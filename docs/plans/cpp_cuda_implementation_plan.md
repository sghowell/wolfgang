# FastPauli C++/CUDA Implementation Plan

## 1. Project summary

Build **FastPauli**: a high-performance C++/CUDA-backed Python package for sparse sums of Pauli strings and Hamiltonians.

FastPauli should **not** replace Qiskit, OpenFermion, Qiskit Nature, or a full simulator. It should be a narrow accelerator for the hot paths around Pauli/Hamiltonian manipulation:

```python
from fastpauli import PauliSum

h = PauliSum.from_qiskit(spo)
h = h.simplify()

groups = h.group_commuting(mode="qwc")
energy = h.expectation_statevector(psi)

spo2 = h.to_qiskit()
```

The project targets sparse Pauli-basis operators: objects represented as a list of Pauli strings plus complex coefficients. This matches Qiskit `SparsePauliOp` and OpenFermion `QubitOperator` conceptually, but FastPauli should use a much lower-level packed representation and optimized kernels.

This plan is supported by source-of-truth architecture, quality, and benchmark contracts that implementation work must preserve:

```text
AGENTS.md
docs/architecture/semantic_contracts.md
docs/architecture/cuda_backend.md
docs/architecture/hardware_targets_and_testing.md
docs/architecture/testing_and_ci.md
docs/architecture/adapter_contracts.md
docs/benchmarks/protocol.md
docs/quality/phase_quality_gates.md
docs/quality/agent_harness.md
docs/quality/code_review.md
docs/quality/code_standards.md
docs/quality/documentation_standards.md
docs/architecture/api_stability.md
docs/quality/security_and_supply_chain.md
docs/quality/release_and_packaging.md
CONTRIBUTING.md
```

The roadmap view is maintained in:

```text
docs/roadmap.md
```

---

## 2. Language and stack decision

Use:

```text
Core language:      C++20 baseline
GPU backend:        CUDA C++ required product milestone
Python bindings:    nanobind
Build backend:      scikit-build-core
Build system:       CMake + Ninja
CPU parallelism:    oneTBB first, OpenMP optional later
GPU primitives:     CUB / Thrust first, custom kernels later
Target policy:      docs/architecture/hardware_targets_and_testing.md
Testing:            pytest + optional Qiskit/OpenFermion comparison tests
Benchmarking:       pytest-benchmark or asv + Google Benchmark for C++
```

Rationale:

- **C++20/CUDA has the highest performance ceiling** because it gives direct access to CUDA C++, CUB, Thrust, oneTBB, CPU intrinsics, custom allocators, and explicit SIMD dispatch while keeping the initial wheel/compiler matrix manageable.
- **nanobind** is a small C++/Python binding library similar to pybind11 and is designed for low-overhead Python extension modules.
- **scikit-build-core** is a Python build backend for CMake-based extension modules, suitable for compiled scientific Python packages.
- **CUB/Thrust** map directly to sort, scan, reduce, and compact workloads that FastPauli will need for simplify/deduplicate and potentially GPU expectation kernels.
- **oneTBB** gives a mature path for parallel CPU sort/reduce and blocked parallel loops.

---

## 3. Core product scope

FastPauli should accelerate these operations:

```text
1. Construction from dense labels
2. Construction from sparse Pauli triples
3. Conversion from Qiskit SparsePauliOp
4. Conversion from OpenFermion QubitOperator
5. Canonicalization / simplification / duplicate reduction
6. Addition and scalar multiplication
7. PauliSum @ PauliSum multiplication with guardrails
8. Commutation checks
9. Qubit-wise commuting grouping
10. Full commuting grouping heuristic
11. Statevector expectation values
12. Z-diagonal expectation from counts / bitstrings
```

The first implementation slice should be **CPU-only**, but CUDA is a required backend milestone. The package scaffold, CPU correctness path, semantic tests, and benchmark baselines come before CUDA kernels so GPU work has unambiguous behavior and measurable win conditions. Host-side data structures must remain compatible with the device mirror described in `docs/architecture/cuda_backend.md`, and CPU/CUDA build options must follow `docs/architecture/hardware_targets_and_testing.md`.

---

## 4. Non-goals

Do **not** implement these in the initial version:

```text
Full Qiskit replacement
Full OpenFermion replacement
General quantum circuit simulation
General dense matrix operator support
Symbolic ParameterExpression coefficients
Optimal graph coloring
FermionOperator -> QubitOperator mappings
Noise simulation
Tensor-network simulation
GPU kernels before CPU correctness
Qiskit C API integration
```

FastPauli should remain a **narrow operator/Hamiltonian accelerator**.

---

## 5. MVP success criteria

The MVP is successful if it can:

```python
from fastpauli import PauliSum

h = PauliSum.from_labels(["XXI", "IZZ", "YYI"], [1.0, 2.0, -0.5j])
h2 = h.simplify()

labels, coeffs = h2.to_labels()
```

and also:

```python
h = PauliSum.from_sparse_list(
    [("ZX", [1, 4], 1.0)],
    num_qubits=5,
)

labels, coeffs = h.to_labels()
assert labels == ["XIIZI"]
```

Initial semantic MVP operations, beginning in Phase 2 after the Phase 1 package and harness scaffold:

```text
from_labels
to_labels
from_sparse_list
to_sparse_list
num_qubits
num_terms
copy
basic validation
basic tests
```

Pre-implementation planning gate:

```text
semantic contracts documented
CUDA backend architecture documented
CPU and CUDA hardware target policy documented
testing and CI architecture documented
adapter contracts documented
benchmark protocol documented
phase quality gates documented
agent guide documented
agent harness documented
agent-driven review policy documented
code standards documented
documentation standards documented
API stability documented
security and supply-chain standards documented
release and packaging standards documented
contribution and review surface documented
roadmap phases documented
first PR scope confirmed as CPU-only but CUDA-compatible
```

Second MVP milestone:

```text
from_qiskit
to_qiskit
simplify
sort
addition
scalar multiplication
```

Third MVP milestone:

```text
multiplication
commutation
grouping
OpenFermion adapter
statevector expectation
Z-count expectation
benchmarks
```

---

## 6. Repository layout

Target this full-roadmap layout as phases add implementation surface.
`include/fastpauli/` is reserved for documented public C++ API headers. Native
implementation helpers belong under `src/detail/` unless a phase explicitly promotes
them to public API with API-stability review.

```text
fastpauli/
  pyproject.toml
  CMakeLists.txt
  README.md
  LICENSE

  include/
    fastpauli/
      pauli_sum.hpp
      errors.hpp
      cpu_backend.hpp

  src/
    pauli_sum.cpp
    parse.cpp
    export.cpp
    arithmetic.cpp
    simplify.cpp
    multiply.cpp
    commute.cpp
    grouping.cpp
    expectation.cpp
    cpu_backend.cpp

  src/detail/
    bitops.hpp
    checked_arithmetic.hpp
    commutation.hpp
    commute_kernels.hpp
    packed_key.hpp
    phase.hpp
    ... additional private implementation headers as needed

  src/simd/
    commute_kernels_scalar.cpp
    commute_kernels_avx2.cpp
    commute_kernels_avx512.cpp
    commute_kernels_neon.cpp
    ... scalar fallback and ISA-specialized CPU kernels as needed

  src/parallel/
    commute_kernels_tbb.cpp
    ... threaded CPU kernels as needed

  src/cuda/
    CMakeLists.txt
    device_pauli_sum.cuh
    simplify_cuda.cu
    expectation_cuda.cu
    commutation_cuda.cu
    matmul_cuda.cu

  bindings/
    python/
      module.cpp
      pauli_sum_py.cpp
      ndarray_adapters.hpp

  python/
    fastpauli/
      __init__.py
      qiskit.py
      openfermion.py
      typing.py
      _version.py

  tests/
    test_basic.py
    test_endianness.py
    test_sparse_list.py
    test_qiskit_roundtrip.py
    test_openfermion_roundtrip.py
    test_simplify.py
    test_arithmetic.py
    test_multiply.py
    test_commutation.py
    test_grouping.py
    test_expectation.py

  benchmarks/
    bench_from_labels.py
    bench_from_sparse_list.py
    bench_qiskit_conversion.py
    bench_openfermion_conversion.py
    bench_simplify.py
    bench_multiply.py
    bench_commutation.py
    bench_grouping.py
    bench_expectation.py

  cpp_benchmarks/
    CMakeLists.txt
    bench_simplify.cpp
    bench_commutation.cpp
    bench_expectation.cpp
```

The top-level `src/` directory is for backend-neutral operation orchestration
and scalar data-structure behavior. Backend-specialized CPU implementation files
belong under `src/simd/` or `src/parallel/`; scalar fallback kernels may live in
`src/simd/` when they are the dispatch-family baseline for adjacent ISA-specific
files. CUDA implementation files belong under `src/cuda/`. `scripts/validate.py`
and `tests/test_native_layout.py` enforce the current native source layout so
future phases fail quickly if new backend files are added in the wrong directory.

---

## 7. Python package build

### `pyproject.toml`

Use scikit-build-core and nanobind:

```toml
[build-system]
requires = [
  "scikit-build-core>=0.10",
  "nanobind>=2.0",
]
build-backend = "scikit_build_core.build"

[project]
name = "fastpauli"
version = "0.1.0"
description = "High-performance C++/CUDA accelerator for sparse Pauli sums"
requires-python = ">=3.10"
dependencies = [
  "numpy>=1.24",
]

[project.optional-dependencies]
qiskit = ["qiskit"]
openfermion = ["openfermion"]
test = ["pytest", "hypothesis", "pytest-benchmark"]
bench = ["pytest-benchmark", "asv"]

[tool.scikit-build]
cmake.version = ">=3.24"
ninja.version = ">=1.10"
build-dir = "build/{wheel_tag}"
wheel.packages = ["python/fastpauli"]
```

### CMake build options

Expose these options:

```text
FASTPAULI_ENABLE_TBB=ON/OFF
FASTPAULI_ENABLE_OPENMP=ON/OFF
FASTPAULI_ENABLE_CUDA=ON/OFF
FASTPAULI_ENABLE_AVX2=ON/OFF
FASTPAULI_ENABLE_AVX512=ON/OFF
FASTPAULI_ENABLE_ARM_NEON=ON/OFF
FASTPAULI_ENABLE_ARM_SVE=ON/OFF
FASTPAULI_ENABLE_NATIVE=ON/OFF
FASTPAULI_CUDA_ARCHITECTURES=<CMake CUDA architecture list>
FASTPAULI_BUILD_TESTS=ON/OFF
FASTPAULI_BUILD_BENCHMARKS=ON/OFF
```

Default for source builds:

```text
FASTPAULI_ENABLE_TBB=auto
FASTPAULI_ENABLE_CUDA=OFF
FASTPAULI_ENABLE_AVX2=auto
FASTPAULI_ENABLE_AVX512=auto
FASTPAULI_ENABLE_ARM_NEON=auto
FASTPAULI_ENABLE_ARM_SVE=auto
FASTPAULI_ENABLE_NATIVE=OFF
```

Default for wheels:

```text
FASTPAULI_ENABLE_CUDA=OFF
portable scalar baseline included
runtime CPU dispatch enabled where possible
native CPU tuning disabled
```

CUDA wheels should be a separate packaging effort after the CPU package is stable.

CUDA build policy:

```text
FASTPAULI_ENABLE_CUDA=OFF must build without CUDA installed.
FASTPAULI_ENABLE_CUDA=ON is required to build the CUDA backend from source.
CPU-only public headers must not include CUDA headers.
CUDA source-build support initially targets CUDA 12.9.x or the current CUDA 12.x line, with architecture targets defined in docs/architecture/hardware_targets_and_testing.md.
```

CPU and CUDA target, packaging, dispatch, and source-build contracts are defined in `docs/architecture/hardware_targets_and_testing.md` and `docs/architecture/cuda_backend.md`.

Before CUDA implementation begins, complete the CPU performance hardening
checkpoint in `docs/roadmap.md`. The checkpoint must profile and optimize the
existing scalar CPU implementation, extend benchmarks where needed, and record
Apple Silicon plus x86_64 evidence without changing public semantics or implying
that unavailable oneTBB, SIMD, or CUDA paths exist.

---

## 8. Internal representation

Use a packed bit representation.

The detailed semantic contract for representation, ordering, empty operators, coefficients, tolerances, multiplication, commutation, grouping, and expectation behavior is `docs/architecture/semantic_contracts.md`.

```cpp
namespace fastpauli {

struct PauliSum {
    std::size_t num_qubits = 0;
    std::size_t words = 0;
    std::size_t num_terms = 0;

    std::vector<std::uint64_t> x;
    std::vector<std::uint64_t> z;
    std::vector<std::complex<double>> coeffs;
};

}
```

Representation invariants:

```text
bit q represents qubit q
qubit 0 is bit 0
word index = q / 64
bit offset = q % 64

x=0, z=0 means I
x=1, z=0 means X
x=0, z=1 means Z
x=1, z=1 means Y

All Pauli phases are folded into coeffs.
Unused high bits in the final word are always zeroed.
```

Dense-label convention:

```text
FastPauli internal qubit 0 = bit 0
Qiskit dense label qubit 0 = right-most character
```

Therefore:

```text
"XYZ" means:
  X on qubit 2
  Y on qubit 1
  Z on qubit 0
```

Sparse-list convention:

```text
("ZX", [1, 4], coeff) means:
  Z on qubit 1
  X on qubit 4
```

Canonical ordering:

```text
simplify() returns lexicographic order over x word 0, z word 0, x word 1, z word 1, ...
sort(by_weight=True) orders by ascending Pauli weight, then canonical order.
Construction methods preserve input term order until simplify() or sort() is called.
```

---

## 9. Public Python API

Initial semantic API, beginning in Phase 2 after the Phase 1 package and harness scaffold:

Phase 1 exposes only the minimal `fastpauli.PauliSum` scaffold and properties needed for install/import smoke tests. The semantic construction, export, adapter, and algebra APIs below are not part of the first PR.

```python
class PauliSum:
    @classmethod
    def from_labels(
        cls,
        labels: list[str],
        coeffs: object | None = None,
    ) -> "PauliSum": ...

    @classmethod
    def from_sparse_list(
        cls,
        triples: list[tuple[str, list[int], complex]],
        num_qubits: int,
    ) -> "PauliSum": ...

    @classmethod
    def empty(cls, num_qubits: int) -> "PauliSum": ...

    @classmethod
    def from_qiskit(cls, op) -> "PauliSum": ...

    @classmethod
    def from_openfermion(
        cls,
        op,
        num_qubits: int | None = None,
    ) -> "PauliSum": ...

    @property
    def num_qubits(self) -> int: ...

    @property
    def num_terms(self) -> int: ...

    def copy(self) -> "PauliSum": ...

    def simplify(
        self,
        atol: float = 1e-12,
        rtol: float = 0.0,
    ) -> "PauliSum": ...

    def sort(
        self,
        by_weight: bool = False,
    ) -> "PauliSum": ...

    def __add__(self, other: "PauliSum") -> "PauliSum": ...

    def __mul__(self, scalar: complex) -> "PauliSum": ...

    def __rmul__(self, scalar: complex) -> "PauliSum": ...

    def __matmul__(self, other: "PauliSum") -> "PauliSum": ...

    def matmul(
        self,
        other: "PauliSum",
        *,
        simplify: bool = True,
        max_intermediate_terms: int = 50_000_000,
    ) -> "PauliSum": ...

    def commutes_with(self, other: "PauliSum") -> object: ...

    def group_commuting(
        self,
        mode: str = "qwc",
        strategy: str = "largest_first",
        max_terms_for_graph: int = 50_000,
    ) -> list["PauliSum"]: ...

    def expectation_statevector(self, psi) -> complex: ...

    def expectation_z_counts(self, counts) -> complex: ...

    def to_labels(self) -> tuple[list[str], object]: ...

    def to_sparse_list(self) -> list[tuple[str, list[int], complex]]: ...

    def to_qiskit(self): ...

    def to_openfermion(self): ...
```

CUDA API additions after the CPU contract is stable:

```python
class PauliSum:
    def to_device(self, device: int = 0) -> "DevicePauliSum": ...

class DevicePauliSum:
    @property
    def num_qubits(self) -> int: ...

    @property
    def num_terms(self) -> int: ...

    def to_host(self) -> PauliSum: ...
    def simplify(self, atol: float = 1e-12, rtol: float = 0.0) -> "DevicePauliSum": ...
    def expectation_statevector(self, psi) -> complex: ...
```

Once CUDA lands, backend-selectable host methods use:

```python
h.simplify(backend="auto")  # "cpu", "cuda", or "auto"
h.expectation_statevector(psi, backend="auto")  # "cpu", "cuda", or "auto"
```

`backend="cuda"` raises `RuntimeError` when the package was built without CUDA support.

---

## 10. C++ API

Expose a small native API independent of Python:

```cpp
namespace fastpauli {

class PauliSumView;

PauliSum from_labels(
    std::span<const std::string> labels,
    std::span<const std::complex<double>> coeffs
);

PauliSum from_sparse_terms(
    std::span<const SparseTermInput> terms,
    std::size_t num_qubits
);

PauliSum empty(std::size_t num_qubits);

std::vector<std::string> to_labels(const PauliSum& op);

std::vector<SparseTermOutput> to_sparse_terms(const PauliSum& op);

PauliSum simplify(
    const PauliSum& op,
    double atol,
    double rtol
);

PauliSum add(
    const PauliSum& lhs,
    const PauliSum& rhs,
    bool simplify_output
);

PauliSum multiply(
    const PauliSum& lhs,
    const PauliSum& rhs,
    bool simplify_output,
    std::uint64_t max_intermediate_terms
);

bool commute_terms(
    const std::uint64_t* x1,
    const std::uint64_t* z1,
    const std::uint64_t* x2,
    const std::uint64_t* z2,
    std::size_t words
);

std::vector<PauliSum> group_commuting_qwc(const PauliSum& op);
std::vector<PauliSum> group_commuting_full(const PauliSum& op);

std::complex<double> expectation_statevector(
    const PauliSum& op,
    std::span<const std::complex<double>> psi
);

}
```

This keeps the core usable from C++ benchmarks and future non-Python bindings.

---

## 11. Parsing and exporting

### Dense labels

Algorithm:

```text
Input:
  labels = ["XYZ", ...]

For each label:
  num_qubits = len(label)

For each character position p from left to right:
  q = num_qubits - 1 - p

  I: x[q]=0, z[q]=0
  X: x[q]=1, z[q]=0
  Y: x[q]=1, z[q]=1
  Z: x[q]=0, z[q]=1
```

Validation:

```text
All labels have equal length
Only I, X, Y, Z are accepted
Coefficient count matches label count
Empty input requires PauliSum.empty(num_qubits)
```

`PauliSum.from_labels([])` is invalid. Use `PauliSum.empty(num_qubits)` for zero-term operators.

### Sparse list

Input shape:

```python
[
    ("ZX", [1, 4], 1.0),
    ("YY", [0, 3], -1.0 + 1.0j),
]
```

Algorithm:

```text
For each sparse term:
  local_pauli_string length must equal qubit_indices length

  For k in range(len(local_pauli_string)):
      p = local_pauli_string[k]
      q = qubit_indices[k]
      set bit q according to p
```

Validation:

```text
All qubit indices in range
No duplicate qubit index inside one term
Only I, X, Y, Z are accepted
Coefficient is numeric complex
```

---

## 12. Qiskit adapter

Implement in Python first:

```text
python/fastpauli/qiskit.py
```

Public methods:

```python
PauliSum.from_qiskit(op)
PauliSum.to_qiskit()
```

Conversion strategy:

```text
Qiskit SparsePauliOp
  -> extract labels or PauliList x/z arrays and coeffs
  -> call C++ batch constructor
```

Start with a robust label/sparse-list path, then optimize to direct `x`, `z`, `phase`, and `coeffs` extraction after tests are stable.

Important semantic requirement:

```text
All Qiskit Pauli phases must be folded into FastPauli coefficients.
FastPauli should export zero-phase Pauli strings back to Qiskit.
```

Do not materialize dense matrices during conversion.

---

## 13. OpenFermion adapter

Implement in Python first:

```text
python/fastpauli/openfermion.py
```

OpenFermion terms look like:

```python
{
    ((0, "X"), (5, "X")): 0.5,
    ((1, "Z"), (2, "Y")): -1.0j,
}
```

Adapter behavior:

```text
from_openfermion:
  infer num_qubits if not provided
  convert each term tuple to sparse-list form
  call PauliSum.from_sparse_list

to_openfermion:
  export sparse terms
  build openfermion.QubitOperator
```

---

## 14. Simplify / deduplicate

This is the first major performance target.

Goal:

```python
h2 = h.simplify(atol=1e-12, rtol=0.0)
```

Semantics:

```text
Combine duplicate Pauli strings.
Sum their coefficients.
Drop coefficients using abs(c) <= atol + rtol * max_abs_input_coefficient.
Return a canonical PauliSum in lexicographic packed-word order.
Reject negative tolerances with ValueError.
```

CPU algorithm:

```text
1. Create an index array [0, 1, ..., num_terms - 1].
2. Sort indices by packed Pauli key:
     x word 0, z word 0, x word 1, z word 1, ...
3. Linear scan sorted indices.
4. Accumulate coefficients for equal keys.
5. Drop near-zero coefficients.
6. Emit compact output arrays.
```

Specialized paths:

```text
words == 1:
  key = (x0, z0)

words == 2:
  key = (x0, z0, x1, z1)

words <= 4:
  fixed-size inline comparator

words > 4:
  slice comparator
```

Parallel CPU path:

```text
oneTBB parallel_sort
parallel chunk reduction
final serial or parallel merge of chunk boundaries
```

CUDA path, after CPU benchmark gate:

```text
Pack keys
Sort by key with CUB or Thrust
Reduce by key
Compact nonzero coefficients
Copy result back or keep on device
```

Do not implement GPU simplify until the CPU benchmark suite exists.

---

## 15. Addition and scalar multiplication

Addition:

```python
h3 = h1 + h2
```

Implementation:

```text
Check same num_qubits.
Allocate output with len = len(h1) + len(h2).
Copy x/z/coeff buffers.
Do not simplify automatically unless explicit add(..., simplify=True) is added.
Preserve left-then-right term order before any explicit simplify.
```

Scalar multiplication:

```python
h2 = 2.5 * h
h3 = h * (1.0 - 0.5j)
```

Implementation:

```text
Copy x and z buffers.
Scale coefficient buffer.
Parallelize coefficient scaling for large term counts.
Multiplication by zero keeps terms until simplify() is called.
```

---

## 16. Pauli multiplication

Support:

```python
h3 = h1 @ h2
```

`h1 @ h2` follows matrix multiplication semantics: `h2` acts first and `h1` acts second.

Single-qubit phase fixtures:

```text
X @ Y =  i Z
Y @ X = -i Z
Y @ Z =  i X
Z @ Y = -i X
Z @ X =  i Y
X @ Z = -i Y
```

Guardrail:

```python
h3 = h1.matmul(
    h2,
    simplify=True,
    max_intermediate_terms=50_000_000,
)
```

Reject accidental blowups:

```text
if len(h1) * len(h2) > max_intermediate_terms:
    raise ValueError
```

Term product:

```text
x3 = x1 XOR x2
z3 = z1 XOR z2
```

For the canonical representation:

```text
P(x, z) = (-i)^(popcnt(x & z)) Z^z X^x
```

A correct phase update is:

```text
e = popcnt(x3 & z3)
  - popcnt(x1 & z1)
  - popcnt(x2 & z2)
  + 2 * popcnt(x1 & z2)

phase = i^e
coeff3 = coeff1 * coeff2 * phase
```

where `e` is taken modulo 4.

Implementation:

```text
1. Block over lhs terms.
2. Block over rhs terms.
3. Generate output x/z/coeff buffers.
4. Use popcount for phase.
5. Simplify output if requested.
```

CPU optimization:

```text
Use std::popcount where available.
Specialize words == 1, 2, 4.
Parallelize product generation by output blocks.
```

CUDA optimization, later:

```text
One thread per product term for moderate sizes.
Block-level output generation.
Then call GPU simplify.
```

---

## 17. Commutation checks

Full Pauli commutation:

```text
anticommutes iff parity(popcnt((x1 & z2) XOR (z1 & x2))) == 1
commutes otherwise
```

Qubit-wise commutation conflict:

```text
qwc conflict iff ((x1 & z2) XOR (z1 & x2)) != 0
```

Expose:

```python
h.commutes_with(other)
```

Possible return shape:

```text
If both operands have one term:
  return bool

If exactly one operand has one term:
  return boolean vector shape (h.num_terms,)

If both operands have many terms:
  return boolean matrix shape (h.num_terms, other.num_terms)
  only when h.num_terms * other.num_terms <= max_commutation_matrix_entries
```

Default dense-matrix guardrail:

```text
max_commutation_matrix_entries = 100_000_000
```

CPU implementation:

```text
Nested blocked loops.
Specialize small word counts.
Parallelize outer blocks.
```

CUDA implementation, later:

```text
One thread per pair or tile.
Useful for dense pairwise commutation matrices.
Less useful for greedy grouping itself.
```

---

## 18. Grouping

Expose:

```python
groups = h.group_commuting(mode="qwc")
groups = h.group_commuting(mode="full")
```

### QWC grouping

Use greedy largest-first grouping.

For each group, maintain basis masks:

```cpp
struct QWCGroupState {
    std::vector<std::uint64_t> x_basis;
    std::vector<std::uint64_t> z_basis;
    std::vector<std::size_t> term_indices;
};
```

A term fits a QWC group if:

```text
((term.x & group.z_basis) XOR (term.z & group.x_basis)) == 0
```

Algorithm:

```text
1. Sort terms by descending Pauli weight, then canonical order.
2. For each term:
     try groups in order
     place into first compatible group
     otherwise create new group
3. Return PauliSum objects for each group.
```

### Full commuting grouping

Use two modes:

```text
Small n_terms:
  Build noncommutation graph.
  Greedy color graph.

Large n_terms:
  Streaming greedy grouping.
  Each term must commute with all terms already in the candidate group.
```

Do **not** claim optimality. The method should be documented as a deterministic heuristic.

Group output order follows deterministic greedy placement order. Grouping must never rely on unordered hash iteration for externally visible order.

---

## 19. Expectation value: statevector

Expose:

```python
energy = h.expectation_statevector(psi)
```

Constraints:

```text
num_qubits <= 63 initially
len(psi) == 2 ** num_qubits
psi dtype complex64 or complex128
```

For a Pauli term:

```text
P |j> = phase(j) |j XOR x_mask>
```

where:

```text
phase(j) = i^popcnt(x & z) * (-1)^popcnt(z & j)
```

Expectation:

```text
<psi|P|psi> = sum_j conj(psi[j XOR x]) * phase(j) * psi[j]
```

Then multiply by the term coefficient and sum across terms.

CPU strategy:

```text
If many terms:
  parallelize over terms

If few terms and large statevector:
  parallelize over statevector chunks

Use a heuristic threshold.
```

CUDA strategy, later:

```text
Kernel computes partial sums per term or per term block.
Block-level reductions.
Final reduction on GPU or CPU.
Group terms by x_mask if that improves memory access.
```

CUDA statevector interop:

```text
Host NumPy arrays may be copied to device internally.
Device-resident arrays are accepted through __cuda_array_interface__ first.
DLPack support can be added after CUDA-array-interface tests are stable.
Accepted device arrays must be 1-dimensional, contiguous, complex64 or complex128, and on the same CUDA device as DevicePauliSum.
```

---

## 20. Expectation value: Z counts

Expose:

```python
energy = h.expectation_z_counts(counts)
```

Only diagonal terms are supported initially:

```text
Reject if any term has x != 0.
```

For each bitstring/count pair:

```text
term_value = (-1) ** parity(z_mask & bitstring)
weighted contribution = count * coeff * term_value
```

Input forms:

```python
{"0101": 123, "1101": 456}
[(bitstring_int, count), ...]
np.ndarray of bitstring integers
```

Start with Python dict support. Add vectorized NumPy support later.

String bitstrings use the dense-label convention: the right-most bit is qubit 0.

---

## 21. CPU performance plan

### Memory layout

Use structure-of-arrays:

```text
x:      contiguous uint64 words
z:      contiguous uint64 words
coeffs: contiguous complex<double>
```

This is preferable to per-term objects.

### Alignment

Use an aligned allocator later:

```text
64-byte alignment for AVX-512-friendly loads
32-byte alignment for AVX2-friendly loads
```

Do not add allocator complexity in Phase 1.

### Dispatch

Use runtime CPU feature detection:

```text
scalar baseline
AVX2 path
AVX-512 path
```

Initial implementation can use scalar loops plus compiler autovectorization. Add explicit SIMD only after benchmarks identify bottlenecks and after the forced-path testing rules in `docs/architecture/hardware_targets_and_testing.md` can be enforced.

### Parallelism

Use oneTBB for:

```text
parallel_sort
parallel_for
parallel_reduce
blocked_range
```

Start with oneTBB optional. Keep a portable fallback path and preserve release-wheel defaults from `docs/architecture/hardware_targets_and_testing.md`.

---

## 22. CUDA backend plan

CUDA is a required backend milestone. It is still implemented after CPU semantics and benchmark baselines are stable. The detailed backend contract is `docs/architecture/cuda_backend.md`.

### CUDA build option

```bash
pip install . --config-settings=cmake.define.FASTPAULI_ENABLE_CUDA=ON
```

### Device representation

```cpp
struct DevicePauliSum {
    std::size_t num_qubits;
    std::size_t words;
    std::size_t num_terms;

    std::uint64_t* x;
    std::uint64_t* z;
    thrust::complex<double>* coeffs;
    int device_ordinal;
};
```

`DevicePauliSum` is an owning device mirror. It uses explicit transfers and must not rely on CUDA unified memory in the first CUDA milestone.

### Initial CUDA targets

CUDA toolkit, driver, and `sm_*` targets are defined in `docs/architecture/hardware_targets_and_testing.md`. The CUDA 12.x source-build lane keeps Volta `sm_70` viable; the CUDA 13.x lane must not be adopted as the only baseline without revisiting that policy.

Implement in this order:

```text
1. Device transfer and equality tests
2. CUDA statevector expectation or CUDA simplify, selected from benchmark evidence
3. The remaining CUDA expectation/simplify target
4. CUDA pairwise commutation matrix with dense-output guardrails
5. CUDA multiplication product generation + simplify
```

CUDA benchmarks must report both transfer-inclusive and device-resident timings.

### Do not start with GPU grouping

Greedy grouping is branchy and sequential compared with sort/reduce and expectation kernels. GPU can help build pairwise conflict data, but the grouping heuristic itself should remain CPU-first until there is a clear benchmark reason to move it.

### CUDA primitives

Use CUB or Thrust first for:

```text
sort
reduce_by_key
scan
compact
```

Only write custom kernels where primitives do not give the needed performance or memory layout.

---

## 23. Testing plan

### Mandatory endianness tests

```python
def test_dense_label_endianness():
    h = PauliSum.from_labels(["XYZ"])
    sparse = h.to_sparse_list()
    assert sparse == [("ZYX", [0, 1, 2], 1.0 + 0.0j)]
```

```python
def test_sparse_list_example():
    h = PauliSum.from_sparse_list(
        [("ZX", [1, 4], 1.0)],
        num_qubits=5,
    )
    labels, coeffs = h.to_labels()
    assert labels == ["XIIZI"]
```

### Basic tests

```text
from_labels / to_labels round-trip
from_sparse_list / to_sparse_list round-trip
invalid characters raise ValueError
duplicate sparse indices raise ValueError
out-of-range sparse indices raise ValueError
coefficient length mismatch raises ValueError
empty input behavior is explicit
PauliSum.empty(num_qubits) creates a zero-term operator
construction order is preserved before simplify or sort
```

### Qiskit comparison tests

For random small operators:

```text
FastPauli.to_qiskit() equals original SparsePauliOp after simplify
Matrix equality for n <= 8
Simplify matches Qiskit semantics
Addition matches Qiskit
Multiplication matches Qiskit by dense matrix comparison
Grouping outputs internally valid groups
```

### Multiplication phase fixtures

```text
X @ Y simplifies to i Z
Y @ X simplifies to -i Z
Y @ Z simplifies to i X
Z @ Y simplifies to -i X
Z @ X simplifies to i Y
X @ Z simplifies to -i Y
```

### OpenFermion comparison tests

```text
QubitOperator -> PauliSum -> QubitOperator preserves terms
Coefficients match after simplification
Random sparse OpenFermion terms round-trip
```

### Property tests

Use Hypothesis or a custom generator.

Properties:

```text
simplify is idempotent
sort does not change operator semantics
A + B == B + A after simplify
(A @ B) @ C == A @ (B @ C) for small random operators
commutation is symmetric
every QWC group is internally QWC-compatible
every full group is internally commuting
statevector expectation matches dense matrix for n <= 8
```

### CUDA comparison tests

CUDA tests are required once `FASTPAULI_ENABLE_CUDA=ON` lands. They must skip cleanly when CUDA is not built or no runtime device is available.

Required properties:

```text
PauliSum.to_device().to_host() preserves labels and coefficients
DevicePauliSum.simplify().to_host() matches CPU simplify canonical output
CUDA expectation matches CPU expectation within dtype-specific tolerance
CUDA errors match CPU guardrails for invalid sizes and unsupported dtypes
```

---

## 24. Benchmark plan

Benchmarks should exist from the first simplification PR onward.

### Benchmark dimensions

```text
num_qubits:
  32, 64, 128, 512, 2048

num_terms:
  10_000, 100_000, 1_000_000

term_weight:
  2, 4, 8, 16, 64

duplicate_rate:
  0%, 10%, 50%, 90%

coefficient dtype:
  complex128
```

### Baselines

Compare against:

```text
Qiskit SparsePauliOp
OpenFermion QubitOperator
pure NumPy helper where relevant
FastPauli scalar CPU
FastPauli oneTBB CPU
FastPauli CUDA transfer-inclusive when available
FastPauli CUDA device-resident when available
```

### Benchmarks

```text
bench_from_labels
bench_from_sparse_list
bench_from_qiskit
bench_to_qiskit
bench_from_openfermion
bench_to_openfermion
bench_simplify_low_duplicate
bench_simplify_high_duplicate
bench_add_then_simplify
bench_multiply_by_single_term
bench_multiply_sum_small_cross_product
bench_pairwise_commutation
bench_qwc_grouping
bench_full_grouping
bench_expectation_statevector
bench_expectation_z_counts
```

### Initial performance goals

Directional goals, not promises:

```text
from_sparse_list:
  10x+ over Python-heavy construction for large sparse operators

simplify:
  10x-100x over Python/object-heavy paths on 100k-1M terms

OpenFermion conversion/manipulation:
  50x-200x on large QubitOperator-style workloads

commutation:
  10x-50x on packed-bit pairwise checks

expectation_statevector:
  5x-50x over naive Python loops
```

---

## 25. Implementation phases

### Phase 0: planning and architecture lock

Acceptance criteria:

```text
docs/architecture/semantic_contracts.md defines correctness contracts
docs/architecture/cuda_backend.md defines required CUDA architecture
docs/architecture/hardware_targets_and_testing.md defines CPU/CUDA target and hardware validation expectations
docs/architecture/testing_and_ci.md defines validation and CI expectations
docs/architecture/adapter_contracts.md defines optional integration behavior
docs/benchmarks/protocol.md defines performance evidence requirements
docs/quality/phase_quality_gates.md defines phase completion gates
docs/quality/agent_harness.md defines Codex-driven harness expectations
docs/quality/code_review.md defines independent agent-driven review expectations
docs/quality/code_standards.md defines C++/CUDA/Python code quality expectations
docs/quality/documentation_standards.md defines API and user-facing documentation expectations
docs/architecture/api_stability.md defines public API compatibility expectations
docs/quality/security_and_supply_chain.md defines native-code and dependency safety expectations
docs/quality/release_and_packaging.md defines packaging, versioning, and release expectations
CONTRIBUTING.md defines the review and contribution workflow
AGENTS.md maps agents to the source-of-truth docs
docs/roadmap.md tracks phase order and release gates
this implementation plan references those documents
```

Implement:

```text
semantic contract documentation
CUDA backend architecture documentation
testing and CI architecture documentation
adapter contract documentation
benchmark protocol documentation
phase quality gate documentation
agent guide documentation
agent harness documentation
agent-driven review documentation
code standards documentation
documentation standards documentation
API stability documentation
security and supply-chain documentation
release and packaging documentation
contribution and review documentation
roadmap documentation
plan updates for resolved decisions
```

---

### Phase 1: C++/nanobind scaffold

Acceptance criteria:

```text
pip install -e . works
import fastpauli works
fastpauli.PauliSum exists
pytest runs
CPU-only build does not require CUDA headers or toolkit
repo-local validation command exists
first mechanical harness checks exist
initial review-policy checks exist
initial code/documentation standard checks exist
initial API/security/release standard checks exist
Phase 1 gates in docs/quality/phase_quality_gates.md are satisfied
```

Implement:

```text
pyproject.toml
CMakeLists.txt
nanobind module
minimal PauliSum class
num_qubits property
num_terms property
scripts/validate.py
CPU-only CMake default configuration
FASTPAULI_ENABLE_CUDA=OFF and FASTPAULI_ENABLE_NATIVE=OFF build-default checks
initial .github/workflows/ci.yml
source-doc existence checks
AGENTS.md and README.md local-link checks
stale-marker scan
review-policy closeout checks
```

Do not implement packed parsing/export APIs in Phase 1. Keep `from_labels`, `to_labels`, `from_sparse_list`, `to_sparse_list`, `empty(num_qubits)`, endianness behavior, and invalid-input semantics for Phase 2.

No CUDA kernels yet. Keep the public CPU core compatible with the future device mirror.

---

### Phase 2: packed representation and parsing

Acceptance criteria:

```text
PauliSum.from_labels works
PauliSum.to_labels works
PauliSum.from_sparse_list works
PauliSum.to_sparse_list works
PauliSum.empty(num_qubits) works
construction order is preserved before simplify or sort
endianness tests pass
Phase 2 gates in docs/quality/phase_quality_gates.md are satisfied
```

Implement:

```text
C++ PauliSum struct
dense label parser
dense label exporter
sparse-list parser
sparse-list exporter
input validation
Python binding
```

---

### Phase 3: Qiskit adapter

Acceptance criteria:

```text
PauliSum.from_qiskit works
PauliSum.to_qiskit works
small random operators round-trip
n <= 8 matrix comparisons pass
Qiskit behavior follows docs/architecture/adapter_contracts.md
Phase 3 gates in docs/quality/phase_quality_gates.md are satisfied
```

Implement:

```text
python/fastpauli/qiskit.py
optional dependency handling
tests skipped when qiskit not installed
```

---

### Phase 4: simplify

Acceptance criteria:

```text
simplify combines duplicates
simplify removes near-zero coefficients using documented tolerance semantics
simplify returns canonical order
simplify is idempotent
simplify matches Qiskit for small random tests
bench_simplify.py exists
benchmark behavior follows docs/benchmarks/protocol.md
Phase 4 gates in docs/quality/phase_quality_gates.md are satisfied
```

Implement:

```text
sort-based deduplication
scalar C++ baseline
optional oneTBB parallel_sort
specialized words == 1 and words == 2 paths
```

---

### Phase 5: arithmetic

Acceptance criteria:

```text
addition works
scalar multiplication works
single-term multiplication works
PauliSum @ PauliSum works with guardrail
small dense matrix comparisons pass
bench_multiply.py exists
Phase 5 gates in docs/quality/phase_quality_gates.md are satisfied
```

Implement:

```text
concat addition
coefficient scaling
blocked product generation
phase-correct multiplication
post-product simplify
```

---

### Phase 6: commutation and grouping

Acceptance criteria:

```text
commutes_with works
commutes_with enforces max_commutation_matrix_entries before dense allocation
qwc grouping returns valid QWC groups
full grouping returns valid commuting groups
bench_grouping.py exists
Phase 6 gates in docs/quality/phase_quality_gates.md are satisfied
```

Implement:

```text
pairwise commutation kernels
QWC greedy grouping
full commuting greedy grouping
optional noncommutation graph for small term counts
```

---

### Phase 7: OpenFermion adapter

Acceptance criteria:

```text
from_openfermion works
to_openfermion works
round-trip tests pass
bench_openfermion_conversion.py exists
OpenFermion behavior follows docs/architecture/adapter_contracts.md
Phase 7 gates in docs/quality/phase_quality_gates.md are satisfied
```

Implement:

```text
python/fastpauli/openfermion.py
optional dependency handling
term tuple conversion
```

---

### Phase 8: expectation kernels

Acceptance criteria:

```text
expectation_statevector matches dense matrix for n <= 8
expectation_z_counts matches direct Python computation
Z-count bitstring endianness follows dense-label convention
bench_expectation.py exists
Phase 8 gates in docs/quality/phase_quality_gates.md are satisfied
```

Implement:

```text
statevector expectation scalar baseline
parallel term-wise expectation
Z-count expectation
dtype validation
```

---

### Phase 9: CPU optimization

Acceptance criteria:

```text
benchmarks report scalar timings and optimized-path availability
runtime CPU dispatch works for auto and forced scalar
uncompiled or hardware-unavailable optimized selectors fail clearly
AVX2 path tested where compiled and available
oneTBB path tested where compiled and available
forced scalar and optimized paths follow docs/architecture/hardware_targets_and_testing.md
Phase 9 gates in docs/quality/phase_quality_gates.md are satisfied
```

Implement:

```text
aligned buffers if needed
AVX2 kernels for hot loops
AVX-512 kernels if justified
runtime dispatch
cache-aware blocking
```

---

### Phase 10: CUDA backend foundation

Acceptance criteria:

```text
FASTPAULI_ENABLE_CUDA=ON builds locally
FASTPAULI_ENABLE_CUDA=OFF builds without CUDA installed
PauliSum.to_device works
DevicePauliSum.to_host works
device transfer equality tests pass
CUDA skip reasons distinguish build-time absence from runtime device absence
Phase 10 gates in docs/quality/phase_quality_gates.md are satisfied
```

Implement:

```text
DevicePauliSum
host/device transfer
Python DevicePauliSum binding
CUDA availability detection
CUDA transfer tests behind availability checks
```

---

### Phase 11: CUDA kernels

Acceptance criteria:

```text
first CUDA kernel selected by benchmark evidence
CUDA simplify and CUDA expectation both match CPU output
CPU/GPU benchmarks include transfer-inclusive and device-resident timings
CUDA pairwise commutation enforces dense-output guardrails
CUDA multiplication enforces max_intermediate_terms
CUDA benchmark reporting follows docs/benchmarks/protocol.md
extended CPU/CUDA hillclimb runs end with the comprehensive checked-in optimization and profiling report required by docs/benchmarks/protocol.md, including open-source competitor installation/comparison and publication-quality technical visuals
Phase 11 gates in docs/quality/phase_quality_gates.md are satisfied
```

Implement:

```text
CUDA statevector expectation
CUDA simplify using Thrust or CUB
CUDA pairwise commutation matrix
CUDA multiplication product generation followed by simplify
CPU/GPU equivalence tests
CUDA benchmarks
```

---

## 26. First PR scope

The first PR should be deliberately small and limited to the Phase 1 package and harness scaffold.

Implement only:

```text
pyproject.toml with scikit-build-core and nanobind
CMakeLists.txt with CPU-only defaults
FASTPAULI_ENABLE_CUDA=OFF and FASTPAULI_ENABLE_NATIVE=OFF build-default validation
minimal C++ extension module
minimal nanobind Python class fastpauli.PauliSum
package __init__.py and __version__
num_qubits
num_terms
pytest import smoke tests
scripts/validate.py
initial GitHub Actions CPU CI workflow
docs/source-of-truth existence checks
README.md and AGENTS.md local-link checks
stale-marker scan
review-policy closeout checks
```

Do **not** implement:

```text
from_labels
to_labels
from_sparse_list
to_sparse_list
empty(num_qubits)
packed x/z/coeff buffers
endianness parsing/export tests
Qiskit adapter
OpenFermion adapter
simplify
multiplication
commutation
grouping
expectation
CUDA
SIMD
oneTBB
```

The goal of PR 1 is to make FastPauli installable, importable, reviewable, and mechanically validated without taking on representation semantics.

PR 1 must also keep the package and build layout compatible with the future host and `DevicePauliSum` mirror in `docs/architecture/cuda_backend.md`.

---

## 27. First Codex prompt

```text
We are implementing FastPauli, a high-performance C++/CUDA-backed Python package for sparse sums of Pauli strings. The first task is the Phase 1 package and harness scaffold. It is CPU-only. Do not implement CUDA, packed representation parsing/export, or Pauli algebra yet.

Create the Phase 1 package and harness scaffold using:
- C++20
- CMake
- scikit-build-core
- nanobind
- pytest

Use this layout:

fastpauli/
  pyproject.toml
  CMakeLists.txt
  include/
    fastpauli/
      pauli_sum.hpp
      errors.hpp
  src/
    pauli_sum.cpp
  bindings/
    python/
      module.cpp
      pauli_sum_py.cpp
  python/
    fastpauli/
      __init__.py
      _version.py
  scripts/
    validate.py
  tests/
    test_basic.py
  .github/
    workflows/
      ci.yml

Implement only the minimal C++ class needed for import and scaffold tests:

namespace fastpauli {

struct PauliSum {
    std::size_t num_qubits;
    std::size_t num_terms;
};

}

Expose a Python class fastpauli.PauliSum with:
- .num_qubits
- .num_terms

Phase 1 may use a minimal internal constructor or test helper if needed for smoke tests, but it must not expose public parsing/export APIs before Phase 2.

Add scripts/validate.py. It must print each check name, return nonzero on failure, and run at least:
- python -m pytest
- python -c "import fastpauli"
- source-doc existence checks for the docs listed in AGENTS.md and README.md
- README.md and AGENTS.md local-link checks
- stale-marker scan for unsupported planning markers
- review-policy existence and closeout checklist checks
- CPU-only build-default checks showing FASTPAULI_ENABLE_CUDA=OFF does not require CUDA headers, libraries, or toolkit discovery
- portable scalar build-default checks showing FASTPAULI_ENABLE_NATIVE=OFF and identifying the scalar CPU path in validation output

Add .github/workflows/ci.yml with CPU-only Linux and macOS jobs that run python scripts/validate.py.

Tests:
1. import fastpauli succeeds.
2. fastpauli.__version__ exists.
3. fastpauli.PauliSum exists.
4. minimal PauliSum smoke test exposes num_qubits and num_terms if the scaffold constructor is present.
5. scripts/validate.py succeeds locally.

Keep representation parsing/export, Qiskit, OpenFermion, simplify, arithmetic, commutation, grouping, expectation, SIMD, oneTBB, and CUDA out of this first PR.
```

---

## 28. Strategic order

Build in this exact order:

```text
0. Planning and architecture lock
1. Package, C++/nanobind, validation, CI, and review-harness scaffold
2. Packed representation and endianness-correct parsing/export
3. Qiskit adapter
4. Simplify benchmark
5. Sort-based simplify
6. Addition and scalar multiplication
7. Multiplication
8. Commutation
9. QWC grouping
10. Full grouping
11. OpenFermion adapter
12. Statevector expectation
13. Z-count expectation
14. CPU optimization
15. CUDA backend foundation
16. First CUDA kernel selected by benchmark evidence
17. Remaining CUDA simplify/expectation target
18. CUDA commutation
19. CUDA multiplication
```

The highest-risk early bug class is **endianness and phase handling**, not raw performance. Lock those down before optimizing.

---

## 29. References

- Qiskit `SparsePauliOp`: <https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.quantum_info.SparsePauliOp>
- Qiskit operators overview and sparse-list conventions: <https://quantum.cloud.ibm.com/docs/guides/operators-overview>
- OpenFermion `QubitOperator`: <https://quantumai.google/reference/python/openfermion/ops/QubitOperator>
- nanobind documentation: <https://nanobind.readthedocs.io/>
- scikit-build-core documentation: <https://scikit-build-core.readthedocs.io/>
- NVIDIA CUB documentation: <https://docs.nvidia.com/cuda/cub/index.html>
- NVIDIA Thrust: <https://developer.nvidia.com/thrust>
- NVIDIA CUDA Toolkit release notes: <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/>
- NVIDIA CUDA GPU compute capability table: <https://developer.nvidia.com/cuda/gpus>
- NVIDIA CUDA architecture support guidance: <https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/>
- Python packaging platform compatibility tags: <https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/>
- PyPA manylinux project: <https://github.com/pypa/manylinux>
- oneTBB `parallel_sort`: <https://oneapi-spec.uxlfoundation.org/specifications/oneapi/v1.1-rev-1/elements/onetbb/source/algorithms/functions/parallel_sort_func>

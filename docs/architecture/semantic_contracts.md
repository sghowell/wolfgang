# Wolfgang Semantic Contracts

This document defines the behavior that implementation, tests, adapters, and benchmarks must preserve. The implementation plan can change order or mechanics, but these contracts are source of truth for correctness.

## Representation

Wolfgang stores each Pauli term as packed `x` and `z` bit masks plus one complex coefficient.

```text
bit q represents qubit q
qubit 0 is bit 0
word index = q / 64
bit offset = q % 64

x=0, z=0 means I
x=1, z=0 means X
x=0, z=1 means Z
x=1, z=1 means Y

all Pauli phases are folded into coefficients
unused high bits in the final word are always zeroed
```

The host representation uses structure-of-arrays buffers:

```text
x:      num_terms * words uint64 values
z:      num_terms * words uint64 values
coeffs: num_terms complex128 values
```

`words == 0` is valid only when `num_qubits == 0`. For `num_qubits > 0`, `words == ceil(num_qubits / 64)`.

## Dense Label Convention

Dense labels follow Qiskit-style display order:

```text
Wolfgang internal qubit 0 = bit 0
dense label qubit 0 = right-most character
```

Therefore `"XYZ"` means:

```text
X on qubit 2
Y on qubit 1
Z on qubit 0
```

`to_labels()` must export labels in the same convention. Before `simplify()` or `sort()`, `to_labels()` preserves construction order.

## Sparse List Convention

Sparse terms are `(local_pauli_string, qubit_indices, coefficient)`.

```text
local_pauli_string[k] acts on qubit_indices[k]
```

Example:

```python
PauliSum.from_sparse_list([("ZX", [1, 4], 1.0)], num_qubits=5).to_labels()
```

returns:

```python
(["XIIZI"], np.array([1.0 + 0.0j]))
```

Sparse export order inside one term is ascending by qubit index. The exported local Pauli string follows that sorted qubit order.

## Construction Order And Canonical Order

Construction methods preserve input term order:

```text
from_labels -> term order follows input labels
from_sparse_list -> term order follows input triples
from_qiskit -> term order follows the source SparsePauliOp
from_openfermion -> term order follows the source QubitOperator iteration order only before simplify
```

Canonical order is produced by `simplify()` and `sort()`.

Default canonical order is lexicographic over packed words:

```text
x word 0, z word 0, x word 1, z word 1, ... x word n, z word n
```

`sort(by_weight=True)` orders by:

```text
1. ascending Pauli weight
2. default canonical order
```

`simplify()` must return terms in default canonical order after duplicate reduction and zero dropping.

## Empty, Zero, And Identity Operators

`PauliSum.from_labels([])` is invalid because `num_qubits` cannot be inferred.

Empty operators require an explicit constructor in the packed representation and I/O slice:

```python
PauliSum.empty(num_qubits: int) -> PauliSum
```

An empty operator has:

```text
num_qubits = requested value
num_terms = 0
x, z, coeffs = empty buffers
```

An identity term is a normal term with all-zero `x` and `z` masks. Its coefficient is stored in `coeffs`.

If `simplify()` drops every coefficient, it returns `PauliSum.empty(self.num_qubits)`.

## Coefficients And Dtypes

The initial semantic Python API, beginning in the packed representation and I/O slice, accepts Python numeric scalars and NumPy complex arrays. Internal storage is always `complex128`.

```text
float, int, complex, complex64, and complex128 inputs promote to complex128
to_labels() returns a complex128 NumPy array for coefficients
to_sparse_list() returns Python complex coefficients
statevector expectation accepts complex64 and complex128 inputs
complex64 statevectors may use complex64 arithmetic in future CUDA kernels, but the public return value is Python complex
```

Symbolic coefficients are out of scope.

## Tolerance Semantics

`simplify(atol=1e-12, rtol=0.0)` combines duplicate Pauli strings and drops a combined coefficient `c` when:

```text
abs(c) <= atol + rtol * max_abs_input_coefficient
```

`max_abs_input_coefficient` is computed over the input coefficients before duplicate reduction. For empty inputs it is `0`.

Negative tolerances are invalid and raise `ValueError`.

## Addition And Scalar Multiplication

`h1 + h2` requires equal `num_qubits`, concatenates terms in left-then-right order, and does not simplify automatically.

Scalar multiplication preserves term order. Multiplying by zero keeps the terms until `simplify()` is called.

## Multiplication

`h1 @ h2` means operator composition where `h2` acts first and `h1` acts second, matching matrix multiplication semantics.

For single-qubit terms:

```text
X @ Y =  i Z
Y @ X = -i Z
Y @ Z =  i X
Z @ Y = -i X
Z @ X =  i Y
X @ Z = -i Y
```

The product of sums emits products in nested-loop order before optional simplification:

```text
for lhs term in lhs order:
  for rhs term in rhs order:
    emit lhs_term @ rhs_term
```

`matmul(..., simplify=True)` simplifies and returns canonical order. `matmul(..., simplify=False)` preserves nested-loop product order.

## Commutation

Term-level full commutation uses:

```text
anticommutes iff parity(popcnt((x1 & z2) XOR (z1 & x2))) == 1
```

Term-level qubit-wise commutation conflict uses:

```text
qwc conflict iff ((x1 & z2) XOR (z1 & x2)) != 0
```

`commutes_with(other)` returns a boolean when both operands have one term. It returns a one-dimensional boolean NumPy array when exactly one operand has one term. It returns a two-dimensional boolean NumPy array for pairwise many-to-many checks only when the output element count is within the configured guardrail.

Default guardrail:

```text
max_commutation_matrix_entries = 100_000_000
```

Larger pairwise checks raise `ValueError` until a blocked commutation API is specified and implemented.

## Grouping

Grouping is deterministic and heuristic. Wolfgang does not claim optimal graph coloring.

`group_commuting(mode="qwc")` uses greedy largest-first grouping. Term ordering for greedy placement is:

```text
1. descending Pauli weight
2. default canonical order
```

`group_commuting(mode="full")` may use a precomputed noncommutation graph only when:

```text
num_terms <= max_terms_for_graph
```

`max_terms_for_graph` is a strategy ceiling, not permission to allocate an unsafe graph. Implementations must stream when the graph would exceed the implementation's documented memory-safety cap. The scalar CPU baseline caps precomputed full-grouping graph storage at 10,000,000 pair entries. Streaming greedy grouping checks a candidate term against all terms already in the candidate group.

Each returned group preserves the source coefficients and term masks. Group term order follows the deterministic greedy placement order, not necessarily original input order.

## Expectation Values

`expectation_statevector(psi)` requires:

```text
len(psi) == 2 ** num_qubits
psi dtype complex64 or complex128
num_qubits <= 63 for the initial CPU implementation
```

The return type is Python `complex`.

`expectation_z_counts(counts)` initially accepts Python dictionaries mapping dense bitstrings to counts. Dense bitstrings use the same display convention as dense labels: the right-most bit is qubit 0.

Only diagonal terms are accepted by `expectation_z_counts()` in the initial implementation. If any term has nonzero `x` mask, raise `ValueError`.

## Error Policy

Invalid user input raises `ValueError`. Missing optional integrations raise `ImportError` with an installation hint. Unsupported dtype or array layout raises `TypeError`.

C++ exceptions crossing the Python boundary must be translated to the corresponding Python exception type.

## Accelerator Backend Selection

Host `PauliSum` objects remain CPU-resident. Initial accelerator construction
accepts a backend selector:

```text
backend=None
backend="auto"
backend="cuda"
backend="hip"
backend="metal"
```

`None` and `"auto"` preserve single-backend compatibility when the accelerator
choice is unambiguous. Explicit `"cuda"`, `"hip"`, or `"metal"` selection
fails before allocation when the requested backend was not compiled. Invalid
selector values raise `ValueError`.

Accelerator-resident objects report their object-local backend as a stable
lowercase string. `DevicePauliSum.backend` and `DeviceCommutationMatrix.backend`
must return `"cuda"`, `"hip"`, or `"metal"` for valid accelerator allocations.
Backend identity is a property of the object, not a process-global active
backend.

The Apple Metal selector is source-build-only behind `WOLFGANG_ENABLE_METAL=ON`;
see `docs/architecture/apple_accelerator.md`. It initially covers transfers, pairwise commutation, and retained
`DeviceCommutationMatrix` count consumers. Unsupported accelerator operations
must raise a clear runtime error without falling back silently to CPU.

## Required Test Fixtures

Every implementation phase must preserve these tests:

```python
def test_dense_label_endianness():
    h = PauliSum.from_labels(["XYZ"])
    assert h.to_sparse_list() == [("ZYX", [0, 1, 2], 1.0 + 0.0j)]
```

Multiplication phase tests must include:

```python
assert (PauliSum.from_labels(["X"]) @ PauliSum.from_labels(["Y"])).simplify().to_sparse_list() == [("Z", [0], 1j)]
assert (PauliSum.from_labels(["Y"]) @ PauliSum.from_labels(["X"])).simplify().to_sparse_list() == [("Z", [0], -1j)]
```

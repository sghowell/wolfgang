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

`to_labels()` must export labels in the same convention. Before `simplify()`, `to_labels()` preserves construction order.

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

Canonical order is produced by `simplify()`. There is currently no public `sort()` API.

Default canonical order is lexicographic over packed words:

```text
x word 0, z word 0, x word 1, z word 1, ... x word n, z word n
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

Negative or nonfinite tolerances are invalid and raise `ValueError`.

With `rtol=0`, simplify is idempotent. With `rtol>0`, a second call recomputes
its input scale and can drop additional terms after duplicates have combined.
For example, `X + X + 0.15 Y` with `atol=0, rtol=0.1` first becomes
`2 X + 0.15 Y`, then `2 X`. This input-relative formula is intentional.

Simplify on every backend (CPU, CUDA, HIP and both Metal paths) rejects
nonfinite coefficient components with `ValueError` and raises `OverflowError`
if duplicate accumulation produces nonfinite components, including the
`inf + -inf = NaN` a tree-ordered GPU reduction can form. Finite components
are compared safely even when their complex magnitude exceeds the maximum
double. `tests/test_accelerator_simplify_parity.py` checks each available
accelerator against CPU simplify; CUDA and HIP may associate long duplicate
sums differently, so their coefficients match CPU to rounding. Whether an
intermediate sum overflows also depends on that order: duplicate sums within
about a factor of the run length of the largest double may raise
`OverflowError` on one backend and return a finite result on another.

## Addition And Scalar Multiplication

`h1 + h2` requires equal `num_qubits`, concatenates terms in left-then-right order, and does not simplify automatically.

Scalar multiplication preserves term order. Multiplying by zero keeps the terms until `simplify()` is called.

`h1 - h2` concatenates `h1` with the exactly negated terms of `h2` and does not
simplify. `-h` flips every coefficient's sign exactly; unlike `h * -1`, an
infinite component never becomes NaN. `h.adjoint()` conjugates coefficients and
keeps terms and order, since Pauli strings are Hermitian.

## Packed Array I/O

`PauliSum.from_arrays(x, z, coeffs, num_qubits)` and `to_arrays()` exchange the
packed representation directly:

```text
x, z: uint64, shape (num_terms, ceil(num_qubits / 64)), term-major
qubit q: bit q % 64 of word q // 64 (X sets x, Z sets z, Y sets both)
coeffs: shape (num_terms,); any integer, unsigned, real or complex dtype, stored as complex128
bits at or above num_qubits in each term's last word must be zero (ValueError)
```

`from_arrays` always copies and preserves term order. Word arrays with any dtype
other than `uint64` raise `TypeError` instead of being cast; wrong dimensions or
shapes raise `ValueError`. Non-contiguous inputs are accepted.

`to_arrays()` returns read-only NumPy views of the operator's storage. Each view
keeps the operator alive, and the operator is immutable, so the data never
changes underneath a view. `PauliSum.from_arrays(*op.to_arrays(), op.num_qubits)`
is structurally equal to `op`.

## Indexing And Equality

`len(h)` equals `h.num_terms`. `h[i]` returns a single-term `PauliSum` and accepts
negative indices; out-of-range indices raise `IndexError`, and non-integer keys
raise `TypeError`. `h[a:b:c]` returns the selected terms in slice order with
`num_qubits` preserved. Iteration yields single-term operators in term order.

`h1 == h2` is structural: equal `num_qubits` and the same terms in the same order
with coefficients equal under `==` (so `0.0 == -0.0` and NaN never compares
equal). It never simplifies, so reordered or unmerged operators compare unequal.
`PauliSum` is unhashable. Comparing with a non-`PauliSum` returns `False`.

`h1.equiv(h2, atol=1e-12, rtol=0.0)` is mathematical equality:
`(h1 - h2).simplify(atol, rtol).num_terms == 0`. Operators with different
`num_qubits` are never equivalent. Because `rtol` scales with the largest
coefficient in `h1 - h2` before duplicates merge, `rtol >= 1` makes any two
operators equivalent; keep `rtol` well below 1. Non-finite coefficients raise
`ValueError`, as in `simplify`.

Because `PauliSum` is a sequence of single-term operators, NumPy treats it as
nested data: `np.array([op1, op2], dtype=object)` recurses into the terms.
Build object arrays with `np.empty(n, dtype=object)` and item assignment.

## Commutators

`h1.commutator(h2)` returns `h1 @ h2 - h2 @ h1` and `h1.anticommutator(h2)`
returns `h1 @ h2 + h2 @ h1`. Two Pauli strings either commute or anticommute,
so for terms `a P` and `b Q`:

```text
P, Q anticommute: commutator gets 2ab PQ, anticommutator gets nothing
P, Q commute:     commutator gets nothing, anticommutator gets 2ab PQ
```

Pairs that contribute nothing are never generated. Each kept coefficient is
computed as `2 * (a * b)`, so doubling never overflows before the product. Any
non-finite coefficient in either operand raises `ValueError`, even if its pairs
would all cancel. With `simplify=True` (the
default) the result is simplified with default tolerances into canonical order.
With `simplify=False` the surviving products keep lhs-outer, rhs-inner order.
`max_intermediate_terms` bounds `len(h1) * len(h2)` before any work, as in
`matmul`.

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

A 1-D `psi` returns Python `complex`. A C-contiguous 2-D `psi` of shape
`(batch, 2 ** num_qubits)` is a batch of states and returns a `complex128` array
with one value per row. Other dimensions raise `ValueError`.

## Statevector Tools

All statevector tools use the dense-index convention: bit `q` of a basis index is
qubit `q`, so a Pauli string with masks `(x, z)` maps `|b>` to
`i^popcount(x & z) * (-1)^popcount(z & b) |b ^ x>`. They accept the same 1-D or
batched `complex64`/`complex128` inputs as `expectation_statevector` and require
`num_qubits <= 63`.

```text
apply_statevector(psi)    -> H @ psi as a new complex128 array of psi's shape
variance_statevector(psi) -> variance of H in psi / ||psi||: <H^dagger H>/N - |<H>|^2/N^2, N = <psi|psi>, clamped at 0
to_matrix()               -> dense complex128 (2^n, 2^n) array, entry [r, c] = <r|H|c>
to_matrix(sparse=True)    -> SciPy CSR matrix with sorted column indices
```

`apply_statevector` and `expectation_statevector` use `psi` as given;
`variance_statevector` normalizes and raises `ValueError` for a zero state.
Batched inputs are validated even when empty. `DevicePauliSum.expectation_statevector`
accepts only 1-D states. `to_matrix(max_entries=2**24)` raises `ValueError`
before allocating when `4 ** num_qubits` (dense) or
`2 ** num_qubits * distinct_x_masks + 2 ** num_qubits + 1` (sparse entries plus
row pointers) exceeds `max_entries`. The sparse
form omits entries that cancel to exactly zero. `to_matrix(sparse=True)` raises
`ImportError` with an installation hint when SciPy is absent; SciPy is never
imported at package import.

## Counts Estimation

`expectation_z_counts(counts)` initially accepts Python dictionaries mapping dense bitstrings to counts. Dense bitstrings use the same display convention as dense labels: the right-most bit is qubit 0.

Count weights may be finite nonnegative real numbers and must have positive total
weight. They are normalized after division by their largest value, so large
finite counts do not overflow the total. Rescaling all weights by the same
positive factor preserves the result within floating-point rounding.

Only diagonal terms are accepted by `expectation_z_counts()` in the initial implementation. If any term has nonzero `x` mask, raise `ValueError`.

`estimate_counts(counts, basis=None)` generalizes this to local measurement
bases. `basis` is a dense-label-order string of `X`, `Y`, `Z` or `I`, one per
qubit (right-most is qubit 0), naming the basis each qubit was measured in;
`None` means all `Z`. Each term must act on every qubit as `I` or as that
qubit's basis letter, otherwise `ValueError`. A term's eigenvalue on outcome `b`
is `(-1)^popcount(support & b)`: outcome bit 0 is the +1 eigenstate of the
qubit's basis Pauli, which is what the standard rotations produce (`h` for X,
`sdg` then `h` for Y, nothing for Z). It returns `(mean, standard_error)`:

```text
O(b)            = sum_t c_t * (-1)^popcount(support_t & b)
mean            = sum_b n_b O(b) / N                     (N = total shots)
standard_error  = sqrt(s^2 / N), s^2 = sum_b n_b |O(b) - mean|^2 / (N - 1)
                = sqrt(population variance / (N - 1))
```

Counts are raw shot counts; `standard_error` is NaN when `N <= 1`.

## Measurement Planning

`wolfgang_quantum.measurement` turns commuting groups into runnable
measurements. Gates are `(name, qubits)` tuples, names `h`, `s`, `sdg`, `cx`
(control first) and `cz`, in application order, using Wolfgang qubit indices.

```text
qwc_basis(group)              -> basis string for a qubit-wise commuting group; ValueError otherwise
diagonalizing_clifford(group) -> (gates U, Z-only PauliSum D) with D = U group U^dagger
allocate_shots(groups, total, variances=None) -> integer shots per group summing to total
to_qiskit_circuit(gates, n)   -> Qiskit QuantumCircuit (imports Qiskit lazily)
```

`qwc_basis` and `diagonalizing_clifford` are also `PauliSum` methods.
`diagonalizing_clifford` uses only `h` and `sdg` for qubit-wise commuting groups
and raises `ValueError` when terms do not all commute. `D` keeps the group's term
order, with each coefficient multiplied by the sign picked up under
conjugation, so `<psi|group|psi> = <U psi|D|U psi>` and
`D.estimate_counts(counts)` estimates the group from Z-basis counts of `U psi`.

`allocate_shots` minimizes the variance of the summed estimate,
`sum_g sigma_g^2 / N_g`, by choosing `N_g` proportional to `sigma_g`. Without
`variances`, `sigma_g` is bounded by the sum of `|c|` over the group's
non-identity terms, so identity-only groups get zero shots; with `variances`,
groups with zero variance get zero shots. Every group with nonzero weight gets
at least one shot when `total` allows. Rounding uses largest remainders, so
counts approximate the continuous optimum.

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

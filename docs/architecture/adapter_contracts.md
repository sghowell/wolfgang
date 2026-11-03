# Wolfgang Adapter Contracts

This document defines adapter behavior for optional ecosystem integrations. It applies to the Qiskit, OpenFermion and CUDA-Q adapters and to tests that compare Wolfgang semantics against external libraries.

## Shared Adapter Policy

Adapters are optional. Importing `wolfgang_quantum` must not import Qiskit, OpenFermion or CUDA-Q.

Missing optional dependencies raise `ImportError` with an installation hint when an adapter method is called:

```text
Install Wolfgang with the qiskit extra to use this adapter.
Install Wolfgang with the openfermion extra to use this adapter.
Install Wolfgang with the cudaq extra to use this adapter.
```

Adapters must:

```text
preserve num_qubits
preserve coefficients as complex128-compatible values
fold external Pauli phases into Wolfgang coefficients
avoid dense matrix materialization during conversion
round-trip empty and identity operators
keep conversion tests separate from core tests so optional dependencies can skip cleanly
```

## Qiskit Adapter

Supported input:

```text
qiskit.quantum_info.SparsePauliOp
```

Initial implementation path:

```text
1. extract labels and coefficients using public Qiskit APIs
2. fold any Qiskit Pauli phases into coefficients
3. call PauliSum.from_labels(labels, coeffs)
```

Optimized implementation path after tests are stable:

```text
1. extract PauliList x/z/phase data through public APIs
2. convert x/z arrays to Wolfgang packed representation
3. fold phase data into coefficients
4. compare optimized extraction against label extraction in tests
```

Export behavior:

```text
to_qiskit returns SparsePauliOp
exported Pauli strings have zero explicit Pauli phase
all phase information is represented in coefficients
dense label convention matches Qiskit display order
```

Required tests:

```text
single-qubit I, X, Y, Z round-trip
multi-qubit endianness round-trip
identity operator round-trip
empty operator round-trip when Qiskit supports explicit num_qubits for the chosen construction path
duplicate terms preserve operator semantics after simplify
phased Pauli inputs fold phases into coefficients
small random n <= 8 dense-matrix comparison
optional dependency missing raises ImportError with installation hint
```

Equality policy:

```text
raw term order should follow the source SparsePauliOp before simplify
semantic equality is checked after simplify or by dense matrix for small n
coefficients compare with operation-appropriate numeric tolerance
```

## OpenFermion Adapter

Supported input:

```text
openfermion.ops.QubitOperator
```

Input terms are mapped as:

```text
((0, "X"), (5, "X")) -> local_pauli_string "XX", indices [0, 5]
() -> identity term
```

`from_openfermion(op, num_qubits=None)` behavior:

```text
if num_qubits is provided, validate every term index is in range
if num_qubits is None, infer max_index + 1 from non-identity terms
if op contains only an identity term and num_qubits is None, infer num_qubits = 0
if op has no terms and num_qubits is None, return PauliSum.empty(0)
call PauliSum.from_sparse_list for non-empty terms
```

`to_openfermion()` behavior:

```text
export sparse terms
build QubitOperator terms with coefficients
identity terms export as the OpenFermion identity term
duplicates are accumulated without tolerance pruning; only exact zeros are omitted
zero-term PauliSum exports as an additive zero QubitOperator
```

Required tests:

```text
single-term round-trip
multi-term sparse round-trip
identity term round-trip
empty operator behavior
provided num_qubits validation
inferred num_qubits behavior
duplicate terms match after simplify
optional dependency missing raises ImportError with installation hint
```

Equality policy:

```text
raw iteration order may follow OpenFermion internals before simplify
semantic equality is checked after simplify
coefficients compare with operation-appropriate numeric tolerance
```

## CUDA-Q Adapter

Supported input:

```text
cudaq.SpinOperator
cudaq.SpinOperatorTerm (including a single Pauli such as cudaq.spin.z(5))
```

`PauliSum.from_cudaq(op, num_qubits=None)` behavior:

```text
terms keep CUDA-Q's order and coefficients exactly, including zero coefficients
qubit q of a CUDA-Q term maps to Wolfgang qubit q (both little-endian in statevector indices)
if num_qubits is None, infer one more than the largest qubit index named, including explicit identities such as spin.i(5)
operators that name no qubit (identity-only terms, empty operators) infer num_qubits = 0
coefficients that do not evaluate to numbers (parameterized ones) raise ValueError
indices at or above an explicit num_qubits raise ValueError
```

`to_cudaq()` behavior:

```text
returns cudaq.SpinOperator
CUDA-Q merges duplicate terms at the first occurrence's position, summing in input order
zero-term PauliSum exports as cudaq.SpinOperator.empty()
CUDA-Q operators carry no width; from_cudaq(op.to_cudaq(), op.num_qubits) restores it
finite coefficients transfer exactly; CUDA-Q scales each by 1.0 per Pauli, so an infinite component can make the other NaN
```

Both directions exchange whole operators through `SpinOperator.serialize()` and
the `SpinOperator(list[float])` constructor, whose layout CUDA-Q 0.11 through
0.16 share. Because CUDA-Q does not promise that layout, the adapter checks it
once per process against a known operator and otherwise converts through the
public term API (`get_binary_symplectic_form`, `degrees`,
`evaluate_coefficient`, and `cudaq.spin` products); it also falls back when a
serialized operator does not parse. Every conversion test runs on both paths,
and the serialized variant fails if the probe rejects the installed CUDA-Q.

Required tests:

```text
known operator both ways
random round trips at 1, 7, 64, 65 and 130 qubits, including terms whose Paulis are all above qubit 63
term and single-Pauli inputs
identity, empty and explicit-width operators; explicit identities set the inferred width
duplicates sum like simplify
zero coefficients kept; non-finite and signed-zero coefficients read exactly
parameterized coefficients raise ValueError
input validation and missing-dependency install hint
expectation values equal cudaq.observe on a double-precision simulator
```

## Version Policy

Do not claim support for a dependency version until it is exercised in CI or local validation.

Once adapters land, `pyproject.toml` should declare extras:

```text
qiskit = [...]
openfermion = [...]
cudaq = [...]
```

The minimum supported versions should be set to versions validated by tests, not guessed.
The initial OpenFermion adapter lower bound is `openfermion>=1.7.1`, with CI
covering `openfermion==1.7.1` and the latest compatible `openfermion>=1.7.1`
resolution.
The CUDA-Q adapter lower bound is `cudaq>=0.11`, the first release with the
current operator API, with CI covering `cudaq==0.11.0` and the latest
compatible resolution.

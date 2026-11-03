# Quickstart

## Construct and inspect an operator

```python
import numpy as np
from wolfgang_quantum import PauliSum

op = PauliSum.from_labels(
    ["XX", "YY", "ZI", "IZ"],
    np.array([0.5, 0.5, -1.0, -1.0], dtype=np.complex128),
)

labels, coefficients = op.to_labels()
print(labels)
print(coefficients)
```

`PauliSum` owns packed `x` and `z` bit masks plus `complex128` coefficients. Labels are an input and output format; the kernels operate on the packed masks.

## Algebra and canonicalization

```python
square = (op @ op).simplify(atol=1e-12)
labels, coefficients = square.to_labels()
```

`+` concatenates terms and does not simplify them. `@` performs phase-correct Pauli multiplication. `simplify()` combines identical packed terms, drops values under the documented tolerance, and returns deterministic canonical order.

## Commutation and grouping

```python
commuting = op.commutes_with(op)
groups = op.group_commuting()

print(commuting.shape)
print([group.num_terms for group in groups])
```

The dense pairwise commutation matrix has a guarded output size. In accelerator-heavy workflows, prefer a compact consumer operation where one exists over materializing a dense matrix larger than the task needs.

## Expectation value

```python
psi = np.zeros(4, dtype=np.complex128)
psi[0] = 1.0
value = op.expectation_statevector(psi)
print(value)
```

The statevector length must be exactly `2**num_qubits`, use `complex64` or `complex128`, and be C-contiguous.

## Optional adapters

```python
qiskit_op = op.to_qiskit()
round_trip = PauliSum.from_qiskit(qiskit_op)

spin_op = op.to_cudaq()                              # cudaq.SpinOperator
again = PauliSum.from_cudaq(spin_op, op.num_qubits)  # width is not stored by CUDA-Q
```

Install the matching extra before calling an adapter. Importing `wolfgang_quantum` does not import Qiskit, OpenFermion or CUDA-Q.

## Next

- [Pauli conventions](conventions.md)
- [Architecture](../guide/architecture.md)
- [Python API](../guide/python-api.md)
- [Expectation values](../user/expectation_values.md)

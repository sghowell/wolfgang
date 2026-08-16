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

`PauliSum` owns packed `x` and `z` bit masks plus `complex128` coefficients. Labels are convenient I/O—not the hot-path representation.

## Algebra and canonicalization

```python
square = (op @ op).simplify(atol=1e-12)
labels, coefficients = square.to_labels()
```

`+` concatenates terms without silently simplifying. `@` performs phase-correct Pauli multiplication. `simplify()` combines identical packed terms, drops values under the documented tolerance, and returns deterministic canonical order.

## Commutation and grouping

```python
commuting = op.commutes_with(op)
groups = op.group_commuting()

print(commuting.shape)
print([group.num_terms for group in groups])
```

Dense pairwise commutation has a guarded output size. For accelerator-heavy workflows, prefer compact consumer operations where available rather than materializing an unnecessarily large dense matrix.

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
```

Install the corresponding extra first. Importing `wolfgang_quantum` itself does not eagerly import Qiskit or OpenFermion.

## Next

- [Pauli conventions](conventions.md)
- [Architecture](../guide/architecture.md)
- [Python API](../guide/python-api.md)
- [Expectation values](../user/expectation_values.md)

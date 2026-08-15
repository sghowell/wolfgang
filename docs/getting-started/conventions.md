# Pauli conventions

Wolfgang represents each tensor-product Pauli string with symplectic binary masks `(x, z)`:

| Pauli | x | z |
|---|---:|---:|
| I | 0 | 0 |
| X | 1 | 0 |
| Y | 1 | 1 |
| Z | 0 | 1 |

Qubit zero occupies the least-significant packed bit. Human-readable dense labels follow the common convention that the rightmost character denotes qubit zero. For example, label `"XI"` applies `X` to qubit one and identity to qubit zero.

Each term is:

\[
c_j P_j = c_j \bigotimes_{q=0}^{n-1} P_{j,q},
\]

and a `PauliSum` is `sum_j c_j P_j`.

## Multiplication

Pauli multiplication is not string concatenation. The packed implementation computes output masks with XOR and derives the phase from anti-commuting local factors. `A @ B` means matrix-product order `A B`.

## Canonical order

`simplify()` sorts packed keys deterministically, combines equal terms, and applies the documented absolute/relative drop threshold. Input order is preserved by ordinary construction and addition; canonicalization is explicit rather than surprising.

## Sparse-list I/O

Sparse terms specify a local Pauli string and matching qubit indices. Exported indices are sorted in ascending qubit order. Duplicate indices and out-of-range indices are invalid.

## Numerical policy

Coefficients are stored as `complex128`. Inputs may be converted to this representation. Equality in examples should use an explicit tolerance where floating-point arithmetic is involved; structural/canonical ordering and commutation results are exact.

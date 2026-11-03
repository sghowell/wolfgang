# Usability And Kernel Campaign

Baseline: `d555868`. This campaign closes the usability and performance gaps
found in the November 2026 project review. Earlier decisions are inputs, not
constraints: where measurement shows an existing path or policy costs
correctness or speed, the slice replaces it and records why.

## Slices

Each slice is a separate `codex/` branch with failing tests first, the
repository validation entrypoint, an independent agent review, and fresh
hardware evidence for any accelerator or speed claim.

| # | Slice | Scope | Evidence required |
| --- | --- | --- | --- |
| 1 | CUDA simplify hardening | Restore the CUDA build broken by `ee4f95f`; replace the one-thread generic (>128 qubit) reduction with a parallel segmented reduce-by-key; use stable key sorts on the one- and two-word paths | H100 CUDA validation, CPU-equivalence tests across widths, sizes, duplicates and cancellation; serial vs parallel vs CPU timings; paired before/after timings for the specialized paths |
| 2 | Packed array I/O | `PauliSum.from_arrays` / `to_arrays` on `uint64` x/z words and `complex128` coefficients with packed-word validation; zero-copy device views for CUDA `DevicePauliSum` | Malformed-input and lifetime tests; string-boundary vs array-boundary timings |
| 3 | Operator algebra | Subtraction, negation, `len`, indexing and slicing, structural equality, tolerance equivalence, adjoint, and fused `commutator` / `anticommutator` that skip pairs which vanish | Algebraic identity and CPU-oracle tests; fused vs `matmul`-composed timings |
| 4 | Statevector tools | Apply an operator to a statevector, dense and sparse matrix export, batched expectation, variance | Dense-matrix oracle tests for small widths; memory guardrails |
| 5 | Measurement workflow | Qubit-wise group bases, expectation and standard error from counts in arbitrary local bases, diagonalizing Clifford circuits for fully commuting groups, variance-weighted shot allocation | Statevector and sampled-counts oracles; Clifford conjugation checked term by term |
| 6 | CPU simplify parallelism | oneTBB sort and duplicate reduction that keeps scalar canonical output and input-order sums | Bitwise scalar equivalence; paired timings on x86_64 and Apple Silicon against the scalar control and Qiskit |
| 7 | Device allocation reuse | Remove per-call allocation and synchronization costs that profiling shows dominate chained GPU operations, through private workspaces first and `_inplace` public methods where they pay off | Before/after Nsight Systems API breakdown and same-boundary timings |
| 8 | Metal hybrid simplify | GPU integer key sort and run detection with float64 coefficient reduction on the CPU over shared memory | Promotion only if it beats same-host CPU simplify at the measured sizes |

## Decisions carried into the campaign

- Coefficient storage stays `complex128`. A `complex64` container saves 17-25%
  of term memory, loses about nine significant digits in duplicate-cancelling
  reductions, and does not make Metal simplify competitive.
- Public CUDA streams and graphs stay out unless new profiling shows launch
  overhead dominates after allocation is removed (slice 7); the Campaign 10
  profile attributed cost to allocation and copies, which slice 7 targets.
- HIP-facing API additions are implemented where they share code with CUDA but
  are not claimed without MI300X evidence.

## Status

Slice 1 in progress.

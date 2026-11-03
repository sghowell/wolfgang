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
| 2 | Packed array I/O | `PauliSum.from_arrays` / `to_arrays` on `uint64` x/z words and `complex128` coefficients with packed-word validation (zero-copy CUDA `DevicePauliSum` views moved to slice 7 with the other device work) | Malformed-input and lifetime tests; string-boundary vs array-boundary timings |
| 3 | Operator algebra | Subtraction, negation, `len`, indexing and slicing, structural equality, tolerance equivalence, adjoint, and fused `commutator` / `anticommutator` that skip pairs which vanish | Algebraic identity and CPU-oracle tests; fused vs `matmul`-composed timings |
| 4 | Statevector tools | Apply an operator to a statevector, dense and sparse matrix export, batched expectation, variance | Dense-matrix oracle tests for small widths; memory guardrails |
| 5 | Measurement workflow | Qubit-wise group bases, expectation and standard error from counts in arbitrary local bases, diagonalizing Clifford circuits for fully commuting groups, variance-weighted shot allocation | Statevector and sampled-counts oracles; Clifford conjugation checked term by term |
| 6 | CPU simplify parallelism | oneTBB sort and duplicate reduction that keeps scalar canonical output and input-order sums | Bitwise scalar equivalence; paired timings on x86_64 and Apple Silicon against the scalar control and Qiskit |
| 7 | Device allocation reuse | Remove per-call allocation, discovery and synchronization costs that dominate small GPU operations, through a private pooled allocator first and `_inplace` public methods where they pay off; zero-copy CUDA `DevicePauliSum` views | Before/after Nsight Systems API breakdown and same-boundary timings |
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

Slice 1 merged with H100 evidence:
`docs/benchmarks/reports/cuda_simplify_paths_h100_2026-11-03.md`. Review closeout:
`docs/reviews/usability_kernel_campaign_2026-11-03.md`.

Slices 2 and 3 merged: `docs/benchmarks/reports/host_api_2026-11-03.md`.
Slices 4 and 5 merged (no speed claims; correctness-oracle evidence only).

Slice 6: oneTBB simplify, bit-identical to scalar, auto from 8,192 terms with
at least four oneTBB threads
(`docs/benchmarks/reports/cpu_parallel_simplify_2026-11-03.md`, two x86 hosts).
Deferred with reason:
- **Apple Silicon oneTBB timings.** oneTBB is not installed on the Apple
  Silicon development machine, and building it from downloaded source there
  was not permitted in this campaign. The kernel is portable C++ with no
  architecture-specific code; ARM timings wait for a host with oneTBB
  packaged.

Slice 8: Metal hybrid simplify promoted for one-word operators from 12,288
terms, where it beats same-host CPU simplify and the host bridge in every
measured combination
(`docs/benchmarks/reports/metal_hybrid_simplify_2026-11-03.md`, M4 Pro).
Deferred with reason:
- **Hybrid timings on oneTBB Metal builds.** Same blocker as the slice 6 Apple
  Silicon deferral: oneTBB is not installed on the Apple development machine.
  On such builds the bridge's CPU simplify is parallel and the hybrid's
  reduction is not, so the threshold is unmeasured there.
- **A device-resident float64 reduction.** Metal kernels cannot use `double`;
  an exact reduction would need software float64 or double-float arithmetic,
  and must still match CPU simplify bit for bit. The reduction stays on the
  CPU, which is why the public capability record remains `host_bridge`.

Slice 7: pooled CUDA allocations, cheap device selection, and fewer Thrust
synchronizations and scratch fills
(`docs/benchmarks/reports/cuda_allocation_reuse_h100_2026-11-03.md`, with an
Nsight Systems API breakdown).
Deferred with reason:
- **Zero-copy CUDA `DevicePauliSum` views.** These need backend-neutral pointer
  accessors implemented in the CUDA, HIP and Metal backends, plus an
  export-aware free path for pooled memory. HIP cannot be compiled or validated
  in this campaign's environment.
- **`_inplace` public methods.** Not added. Pooled allocation removed the
  allocation cost they were meant to avoid without new API surface.
- **Public streams and graphs.** These stay out. The remaining small-operator
  floor (about 120-145 us for device-resident simplify after this slice) comes
  from the sort and reduction kernels and the synchronizations that
  value-returning Thrust algorithms still need, so the next lever is a fused
  small-operator simplify kernel, not stream-aware APIs.

# Portability And Interop Campaign

Baseline: `7c69edc`. This campaign follows the usability and kernel campaign
(`docs/plans/usability_and_kernel_campaign_plan.md`). It closes that campaign's
deferrals where hardware now allows, keeps GPU builds from breaking unnoticed,
brings the HIP backend to the CUDA backend's correctness and allocation
behavior, and opens zero-copy interop with other GPU libraries. Earlier
decisions remain inputs, not constraints.

## Slices

Each slice is a separate `codex/` branch with failing tests first, the
repository validation entrypoint, an independent agent review, and fresh
hardware evidence for any accelerator or speed claim. CUDA evidence comes from
an H100, HIP evidence from an MI300X, and Metal evidence from an Apple M4 Pro.

| # | Slice | Scope | Evidence required |
| --- | --- | --- | --- |
| 1 | GPU build health | Compile-only CUDA (with oneTBB), HIP and Metal CI jobs, so a source break fails CI the way `ee4f95f` should have; a baseline HIP validation at `7c69edc` | Green CI jobs; MI300X `WOLFGANG_VALIDATE_HIP=1` result at the baseline |
| 2 | Accelerator simplify parity and HIP allocation | CUDA and HIP simplify follow the CPU numeric contract (`ValueError` for non-finite input, `OverflowError` for duplicate-sum overflow including tree-sum `inf + -inf`); HIP gains the CPU drop rule CUDA already has, `par_nosync` rocThrust execution, uninitialized scratch, a single-term path, MemoryError on out-of-memory, and cheap device selection on transfers (a stream-ordered pool was measured and kept as a benchmark opt-in) | One backend-neutral CPU-equivalence suite (coefficients to rounding) passing on CUDA, HIP and Metal; MI300X before/after timings at the same boundary |
| 3 | Zero-copy device arrays | Read-only DLPack and CUDA Array Interface export of a `DevicePauliSum`'s x words, z words and coefficients that keeps the owner alive, and construction from device arrays without a host round trip. Metal exports follow the existing `DeviceCommutationMatrix` Metal DLPack contract or stay disabled with a recorded reason | Lifetime and aliasing tests with a named consumer per backend (CuPy on CUDA; a ROCm consumer on HIP or the export stays disabled there); export-vs-copy timings |
| 4 | CUDA-Q interop | Converters between `PauliSum` and `cudaq.SpinOperator`, imported lazily behind an optional extra | Round-trip and expectation-equivalence tests in an adapter CI job (per `docs/architecture/testing_and_ci.md`) where CUDA-Q installs |
| 5 | Fused small-operator simplify | One-launch block-level sort and reduction for small operators on CUDA and HIP, removing the separate sort, reduce and synchronization costs that set the post-pooling floor | CPU bit-identity tests; H100 and MI300X timings against the pooled path; threshold sweep |
| 6 | Metal follow-ups | Parallel CPU reduction in the hybrid simplify (chunked at run boundaries, bit-identical); 4M and 16M-term inputs, splitting the sort across command buffers if one hits the GPU timeout; build-time precompiled kernels (`.metallib`) when Xcode's Metal compiler is available, to cut the first-call compile; Apple oneTBB timings if oneTBB is installed | M4 Pro timings; large-input correctness; first-call timings with and without the precompiled library |
| 7 | oneTBB in wheels | Bundle oneTBB in release wheels so `auto` reaches parallel simplify for pip users; update release, security and SBOM docs, and ship the Apache-2.0 notice | Wheel smoke tests on every wheel target; public artifact audit; before/after timings from installed wheels |
| 8 | Closeouts | Record `_inplace` methods as won't-do; measure oneTBB simplify peak memory | Peak-memory measurement on an x86 host |
| 9 | Streams and graphs | Profile after slice 5; where launch and synchronization overhead still dominate, capture fixed-shape small-operation sequences in CUDA/HIP graphs, and evaluate a public stream argument that pairs with slice 3's DLPack stream negotiation | Nsight Systems / rocprofv3 breakdowns before and after; same-boundary timings; ship only what measurably helps |
| 10 | Further performance | Profile-driven improvements across backends found after slices 1-9 | Paired same-host timings per change |
| 11 | Documentation and site | Review every user-facing doc, refresh the README and plots from current evidence, and publish the documentation site to GitHub Pages | Strict docs build; link and stale-claim audit; live site |

## Decisions carried into the campaign

- Coefficient storage stays `complex128`.
- Streams and graphs are evaluated in slice 9, after slice 5 removes the
  separate sort, reduce and synchronization costs, and ship only with evidence.
- HIP claims require MI300X evidence. An MI300X is available for this campaign.
- Publishing a release (tags, PyPI uploads) waits for explicit approval; slice 7
  prepares and validates wheels only.

## Status

Slice 1 merged: compile-only CUDA, HIP and Metal CI lanes; the MI300X baseline
(`7c69edc`, HIP 7.16) passes `WOLFGANG_VALIDATE_HIP=1 python scripts/validate.py
--profile core` (684 tests in the HIP run). Review closeout:
`docs/reviews/portability_interop_campaign_2026-11-03.md`.

Slice 2 merged: CUDA and HIP simplify follow the CPU numeric contract on
every reduction path; HIP small-operation simplify and matmul are 1.5-1.6x
faster and single-term simplify 8x
(`docs/benchmarks/reports/hip_parity_mi300x_2026-11-03.md`). A HIP
stream-ordered pool was measured slower than `hipMalloc` on ROCm and is kept
only as a benchmark opt-in.

Slice 5 merged: a fused single-block simplify for CUDA and HIP operators of up
to 256 qubits and 1,024 terms, 3.0-7.8x faster on H100 and 3.5-5.7x on MI300X,
with coefficients bit-identical to CPU
(`docs/benchmarks/reports/small_simplify_2026-11-03.md`). Every backend now
rounds the drop threshold's product separately.

Slice 6 merged: the Metal hybrid simplify reduces duplicate runs on all CPU
cores from 2^20 terms (1.0-1.2x end to end, bit-identical), ran 4M, 16M and
64M-term inputs without a GPU timeout, splits large sorts across command
buffers, and embeds precompiled kernels when Xcode's Metal compiler is
available (first call 3.1 ms against 3.9 ms)
(`docs/benchmarks/reports/metal_followups_2026-11-03.md`). Apple oneTBB
timings stay deferred: oneTBB is not installed on the M4 Pro.

Slice 3 merged: CUDA `DevicePauliSum.device_arrays()` hands the packed
buffers to CuPy and PyTorch without a copy (36 us against 8.8 ms through the
host at 1M terms), and `DevicePauliSum.from_device_arrays()` builds an operator
from device arrays on any stream (84 us against 21 ms)
(`docs/benchmarks/reports/device_arrays_h100_2026-11-03.md`). HIP and Metal
stay disabled until a consumer on those backends is validated.

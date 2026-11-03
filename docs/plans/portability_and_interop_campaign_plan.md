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
| 2 | Accelerator simplify parity and HIP allocation | CUDA and HIP simplify follow the CPU numeric contract (`ValueError` for non-finite input, `OverflowError` for duplicate-sum overflow including tree-sum `inf + -inf`); HIP gains the CPU drop rule CUDA already has, pooled allocations with out-of-memory trim-and-retry, and cheap device selection on transfers | One backend-neutral CPU-equivalence suite (coefficients to rounding) passing on CUDA, HIP and Metal; MI300X before/after timings at the same boundary |
| 3 | Zero-copy device arrays | Read-only DLPack and CUDA Array Interface export of a `DevicePauliSum`'s x words, z words and coefficients that keeps the owner alive, and construction from device arrays without a host round trip. Metal exports follow the existing `DeviceCommutationMatrix` Metal DLPack contract or stay disabled with a recorded reason | Lifetime and aliasing tests with a named consumer per backend (CuPy on CUDA; a ROCm consumer on HIP or the export stays disabled there); export-vs-copy timings |
| 4 | CUDA-Q interop | Converters between `PauliSum` and `cudaq.SpinOperator`, imported lazily behind an optional extra | Round-trip and expectation-equivalence tests in an adapter CI job (per `docs/architecture/testing_and_ci.md`) where CUDA-Q installs |
| 5 | Fused small-operator simplify | One-launch block-level sort and reduction for small operators on CUDA and HIP, removing the separate sort, reduce and synchronization costs that set the post-pooling floor | CPU bit-identity tests; H100 and MI300X timings against the pooled path; threshold sweep |
| 6 | Metal follow-ups | Parallel CPU reduction in the hybrid simplify (chunked at run boundaries, bit-identical), inputs above 1,048,577 terms, and Apple oneTBB timings if oneTBB is installed | M4 Pro timings; large-input correctness |
| 7 | oneTBB in wheels | Bundle oneTBB in release wheels so `auto` reaches parallel simplify for pip users; update release, security and SBOM docs, and ship the Apache-2.0 notice | Wheel smoke tests on every wheel target; public artifact audit; before/after timings from installed wheels |
| 8 | Closeouts | Record `_inplace` methods as won't-do; measure oneTBB simplify peak memory | Peak-memory measurement on an x86 host |

## Decisions carried into the campaign

- Coefficient storage stays `complex128`.
- Public CUDA streams and graphs stay out unless profiling after slice 5 shows
  launch overhead dominates.
- HIP claims require MI300X evidence. An MI300X is available for this campaign.
- Publishing a release (tags, PyPI uploads) waits for explicit approval; slice 7
  prepares and validates wheels only.

## Status

Slice 1 in progress: GPU build CI on draft PR #16; the MI300X baseline
(`7c69edc`, HIP 7.16) passes `WOLFGANG_VALIDATE_HIP=1 python scripts/validate.py
--profile core` (684 tests in the HIP run).

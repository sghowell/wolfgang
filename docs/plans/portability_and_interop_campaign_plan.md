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
| 1 | GPU build health | Compile-only CUDA and HIP CI jobs in pinned toolkit containers, so a source break fails CI the way `ee4f95f` should have; a baseline HIP validation at `7c69edc` | Green CI jobs; MI300X `WOLFGANG_VALIDATE_HIP=1` result at the baseline |
| 2 | HIP parity | Port the CUDA fixes HIP never received: the CPU drop rule (no NaN threshold from `0 * inf`, no `abs` overflow), up-front non-finite rejection, runs seeded with their first coefficient (signed zeros), pooled allocations with out-of-memory trim-and-retry, and cheap device selection on transfers | CPU bit-identity tests on HIP; MI300X before/after timings at the same boundary |
| 3 | Zero-copy device arrays | Read-only DLPack and CUDA Array Interface export of a `DevicePauliSum`'s x words, z words and coefficients that keeps the owner alive, and construction from device arrays without a host round trip | Lifetime and aliasing tests with a named consumer per backend (CuPy on CUDA; a ROCm consumer on HIP or the export stays disabled there); export-vs-copy timings |
| 4 | CUDA-Q interop | Converters between `PauliSum` and `cudaq.SpinOperator`, imported lazily | Round-trip and expectation-equivalence tests where CUDA-Q installs |
| 5 | Fused small-operator simplify | One-launch block-level sort and reduction for small operators on CUDA and HIP, removing the separate sort, reduce and synchronization costs that set the post-pooling floor | CPU bit-identity tests; H100 and MI300X timings against the pooled path; threshold sweep |
| 6 | Metal follow-ups | Parallel CPU reduction in the hybrid simplify (chunked at run boundaries, bit-identical), inputs above 1,048,577 terms, and Apple oneTBB timings if oneTBB is installed | M4 Pro timings; large-input correctness |
| 7 | oneTBB in wheels | Bundle oneTBB in release wheels so `auto` reaches parallel simplify for pip users | Wheel smoke tests on every wheel target; before/after timings from installed wheels |
| 8 | Closeouts | Record `_inplace` methods as won't-do; measure oneTBB simplify peak memory | Peak-memory measurement on an x86 host |

## Decisions carried into the campaign

- Coefficient storage stays `complex128`.
- Public CUDA streams and graphs stay out unless profiling after slice 5 shows
  launch overhead dominates.
- HIP claims require MI300X evidence. An MI300X is available for this campaign.
- Publishing a release (tags, PyPI uploads) waits for explicit approval; slice 7
  prepares and validates wheels only.

## Status

Not started.

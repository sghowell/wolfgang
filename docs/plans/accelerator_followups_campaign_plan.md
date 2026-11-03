# Accelerator Follow-Ups Campaign

Baseline: `ad6f7a8`. This campaign closes the follow-ups the portability and
interop campaign (`docs/plans/portability_and_interop_campaign_plan.md`) left
open, adds the release-readiness work that bundling oneTBB in wheels exposed,
and finishes with a documentation pass. Earlier decisions remain inputs, not
constraints.

## Slices

Each slice is a separate `codex/` branch with failing tests first, the
repository validation entrypoint, an independent agent review, and fresh
hardware evidence for any accelerator or speed claim. CUDA evidence comes from
an H100, HIP evidence from an MI300X, and CPU evidence from the hosts each
report names.

| # | Slice | Scope | Evidence required |
| --- | --- | --- | --- |
| 1 | Device-side simplify counts (CUDA) | Simplify beyond the fused kernel keeps the input maximum, drop rule, run count and survivor count on the device, so one synchronization replaces the eight the slice 9 profile found; the CUDA and HIP pipeline lives in one shared header. `matmul` reaches it through simplify | CPU-equivalence suite; Nsight Systems breakdown before and after; paired H100 timings against a baseline build |
| 2 | HIP pipeline and per-operation workspace | The shared pipeline on HIP, with its scratch drawn from one reusable per-thread workspace instead of one `hipMalloc` per buffer | CPU-equivalence suite on the MI300X; paired timings against a baseline build |
| 3 | Pinned transfers | Host-device copies of operators and host statevectors staged through pinned memory where that is measurably faster | Paired transfer timings on H100 and MI300X |
| 4 | CPU thread cap | A documented way to cap oneTBB's threads (environment variable and Python call), so processes in pools, MPI ranks and CPU-limited containers do not each start one thread per core; auto dispatch uses the capped count | Tests for the cap and the dispatch floor; timings with and without the cap under oversubscription |
| 5 | oneTBB in the SBOM | The release software bill of materials lists the bundled oneTBB runtime (version, commit, license) beside the Python dependencies | Release workflow dry run with the generated SBOM checked by the release-wheelhouse checks |
| 6 | Multi-GPU | Peer access for pooled exports and multi-GPU tests | Needs a host with two or more GPUs; the current H100 and MI300X hosts have one each, so this slice waits for hardware |
| 7 | Documentation and figures | README and site figures for absolute timings, speedups against baselines and other libraries, and performance over time across the dated reports; fix diagrams that do not render on the site; review every user-facing doc | Strict docs build; diagrams checked in a browser on the live site; link and stale-claim audit |

## Decisions carried into the campaign

- Coefficient storage stays `complex128`.
- No CUDA graphs and no public stream argument (slice 9 of the previous
  campaign); removing the host round trips in slice 1 lets launches queue
  without either.
- Floating reductions may use an implementation-defined parallel order within
  documented tolerances (`docs/architecture/cuda_backend.md`), so the slice 1
  pipeline may associate duplicate sums differently from the Thrust path it
  replaces; canonical order, the drop rule and the error contract do not
  change.
- Publishing a release (tags, PyPI uploads) waits for explicit approval.

## Status

Planned.

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
| 1 | Device-side simplify counts (CUDA) | Simplify beyond the fused kernel keeps the input maximum, drop rule, run count and survivor count on the device, so one wait before sizing the output replaces the eight the slice 9 profile found; the CUDA and HIP pipeline lives in one shared header. `matmul` reaches it through simplify | CPU-equivalence suite; Nsight Systems breakdown before and after; paired H100 timings against a baseline build |
| 2 | HIP pipeline and per-operation workspace | The shared pipeline on HIP, with its scratch drawn from one reusable per-thread workspace instead of one `hipMalloc` per buffer | CPU-equivalence suite on the MI300X; paired timings against a baseline build |
| 3 | Pinned transfers | Host-device copies of operators and host statevectors staged through pinned memory where that is measurably faster | Paired transfer timings on H100 and MI300X |
| 4 | CPU thread cap | A documented way to cap oneTBB's threads (environment variable and Python call), so processes in pools, MPI ranks and CPU-limited containers do not each start one thread per core; auto dispatch uses the capped count | Tests for the cap and the dispatch floor; timings with and without the cap under oversubscription |
| 5 | oneTBB in the SBOM | The release software bill of materials lists the bundled oneTBB runtime (version, commit, license) beside the Python dependencies | Release workflow dry run with the generated SBOM checked by the release-wheelhouse checks |
| 6 | Multi-GPU | Peer access for pooled exports and multi-GPU tests | Needs a host with two or more GPUs; the current H100 and MI300X hosts have one each, so this slice waits for hardware |
| 7 | Documentation and figures | README and site figures for absolute timings, speedups against baselines and other libraries, and performance over time across the dated reports; fix diagrams that do not render on the site; review every user-facing doc, including an editing pass so the prose reads as written by an engineer and technical writer, without stock phrasing, inflated adjectives, reflexive lists of three, bolded lead-ins, hedging or filler transitions | Strict docs build; diagrams checked in a browser on the live site; link and stale-claim audit; a phrase-pattern scan over README and `docs/` plus an independent editorial review |

## Follow-ups found during the campaign

- The four-thread oneTBB floor for auto dispatch was never measured. With the
  slice 4 cap, oneTBB at two and three threads beat scalar simplify by 1.6x and
  2.4x at 500,000 terms on one host. Measure it on more hosts and sizes,
  separately for simplify (which could take its own floor) and pairwise
  commutation (where the alternative is a SIMD kernel), before moving it.

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

Slice 1 merged: CUDA simplify beyond the fused kernel runs the device-count
pipeline in `src/detail/device_simplify_pipeline.cuh`, which waits for the
GPU once before sizing its output instead of eight times; `simplify` on 4,096 terms takes 108 us instead of 188 us and
`matmul` on 128 x 128 terms 123 us instead of 202 us on an H100
(`docs/benchmarks/reports/cuda_device_counts_h100_2026-11-03.md`). Pinned
per-thread result buffers are freed at thread exit. Review closeout:
`docs/reviews/accelerator_followups_campaign_2026-11-03.md`.

Slice 2 merged: HIP simplify runs the same pipeline with hipCUB, and its
scratch and rocThrust temporaries come from a per-thread block cache; the
pipeline alone was slower than rocThrust at 50,000 terms because freeing its
multi-megabyte scratch cost up to about 100 us per call. On an MI300X,
`simplify` on 4,096 terms takes 98 us instead of 182 us and device-resident
simplify is 1.6-2.7x faster from 50,000 to 1M terms
(`docs/benchmarks/reports/hip_device_counts_mi300x_2026-11-03.md`).

Slice 3 merged: pinned staging measured faster only for device-to-host copies.
`to_host` stages through pinned buffers from a process-wide pool (CUDA at
every size, HIP up to 64 KiB) and advises large results for huge pages,
because page faults on fresh host memory, not PCIe, dominated large copies on
the KVM test hosts. `to_host` is 1.9-3.2x faster on an H100, and on an MI300X
1.8-1.9x faster for small operators and 2.2-3.2x at 1M (24 qubits) and 4M
terms; `to_device` keeps the driver's pageable path, which beat staging
(`docs/benchmarks/reports/host_transfers_2026-11-03.md`).

Slice 4 merged: `set_num_threads()`, `get_num_threads()` and
`WOLFGANG_NUM_THREADS` cap the oneTBB CPU kernels for the whole process, and
auto dispatch's four-thread floor counts the capped threads. Inside a 4-CPU
quota on a 26-vCPU host a cap of 4 made simplify 1.25x faster on average with
a 2.6x shorter 90th percentile; without a quota, oversubscription cost nothing
measurable (`docs/benchmarks/reports/cpu_threads_2026-11-03.md`).

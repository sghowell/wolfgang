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
| 6 | Multi-GPU | Peer access for pooled exports and multi-GPU tests | A host with two or more GPUs (two NVLink-connected H100s) |
| 7 | Documentation and figures | README and site figures for absolute timings, speedups against baselines and other libraries, and performance over time across the dated reports; fix diagrams that do not render on the site; review every user-facing doc, including an editing pass so the prose reads as written by an engineer and technical writer, without stock phrasing, inflated adjectives, reflexive lists of three, bolded lead-ins, hedging or filler transitions | Strict docs build; diagrams checked in a browser on the live site; link and stale-claim audit; a phrase-pattern scan over README and `docs/` plus an independent editorial review |

## Follow-ups found during the campaign

- The four-thread oneTBB floor for auto dispatch was never measured. With the
  slice 4 cap, oneTBB at two and three threads beat scalar simplify by 1.6x and
  2.4x at 500,000 terms on one host. Measure it on more hosts and sizes,
  separately for simplify (which could take its own floor) and pairwise
  commutation (where the alternative is a SIMD kernel), before moving it.
  Taken up as slice 8 (approved rule: lower each kernel's floor to the smallest
  thread count at which oneTBB is at least as fast as the alternative from the
  switch point up on every host measured, and raise the switch point instead
  where oneTBB loses only near it).
- Host pairwise commutation filled a zeroed vector that the binding then
  copied into NumPy. Taken up after slice 8 as a follow-up.

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

Slice 5 merged: the release CycloneDX SBOM lists the bundled oneTBB runtime
(version, source commit, `Apache-2.0 AND BSD-3-Clause`) and the vendored DLPack
header (version, SHA-256) next to the Python dependencies, added by `scripts/sbom_bundled_components.py` from the
files that pin them and checked against the CycloneDX schema in the release
workflow.

Slice 7 merged: the README, the site home page, the performance guide and the
evidence index show three figures that `scripts/render_readme_figures.py`
draws from committed benchmark data (simplify time per host, each accelerator
beside that host's CPU; Wolfgang against Qiskit, OpenFermion and cuStateVec;
per-call GPU times across the November campaigns), and
`tests/test_readme_figures.py` fails when one is stale. Mermaid diagrams render
on the site again: the initializer looked only for `pre.mermaid`, while the
fence format emits `div.mermaid`. The user and engineering docs were edited
for plain, specific prose, `docs/quality/documentation_standards.md` gained a
prose-patterns section, and the benchmark protocol's README-plot rules now
describe the three figures.

Slice 6 merged: the first multi-GPU test run, on two NVLink-connected H100s,
found three problems, all fixed. A peer kernel reading an exported buffer
faulted with an illegal address; the first export from a device now maps its
memory pool into every peer that can access it (26 ms once). Device calls held
the GIL, so two threads on two GPUs ran no faster than one; CUDA calls now
release it and reach 1.88-1.98 times one thread's throughput, with a per-device
lock holding same-device threads at 0.95-0.99 times one thread (0.99-1.02
before; 0.63-0.65 without the lock). A process that used
only GPU 1 created a 516 MiB context on GPU 0; device switches now go back
only to a device that has a context. Single-GPU call overhead is unchanged
(`docs/benchmarks/reports/cuda_multi_gpu_h100_2026-11-03.md`).

Slice 8 (oneTBB thread floors) merged: measured on an EPYC 9654 and a Xeon
8480+ pinned to one core's two hyperthreads, two cores, three and four vCPUs
(and the Xeon unpinned), and on an M4 Pro. Simplify runs on oneTBB from two
threads. The oneTBB commutation kernel ran a scalar loop per thread, at
0.23-0.26 times one AVX-512 thread with four threads; each chunk now runs the
best SIMD row kernel, and the single-threaded kernels are 1.02-1.73x faster.
Pairwise commutation uses oneTBB from two threads (three over AVX2 or AVX-512),
and one-word operators on AVX2 or AVX-512 from 2048 x 2048 entries
(`docs/benchmarks/reports/cpu_thread_floor_2026-11-03.md`).

Host commutation follow-up merged: the kernels write straight into the NumPy
result, and `PauliSum.commutes_with_into` fills a caller-owned array. The old
path's two buffers per call made glibc fault pages back in, depending on heap
layout; on an EPYC 9654, auto on four vCPUs is 1.4-10x faster for one- and
two-word operators and one AVX-512 thread 1.07-5.8x, and on an M4 Pro auto
with oneTBB 1.2-2.4x. That cost was what slice 8's one-word 2048 x 2048 threshold had
measured, so one-word operators on AVX2 or AVX-512 now switch to oneTBB at
331,776 entries like wider ones
(`docs/benchmarks/reports/host_commutation_into_2026-11-03.md`).

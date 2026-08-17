# Wolfgang Cross-Backend Kernel Performance Campaign Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Turn the checked CUDA, ROCm/HIP, CPU, and Apple Metal analyses at source commit `d14b4960a5197485e41d81a5dc426af5fce7cbae` into one ranked, falsifiable optimization campaign that improves user-visible latency and throughput without broadening public semantics or renting hardware before local prestaging is complete.

**Architecture:** Treat this as a cross-backend evidence-and-ranking campaign, not a kernel-coding sprint. The first wave freezes immutable baselines, benchmark schemas, correctness gates, and cheap local prototypes that can be developed on CPU and Apple Silicon without paid hardware. Only after those results show repeatable signal do later waves spend H100, MI300X, A100, or RTX PRO 6000 Blackwell time on backend-specific kernel or boundary work. Boundary/API wins rank above kernel-internal micro-optimization when the checked reports already show allocation, synchronization, host materialization, and reuse gaps dominating retained workloads.

**Tech Stack:** C++20, nanobind, Python benchmark harnesses, `scripts/validate.py`, deterministic benchmark JSON, Compute Sanitizer for CUDA, `rocprof` for HIP, local Apple Metal validation on Apple Silicon, and checked reports under `docs/benchmarks/`.

---

## Campaign status

Status: planned_cross_backend_execution_handoff

Immutable source baseline:

```text
commit: d14b4960a5197485e41d81a5dc426af5fce7cbae
branch baseline: main at d14b4960a5197485e41d81a5dc426af5fce7cbae
production-kernel rule: do not alter production kernels during this planning slice
hardware rule: do not rent hardware until all first-wave local prestaging tasks are complete and reviewed
```

Parent evidence inputs to synthesize before any implementation branch:

```text
CUDA parent analysis: t_0e70156a
ROCm/HIP parent analysis: t_2465fc9e
CPU + Apple Metal parent analysis: t_ebfe34d2
benchmark protocol: docs/benchmarks/protocol.md
backend contracts: docs/architecture/backend_neutral_accelerators.md
latest CUDA report: docs/benchmarks/reports/cuda_cross_architecture_campaign10_2026-04-29.md
latest ROCm report: docs/benchmarks/reports/rocm_campaign8_architecture_readiness_2026-05-01.md
latest Apple Metal report: docs/benchmarks/reports/apple_metal_optimization_campaign8_2026-05-07.md
CPU Apple Silicon evidence: docs/benchmarks/reports/cpu_phase9_apple_silicon_m4_pro_2026-04-25.md
```

## What the evidence already says

### Shared conclusion

The best remaining wins are mostly boundary and lifetime wins, not generic “optimize the kernel” folklore.

```text
CUDA: host output, allocation, temp storage, and generic multiword simplify dominate more than launch overhead
ROCm/HIP: retained paths already win; host materialization, synchronization, and fresh per-call buffers remain the main headroom
Apple Metal: device-resident compact consumers can be reasonable, but transfer-inclusive paths and per-call allocation still dominate host-facing work
CPU: pairwise commutation dispatch is good already; remaining cheap win is dispatch-policy correction for small full-grouping cases, not broad kernel rewrites
```

### Workload split that must stay explicit

```text
small workloads: CPU selector policy and Metal/host bridge costs dominate; regressions here are easy to create and hard to justify
medium/large retained workloads: buffer reuse, compact consumers, and avoiding dense host materialization dominate on CUDA and ROCm/HIP
large multiword simplify workloads: generic sort/reduce structure dominates on CUDA and ROCm/HIP; current CPU and Metal evidence does not justify treating them as the same optimization slice
expectation workloads: host-to-device statevector movement and accumulation strategy matter more than tiny instruction tuning
```

## Global ranking by expected user-visible speedup per engineering plus hardware cost

| Rank | Experiment family | Type | Shared vs specific | Primary workload | Why it ranks here |
| --- | --- | --- | --- | --- | --- |
| 1 | Retained commutation result reuse and compact-consumer-first boundaries | boundary/API | shared CUDA + HIP + Metal direction, backend-specific implementation | medium/large graph, grouping, count, degree consumers | Strongest checked evidence across all accelerator parents: host materialization and allocation dominate more than launch or arithmetic. |
| 2 | Workspace / scratch / output lifetime reuse for repeated accelerator calls | boundary/internal | shared pattern across CUDA + HIP + Metal | repeated simplify, commutation, retained output reuse loops | CUDA and HIP both show per-call temporaries as first-order cost; Metal also shows allocation gap between allocating and reused device outputs. |
| 3 | CPU auto-dispatch selector split for small full-grouping and shape-sensitive cases | boundary/policy | CPU-specific but local and cheap | small full-grouping on Apple Silicon and other CPUs | Fast, cheap, locally testable; local evidence already shows auto/neon can lose to scalar on a small full-grouping case. |
| 4 | Multiword simplify algorithm path for CUDA + HIP | kernel-internal | shared CUDA + HIP direction | words > 2 simplify, duplicate reduction | Valuable on the right workloads, but lower rank because it is more engineering-heavy and less universally user-visible than reuse/boundary work. |
| 5 | Expectation statevector residency and reduced accumulation overhead | boundary + kernel | shared CUDA + HIP, contract-sensitive | statevector expectation with repeated state reuse | Evidence is meaningful, but API/lifetime semantics are riskier and should wait for first-wave boundary prestaging. |
| 6 | Metal device-output reuse for commutation consumers | boundary/internal | Metal-specific | Apple local retained commutation workflows | Important local quality-of-life win, but narrower install base and lower ceiling than rank 1-2 shared accelerator work. |
| 7 | Metal lower-pass deterministic simplify design | kernel-internal | Metal-specific | one-word simplify candidate only | Evidence says current candidate is still slower than CPU and dominated by 108-139 dispatches; keep experimental until a lower-pass design exists. |
| 8 | Generic launch-overhead, CUDA Graph, public stream, or CSR-scatter work | boundary/API or kernel folklore | backend-specific and mostly rejected | n/a | Latest checked CUDA and ROCm decisions explicitly do not support prioritizing this now. |

## Ranking notes grounded in checked evidence

```text
rank-1 evidence:
- CUDA Campaign 10 compact graph consumer at 8192x8192 is 0.001043 s on A100 and 0.000558 s on RTX PRO 6000 Blackwell, versus 0.015406/0.006399 s transfer-inclusive host-facing rows
- ROCm parent analysis reports host copy at 32.8x reuse-kernel cost and allocation at 2.55x reuse-kernel cost on large retained commutation rows
- Metal parent analysis reports transfer-inclusive commutation remaining about 6x-24x slower than CPU, while compact consumers become reasonable only when results stay on device and buffers are reused

rank-2 evidence:
- CUDA simplify and retained commutation both still pay temp traffic and per-call allocation costs in checked reports
- HIP `HipTemporaryWorkspace` exists but is unused; simplify and compact consumers still allocate per call
- Metal parent analysis explicitly calls out the gap between `metal_device_output_allocating` and `metal_device_output_reused`

rank-3 evidence:
- local Apple Silicon full-grouping case: scalar 26.4 us versus auto/neon 28.9 us
- pairwise commutation remains strongly in favor of auto/neon, so the selector split must be shape-specific rather than a broad rollback

rank-4 evidence:
- CUDA multiword simplify remains limited by generic index sort and generic reduction kernel
- HIP campaign 4 fixed the pathological generic multiword path but still retained a generic sorted-index reduce-by-key structure

rank-5 evidence:
- HIP expectation transfer-inclusive cost is about 1.9x resident on a checked row and much worse on profiler rows
- CUDA expectation parent analysis still points at host-entry statevector allocation/copy and atomic accumulation

rank-7 evidence:
- Apple Metal Campaign 8 shows 108-139 dispatches for checked simplify rows and the candidate remains slower than same-host CPU default on every measured row
```

## Immutable benchmark and correctness contract

### Required immutable baselines

All experiments compare against this exact checked baseline first:

```text
source revision: d14b4960a5197485e41d81a5dc426af5fce7cbae
CPU baseline: current auto dispatch policy and forced scalar fallback
CUDA baseline: latest retained compact-consumer and transfer-inclusive rows from Campaign 10 / Campaign 11 evidence family
ROCm/HIP baseline: latest retained MI300X source-build rows from Campaign 7 / Campaign 8 evidence family
Metal baseline: Apple Metal Campaign 8 timing-decomposed simplify evidence and current retained commutation paths
```

### Minimum benchmark schema for every new row

Every new campaign row must record:

```text
campaign
experiment_branch
baseline_git_revision
candidate_git_revision
backend
boundary
operation
profile
dataset_name
num_qubits
num_terms or lhs_terms/rhs_terms
term_weight or term_weight_distribution
duplicate_rate or duplicate-pressure descriptor
packed_words
random_seed
warmup_count
repeat_count
median_seconds
correctness_checked
correctness_oracle
hardware_host_label
tooling_status
```

Boundary must be one of:

```text
cpu_only
transfer_inclusive
device_resident
reused_device_output
compact_consumer
private_candidate
profiler_only
status_only
```

### Correctness and sanitizer gates

No performance row is promotable unless the corresponding correctness lane passes on the same revision.

```text
CPU changes: forced-scalar equivalence plus existing semantic pytest coverage
CUDA changes: CPU/CUDA equivalence, Compute Sanitizer memcheck, and targeted initcheck/synccheck/racecheck for touched kernels when the kernel body changes
ROCm/HIP changes: CPU/HIP equivalence and rocprof availability or exact profiler-unavailable reason
Metal changes: CPU-vs-Metal equality on every changed row and `FASTPAULI_VALIDATE_METAL=1 uv run python scripts/validate.py`
Docs-only or benchmark-schema-only changes: `git diff --check`, focused pytest, and repo-local validation
```

### Repeated-evidence promotion rule

Promote only on repeated evidence.

```text
single-seed or single-run wins are never enough
for timing-only experiments, require 3 independent reruns on the same hardware lane and use mean-of-medians as the promotion metric
for branch promotion, candidate mean must beat baseline mean on the target workload family
small-workload regression budget: reject if any documented small workload regresses by >5% unless the plan already carved out an explicit selector split
medium/large retained-workload target: require >=10% win to justify extra complexity
API-surface-expanding work: require both a measurable win and an accepted semantic contract before merge
```

## Affected-backend rerun rules

Rerun only the lanes actually affected by the change, plus the required CPU oracle lane.

```text
CPU selector-only changes:
- rerun bench_cpu_dispatch.py
- rerun bench_cpu_thresholds.py
- rerun full-grouping rows on the local CPU host
- rerun Metal same-host rows only if shared host-side grouping code changed

shared benchmark-schema or docs changes:
- rerun focused pytest for affected benchmark/report tests
- rerun scripts/validate.py locally
- do not spend accelerator time

CUDA-only kernel or workspace changes:
- rerun CUDA correctness + touched benchmark rows on one NVIDIA host
- rerun non-H100 NVIDIA lane only if portability-sensitive code or compiled-architecture assumptions changed
- do not rerun MI300X or Metal

HIP-only kernel or workspace changes:
- rerun HIP correctness + touched benchmark rows on MI300X
- do not rerun CUDA or Metal unless shared contract code changed

shared accelerator boundary contract changes:
- rerun local CPU control
- rerun local Metal if Apple path touched
- rerun one CUDA lane and one HIP lane after local prestaging is accepted
```

## Hardware matrix and capacity windows

Do not book paid hardware before first-wave prestaging is green and reviewed.

| Lane | Purpose | Capacity window after prestaging | Notes |
| --- | --- | --- | --- |
| Local Apple M4 Pro | CPU dispatch, docs/tests, Metal local prototypes | immediate, 0 paid hours | default first-wave development host |
| H100 | CUDA retained baseline and sanitizer/profiler follow-up | 4-6 focused hours | use only after a CUDA-specific candidate survives local/schema review |
| A100 | non-H100 CUDA portability confirmation | 2-3 hours | use only for portability-sensitive CUDA changes |
| RTX PRO 6000 Blackwell | second NVIDIA portability lane | 2-3 hours | use only when architecture sensitivity matters |
| MI300X | HIP retained baseline and profiler-backed reopen work | 4-6 focused hours | notify before booking capacity; do not consume capacity for same-host repetition without a retained-operation bottleneck |

MI300X operator rule:

```text
before any MI300X booking, notify the operator that first-wave prestaging is complete, name the exact retained operation to rerun, name the expected bottleneck, and name the rejection criterion
```

## First wave: minimal experiments that can be developed locally before paid hardware

These are the only experiments that should start immediately.

### Wave 1A: Freeze the cross-backend benchmark contract and ranking harness

**Objective:** Make every later result comparable and reviewable before any kernel changes.

**Likely files:**
- Modify: `docs/benchmarks/protocol.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/research/provenance.md`
- Modify: `AGENTS.md`
- Create: `docs/plans/wolfgang-kernel-performance-campaign.md`
- Create: `tests/test_wolfgang_kernel_performance_campaign_plan.py`

**Branch:** `exp/2026-08-16-xbackend-benchmark-contract`

**Run locally:**
```bash
git diff --check
uv run python -m pytest tests/test_wolfgang_kernel_performance_campaign_plan.py -q
uv run python scripts/validate.py
```

**Promotion rule:** If the contract test passes and review confirms the ranking is evidence-grounded, this wave is complete.

### Wave 1B: CPU small-shape dispatch split prototype

**Objective:** Correct the known small full-grouping mis-selection without weakening pairwise commutation wins.

**Likely files:**
- Modify: `src/cpu_backend.cpp`
- Modify: `src/grouping.cpp`
- Modify: `benchmarks/bench_cpu_dispatch.py`
- Modify: `benchmarks/bench_cpu_thresholds.py`
- Modify: `tests/` CPU dispatch coverage files if needed

**Branch:** `exp/2026-08-16-cpu-small-shape-dispatch`

**Run locally:**
```bash
uv run python benchmarks/bench_cpu_dispatch.py --repeat 5 --warmup 1 --json
uv run python benchmarks/bench_cpu_thresholds.py --repeat 5 --warmup 1 --json
uv run python scripts/validate.py
```

**Success criterion:** small full-grouping improves or selector cleanly splits, while pairwise commutation remains within the regression budget.

### Wave 1C: Shared accelerator workspace/reuse contract prestaging

**Objective:** Define and test the ownership/lifetime boundary for reusable scratch and output buffers without changing production kernels yet.

**Likely files:**
- Modify: `docs/architecture/backend_neutral_accelerators.md`
- Modify: `docs/architecture/cuda_backend.md`
- Modify: `docs/architecture/rocm_backend.md`
- Modify: `docs/architecture/apple_accelerator.md`
- Modify: `src/cuda/workspace.cuh`
- Modify: `src/hip/workspace_hip.hip.hpp`
- Modify: `src/metal/workspace_metal.hpp`
- Modify: `src/detail/accelerator_host_helpers.hpp`
- Modify: benchmark schema tests only if the contract becomes user-visible

**Branch:** `exp/2026-08-16-xbackend-workspace-contract`

**Run locally:**
```bash
git diff --check
uv run python -m pytest tests -q -k "backend_neutral or rocm or apple or validate"
uv run python scripts/validate.py
```

**Success criterion:** contract text and tests land without enabling new public behavior or breaking target-specific accelerator policy.

### Wave 1D: Metal commutation reuse prestage on the local Apple host

**Objective:** Prototype the cheapest Apple-local retained win: reusing device outputs and avoiding fresh allocation in repeated commutation consumers.

**Likely files:**
- Modify: `src/metal/device_commutation_matrix_metal.mm`
- Modify: `src/metal/workspace_metal.mm`
- Modify: `benchmarks/bench_metal_kernels.py`
- Modify: `docs/architecture/apple_accelerator.md`
- Modify: Apple Metal benchmark tests as needed

**Branch:** `exp/2026-08-16-metal-commutation-reuse`

**Run locally:**
```bash
env FASTPAULI_VALIDATE_METAL=1 uv run python benchmarks/bench_metal_kernels.py --profile smoke --repeat 3 --json
env FASTPAULI_VALIDATE_METAL=1 uv run python scripts/validate.py
```

**Success criterion:** repeated retained-output cases improve without broadening public semantics or worsening transfer-inclusive small rows.

## Second wave: only after first-wave evidence is green

### Wave 2A: CUDA retained compact-consumer workspace reuse

**Objective:** Spend H100 time only after the reuse contract and local tests are stable.

**Likely files:**
- Modify: `src/cuda/workspace.cu`
- Modify: `src/cuda/device_commutation_matrix.cu`
- Modify: `src/cuda/commutation_cuda.cu`
- Modify: `benchmarks/bench_cuda_scaling.py`
- Modify: `benchmarks/bench_cuda_kernels.py`

**Branch:** `exp/2026-08-16-cuda-retained-reuse`

**Run on H100 after prestaging:**
```bash
env FASTPAULI_VALIDATE_CUDA=1 uv run python benchmarks/bench_cuda_scaling.py --profile default --repeat 5 --warmup 1 --json
env FASTPAULI_VALIDATE_CUDA=1 uv run python scripts/validate.py
```

**Why second wave:** strong upside, but not locally falsifiable on this host.

### Wave 2B: HIP retained compact-consumer workspace reuse

**Objective:** Use MI300X only for a retained-operation bottleneck that survives Wave 1C.

**Likely files:**
- Modify: `src/hip/workspace_hip.hip.cpp`
- Modify: `src/hip/device_commutation_matrix.hip.cpp`
- Modify: `src/hip/commutation_hip.hip.cpp`
- Modify: `benchmarks/bench_rocm_kernels.py`

**Branch:** `exp/2026-08-16-hip-retained-reuse`

**Run on MI300X after notify-before-capacity:**
```bash
env FASTPAULI_VALIDATE_HIP=1 uv run python benchmarks/bench_rocm_kernels.py --profile campaign7-release-smoke --repeat 5 --warmup 1 --json
env FASTPAULI_VALIDATE_HIP=1 uv run python scripts/validate.py
```

### Wave 2C: CUDA + HIP expectation residency follow-up

**Objective:** Revisit expectation only after boundary reuse work is in place.

**Likely files:**
- Modify: `src/cuda/expectation_cuda.cu`
- Modify: `src/hip/expectation_hip.hip.cpp`
- Modify: `include/wolfgang/device_pauli_sum.hpp`
- Modify: `bindings/python/pauli_sum_py.cpp`
- Modify: `benchmarks/bench_expectation.py`

**Branch:** `exp/2026-08-16-xbackend-expectation-residency`

**Run on affected GPU host only after contract acceptance.**

### Wave 2D: CUDA + HIP multiword simplify follow-up

**Objective:** Improve the >2-word simplify structure only if reuse work does not already move the user-visible bottleneck enough.

**Likely files:**
- Modify: `src/cuda/simplify_cuda.cu`
- Modify: `src/hip/simplify_hip.hip.cpp`
- Modify: `benchmarks/bench_cuda_kernels.py`
- Modify: `benchmarks/bench_rocm_kernels.py`

**Branch:** `exp/2026-08-16-xbackend-multiword-simplify`

### Wave 2E: Metal lower-pass simplify probe

**Objective:** Keep this explicitly experimental until a lower-pass design exists.

**Likely files:**
- Modify: `src/metal/simplify_metal.mm`
- Modify: `src/metal/kernels/simplify.metal`
- Modify: `benchmarks/bench_metal_kernels.py`

**Branch:** `exp/2026-08-16-metal-lower-pass-simplify`

## Explicit deprioritizations

Do not start these unless new checked evidence overturns the current reports:

```text
public streams or graph replay on CUDA or HIP
full CSR scatter tuning for retained consumers
mixed CUDA+HIP runtime packaging work
Apple Metal public simplify promotion from the current Campaign 8 candidate
generic instruction-level CUDA or HIP kernel tweaks without a retained-operation bottleneck
same-host MI300X repetition without a new profiler artifact
```

## Rollback plan

Every experiment branch must be disposable.

```text
1. Start from main at or rebased onto the current accepted baseline.
2. Keep one coherent hypothesis per `exp/<date>-<slug>` branch.
3. If the screen loses or violates regression budget, delete the branch.
4. Do not merge a branch on a best-single-run win.
5. If a shared contract change destabilizes a backend, revert the contract branch and reopen with a narrower backend-specific slice.
6. Preserve checked benchmark JSON, commands, and summary notes even for rejected branches.
```

## Stop conditions

Stop the campaign when any one of these becomes true:

```text
three consecutive first-wave branches fail to produce a promotable signal
rank-1 and rank-2 reuse work fail to beat the baseline mean by >=10% on their intended workloads
the only surviving wins come from regressions on small workloads that need selector splits the team does not want to own
remaining opportunities require new public lifetime semantics that have not been accepted in docs/architecture or docs/architecture/api_stability.md
available hardware time is exhausted before any retained-operation bottleneck survives local prestaging
```

## Implementation handoff summary

The first implementation branch should not touch production CUDA or HIP kernels. It should land this planning contract, route it through source-of-truth docs, and add the plan test. After that, execute Wave 1B through Wave 1D in rank order, promoting only when repeated evidence beats the immutable baseline.

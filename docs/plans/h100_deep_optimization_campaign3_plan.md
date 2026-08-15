# H100 Deep Optimization Campaign 3 Plan

Status: completed on 2026-04-28. The checked-in report is
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign3_2026-04-28.md`;
raw JSON, summary JSON, and metadata are under
`docs/benchmarks/data/cuda_deep_optimization_h100_campaign3_2026-04-28/`;
generated SVGs are under `docs/benchmarks/plots/`.

> For agentic workers: execute this plan task-by-task on a short-lived
> `codex/` branch. Use `superpowers:subagent-driven-development` or
> `superpowers:executing-plans` when carrying out the implementation campaign.
> H100 profiling, benchmarking, sanitizer, and performance experiments must run
> on the H100 host. Local Apple Silicon may be used only for repository editing,
> documentation checks, and non-performance validation.

## Goal

Convert the remaining headroom from
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign2_2026-04-28.md`
into a third H100 optimization campaign focused on allocation, temporary
storage, output materialization, and reduction topology.

The campaign succeeds only if it either retains production changes with
same-boundary speedups and fresh correctness evidence, or proves with profiler
and benchmark data why each explored path is exhausted or deferred.

## Source Inputs

Read these before implementation begins:

```text
README.md
docs/roadmap.md
docs/plans/cuda_deep_optimization_plan.md
docs/plans/h100_deep_optimization_campaign2_plan.md
docs/benchmarks/reports/cuda_deep_optimization_h100_campaign2_2026-04-28.md
docs/architecture/cuda_backend.md
docs/architecture/api_stability.md
docs/architecture/hardware_targets_and_testing.md
docs/benchmarks/protocol.md
docs/quality/code_review.md
docs/quality/code_standards.md
docs/quality/documentation_standards.md
```

## Scope

Campaign 3 starts from the campaign-2 conclusion that the next CUDA performance
frontier is allocation and materialization rather than instruction-level PTX
work.

In scope:

```text
allocation and temporary-storage instrumentation for CUDA hot paths
private reusable workspace prototypes for simplify and matmul+simplify
CUB-backed or lower-allocation duplicate-reduction experiments
same-boundary A/B benchmarks for Thrust/CCCL, CUB, and custom materialization choices
commutation output materialization alternatives behind private or benchmark-only surfaces
statevector expectation reduction-topology experiments when repeatability is measured
stream-aware private helper experiments only when lifetime and synchronization stay explicit
competitive baseline refreshes for semantically comparable GPU and CPU paths
publication-quality campaign report, plots, raw data, and decision log
```

Out of scope unless profiler evidence changes the priority:

```text
public CUDA workspace API without an API review and user-facing docs
public stream or async APIs without documented lifetime and synchronization semantics
public device-output commutation API without ownership, dtype, shape, and error tests
raw PTX rewrites without a specific SASS/code-generation defect
custom GPU sorting before CUB/CCCL temp-storage options are measured
portable CUDA wheel, A100, RTX, AMD GPU, Apple GPU, HIP, or Metal performance claims
unsupported comparisons that treat framework-level baselines as sparse-Pauli primitive baselines
```

## Hypotheses

The campaign tests these hypotheses in order:

1. Host-side and device temporary allocation costs still hide useful H100 kernel
   throughput for simplify and matmul+simplify.
2. A private reusable CUDA workspace can remove repeated allocation overhead
   without changing canonical ordering, tolerance semantics, device ownership,
   or public synchronization behavior.
3. CUB temporary-storage APIs or a more direct duplicate-reduction pipeline can
   reduce simplify and matmul+simplify overhead compared with the current
   Thrust-heavy implementation.
4. Dense commutation remains bounded by public host-output materialization; a
   device-byte or bit-packed prototype can quantify the gap without changing
   public API.
5. Statevector expectation may benefit from an alternate reduction topology
   only if accuracy, tolerance, and repeatability evidence justify the change.

## H100 Execution Contract

All performance evidence for this campaign must be produced on the H100 host.
Use an environment variable for the current session target rather than hard
coding an ephemeral host in committed docs:

```bash
export FASTPAULI_H100_SSH_TARGET="${FASTPAULI_H100_SSH_TARGET:?set to the current H100 SSH target}"
export FASTPAULI_H100_BASELINE_DIR=<private-path>
export FASTPAULI_H100_EXPERIMENT_DIR=<private-path>
export FASTPAULI_H100_ARTIFACT_ROOT=<private-path>
export FASTPAULI_BRANCH=codex/h100-deep-optimization-campaign3
```

Push the campaign branch before running remote experiments:

```bash
git push -u origin "$FASTPAULI_BRANCH"
```

The H100 host must use two independent checkouts:

```text
baseline: origin/main reproduction only, safe to reset
experiment: FASTPAULI_BRANCH or an exact campaign commit, never reset over dirty work
```

The first H100 command must prepare the checkouts and record hardware/software
identity into the artifact root:

```bash
ssh "$FASTPAULI_H100_SSH_TARGET" \
  'if [ ! -d <private-path> ]; then \
     git clone https://github.com/sghowell/FastPauli.git <private-path> \
   fi && \
   if [ ! -d <private-path> ]; then \
     git clone https://github.com/sghowell/FastPauli.git <private-path> \
   fi && \
   cd <private-path> && \
   git fetch origin && \
   git checkout main && \
   git reset --hard origin/main && \
   cd <private-path> && \
   git fetch origin && \
   test -z "$(git status --porcelain)" && \
   git checkout codex/h100-deep-optimization-campaign3 && \
   git pull --ff-only origin codex/h100-deep-optimization-campaign3 && \
   mkdir -p <private-path> && \
   git rev-parse HEAD > <private-path> && \
   hostname > <private-path> && \
   nvidia-smi -q > <private-path> && \
   nvidia-smi --query-gpu=name,uuid,driver_version,cuda_version,compute_cap --format=csv \
     > <private-path> && \
   lscpu > <private-path> && \
   /usr/local/cuda/bin/nvcc --version > <private-path>'
```

## Required Artifacts

Expected checked-in outputs:

```text
docs/benchmarks/reports/cuda_deep_optimization_h100_campaign3_YYYY-MM-DD.md
docs/benchmarks/data/cuda_deep_optimization_h100_campaign3_YYYY-MM-DD/summary.json
docs/benchmarks/data/cuda_deep_optimization_h100_campaign3_YYYY-MM-DD/raw/*.json
docs/benchmarks/plots/cuda_h100_campaign3_*.svg
docs/benchmarks/plots/cuda_h100_campaign3_readme_cross_comparison.svg
```

Expected remote artifact root:

```text
<private-path>
```

The final checked-in report must include:

```text
exact git revisions for baseline and experiment
exact H100 hardware, driver, CUDA toolkit, compiled architectures, and compiler metadata
exact commands for validation, profiling, benchmarking, and competitor baselines
raw JSON paths and plot-rendering commands
availability, version, semantic mapping, correctness, and limitation status for every competitor baseline
README plot selection rationale, including which CPU, CUDA, and external baseline paths appear in the README-facing chart
profiler availability and permission status
retained, rejected, and deferred experiment decisions
residual headroom and next-platform recommendations
```

## Decision Gates

Record these decisions in the branch diff before retaining production changes:

1. Workspace ownership: keep workspace internal/benchmark-only, expose an
   experimental public API, or reject the workspace path.
2. Workspace lifetime: define device ordinal, stream compatibility, capacity
   growth, reset/release, and moved-from behavior.
3. Temporary storage: choose Thrust/CCCL default allocation, explicit CUB
   scratch buffers, custom scratch layout, or no change.
4. Commutation materialization: keep public host vector return, retain
   caller-owned host fill only, add a private device-byte prototype, add a
   private bit-packed prototype, or reject output alternatives.
5. Statevector reduction: keep current atomic/fused accumulation, switch to a
   CUB-backed reduction, add an optional deterministic benchmark mode, or reject
   alternate reduction topology.
6. Stream semantics: keep public default-stream synchronize-before-return,
   retain private stream helpers, or start a separate public async API plan.
7. Timing boundary: label every retained or reported result as allocation
   inclusive, preallocated, device-resident, transfer-inclusive, host-output,
   caller-owned output, or private prototype.

Update `docs/architecture/cuda_backend.md`, `docs/architecture/api_stability.md`,
and `docs/benchmarks/protocol.md` in the same branch when any decision changes
public behavior, timing boundaries, report schemas, or API policy.

## Task 0: Baseline Reproduction And Profiler Preflight

- [ ] Prepare clean baseline and experiment checkouts on the H100 host.
- [ ] Install an editable CUDA source build in both checkouts with
      `FASTPAULI_CUDA_ARCHITECTURES=90`.
- [ ] Run `FASTPAULI_VALIDATE_CUDA=1 python scripts/validate.py` in the
      baseline checkout.
- [ ] Reproduce campaign-2 stress and extreme benchmark profiles from
      `origin/main` and save raw JSON under the campaign-3 artifact root.
- [ ] Record `nsys`, `ncu`, `compute-sanitizer`, `cuobjdump`, and `nvdisasm`
      availability.
- [ ] Run an Nsight Compute permission probe and record whether privileged
      profiler collection is required.
- [ ] Stop and report a blocker if H100 profiler evidence cannot be collected
      and the missing tool prevents a decision gate from being evaluated.

## Task 1: Allocation And Materialization Instrumentation

- [ ] Extend CUDA benchmark output to record temporary-storage bytes,
      workspace mode, allocation count or unavailable reason, result
      materialization target, and duplicate survivor counts for every affected
      operation.
- [ ] Add an allocation/materialization stress profile that isolates:
      simplify duplicate pressure, matmul+simplify duplicate pressure,
      pairwise commutation host-output materialization, and statevector
      expectation resident execution.
- [ ] Ensure smoke/default/stress/extreme profiles remain deterministic and
      correctness-checked.
- [ ] Capture Nsight Systems timelines for current simplify, matmul+simplify,
      pairwise commutation, and statevector expectation before code changes.
- [ ] Capture Nsight Compute evidence for representative kernels and
      CCCL/Thrust-heavy calls.

Acceptance:

```text
benchmark JSON and reports expose allocation/materialization boundaries
existing benchmark consumers remain compatible or are updated in the same slice
no speedup claim is made without same-boundary labels
```

## Task 2: Private CUDA Workspace Prototype

- [ ] Design a private C++ CUDA workspace type that is not installed and not
      exposed to Python.
- [ ] Tie workspace storage to one CUDA device ordinal and reject cross-device
      reuse.
- [ ] Use monotonic capacity growth within a run and explicit reset/release
      paths before shrinking.
- [ ] Add benchmark-only paths that can run with workspace absent,
      growth-inside-timing, and pre-reserved-outside-timing modes.
- [ ] Keep public methods default-stream and synchronize-before-return.
- [ ] Add CUDA tests for wrong device, capacity growth, reset/release, and
      unchanged results when the workspace path is used internally.

Acceptance:

```text
workspace mode changes timing boundaries only when labels say so
canonical simplify order and tolerance behavior are unchanged
workspace failures translate to existing CUDA exception policy
```

## Task 3: Simplify And Matmul Duplicate-Reduction Experiments

- [ ] Compare current Thrust/CCCL simplify against an explicit CUB
      temporary-storage implementation where CUB APIs fit the key/value layout.
- [ ] Evaluate whether a two-pass temp-size query and reusable scratch buffer
      removes allocation overhead without increasing kernel time.
- [ ] Profile sort, run-length or reduce-by-key, compaction, coefficient
      accumulation, and output materialization separately.
- [ ] Test low-, medium-, high-, and pathological-duplicate regimes.
- [ ] Retain a production path only if it is faster or materially more stable
      on same-boundary H100 measurements and does not regress CPU-only builds.
- [ ] Reject or defer custom duplicate-reduction kernels unless CUB/CCCL
      evidence proves the library path is the limiting factor.

Acceptance:

```text
same-boundary stress and extreme measurements cover simplify and matmul+simplify
CPU/GPU equivalence tests pass for retained changes
Nsight evidence explains retained, rejected, or deferred duplicate-reduction choices
```

## Task 4: Commutation Output Materialization Frontier

- [ ] Keep the public vector-return and caller-owned host-fill APIs as the
      supported behavior.
- [ ] Add private benchmark-only variants for caller-owned device bytes and
      bit-packed output only if the benchmark surface can prevent accidental
      public exposure.
- [ ] Measure dense host vector return, caller-owned host fill, device-byte
      output, bit-packed output, and transfer-inclusive copies with explicit
      labels.
- [ ] Preserve dense-output guardrails and row-major result semantics.
- [ ] Do not retain a public device-output or bit-packed API without a separate
      API review, user docs, and tests.

Acceptance:

```text
public commutation behavior remains unchanged
benchmark report quantifies host materialization overhead versus private prototypes
prototype-only results are not advertised as public speedups
```

## Task 5: Statevector Expectation Reduction Topology

- [ ] Compare the current fused accumulation against a CUB-backed or staged
      reduction when profiler evidence shows atomic or reduction pressure.
- [ ] Measure complex64 and complex128 separately.
- [ ] Record numerical error versus CPU and current CUDA results.
- [ ] Run repeatability checks across repeated runs with fixed seeds.
- [ ] Retain a new path only if it improves same-boundary performance and stays
      inside documented dtype tolerances.
- [ ] Document deterministic-mode tradeoffs if a deterministic prototype is
      useful but slower.

Acceptance:

```text
accuracy and repeatability are reported for every reduction variant
statevector CUDA-array-interface behavior remains public and unchanged
```

## Task 6: Competitor And CPU Baseline Refresh

- [ ] Refresh all FastPauli CPU selector timings available on the H100 host:
      scalar, oneTBB, SIMD, native source builds, and auto.
- [ ] Refresh Qiskit, OpenFermion, CuPy, cuQuantum/cuStateVec, cuQuantum
      cuPauliProp when semantically mapped, CUDA-Q, and Qiskit Aer GPU
      availability.
- [ ] Build a same-dataset comparison table for each README-eligible operation
      that includes FastPauli scalar CPU, auto CPU dispatch, each available
      optimized CPU selector, CUDA transfer-inclusive, CUDA device-resident,
      CUDA preallocated or reused-output paths when they are public or clearly
      labeled prototypes, and external packages with comparable semantics.
- [ ] Run only semantically comparable competitor timings; record unavailable
      or framework-level-only status instead of forcing invalid comparisons.
- [ ] Check competitor correctness wherever an output can be mapped to a
      FastPauli oracle.

Acceptance:

```text
plots include all captured FastPauli CPU and CUDA variants
README-eligible plots include CPU scalar, every captured optimized CPU selector, CUDA transfer-inclusive, CUDA device-resident, and semantically comparable external package baselines where available
CUDA-only before/after speedup plots are allowed in the report but cannot be the only README benchmark visual
competitor results include version, availability, semantic mapping, timing boundary, and correctness status
unsupported GPU-library comparisons are clearly labeled or omitted with reasons
```

## Task 7: Production Retention, Validation, And Review

- [ ] Keep production changes only when they pass the relevant decision gates.
- [ ] Run local Apple Silicon validation for docs and CPU-only checks.
- [ ] Run H100 CUDA validation with `FASTPAULI_VALIDATE_CUDA=1`.
- [ ] Run Compute Sanitizer for retained CUDA changes.
- [ ] Run benchmark smoke/default/stress profiles after retained changes.
- [ ] Request independent agent review before merge, with CUDA, benchmark, and
      API-policy scope called out.
- [ ] Resolve P0/P1 findings and rerun validation after fixes.

Acceptance:

```text
python scripts/validate.py passes locally
FASTPAULI_VALIDATE_CUDA=1 python scripts/validate.py passes on H100
Compute Sanitizer is clean for retained CUDA paths
review evidence records reviewer, scope, findings, resolutions, validation, and residual risk
```

## Task 8: Campaign Report And Closeout

- [ ] Check in the campaign-3 raw JSON, summary JSON, plots, and final report.
- [ ] Replace or augment the README benchmark snapshot with a broad
      cross-comparison plot generated from checked-in campaign-3 evidence. The
      README-facing plot must compare FastPauli CPU scalar, captured optimized
      CPU selectors, CUDA transfer-inclusive, CUDA device-resident, and
      semantically comparable external package baselines where those data exist.
- [ ] Keep CUDA-only before/after improvement plots in the report. Do not use
      them in README unless they are paired with the broad cross-comparison
      plot.
- [ ] Include diagrams or visuals only when they explain architecture,
      allocation/materialization boundaries, kernel flow, reduction topology, or
      hardware behavior better than text alone.
- [ ] Keep visuals clean, legible, publication quality, and generated from
      checked-in evidence whenever they represent benchmark data.
- [ ] Document rejected, retained, and deferred experiments in the report.
- [ ] Update roadmap and any affected architecture docs with the next
      remaining-headroom decisions.
- [ ] Merge locally to `main`, validate the merged result, push `main`, confirm
      CI is green, and delete the merged feature branch.

Acceptance:

```text
checked-in report and plots are reproducible from checked-in raw data
README benchmark snapshot includes a broad CPU/CUDA/external cross-comparison, not only CUDA before/after speedups
README, roadmap, architecture docs, and benchmark protocol agree
CI is green on pushed main
no untracked benchmark artifacts, profiler dumps, or build products remain in the worktree
```

## Exhaustion Criteria

Campaign 3 is complete when all of these are true:

```text
allocation and materialization boundaries have been measured for every CUDA hot path in scope
workspace, CUB/CCCL temp storage, commutation materialization, and reduction-topology hypotheses have retained/rejected/deferred decisions
retained production optimizations have same-boundary speedups or stability improvements with correctness evidence
rejected paths have profiler or benchmark evidence explaining why they are not worth retaining
competitor and CPU baselines have been refreshed or recorded as unavailable with reasons
README-ready cross-comparison plots have been generated from checked-in evidence and published in README with unavailable external baselines labeled when necessary
the final report identifies remaining headroom with concrete next actions or states that current H100-side work is exhausted
```

# H100 Deep Optimization Campaign 2 Plan

> For agentic workers: implement this plan task-by-task on a short-lived
> `codex/` branch. H100 profiling, benchmarking, sanitizer, and performance
> experiments must run on the H100 host. Local Apple Silicon may be used only
> for repository editing, documentation checks, and non-performance validation.

## Goal

Convert the remaining headroom from
`docs/benchmarks/reports/cuda_deep_optimization_h100_2026-04-28.md` into a
second H100 optimization campaign that can either retain new production changes
or prove, with profiling evidence, why a path is exhausted.

## Scope

This campaign starts from the 2026-04-28 H100 report and focuses on work that
changes materialization, allocation, or lifetime boundaries. It does not reopen
small instruction-level commutation edits unless new H100 profiler evidence
shows that the materialization boundary is no longer dominant.

In scope:

```text
reusable CUDA/CCCL/CUB temporary storage for simplify and matmul+simplify
device-resident statevector expectation throughput and reduction decomposition
caller-owned workspace API design and implementation if evidence supports it
stream-aware or async API design only when lifetime and error semantics are clear
dense commutation output materialization alternatives after current bottlenecks are separated
workload-specific external GPU baselines when semantics are comparable
publication-quality H100 campaign report, plots, raw data, and diagrams
```

Out of scope unless profiling evidence changes the priority:

```text
raw PTX rewrites without a specific SASS/code-generation defect
custom GPU sorting or duplicate reduction before CCCL/CUB options are measured
portable CUDA wheel claims
Apple Silicon GPU, MPS, HIP, ROCm, AMD GPU, A100, or RTX Pro 6000 claims
CUDA-Q or Qiskit Aer comparisons presented as primitive-equivalent sparse-Pauli baselines
```

## Source Inputs

Read these before running the campaign:

```text
docs/plans/cuda_deep_optimization_plan.md
docs/benchmarks/reports/cuda_deep_optimization_h100_2026-04-28.md
docs/architecture/cuda_backend.md
docs/architecture/hardware_targets_and_testing.md
docs/architecture/api_stability.md
docs/benchmarks/protocol.md
docs/quality/code_review.md
docs/quality/code_standards.md
docs/roadmap.md
```

## H100 Execution Contract

All performance evidence for this campaign must be produced on the H100 host.
Use an environment variable for the current session target rather than hard
coding an ephemeral host in committed docs:

```bash
export FASTPAULI_H100_SSH_TARGET="${FASTPAULI_H100_SSH_TARGET:?set to the current H100 SSH target}"
export FASTPAULI_H100_BASELINE_DIR=<private-path>
export FASTPAULI_H100_EXPERIMENT_DIR=<private-path>
export FASTPAULI_H100_ARTIFACT_ROOT=<private-path>
export FASTPAULI_BRANCH=codex/h100-deep-optimization-campaign2
```

Push the campaign branch before running remote experiments so the H100 host can
checkout the exact revision under test:

```bash
git push -u origin "$FASTPAULI_BRANCH"
```

The H100 campaign uses two independent checkouts:

```text
baseline: origin/main reproduction only, safe to reset
experiment: FASTPAULI_BRANCH or an exact campaign commit, never reset over dirty work
```

The first command on the H100 host must prepare those checkouts and record the
hardware and software identity into the campaign artifact root:

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
   git checkout codex/h100-deep-optimization-campaign2 && \
   git pull --ff-only origin codex/h100-deep-optimization-campaign2 && \
   mkdir -p <private-path> && \
   git rev-parse HEAD > <private-path> && \
   hostname > <private-path> && \
   nvidia-smi -q > <private-path> && \
   nvidia-smi --query-gpu=name,uuid,driver_version,cuda_version,compute_cap --format=csv \
     > <private-path> && \
   lscpu > <private-path> && \
   /usr/local/cuda/bin/nvcc --version > <private-path>'
```

The final checked-in report must include the H100 artifact root, exact git
revision, exact commands, and any unavailable tool or competitor reasons.

## Campaign Artifacts

Expected checked-in outputs:

```text
docs/benchmarks/reports/cuda_deep_optimization_h100_campaign2_2026-04-28.md
docs/benchmarks/data/cuda_deep_optimization_h100_campaign2_2026-04-28/summary.json
docs/benchmarks/data/cuda_deep_optimization_h100_campaign2_2026-04-28/raw/*.json
docs/benchmarks/plots/cuda_h100_campaign2_*.svg
```

Expected remote artifact root:

```text
<private-path>
```

## Decision Gates

Do not start production code changes until these decisions are written into the
branch diff:

1. Workspace ownership: whether `CudaWorkspace` is public API, experimental
   API, or an internal C++/benchmark-only object.
2. Workspace lifetime: which operations may reuse temporary storage, how
   device ordinal and CUDA stream compatibility are checked, and how capacity
   grows.
3. Statevector residency: whether high-throughput expectation uses the current
   CUDA-array-interface path, a new explicit device-statevector wrapper, or an
   experimental API.
4. Stream semantics: whether stream support remains out of scope, benchmark
   internal only, or public; if public, document synchronization and error
   reporting before implementation.
5. Result materialization: whether commutation experiments return host bytes,
   fill caller host output, fill caller device output, return bit-packed output,
   or only measure private prototypes.
6. Determinism: which reductions may be non-deterministic within documented
   floating-point tolerances and which modes must preserve deterministic order.

These decisions should update `docs/architecture/cuda_backend.md`,
`docs/architecture/api_stability.md`, and `docs/benchmarks/protocol.md` when
they affect public behavior, timing boundaries, or benchmark interpretation.

## Task 0: H100 Baseline Reproduction

Purpose: prove the campaign starts from a clean H100 source build and reproduces
the previous report's timing boundaries.

Commands:

```bash
ssh "$FASTPAULI_H100_SSH_TARGET" \
  'cd <private-path> && \
   git fetch origin && \
   git checkout main && \
   git reset --hard origin/main && \
   python3 -m venv .venv && \
   .venv/bin/python -m pip install --upgrade pip && \
   PATH=/usr/local/cuda/bin:$PATH \
   CUDACXX=/usr/local/cuda/bin/nvcc \
   CUDAHOSTCXX=/usr/bin/g++ \
   FASTPAULI_CUDA_ARCHITECTURES=90 \
   .venv/bin/python -m pip install -e ".[test]" \
     --config-settings=cmake.define.FASTPAULI_ENABLE_CUDA=ON \
     --config-settings=cmake.define.FASTPAULI_CUDA_ARCHITECTURES=90 \
     --config-settings=cmake.define.CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++'
```

```bash
ssh "$FASTPAULI_H100_SSH_TARGET" \
  'cd <private-path> && \
   PATH=/usr/local/cuda/bin:$PATH \
   CUDACXX=/usr/local/cuda/bin/nvcc \
   CUDAHOSTCXX=/usr/bin/g++ \
   FASTPAULI_VALIDATE_CUDA=1 \
   FASTPAULI_CUDA_ARCHITECTURES=90 \
   .venv/bin/python scripts/validate.py'
```

Before running required profiling, record profiler availability and test Nsight
Compute counter permissions with a small CUDA benchmark:

```bash
ssh "$FASTPAULI_H100_SSH_TARGET" \
  'cd <private-path> && \
   python3 -m venv .venv && \
   .venv/bin/python -m pip install --upgrade pip && \
   PATH=/usr/local/cuda/bin:$PATH \
   CUDACXX=/usr/local/cuda/bin/nvcc \
   CUDAHOSTCXX=/usr/bin/g++ \
   FASTPAULI_CUDA_ARCHITECTURES=90 \
   .venv/bin/python -m pip install -e ".[test]" \
     --config-settings=cmake.define.FASTPAULI_ENABLE_CUDA=ON \
     --config-settings=cmake.define.FASTPAULI_CUDA_ARCHITECTURES=90 \
     --config-settings=cmake.define.CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ && \
   mkdir -p <private-path> && \
   command -v nsys > <private-path> || echo missing > <private-path> && \
   command -v ncu > <private-path> || echo missing > <private-path> && \
   command -v compute-sanitizer > <private-path> || echo missing > <private-path> && \
   if grep -q missing <private-path> <private-path> <private-path> then \
     echo profiler_tool_missing > <private-path> \
     exit 2; \
   fi && \
   PATH=/usr/local/cuda/bin:$PATH \
   FASTPAULI_VALIDATE_CUDA=1 \
   ncu --target-processes all --set default --launch-count 1 --force-overwrite \
     --export <private-path> \
     .venv/bin/python benchmarks/bench_cuda_scaling.py \
       --profile smoke --operation statevector_expectation --repeat 1 --warmup 0 --json \
     > <private-path> 2>&1 || true && \
   if grep -q ERR_NVGPUCTRPERM <private-path> then \
     echo sudo_required > <private-path> \
   else \
     echo user_access_ok > <private-path> \
   fi'
```

If `ncu-permission-status.txt` contains `sudo_required`, collect required NCU
evidence through a privileged replacement pass or stop and report the missing
permission as a blocker if sudo is unavailable:

```bash
ssh "$FASTPAULI_H100_SSH_TARGET" \
  'cd <private-path> && \
   if grep -q sudo_required <private-path> then \
     sudo env PATH=/usr/local/cuda/bin:$PATH \
       FASTPAULI_VALIDATE_CUDA=1 \
       FASTPAULI_CUDA_ARCHITECTURES=90 \
       CUDACXX=/usr/local/cuda/bin/nvcc \
       CUDAHOSTCXX=/usr/bin/g++ \
       .venv/bin/python scripts/cuda_deep_profile.py \
         --execute --json --profile stress --competitor-set none \
         --require-profiler-artifacts \
         --output-root <private-path> && \
     sudo chown -R ubuntu:ubuntu <private-path> \
   fi'
```

```bash
ssh "$FASTPAULI_H100_SSH_TARGET" \
  'cd <private-path> && \
   if grep -q sudo_required <private-path> then \
     PATH=/usr/local/cuda/bin:$PATH \
       FASTPAULI_VALIDATE_CUDA=1 \
       .venv/bin/python scripts/cuda_deep_profile.py \
         --execute --json --profile stress --competitor-set all \
         --continue-on-error \
         --output-root <private-path> && \
     sudo env PATH=/usr/local/cuda/bin:$PATH \
       FASTPAULI_VALIDATE_CUDA=1 \
       FASTPAULI_CUDA_ARCHITECTURES=90 \
       CUDACXX=/usr/local/cuda/bin/nvcc \
       CUDAHOSTCXX=/usr/bin/g++ \
       .venv/bin/python scripts/cuda_deep_profile.py \
         --execute --json --profile stress --competitor-set none \
         --require-profiler-artifacts \
         --output-root <private-path> && \
     sudo chown -R ubuntu:ubuntu <private-path> \
   else \
     PATH=/usr/local/cuda/bin:$PATH \
       FASTPAULI_VALIDATE_CUDA=1 \
       .venv/bin/python scripts/cuda_deep_profile.py \
         --execute --json --profile stress --competitor-set all \
         --require-profiler-artifacts \
         --output-root <private-path> \
   fi'
```

Acceptance:

```text
validation passes on the H100 source build
profiler preflight records nsys, ncu, compute-sanitizer availability, and NCU permission status
baseline profile records Nsight Systems, Nsight Compute, sanitizer, and binary-inspection artifacts
when NCU requires sudo, nonprivileged baseline benchmark/competitor evidence and privileged profiler evidence are both recorded
baseline benchmark JSON records transfer-inclusive, device-resident, and preallocated-output boundaries where available
summary states whether baseline timings are within expected variance of the 2026-04-28 report
experiment profile commands run from the campaign branch or exact commit, not from the reset baseline checkout
```

## Task 1: Workspace And API Design Slice

Purpose: decide the minimum API surface needed to measure and retain reusable
temporary storage without compromising CPU-only builds or stable semantics.

Candidate Python shape:

```python
workspace = fastpauli.cuda.Workspace(device=0)
dh_simplified = dh.simplify(workspace=workspace)
dh_product = dh.matmul(other, simplify=True, workspace=workspace)
workspace.reserve_sort_pairs(num_items=dh.num_terms)
workspace.reset()
```

Candidate C++ shape:

```cpp
fastpauli::cuda::Workspace workspace(device_ordinal);
workspace.reserve_sort_pairs(num_items);
DevicePauliSum simplified = simplify(device_pauli_sum, workspace);
```

Required design checks:

```text
CPU-only import does not import CUDA libraries
workspace construction fails clearly when CUDA support is absent
workspace device ordinal must match every DevicePauliSum operand
workspace capacity growth is monotonic unless reset or release is called
workspace use does not change canonical ordering or coefficient tolerance semantics
workspace methods do not hide host-device transfer costs in benchmark labels
```

Acceptance:

```text
public, experimental, or internal-only status is explicitly chosen
affected architecture/API docs are updated before code lands
unit tests or planned tests are named for absent-CUDA, wrong-device, moved-from, and reuse cases
```

## Task 2: Benchmark And Profiler Instrumentation

Purpose: make the H100 campaign able to attribute time to allocation,
temporary-storage query, sort/reduce, final compaction, host synchronization,
and result materialization.

Add or extend benchmark outputs so each experiment records:

```text
operation name
timing boundary
num_qubits
num_terms
words
duplicate_rate
survivor_count
workspace enabled or disabled
temporary storage bytes requested
allocation count when available
CUDA stream mode
median, p10, p90, min, max, repeat count, warmup count
correctness oracle
```

H100 commands:

```bash
ssh "$FASTPAULI_H100_SSH_TARGET" \
  'cd <private-path> && \
   FASTPAULI_VALIDATE_CUDA=1 \
   .venv/bin/python benchmarks/bench_cuda_kernels.py \
     --json --repeat 7 --warmup 3 \
     --output <private-path>'
```

```bash
ssh "$FASTPAULI_H100_SSH_TARGET" \
  'cd <private-path> && \
   FASTPAULI_VALIDATE_CUDA=1 \
   .venv/bin/python benchmarks/bench_cuda_scaling.py \
     --profile extreme --json --repeat 5 --warmup 2 \
     --output <private-path>'
```

Acceptance:

```text
new JSON fields are covered by local tests
renderer rejects missing case keys for A/B summaries
benchmarks still run in smoke mode without an H100
H100 stress/extreme runs include correctness checks
```

## Task 3: Reusable CCCL/CUB Workspace Experiments

Purpose: measure whether explicit reusable storage changes simplify and
matmul+simplify performance enough to justify an API.

Experiments:

```text
simplify without explicit workspace
simplify with explicit workspace
matmul+simplify without explicit workspace
matmul+simplify with explicit workspace
workspace pre-reserved before timing
workspace allowed to grow inside timing
low, medium, high duplicate-rate datasets
small, default, stress, and extreme term-count profiles
```

Profiler requirements:

```text
Nsight Systems trace for each retained candidate
Nsight Compute report for relevant CCCL/CUB kernels
temporary storage query and allocation counts in JSON
Compute Sanitizer memcheck after retained changes
```

Acceptance:

```text
retain only if median speedup is material on at least one target workload and no target workload regresses beyond noise without explanation
reject with evidence if allocation reuse is not the limiting factor
canonical ordering, zero-tolerance filtering, and coefficient accuracy tests pass
```

## Task 4: Duplicate-Reduction Pipeline Experiments

Purpose: determine whether a more explicit CUB/CCCL duplicate-reduction path
can improve over the current Thrust-heavy simplify/matmul+simplify path.

Candidate variants:

```text
current Thrust sort/reduce baseline
explicit CUB temporary storage reuse around radix sort where compatible
separate key/value compaction with explicit workspace
special-case low-word keys only if profiler evidence shows key-width overhead
```

Rules:

```text
do not ship a custom sort
do not change canonical key ordering
do not change tolerance semantics
do not keep a variant that wins only by skipping correctness work
```

Acceptance:

```text
A/B table includes same revision family, same H100, same dataset, and same timing boundary
SASS/PTX inspection is captured for any launch-bound, register-pressure, or code-generation claim
retained code has CUDA tests and CPU-only build tests
```

## Task 5: Device-Resident Expectation Experiments

Purpose: separate host-copy, device-pointer, per-term reduction, final
coefficient accumulation, synchronization, and deterministic-mode costs for
statevector expectation.

Experiments:

```text
host NumPy statevector copied each call
CUDA-array-interface statevector reused across calls
operator-resident host-statevector boundary
device-statevector resident boundary
fused final reduction prototype
CUB reduction prototype
deterministic reduction mode if floating-point variance is material
```

Correctness checks:

```text
complex64 and complex128 statevectors
normalized and intentionally non-normalized fixtures
small exact fixtures against dense NumPy references
large random fixtures against current CUDA path within documented tolerances
wrong-device, non-contiguous, wrong dtype, wrong length, and moved-from errors
```

Acceptance:

```text
device-resident timing is not mixed with transfer-inclusive timing
retained changes preserve documented floating-point tolerances
cuStateVec baseline remains semantically matched only for Pauli-basis expectation
```

## Task 6: Dense Commutation Materialization Experiments

Purpose: decide whether the remaining commutation headroom is API/materialization
bound rather than kernel-instruction bound.

Experiments:

```text
existing vector-returning host output
existing caller-owned host output
caller-owned pinned host output
caller-owned device byte output
bit-packed output prototype
async copy prototype if stream semantics are already documented
words==1 and words==2 specialization only after output-bound variants are separated
```

Acceptance:

```text
large dense outputs keep guardrail behavior
all output forms use the same row-major semantic oracle
host-output registration cost is measured separately from kernel cost where possible
no public async behavior ships without documented synchronization and error semantics
```

## Task 7: External GPU Baseline Refresh

Purpose: refresh open-source GPU comparisons without overstating semantic
equivalence.

Benchmark categories:

```text
cuQuantum cuStateVec: primitive-equivalent only for statevector Pauli expectation
CUDA-Q: end-to-end spin-operator observe workflows only
Qiskit Aer GPU: framework-level circuit/statevector workflows only
Qiskit SparsePauliOp and OpenFermion: CPU primitive baselines
```

Each baseline record must include:

```text
package name and version
install command
GPU enabled or unavailable reason
semantic mapping
timing boundary
correctness oracle
median and distribution statistics
raw exception text when unavailable
```

Acceptance:

```text
unavailable baselines are represented in machine-readable JSON
README and report do not present framework-level baselines as sparse-Pauli primitive speedups
```

## Task 8: Campaign Report And Closeout

Purpose: make the campaign durable and reviewable.

Required checked-in report sections:

```text
executive summary
hardware and environment identity
git revision provenance
exact H100 setup, validation, profiling, and benchmark commands
retained experiments
rejected experiments
correctness and sanitizer evidence
Nsight Systems and Nsight Compute interpretation
SASS/PTX findings where relevant
competitor baseline table
remaining headroom and next decisions
limitations and portability boundaries
```

Required visuals:

```text
speedup and latency plots from checked-in JSON
allocation and temporary-storage attribution plot
Nsight bottleneck summary plot
architecture diagram showing host, Python, C++ binding, CUDA runtime, H100 SMs, and HBM
kernel/dataflow diagram for retained CUDA changes
```

Closeout sequence:

```bash
python scripts/validate.py
git diff --check
git add -A
git commit -m "Complete H100 CUDA deep optimization campaign 2"
git switch main
git merge --ff-only "$FASTPAULI_BRANCH"
python scripts/validate.py
git push origin main
gh run list --branch main --limit 1
gh run watch "$(gh run list --branch main --limit 1 --json databaseId --jq '.[0].databaseId')" --exit-status
git branch -d "$FASTPAULI_BRANCH"
```

Acceptance:

```text
independent review covers CUDA API/lifetime, benchmark methodology, docs, and report claims
P0/P1 review findings are fixed before merge
merged main validates locally
remote CI is green
local feature branch is deleted after merge
```

## Stop Conditions

Stop and report a blocker instead of forcing a change when:

```text
the H100 host is unavailable or no longer has the required CUDA/Nsight tools
retained code would require changing public semantics without prior doc updates
profiler evidence contradicts the assumed bottleneck model
correctness, sanitizer, or CPU-only validation fails and cannot be fixed in scope
competitor package installation requires unsupported system changes
```

## Exhaustion Standard For Campaign 2

This campaign is exhausted only when every in-scope path has one of:

```text
retained implementation with correctness, sanitizer, profiler, A/B, and report evidence
rejected implementation with same-boundary A/B evidence and a technical rejection reason
documented design blocker with exact missing API/lifetime/security/compatibility decision
documented hardware/tool blocker with exact command, error, and next action
```

The final answer for the execution campaign must not claim exhaustion from a
single benchmark pass. It must list the experiment matrix, retained changes,
rejected changes, validation, CI status, and residual headroom.

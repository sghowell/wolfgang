# H100 CUDA Workspace And Device-Output Campaign 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert Campaign 3's remaining H100 headroom into API-safe CUDA workspace, CUB scratch-buffer, and output-materialization experiments with retained production changes only when correctness, ownership, and same-boundary benchmark evidence justify them.

**Execution status:** Completed on 2026-04-29. The checked report is
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign4_2026-04-29.md`;
raw JSON, profiler summaries, and plots are checked in under
`docs/benchmarks/data/cuda_deep_optimization_h100_campaign4_2026-04-29/` and
`docs/benchmarks/plots/cuda_h100_campaign4_*.svg`.

**Architecture:** Campaign 4 starts with a private internal CUDA workspace that owns reusable temporary storage for one device ordinal and one synchronization model. Explicit CUB/CCCL duplicate-reduction experiments, commutation device-output prototypes, and any statevector reduction variants must run behind that workspace or behind benchmark-only switches until a separate API review promotes a public surface.

**Tech Stack:** C++20, CUDA C++ 12.x, CCCL/CUB/Thrust, nanobind, pytest, Nsight Systems, Nsight Compute, Compute Sanitizer, Python benchmark/report renderers, H100 source builds with `FASTPAULI_CUDA_ARCHITECTURES=90`.

---

## Status

Status: planned after Campaign 3.

Campaign 3 source-of-truth evidence:

```text
report: docs/benchmarks/reports/cuda_deep_optimization_h100_campaign3_2026-04-28.md
data: docs/benchmarks/data/cuda_deep_optimization_h100_campaign3_2026-04-28/
primary retained optimization: packed 32-bit x/z CUDA simplify key for one-word operators with num_qubits <= 32
primary remaining headroom: private workspace design, explicit CUB scratch-buffer experiments, and commutation output materialization
```

Campaign 4 is not a raw PTX or small instruction-tuning campaign. Raw PTX,
inline PTX, launch-bound changes, or hand-written sort/reduce kernels are in
scope only after Nsight Compute and SASS evidence identify a specific compiler
or library bottleneck that the workspace and CUB experiments cannot address.

## Source Inputs

Read these files before writing code:

```text
README.md
docs/roadmap.md
docs/plans/cuda_deep_optimization_plan.md
docs/plans/h100_deep_optimization_campaign3_plan.md
docs/benchmarks/reports/cuda_deep_optimization_h100_campaign3_2026-04-28.md
docs/architecture/cuda_backend.md
docs/architecture/api_stability.md
docs/architecture/hardware_targets_and_testing.md
docs/benchmarks/protocol.md
docs/quality/code_review.md
docs/quality/code_standards.md
docs/quality/documentation_standards.md
```

## Scope

In scope:

```text
private internal CUDA workspace design and implementation
explicit CUB temporary-storage and duplicate-reduction experiments for simplify
workspace-backed matmul+simplify experiments
commutation caller-owned device-byte and bit-packed output prototypes behind benchmark-only surfaces
same-boundary A/B benchmarks against Campaign 3 baseline behavior
profiler evidence for workspace allocation, CUB/CCCL calls, device-output materialization, and any retained kernels
Compute Sanitizer coverage for retained CUDA changes
README-compatible broad CPU/CUDA/external comparison plots only if Campaign 4 updates benchmark evidence materially
final checked-in Campaign 4 report with retained, rejected, deferred, and exhausted paths
```

Out of scope for this campaign unless an explicit API-review task promotes the
surface first:

```text
public Python CUDA workspace object
public stream or async API
public commutation device-output API
public bit-packed commutation output API
public device-statevector wrapper
CUDA wheel release claims
A100, RTX, AMD/HIP, or Apple Metal/MPS performance claims
raw PTX rewrites without profiler/SASS proof
```

## File Structure

Planned implementation files:

```text
src/cuda/workspace.cuh
  Private CUDA workspace declarations, device binding, capacity records,
  typed scratch-buffer views, reset/release helpers, and error translation.

src/cuda/workspace.cu
  Private CUDA workspace definitions, allocation/growth/release logic,
  CUB temporary-storage reservation helpers, and benchmark-only mode parsing.

src/cuda/simplify_cuda.cu
  Workspace-aware simplify and CUB duplicate-reduction experiments.

src/cuda/matmul_cuda.cu
  Workspace-aware matmul+simplify handoff and duplicate-reduction timing labels.

src/cuda/commutation_cuda.cu
  Benchmark-only caller-owned device-byte and bit-packed output prototypes.

src/cuda/device_pauli_sum.cuh
  Private declarations only when workspace helpers need shared CUDA internals.
  Do not expose workspace types through include/fastpauli headers in this
  campaign.
```

Planned benchmark and report files:

```text
benchmarks/bench_cuda_kernels.py
  Per-operation workspace mode, scratch bytes, allocation/growth count,
  CUB strategy, output materialization, and prototype label fields.

benchmarks/bench_cuda_scaling.py
  Campaign 4 profiles for workspace duplicate reduction, matmul+simplify
  duplicate pressure, commutation materialization, and statevector reduction
  guardrails.

scripts/render_cuda_campaign4_assets.py
  Checked-in summary and SVG renderer for Campaign 4 data.

tests/test_cuda_scaling_benchmark.py
  JSON schema and deterministic profile coverage for new benchmark fields.

tests/test_cuda_deep_report_assets.py
  Renderer freshness, evidence normalization, and plot existence coverage.

tests/test_phase11_cuda_kernels.py
  CUDA correctness coverage for any retained workspace-backed behavior.
```

Planned docs and evidence files:

```text
docs/benchmarks/data/cuda_deep_optimization_h100_campaign4_YYYY-MM-DD/
docs/benchmarks/reports/cuda_deep_optimization_h100_campaign4_YYYY-MM-DD.md
docs/benchmarks/plots/cuda_h100_campaign4_*.svg
docs/plans/cuda_commutation_device_output_api_review.md
docs/architecture/cuda_backend.md
docs/architecture/api_stability.md
docs/benchmarks/protocol.md
docs/roadmap.md
```

## H100 Execution Contract

Use environment variables for the current H100 host and artifact roots:

```bash
export FASTPAULI_H100_SSH_TARGET="${FASTPAULI_H100_SSH_TARGET:?set to the current H100 SSH target}"
export FASTPAULI_H100_BASELINE_DIR=<private-path>
export FASTPAULI_H100_EXPERIMENT_DIR=<private-path>
export FASTPAULI_H100_ARTIFACT_ROOT=<private-path>
export FASTPAULI_H100_BASELINE_REVISION=bc68079f1db97822dd4c8ec35712f77c494ca2ed
export FASTPAULI_BRANCH=codex/h100-deep-optimization-campaign4
```

Baseline and experiment checkouts must stay independent:

```text
baseline checkout: exact Campaign 3 final main revision, pinned by FASTPAULI_H100_BASELINE_REVISION
experiment checkout: Campaign 4 branch or exact experiment commit
never reset a dirty experiment checkout
record both exact revisions in the final report
```

`FASTPAULI_H100_BASELINE_REVISION` is intentionally pinned to the Campaign 3
final `main` revision that produced the checked-in Campaign 3 report. Do not
replace it with moving `origin/main` during execution. If the human owner asks
to rebase Campaign 4 before execution, update the pinned baseline revision and
record the reason in the Campaign 4 report.

Minimum remote preflight command:

```bash
ssh "$FASTPAULI_H100_SSH_TARGET" \
  'set -eu; \
   ART=<private-path> \
   mkdir -p "$ART" && \
   nvidia-smi --query-gpu=name,uuid,driver_version,cuda_version,compute_cap --format=csv \
     > "$ART/gpu.csv" && \
   nvidia-smi -q > "$ART/nvidia-smi-q.txt" && \
   lscpu > "$ART/lscpu.txt" && \
   /usr/local/cuda/bin/nvcc --version > "$ART/nvcc-version.txt" && \
   nsys --version > "$ART/nsys-version.txt" && \
   ncu --version > "$ART/ncu-version.txt" && \
   compute-sanitizer --version > "$ART/compute-sanitizer-version.txt"'
```

## Decision Gates

Record each decision in the Campaign 4 report and update architecture docs in
the same branch when a decision changes a source-of-truth contract.

```text
workspace ownership: private C++ only, benchmark-only Python switch, or public experimental API review
workspace lifetime: device ordinal binding, moved-from behavior, growth, reset, release, and destruction
workspace timing boundary: absent, grows inside timing, pre-reserved outside timing, or reused across timed iterations
CUB scratch policy: Thrust/CCCL default, explicit CUB scratch, custom scratch layout, or rejected
duplicate-reduction policy: current packed-key path, CUB sort+reduce path, hybrid path, or rejected
commutation output policy: public host vector/fill only, private device-byte prototype, private bit-packed prototype, or separate public API plan
stream policy: public default-stream synchronize-before-return, private stream helper only, or separate public async API plan
statevector reduction policy: keep fused accumulator, CUB staged reduction, deterministic prototype, or rejected
retention policy: production, benchmark-only, report-only rejected, or design-deferred
```

## Task 0: Baseline Reproduction And Profiling Preflight

**Files:**

```text
No repository edits unless benchmark/profiler scripts fail before experiments begin.
Remote artifacts under <private-path>
```

- [ ] **Step 0.1: Create the Campaign 4 branch locally**

Run:

```bash
git switch main
git pull --ff-only origin main
git switch -c codex/h100-deep-optimization-campaign4
```

Expected: the new branch starts from the Campaign 3 final `main` revision.

- [ ] **Step 0.2: Push the branch before remote experiments**

Run:

```bash
git push -u origin codex/h100-deep-optimization-campaign4
```

Expected: the H100 experiment checkout can fetch the exact branch.

- [ ] **Step 0.3: Prepare clean H100 checkouts**

Run from the local machine:

```bash
ssh "$FASTPAULI_H100_SSH_TARGET" \
  'set -eu; \
   BASE=<private-path> \
   EXP=<private-path> \
   ART=<private-path> \
   BASELINE_REV=bc68079f1db97822dd4c8ec35712f77c494ca2ed; \
   if [ ! -d "$BASE/.git" ]; then \
     git clone https://github.com/sghowell/FastPauli.git "$BASE"; \
   fi; \
   if [ ! -d "$EXP/.git" ]; then \
     git clone https://github.com/sghowell/FastPauli.git "$EXP"; \
   fi; \
   cd "$BASE"; \
   git fetch origin; \
   git checkout --detach "$BASELINE_REV"; \
   git reset --hard "$BASELINE_REV"; \
   cd "$EXP"; \
   git fetch origin; \
   test -z "$(git status --porcelain)"; \
   git checkout codex/h100-deep-optimization-campaign4; \
   git pull --ff-only origin codex/h100-deep-optimization-campaign4; \
   mkdir -p "$ART/raw"; \
   git -C "$BASE" rev-parse HEAD > "$ART/baseline-revision.txt"; \
   git -C "$EXP" rev-parse HEAD > "$ART/experiment-revision.txt"'
```

Expected: both checkouts are clean, the baseline checkout is detached at
`bc68079f1db97822dd4c8ec35712f77c494ca2ed`, and both revisions are recorded.

- [ ] **Step 0.4: Validate baseline CUDA behavior**

Run on the H100 baseline checkout:

```bash
cd "$FASTPAULI_H100_BASELINE_DIR"
FASTPAULI_CUDA_ARCHITECTURES=90 uv run python -m pip install -e ".[test]" \
  --config-settings=cmake.define.FASTPAULI_ENABLE_CUDA=ON \
  --config-settings=cmake.define.FASTPAULI_CUDA_ARCHITECTURES=90 \
  --config-settings=cmake.define.CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++
FASTPAULI_VALIDATE_CUDA=1 uv run python scripts/validate.py
```

Expected: validation passes before any Campaign 4 code is judged.

- [ ] **Step 0.5: Capture Campaign 3-equivalent baseline timings**

Run on the H100 baseline checkout:

```bash
cd "$FASTPAULI_H100_BASELINE_DIR"
uv run python benchmarks/bench_cuda_scaling.py --profile stress --repeat 5 --warmup 2 --json \
  --output "$FASTPAULI_H100_ARTIFACT_ROOT/raw/baseline_cuda_scaling_stress.json"
uv run python benchmarks/bench_cuda_scaling.py --profile extreme --repeat 3 --warmup 1 --json \
  --output "$FASTPAULI_H100_ARTIFACT_ROOT/raw/baseline_cuda_scaling_extreme.json"
uv run python benchmarks/bench_cuda_scaling.py --profile materialization --repeat 5 --warmup 2 --json \
  --output "$FASTPAULI_H100_ARTIFACT_ROOT/raw/baseline_cuda_scaling_materialization.json"
```

Expected: baseline JSON includes correctness-checked simplify,
matmul+simplify, commutation, and statevector rows.

Acceptance:

```text
baseline and experiment revisions are recorded
baseline H100 CUDA validation passes
profiler tools are available or missing-tool reasons are recorded
baseline stress, extreme, and materialization JSON exists before optimization code is retained
```

## Task 1: Private CUDA Workspace Contract

**Files:**

```text
Create: src/cuda/workspace.cuh
Create: src/cuda/workspace.cu
Modify: src/cuda/device_pauli_sum.cuh
Modify: bindings/python/module.cpp
Modify: CMakeLists.txt
Test: tests/test_phase11_cuda_kernels.py
Docs: docs/architecture/cuda_backend.md
Docs: docs/architecture/api_stability.md
```

- [ ] **Step 1.1: Write CUDA workspace behavior tests**

Add CUDA-gated tests that exercise:

```text
workspace absent path returns the same simplify result as current CUDA simplify
workspace pre-reserved path returns the same simplify result
workspace growth-inside-timing path returns the same simplify result
workspace rejects cross-device reuse when multiple devices are present
workspace reset keeps capacity reusable for the same device
workspace release drops owned storage and allows later regrowth
private test hook reports snapshots without exposing device pointers
```

Run:

```bash
FASTPAULI_VALIDATE_CUDA=1 uv run python -m pytest tests/test_phase11_cuda_kernels.py -q
```

Expected before implementation: new workspace-specific tests fail or skip only
when CUDA is not built.

- [ ] **Step 1.2: Implement a private workspace owner**

Implement `src/cuda/workspace.cuh` with these internal-only concepts:

```cpp
namespace fastpauli::cuda::detail {

enum class WorkspaceTimingMode {
    kAbsent,
    kGrowInsideTiming,
    kPreReservedOutsideTiming,
};

struct WorkspaceSnapshot {
    int device_ordinal;
    std::size_t reserved_bytes;
    std::size_t high_watermark_bytes;
    std::size_t allocation_count;
    std::size_t growth_count;
};

class CudaWorkspace {
public:
    explicit CudaWorkspace(int device_ordinal);
    CudaWorkspace(const CudaWorkspace&) = delete;
    CudaWorkspace& operator=(const CudaWorkspace&) = delete;
    CudaWorkspace(CudaWorkspace&&) noexcept;
    CudaWorkspace& operator=(CudaWorkspace&&) noexcept;
    ~CudaWorkspace();

    int device_ordinal() const noexcept;
    void ensure_device(int operand_device_ordinal) const;
    void* reserve_bytes(std::size_t bytes, std::size_t alignment);
    void reset() noexcept;
    void release() noexcept;
    WorkspaceSnapshot snapshot() const noexcept;
};

WorkspaceTimingMode workspace_timing_mode_from_env();

}  // namespace fastpauli::cuda::detail
```

The workspace must remain under `src/cuda/`; do not install or include it from
`include/fastpauli`.

- [ ] **Step 1.3: Add clear error translation**

Map allocation, device-mismatch, and moved-from workspace misuse into the same
CUDA exception policy used by `DevicePauliSum`. Error messages must include:

```text
operation name
workspace device ordinal
operand device ordinal
requested bytes when allocation fails
```

- [ ] **Step 1.4: Add a private test hook, not a public API**

Expose a deliberately private, underscored test hook in `bindings/python/module.cpp`
only when CUDA support is compiled:

```text
_cuda_workspace_probe_for_testing(device=0, reserve_bytes=(4096, 8192), reset=True, release=True)
```

The hook returns a dictionary with:

```text
cuda_enabled
runtime_available
device_ordinal
snapshots before reserve, after each reserve, after reset, and after release
allocation_count
growth_count
high_watermark_bytes
status or skip_reason
```

The hook must not return raw device pointers, must not be documented in
README.md or user docs, and must raise or report the same absent-CUDA and
runtime-unavailable states as existing CUDA test helpers. This hook exists so
pytest can verify private workspace ownership and lifetime without promoting a
public workspace API.

- [ ] **Step 1.5: Validate CPU-only build safety**

Run locally:

```bash
uv run python scripts/validate.py
```

Expected: CPU-only local validation still passes without CUDA headers in public
headers.

Acceptance:

```text
workspace is private and not installed
workspace ownership is tied to one device ordinal
workspace growth, reset, release, and snapshot semantics are tested
private test hook is underscored, undocumented, and does not expose device pointers
CPU-only builds do not include CUDA public headers
docs state that the workspace is not public API
```

Suggested commit:

```bash
git add src/cuda/workspace.cuh src/cuda/workspace.cu src/cuda/device_pauli_sum.cuh bindings/python/module.cpp CMakeLists.txt tests/test_phase11_cuda_kernels.py docs/architecture/cuda_backend.md docs/architecture/api_stability.md
git commit -m "Add private CUDA workspace contract"
```

## Task 2: Workspace Benchmark Instrumentation

**Files:**

```text
Modify: benchmarks/bench_cuda_kernels.py
Modify: benchmarks/bench_cuda_scaling.py
Modify: tests/test_cuda_scaling_benchmark.py
Modify: tests/test_benchmark_metadata.py
Modify: docs/benchmarks/protocol.md
```

- [ ] **Step 2.1: Extend benchmark JSON schema**

For each CUDA case affected by workspace experiments, emit these fields:

```json
{
  "workspace_mode": "absent|grow_inside_timing|pre_reserved_outside_timing",
  "workspace_reserved_bytes": 0,
  "workspace_high_watermark_bytes": 0,
  "workspace_allocation_count": 0,
  "workspace_growth_count": 0,
  "cub_strategy": "none|device_radix_sort_reduce|device_run_length_encode|device_reduce_by_key",
  "scratch_bytes_requested": 0,
  "result_materialization_target": "host_vector|caller_owned_host_bytes|caller_owned_device_bytes|bitpacked_device_prototype|none",
  "timing_boundary": "transfer_inclusive|device_resident|preallocated|prototype"
}
```

- [ ] **Step 2.2: Add Campaign 4 benchmark profiles**

Add deterministic `bench_cuda_scaling.py --profile campaign4_workspace`
covering:

```text
simplify one-word <=32 qubits with low, medium, high, and pathological duplicate rates
simplify multi-word >64 qubits with medium and high duplicate rates
matmul+simplify duplicate pressure at 512x512, 1024x1024, 2048x2048, 4096x4096
pairwise commutation host vector, caller-owned host bytes, device-byte prototype, and bit-packed prototype
statevector expectation guardrail rows for complex64 and complex128
```

Every row must keep correctness checks enabled. Extreme rows may be opt-in, but
their JSON must still record the skipped or unavailable reason when not run.

- [ ] **Step 2.3: Test benchmark compatibility**

Run:

```bash
uv run python -m pytest tests/test_cuda_scaling_benchmark.py tests/test_benchmark_metadata.py -q
uv run python benchmarks/bench_cuda_scaling.py --profile smoke --repeat 1 --json
uv run python benchmarks/bench_cuda_scaling.py --profile campaign4_workspace --repeat 1 --warmup 0 --json
```

Expected: smoke profiles remain compatible, and Campaign 4 rows include the
new workspace/materialization fields.

Acceptance:

```text
benchmark JSON labels every workspace and materialization timing boundary
old smoke/default/stress/extreme consumers remain compatible
unavailable workspace or prototype rows report explicit reasons
benchmark protocol documents the new fields and timing labels
```

Suggested commit:

```bash
git add benchmarks/bench_cuda_kernels.py benchmarks/bench_cuda_scaling.py tests/test_cuda_scaling_benchmark.py tests/test_benchmark_metadata.py docs/benchmarks/protocol.md
git commit -m "Add Campaign 4 CUDA workspace benchmark fields"
```

## Task 3: Explicit CUB Duplicate-Reduction Experiments

**Files:**

```text
Modify: src/cuda/simplify_cuda.cu
Modify: src/cuda/matmul_cuda.cu
Modify: src/cuda/workspace.cuh
Modify: src/cuda/workspace.cu
Test: tests/test_phase11_cuda_kernels.py
Benchmark: benchmarks/bench_cuda_scaling.py
```

- [ ] **Step 3.1: Add env-gated CUB strategies**

Support benchmark-only strategy selection with:

```bash
export FASTPAULI_CUDA_BENCH_DUPLICATE_REDUCTION=thrust_default
export FASTPAULI_CUDA_BENCH_DUPLICATE_REDUCTION=cub_radix_sort_reduce
export FASTPAULI_CUDA_BENCH_DUPLICATE_REDUCTION=cub_radix_sort_run_length
```

Invalid values must fail clearly in benchmarks and tests. Public Python
methods must continue to use the production default unless the benchmark
harness explicitly opts into a strategy.

- [ ] **Step 3.2: Prototype CUB sort plus reduce path**

Implement only behind the benchmark selector:

```text
key layout: existing packed-key32 path for words == 1 && num_qubits <= 32, existing full key layout otherwise
value layout: coefficient values preserve complex128 semantics
ordering: canonical x-then-z order exactly matches current simplify
tolerance: atol/rtol zero filtering exactly matches current simplify
workspace: all CUB temporary storage comes from CudaWorkspace when workspace mode is not absent
fallback: if CUB strategy rejects a case, benchmark row records unavailable reason and runs current production path for correctness comparison
```

- [ ] **Step 3.3: Separate temporary-storage measurements from timed kernels**

For each strategy, benchmark:

```text
absent workspace
workspace grows inside timing
workspace pre-reserved outside timing
current production default
```

Do not compare pre-reserved workspace timings against allocation-inclusive
production timings without showing both labels.

- [ ] **Step 3.4: Validate correctness and profiler evidence**

Run on H100:

```bash
cd "$FASTPAULI_H100_EXPERIMENT_DIR"
FASTPAULI_VALIDATE_CUDA=1 uv run python -m pytest tests/test_phase11_cuda_kernels.py -q
uv run python benchmarks/bench_cuda_scaling.py --profile campaign4_workspace --repeat 5 --warmup 2 --json \
  --output "$FASTPAULI_H100_ARTIFACT_ROOT/raw/experiment_campaign4_workspace.json"
python scripts/cuda_deep_profile.py --execute --json --profile stress --repeat 3 --warmup 1 \
  --competitor-set none --require-profiler-artifacts --continue-on-error \
  --output-root "$FASTPAULI_H100_ARTIFACT_ROOT/experiment_profile" \
  > "$FASTPAULI_H100_ARTIFACT_ROOT/raw/experiment_profile_report.json"
```

Acceptance:

```text
CUB strategy results match current CUDA simplify and CPU canonical output
same-boundary speedups or regressions are recorded for every strategy
Nsight evidence explains allocation, sort, reduce, compaction, and memory-traffic behavior
production code is retained only if same-boundary H100 evidence improves performance or stability
```

Suggested commit:

```bash
git add src/cuda/simplify_cuda.cu src/cuda/matmul_cuda.cu src/cuda/workspace.cuh src/cuda/workspace.cu tests/test_phase11_cuda_kernels.py benchmarks/bench_cuda_scaling.py
git commit -m "Prototype workspace-backed CUDA duplicate reduction"
```

## Task 4: Commutation Device-Output API Review And Prototypes

**Files:**

```text
Create: docs/plans/cuda_commutation_device_output_api_review.md
Modify: src/cuda/commutation_cuda.cu
Modify: benchmarks/bench_cuda_kernels.py
Modify: benchmarks/bench_cuda_scaling.py
Modify: tests/test_phase11_cuda_kernels.py
Docs: docs/architecture/cuda_backend.md
Docs: docs/architecture/api_stability.md
```

- [ ] **Step 4.1: Write the API review before exposing any public method**

Create `docs/plans/cuda_commutation_device_output_api_review.md` with these
decisions:

```text
public status: no public device-output API in Campaign 4 unless this review is explicitly accepted
output ownership: caller-owned device bytes, caller-owned bit-packed device words, or owned FastPauli device object
dtype and shape: uint8 row-major dense matrix or uint64 bit-packed row-major matrix
device ordinal: operands and output must live on the same CUDA device
stream behavior: default-stream synchronize-before-return unless a separate async API plan is accepted
errors: absent CUDA, moved-from operands, wrong device, wrong shape, wrong dtype, oversized dense output, and allocation failure
interop: accepted Python device-array protocols if public exposure is proposed
benchmark labels: public host vector, public caller-owned host bytes, private device bytes, private bit-packed device words
```

- [ ] **Step 4.2: Keep prototypes benchmark-only**

Implement private prototype timing paths only when the benchmark harness sets:

```bash
export FASTPAULI_CUDA_BENCH_COMMUTATION_OUTPUT=host_vector
export FASTPAULI_CUDA_BENCH_COMMUTATION_OUTPUT=caller_owned_host_bytes
export FASTPAULI_CUDA_BENCH_COMMUTATION_OUTPUT=caller_owned_device_bytes
export FASTPAULI_CUDA_BENCH_COMMUTATION_OUTPUT=bitpacked_device_words
```

Invalid values fail clearly. Public `commutes_with()` and
`commutes_with_into()` behavior stays unchanged.

- [ ] **Step 4.3: Measure materialization boundaries**

Run on H100:

```bash
uv run python benchmarks/bench_cuda_scaling.py --profile campaign4_workspace --repeat 5 --warmup 2 --json \
  --output "$FASTPAULI_H100_ARTIFACT_ROOT/raw/experiment_commutation_materialization.json"
```

Expected rows:

```text
host vector public path
caller-owned host bytes public path
caller-owned device bytes private prototype
bit-packed device words private prototype
transfer-inclusive copy back to host when measured
```

Acceptance:

```text
public commutation behavior and docs remain unchanged unless the API review is accepted
private prototype results are labeled as prototype timing boundaries
dense-output guardrails and row-major semantics stay enforced
bit-packed output documents bit order and matrix shape in the report
```

Suggested commit:

```bash
git add docs/plans/cuda_commutation_device_output_api_review.md src/cuda/commutation_cuda.cu benchmarks/bench_cuda_kernels.py benchmarks/bench_cuda_scaling.py tests/test_phase11_cuda_kernels.py docs/architecture/cuda_backend.md docs/architecture/api_stability.md
git commit -m "Plan commutation device-output CUDA boundary"
```

## Task 5: Statevector Reduction Recheck

**Files:**

```text
Modify: src/cuda/expectation_cuda.cu only if profiler evidence supports an experiment
Modify: benchmarks/bench_cuda_scaling.py
Test: tests/test_phase11_cuda_kernels.py
```

- [ ] **Step 5.1: Profile the current fused accumulator first**

Run on H100:

```bash
ncu --target-processes all --set detailed \
  --kernel-name regex:statevector \
  --export "$FASTPAULI_H100_ARTIFACT_ROOT/ncu_statevector_campaign4" \
  uv run python benchmarks/bench_cuda_scaling.py --profile campaign4_workspace --repeat 3 --warmup 1 --json
```

Expected: the report identifies whether reduction, memory bandwidth, atomics,
launch overhead, or host synchronization is material.

- [ ] **Step 5.2: Add a staged or CUB reduction only when evidence supports it**

Prototype a statevector reduction variant only if Step 5.1 shows reduction or
atomic pressure. Measure:

```text
complex64
complex128
host NumPy statevector copied internally
CUDA-array-interface device-resident statevector
repeatability across fixed seeds
absolute and relative error against CPU scalar and current CUDA fused accumulator
```

Acceptance:

```text
statevector public API remains CUDA-array-interface based
new reduction topology is retained only when it improves same-boundary timings and stays within documented dtype tolerances
otherwise the report records the current fused accumulator as intentionally retained
```

Suggested commit:

```bash
git add src/cuda/expectation_cuda.cu benchmarks/bench_cuda_scaling.py tests/test_phase11_cuda_kernels.py
git commit -m "Evaluate Campaign 4 statevector reduction topology"
```

## Task 6: Report Renderer And Evidence Assets

**Files:**

```text
Create: scripts/render_cuda_campaign4_assets.py
Modify: tests/test_cuda_deep_report_assets.py
Create: docs/benchmarks/data/cuda_deep_optimization_h100_campaign4_YYYY-MM-DD/summary.json
Create: docs/benchmarks/plots/cuda_h100_campaign4_workspace_boundaries.svg
Create: docs/benchmarks/plots/cuda_h100_campaign4_duplicate_reduction.svg
Create: docs/benchmarks/plots/cuda_h100_campaign4_commutation_materialization.svg
Create: docs/benchmarks/plots/cuda_h100_campaign4_cross_comparison.svg
Create: docs/benchmarks/plots/cuda_h100_campaign4_evidence_status.svg
```

- [ ] **Step 6.1: Build a checked-data renderer**

The renderer must load raw JSON from:

```text
docs/benchmarks/data/cuda_deep_optimization_h100_campaign4_YYYY-MM-DD/raw/
```

and emit:

```text
summary.json with retained/rejected/deferred decisions
workspace boundary plot
duplicate-reduction A/B plot
commutation materialization plot
cross-comparison plot with CPU scalar, CPU auto, available CPU selectors, CUDA transfer, CUDA resident, and external baselines
evidence status plot
```

- [ ] **Step 6.2: Add freshness tests**

Add tests that:

```text
run the renderer against checked raw JSON
compare generated summary JSON with the checked summary JSON
compare generated SVGs with checked SVGs
verify privileged NCU evidence supersedes nonprivileged ERR_NVGPUCTRPERM rows when both exist
fail when required retained-change evidence is missing
```

Run:

```bash
uv run python -m pytest tests/test_cuda_deep_report_assets.py -q
```

Acceptance:

```text
plots and summary are reproducible from checked raw data
README-facing plot, when updated, is broad cross-comparison evidence rather than CUDA-only A/B evidence
private prototype rows are visually labeled as prototypes
```

Suggested commit:

```bash
git add scripts/render_cuda_campaign4_assets.py tests/test_cuda_deep_report_assets.py docs/benchmarks/data/cuda_deep_optimization_h100_campaign4_*/ docs/benchmarks/plots/cuda_h100_campaign4_*.svg
git commit -m "Add Campaign 4 CUDA report renderer"
```

## Task 7: Competitor And CPU Baseline Refresh

**Files:**

```text
Modify: benchmarks/bench_competitive_baselines.py only if Campaign 4 adds comparable workload mappings
Create: docs/benchmarks/data/cuda_deep_optimization_h100_campaign4_YYYY-MM-DD/raw/competitive_baselines_final.json
```

- [ ] **Step 7.1: Refresh FastPauli CPU selector timings on the H100 host**

Run:

```bash
uv run python benchmarks/bench_cuda_scaling.py --profile default --repeat 7 --warmup 3 --json \
  --output "$FASTPAULI_H100_ARTIFACT_ROOT/raw/experiment_cuda_scaling_default.json"
```

Expected: scalar CPU, CPU auto, oneTBB, AVX2, and AVX-512 rows are captured
when compiled and available; unavailable selectors include reasons.

- [ ] **Step 7.2: Refresh external package baselines**

Run in an isolated H100 environment:

```bash
uv run python -m pip install qiskit openfermion cupy-cuda12x cuquantum-python-cu12 cudaq qiskit-aer-gpu
uv run python benchmarks/bench_competitive_baselines.py --repeat 5 --warmup 2 --json \
  --output "$FASTPAULI_H100_ARTIFACT_ROOT/raw/competitive_baselines_final.json"
```

Record package versions, install commands, GPU enablement, semantic mapping,
timing boundary, correctness oracle, and unavailable reasons.

Acceptance:

```text
external comparisons are used only when semantically comparable
CUDA-Q and Aer remain framework-level unless a sparse-Pauli primitive mapping is documented
cuStateVec and cuPauliProp rows state exact workload mapping and correctness tolerance
README cross-comparison includes all captured FastPauli CPU variants and comparable external baselines when Campaign 4 updates README performance evidence
```

Suggested commit:

```bash
git add benchmarks/bench_competitive_baselines.py docs/benchmarks/data/cuda_deep_optimization_h100_campaign4_*/raw/competitive_baselines_final.json
git commit -m "Refresh Campaign 4 CUDA competitor baselines"
```

## Task 8: Production Retention And Correctness Gate

**Files:**

```text
Modify only files touched by retained production changes.
Docs: docs/architecture/cuda_backend.md
Docs: docs/architecture/api_stability.md
Docs: docs/benchmarks/protocol.md
```

- [ ] **Step 8.1: Classify every experiment**

Use this table in the report:

```text
experiment
status: production | benchmark-only | rejected | deferred-to-api-review
same-boundary speedup or regression
correctness evidence
profiler evidence
API or ownership impact
reason for retaining or rejecting
```

- [ ] **Step 8.2: Run H100 retained-change validation**

Run:

```bash
FASTPAULI_VALIDATE_CUDA=1 uv run python scripts/validate.py
compute-sanitizer --tool memcheck uv run python -m pytest tests/test_phase11_cuda_kernels.py -q
compute-sanitizer --tool racecheck uv run python -m pytest tests/test_phase11_cuda_kernels.py -q
compute-sanitizer --tool initcheck uv run python -m pytest tests/test_phase11_cuda_kernels.py -q
compute-sanitizer --tool synccheck uv run python -m pytest tests/test_phase11_cuda_kernels.py -q
```

Expected: validation and sanitizer passes are clean for retained CUDA paths.

- [ ] **Step 8.3: Run local CPU-only validation**

Run locally on Apple Silicon:

```bash
uv run python scripts/validate.py
git diff --check
```

Expected: CPU-only validation passes, docs whitespace is clean, and no CUDA
header leaked into public CPU-only headers.

Acceptance:

```text
retained production changes have same-boundary performance evidence
retained production changes pass CPU-only and H100 CUDA validation
benchmark-only prototypes cannot be invoked as supported public APIs
docs and benchmark labels match implementation behavior
```

Suggested commit:

```bash
git add src/cuda benchmarks tests docs
git commit -m "Retain Campaign 4 CUDA optimization decisions"
```

## Task 9: Campaign 4 Report, Review, Merge, And CI

**Files:**

```text
Create: docs/benchmarks/reports/cuda_deep_optimization_h100_campaign4_YYYY-MM-DD.md
Modify: README.md only if benchmark evidence shown there changes
Modify: docs/roadmap.md
Modify: docs/plans/cuda_deep_optimization_plan.md
Modify: docs/plans/h100_deep_optimization_campaign4_plan.md
```

- [ ] **Step 9.1: Write the final report**

The report must include:

```text
exact baseline and experiment revisions
H100 hardware, driver, CUDA toolkit, compiled architecture, compiler, CPU, OS, oneTBB, and CPU selector metadata
commands for validation, profiler runs, sanitizer runs, benchmark runs, competitor installs, and plot rendering
retained, rejected, deferred, and exhausted experiment table
workspace ownership and lifetime decision
CUB scratch-buffer decision
commutation device-output API-review decision
statevector reduction decision
CPU/CUDA/external cross-comparison plot when README evidence changes
profiler interpretation separating Python, CUDA API, allocation, transfer, kernel, CUB/CCCL, synchronization, and materialization costs
remaining headroom with concrete next actions
```

- [ ] **Step 9.2: Request independent review**

Review scope must include:

```text
CUDA correctness and sanitizer evidence
workspace ownership and lifetime semantics
public/private API boundary
benchmark fairness and same-boundary labeling
renderer freshness and README plot evidence
docs consistency across roadmap, CUDA architecture, API stability, and benchmark protocol
```

Resolve all P0/P1 findings. Resolve P2 findings that affect correctness,
public API, benchmark claims, or report reproducibility before merge.

- [ ] **Step 9.3: Validate, merge, push, and confirm CI**

Run:

```bash
uv run python scripts/validate.py
git status --short
git switch main
git pull --ff-only origin main
git merge --ff-only codex/h100-deep-optimization-campaign4
uv run python scripts/validate.py
git push origin main
gh run list --branch main --limit 5
gh run watch <main-run-id> --exit-status
git branch -d codex/h100-deep-optimization-campaign4
git push origin --delete codex/h100-deep-optimization-campaign4
```

Acceptance:

```text
final report and plots are checked in and reproducible
review evidence is recorded in closeout
merged main validation passes
main CI is green
local and remote Campaign 4 branches are deleted after merge
worktree is clean
```

## Exhaustion Criteria

Campaign 4 is complete only when all statements are true:

```text
private workspace ownership, lifetime, reset/release, growth, timing-boundary, and device-ordinal behavior are implemented or explicitly rejected with evidence
CUB scratch-buffer duplicate-reduction experiments are measured against Campaign 3 production behavior at the same timing boundaries
matmul+simplify duplicate-reduction experiments are measured separately from product generation
commutation output materialization is measured across public host paths and private device-output prototypes, with prototype labels preserved
statevector reduction topology is either retained unchanged with profiler evidence or replaced with same-boundary speedup and tolerance evidence
Compute Sanitizer is clean for retained CUDA changes
Nsight Systems and Nsight Compute evidence exists for retained and rejected CUDA performance decisions
CPU scalar, CPU auto, available optimized CPU selectors, CUDA transfer-inclusive, CUDA device-resident, and comparable external baselines are refreshed when README/report comparisons change
the report identifies remaining headroom as either public API design, cross-GPU portability, backend expansion, or exhausted H100-local work
```

## Stop Conditions

Stop implementation and report the blocker instead of forcing a result when:

```text
H100 profiler access is unavailable and the missing profiler data is required for a retention decision
Compute Sanitizer reports a retained-path correctness or memory-safety issue
workspace-backed paths change canonical ordering, tolerance behavior, or documented synchronization semantics
CUB prototypes require a public workspace, stream, or output API before correctness can be expressed
external baselines cannot be installed in a compatible environment and no semantically comparable local package remains
remote H100 instance instability prevents repeated benchmark confirmation across at least two independent runs for retained production changes
```

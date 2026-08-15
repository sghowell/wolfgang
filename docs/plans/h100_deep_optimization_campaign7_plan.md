# H100 CUDA Fused Consumer Campaign 7 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]` or `- [x]`) syntax for tracking.

**Goal:** Convert the five Campaign 6 remaining-headroom items into a measured CUDA campaign that fuses real downstream commutation workflows onto `DeviceCommutationMatrix`, then uses profiler evidence to decide whether count reductions, async/stream APIs, bit-packed layouts, and broader GPU claims are warranted.

**Architecture:** Campaign 7 is a fused-consumer campaign, not another isolated dense-output or fill-only campaign. The dense row-major `DeviceCommutationMatrix` and the synchronous public CUDA API remain the compatibility baseline while benchmark-only fused graph and grouping consumers prove whether downstream algorithms can stay GPU-resident. Public API expansion is allowed only after a written contract accepts lifetime, ownership, synchronization, error propagation, layout, and Python return semantics.

**Tech Stack:** C++20, CUDA C++ 12.x, nanobind, NumPy, CuPy, CUDA Array Interface v3, pytest, `bench_cuda_scaling.py`, `bench_cuda_kernels.py`, `bench_competitive_baselines.py`, Nsight Systems, Nsight Compute, Compute Sanitizer, H100 source builds with `FASTPAULI_CUDA_ARCHITECTURES=90`, and one additional NVIDIA architecture run before non-H100 GPU claims.

---

## Status

Status: completed on H100; report published at
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign7_2026-04-29.md`.

Campaign 6 source-of-truth evidence:

```text
plan: docs/plans/h100_deep_optimization_campaign6_plan.md
report: docs/benchmarks/reports/cuda_deep_optimization_h100_campaign6_2026-04-29.md
data: docs/benchmarks/data/cuda_deep_optimization_h100_campaign6_2026-04-29/
retained public consumer API: DeviceCommutationMatrix.count_commuting(axis=None|0|1)
retained public commutation matrix API: DevicePauliSum.commutes_with_device()
```

Campaign 6 remaining-headroom items that Campaign 7 must cover:

```text
1. Fuse real downstream commutation algorithms, such as graph construction or grouping, directly onto DeviceCommutationMatrix data.
2. Specialize count reductions only if those fused consumers still need standalone count summaries and profiler evidence shows reduction kernels dominate.
3. Revisit public async/stream APIs only after an accepted lifetime, event, stream capture, error propagation, and Python ownership contract.
4. Revisit bit-packed output only with a consumer whose measured memory capacity or bandwidth limit cannot be addressed by dense-layout fused kernels.
5. Add non-H100 portability runs for the retained consumer API on another NVIDIA architecture before making broader GPU claims.
```

## Source Inputs

Read these files before implementation:

```text
README.md
docs/roadmap.md
docs/plans/cuda_deep_optimization_plan.md
docs/plans/h100_deep_optimization_campaign6_plan.md
docs/plans/cuda_async_stream_api_review.md
docs/plans/cuda_commutation_consumer_api_review.md
docs/benchmarks/reports/cuda_deep_optimization_h100_campaign6_2026-04-29.md
docs/architecture/cuda_backend.md
docs/architecture/api_stability.md
docs/architecture/hardware_targets_and_testing.md
docs/architecture/testing_and_ci.md
docs/benchmarks/protocol.md
docs/quality/code_review.md
docs/quality/code_standards.md
docs/quality/documentation_standards.md
docs/user/performance.md
include/fastpauli/device_commutation_matrix.hpp
include/fastpauli/device_pauli_sum.hpp
src/cuda/commutation_cuda.cu
src/cuda/device_commutation_matrix.cu
src/cuda/device_commutation_matrix.cuh
src/cuda/workspace.cu
src/cuda/workspace.cuh
bindings/python/pauli_sum_py.cpp
benchmarks/bench_cuda_scaling.py
benchmarks/bench_cuda_kernels.py
benchmarks/bench_competitive_baselines.py
scripts/cuda_deep_profile.py
tests/test_phase11_cuda_kernels.py
tests/test_cuda_scaling_benchmark.py
tests/test_cuda_deep_report_assets.py
```

## Scope

In scope:

```text
written fused-consumer API and benchmark-surface review before public API changes
benchmark-only CUDA graph-construction consumers that operate on DeviceCommutationMatrix data without host dense materialization
benchmark-only CUDA grouping-oriented summaries that can feed CPU or future GPU grouping without copying the full dense matrix
same-boundary H100 A/B comparisons against host materialization, compact counts, CuPy dense consumers, CPU scalar, and optimized CPU selectors
profiler-gated count-reduction specialization only when fused workflows still depend on standalone counts and reduction kernels dominate measured time
async/stream API reconsideration only through an accepted lifetime, event, stream capture, error propagation, and Python ownership contract
bit-packed prototype only when dense-layout fused consumers show measured capacity or bandwidth limits that packed layout can address
one non-H100 NVIDIA portability run for the retained consumer API before broadening GPU claims beyond H100 source-build evidence
Campaign 7 report, raw data, metadata, profiler evidence, generated plots, and README broad landscape refresh when evidence supersedes Campaign 6
```

Out of scope unless this plan's decision gates explicitly accept the surface:

```text
public async methods
public stream-handle arguments
public event classes
public bit-packed commutation output
raw device pointer APIs beyond existing CUDA Array Interface metadata
CUDA wheel release claims
HIP/AMD, Metal/MPS, Apple GPU, or non-NVIDIA backend work
multi-GPU claims
raw PTX or inline PTX without Nsight and SASS evidence for a specific compiler-codegen limit
```

## File Structure

Planned files for the implementation slice:

```text
docs/plans/cuda_fused_commutation_consumer_api_review.md
  Fused graph/grouping consumer contract, benchmark-only scope, public API decision gate, ownership rules, and correctness oracle.

docs/plans/cuda_async_stream_campaign7_decision.md
  Campaign 7-specific async/stream decision, building on docs/plans/cuda_async_stream_api_review.md and covering lifetime, events, capture, error propagation, and Python ownership.

docs/plans/cuda_bitpacked_commutation_campaign7_decision.md
  Campaign 7-specific bit-packed decision, trigger evidence, layout requirements, and retained or rejected status.

docs/architecture/cuda_backend.md
  Updated status for fused consumers, count-reduction specialization, async/stream deferral or acceptance, bit-packed deferral or acceptance, and portability-claim boundaries.

docs/architecture/api_stability.md
  Updated only if a public fused-consumer, async, stream, or bit-packed API is retained.

include/fastpauli/device_commutation_matrix.hpp
  Public declarations only for accepted public fused-consumer methods. Benchmark-only helpers stay out of this header.

src/cuda/device_commutation_matrix.cu
src/cuda/device_commutation_matrix.cuh
  Dense-matrix fused consumer kernels, count-specialization kernels retained by evidence, and private launch helpers.

src/cuda/commutation_cuda.cu
  Commutation population integration only when the fused consumer can share population-side data or workspace safely.

src/cuda/workspace.cu
src/cuda/workspace.cuh
  Reusable temporary-storage support for graph edge counts, prefix sums, CSR scatter, grouping summaries, and any accepted CUB/CCCL primitive use.

bindings/python/pauli_sum_py.cpp
  Python bindings only for public surfaces accepted by the fused-consumer API review.

benchmarks/bench_cuda_scaling.py
  Add Campaign 7 profiles for fused graph construction, grouping-oriented summaries, count specializations, bit-packed prototypes, and portability runs.

benchmarks/bench_cuda_kernels.py
  Add reusable timing schema fields for fused-consumer boundaries, reduction specialization status, dense-vs-packed evidence, and non-H100 metadata.

benchmarks/bench_competitive_baselines.py
  Add or extend comparable CPU, CuPy, and external rows only where semantics match the fused consumer workload.

scripts/render_cuda_campaign7_assets.py
  Generate Campaign 7 summary JSON and report plots from checked raw data.

tests/test_phase11_cuda_kernels.py
  CUDA correctness tests for any retained public fused-consumer behavior and CPU-only error behavior.

tests/test_cuda_scaling_benchmark.py
  Non-CUDA tests proving Campaign 7 benchmark profiles, schema fields, and unavailable-status rows are present without requiring a GPU.

tests/test_cuda_deep_report_assets.py
  Renderer freshness tests for Campaign 7 checked summary and plots.

docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29/
docs/benchmarks/reports/cuda_deep_optimization_h100_campaign7_2026-04-29.md
docs/benchmarks/plots/cuda_h100_campaign7_*.svg
  Final H100 evidence bundle after execution.

docs/benchmarks/data/cuda_portability_campaign7_non_h100_nvidia_2026-04-29/
docs/benchmarks/reports/cuda_portability_campaign7_non_h100_nvidia_2026-04-29.md
  Non-H100 NVIDIA portability evidence bundle after execution.

README.md
docs/roadmap.md
docs/plans/cuda_deep_optimization_plan.md
docs/user/performance.md
  Source-of-truth links, next-slice status, portability wording, and broad performance visual policy.
```

The non-H100 report must record a lowercase hardware identifier in metadata, such as `a100_sm80`, `rtx6000ada_sm89`, or `l4_sm89`, before the first portability command is run.

## Public API Decision Gate

Campaign 7 starts conservative:

```text
existing public CUDA APIs remain default-stream and synchronize-before-return
existing DeviceCommutationMatrix dense uint8 layout remains the public matrix representation
existing DeviceCommutationMatrix.count_commuting(axis=None|0|1) remains the only public compact consumer unless the fused-consumer review accepts another method
fused graph and grouping consumers start as benchmark-only private helpers
count-reduction specializations are implementation details unless they change public performance evidence only
public stream handles remain deferred unless the async/stream decision accepts exact semantics
public async return objects remain deferred unless the async/stream decision accepts exact semantics
public bit-packed output remains deferred unless the bit-packed decision accepts exact layout, interop, ownership, and consumer semantics
```

Required decision artifacts:

```text
docs/plans/cuda_fused_commutation_consumer_api_review.md
docs/plans/cuda_async_stream_campaign7_decision.md
docs/plans/cuda_bitpacked_commutation_campaign7_decision.md
```

The fused-consumer review must accept or reject each candidate before code is exposed outside benchmark helpers:

```text
dense anti-commutation CSR graph construction from DeviceCommutationMatrix
dense row and column graph-degree summaries
grouping-oriented conflict summary suitable for greedy commuting-group construction
public Python method returning compact graph/grouping output
benchmark-only private method used only by H100 measurement scripts
```

Benchmark-only fused consumers must use this private access pattern unless the
review accepts a different one:

```text
private Python hook: fastpauli._fastpauli_core._benchmark_cuda_fused_commutation_consumer
allowed callers: benchmarks/bench_cuda_scaling.py, benchmarks/bench_cuda_kernels.py, and CUDA-gated tests
return shape: JSON-serializable dict with mode, rows, cols, timings, output_sizes, correctness_digest, and unavailable_reason
supported modes: csr_anticommutation_graph, conflict_degrees, grouping_summary, bitpacked_ab
visibility: never re-export from python/fastpauli/__init__.py and never document as user-facing API
CPU-only behavior: return unavailable status for benchmark scripts and raise the existing CUDA rebuild-guidance RuntimeError in CUDA-gated tests
```

The private hook may call C++ helpers in `src/cuda/device_commutation_matrix.*`
or a dedicated benchmark translation unit, but those helpers must remain out of
installed public headers unless the fused-consumer review accepts a public API.

If a public fused consumer is retained, it must specify:

```text
exact Python method name and C++ method name
return type and shape
commuting or anti-commuting edge convention
stable ordering of returned edges, rows, groups, or summaries
device and stream synchronization semantics
host copy size and transfer boundary
CPU-only error behavior
moved-from object behavior
memory allocation limit and failure mode
correctness oracle against a CPU reference
benchmark labels that distinguish fill, fused consumer, compact host copy, and full to_host()
```

No public method may be added if the review cannot define those fields.

## Fused Consumer Workloads

Campaign 7 must implement or explicitly reject the following workloads with evidence:

```text
1. Anti-commutation CSR graph construction
   Input: DeviceCommutationMatrix dense row-major uint8 flags.
   Edge convention: matrix value 0 is an anti-commuting edge.
   Output boundary: CSR row offsets and column indices, copied to host only for validation or benchmark reporting unless a public API is accepted.
   Correctness oracle: NumPy extraction of zero entries from matrix.to_host() on small and default-size references.

2. Row and column conflict-degree summaries
   Input: DeviceCommutationMatrix dense row-major uint8 flags.
   Degree convention: count of anti-commuting entries, which equals cols - commuting_count for rows and rows - commuting_count for columns.
   Output boundary: uint64 host vectors unless a fully device-resident follow-on grouping step is retained.
   Correctness oracle: CPU row and column sums over the inverted bool matrix.

3. Grouping-oriented conflict summary
   Input: DeviceCommutationMatrix dense row-major uint8 flags and deterministic term ordering.
   Summary convention: deterministic per-term conflict degrees, optional top-k conflict candidates, and enough metadata to drive a greedy grouping reference without full dense host materialization.
   Output boundary: compact host summary for Campaign 7 unless the fused-consumer review accepts a device-resident grouping prototype.
   Correctness oracle: existing CPU grouping tests plus pairwise commutation validation for every emitted group.
```

The CSR graph construction workload is the primary fused-consumer target. The degree summaries are retained only if they materially improve grouping or graph construction evidence beyond the existing `count_commuting(axis=0|1)` API.

## Task 0: Branch, H100 Setup, And Baseline Capture

**Files:**
- Read: `README.md`
- Read: `docs/roadmap.md`
- Read: `docs/plans/h100_deep_optimization_campaign7_plan.md`
- Create during execution: `docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29/metadata/`

- [x] **Step 0.1: Create the implementation branch**

Run:

```bash
git switch main
git pull --ff-only origin main
git switch -c codex/h100-campaign7
```

Expected: the branch starts from the current pushed `main`.

- [x] **Step 0.2: Prepare the H100 source-build checkout**

Run on the H100 host:

```bash
python -m venv .venv
.venv/bin/python -m pip install -U pip wheel build
FASTPAULI_CUDA_ARCHITECTURES=90 FASTPAULI_ENABLE_CUDA=ON .venv/bin/python -m pip install -e ".[test,qiskit,openfermion]"
.venv/bin/python -m pip install cupy-cuda12x cuquantum-python-cu12 qiskit-aer
```

Expected: `import fastpauli`, `import cupy`, and `fastpauli._fastpauli_core._build_info()["cuda_enabled"]` all succeed.

- [x] **Step 0.3: Capture baseline metadata**

Run on the H100 host:

```bash
mkdir -p docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29/metadata
git rev-parse HEAD > docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29/metadata/experiment-revision.txt
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv > docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29/metadata/gpu.csv
/usr/local/cuda/bin/nvcc --version > docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29/metadata/nvcc-version.txt
.venv/bin/python - <<'PY' > docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29/metadata/python-package-versions.txt
import importlib.metadata as md
for name in ["fastpauli", "numpy", "pytest", "cupy-cuda12x", "cuquantum-python-cu12", "qiskit", "openfermion", "qiskit-aer"]:
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}: not installed")
PY
```

Expected: metadata records the H100 device, CUDA toolkit, experiment revision, and package versions.

## Task 1: Fused Consumer Contract And Benchmark Schema

**Files:**
- Create: `docs/plans/cuda_fused_commutation_consumer_api_review.md`
- Modify: `benchmarks/bench_cuda_scaling.py`
- Modify: `benchmarks/bench_cuda_kernels.py`
- Modify: `tests/test_cuda_scaling_benchmark.py`
- Modify: `docs/architecture/cuda_backend.md`
- Modify: `docs/user/performance.md`

- [x] **Step 1.1: Write the fused-consumer API review**

Create `docs/plans/cuda_fused_commutation_consumer_api_review.md` with these required sections:

```text
# CUDA Fused Commutation Consumer API Review
Status: benchmark-only fused consumers first; public fused API deferred unless Campaign 7 evidence accepts exact return semantics.

Existing invariant: DeviceCommutationMatrix owns dense row-major uint8 flags.
Primary fused consumer: anti-commutation CSR graph construction.
Secondary fused consumer: row and column conflict-degree summaries.
Grouping-oriented consumer: deterministic compact conflict summary for greedy grouping.
Public API decision: accepted or rejected per method after correctness, timing, and ownership evidence.
Memory ownership: FastPauli owns device buffers; benchmark helpers must synchronize before host results are observed.
Ordering: row-major by lhs row, then rhs column.
Correctness oracle: CPU extraction from matrix.to_host() for small cases and existing CPU grouping validation for grouping summaries.
Failure modes: allocation failure, unsupported CUDA build, moved-from matrix, unsupported axis or mode.
Benchmark labels: fill, fused CSR, fused degree, grouping summary, compact host copy, full dense to_host, CuPy dense consumer.
```

Expected: the review rejects public fused APIs by default and defines exact evidence needed to retain one later in the campaign.

- [x] **Step 1.2: Add Campaign 7 benchmark schema tests before implementation**

Extend `tests/test_cuda_scaling_benchmark.py` so the non-CUDA test suite requires Campaign 7 profiles or schema fields for:

```text
fused_graph_csr
fused_conflict_degrees
fused_grouping_summary
count_specialization_status
bitpacked_decision_status
portability_gpu
```

Run:

```bash
python -m pytest tests/test_cuda_scaling_benchmark.py -q
```

Expected before implementation: failure naming the missing Campaign 7 profile or schema field.

- [x] **Step 1.3: Add benchmark CLI schema support**

Modify `benchmarks/bench_cuda_scaling.py` and `benchmarks/bench_cuda_kernels.py` so they emit the Campaign 7 schema fields. The benchmark must emit explicit unavailable rows when CUDA, CuPy, or a required fused helper is not available.

Run:

```bash
python -m pytest tests/test_cuda_scaling_benchmark.py -q
```

Expected after implementation: all tests in `tests/test_cuda_scaling_benchmark.py` pass on a CPU-only local machine.

- [x] **Step 1.4: Commit the contract and schema gate**

Run:

```bash
git add docs/plans/cuda_fused_commutation_consumer_api_review.md benchmarks/bench_cuda_scaling.py benchmarks/bench_cuda_kernels.py tests/test_cuda_scaling_benchmark.py docs/architecture/cuda_backend.md docs/user/performance.md
git commit -m "plan cuda fused consumer campaign"
```

Expected: the first Campaign 7 implementation commit contains the written contract and schema gate.

## Task 2: Anti-Commutation CSR Graph Construction

**Files:**
- Modify: `src/cuda/device_commutation_matrix.cu`
- Modify: `src/cuda/device_commutation_matrix.cuh`
- Modify: `src/cuda/workspace.cu`
- Modify: `src/cuda/workspace.cuh`
- Modify: `benchmarks/bench_cuda_scaling.py`
- Modify: `tests/test_phase11_cuda_kernels.py`

- [x] **Step 2.1: Add failing CUDA correctness tests**

Add CUDA-gated tests that construct small matrices with known commuting and anti-commuting pairs, then validate the fused CSR graph boundary against `matrix.to_host()`.

The expected CSR convention is:

```text
row_offsets length = rows + 1
row_offsets[0] = 0
row_offsets[i + 1] - row_offsets[i] = number of anti-commuting entries in row i
col_indices are sorted ascending within each row
col_indices contain rhs column indices where DeviceCommutationMatrix[row, col] == 0
```

Run on the H100 host:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 .venv/bin/python -m pytest tests/test_phase11_cuda_kernels.py -q
```

Expected before implementation: failure because the fused CSR helper is absent.

- [x] **Step 2.2: Implement benchmark-only CSR graph construction**

Implement private CUDA helpers that:

```text
count anti-commuting entries per row from the dense uint8 matrix
prefix-sum row counts into CSR row offsets using the existing workspace abstraction
scatter sorted column indices by row
copy CSR buffers to host only for validation and benchmark reporting
do not expose a public Python method unless docs/plans/cuda_fused_commutation_consumer_api_review.md is updated to accept one
```

Use CUB/CCCL primitives through the existing private workspace only when they reduce allocation churn or simplify prefix-sum correctness.

- [x] **Step 2.3: Add CSR graph benchmark rows**

Extend `benchmarks/bench_cuda_scaling.py` with Campaign 7 rows that separately time:

```text
commutation fill allocation path
commutation fill reuse path
fused CSR row-count pass
CSR prefix sum
CSR column-index scatter
compact CSR host copy
full dense to_host() baseline
CuPy dense consumer baseline when CuPy is importable
CPU scalar graph extraction baseline
best available optimized CPU graph extraction baseline
```

Run on H100:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 .venv/bin/python benchmarks/bench_cuda_scaling.py --profile stress --repeat 7 --warmup 2 --json --output docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29/raw/fused_graph_stress.json
```

Expected: raw JSON contains every timing boundary listed above or an explicit unavailable status with a reason.

- [x] **Step 2.4: Validate CSR graph correctness and commit**

Run:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 .venv/bin/python -m pytest tests/test_phase11_cuda_kernels.py -q
python -m pytest tests/test_cuda_scaling_benchmark.py -q
```

Expected: CUDA correctness tests pass on H100, and non-CUDA schema tests pass locally.

Commit:

```bash
git add src/cuda/device_commutation_matrix.cu src/cuda/device_commutation_matrix.cuh src/cuda/workspace.cu src/cuda/workspace.cuh benchmarks/bench_cuda_scaling.py tests/test_phase11_cuda_kernels.py tests/test_cuda_scaling_benchmark.py
git commit -m "bench cuda fused commutation graph"
```

## Task 3: Grouping-Oriented Fused Summaries

**Files:**
- Modify: `src/cuda/device_commutation_matrix.cu`
- Modify: `src/cuda/device_commutation_matrix.cuh`
- Modify: `benchmarks/bench_cuda_scaling.py`
- Modify: `tests/test_phase11_cuda_kernels.py`
- Modify: `tests/test_phase6_commutation_grouping.py`

- [x] **Step 3.1: Add correctness tests for conflict-degree summaries**

Add CUDA-gated tests for row and column anti-commutation degrees:

```text
row_conflicts[i] = number of rhs terms that anti-commute with lhs row i
col_conflicts[j] = number of lhs terms that anti-commute with rhs column j
row_conflicts[i] = matrix.cols - matrix.count_commuting(axis=1)[i]
col_conflicts[j] = matrix.rows - matrix.count_commuting(axis=0)[j]
```

Run on H100:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 .venv/bin/python -m pytest tests/test_phase11_cuda_kernels.py -q
```

Expected before implementation: failure because conflict-degree helpers are absent.

- [x] **Step 3.2: Implement benchmark-only conflict summaries**

Implement private CUDA helpers for row and column conflict degrees. Reuse count-reduction infrastructure only when it is faster or simpler than maintaining a separate inverted-count kernel; record the choice in benchmark metadata.

- [x] **Step 3.3: Add grouping-oriented benchmark rows**

Add benchmark rows that measure whether compact conflict summaries reduce end-to-end grouping preparation time compared with:

```text
full dense to_host() followed by CPU grouping preparation
existing count_commuting(axis=0|1) followed by CPU grouping preparation
CSR graph construction followed by CPU grouping preparation
CuPy dense consumer summaries when CuPy is available
```

Run on H100:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 .venv/bin/python benchmarks/bench_cuda_scaling.py --profile stress --repeat 7 --warmup 2 --json --output docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29/raw/fused_grouping_stress.json
```

Expected: raw JSON shows whether grouping-oriented summaries avoid dense host materialization and whether they outperform existing compact count consumers.

- [x] **Step 3.4: Commit grouping-oriented fused summaries**

Run:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 .venv/bin/python -m pytest tests/test_phase11_cuda_kernels.py -q
python -m pytest tests/test_cuda_scaling_benchmark.py tests/test_phase6_commutation_grouping.py -q
git add src/cuda/device_commutation_matrix.cu src/cuda/device_commutation_matrix.cuh benchmarks/bench_cuda_scaling.py tests/test_phase11_cuda_kernels.py tests/test_phase6_commutation_grouping.py tests/test_cuda_scaling_benchmark.py
git commit -m "bench cuda fused grouping summaries"
```

Expected: grouping summary tests and benchmark schema tests pass.

## Task 4: Profiler-Gated Count-Reduction Specialization

**Files:**
- Modify: `src/cuda/device_commutation_matrix.cu`
- Modify: `src/cuda/device_commutation_matrix.cuh`
- Modify: `benchmarks/bench_cuda_kernels.py`
- Modify: `benchmarks/bench_cuda_scaling.py`
- Create during execution: `docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29/profiler/`

- [x] **Step 4.1: Profile fused consumers before specializing counts**

Run on H100:

```bash
mkdir -p docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29/profiler
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 .venv/bin/python scripts/cuda_deep_profile.py --profile stress --json --output docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29/profiler/profile_plan.json
nsys profile --stats=true --force-overwrite=true --output docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29/profiler/nsys_campaign7_fused_consumers .venv/bin/python benchmarks/bench_cuda_scaling.py --profile stress --repeat 3 --warmup 1 --json --output docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29/raw/nsys_fused_consumers.json
```

Expected: Nsight Systems identifies fused CSR, grouping-summary, and count-reduction kernel time separately.

- [x] **Step 4.2: Apply the count-specialization gate**

Use this gate:

```text
Implement count-reduction specialization only if standalone count reductions are still required by the retained fused consumer and profiler evidence shows count kernels are the dominant retained-consumer bottleneck. Dominant means count-reduction kernels are the largest measured GPU-time bucket in the retained fused workflow and optimizing that bucket can change end-to-end workflow time.
Reject count specialization for Campaign 7 if fused CSR or grouping workflows are dominated by fill, scatter, host copy, allocation, or external consumer time.
```

Record the decision in the Campaign 7 raw summary under `count_specialization_status` with one of:

```text
retained
rejected_not_dominant
rejected_not_required_by_fused_consumer
deferred_needs_more_architectures
```

- [x] **Step 4.3: If retained, implement count specializations**

Allowed specializations:

```text
row-major row reductions that maximize contiguous reads
column reductions that use tiled shared-memory or transpose-style access to reduce strided global loads
total reductions that use warp-level reductions and block reductions without changing count semantics
CUB/CCCL reductions only when workspace reuse removes repeated allocation overhead
```

Run A/B:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 .venv/bin/python benchmarks/bench_cuda_scaling.py --profile stress --repeat 7 --warmup 2 --json --output docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29/raw/count_specialization_ab.json
```

Expected: retained specializations improve at least one retained count or fused-consumer row without regressing correctness or widening public semantics.

- [x] **Step 4.4: Commit count decision**

If retained, commit code and evidence hooks:

```bash
git add src/cuda/device_commutation_matrix.cu src/cuda/device_commutation_matrix.cuh benchmarks/bench_cuda_kernels.py benchmarks/bench_cuda_scaling.py tests/test_phase11_cuda_kernels.py
git commit -m "optimize cuda count reductions for fused consumers"
```

If rejected, commit the documented gate and benchmark schema only:

```bash
git add benchmarks/bench_cuda_kernels.py benchmarks/bench_cuda_scaling.py docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29/raw/count_specialization_ab.json
git commit -m "record cuda count reduction gate"
```

Expected: the commit message reflects whether code changed or the specialization was rejected by evidence.

## Task 5: Async And Stream API Decision

**Files:**
- Create: `docs/plans/cuda_async_stream_campaign7_decision.md`
- Modify: `docs/architecture/cuda_backend.md`
- Modify: `docs/architecture/api_stability.md` only if a public API is retained
- Modify: `benchmarks/bench_cuda_scaling.py` only if private stream/event timing is retained

- [x] **Step 5.1: Write the Campaign 7 async/stream decision**

Create `docs/plans/cuda_async_stream_campaign7_decision.md` with a status of either:

```text
public async/stream API deferred
private stream/event benchmark probe retained
public async/stream API retained with exact contract
```

The decision must cover:

```text
stream owner and device ordinal
event owner and destruction behavior
which objects are kept alive until completion
CUDA graph capture support or explicit rejection
where deferred CUDA errors become Python exceptions
whether timing is enqueue-only, event-elapsed, synchronization-only, or end-to-end
interaction with DeviceCommutationMatrix and private workspace allocations
```

- [x] **Step 5.2: Retain no public stream API unless the contract is complete**

The public API remains deferred unless the decision document defines every field above and the fused-consumer benchmarks need async execution to answer an end-to-end performance question.

If public async/stream remains deferred, update docs to say Campaign 7 reconsidered the surface and left the synchronous public invariant intact.

- [x] **Step 5.3: If private stream/event probes are retained, benchmark them with labels**

Run on H100:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 .venv/bin/python benchmarks/bench_cuda_scaling.py --profile stress --repeat 7 --warmup 2 --json --output docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29/raw/stream_event_probe.json
```

Expected: every stream/event row is labeled `private_benchmark_only` and is not compared against synchronized public API timing without a synchronized boundary row.

- [x] **Step 5.4: Commit async/stream decision**

Run:

```bash
git add docs/plans/cuda_async_stream_campaign7_decision.md docs/architecture/cuda_backend.md docs/architecture/api_stability.md benchmarks/bench_cuda_scaling.py
git commit -m "decide cuda stream api for fused consumers"
```

Expected: commit records an accepted, rejected, or deferred async/stream decision with no ambiguous public API surface.

## Task 6: Bit-Packed Output Decision

**Files:**
- Create: `docs/plans/cuda_bitpacked_commutation_campaign7_decision.md`
- Modify: `benchmarks/bench_cuda_scaling.py` only if a packed prototype is retained
- Modify: `src/cuda/device_commutation_matrix.cu` only if a packed prototype is retained
- Modify: `src/cuda/device_commutation_matrix.cuh` only if a packed prototype is retained
- Modify: `docs/architecture/cuda_backend.md`
- Modify: `docs/architecture/api_stability.md` only if a public API is retained

- [x] **Step 6.1: Apply the bit-packed trigger gate**

Retain a bit-packed prototype only when the dense fused consumer evidence shows one of:

```text
dense DeviceCommutationMatrix memory capacity prevents a target stress or extreme workload from running
dense fused CSR or grouping-summary kernels are bandwidth-bound and bit-packed access reduces end-to-end retained consumer time
dense host or device copy size is the measured bottleneck and packed layout avoids that copy without immediate unpacking
```

Reject bit-packed output for Campaign 7 if dense fused consumers solve the measured workload or if the packed path immediately unpacks back to dense flags.

- [x] **Step 6.2: Write the bit-packed decision**

Create `docs/plans/cuda_bitpacked_commutation_campaign7_decision.md` with:

```text
Status: retained, rejected, or deferred.
Trigger evidence: exact benchmark and profiler rows that motivated the decision.
Layout if retained: row-major uint64 words, word count per row, bit order, padding, and bit meaning.
Interop if retained: CUDA Array Interface unsupported unless a valid dtype/shape contract is defined.
Host materialization if retained: exact conversion behavior and synchronization semantics.
Consumer semantics if retained: which fused consumer uses packed data without unpacking to dense.
```

- [x] **Step 6.3: If retained, benchmark dense vs packed fused consumers**

Run on H100:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 .venv/bin/python benchmarks/bench_cuda_scaling.py --profile extreme --repeat 5 --warmup 2 --json --output docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29/raw/bitpacked_dense_ab.json
```

Expected: the packed row reports end-to-end fused consumer time, memory footprint, and correctness against the dense CSR or grouping oracle.

- [x] **Step 6.4: Commit bit-packed decision**

Run:

```bash
git add docs/plans/cuda_bitpacked_commutation_campaign7_decision.md docs/architecture/cuda_backend.md docs/architecture/api_stability.md benchmarks/bench_cuda_scaling.py src/cuda/device_commutation_matrix.cu src/cuda/device_commutation_matrix.cuh tests/test_phase11_cuda_kernels.py
git commit -m "decide cuda bitpacked fused consumer path"
```

Expected: commit records the retained, rejected, or deferred bit-packed decision and any supporting implementation.

## Task 7: Non-H100 NVIDIA Portability Run

**Files:**
- Modify: `benchmarks/bench_cuda_scaling.py`
- Modify: `docs/architecture/hardware_targets_and_testing.md`
- Modify: `docs/user/performance.md`
- Create during execution: `docs/benchmarks/data/cuda_portability_campaign7_non_h100_nvidia_2026-04-29/`
- Create during execution: `docs/benchmarks/reports/cuda_portability_campaign7_non_h100_nvidia_2026-04-29.md`

- [x] **Step 7.1: Select the non-H100 NVIDIA target**

Select one target in this priority order:

```text
1. A100, SM80, because it tests an older datacenter architecture with high memory bandwidth.
2. RTX 6000 Ada, SM89, because it tests current workstation Ada behavior.
3. L4 or A10, SM89 or SM86, because it tests lower-power deployment behavior.
```

Record the selected hardware identifier in `docs/benchmarks/data/cuda_portability_campaign7_non_h100_nvidia_2026-04-29/metadata/gpu.csv`.

- [x] **Step 7.2: Build and run the retained Campaign 7 consumer API**

Run on the non-H100 NVIDIA host:

```bash
python -m venv .venv
.venv/bin/python -m pip install -U pip wheel build
FASTPAULI_ENABLE_CUDA=ON .venv/bin/python -m pip install -e ".[test,qiskit,openfermion]"
FASTPAULI_VALIDATE_CUDA=1 .venv/bin/python scripts/validate.py
FASTPAULI_VALIDATE_CUDA=1 .venv/bin/python benchmarks/bench_cuda_scaling.py --profile stress --repeat 7 --warmup 2 --json --output docs/benchmarks/data/cuda_portability_campaign7_non_h100_nvidia_2026-04-29/raw/campaign7_portability_stress.json
```

Expected: validation passes, or the report records an explicit portability blocker with compiler, driver, toolkit, architecture, and failure output.

- [x] **Step 7.3: Add portability report**

The portability report must state:

```text
hardware and compute capability
driver and CUDA toolkit
compiled CUDA architectures
which Campaign 7 consumer surfaces ran
which benchmark rows are comparable to H100
which rows are unavailable and why
whether README claims can broaden beyond H100
```

- [x] **Step 7.4: Commit portability evidence**

Run:

```bash
git add docs/benchmarks/data/cuda_portability_campaign7_non_h100_nvidia_2026-04-29 docs/benchmarks/reports/cuda_portability_campaign7_non_h100_nvidia_2026-04-29.md docs/architecture/hardware_targets_and_testing.md docs/user/performance.md benchmarks/bench_cuda_scaling.py
git commit -m "record cuda campaign7 portability evidence"
```

Expected: commit includes the non-H100 run or a documented infrastructure blocker. No broad GPU claim is added without a passing retained consumer run.

## Task 8: Campaign 7 Report, Plots, README, And Roadmap

**Files:**
- Create: `scripts/render_cuda_campaign7_assets.py`
- Modify: `tests/test_cuda_deep_report_assets.py`
- Create: `docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29/summary.json`
- Create: `docs/benchmarks/reports/cuda_deep_optimization_h100_campaign7_2026-04-29.md`
- Create: `docs/benchmarks/plots/cuda_h100_campaign7_*.svg`
- Modify: `README.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/plans/cuda_deep_optimization_plan.md`
- Modify: `docs/user/performance.md`

- [x] **Step 8.1: Add the Campaign 7 renderer and freshness tests**

Implement `scripts/render_cuda_campaign7_assets.py` so it reads checked raw JSON, writes one summary JSON, and emits publication-quality SVG plots for:

```text
fused CSR graph construction vs full dense to_host()
grouping-oriented summaries vs count_commuting and dense host materialization
count-specialization A/B if retained
dense vs bit-packed fused consumer if retained
H100 vs non-H100 retained consumer portability
broad CPU/CUDA/external performance landscape if Campaign 7 supersedes Campaign 6 for README
```

Run:

```bash
python -m pytest tests/test_cuda_deep_report_assets.py -q
python scripts/render_cuda_campaign7_assets.py --data-dir docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29 --plot-dir docs/benchmarks/plots
```

Expected: renderer tests pass and generated plots are deterministic.

- [x] **Step 8.2: Write the Campaign 7 report**

The report must include:

```text
Evidence table with plan, report, summary, raw data, metadata, profiler, and plots.
Hardware and build metadata.
Validation status including pytest, scripts/validate.py, Compute Sanitizer, Nsight Systems, and Nsight Compute.
Fused-consumer API decision and public API status.
CSR graph construction correctness, timing, and memory footprint.
Grouping-oriented summary correctness, timing, and memory footprint.
Profiler findings with kernel-level bottlenecks.
Count-specialization retained or rejected decision.
Async/stream retained, rejected, or deferred decision.
Bit-packed retained, rejected, or deferred decision.
Non-H100 portability result or blocker.
Broad performance landscape update if Campaign 7 supersedes Campaign 6.
Remaining headroom for the next CUDA campaign.
```

- [x] **Step 8.3: Refresh README and roadmap**

Update the README performance section only if Campaign 7 evidence supersedes Campaign 6 for the broad landscape. Specialized fused-consumer plots can remain report-only unless they are integrated into the full CPU/CUDA/external comparison.

Update these source-of-truth links in every case:

```text
README.md
docs/roadmap.md
docs/plans/cuda_deep_optimization_plan.md
docs/user/performance.md
```

- [x] **Step 8.4: Commit report assets and docs**

Run:

```bash
python scripts/render_cuda_campaign7_assets.py --data-dir docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29 --plot-dir docs/benchmarks/plots
python -m pytest tests/test_cuda_deep_report_assets.py -q
git add scripts/render_cuda_campaign7_assets.py tests/test_cuda_deep_report_assets.py docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29 docs/benchmarks/reports/cuda_deep_optimization_h100_campaign7_2026-04-29.md docs/benchmarks/plots/cuda_h100_campaign7_*.svg README.md docs/roadmap.md docs/plans/cuda_deep_optimization_plan.md docs/user/performance.md
git commit -m "report cuda campaign7 fused consumers"
```

Expected: report, plots, README, roadmap, and performance guide agree on the same Campaign 7 evidence.

## Task 9: Review, Validation, Merge, Push, And CI

**Files:**
- Read: `docs/quality/code_review.md`
- Read: `docs/quality/phase_quality_gates.md`
- Read: `docs/benchmarks/protocol.md`

- [x] **Step 9.1: Run local validation**

Run:

```bash
python scripts/validate.py
git diff --check
```

Expected: validation passes and whitespace checks are clean.

- [x] **Step 9.2: Run H100 validation**

Run on H100:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 .venv/bin/python scripts/validate.py
compute-sanitizer --tool memcheck .venv/bin/python -m pytest tests/test_phase11_cuda_kernels.py -q
compute-sanitizer --tool racecheck .venv/bin/python -m pytest tests/test_phase11_cuda_kernels.py -q
compute-sanitizer --tool initcheck .venv/bin/python -m pytest tests/test_phase11_cuda_kernels.py -q
compute-sanitizer --tool synccheck .venv/bin/python -m pytest tests/test_phase11_cuda_kernels.py -q
```

Expected: validation passes and sanitizer summaries report zero CUDA memory errors or hazards. Known nanobind process-exit leak diagnostics must be reported separately from CUDA correctness.

- [ ] **Step 9.3: Complete review stage**

Follow `docs/quality/code_review.md`. The closeout must record:

```text
review scope
blocking findings and fixes
non-blocking findings and deferrals
validation after fixes
residual risk
```

- [ ] **Step 9.4: Merge, push, and confirm CI**

Run:

```bash
git switch main
git pull --ff-only origin main
git merge --ff-only codex/h100-campaign7
python scripts/validate.py
git push origin main
gh run list --limit 5 --json databaseId,headSha,status,conclusion,workflowName,displayTitle
gh run watch <campaign7-main-run-id> --exit-status
git branch -d codex/h100-campaign7
```

Expected: pushed `main` contains Campaign 7, local validation passes on merged `main`, remote CI is green, and the local feature branch is deleted.

## Exhaustion Criteria

Campaign 7 is complete only when all five Campaign 6 remaining-headroom items have evidence:

```text
1. Fused downstream commutation algorithms: CSR graph construction and grouping-oriented summaries are benchmarked on H100 or rejected with correctness/performance evidence.
2. Count reductions: specialization is retained only with profiler evidence that count kernels dominate a retained fused workflow; otherwise the rejection is recorded.
3. Async/stream APIs: public APIs remain deferred or are retained only after a complete lifetime, event, stream capture, error propagation, and Python ownership contract.
4. Bit-packed output: dense fused consumers are retained or a packed prototype is retained only with measured memory capacity or bandwidth evidence from a real consumer.
5. Non-H100 portability: one additional NVIDIA architecture runs the retained consumer API, or an infrastructure blocker is documented with exact commands and logs.
```

Required validation evidence:

```text
H100 source-build validation
CUDA correctness tests for retained public or benchmark-only helper behavior
Compute Sanitizer memcheck, racecheck, initcheck, and synccheck
Nsight Systems trace for fused consumer workflow
Nsight Compute profile for kernels that drive retained claims
raw benchmark JSON for every plotted result
metadata for CPU, GPU, driver, CUDA toolkit, compiled architectures, package versions, and git revision
README broad landscape update when Campaign 7 supersedes Campaign 6
review closeout following docs/quality/code_review.md
```

## Remaining Headroom After Campaign 7

Acceptable remaining headroom after Campaign 7 is limited to work that requires new evidence or hardware:

```text
public fused graph or grouping API if Campaign 7 retains only benchmark-only helpers
CUDA Graphs or stream-aware public execution if private stream/event evidence is positive but the public API remains deferred
multi-GPU or distributed GPU commutation workflows
additional non-H100 NVIDIA architecture coverage
DLPack interop for retained device outputs
HIP/AMD, Metal/MPS, or Apple GPU exploration
raw PTX or inline assembly only for a profiled compiler-codegen limit not expressible in CUDA C++ or CCCL/CUB
```

## Self-Review Checklist

Campaign 7 planning is acceptable when:

```text
every Campaign 6 remaining-headroom item maps to at least one task and one exhaustion criterion
public API changes are impossible without a written accepted contract
benchmark-only helpers are labeled and cannot be mistaken for public API promises
README updates preserve the broad CPU/CUDA/external performance landscape
non-H100 portability is required before broader GPU claims
all report paths, raw data paths, profiler paths, commands, and validation gates are explicit
```

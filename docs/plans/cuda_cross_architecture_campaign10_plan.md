# CUDA Cross-Architecture Headroom Campaign 10 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address every new remaining-headroom item from Campaign 9 with cross-architecture CUDA evidence, implemented code where the contract is accepted, or explicit rejection/blocker evidence that leaves no ambiguous deferred status.

**Architecture:** Campaign 10 is a portability-first campaign, not another H100-only hillclimb. It keeps H100 as the reference baseline, adds A100 as the required non-H100 datacenter lane, adds an RTX-class lane when available, and reopens public grouping, PyTorch DLPack, stream/CUDA Graph, and CSR scatter only behind exact contracts plus measured evidence.

**Tech Stack:** C++20, CUDA C++ 12.x or the installed CUDA toolkit on each GPU host, nanobind, NumPy, optional CuPy, optional PyTorch CUDA, pytest, Nsight Systems, Nsight Compute, Compute Sanitizer, `bench_cuda_scaling.py`, `bench_cuda_kernels.py`, `bench_competitive_baselines.py`, and checked SVG/JSON/Markdown benchmark artifacts.

---

## Status

Status: completed.

Completion evidence:

```text
report: docs/benchmarks/reports/cuda_cross_architecture_campaign10_2026-04-29.md
summary: docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29/summary.json
raw data: docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29/raw/
logs: docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29/logs/
plots: docs/benchmarks/plots/cuda_campaign10_*.svg
```

Final outcomes:

```text
1. Non-H100 NVIDIA portability: passed on A100 sm_80 and RTX PRO 6000 Blackwell sm_120.
2. PyTorch CUDA DLPack: passed on both Campaign 10 hosts.
3. Public grouping API: rejected with evidence; conflict_degrees remains the compact public summary.
4. Stream/CUDA Graph reprobe: rejected with evidence; launch overhead is not dominant.
5. CSR scatter reprobe: rejected with evidence; retained consumers do not need full CSR edge lists.
```

Campaign 10 starts from:

```text
report: docs/benchmarks/reports/cuda_deferred_headroom_campaign9_2026-04-29.md
summary: docs/benchmarks/data/cuda_deferred_headroom_campaign9_2026-04-29/summary.json
current README landscape: docs/benchmarks/plots/cuda_campaign9_performance_landscape.svg
```

Campaign 9 remaining headroom to address:

```text
1. Provision a real non-H100 NVIDIA source-build host and rerun portability.
2. Broaden DLPack consumer coverage to PyTorch CUDA when installed.
3. Consider a true public grouping API only after exact return, ownership, ordering, and docs contracts are accepted.
4. Consider stream/CUDA Graph work only if new profiler evidence shows launch or replay overhead dominates a retained consumer.
5. Reopen CSR scatter only if a future retained consumer requires full CSR edge lists.
```

Campaign 10 completion requires each item above to have a final non-deferred
outcome in the checked summary and report. Allowed terminal outcomes are:

```text
implemented
passed
rejected_with_evidence
blocked_external
blocked_toolchain
blocked_dependency
```

The final Campaign 10 summary must not contain `final_status: "deferred"`.

## Hardware Recommendation

Use both non-H100 hosts if available:

```text
A100: required first portability lane, because it covers sm_80 and an older data-center architecture.
RTX-class: recommended second portability lane, because it covers workstation/newer architecture behavior that A100 cannot expose.
```

If only one non-H100 host can be used, use A100 first. If both are available,
run A100 and RTX-class before making broader NVIDIA portability claims.

The RTX name must be verified from `nvidia-smi` before selecting the compile
architecture:

```text
NVIDIA RTX 6000 Ada: use FASTPAULI_CUDA_ARCHITECTURES=89.
NVIDIA RTX PRO 6000 Blackwell: use FASTPAULI_CUDA_ARCHITECTURES=120 if the installed toolkit accepts it.
```

The official NVIDIA CUDA GPU compute-capability table lists A100 as compute
capability 8.0, RTX 6000 Ada as 8.9, H100 as 9.0, and RTX PRO 6000 Blackwell
as 12.0. The execution report must still record the actual value from
`nvidia-smi --query-gpu=name,compute_cap,driver_version`.

Blackwell support is not a release claim until source build, runtime tests,
CUDA validation, and benchmark evidence pass on the actual RTX PRO 6000 host.
If CUDA/CMake rejects `FASTPAULI_CUDA_ARCHITECTURES=120`, record
`blocked_toolchain` for the Blackwell compile lane and do not silently compile
only older architectures while calling the result Blackwell-validated.

## Source Inputs

Read these before editing code or benchmark logic:

```text
README.md
docs/roadmap.md
docs/plans/cuda_deep_optimization_plan.md
docs/plans/h100_deep_optimization_campaign9_plan.md
docs/plans/cuda_fused_grouping_public_api_campaign9_contract.md
docs/plans/cuda_dlpack_interop_campaign9_contract.md
docs/plans/cuda_stream_graph_campaign9_contract.md
docs/plans/cuda_csr_scatter_campaign9_decision.md
docs/benchmarks/reports/cuda_deferred_headroom_campaign9_2026-04-29.md
docs/benchmarks/reports/cuda_portability_campaign9_non_h100_nvidia_2026-04-29.md
docs/architecture/cuda_backend.md
docs/architecture/api_stability.md
docs/architecture/hardware_targets_and_testing.md
docs/architecture/testing_and_ci.md
docs/benchmarks/protocol.md
docs/quality/code_review.md
docs/quality/code_standards.md
docs/quality/documentation_standards.md
docs/user/performance.md
bindings/python/pauli_sum_py.cpp
include/fastpauli/device_commutation_matrix.hpp
include/fastpauli/device_pauli_sum.hpp
include/fastpauli/pauli_sum.hpp
src/cuda/commutation_cuda.cu
src/cuda/device_commutation_matrix.cu
src/cuda/device_commutation_matrix.cuh
src/cuda/workspace.cu
src/cuda/workspace.cuh
src/grouping.cpp
benchmarks/bench_cuda_scaling.py
benchmarks/bench_cuda_kernels.py
benchmarks/bench_competitive_baselines.py
scripts/render_cuda_campaign9_assets.py
tests/test_phase11_cuda_kernels.py
tests/test_cuda_scaling_benchmark.py
tests/test_cuda_deep_report_assets.py
tests/test_phase6_commutation_grouping.py
```

External primary references:

```text
NVIDIA CUDA GPU compute capabilities: https://developer.nvidia.com/cuda/gpus
DLPack Python specification: https://dmlc.github.io/dlpack/latest/python_spec.html
CUDA streams and CUDA Graph stream-capture rules: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
Nsight Compute counter permissions: https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-nsightcompute
```

## Scope

In scope:

```text
cross-architecture source-build validation on A100 and RTX-class hosts
runtime, correctness, sanitizer, and benchmark evidence for non-H100 hosts
architecture-specific compile target handling for sm_80, sm_89, and sm_120 when hardware is available
PyTorch CUDA DLPack consumer tests and benchmark rows when a CUDA-enabled PyTorch wheel is installed
an exact public grouping API contract and either implementation or rejection with evidence
new Nsight Systems and Nsight Compute evidence for retained compact consumers before any stream/CUDA Graph work is accepted
CSR scatter reopening only if a retained public or private Campaign 10 consumer requires full CSR edge lists
new Campaign 10 raw JSON, metadata, plots, report, README landscape, roadmap, and performance-guide updates
independent agent review before merge
```

Out of scope:

```text
CUDA wheel release claims
multi-GPU public behavior
HIP, ROCm, AMD GPU, Metal, MPS, or Apple GPU implementation
raw device pointer public APIs
public async/event/stream objects without accepted API-stability text
mutable DLPack exports
silent fallback from a requested architecture to an older compiled architecture
CSR scatter tuning that improves only an unretained full-CSR baseline
raw PTX or inline PTX without a specific SASS/PTX code-generation finding
```

## Campaign 10 Status Schema

Every Campaign 10 raw row must carry:

```text
campaign: "cuda_cross_architecture_campaign10"
mode: one of "cross_arch_portability", "dlpack_pytorch", "public_grouping_api", "stream_graph_reprobe", "csr_scatter_reprobe", "readme_landscape"
campaign9_headroom_item: integer from 1 through 5
final_status: one of "implemented", "passed", "rejected_with_evidence", "blocked_external", "blocked_toolchain", "blocked_dependency"
deferred_status_allowed: false
decision_doc: repo-relative path to the contract or decision document
ssh_target: exact SSH target used for remote evidence, or "local" for local-only rows
provider_instance_type: exact provider instance type when available, or "not_available_to_agent" with an access note
gpu_name: exact GPU name from runtime evidence
gpu_compute_capability: major.minor compute capability from runtime evidence
cuda_driver: driver version
cuda_runtime: runtime version
cuda_toolkit: toolkit version
compiled_architectures: semicolon-separated architecture list
git_revision: full revision used for the run
command: exact command used for the row
correctness_digest: stable digest for deterministic result rows or empty string for profiler-only rows
unavailable_reason: empty string when available; exact reason otherwise
```

Renderer tests must fail if the summary omits any Campaign 9 remaining-headroom
item, contains `final_status: "deferred"`, or presents a Blackwell row without
recording whether `sm_120` compiled and ran.

## File Structure

Planned files for execution:

```text
docs/plans/cuda_cross_architecture_campaign10_plan.md
  This plan and the source-of-truth checklist for Campaign 10.

docs/plans/cuda_grouping_public_api_campaign10_contract.md
  Exact accept/reject contract for a true public CUDA grouping API.

docs/plans/cuda_stream_graph_campaign10_decision.md
  Exact accept/reject decision for stream and CUDA Graph work after fresh profiler evidence.

docs/plans/cuda_csr_scatter_campaign10_decision.md
  Exact accept/reject decision for CSR scatter after checking retained-consumer requirements.

docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29/
  Raw JSON, profiler exports, sanitizer logs, validation logs, environment metadata, and summary JSON.

docs/benchmarks/reports/cuda_cross_architecture_campaign10_2026-04-29.md
  Final Campaign 10 report with one subsection per remaining-headroom item and one hardware subsection per host.

docs/benchmarks/plots/cuda_campaign10_cross_architecture.svg
docs/benchmarks/plots/cuda_campaign10_dlpack_consumers.svg
docs/benchmarks/plots/cuda_campaign10_headroom_status.svg
docs/benchmarks/plots/cuda_campaign10_performance_landscape.svg
  Checked visual assets generated from Campaign 10 summary data.

scripts/render_cuda_campaign10_assets.py
  Campaign 10 summary and plot renderer. It may reuse Campaign 9 helpers only if tests preserve Campaign 9 outputs.

benchmarks/bench_cuda_scaling.py
benchmarks/bench_cuda_kernels.py
benchmarks/bench_competitive_baselines.py
  Campaign 10 benchmark profiles and optional PyTorch CUDA consumer rows.

bindings/python/pauli_sum_py.cpp
include/fastpauli/device_pauli_sum.hpp
src/cuda/commutation_cuda.cu
src/grouping.cpp
  Code changes only if the public grouping contract is accepted or a private stream/CSR probe is justified.

tests/test_phase11_cuda_kernels.py
tests/test_cuda_scaling_benchmark.py
tests/test_cuda_deep_report_assets.py
tests/test_phase6_commutation_grouping.py
  DLPack PyTorch coverage, Campaign 10 schema tests, renderer freshness tests, and grouping contract tests.

README.md
docs/roadmap.md
docs/plans/cuda_deep_optimization_plan.md
docs/architecture/hardware_targets_and_testing.md
docs/user/performance.md
AGENTS.md
  Source-of-truth status and planning updates.
```

## Task 1: Schema, Renderer, And Plan Tests

**Files:**
- Create: `scripts/render_cuda_campaign10_assets.py`
- Modify: `tests/test_cuda_deep_report_assets.py`
- Modify: `tests/test_cuda_scaling_benchmark.py`

- [x] **Step 1: Add Campaign 10 summary schema tests**

Add tests that load a fixture summary containing five items and assert:

```text
all campaign9_headroom_item values 1..5 are present
no final_status is deferred
Blackwell rows must record compiled_architectures and a compile outcome for 120
readme performance landscape excludes profiler-only rows
```

Run:

```bash
.venv/bin/python -m pytest tests/test_cuda_deep_report_assets.py -q
```

Expected before implementation: failure naming missing Campaign 10 renderer or
schema support.

- [x] **Step 2: Implement the renderer**

Create `scripts/render_cuda_campaign10_assets.py` with:

```text
summary loading
status validation
cross-architecture plot generation
DLPack consumer plot generation
headroom status plot generation
broad performance landscape plot generation
```

The renderer must preserve Campaign 9 plot behavior and must reject rows that
make architecture claims without hardware metadata.

- [x] **Step 3: Add benchmark profile declarations**

Add benchmark profile tests for:

```text
campaign10-portability
campaign10-dlpack-pytorch
campaign10-stream-graph-reprobe
campaign10-csr-scatter-reprobe
```

Run:

```bash
.venv/bin/python -m pytest tests/test_cuda_scaling_benchmark.py -q
```

Expected after implementation: all Campaign 10 profile tests pass locally, with
CUDA-only rows skipped or marked unavailable on CPU-only builds.

- [x] **Step 4: Commit schema and renderer scaffolding**

```bash
git add scripts/render_cuda_campaign10_assets.py tests/test_cuda_deep_report_assets.py tests/test_cuda_scaling_benchmark.py
git commit -m "test cuda campaign10 report schema"
```

## Task 2: Cross-Architecture Portability Evidence

**Files:**
- Modify: `benchmarks/bench_cuda_scaling.py`
- Modify: `scripts/render_cuda_campaign10_assets.py`
- Create: `docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29/`
- Create: `docs/benchmarks/reports/cuda_cross_architecture_campaign10_2026-04-29.md`

- [x] **Step 1: Record host inventory for each GPU host**

For each available host, run:

```bash
: "${FASTPAULI_CAMPAIGN10_HOST:?set to the exact user-provided SSH target, such as ubuntu@<private-address>}"
ssh "$FASTPAULI_CAMPAIGN10_HOST" 'hostname'
ssh "$FASTPAULI_CAMPAIGN10_HOST" 'command -v nvidia-smi'
ssh "$FASTPAULI_CAMPAIGN10_HOST" 'nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader'
ssh "$FASTPAULI_CAMPAIGN10_HOST" 'command -v nvcc || command -v /usr/local/cuda/bin/nvcc || true'
```

The report must record exact stdout, stderr, exit code, SSH target, provider
instance type when available, and whether the host is A100, RTX 6000 Ada, RTX
PRO 6000 Blackwell, or another named NVIDIA GPU. The `nvidia-smi` command is
separate so a missing driver or failed GPU query preserves its own nonzero exit
code and stderr instead of being masked by CUDA toolkit discovery.

- [x] **Step 2: Select compile architecture from the measured GPU**

Use:

```text
A100 with compute capability 8.0: FASTPAULI_CUDA_ARCHITECTURES=80
RTX 6000 Ada with compute capability 8.9: FASTPAULI_CUDA_ARCHITECTURES=89
RTX PRO 6000 Blackwell with compute capability 12.0: FASTPAULI_CUDA_ARCHITECTURES=120
```

Set the execution variables before validation and benchmark commands:

```bash
export FASTPAULI_CAMPAIGN10_GPU_ROLE=a100
export FASTPAULI_CAMPAIGN10_ARCH=80
```

For RTX 6000 Ada use `FASTPAULI_CAMPAIGN10_GPU_ROLE=rtx6000ada` and
`FASTPAULI_CAMPAIGN10_ARCH=89`. For RTX PRO 6000 Blackwell use
`FASTPAULI_CAMPAIGN10_GPU_ROLE=rtxpro6000blackwell` and
`FASTPAULI_CAMPAIGN10_ARCH=120`.

If a host reports a different compute capability, use the exact integer
architecture corresponding to the reported value and record the reason in the
summary.

- [x] **Step 3: Run source-build validation on each host**

After syncing the current branch to the remote workdir, run:

```bash
cd <private-path>
PATH=/usr/local/cuda/bin:/usr/local/cuda-12.9/bin:$PATH \
FASTPAULI_VALIDATE_CUDA=1 \
FASTPAULI_CUDA_ARCHITECTURES="$FASTPAULI_CAMPAIGN10_ARCH" \
.venv/bin/python scripts/validate.py
```

Expected success condition:

```text
CUDA build info reports cuda_enabled=True
compiled_architectures equals the requested architecture
pytest passes
Phase 10 and Phase 11 CUDA tests pass
CUDA benchmark smoke completes
sdist smoke completes
```

If the build fails because the toolkit does not accept the requested
architecture, record `blocked_toolchain` for that host and keep the failure log.

- [x] **Step 4: Run portability benchmarks**

On each validation-passing host, run:

```bash
cd <private-path>
PATH=/usr/local/cuda/bin:/usr/local/cuda-12.9/bin:$PATH \
FASTPAULI_CUDA_ARCHITECTURES="$FASTPAULI_CAMPAIGN10_ARCH" \
.venv/bin/python benchmarks/bench_cuda_scaling.py \
  --profile campaign10-portability \
  --repeat 7 \
  --warmup 2 \
  --json \
  --output "docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29/raw/portability_${FASTPAULI_CAMPAIGN10_GPU_ROLE}.json"
```

The benchmark profile must include retained compact consumers,
transfer-inclusive and device-resident CUDA rows, CPU scalar/default/optimized
selector rows where available, and exact unavailable reasons for optional
competitors.

- [x] **Step 5: Run sanitizer coverage**

On each validation-passing host, run:

```bash
cd <private-path>
PATH=/usr/local/cuda/bin:/usr/local/cuda-12.9/bin:$PATH \
compute-sanitizer --tool memcheck --error-exitcode 99 \
  .venv/bin/python -m pytest tests/test_phase11_cuda_kernels.py -q
```

Run racecheck/initcheck/synccheck on at least A100 and on the RTX host if the
runtime supports them without known profiler-permission blockers.

- [x] **Step 6: Commit portability evidence**

```bash
git add benchmarks/bench_cuda_scaling.py scripts/render_cuda_campaign10_assets.py docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29 docs/benchmarks/reports/cuda_cross_architecture_campaign10_2026-04-29.md
git commit -m "bench cuda campaign10 cross architecture portability"
```

## Task 3: PyTorch CUDA DLPack Consumer Coverage

**Files:**
- Modify: `tests/test_phase11_cuda_kernels.py`
- Modify: `benchmarks/bench_cuda_scaling.py`
- Modify: `docs/plans/cuda_dlpack_interop_campaign9_contract.md`
- Create or update: `docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29/raw/dlpack_pytorch.json`

- [x] **Step 1: Tighten the PyTorch CUDA test gate**

Update the optional PyTorch test so it records these skip reasons distinctly:

```text
torch not importable
torch importable but torch.cuda.is_available() is false
torch CUDA available but torch.utils.dlpack cannot consume the versioned read-only capsule
```

The test must consume `DeviceCommutationMatrix.__dlpack__(max_version=(1, 0))`
and must verify shape, dtype, device, values, and read-only behavior when the
consumer exposes mutability checks.

- [x] **Step 2: Install and verify PyTorch CUDA on at least one CUDA host**

Use the PyTorch installation command selected for the host's supported CUDA
wheel, then record the exact command in metadata. Verify with:

```bash
.venv/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
assert torch.cuda.is_available()
PY
```

If no CUDA-enabled PyTorch wheel is installable on the available hosts, record
`blocked_dependency` with the exact pip command and error output.

- [x] **Step 3: Run DLPack tests and benchmarks**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_phase11_cuda_kernels.py::test_cuda_device_commutation_matrix_dlpack_pytorch_consumer_matches_numpy \
  tests/test_phase11_cuda_kernels.py::test_cuda_device_commutation_matrix_dlpack_cupy_consumer_matches_numpy \
  -q
```

Then run:

```bash
.venv/bin/python benchmarks/bench_cuda_scaling.py \
  --profile campaign10-dlpack-pytorch \
  --repeat 7 \
  --warmup 2 \
  --json \
  --output docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29/raw/dlpack_pytorch.json
```

- [x] **Step 4: Commit DLPack evidence**

```bash
git add tests/test_phase11_cuda_kernels.py benchmarks/bench_cuda_scaling.py docs/plans/cuda_dlpack_interop_campaign9_contract.md docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29
git commit -m "bench cuda campaign10 pytorch dlpack coverage"
```

## Task 4: Public Grouping API Contract And Decision

**Files:**
- Create: `docs/plans/cuda_grouping_public_api_campaign10_contract.md`
- Modify if accepted: `include/fastpauli/device_pauli_sum.hpp`
- Modify if accepted: `bindings/python/pauli_sum_py.cpp`
- Modify if accepted: `src/cuda/commutation_cuda.cu`
- Modify if accepted: `src/grouping.cpp`
- Modify: `tests/test_phase11_cuda_kernels.py`
- Modify: `tests/test_phase6_commutation_grouping.py`

- [x] **Step 1: Write the exact candidate contract**

The candidate public surface is:

```python
DevicePauliSum.group_commuting_device(
    mode: str = "full",
    strategy: str = "largest_first",
    max_terms_for_graph: int = 50000,
) -> list[PauliSum]
```

Contract requirements:

```text
mode accepts only "full" in Campaign 10
strategy accepts only "largest_first"
return type is list[PauliSum], matching CPU group_commuting host ownership
group ordering and term ordering must match CPU group_commuting(mode="full", strategy="largest_first")
method is synchronous and default-stream compatible
method validates same device, moved-from state, qubit counts, and max_terms_for_graph before allocation
CPU-only builds do not expose DevicePauliSum, matching existing CUDA behavior
no public device-resident grouping metadata object is introduced in Campaign 10
```

- [x] **Step 2: Write failing tests**

Tests must cover:

```text
GPU grouping output matches CPU grouping labels and coefficients
invalid mode and strategy raise ValueError
max_terms_for_graph guard raises before allocation
CPU-only unavailable behavior remains unchanged
```

Run:

```bash
.venv/bin/python -m pytest tests/test_phase11_cuda_kernels.py tests/test_phase6_commutation_grouping.py -q
```

Expected before implementation: missing attribute or contract failure.

- [x] **Step 3: Accept or reject the API before implementation**

Accept only if all are true:

```text
contract above is accepted in docs/plans/cuda_grouping_public_api_campaign10_contract.md
implementation can reuse existing CPU grouping ownership semantics
H100 or non-H100 benchmark evidence shows a useful retained boundary compared with CPU grouping on at least one medium or large row
no new stream/lifetime public surface is needed
```

If any condition fails, record `rejected_with_evidence` and do not expose
`group_commuting_device`.

- [x] **Step 4: Implement only if accepted**

If accepted, implement the exact API. It must use CUDA to compute the full
commutation graph or dense matrix and then produce host-owned `PauliSum`
groups with CPU-equivalent ordering.

Run:

```bash
.venv/bin/python -m pytest tests/test_phase11_cuda_kernels.py::test_cuda_group_commuting_device_matches_cpu_full tests/test_phase6_commutation_grouping.py -q
```

- [x] **Step 5: Commit grouping decision or implementation**

```bash
git add docs/plans/cuda_grouping_public_api_campaign10_contract.md tests/test_phase11_cuda_kernels.py tests/test_phase6_commutation_grouping.py include/fastpauli/device_pauli_sum.hpp bindings/python/pauli_sum_py.cpp src/cuda/commutation_cuda.cu src/grouping.cpp
git commit -m "docs cuda campaign10 grouping api decision"
```

If implementation is accepted and code lands, use:

```bash
git commit -m "feat cuda public device grouping"
```

## Task 5: Stream And CUDA Graph Reprobe

**Files:**
- Create: `docs/plans/cuda_stream_graph_campaign10_decision.md`
- Modify if accepted: `benchmarks/bench_cuda_scaling.py`
- Modify if accepted: `src/cuda/workspace.cu`
- Modify if accepted: `src/cuda/workspace.cuh`
- Modify if accepted: `bindings/python/pauli_sum_py.cpp`

- [x] **Step 1: Capture new launch-overhead evidence**

Use the latest H100 reference evidence and run Nsight Systems for retained
compact consumers on each available non-H100 host:

```bash
nsys profile --force-overwrite=true --stats=true --trace=cuda,nvtx,osrt \
  --output "docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29/profiler/nsys_campaign10_compact_consumers_${FASTPAULI_CAMPAIGN10_GPU_ROLE}" \
  .venv/bin/python benchmarks/bench_cuda_scaling.py \
    --profile campaign10-stream-graph-reprobe \
    --repeat 3 \
    --warmup 1 \
    --json
```

- [x] **Step 2: Apply acceptance thresholds**

Accept a private CUDA Graph replay probe only if:

```text
launch/API overhead is at least 15% of a retained compact consumer on one measured host
graph replay improves that retained consumer by at least 10% on one host or at least 5% on two hosts
workspace addresses are stable and no pageable host allocation is captured
Python exceptions surface before returning from public synchronous methods
```

Public stream/event/graph APIs remain rejected in Campaign 10 unless a separate
API stability contract is written and accepted first.

- [x] **Step 3: Implement or reject**

If accepted, implement only a private benchmark-only graph replay hook and label
all rows `private_benchmark_only`. If rejected, write the exact profiler
evidence and threshold comparison into
`docs/plans/cuda_stream_graph_campaign10_decision.md`.

- [x] **Step 4: Commit stream/graph decision**

```bash
git add docs/plans/cuda_stream_graph_campaign10_decision.md benchmarks/bench_cuda_scaling.py src/cuda/workspace.cu src/cuda/workspace.cuh bindings/python/pauli_sum_py.cpp docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29
git commit -m "docs cuda campaign10 stream graph decision"
```

## Task 6: CSR Scatter Reprobe

**Files:**
- Create: `docs/plans/cuda_csr_scatter_campaign10_decision.md`
- Modify if accepted: `benchmarks/bench_cuda_scaling.py`
- Modify if accepted: `src/cuda/device_commutation_matrix.cu`
- Modify if accepted: `src/cuda/device_commutation_matrix.cuh`

- [x] **Step 1: Determine whether a retained consumer needs full CSR**

Check Campaign 10 grouping and stream/graph decisions. CSR scatter may reopen
only if a retained Campaign 10 consumer exports or internally consumes full CSR
edge lists.

- [x] **Step 2: Run the CSR scatter reprobe profile**

Run:

```bash
.venv/bin/python benchmarks/bench_cuda_scaling.py \
  --profile campaign10-csr-scatter-reprobe \
  --repeat 7 \
  --warmup 2 \
  --json \
  --output "docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29/raw/csr_scatter_reprobe_${FASTPAULI_CAMPAIGN10_GPU_ROLE}.json"
```

If privileged Nsight Compute is available, capture:

```bash
ncu --target-processes all --set detailed \
  --section SpeedOfLight --section MemoryWorkloadAnalysis --section LaunchStats \
  --section Occupancy --section SchedulerStats --section WarpStateStats \
  --force-overwrite \
  --export "docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29/profiler/ncu_csr_scatter_reprobe_${FASTPAULI_CAMPAIGN10_GPU_ROLE}" \
  .venv/bin/python benchmarks/bench_cuda_scaling.py \
    --profile campaign10-csr-scatter-reprobe \
    --repeat 1 \
    --warmup 0 \
    --json
```

- [x] **Step 3: Apply materiality threshold**

Implement CSR scatter tuning only if:

```text
a retained consumer requires full CSR edge lists
CSR scatter is visible as a material bottleneck in profiler evidence
projected improvement is at least 10% on one retained high-scale row or at least 5% on a broad landscape row
deterministic edge ordering and correctness are preserved
```

Otherwise record `rejected_with_evidence`.

- [x] **Step 4: Commit CSR decision**

```bash
git add docs/plans/cuda_csr_scatter_campaign10_decision.md benchmarks/bench_cuda_scaling.py src/cuda/device_commutation_matrix.cu src/cuda/device_commutation_matrix.cuh docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29
git commit -m "docs cuda campaign10 csr scatter decision"
```

## Task 7: Final Report, README Landscape, And Docs

**Files:**
- Create: `docs/benchmarks/reports/cuda_cross_architecture_campaign10_2026-04-29.md`
- Create: `docs/benchmarks/plots/cuda_campaign10_cross_architecture.svg`
- Create: `docs/benchmarks/plots/cuda_campaign10_dlpack_consumers.svg`
- Create: `docs/benchmarks/plots/cuda_campaign10_headroom_status.svg`
- Create: `docs/benchmarks/plots/cuda_campaign10_performance_landscape.svg`
- Modify: `README.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/plans/cuda_deep_optimization_plan.md`
- Modify: `docs/architecture/hardware_targets_and_testing.md`
- Modify: `docs/user/performance.md`
- Modify: `AGENTS.md`

- [x] **Step 1: Generate final assets**

Run:

```bash
.venv/bin/python scripts/render_cuda_campaign10_assets.py \
  --data-dir docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29 \
  --plot-dir docs/benchmarks/plots
```

Expected:

```text
all five Campaign 9 headroom items present
no deferred statuses
README landscape includes CPU, H100, A100, RTX, CUDA transfer-inclusive, CUDA device-resident, compact consumers, DLPack consumers, and external baselines where available
profiler-only rows excluded from user-facing performance landscape
```

- [x] **Step 2: Write final report**

The report must include:

```text
hardware table for H100, A100, and RTX-class hosts
source-build validation matrix
benchmark comparison matrix
DLPack CuPy versus PyTorch CUDA consumer evidence
public grouping API accept/reject outcome
stream/CUDA Graph accept/reject outcome
CSR scatter accept/reject outcome
remaining headroom section containing only new future work, not unresolved Campaign 9 items
```

- [x] **Step 3: Refresh source-of-truth docs**

Update README, roadmap, CUDA deep optimization plan, hardware targets, and user
performance guide so the latest CUDA source-of-truth points to Campaign 10.

- [x] **Step 4: Commit report and docs**

```bash
git add README.md AGENTS.md docs/roadmap.md docs/plans/cuda_deep_optimization_plan.md docs/architecture/hardware_targets_and_testing.md docs/user/performance.md docs/benchmarks/reports/cuda_cross_architecture_campaign10_2026-04-29.md docs/benchmarks/plots/cuda_campaign10_*.svg docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29
git commit -m "docs cuda campaign10 cross architecture report"
```

## Task 8: Validation, Review, Merge, Push, And Cleanup

**Files:**
- No new files unless review fixes are required.

- [x] **Step 1: Validate locally**

Run:

```bash
.venv/bin/python scripts/validate.py
git diff --check
```

Expected:

```text
repo validation passes
whitespace check passes
```

- [x] **Step 2: Validate on every successful CUDA host**

Run the CUDA validation command again on every host whose final status is
`passed` or `implemented`:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES="$FASTPAULI_CAMPAIGN10_ARCH" .venv/bin/python scripts/validate.py
```

- [x] **Step 3: Request independent review**

The review request must include:

```text
base SHA and head SHA
list of accepted, implemented, rejected, and blocked items
validation evidence for local, A100, RTX, and H100 runs
exact known residual risks
```

Resolve all blocking findings before merge.

- [x] **Step 4: Merge and push**

Run:

```bash
git switch main
git pull --ff-only
git merge --ff-only codex/cuda-campaign10-headroom
.venv/bin/python scripts/validate.py
git diff --check
git push origin main
```

- [x] **Step 5: Confirm CI and clean up**

Run:

```bash
gh run list --branch main --limit 5 --json databaseId,headSha,status,conclusion,workflowName,displayTitle,url,createdAt
FASTPAULI_CAMPAIGN10_RUN_ID=$(gh run list --branch main --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$FASTPAULI_CAMPAIGN10_RUN_ID" --exit-status
git branch -d codex/cuda-campaign10-headroom
git push origin --delete codex/cuda-campaign10-headroom
```

Do not call Campaign 10 complete until CI is green or a real CI-service blocker
is recorded with the exact run URL and status.

## Completion Definition

Campaign 10 is complete only when:

```text
A100 portability either passes or records a concrete external/toolchain blocker
RTX-class portability either passes or records a concrete external/toolchain blocker
PyTorch CUDA DLPack either passes or records a concrete dependency blocker
public grouping API is implemented or rejected with exact contract/performance evidence
stream/CUDA Graph work is implemented privately or rejected with profiler evidence
CSR scatter is implemented or rejected with retained-consumer evidence
README performance landscape shows the cross-architecture comparison
report and raw evidence are checked in
local validation passes
CUDA validation passes on every passing CUDA host
independent review has no blocking findings
main is pushed and CI is green
feature branch is deleted locally and remotely
```

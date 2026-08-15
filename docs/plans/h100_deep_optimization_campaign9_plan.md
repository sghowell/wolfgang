# CUDA Deferred Headroom Campaign 9 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve every deferred or blocked Campaign 8 CUDA headroom item with measured evidence, accepted public contracts, implemented code where accepted, or an explicit rejection/blocker that leaves no ambiguous deferred surface.

**Architecture:** Campaign 9 is a closure campaign for the Campaign 8 remaining-headroom list. It does not start by adding public APIs; it first turns each deferred surface into a concrete contract, then implements and benchmarks only the surfaces whose ownership, lifetime, synchronization, error, documentation, and performance contracts are accepted. Final status may be `accepted`, `implemented`, `rejected_with_evidence`, `passed`, `failed`, or `blocked_external`; `deferred` is not an allowed Campaign 9 closeout state.

**Execution status:** Complete on H100. The final report is
`docs/benchmarks/reports/cuda_deferred_headroom_campaign9_2026-04-29.md`.
Campaign 9 closed every Campaign 8 deferred or blocked item with a
non-deferred status and retained only the accepted public surfaces documented
there.

**Tech Stack:** C++20, CUDA C++ 12.x, nanobind, NumPy, optional CuPy/PyTorch for CUDA Array Interface and DLPack consumer checks, optional DLPack headers, pytest, `bench_cuda_scaling.py`, `bench_cuda_kernels.py`, `bench_competitive_baselines.py`, Nsight Systems, privileged Nsight Compute, Compute Sanitizer, H100 source builds with `FASTPAULI_CUDA_ARCHITECTURES=90`, and one named non-H100 NVIDIA source-build validation host.

---

## Status

Status: planned.

Campaign 9 starts from:

```text
report: docs/benchmarks/reports/cuda_deep_optimization_h100_campaign8_2026-04-29.md
plan: docs/plans/h100_deep_optimization_campaign8_plan.md
summary: docs/benchmarks/data/cuda_deep_optimization_h100_campaign8_2026-04-29/summary.json
```

Campaign 8 remaining headroom to close:

```text
1. Run Campaign 8 portability on one named non-H100 NVIDIA host.
2. Capture privileged Nsight Compute counters for retained compact graph/grouping consumers.
3. Promote a public fused grouping API only after exact semantics and documentation are accepted.
4. Revisit DLPack only with a complete ownership, stream, and lifetime test plan.
5. Consider stream or CUDA Graph replay only after enqueue, synchronization, workspace, and error semantics are fully specified.
6. Reopen CSR scatter tuning only if a retained consumer again needs full CSR scatter and NCU proves it is material.
```

Campaign 9 completion requires every item above to have a final non-deferred
outcome. If a public surface is rejected, the rejection must include exact
contract or evidence reasons and must preserve the current public CUDA API
invariants.

## Source Inputs

Read these before editing code or benchmark logic:

```text
README.md
docs/roadmap.md
docs/plans/cuda_deep_optimization_plan.md
docs/plans/h100_deep_optimization_campaign8_plan.md
docs/plans/cuda_fused_grouping_public_api_campaign8_review.md
docs/plans/cuda_dlpack_interop_campaign8_review.md
docs/plans/cuda_graphs_stream_campaign8_decision.md
docs/benchmarks/reports/cuda_deep_optimization_h100_campaign8_2026-04-29.md
docs/benchmarks/reports/cuda_portability_campaign8_non_h100_nvidia_2026-04-29.md
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
scripts/render_cuda_campaign8_assets.py
tests/test_phase11_cuda_kernels.py
tests/test_cuda_scaling_benchmark.py
tests/test_cuda_deep_report_assets.py
```

External primary references for the accepted contracts:

```text
Nsight Compute performance-counter permissions:
https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-nsightcompute

DLPack Python specification:
https://dmlc.github.io/dlpack/latest/python_spec.html

CUDA streams and CUDA Graph stream-capture rules:
https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html
```

## Scope

In scope:

```text
Campaign 9 decision contracts for every Campaign 8 deferred or blocked item
privileged Nsight Compute counters for retained Campaign 8 compact graph/grouping consumers
one named non-H100 NVIDIA source-build portability run
public fused grouping API implementation only if its exact contract is accepted in writing first
DLPack implementation only if its exact PyCapsule, stream, lifetime, dtype, shape, and mutability contract is accepted in writing first
private or public stream/CUDA Graph implementation only if enqueue, synchronization, workspace, capture, and error semantics are accepted in writing first
CSR scatter tuning only if a retained Campaign 9 consumer needs full CSR scatter and Nsight Compute proves scatter materially affects end-to-end time
new Campaign 9 benchmark profiles, raw JSON, metadata, plots, checked report, README broad performance landscape, and roadmap updates
CUDA correctness tests, CPU-only unavailable tests, sanitizer evidence, profiler evidence, and independent agent review
```

Out of scope:

```text
CUDA wheel release claims
raw device pointer public APIs
multi-GPU public behavior
HIP/AMD, Metal/MPS, Apple GPU, or non-NVIDIA backend implementation
public async/event/stream objects without accepted API-stability text
public DLPack export for mutable FastPauli-owned buffers without single-consumer ownership tests
full CSR edge-list export as the default high-scale graph/grouping boundary
raw PTX or inline PTX unless Nsight Compute identifies a compiler-codegen bottleneck that CUDA C++ cannot express
```

## File Structure

Planned files for the execution slice:

```text
docs/plans/cuda_fused_grouping_public_api_campaign9_contract.md
  Final Campaign 9 accept/reject contract for public fused grouping. It must name the public Python/C++ symbols if accepted, or state the exact reason no symbol is exposed.

docs/plans/cuda_dlpack_interop_campaign9_contract.md
  Final Campaign 9 accept/reject contract for DLPack producer behavior, PyCapsule ownership, stream synchronization, same-device checks, dtype/shape rules, and consumer tests.

docs/plans/cuda_stream_graph_campaign9_contract.md
  Final Campaign 9 accept/reject contract for stream-aware execution and CUDA Graph replay. It must distinguish public API, private benchmark-only probes, and rejection.

docs/plans/cuda_csr_scatter_campaign9_decision.md
  Final Campaign 9 CSR scatter reopen decision using privileged Nsight Compute evidence.

docs/benchmarks/data/cuda_deferred_headroom_campaign9_2026-04-29/
  Raw H100 benchmark JSON, profiler exports, sanitizer logs, validation logs, environment metadata, and summary JSON.

docs/benchmarks/reports/cuda_deferred_headroom_campaign9_2026-04-29.md
  Final Campaign 9 report with one status subsection per deferred Campaign 8 item.

docs/benchmarks/data/cuda_portability_campaign9_non_h100_nvidia_2026-04-29/
docs/benchmarks/reports/cuda_portability_campaign9_non_h100_nvidia_2026-04-29.md
  Named non-H100 source-build portability evidence.

scripts/render_cuda_campaign9_assets.py
  Campaign 9 summary and plot renderer. It may share helpers with the Campaign 8 renderer only if tests keep Campaign 8 outputs stable.

benchmarks/bench_cuda_scaling.py
benchmarks/bench_cuda_kernels.py
benchmarks/bench_competitive_baselines.py
  Campaign 9 profiles, status schema, timing labels, and optional framework baselines.

bindings/python/pauli_sum_py.cpp
include/fastpauli/device_commutation_matrix.hpp
include/fastpauli/device_pauli_sum.hpp
src/cuda/device_commutation_matrix.cu
src/cuda/device_commutation_matrix.cuh
src/cuda/commutation_cuda.cu
src/cuda/workspace.cu
src/cuda/workspace.cuh
  Code changes only for accepted public or private benchmark surfaces.

tests/test_phase11_cuda_kernels.py
tests/test_cuda_scaling_benchmark.py
tests/test_cuda_deep_report_assets.py
  CUDA correctness tests, CPU-only unavailable rows, benchmark schema tests, and renderer freshness checks.

README.md
docs/roadmap.md
docs/plans/cuda_deep_optimization_plan.md
docs/architecture/cuda_backend.md
docs/architecture/api_stability.md
docs/user/performance.md
  Source-of-truth status and performance documentation updates.
```

## Required Campaign 9 Status Schema

Every Campaign 9 raw row must carry:

```text
campaign: "cuda_deferred_headroom_campaign9"
mode: one of "privileged_ncu", "non_h100_portability", "public_grouping_api", "dlpack_interop", "stream_graph", "csr_scatter_reopen"
boundary: one of "device_resident", "compact_host_copy", "transfer_inclusive", "framework_consumer", "private_benchmark_only", "public_api", "profiler_only"
campaign8_headroom_item: integer from 1 through 6
final_status: one of "accepted", "implemented", "rejected_with_evidence", "passed", "failed", "blocked_external", "not_applicable"
deferred_status_allowed: false
decision_doc: repo-relative path to the contract or decision document
correctness_digest: stable digest for deterministic result rows or empty string for profiler-only rows
unavailable_reason: empty string when available; exact reason otherwise
git_revision: full revision used for the run
cuda_driver: driver version
cuda_runtime: CUDA runtime version
cuda_toolkit: toolkit version
compiled_architectures: semicolon-separated architecture list
gpu_name: exact GPU name
gpu_compute_capability: major.minor compute capability
```

Renderer tests must fail if a final Campaign 9 summary contains
`final_status: "deferred"` or omits any Campaign 8 remaining-headroom item.

## Task 1: Decision Contracts And Benchmark Schema

**Files:**
- Create: `docs/plans/cuda_fused_grouping_public_api_campaign9_contract.md`
- Create: `docs/plans/cuda_dlpack_interop_campaign9_contract.md`
- Create: `docs/plans/cuda_stream_graph_campaign9_contract.md`
- Create: `docs/plans/cuda_csr_scatter_campaign9_decision.md`
- Modify: `benchmarks/bench_cuda_kernels.py`
- Modify: `benchmarks/bench_cuda_scaling.py`
- Modify: `tests/test_cuda_scaling_benchmark.py`

- [ ] Write the Campaign 9 fused grouping contract.

Required initial decision text:

```text
Status: pending Campaign 9 evidence; final status cannot remain deferred.
Candidate public Python symbols:
  DeviceCommutationMatrix.conflict_degrees(axis=None|0|1, *, top_k=None)
  DeviceCommutationMatrix.top_conflicts(axis=1, *, k=8)
  DevicePauliSum.group_commuting_device(mode="full", strategy="largest_first", max_terms_for_graph=50000)
Accepted only if:
  return shape, dtype, ordering, ownership, synchronization, CPU-only errors, moved-from behavior, allocation limits, docstrings, user docs, and CPU correctness oracle are specified before implementation.
Default closeout if not accepted:
  rejected_with_evidence, with no installed public C++ declarations or fastpauli.__init__ exports.
```

- [ ] Write the Campaign 9 DLPack contract.

Required initial decision text:

```text
Status: pending Campaign 9 evidence; final status cannot remain deferred.
Candidate scope:
  read-only DLPack producer for DeviceCommutationMatrix dense uint8 buffer
Rejected by default for:
  mutable exports, cross-device copies, raw pointer exports, and compact grouping metadata whose lifetime is not tied to a stable owning object.
Accepted only if:
  __dlpack__(stream=None|1|2|>2, max_version=positive tuple, copy=None|False)
  __dlpack_device__()
  single-consumer capsule behavior
  used_dltensor rename/deleter behavior
  same-device enforcement
  stream wait/event behavior
  dtype "|u1", row-major compact strides, shape (rows, cols)
  owner lifetime and moved-from behavior
  CuPy and PyTorch consumer tests
  CPU-only BufferError or RuntimeError behavior
are all specified and tested.
```

- [ ] Write the Campaign 9 stream/CUDA Graph contract.

Required initial decision text:

```text
Status: pending Campaign 9 evidence; final status cannot remain deferred.
Public baseline:
  existing public CUDA methods remain synchronous and default-stream compatible unless this contract explicitly accepts a new API.
Private probe candidate:
  replay the retained compact graph/grouping consumer as a benchmark-only CUDA Graph with fixed shapes and stable workspace addresses.
Accepted only if:
  enqueue timing, event-elapsed timing, host synchronization, workspace lifetime, graph-capture safety, graph-update behavior, Python exception timing, and CPU-only behavior are specified.
Rejected if:
  retained workloads are dominated by kernels rather than launch overhead, workspace allocation is not graph-safe, or capture requires changing public synchronous semantics.
```

- [ ] Write the CSR scatter reopen decision.

Required initial decision text:

```text
Status: pending privileged Nsight Compute evidence; final status cannot remain deferred.
Reopen condition:
  a retained Campaign 9 consumer exports or internally consumes full CSR edge lists and Nsight Compute shows scatter kernels materially affect end-to-end retained-consumer time.
Reject condition:
  retained consumers remain compact and avoid full CSR scatter, or scatter is not a top profiler bottleneck.
Material threshold:
  scatter tuning is worth implementation only if projected improvement is at least 10% on one retained high-scale row or at least 5% on the broad landscape row it affects.
```

- [ ] Add Campaign 9 schema constants and CPU-only unavailable row tests.

Run:

```bash
python -m pytest tests/test_cuda_scaling_benchmark.py -q
```

Expected before implementation: tests fail because Campaign 9 schema/profile
entries do not exist. Expected after implementation: tests pass and prove every
Campaign 9 row has `deferred_status_allowed: false`.

- [ ] Commit the contracts and schema.

Commit message:

```bash
git commit -m "plan cuda campaign9 deferred headroom"
```

## Task 2: Privileged Nsight Compute Evidence On H100

**Files:**
- Modify: `benchmarks/bench_cuda_scaling.py`
- Modify: `scripts/render_cuda_campaign9_assets.py`
- Create: `docs/benchmarks/data/cuda_deferred_headroom_campaign9_2026-04-29/profiler/`
- Create: `docs/benchmarks/data/cuda_deferred_headroom_campaign9_2026-04-29/raw/`
- Create: `docs/benchmarks/data/cuda_deferred_headroom_campaign9_2026-04-29/metadata/`

- [ ] Verify the H100 host and source build.

Run on the H100 host:

```bash
cd ~/FastPauli-campaign9
git rev-parse HEAD
nvidia-smi
nvcc --version
python -m pytest tests/test_phase11_cuda_kernels.py -q
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 python scripts/validate.py
```

Expected: CUDA validation passes and the metadata records H100 PCIe, compute
capability 9.0, CUDA driver/runtime/toolkit, compiler, Python version, and git
revision.

- [ ] Capture privileged Nsight Compute counters.

Use root or `CAP_SYS_ADMIN` for the profiled target when the driver restricts
hardware performance counters.

Run:

```bash
sudo -E ncu --set full --target-processes all --force-overwrite \
  --export docs/benchmarks/data/cuda_deferred_headroom_campaign9_2026-04-29/profiler/ncu_campaign9_compact_consumers \
  python benchmarks/bench_cuda_scaling.py \
    --profile campaign9-privileged-ncu \
    --repeat 5 \
    --warmup 2 \
    --json \
    --output docs/benchmarks/data/cuda_deferred_headroom_campaign9_2026-04-29/raw/privileged_ncu_compact_consumers.json
```

Expected: no `ERR_NVGPUCTRPERM`. If `ERR_NVGPUCTRPERM` remains, record
`final_status: "blocked_external"` with the exact driver permission state and
the failed command; do not mark this item deferred.

- [ ] Restore evidence-file ownership after privileged profiling.

If `sudo -E ncu` writes root-owned files under the repo checkout, restore
ownership before rendering, staging, or deleting artifacts:

```bash
sudo chown -R "$(id -u):$(id -g)" \
  docs/benchmarks/data/cuda_deferred_headroom_campaign9_2026-04-29
```

Expected: the normal SSH user can read, render, stage, and archive profiler
artifacts without later permission prompts or root-owned worktree files.

- [ ] Export NCU details to CSV.

Run:

```bash
ncu --import docs/benchmarks/data/cuda_deferred_headroom_campaign9_2026-04-29/profiler/ncu_campaign9_compact_consumers.ncu-rep \
  --csv \
  --page details \
  > docs/benchmarks/data/cuda_deferred_headroom_campaign9_2026-04-29/profiler/ncu_campaign9_compact_consumers_details.csv
```

Expected: CSV includes retained compact graph/grouping kernels, memory
throughput, occupancy, achieved bandwidth, launch statistics, and any
source-level stall evidence available from the selected sections.

- [ ] Decide whether kernel tuning remains justified.

Required decision rule:

```text
If commutation fill remains dominant and compact consumers are below the material threshold, reject deeper compact-consumer tuning for Campaign 9.
If count/grouping kernels dominate a retained row and a local kernel change can improve end-to-end time without changing semantics, implement the smallest isolated kernel optimization and rerun validation.
If CSR scatter dominates only the full CSR baseline and not a retained consumer, keep CSR scatter rejected_no_consumer.
```

- [ ] Commit profiler evidence.

Commit message:

```bash
git commit -m "bench cuda campaign9 privileged ncu evidence"
```

## Task 3: Non-H100 NVIDIA Portability

**Files:**
- Modify: `benchmarks/bench_cuda_scaling.py`
- Create: `docs/benchmarks/data/cuda_portability_campaign9_non_h100_nvidia_2026-04-29/`
- Create: `docs/benchmarks/reports/cuda_portability_campaign9_non_h100_nvidia_2026-04-29.md`
- Modify: `docs/architecture/hardware_targets_and_testing.md`

- [ ] Select exactly one named non-H100 NVIDIA host before running commands.

Valid first targets:

```text
a100_sm80
rtx6000ada_sm89
l4_sm89
a10_sm86
```

Record this metadata before validation:

```text
host_label
gpu_name
compute_capability
driver_version
runtime_version
toolkit_version
compiled_architectures
git_revision
```

- [ ] Run source-build validation on the non-H100 host.

Use the matching architecture for the selected host:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=<sm_major><sm_minor> python scripts/validate.py
```

Expected: CUDA validation passes. If the source build fails, record
`final_status: "failed"` with exact compiler/runtime failure and do not broaden
non-H100 claims.

- [ ] Run the retained-consumer portability profile.

Run:

```bash
python benchmarks/bench_cuda_scaling.py \
  --profile campaign9-non-h100-portability \
  --repeat 5 \
  --warmup 2 \
  --json \
  --output docs/benchmarks/data/cuda_portability_campaign9_non_h100_nvidia_2026-04-29/raw/portability.json
```

Expected: compact graph/grouping correctness digests match H100 semantic
expectations for deterministic datasets, and performance rows are labeled as
non-H100 source-build evidence rather than H100 evidence.

- [ ] Write the portability report.

The report must state one of:

```text
passed: named non-H100 source build validates retained Campaign 8/Campaign 9 boundaries
failed: named non-H100 source build or retained-consumer benchmark failed with exact logs
blocked_external: no named non-H100 host was actually available after a concrete provisioning attempt
```

`blocked_external` is allowed only with a provider, instance type, date, and
failed provisioning or access evidence.

- [ ] Commit portability evidence.

Commit message:

```bash
git commit -m "bench cuda campaign9 non-h100 portability"
```

## Task 4: Public Fused Grouping API Decision And Implementation

**Files:**
- Modify: `docs/plans/cuda_fused_grouping_public_api_campaign9_contract.md`
- Modify only if accepted: `include/fastpauli/device_commutation_matrix.hpp`
- Modify only if accepted: `src/cuda/device_commutation_matrix.cu`
- Modify only if accepted: `src/cuda/device_commutation_matrix.cuh`
- Modify only if accepted: `bindings/python/pauli_sum_py.cpp`
- Modify only if accepted: `docs/architecture/api_stability.md`
- Modify: `tests/test_phase11_cuda_kernels.py`
- Modify: `docs/user/performance.md`

- [ ] Choose one true fused grouping surface to accept or reject.

The Campaign 8 headroom item is a public fused grouping API, not merely a
conflict-count summary. Campaign 9 must therefore give a final status to a true
grouping-returning surface:

```text
DevicePauliSum.group_commuting_device(mode="full", strategy="largest_first", max_terms_for_graph=50000)
```

A true grouping API may be accepted only if the contract specifies:

```text
exact return type and shape for groups
whether groups are returned as host PauliSum objects, host index arrays, or device-resident metadata
stable ordering relative to CPU group_commuting(mode="full", strategy="largest_first")
ownership and lifetime of every returned object
device and stream synchronization semantics
allocation guardrails and failure mode
CPU-only behavior and moved-from behavior
correctness oracle against CPU group_commuting
user docs and API-stability status
```

`DeviceCommutationMatrix.conflict_degrees(axis=None|0|1)` may still be
accepted as a separate compact summary API, but it does not by itself close the
public fused grouping headroom item. If Campaign 9 implements only
`conflict_degrees`, the true grouping API final status must still be
`rejected_with_evidence` with an exact explanation of why the stable group
return contract is not accepted.

Optional compact summary semantics if accepted:

```text
DeviceCommutationMatrix.conflict_degrees(axis=None|0|1)
axis=None returns a Python int equal to rows * cols - count_commuting()
axis=1 returns a uint64 NumPy array of length rows with per-row anti-commuting counts
axis=0 returns a uint64 NumPy array of length cols with per-column anti-commuting counts
ordering follows matrix row-major term order
the method is synchronous and default-stream compatible
CPU-only builds expose no DeviceCommutationMatrix object, matching existing CUDA behavior
moved-from matrices raise RuntimeError
bad axis values raise ValueError
```

- [ ] Write failing tests for the accepted surface or rejection status.

Accepted true grouping API test command:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 python -m pytest \
  tests/test_phase11_cuda_kernels.py::test_cuda_group_commuting_device_matches_cpu_full_grouping -q
```

Optional compact summary test command:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 python -m pytest \
  tests/test_phase11_cuda_kernels.py::test_cuda_device_commutation_matrix_conflict_degrees_matches_numpy -q
```

Rejection status test command:

```bash
python -m pytest tests/test_cuda_scaling_benchmark.py::test_cuda_campaign9_public_grouping_final_status_is_not_deferred -q
```

Expected before implementation: accepted true-grouping or compact-summary tests
fail because the methods are absent, or rejection-status tests fail until the
Campaign 9 summary records `rejected_with_evidence` for the true grouping API.

- [ ] Implement only the accepted surface.

If `group_commuting_device` is accepted, implement the documented group-return
format and prove it preserves CPU grouping semantics. If only
`conflict_degrees` is accepted, implement it as the anti-commuting complement
of existing compact count kernels, not by copying the dense matrix to host, and
keep the true grouping API rejected with evidence.

Required benchmark labels:

```text
group_commuting_device
cpu_group_commuting_reference
count_commuting_existing
conflict_degrees_axis_none
conflict_degrees_axis_0
conflict_degrees_axis_1
dense_to_host_plus_numpy_conflicts
private_grouping_consumer
```

- [ ] Update docs and API stability.

Required docs if accepted:

```text
README current status list
docs/user/performance.md CUDA commutation section
docs/architecture/api_stability.md public CUDA API list
docs/architecture/cuda_backend.md Campaign 9 status
```

If rejected, update only the contract, Campaign 9 report, roadmap, and CUDA
backend architecture with the exact rejection reason.

- [ ] Commit the public grouping decision.

Commit message:

```bash
git commit -m "decide cuda campaign9 grouping api"
```

## Task 5: DLPack Interop Decision And Implementation

**Files:**
- Modify: `docs/plans/cuda_dlpack_interop_campaign9_contract.md`
- Modify only if accepted: `bindings/python/pauli_sum_py.cpp`
- Modify only if accepted: `include/fastpauli/device_commutation_matrix.hpp`
- Modify: `benchmarks/bench_competitive_baselines.py`
- Modify: `tests/test_phase11_cuda_kernels.py`
- Modify: `tests/test_cuda_scaling_benchmark.py`
- Modify: `docs/architecture/cuda_backend.md`
- Modify only if public: `docs/architecture/api_stability.md`
- Modify only if public: `docs/user/performance.md`

- [ ] Accept only a read-only dense `DeviceCommutationMatrix` DLPack producer, or reject DLPack for Campaign 9.

Accepted public method names:

```text
DeviceCommutationMatrix.__dlpack__(stream=None, max_version=(1, 0), copy=None)
DeviceCommutationMatrix.__dlpack_device__()
```

Required `__dlpack_device__` return convention:

```text
(2, device_ordinal)
```

where `2` is `kDLCUDA`.

- [ ] Write ownership and consumer tests before implementation.

Required CUDA-gated tests:

```text
At least one real CUDA DLPack consumer must be installed and pass before DLPack can be accepted.
CuPy from_dlpack consumes the matrix without host copy when CuPy is the selected real consumer.
PyTorch from_dlpack consumes the matrix without host copy when torch with CUDA is the selected real consumer.
Both CuPy and PyTorch consumers are preferred; absence of one must be recorded with the package/version/provisioning reason.
Consuming the same capsule twice fails or is impossible through the standard method path.
The FastPauli owner remains alive for the consumer view lifetime.
Moved-from matrix export raises RuntimeError.
CPU-only unavailable behavior is explicit.
stream=0 is rejected because the Python Array API marks CUDA stream 0 ambiguous.
stream=None, stream=1, and stream=2 behavior is documented and tested.
copy=True raises BufferError unless an explicit copy implementation is added.
```

If no real CUDA DLPack consumer can be installed and validated on the H100 host,
Campaign 9 must record DLPack as `blocked_external` or
`rejected_with_evidence`; it may not record `accepted` or `implemented`.

- [ ] Implement DLPack only after tests define the accepted behavior.

Implementation requirements:

```text
capsule name starts as "dltensor" or "dltensor_versioned" according to max_version
consumer-owned capsule rename to "used_dltensor" or "used_dltensor_versioned" prevents double deleter calls
deleter holds enough state to keep the FastPauli owner alive until consumer destruction
DLTensor shape is (rows, cols), dtype is uint8, strides are null for compact row-major
same-device pointer and device ordinal are recorded
stream synchronization follows the accepted contract
```

- [ ] Benchmark DLPack consumers against CUDA Array Interface.

Run:

```bash
python benchmarks/bench_competitive_baselines.py \
  --profile cuda-interop \
  --repeat 7 \
  --warmup 2 \
  --json \
  --output docs/benchmarks/data/cuda_deferred_headroom_campaign9_2026-04-29/raw/dlpack_interop.json
```

Expected: rows include CuPy CUDA Array Interface, at least one passing real
DLPack CuPy or PyTorch consumer if DLPack is accepted, dense `to_host()`, and
retained compact consumers. Optional framework absence may be recorded without
failing broad comparison rows only when DLPack is not accepted; accepted DLPack
requires a passing real CUDA consumer.

- [ ] Commit the DLPack decision.

Commit message:

```bash
git commit -m "decide cuda campaign9 dlpack interop"
```

## Task 6: Stream And CUDA Graph Decision And Implementation

**Files:**
- Modify: `docs/plans/cuda_stream_graph_campaign9_contract.md`
- Modify only if accepted private probe: `src/cuda/workspace.cu`
- Modify only if accepted private probe: `src/cuda/workspace.cuh`
- Modify only if accepted private probe: `src/cuda/device_commutation_matrix.cu`
- Modify only if accepted private probe: `bindings/python/pauli_sum_py.cpp`
- Modify: `benchmarks/bench_cuda_scaling.py`
- Modify: `tests/test_phase11_cuda_kernels.py`
- Modify: `tests/test_cuda_scaling_benchmark.py`
- Modify: `docs/architecture/cuda_backend.md`

- [ ] Decide public API versus private benchmark-only scope.

Default Campaign 9 stance:

```text
public stream/event/graph API remains rejected unless the contract accepts exact object ownership and Python exception timing
private CUDA Graph replay probe may be accepted for fixed-shape compact graph/grouping consumers
```

- [ ] Write graph replay tests before implementation.

Required CUDA-gated tests for a private probe:

```text
first replay and second replay produce identical correctness_digest
workspace addresses remain stable across graph instantiation and replay
input shape changes reject graph reuse with a clear error
Python-visible report separates graph_build_seconds, graph_instantiate_seconds, graph_replay_seconds, and synchronous_end_to_end_seconds
public methods remain synchronous and default-stream compatible
```

- [ ] Implement a private fixed-shape graph replay probe only if accepted.

Required implementation constraints:

```text
no public stream handle argument
no public event object
no public CUDA graph handle
no capture of APIs that allocate pageable/pinned host memory inside capture
no capture using cudaStreamLegacy as the capture stream
all CUDA errors become Python exceptions before returning from the private hook
```

- [ ] Benchmark launch overhead and graph replay.

Run:

```bash
python benchmarks/bench_cuda_scaling.py \
  --profile campaign9-stream-graph \
  --repeat 20 \
  --warmup 5 \
  --json \
  --output docs/benchmarks/data/cuda_deferred_headroom_campaign9_2026-04-29/raw/stream_graph.json
```

Accept graph replay only if it improves a retained compact consumer by at least
5% end-to-end or by at least 10% on repeated small/medium fixed-shape rows
without changing public semantics. Otherwise record `rejected_with_evidence`.

- [ ] Commit the stream/graph decision.

Commit message:

```bash
git commit -m "decide cuda campaign9 stream graph path"
```

## Task 7: CSR Scatter Reopen Decision

**Files:**
- Modify: `docs/plans/cuda_csr_scatter_campaign9_decision.md`
- Modify only if reopened: `src/cuda/device_commutation_matrix.cu`
- Modify only if reopened: `src/cuda/device_commutation_matrix.cuh`
- Modify: `benchmarks/bench_cuda_scaling.py`
- Modify: `tests/test_phase11_cuda_kernels.py`
- Modify: `docs/benchmarks/reports/cuda_deferred_headroom_campaign9_2026-04-29.md`

- [ ] Determine whether any retained Campaign 9 consumer needs full CSR scatter.

Required logic:

```text
If public grouping, DLPack, and stream/graph decisions retain only compact consumers, keep CSR scatter rejected_no_consumer.
If a retained public or private consumer uses full CSR edge lists, run NCU-guided scatter A/B before deciding.
```

- [ ] Run scatter A/B only if the reopen condition is met.

Run:

```bash
python benchmarks/bench_cuda_scaling.py \
  --profile campaign9-csr-scatter-ab \
  --repeat 7 \
  --warmup 2 \
  --json \
  --output docs/benchmarks/data/cuda_deferred_headroom_campaign9_2026-04-29/raw/csr_scatter_ab.json
```

Expected: rows compare existing scatter, any tuned scatter variant, compact
consumer alternatives, and full dense host materialization with correctness
digests.

- [ ] Implement scatter tuning only if material.

Allowed implementation candidates:

```text
coalesced row-chunk scatter for dense high-conflict rows
CUB prefix-sum temporary-storage reuse if allocation appears in NCU as material
vectorized index writes only if alignment and bounds are proven
```

Rejected implementation candidates:

```text
raw PTX without SASS evidence
atomic-heavy scatter that changes deterministic ordering
scatter tuning that improves only an unretained full CSR baseline
```

- [ ] Commit the CSR decision.

Commit message:

```bash
git commit -m "decide cuda campaign9 csr scatter"
```

## Task 8: Report, Plots, README, And Roadmap

**Files:**
- Create: `scripts/render_cuda_campaign9_assets.py`
- Create: `docs/benchmarks/reports/cuda_deferred_headroom_campaign9_2026-04-29.md`
- Create: `docs/benchmarks/plots/cuda_campaign9_deferred_headroom_status.svg`
- Create: `docs/benchmarks/plots/cuda_campaign9_privileged_ncu.svg`
- Create: `docs/benchmarks/plots/cuda_campaign9_portability.svg`
- Create: `docs/benchmarks/plots/cuda_campaign9_performance_landscape.svg`
- Modify: `tests/test_cuda_deep_report_assets.py`
- Modify: `README.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/plans/cuda_deep_optimization_plan.md`
- Modify: `docs/user/performance.md`
- Modify: `docs/architecture/cuda_backend.md`

- [ ] Render Campaign 9 summary and plots.

Run:

```bash
python scripts/render_cuda_campaign9_assets.py \
  --data-dir docs/benchmarks/data/cuda_deferred_headroom_campaign9_2026-04-29
python -m pytest tests/test_cuda_deep_report_assets.py -q
```

Expected: renderer tests prove plots and summary are fresh and no Campaign 9
final status is `deferred`.

- [ ] Write the Campaign 9 report.

Required sections:

```text
Evidence
Hardware
Validation
Privileged Nsight Compute
Non-H100 Portability
Public Fused Grouping API Decision
DLPack Interop Decision
Stream And CUDA Graph Decision
CSR Scatter Decision
Performance Landscape
Remaining Headroom
```

The `Remaining Headroom` section must not list unresolved Campaign 8 deferred
items. It may list new follow-up work only when it is discovered by Campaign 9
evidence and is not a restatement of an unresolved Campaign 8 deferral.

- [ ] Refresh the README broad landscape.

README requirement:

```text
The README plot remains a broad across-the-board performance comparison and includes CPU scalar, captured optimized CPU selectors, CUDA transfer-inclusive rows, CUDA device-resident rows, compact consumers, accepted/rejected Campaign 9 surface rows, framework consumer rows where available, and semantically comparable external baselines.
```

- [ ] Commit reports and docs.

Commit message:

```bash
git commit -m "docs cuda campaign9 deferred headroom report"
```

## Task 9: Review, Merge, Push, And CI Closeout

**Files:**
- Read: `docs/quality/code_review.md`
- Read: `docs/benchmarks/reports/cuda_deferred_headroom_campaign9_2026-04-29.md`
- Modify only for fixes: files named by review findings

- [ ] Run final H100 validation.

Run on H100:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 python scripts/validate.py
compute-sanitizer --tool memcheck python -m pytest tests/test_phase11_cuda_kernels.py -q
compute-sanitizer --tool racecheck python -m pytest tests/test_phase11_cuda_kernels.py -q
compute-sanitizer --tool initcheck python -m pytest tests/test_phase11_cuda_kernels.py -q
compute-sanitizer --tool synccheck python -m pytest tests/test_phase11_cuda_kernels.py -q
```

Expected: CUDA validation passes and Compute Sanitizer reports zero memory,
race, init, and synchronization errors for retained Campaign 9 changes.

- [ ] Run local CPU validation.

Run locally:

```bash
python scripts/validate.py
git diff --check
```

Expected: validation and whitespace checks pass.

- [ ] Request independent agent-driven review.

Reviewer inputs:

```text
branch name
commit list
Campaign 9 goal and non-deferred closeout rule
decision docs for all six Campaign 8 remaining-headroom items
H100 validation and profiler evidence
non-H100 NVIDIA portability evidence
public API changes, if any
DLPack/stream/graph decisions
CSR scatter decision
README broad landscape update
local and H100 validation commands/results
```

- [ ] Resolve every blocking review finding.

Expected: all P0/P1 findings are fixed and revalidated. P2 findings are either
fixed or recorded with exact rationale and owner.

- [ ] Merge, validate, push, confirm CI, and cleanup.

Run:

```bash
git switch main
git pull --ff-only
git merge --ff-only codex/h100-campaign9-deferred-headroom
python scripts/validate.py
git diff --check
git push origin main
gh run list --branch main --limit 5
gh run watch "$(gh run list --branch main --limit 1 --json databaseId --jq '.[0].databaseId')"
gh run view "$(gh run list --branch main --limit 1 --json databaseId --jq '.[0].databaseId')" --json conclusion,status,url,headSha
git branch -d codex/h100-campaign9-deferred-headroom
```

Expected: merged `main` validates locally, pushed CI completes successfully,
and the local feature branch is deleted.

## Exhaustion Criteria

Campaign 9 is complete only when all of the following are true:

```text
every Campaign 8 remaining-headroom item has a final non-deferred Campaign 9 status
one named non-H100 NVIDIA portability run passes, fails, or is blocked_external with provisioning/access evidence
privileged Nsight Compute counters are captured or blocked_external with exact permission evidence
public fused grouping API is accepted and implemented with tests/docs or rejected_with_evidence
DLPack is accepted and implemented with ownership/stream/lifetime tests or rejected_with_evidence
stream/CUDA Graph replay is accepted and implemented as public or private benchmark-only with tests, or rejected_with_evidence
CSR scatter is reopened only if a retained consumer needs it and NCU proves it is material
no Campaign 9 summary or report field uses final_status: deferred
H100 CUDA validation passes on the final branch
Compute Sanitizer memcheck, racecheck, initcheck, and synccheck are clean for retained CUDA changes
raw JSON, metadata, profiler files, plots, and reports are checked in under Campaign 9 paths
README performance plot remains an across-the-board CPU/CUDA/external comparison
independent agent-driven review is completed and blocking findings are resolved
merged main is validated, pushed, CI is green, and the local feature branch is cleaned up
```

## Remaining-Headroom Mapping

```text
Run Campaign 8 portability on one named non-H100 NVIDIA host.
  Covered by Task 3.

Capture privileged Nsight Compute counters for retained compact graph/grouping consumers.
  Covered by Task 2.

Promote a public fused grouping API only after exact semantics and documentation are accepted.
  Covered by Tasks 1 and 4.

Revisit DLPack only with a complete ownership, stream, and lifetime test plan.
  Covered by Tasks 1 and 5.

Consider stream or CUDA Graph replay only after enqueue, synchronization, workspace, and error semantics are fully specified.
  Covered by Tasks 1 and 6.

Reopen CSR scatter tuning only if a retained consumer again needs full CSR scatter and NCU proves it is material.
  Covered by Tasks 2 and 7.
```

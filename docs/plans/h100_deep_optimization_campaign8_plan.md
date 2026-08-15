# H100 CUDA Device-Resident Consumer Campaign 8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the Campaign 7 remaining-headroom list into measured H100 work that keeps downstream graph and grouping consumers device-resident, decides whether any new public CUDA API is justified, and records portability evidence before broadening GPU claims.

**Architecture:** Campaign 8 starts from the Campaign 7 result that exporting full CSR edge lists is the wrong public boundary for large random anti-commutation graphs. The primary design is a private device-resident graph/grouping consumer pipeline that returns compact host evidence only when a consumer truly needs it, while existing public CUDA APIs remain synchronous, default-stream, and compatible. Public fused grouping, DLPack, stream-aware, or CUDA Graph surfaces are allowed only after written decision artifacts accept exact ownership, lifetime, ordering, synchronization, and failure semantics.

**Tech Stack:** C++20, CUDA C++ 12.x, CCCL/CUB where justified by profiles, nanobind, NumPy, CuPy, optional PyTorch for DLPack consumer checks, DLPack/PyCapsule only if accepted by review, pytest, `bench_cuda_scaling.py`, `bench_cuda_kernels.py`, `bench_competitive_baselines.py`, Nsight Systems, Nsight Compute, Compute Sanitizer, H100 source builds with `FASTPAULI_CUDA_ARCHITECTURES=90`, and at least one non-H100 NVIDIA source-build validation host before non-H100 claims.

---

## Status

Status: completed on H100 in
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign8_2026-04-29.md`.
Non-H100 NVIDIA portability is blocked by hardware availability and recorded in
`docs/benchmarks/reports/cuda_portability_campaign8_non_h100_nvidia_2026-04-29.md`.

Campaign 7 source-of-truth evidence:

```text
plan: docs/plans/h100_deep_optimization_campaign7_plan.md
report: docs/benchmarks/reports/cuda_deep_optimization_h100_campaign7_2026-04-29.md
data: docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29/
plot: docs/benchmarks/plots/cuda_h100_campaign7_performance_landscape.svg
private fused consumers: CSR graph export, conflict degrees, grouping summary
retained public consumer API: DeviceCommutationMatrix.count_commuting(axis=None|0|1)
retained public commutation matrix API: DevicePauliSum.commutes_with_device()
```

Campaign 7 remaining-headroom items covered by Campaign 8:

```text
1. Device-resident graph consumers that avoid exporting full CSR edge lists: retained as private benchmark-only compact consumers.
2. Public fused grouping API only after exact return semantics and ownership are accepted: deferred by review.
3. DLPack or framework interop for retained device outputs: DLPack deferred; CUDA Array Interface remains retained.
4. Non-H100 NVIDIA retained-consumer portability evidence: blocked by hardware availability.
5. CUDA Graphs or stream-aware execution only after a complete public contract: deferred by decision document.
6. Additional NCU-guided CSR scatter tuning only if a fully device-resident graph consumer needs it: rejected because retained consumers avoid full CSR scatter.
```

## Source Inputs

Read these files before implementation:

```text
README.md
docs/roadmap.md
docs/plans/cuda_deep_optimization_plan.md
docs/plans/h100_deep_optimization_campaign7_plan.md
docs/plans/cuda_fused_commutation_consumer_api_review.md
docs/plans/cuda_async_stream_campaign7_decision.md
docs/plans/cuda_bitpacked_commutation_campaign7_decision.md
docs/benchmarks/reports/cuda_deep_optimization_h100_campaign7_2026-04-29.md
docs/benchmarks/reports/cuda_portability_campaign7_non_h100_nvidia_2026-04-29.md
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
scripts/render_cuda_campaign7_assets.py
tests/test_phase11_cuda_kernels.py
tests/test_cuda_scaling_benchmark.py
tests/test_cuda_deep_report_assets.py
```

## Scope

In scope:

```text
written API and interop decisions before public CUDA surface changes
private device-resident graph/grouping consumers that avoid full CSR edge-list host export
compact host result paths for graph/grouping evidence only when needed by the consumer contract
same-boundary H100 A/B comparisons against Campaign 7 CSR export, Campaign 7 grouping summaries, dense to_host(), compact counts, CuPy consumers, CPU scalar, and optimized CPU selectors
DLPack or framework interop prototype only after ownership, deleter, stream, dtype, shape, mutability, and lifetime semantics are accepted
non-H100 NVIDIA validation and benchmark evidence for retained public and private benchmark consumer boundaries
CUDA Graph or stream-aware execution only after a complete public or private benchmark contract is accepted
NCU-guided CSR scatter tuning only when a device-resident graph consumer still uses scatter and Nsight Compute proves scatter remains material
Campaign 8 report, raw data, metadata, profiler evidence, generated plots, README broad landscape refresh, and roadmap updates when evidence supersedes Campaign 7
```

Out of scope unless this plan's decision gates explicitly accept the surface:

```text
public async Python methods
public external stream-handle arguments
public event classes
public CUDA Graph capture handles
public DLPack export for mutable FastPauli-owned buffers
public raw device pointer APIs
public CSR graph export of full edge lists as the primary high-scale graph API
CUDA wheel release claims
HIP/AMD, Metal/MPS, Apple GPU, or non-NVIDIA backend work
multi-GPU claims
raw PTX or inline PTX without Nsight and SASS evidence for a specific compiler-codegen limit
```

## File Structure

Planned files for the implementation slice:

```text
docs/plans/cuda_fused_grouping_public_api_campaign8_review.md
  Public fused grouping API contract, exact return semantics, ordering, ownership, synchronization, CPU-only behavior, and accept/reject decision.

docs/plans/cuda_dlpack_interop_campaign8_review.md
  DLPack/framework interop contract, PyCapsule ownership, deleter behavior, stream semantics, device checks, dtype/shape rules, mutability, and accept/reject decision.

docs/plans/cuda_graphs_stream_campaign8_decision.md
  Campaign 8 stream-aware and CUDA Graph decision, including capture safety, event/error propagation, object lifetime, and benchmark-only versus public scope.

docs/architecture/cuda_backend.md
  Updated Campaign 8 status for device-resident graph consumers, public grouping API gate, DLPack/interop, stream/graph deferral or acceptance, and scatter tuning.

docs/architecture/api_stability.md
  Updated only if a public fused grouping, DLPack, stream, or graph API is accepted.

include/fastpauli/device_commutation_matrix.hpp
  Public declarations only for accepted public methods. Private benchmark consumers stay out of installed public headers.

include/fastpauli/device_pauli_sum.hpp
  Public declarations only if a retained API needs DevicePauliSum-level construction or ownership semantics.

src/cuda/device_commutation_matrix.cu
src/cuda/device_commutation_matrix.cuh
  Device-resident graph/grouping consumer kernels, compact-output kernels, optional DLPack export support, and private launch helpers.

src/cuda/commutation_cuda.cu
  Integration only when device-resident graph/grouping consumers can safely share commutation fill data or workspace.

src/cuda/workspace.cu
src/cuda/workspace.cuh
  Reusable temporary storage for device-resident graph/grouping consumers, optional stream-aware workspace probes, and gated CSR scatter A/B variants.

bindings/python/pauli_sum_py.cpp
  Python bindings only for public surfaces accepted by Campaign 8 review artifacts and private benchmark hooks used by CUDA benchmark scripts.

benchmarks/bench_cuda_scaling.py
  Campaign 8 stress profiles for device-resident graph consumers, compact grouping, interop consumers, stream/graph probes, non-H100 runs, and scatter A/B rows.

benchmarks/bench_cuda_kernels.py
  Timing schema fields for Campaign 8 statuses and boundary labels.

benchmarks/bench_competitive_baselines.py
  Comparable CPU, CuPy, PyTorch, CUDA-Q, cuQuantum, OpenFermion, and Qiskit rows only where semantics match the measured workload.

scripts/render_cuda_campaign8_assets.py
  Generate Campaign 8 summary JSON and report plots from checked raw data.

tests/test_phase11_cuda_kernels.py
  CUDA correctness tests for any retained public Campaign 8 behavior and private benchmark-only device-resident consumer checks.

tests/test_cuda_scaling_benchmark.py
  Non-CUDA tests proving Campaign 8 benchmark profiles, schema fields, and unavailable-status rows are present without requiring a GPU.

tests/test_cuda_deep_report_assets.py
  Renderer freshness tests for Campaign 8 checked summary and plots.

docs/benchmarks/data/cuda_deep_optimization_h100_campaign8_2026-04-29/
docs/benchmarks/reports/cuda_deep_optimization_h100_campaign8_2026-04-29.md
docs/benchmarks/plots/cuda_h100_campaign8_*.svg
  Final H100 evidence bundle after execution.

docs/benchmarks/data/cuda_portability_campaign8_non_h100_nvidia_2026-04-29/
docs/benchmarks/reports/cuda_portability_campaign8_non_h100_nvidia_2026-04-29.md
  Non-H100 NVIDIA portability evidence bundle after execution.

README.md
docs/roadmap.md
docs/plans/cuda_deep_optimization_plan.md
docs/user/performance.md
  Source-of-truth links, next-slice status, portability wording, broad performance visual policy, and latest evidence references.
```

The non-H100 report must record a lowercase hardware identifier in metadata,
such as `a100_sm80`, `rtx6000ada_sm89`, `l4_sm89`, or `a10_sm86`, before the
first portability command is run.

## Required Benchmark Schema

Campaign 8 benchmark JSON rows must include these fields for every CUDA row:

```text
campaign: "h100_campaign8"
mode: one of "device_resident_graph", "device_grouping_consumer", "grouping_public_api", "dlpack_consumer", "stream_graph_probe", "csr_scatter_ab", "portability_check"
boundary: one of "device_resident", "compact_host_copy", "transfer_inclusive", "framework_consumer", "private_benchmark_only", "public_api"
device_resident_graph_status: one of "retained", "rejected", "unavailable", "not_applicable"
public_grouping_api_status: one of "accepted", "deferred", "rejected", "not_applicable"
dlpack_interop_status: one of "accepted", "accepted_private_probe", "deferred", "rejected", "unavailable", "not_applicable"
non_h100_portability_status: one of "passed", "blocked", "failed", "not_run"
stream_graph_status: one of "accepted_public", "accepted_private_probe", "deferred", "rejected", "not_applicable"
scatter_tuning_status: one of "retained", "rejected_not_dominant", "rejected_no_consumer", "deferred", "not_applicable"
timing_boundary: one of "kernel_only", "device_resident_consumer", "compact_materialization", "full_materialization", "transfer_inclusive"
correctness_digest: stable digest for the deterministic dataset and result convention
unavailable_reason: empty string when available; explicit reason otherwise
git_revision: full revision used for the run
cuda_driver: driver version
cuda_runtime: CUDA runtime version
cuda_toolkit: toolkit version
compiled_architectures: semicolon-separated architecture list
gpu_name: exact GPU name
gpu_compute_capability: major.minor compute capability
```

Renderer tests must fail if the Campaign 8 summary omits any of the six status
fields from the remaining-headroom list.

## Public API Decision Gate

Campaign 8 starts conservative:

```text
existing public CUDA APIs remain default-stream and synchronize-before-return
existing DeviceCommutationMatrix dense uint8 layout remains the public matrix representation
existing DeviceCommutationMatrix.count_commuting(axis=None|0|1) remains the only public compact consumer unless the Campaign 8 grouping review accepts another method
Campaign 7 CSR graph export, conflict-degree, and grouping-summary consumers remain private benchmark-only until an accepted Campaign 8 decision changes their status
DLPack export remains unavailable until the Campaign 8 DLPack review accepts exact ownership and lifetime semantics
stream-aware and CUDA Graph surfaces remain unavailable until the Campaign 8 stream/graph decision accepts exact semantics
```

Required decision artifacts:

```text
docs/plans/cuda_fused_grouping_public_api_campaign8_review.md
docs/plans/cuda_dlpack_interop_campaign8_review.md
docs/plans/cuda_graphs_stream_campaign8_decision.md
```

The public grouping review must accept or reject each candidate before code is
exposed outside benchmark helpers:

```text
compact grouping summary copied to host
device-resident grouping metadata exposed through CUDA Array Interface
DLPack-exported grouping metadata
full CSR anti-commutation graph export
device-resident graph consumer handle with no full edge-list host copy
```

If a public fused grouping API is retained, the review must specify:

```text
exact Python method name and C++ method name
exact return type, shape, dtype, and ownership
commuting or anti-commuting convention
stable ordering of rows, columns, groups, or conflict metadata
device and stream synchronization semantics
host copy size and transfer boundary
CPU-only error behavior
moved-from object behavior
memory allocation limit and failure mode
correctness oracle against a CPU reference
benchmark labels for fill, device-resident consumer, compact host copy, and full to_host()
API stability status and documentation requirements
```

No public method may be added if the review cannot define those fields.

## Private Benchmark Hook Contract

Unless a public API is accepted, Campaign 8 code should use one private hook:

```text
private Python hook: fastpauli._fastpauli_core._benchmark_cuda_device_resident_consumer
allowed callers: benchmarks/bench_cuda_scaling.py, benchmarks/bench_cuda_kernels.py, and CUDA-gated tests
visibility: never re-export from python/fastpauli/__init__.py and never document as user-facing API
CPU-only behavior: benchmark scripts receive an unavailable row with a specific unavailable_reason; CUDA-gated tests receive the existing CUDA rebuild-guidance RuntimeError
return shape: JSON-serializable dict with mode, rows, cols, timings, output_sizes, correctness_digest, statuses, and unavailable_reason
supported modes: device_resident_graph, device_grouping_consumer, dlpack_consumer, stream_graph_probe, csr_scatter_ab
```

The private hook may call C++ helpers in `src/cuda/device_commutation_matrix.*`
or a dedicated benchmark translation unit, but those helpers must remain out of
installed public headers unless the relevant Campaign 8 review accepts a public
API.

## Device-Resident Consumer Workloads

Campaign 8 must implement or explicitly reject the following workloads with evidence:

```text
1. Device-resident graph consumer without full CSR edge-list export
   Input: DeviceCommutationMatrix dense row-major uint8 flags.
   Edge convention: matrix value 0 is an anti-commuting edge.
   Output boundary: compact metadata copied to host only when needed; full CSR row_offsets/column_indices remain validation-only or rejected for high-scale public use.
   Correctness oracle: small deterministic CPU graph extraction and grouping comparison against matrix.to_host() with stable row-major ordering.

2. Device-resident grouping consumer
   Input: DeviceCommutationMatrix dense row-major uint8 flags.
   Consumer goal: produce grouping-compatible conflict metadata without copying O(edges) host output.
   Output boundary: compact per-row/per-group host summary or device-resident metadata handle, depending on accepted review.
   Correctness oracle: exact equality with the CPU grouping summary for deterministic small and default-size datasets.

3. DLPack or framework interop consumer
   Input: retained DeviceCommutationMatrix or retained device-resident grouping metadata.
   Consumer goal: prove a framework can consume FastPauli-owned device output without host materialization.
   Output boundary: DLPack capsule or existing CUDA Array Interface path, with explicit ownership and stream semantics.
   Correctness oracle: CuPy and, when installed, PyTorch digest equality against FastPauli compact host result.

4. Stream or CUDA Graph probe
   Input: repeated device-resident fill plus consumer workloads.
   Consumer goal: quantify launch overhead and synchronization cost only after the decision document defines allowed private or public semantics.
   Output boundary: benchmark-only timing rows unless public API review accepts more.
   Correctness oracle: identical digest to synchronous default-stream execution.

5. CSR scatter A/B
   Input: device-resident graph consumer only if it still needs scatter.
   Consumer goal: reduce kernel time for scatter without increasing host transfer or changing graph convention.
   Output boundary: benchmark-only A/B rows with retained/rejected decision.
   Correctness oracle: exact row_offsets and column_indices equality on small validation cases, and digest equality on stress cases.
```

## Task 0: Branch, Hardware, And Baseline Reproduction

**Files:**
- Read: `AGENTS.md`
- Read: `docs/plans/h100_deep_optimization_campaign8_plan.md`
- Read: `docs/benchmarks/reports/cuda_deep_optimization_h100_campaign7_2026-04-29.md`
- Modify later only if evidence changes: `docs/benchmarks/reports/cuda_deep_optimization_h100_campaign8_2026-04-29.md`

- [ ] Create the execution branch.

Run:

```bash
git switch main
git pull --ff-only
git switch -c codex/h100-campaign8-execution
```

Expected: branch is created from the latest `main`.

- [ ] Record the H100 environment before editing code.

Run on the H100 host:

```bash
nvidia-smi
nvcc --version
python -VV
python -m pip freeze
```

Expected: output records the GPU, driver, CUDA toolkit, Python, and dependency versions for the Campaign 8 metadata bundle.

- [ ] Reproduce the Campaign 7 retained workload on the current H100 host.

Run:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 python scripts/validate.py
python benchmarks/bench_cuda_scaling.py --profile fused-graph-stress --repeat 7 --warmup 2 --json
```

Expected: validation passes and the Campaign 7 fused graph/grouping rows are reproducible within normal H100 run-to-run noise. If the exact profile name has changed, add a Campaign 8-compatible alias before continuing.

- [ ] Save baseline raw JSON under the Campaign 8 data directory.

Expected path:

```text
docs/benchmarks/data/cuda_deep_optimization_h100_campaign8_2026-04-29/raw/baseline_campaign7_reproduction.json
```

- [ ] Commit only if branch setup required docs or benchmark-profile compatibility edits.

Commit message:

```bash
git commit -m "bench: add campaign8 baseline compatibility"
```

## Task 1: Decision Documents And Schema Tests

**Files:**
- Create: `docs/plans/cuda_fused_grouping_public_api_campaign8_review.md`
- Create: `docs/plans/cuda_dlpack_interop_campaign8_review.md`
- Create: `docs/plans/cuda_graphs_stream_campaign8_decision.md`
- Modify: `benchmarks/bench_cuda_kernels.py`
- Modify: `benchmarks/bench_cuda_scaling.py`
- Modify: `tests/test_cuda_scaling_benchmark.py`
- Modify: `tests/test_cuda_deep_report_assets.py`

- [ ] Add failing schema tests for Campaign 8 status fields.

Test intent:

```python
required_status_fields = {
    "device_resident_graph_status",
    "public_grouping_api_status",
    "dlpack_interop_status",
    "non_h100_portability_status",
    "stream_graph_status",
    "scatter_tuning_status",
}
```

Run:

```bash
python -m pytest tests/test_cuda_scaling_benchmark.py tests/test_cuda_deep_report_assets.py -q
```

Expected before implementation: tests fail because Campaign 8 schema fields and renderer inputs do not exist yet.

- [ ] Write the public fused grouping review with an initial conservative decision.

Required initial decision:

```text
public_grouping_api_status: deferred
reason: Campaign 8 must first prove that compact grouping metadata is useful, stable, and ownership-safe without requiring full CSR host export.
allowed implementation: private benchmark hook only.
```

- [ ] Write the DLPack interop review with an initial gated decision.

Required initial decision:

```text
dlpack_interop_status: deferred
reason: DLPack requires exact PyCapsule ownership, deleter, stream, mutability, dtype, shape, and same-device semantics before exposing FastPauli-owned memory.
allowed implementation: private benchmark probe only after tests cover capsule lifetime and same-device enforcement.
```

- [ ] Write the stream/CUDA Graph decision with an initial gated decision.

Required initial decision:

```text
stream_graph_status: deferred
reason: public streams and graph capture require complete error propagation, synchronization, lifetime, and Python ownership contracts.
allowed implementation: private benchmark probe only if the decision doc defines capture-safe scope and preserves synchronous public APIs.
```

- [ ] Add Campaign 8 unavailable rows to benchmark scripts.

Expected behavior on CPU-only hosts:

```text
Campaign 8 profiles return unavailable rows with explicit unavailable_reason.
No CUDA import happens at package import time.
Tests can inspect the Campaign 8 schema without a GPU.
```

- [ ] Run schema tests and full local validation.

Run:

```bash
python -m pytest tests/test_cuda_scaling_benchmark.py tests/test_cuda_deep_report_assets.py -q
python scripts/validate.py
```

Expected: tests and validation pass on a CPU-only or non-CUDA local host.

- [ ] Commit the decision and schema work.

Commit message:

```bash
git commit -m "docs: define campaign8 cuda decision gates"
```

## Task 2: Private Device-Resident Graph Consumer

**Files:**
- Modify: `src/cuda/device_commutation_matrix.cu`
- Modify: `src/cuda/device_commutation_matrix.cuh`
- Modify: `src/cuda/workspace.cu`
- Modify: `src/cuda/workspace.cuh`
- Modify: `bindings/python/pauli_sum_py.cpp`
- Modify: `benchmarks/bench_cuda_scaling.py`
- Modify: `tests/test_phase11_cuda_kernels.py`

- [ ] Add a CUDA-gated failing test for the private device-resident graph hook.

Test behavior:

```text
mode=device_resident_graph returns compact graph metadata.
The compact metadata digest equals the CPU anti-commutation graph digest on small deterministic inputs.
The result does not include full CSR column indices unless validation mode explicitly requests them.
```

Run:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 python -m pytest tests/test_phase11_cuda_kernels.py -q
```

Expected before implementation: the private hook or mode is missing.

- [ ] Implement the private hook mode `device_resident_graph`.

Implementation requirements:

```text
consume an existing DeviceCommutationMatrix buffer
compute compact graph metadata on the CUDA device
copy only compact metadata to host for the default benchmark path
support a validation-only full CSR output for small cases
preserve moved-from, wrong-device, wrong-shape, and allocation guardrail behavior
avoid installed public header declarations
```

- [ ] Add benchmark rows comparing Campaign 7 CSR export and Campaign 8 device-resident graph consumer.

Required row labels:

```text
campaign7_csr_graph_export
campaign8_device_resident_graph_compact
campaign8_device_resident_graph_validation_csr
dense_to_host
count_commuting_axis_none
count_commuting_axis_0
count_commuting_axis_1
```

- [ ] Run CUDA correctness and stress benchmarks.

Run:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 python -m pytest tests/test_phase11_cuda_kernels.py -q
python benchmarks/bench_cuda_scaling.py --profile campaign8-device-graph --repeat 7 --warmup 2 --json
```

Expected: correctness passes, compact graph rows are present, and full CSR output is absent from high-scale default rows.

- [ ] Commit the private device-resident graph consumer.

Commit message:

```bash
git commit -m "perf(cuda): add device-resident graph consumer probe"
```

## Task 3: Device-Resident Grouping Consumer And Public API Gate

**Files:**
- Modify: `docs/plans/cuda_fused_grouping_public_api_campaign8_review.md`
- Modify only if accepted: `include/fastpauli/device_commutation_matrix.hpp`
- Modify: `src/cuda/device_commutation_matrix.cu`
- Modify: `src/cuda/device_commutation_matrix.cuh`
- Modify: `bindings/python/pauli_sum_py.cpp`
- Modify: `benchmarks/bench_cuda_scaling.py`
- Modify: `tests/test_phase11_cuda_kernels.py`
- Modify only if public API is accepted: `docs/user/performance.md`

- [ ] Add failing correctness tests for `mode=device_grouping_consumer`.

Test behavior:

```text
small deterministic Pauli sums produce the same grouping conflict summary as the CPU oracle
row and group ordering is stable across two calls
large stress rows return compact metadata and do not copy full CSR edges by default
```

- [ ] Implement the private device-resident grouping consumer.

Implementation requirements:

```text
consume DeviceCommutationMatrix without calling to_host()
compute grouping-compatible conflict metadata on device
copy only compact metadata needed for deterministic grouping or benchmark reporting
return a correctness digest for stress workloads
reuse private workspace without leaking allocations across failures
```

- [ ] Decide whether public grouping API is accepted.

Accept only if all of these are true:

```text
exact return type, dtype, shape, and ownership are documented
ordering is deterministic and tested
host copy size is bounded and reported
CPU-only behavior matches FastPauli CUDA rebuild guidance
API docs and docstrings can be written without exposing benchmark-only internals
H100 benchmarks show a stable advantage over existing public boundaries for a real user workflow
```

If any condition fails, keep `public_grouping_api_status: deferred` and keep the implementation private.

- [ ] If public API is accepted, add the public method and docs in the same commit.

Required public docs:

```text
method name
input requirements
return type and ordering
synchronization behavior
failure modes
benchmark boundary labels
source-build-only performance evidence caveat
```

- [ ] Run CUDA correctness and benchmark validation.

Run:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 python -m pytest tests/test_phase11_cuda_kernels.py -q
python benchmarks/bench_cuda_scaling.py --profile campaign8-grouping-consumer --repeat 7 --warmup 2 --json
```

Expected: correctness passes and the public API status in benchmark rows matches the decision document.

- [ ] Commit the grouping consumer and API decision.

Commit message if private only:

```bash
git commit -m "perf(cuda): add device-resident grouping consumer probe"
```

Commit message if a public API is accepted:

```bash
git commit -m "feat(cuda): add device-resident grouping consumer"
```

## Task 4: DLPack Or Framework Interop Probe

**Files:**
- Modify: `docs/plans/cuda_dlpack_interop_campaign8_review.md`
- Modify only if accepted: `include/fastpauli/device_commutation_matrix.hpp`
- Modify: `src/cuda/device_commutation_matrix.cu`
- Modify: `src/cuda/device_commutation_matrix.cuh`
- Modify: `bindings/python/pauli_sum_py.cpp`
- Modify: `benchmarks/bench_cuda_scaling.py`
- Modify: `benchmarks/bench_competitive_baselines.py`
- Modify: `tests/test_phase11_cuda_kernels.py`

- [ ] Add tests for the accepted interop boundary or unavailable status.

Test behavior:

```text
CuPy consumer path still works through CUDA Array Interface.
DLPack path is unavailable with explicit reason if the review keeps it deferred.
If DLPack is accepted, capsule ownership is single-consumer, same-device checks are enforced, dtype and shape are exact, and use-after-release is rejected or impossible through the capsule contract.
```

- [ ] Implement only the interop surface accepted by the review.

Allowed outcomes:

```text
retain existing CUDA Array Interface only and record dlpack_interop_status: deferred
add private benchmark-only DLPack probe and record dlpack_interop_status: accepted_private_probe
add public DLPack export only if API stability, ownership, and docs are accepted
```

- [ ] Add framework consumer benchmark rows.

Required row labels when available:

```text
cupy_cuda_array_interface_consumer
pytorch_dlpack_consumer
dlpack_unavailable
```

- [ ] Run interop tests and benchmarks.

Run:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 python -m pytest tests/test_phase11_cuda_kernels.py -q
python benchmarks/bench_cuda_scaling.py --profile campaign8-interop --repeat 7 --warmup 2 --json
```

Expected: available framework rows pass digest checks; unavailable frameworks record explicit reasons without failing the whole benchmark.

- [ ] Commit the interop decision and implementation.

Commit message:

```bash
git commit -m "perf(cuda): evaluate device-output interop consumers"
```

## Task 5: Non-H100 NVIDIA Portability Evidence

**Files:**
- Create: `docs/benchmarks/reports/cuda_portability_campaign8_non_h100_nvidia_2026-04-29.md`
- Create data under: `docs/benchmarks/data/cuda_portability_campaign8_non_h100_nvidia_2026-04-29/`
- Modify: `benchmarks/bench_cuda_scaling.py`
- Modify: `docs/architecture/hardware_targets_and_testing.md`
- Modify: `docs/user/performance.md`

- [ ] Select the portability host and record its hardware identifier.

Priority order:

```text
1. A100, SM80
2. RTX 6000 Ada, SM89
3. L4, SM89
4. A10, SM86
```

- [ ] Build from source on the non-H100 NVIDIA host.

Run with the matching architecture:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=80 python scripts/validate.py
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=89 python scripts/validate.py
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=86 python scripts/validate.py
```

Expected: run only the command matching the selected host architecture; validation
passes or the report records the exact build/runtime blocker.

- [ ] Run retained public and private benchmark consumer checks.

Run:

```bash
python benchmarks/bench_cuda_scaling.py --profile campaign8-portability --repeat 5 --warmup 1 --json
```

Expected: retained public `DeviceCommutationMatrix` APIs and Campaign 8 private benchmark consumers either pass correctness on the non-H100 host or produce a documented blocker.

- [ ] Write the non-H100 portability report.

Required fields:

```text
hardware identifier
GPU name and compute capability
driver and toolkit
compiled architectures
git revision
validation command and result
benchmark command and result
unsupported or unavailable rows with reasons
whether broad claims may remain H100-only or widen to named non-H100 evidence
```

- [ ] Commit portability evidence.

Commit message:

```bash
git commit -m "bench(cuda): add campaign8 non-h100 portability evidence"
```

## Task 6: Stream-Aware Or CUDA Graph Probe

**Files:**
- Modify: `docs/plans/cuda_graphs_stream_campaign8_decision.md`
- Modify only if accepted: `include/fastpauli/device_commutation_matrix.hpp`
- Modify: `src/cuda/device_commutation_matrix.cu`
- Modify: `src/cuda/device_commutation_matrix.cuh`
- Modify: `src/cuda/workspace.cu`
- Modify: `src/cuda/workspace.cuh`
- Modify: `bindings/python/pauli_sum_py.cpp`
- Modify: `benchmarks/bench_cuda_scaling.py`
- Modify: `tests/test_phase11_cuda_kernels.py`

- [ ] Add tests for the accepted stream/graph status.

Test behavior:

```text
public APIs remain synchronous and default-stream unless the decision document accepts a public change
private stream/graph probe returns the same correctness digest as synchronous execution
capture failures or unsupported stream modes return explicit unavailable reasons
```

- [ ] Implement the accepted private or public stream/graph probe.

Allowed private benchmark behavior:

```text
capture repeated fill plus device-resident consumer when CUDA Graph capture is safe
use explicit CUDA events for timing inside benchmark code only
preserve public synchronize-before-return semantics
record graph instantiation cost separately from replay timing
```

- [ ] Run stream/graph benchmarks.

Run:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 python -m pytest tests/test_phase11_cuda_kernels.py -q
python benchmarks/bench_cuda_scaling.py --profile campaign8-stream-graph --repeat 7 --warmup 2 --json
```

Expected: rows distinguish synchronous default-stream, private stream probe, graph instantiation, and graph replay timing.

- [ ] Commit stream/graph evidence or deferral.

Commit message:

```bash
git commit -m "perf(cuda): evaluate campaign8 stream graph probe"
```

## Task 7: NCU-Guided CSR Scatter Tuning

**Files:**
- Modify: `src/cuda/device_commutation_matrix.cu`
- Modify: `src/cuda/device_commutation_matrix.cuh`
- Modify: `src/cuda/workspace.cu`
- Modify: `src/cuda/workspace.cuh`
- Modify: `benchmarks/bench_cuda_scaling.py`
- Modify: `tests/test_phase11_cuda_kernels.py`

- [ ] Gate scatter tuning on profiler evidence.

Proceed only if both conditions are true:

```text
the retained device-resident graph consumer still uses CSR scatter or equivalent edge scatter
Nsight Compute shows scatter is material to the retained device-resident workflow after full CSR host export has been removed
```

If either condition is false, record `scatter_tuning_status: rejected_no_consumer` or `scatter_tuning_status: rejected_not_dominant`.

- [ ] Profile the retained graph consumer before A/B edits.

Run:

```bash
python scripts/cuda_deep_profile.py --campaign campaign8 --profile campaign8-device-graph --tool ncu
```

Expected: profiler output names the scatter kernel, memory throughput, occupancy, achieved active warps, branch efficiency, and stall reasons.

- [ ] Test A/B scatter variants only when profiler evidence justifies them.

Allowed variants:

```text
warp-tiled scatter
block-tiled scatter
vectorized column-index stores where alignment permits
row compaction before scatter
CUB/CCCL prefix/scatter replacement with persistent workspace
```

- [ ] Run exact correctness checks for every A/B variant.

Run:

```bash
FASTPAULI_CUDA_SCATTER_VARIANT=warp_tiled FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 python -m pytest tests/test_phase11_cuda_kernels.py -q
FASTPAULI_CUDA_SCATTER_VARIANT=block_tiled FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 python -m pytest tests/test_phase11_cuda_kernels.py -q
FASTPAULI_CUDA_SCATTER_VARIANT=vectorized_stores FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 python -m pytest tests/test_phase11_cuda_kernels.py -q
FASTPAULI_CUDA_SCATTER_VARIANT=row_compaction FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 python -m pytest tests/test_phase11_cuda_kernels.py -q
FASTPAULI_CUDA_SCATTER_VARIANT=cub_prefix_scatter FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 python -m pytest tests/test_phase11_cuda_kernels.py -q
```

Expected: every retained variant produces exact graph equality on small cases and digest equality on stress cases.

- [ ] Retain only variants that improve the retained device-resident workflow.

Retention rule:

```text
retain a variant only if median repeat-7 device-resident graph workflow time improves by at least 5% on H100 without correctness regressions, allocation growth that changes the documented boundary, or worse non-H100 portability behavior
```

- [ ] Commit retained scatter tuning or documented rejection.

Commit message if retained:

```bash
git commit -m "perf(cuda): tune device-resident graph scatter"
```

Commit message if rejected:

```bash
git commit -m "docs(cuda): record campaign8 scatter tuning rejection"
```

## Task 8: Profiling, Report, Plots, And README Landscape

**Files:**
- Create: `scripts/render_cuda_campaign8_assets.py`
- Create: `docs/benchmarks/reports/cuda_deep_optimization_h100_campaign8_2026-04-29.md`
- Create data under: `docs/benchmarks/data/cuda_deep_optimization_h100_campaign8_2026-04-29/`
- Create plots under: `docs/benchmarks/plots/cuda_h100_campaign8_*.svg`
- Modify: `README.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/plans/cuda_deep_optimization_plan.md`
- Modify: `docs/user/performance.md`
- Modify: `docs/architecture/cuda_backend.md`
- Modify: `tests/test_cuda_deep_report_assets.py`

- [ ] Capture final H100 validation.

Run:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 python scripts/validate.py
```

Expected: validation passes on the H100 source-build host.

- [ ] Run Compute Sanitizer on retained CUDA paths.

Run:

```bash
compute-sanitizer --tool memcheck python -m pytest tests/test_phase11_cuda_kernels.py -q
compute-sanitizer --tool racecheck python -m pytest tests/test_phase11_cuda_kernels.py -q
compute-sanitizer --tool initcheck python -m pytest tests/test_phase11_cuda_kernels.py -q
compute-sanitizer --tool synccheck python -m pytest tests/test_phase11_cuda_kernels.py -q
```

Expected: CUDA error summaries are clean. Known Python-extension exit diagnostics must be separated from CUDA sanitizer errors in the report.

- [ ] Capture Nsight Systems and Nsight Compute profiles.

Run:

```bash
python scripts/cuda_deep_profile.py --campaign campaign8 --profile campaign8-final --tool nsys
python scripts/cuda_deep_profile.py --campaign campaign8 --profile campaign8-final --tool ncu
```

Expected: committed profiler metadata identifies the commands, output paths, kernel focus, and any unavailable profiler permissions.

- [ ] Generate Campaign 8 plots and summary JSON.

Run:

```bash
python scripts/render_cuda_campaign8_assets.py \
  --data-dir docs/benchmarks/data/cuda_deep_optimization_h100_campaign8_2026-04-29 \
  --plot-dir docs/benchmarks/plots
```

Required plots:

```text
cuda_h100_campaign8_device_resident_consumers.svg
cuda_h100_campaign8_interop_consumers.svg
cuda_h100_campaign8_stream_graph.svg
cuda_h100_campaign8_scatter_ab.svg
cuda_h100_campaign8_portability.svg
cuda_h100_campaign8_performance_landscape.svg
```

- [ ] Write the Campaign 8 report.

Required sections:

```text
Evidence
Hardware
Validation
Results
Profiler Findings
Decision Outcomes
Non-H100 Portability
Broad Landscape
Remaining Headroom
```

- [ ] Refresh README and user docs only with checked evidence.

README requirement:

```text
The performance plot remains a broad across-the-board landscape, including CPU scalar, every captured optimized CPU selector, CUDA transfer-inclusive rows, boundary-specific CUDA rows, CUDA device-resident rows, compact consumer rows, framework consumer rows where available, and semantically comparable external baselines.
```

- [ ] Run renderer and docs validation.

Run:

```bash
python -m pytest tests/test_cuda_deep_report_assets.py tests/test_cuda_scaling_benchmark.py -q
python scripts/validate.py
git diff --check
```

Expected: validation passes, renderer tests prove the Campaign 8 report/plot assets are fresh, and whitespace checks pass.

- [ ] Commit final evidence and docs.

Commit message:

```bash
git commit -m "docs: publish campaign8 cuda optimization evidence"
```

## Task 9: Review, Merge, Push, And CI Closeout

**Files:**
- Read: `docs/quality/code_review.md`
- Read: `docs/benchmarks/reports/cuda_deep_optimization_h100_campaign8_2026-04-29.md`
- Modify only for fixes: files named by review findings

- [ ] Run final branch validation.

Run:

```bash
python scripts/validate.py
git diff --check
```

Expected: validation and whitespace checks pass on the feature branch.

- [ ] Request independent review before merge.

Reviewer inputs:

```text
branch name
commit list
Campaign 8 goal and scope
validation commands and results
H100 hardware metadata
non-H100 portability metadata or blocker
Nsight and Compute Sanitizer evidence
known deferred surfaces
relevant docs listed in Source Inputs
```

- [ ] Resolve every P0/P1 finding and either fix or explicitly defer P2 findings.

Expected: review closeout records finding counts, resolutions, validation rerun, and residual risk.

- [ ] Merge locally to `main`.

Run:

```bash
git switch main
git pull --ff-only
git merge --ff-only codex/h100-campaign8-execution
```

Expected: fast-forward merge succeeds.

- [ ] Validate merged `main`.

Run:

```bash
python scripts/validate.py
git diff --check
```

Expected: validation and whitespace checks pass on merged `main`.

- [ ] Push and confirm CI.

Run:

```bash
git push origin main
gh run list --branch main --limit 5
gh run watch "$(gh run list --branch main --limit 1 --json databaseId --jq '.[0].databaseId')"
gh run view "$(gh run list --branch main --limit 1 --json databaseId --jq '.[0].databaseId')" --json conclusion,status,url
```

Expected: the pushed `main` CI run completes successfully, or the closeout records a real blocker with exact failing job URLs.

- [ ] Delete the merged local branch.

Run:

```bash
git branch -d codex/h100-campaign8-execution
```

Expected: local feature branch is removed after merge.

## Exhaustion Criteria

Campaign 8 is complete only when all of the following are true:

```text
all six Campaign 7 remaining-headroom items have an accepted, rejected, deferred, or blocked decision backed by evidence
device-resident graph consumer evidence avoids full CSR edge-list host export for the retained high-scale path
public fused grouping API is either accepted with exact semantics and docs or explicitly deferred with rationale
DLPack/framework interop is either accepted with ownership/lifetime tests or explicitly deferred with rationale
non-H100 NVIDIA retained-consumer portability is passed or blocked with a hardware-specific report
CUDA Graphs or stream-aware execution is accepted only with complete semantics, or deferred with rationale
CSR scatter tuning is performed only when NCU shows it matters to the retained device-resident workflow
H100 validation passes on the final branch
Compute Sanitizer memcheck, racecheck, initcheck, and synccheck are clean for retained CUDA changes
Nsight Systems and Nsight Compute evidence is captured or profiler-permission blockers are recorded
raw JSON, metadata, profiler files, plots, and reports are checked in under Campaign 8 paths
README performance plot remains an across-the-board CPU/CUDA/external comparison, not a narrow single-campaign slice
independent agent-driven review is completed and blocking findings are resolved
merged main is validated, pushed, CI is green, and the local feature branch is cleaned up
```

## Remaining-Headroom Mapping

```text
device-resident graph consumers that avoid exporting full CSR edge lists
  Covered by Tasks 2, 3, 7, and 8.

public fused grouping API only after exact return semantics and ownership are accepted
  Covered by Tasks 1 and 3.

DLPack or framework interop for retained device outputs
  Covered by Tasks 1 and 4.

non-H100 NVIDIA retained-consumer portability evidence
  Covered by Task 5.

CUDA Graphs or stream-aware execution only after a complete public contract
  Covered by Tasks 1 and 6.

additional NCU-guided CSR scatter tuning only if a fully device-resident graph consumer needs it
  Covered by Task 7.
```

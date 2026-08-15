# Backend-Neutral Accelerator Campaign 9 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking, and no implementation slice may claim completion until the validation ladder and review closeout in this document are satisfied.

**Goal:** Implement the backend-neutral accelerator object model for
target-specific accelerator builds. FastPauli keeps one public Python contract
for CPU, CUDA, ROCm/HIP, and future Apple accelerator targets, while normal
source builds compile at most one accelerator runtime. CPU-only imports,
current CUDA-only and HIP-only behavior, object-local backend identity,
deterministic unsupported-backend errors, and evidence-based benchmark
reporting remain mandatory.

**Architecture:** Host `PauliSum` remains CPU-resident. Accelerator-resident
objects own memory on exactly one backend and one backend-local device ordinal.
CUDA and HIP implementation units remain disjoint and selectable at build time;
they are not linked into one normal extension. Runtime dispatch must be selected
from each object identity rather than a process-global active backend. Wrong
device operations fail before allocation or kernel launch. Cross-backend object
operations are unavailable under the target-specific build policy because CUDA
and HIP device objects cannot coexist in one normal extension; a future mixed
runtime or explicit host-staged copy plan must reopen that design before any
claim is made. Backend-neutral APIs do not imply CUDA wheels, ROCm wheels,
combined accelerator wheels, multi-GPU execution, HIP DLPack, HIP CUDA Array
Interface, public streams, public graphs, public workspaces, Metal/MPS support,
or non-MI300X AMD portability claims.

**Tech Stack:** C++20, nanobind, scikit-build-core, CMake CUDA language support, CMake HIP language support, CUDA 12.x source builds, ROCm/HIP source builds, pytest, NumPy, existing FastPauli CPU/CUDA/ROCm benchmark and validation infrastructure.

---

## Status And Trigger

```text
status: completed_target_specific_closeout_lanes
campaign: backend_neutral_accelerator_campaign9
trigger: accepted backend-neutral API scope for target-specific accelerator builds
decision contract: docs/architecture/backend_neutral_accelerators.md
previous readiness plan: docs/plans/mi300x_rocm_optimization_campaign8_plan.md
previous readiness report: docs/benchmarks/reports/rocm_campaign8_architecture_readiness_2026-05-01.md
closeout report: docs/benchmarks/reports/backend_neutral_accelerator_campaign9_2026-05-01.md
```

This campaign is the implementation handoff for the Campaign 8 backend-neutral
object-model gate. It may proceed without a non-MI300X AMD host because it does
not broaden AMD portability claims. It does not require an uncommon mixed
NVIDIA+AMD host because mixed CUDA+HIP source builds are not the normal support
target. It still requires accelerator-host validation before completion claims:

```text
CUDA-only regression lane on an NVIDIA host
HIP-only regression lane on the MI300X host or another already accepted HIP lane
configure-time rejection lane for FASTPAULI_ENABLE_CUDA=ON with FASTPAULI_ENABLE_HIP=ON
CPU-only control lane on the local development host
```

If a future product need requires one extension to link CUDA and HIP together,
that work must start with a separate accepted plan. It is not a Campaign 9
completion gate.

Current implementation and validation status:

```text
implemented: shared accelerator backend identity helper
implemented: structured _accelerator_status() compiled and available backend sets
implemented: PauliSum.to_device(backend=None|"auto"|"cuda"|"hip")
implemented: DeviceCommutationMatrix.empty(..., backend=None|"auto"|"cuda"|"hip")
implemented: DeviceCommutationMatrix.backend
implemented: CPU-only selector/status tests that simulate future mixed-runtime ambiguity
implemented: disjoint CMake CUDA, HIP, and accelerator-stub source-set declarations
implemented: CPU-only source-shape tests for target-specific source-set separation
implemented: CPU-only backend/device validation helper tests that simulate cross-backend and wrong-device failures
implemented: _build_info() and benchmark metadata fields for Campaign 9 build and timing boundaries
completed: CUDA-only remote regression refresh for this campaign on H100
completed: HIP-only remote regression refresh for this campaign on MI300X
completed: configure-time rejection evidence for FASTPAULI_ENABLE_CUDA=ON with FASTPAULI_ENABLE_HIP=ON under the target-specific policy
completed: Campaign 9 report row that records target-specific CPU, CUDA, and HIP build boundaries
```

Completion evidence is checked into
`docs/benchmarks/data/backend_neutral_accelerator_campaign9_2026-05-01/` and
summarized in
`docs/benchmarks/data/backend_neutral_accelerator_campaign9_2026-05-01/summary.json`
and
`docs/benchmarks/reports/backend_neutral_accelerator_campaign9_2026-05-01.md`.
The evidence lanes cover local CPU-only validation, H100 CUDA-only validation,
MI300X HIP-only validation, and the documented CUDA+HIP configure-time rejection.

## In Scope

```text
backend_neutral_status_schema
object_local_backend_identity
backend_construction_selector_contract
device_commutation_matrix_backend_property
ambiguous_dual_runtime_policy
target_specific_accelerator_builds
mixed_cuda_hip_build_rejection
future_multi_runtime_design_gate
same_backend_same_device_validation
cpu_only_header_safety
cuda_target_regression_lane
hip_target_regression_lane
benchmark_boundary_reporting
no_wheel_or_portability_claim
```

This campaign may change:

```text
shared accelerator backend identity and status helpers
explicit backend selection for initial accelerator object construction
DevicePauliSum backend dispatch plumbing
DeviceCommutationMatrix backend identity and Python property
Python _accelerator_status() schema
wrong-device validation helpers and future-only cross-backend simulation helpers
tests, docs, validation scripts, benchmark protocol, and README planning links
```

## Out Of Scope

```text
ROCm wheels
CUDA wheels
combined CUDA+ROCm wheels
non-MI300X AMD portability claims
new HIP kernels beyond existing retained HIP operations
new CUDA kernels beyond dispatch and build-graph changes needed by this campaign
multi-GPU CUDA or ROCm execution
cross-device copies
cross-backend copies
HIP DLPack
HIP CUDA Array Interface
public stream handles
public graph execution
public workspace handles
Metal/MPS implementation
new performance speedup claims without benchmark evidence
normal builds that link both CUDA and HIP runtimes
mixed NVIDIA+AMD host validation as a completion requirement
```

## Acceptance Criteria

The campaign is complete only when all criteria below pass or a blocking
external dependency is recorded with an explicit incomplete status.

```text
CPU-only build imports without CUDA, ROCm, HIP, or accelerator environment variables
CUDA-only source build still exposes existing CUDA behavior and passes current CUDA tests
HIP-only source build still exposes existing MI300X-evidenced HIP behavior and passes current HIP tests
FASTPAULI_ENABLE_CUDA=ON with FASTPAULI_ENABLE_HIP=ON remains a documented configure-time error
_accelerator_status() reports structured compiled and runtime state for cpu and the selected accelerator target without selecting one global accelerator backend as an object-dispatch source
DevicePauliSum.backend reports the owning backend for CUDA and HIP objects
DeviceCommutationMatrix.backend reports the owning backend for CUDA and HIP objects
same-backend same-device operations preserve existing semantics
same-backend wrong-device operations fail deterministically before kernel launch
cross-backend operations remain unavailable unless a future accepted mixed-runtime or explicit-copy plan defines them
public CPU-only headers include no CUDA, HIP, or ROCm runtime headers
benchmark rows distinguish CPU-only, CUDA-only target, HIP-only target, and future Apple-target timing boundaries
README and architecture docs do not imply wheel support, broader AMD support, multi-GPU support, HIP DLPack, or Metal/MPS support
```

## Task 1: Confirm Source-Of-Truth And Red Tests

- [ ] Confirm this plan remains in the source-of-truth route in `README.md`,
  `AGENTS.md`, `docs/roadmap.md`, `docs/plans/rocm_next_waves_plan.md`,
  `docs/architecture/backend_neutral_accelerators.md`,
  `docs/architecture/cuda_backend.md`, `docs/architecture/rocm_backend.md`,
  `docs/architecture/hardware_targets_and_testing.md`,
  `docs/benchmarks/protocol.md`, and `scripts/validate.py`.
- [ ] Keep `tests/test_backend_neutral_campaign9_plan.py` aligned with this
  plan so future agents cannot drop the in-scope key set, source-of-truth
  links, benchmark protocol fields, or out-of-scope support boundaries.
- [ ] Keep the current runtime/build implementation aligned with the
  target-specific contract. The implementation campaign must preserve the
  CUDA+HIP configure-time rejection unless a later accepted mixed-runtime plan
  replaces it.

Expected red-test coverage:

```text
backend_neutral_status_schema
object_local_backend_identity
device_commutation_matrix_backend_property
target_specific_accelerator_builds
mixed_cuda_hip_build_rejection
future_multi_runtime_design_gate
same_backend_same_device_validation
cpu_only_header_safety
cuda_target_regression_lane
hip_target_regression_lane
benchmark_boundary_reporting
no_wheel_or_portability_claim
```

## Task 2: Shared Backend Identity And Status Schema

- [ ] Add a shared C++ backend identity surface that is independent of CUDA or
  HIP headers. Candidate files:

```text
include/fastpauli/accelerator_status.hpp
src/accelerator_status.cpp
```

- [ ] Define a small enum or equivalent value type for:

```text
none
cpu
cuda
hip
```

- [ ] Provide conversion helpers that return stable lowercase names and
  deterministic errors for unknown values.
- [ ] Replace scattered string literals in device object code with the shared
  helpers where doing so does not pull accelerator headers into CPU-only
  public headers.
- [ ] Update Python status conversion so `_accelerator_status()` reports both
  compiled and runtime-visible state. The accepted schema must support these
  rows:

```text
cpu_only_no_accelerator_runtime
cuda_only_runtime_visible
cuda_only_runtime_hidden
hip_only_runtime_visible
hip_only_runtime_hidden
future_mixed_runtime_no_runtime_visible
future_mixed_runtime_cuda_runtime_visible
future_mixed_runtime_hip_runtime_visible
future_mixed_runtime_both_runtimes_visible
```

Minimum Python shape:

```python
{
    "active_backend": "none",
    "compiled_backends": ["cpu", "cuda", "hip"],
    "available_backends": ["cuda", "hip"],
    "cuda": {...},
    "hip": {...},
}
```

`active_backend` remains for compatibility, but target-specific builds must not
use it as a dispatch source for accelerator-resident object methods.

## Task 3: Target-Specific Build Graph

- [ ] Keep CPU-only configuration as the default and preserve current
  CUDA-only and HIP-only source-build commands.
- [ ] Preserve disjoint CUDA, HIP, and accelerator-stub implementation units so
  target-specific builds cannot accidentally link the wrong runtime.
- [ ] Keep CUDA headers out of HIP public surfaces, HIP headers out of CUDA
  public surfaces, and both out of CPU-only public headers.
- [ ] Keep `FASTPAULI_ENABLE_CUDA=ON` with `FASTPAULI_ENABLE_HIP=ON` as a
  configure-time error and update the error text so it cites the
  target-specific accelerator policy rather than a missing mixed-host gate.
- [ ] Add or keep configure/source-shape tests that prove the mutual exclusion
  is intentional and that the CUDA/HIP source sets remain separable for future
  target-specific or mixed-runtime planning.
- [ ] Update validation docs to record CPU-only, CUDA-only target, HIP-only target, and
  dual-request rejection commands used for evidence.

Required validation commands:

```bash
python scripts/validate.py
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 python scripts/validate.py
FASTPAULI_VALIDATE_HIP=1 FASTPAULI_HIP_ARCHITECTURES=gfx942 python scripts/validate.py
python -m pip install -v -e ".[test]" --no-build-isolation --config-settings=cmake.define.FASTPAULI_ENABLE_CUDA=ON --config-settings=cmake.define.FASTPAULI_ENABLE_HIP=ON  # expected configure-time rejection
python -m pytest tests/test_phase10_cuda_foundation.py tests/test_phase12_rocm_foundation.py
```

## Task 4: Backend Construction Selector Contract

- [ ] Add an explicit backend selector to initial accelerator object
  construction. Python `PauliSum.to_device()` must accept an explicit selector,
  for example `backend="cuda"` or `backend="hip"`, while preserving current
  single-backend behavior for existing callers.
- [ ] Add the corresponding C++ construction path for `DevicePauliSum::from_host`
  without making CPU-only public headers include CUDA, HIP, or ROCm runtime
  headers.
- [ ] Add the same backend selector to direct `DeviceCommutationMatrix.empty()`
  construction so tests, reuse paths, and advanced users do not need a global
  active backend to allocate an output matrix.
- [ ] Treat device ordinals as backend-local after backend selection. CUDA
  device `0` and HIP device `0` must never be interpreted as the same device.
- [ ] Keep `backend=None` or the accepted auto selector compatible only when the
  choice is unambiguous.

Accepted default and error behavior:

```text
CPU-only build with omitted backend: raise the current absent-accelerator error
single compiled/runtime-visible backend with omitted backend: choose that backend for backward compatibility
future mixed-runtime build with exactly one runtime-visible backend and omitted backend: choose the visible backend
future mixed-runtime build with both CUDA and HIP runtimes visible and omitted backend: raise an ambiguous-backend error that requires backend="cuda" or backend="hip"
explicit backend with invalid value: raise ValueError before allocation
explicit backend that was not compiled: raise absent-backend error before allocation
explicit backend with hidden or unavailable runtime: raise runtime-unavailable error before allocation
explicit backend and invalid device ordinal: raise backend-specific invalid-device error before allocation when possible
```

Required selector tests:

```text
PauliSum.to_device(backend="cuda") constructs a CUDA DevicePauliSum on CUDA builds
PauliSum.to_device(backend="hip") constructs a HIP DevicePauliSum on HIP builds
PauliSum.to_device(backend="bogus") fails deterministically
future mixed-runtime selector simulation with both runtimes visible rejects omitted backend as ambiguous
future mixed-runtime selector simulation with exactly one runtime visible accepts omitted backend for compatibility
DeviceCommutationMatrix.empty(..., backend="cuda") constructs a CUDA matrix on CUDA builds
DeviceCommutationMatrix.empty(..., backend="hip") constructs a HIP matrix on HIP builds
```

## Task 5: Object-Local Backend Dispatch

- [ ] Ensure `DevicePauliSum` stores and reports object-local backend identity
  across moves, method returns, and errors.
- [ ] Add `DeviceCommutationMatrix.backend` to the C++ and Python public API
  with the same lowercase identity contract as `DevicePauliSum.backend`.
- [ ] Update CUDA object methods to dispatch through CUDA implementation units
  only when the receiver owns CUDA memory.
- [ ] Update HIP object methods to dispatch through HIP implementation units
  only when the receiver owns HIP memory.
- [ ] Ensure CPU-only stubs continue to raise deterministic absent-backend
  errors without linking CUDA, HIP, or ROCm runtime symbols.
- [ ] Add tests for moved-from objects so backend identity does not mask
  ownership errors.

Required object behavior:

```text
CUDA DevicePauliSum.backend == "cuda"
HIP DevicePauliSum.backend == "hip"
CUDA DeviceCommutationMatrix.backend == "cuda"
HIP DeviceCommutationMatrix.backend == "hip"
CPU-only stubs do not report a fake accelerator backend
moved-from objects raise moved-from or invalid-object errors before dispatch
```

## Task 6: Cross-Backend And Device Validation

- [ ] Add one validation helper for binary accelerator operations that checks:

```text
left backend
right backend
left device ordinal
right device ordinal
operation name
```

- [ ] Use the helper before any allocation or kernel launch in operations that
  combine or reuse accelerator-resident objects.
- [ ] Preserve same-backend same-device semantics for existing CUDA and HIP
  operations.
- [ ] Add deterministic errors for same-backend wrong-device operations.
- [ ] Keep CUDA/HIP cross-backend runtime object operations unavailable under
  the target-specific build policy.
- [ ] Keep cross-backend copy semantics unavailable in this campaign.

Minimum error fields:

```text
operation
left backend
left device ordinal
right backend
right device ordinal
failure stage
```

## Task 7: Benchmark And Report Boundaries

- [ ] Extend benchmark metadata so accelerator rows can record:

```text
build_mode: cpu_only, cuda_only, or hip_only; a future Apple target must add an explicit value when implemented
object_backend: cpu, cuda, or hip
compiled_backends
runtime_visible_backends
transfer_boundary: transfer_inclusive, device_resident, host_materialized, compact_consumer, or status_only
```

- [ ] Preserve existing CUDA and ROCm raw data and plots unless a measured row
  is refreshed.
- [ ] Compare CUDA-only target and HIP-only target timing evidence only against rows with
  the same operation and timing boundary.
- [ ] Do not present target-specific configure success or dual-request
  rejection as a performance speedup.
- [ ] Keep the README performance landscape as a broad CPU/CUDA/ROCm/external
  view rather than a narrow backend-neutral snapshot.

## Task 8: Remote Validation Ladder

Execute and record all applicable lanes before completion:

```text
local CPU-only lane: python scripts/validate.py
NVIDIA CUDA-only lane: FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=<sm> python scripts/validate.py
MI300X HIP-only lane: FASTPAULI_VALIDATE_HIP=1 FASTPAULI_HIP_ARCHITECTURES=gfx942 python scripts/validate.py
dual-request rejection lane: configure with FASTPAULI_ENABLE_CUDA=ON and FASTPAULI_ENABLE_HIP=ON and confirm the documented configure-time error
```

Required evidence fields:

```text
host
cpu model
gpu model
driver/runtime versions
CUDA toolkit version when used
ROCm/HIP toolkit version when used
compiled CUDA architectures
compiled HIP architectures
git revision
commands
test results
benchmark rows or explicit status-only evidence
limitations
```

## Task 9: Documentation, Review, Merge, And Closeout

- [ ] Update user-facing docs only after behavior is implemented and validated.
  Planning docs may name accepted scope, but README support claims must not get
  ahead of runtime evidence.
- [ ] Complete the independent review stage required by
  `docs/quality/code_review.md`.
- [ ] Resolve blocking review findings and rerun affected validation.
- [ ] Fast-forward merge to `main` when possible.
- [ ] Validate on merged `main`.
- [ ] Push `main`.
- [ ] Confirm remote CI is green when CI exists.
- [ ] Delete the merged local feature branch.

Closeout must record:

```text
commit revisions
validation commands and results
accelerator hosts used
target-specific accelerator support status
deferred out-of-scope items
residual risk
next recommended campaign
```

## Non-Goals That Must Stay Explicit

The implementation campaign must not soften these boundaries:

```text
ROCm wheels remain unavailable
non-MI300X AMD portability remains blocked without hardware evidence
HIP DLPack remains unavailable until a real ROCm consumer enforces read-only behavior
HIP CUDA Array Interface remains unavailable
multi-GPU ROCm execution remains unavailable
Metal/MPS remains a separate design track
dual CUDA+HIP source-build success does not imply combined binary wheel support
dual CUDA+HIP source builds require a future accepted mixed-runtime plan before any support claim
```

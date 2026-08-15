# Backend-Neutral Accelerator Architecture

This document defines Wolfgang's backend-neutral accelerator API and
target-specific accelerator build policy. CUDA and ROCm/HIP share public Python
contracts where the semantics are the same, but normal Wolfgang builds target
one accelerator runtime at a time. A single extension that links CUDA,
ROCm/HIP, and Metal together is not required for supported CUDA, HIP, CPU, or
Apple accelerator targets.

The executable implementation handoff for this contract is
`docs/plans/backend_neutral_accelerator_campaign9_plan.md`. That plan is
complete under the target-specific policy, with validation evidence in
`docs/benchmarks/reports/backend_neutral_accelerator_campaign9_2026-05-01.md`.

## Current Decision

The current supported build modes remain:

```text
CPU-only
CUDA-only source build
HIP-only source build
Metal-only source build
```

This mode remains deliberately unavailable for normal builds:

```text
FASTPAULI_ENABLE_CUDA=ON with FASTPAULI_ENABLE_HIP=ON
```

`FASTPAULI_ENABLE_CUDA=ON` with `FASTPAULI_ENABLE_HIP=ON` is a
configure-time error by policy. It should not be treated as a blocked release
gate unless a later accepted plan explicitly reopens mixed-runtime packaging
and validation.

Campaign 9 provides a shared CUDA/HIP backend selector for initial device
construction, structured `_accelerator_status()` compiled and available
backend sets, `DeviceCommutationMatrix.backend`, disjoint CMake CUDA/HIP/Metal/stub
source-set declarations, simulated backend/device validation tests,
benchmark/build metadata fields, H100 CUDA-only validation, MI300X HIP-only
validation, and documented CUDA+HIP configure-time rejection evidence. These
surfaces are compatible with CPU-only, CUDA-only, HIP-only, and Metal-only
source builds. The Apple contract is `docs/architecture/apple_accelerator.md`,
with execution handoff in `docs/plans/apple_metal_mps_bringup_plan.md`.

## Backend Identity

Every accelerator-resident object must report one explicit backend identity:

```text
cpu
cuda
hip
metal
```

`metal` is the Apple Silicon source-build backend identity. Public constructors
may accept it only when Wolfgang is built with `FASTPAULI_ENABLE_METAL=ON`;
CPU-only, CUDA-only, and HIP-only builds must reject it with rebuild guidance.

Host `PauliSum` remains CPU-resident and does not become a multi-backend owner.
Accelerator objects must not infer their backend from the active build mode
when the object already owns device memory. Backend identity is object-local and
must survive moves, method returns, and error reporting.

## Build Modes

CPU-only builds remain the default wheel and import path. They must not require
CUDA, ROCm, HIP, or accelerator environment variables.

CUDA-only builds may expose CUDA-backed `DevicePauliSum` and
`DeviceCommutationMatrix` objects. HIP symbols must not be required at import
time or linked into CPU-only public headers.

HIP-only builds may expose HIP-backed `DevicePauliSum` and
`DeviceCommutationMatrix` objects. CUDA symbols must not be required at import
time or linked into CPU-only public headers.

Metal-only builds may expose Metal-backed `DevicePauliSum` and
`DeviceCommutationMatrix` objects on Apple platforms. CUDA and HIP symbols must
not be required, and public CPU headers must not include Apple framework
headers or expose raw Metal object types.

Mixed CUDA+HIP builds are not part of the normal build matrix. If a future
campaign reopens mixed accelerator builds, it must prove:

```text
both CUDA and HIP extension code can compile in one source build
runtime status reports both backends without selecting one as globally active
object methods dispatch from object backend identity, not global build flags
cross-backend operations fail deterministically unless explicit copy semantics are implemented
packaging docs distinguish source-build multi-backend support from wheel support
```

Until such a plan is accepted, target-specific source builds are the support
boundary:

```text
CPU-only default package
CUDA target build
ROCm/HIP target build
Apple/Metal target build
```

The Metal target build must follow `docs/architecture/apple_accelerator.md`:
`FASTPAULI_ENABLE_METAL=ON` is source-build-only and initially mutually
exclusive with CUDA and HIP.

## Device Object Ownership

`DevicePauliSum` owns its packed `x`, packed `z`, and coefficient buffers on
one backend and one device ordinal. Copying remains unavailable unless a future
`.copy()` contract defines allocation, stream, and error behavior. Moving
transfers ownership and leaves a moved-from object that raises deterministic
exceptions on method calls.

`DeviceCommutationMatrix` owns a dense row-major `uint8` flag matrix on one
backend and one device ordinal. Host materialization and compact consumers must
use the owning backend's copy and reduction path. External users may view
accepted interop exports, but they must not free Wolfgang-owned memory.

## Read-Only Export Policy

CUDA read-only DLPack export exists only for accepted CUDA
`DeviceCommutationMatrix` consumers. HIP DLPack remains unavailable after the
Campaign 5 mutation-safety rejection until a real ROCm consumer rejects
mutation of a read-only exported view.

HIP `__cuda_array_interface__` remains permanently unavailable. Exposing a HIP
pointer through CUDA Array Interface would let consumers treat HIP memory as
CUDA memory and is a correctness bug.

## Device Ordinal Semantics

Device ordinals are backend-local. CUDA device `0` and HIP device `0` are not
the same device. Every accelerator object records:

```text
backend identity
device ordinal
runtime status at construction when available
```

Same-backend operations require matching device ordinals unless a later
campaign implements explicit peer-copy or host-staging semantics. Cross-backend
operations are rejected before allocation or kernel launch. Error messages must
name the left backend/device and right backend/device when both are available.

## Status Reporting

`_accelerator_status()` must remain safe on CPU-only machines. Current builds
report structured backend sets:

```text
compiled_backends
compiled_accelerator_backends
available_backends
available_accelerator_backends
active_backend
```

`active_backend` remains a compatibility field and is not an object dispatch
source. Target-specific accelerator builds must preserve the structured
compiled and runtime-available lists rather than collapsing API behavior into
one hard-coded global backend.
`_build_info()` and benchmark metadata must carry the corresponding Campaign 9
build boundary:

```text
accelerator_build_mode
compiled_accelerator_backends
runtime_visible_accelerator_backends
compiled_backends
runtime_visible_backends
```

That behavior requires tests for:

```text
CPU-only build
CUDA-only build with and without visible CUDA runtime
HIP-only build with and without visible HIP runtime
Metal-only build with and without visible Metal runtime
configure-time rejection when CUDA and HIP are both requested
configure-time rejection when Metal is combined with CUDA or HIP
future-only simulation coverage for ambiguous selector behavior
```

## Errors

Accelerator errors must continue to be deterministic Python exceptions. They
must identify:

```text
operation name
backend
device ordinal when known
runtime availability state when relevant
allocation or copy stage when relevant
```

Cross-backend operations must fail before any partial output object is created.
Wrong-device and moved-from errors must remain distinct from absent-runtime
errors.

## Packaging Impact

Backend-neutral APIs do not imply CUDA wheels, ROCm wheels, combined
CUDA+ROCm wheels, or a single extension that links multiple accelerator
runtimes. A future mixed-runtime package requires a release plan that names:

```text
package channel
runtime dependency policy for CUDA and ROCm libraries
build host matrix
runtime test hardware
clean-machine install tests
support-matrix wording
artifact size expectations
```

Until that release plan is accepted and validated, Wolfgang should keep the
default distribution CPU-only and treat CUDA and ROCm as target-specific
source-build evidence.

## Required Tests Before Reopening Mixed CUDA/HIP Builds

A future simultaneous CUDA+HIP campaign must start from a new accepted plan and
add tests for:

```text
configure succeeds with FASTPAULI_ENABLE_CUDA=ON and FASTPAULI_ENABLE_HIP=ON
CPU-only import remains independent of accelerator libraries
_cuda_status(), _hip_status(), and _accelerator_status() all report structured state
CUDA DevicePauliSum.backend remains "cuda"
HIP DevicePauliSum.backend remains "hip"
same-backend same-device operations preserve current semantics
same-backend wrong-device operations fail deterministically
cross-backend operations fail deterministically or use explicitly documented copy semantics
public headers still include no CUDA, HIP, or ROCm runtime headers in CPU-only surfaces
benchmark rows distinguish CUDA-only, HIP-only, and multi-backend timing boundaries
```

No implementation campaign may claim simultaneous CUDA+HIP support until these
tests and the corresponding packaging documentation exist. The absence of this
mixed-host lane is not a blocker for completing CUDA-target, HIP-target, or
Apple Metal-target work.

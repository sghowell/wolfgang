# Wolfgang API Stability And Compatibility

This document defines how Wolfgang treats public APIs, compatibility, versioning, and breaking changes.

## Public Surfaces

Public surfaces include:

```text
Python package name and import paths
public Python classes, methods, functions, properties, and exceptions
documented Python optional extras
documented C++ headers under include/wolfgang
documented CMake options
documented CLI or validation scripts
documented benchmark commands and report schemas
documented CUDA build and runtime behavior
documented ROCm/HIP build and runtime behavior
```

Internal implementation details include:

```text
private Python modules or names prefixed with _
src/ implementation files
bindings internals
undocumented CMake helper targets
benchmark helper internals
test helper internals
```

If a behavior is documented in `README.md`, `docs/user/`, API docs, or source-of-truth architecture docs, treat it as public unless it is explicitly marked experimental.

## Versioning Policy

Wolfgang uses semantic versioning after the first stable release:

```text
MAJOR: incompatible public API or semantic behavior changes
MINOR: backward-compatible functionality
PATCH: backward-compatible fixes
```

Before `1.0.0`, the project may make breaking changes, but they must still be:

```text
documented in changelog or release notes
reflected in tests and source-of-truth docs
called out in migration notes when user-facing
```

Do not claim stable API support before release evidence exists.

## Breaking Changes

Breaking changes include:

```text
renaming or removing public Python APIs
changing dense-label or sparse-list conventions
changing canonical ordering
changing exception types for documented errors
changing coefficient dtype policy
changing CUDA transfer or synchronization behavior
changing ROCm/HIP transfer or synchronization behavior
changing documented CMake option names or defaults
changing optional dependency import behavior
changing benchmark report format after reports are consumed by tooling
```

Non-breaking changes include:

```text
adding new optional methods
adding new optional backend support
adding stricter validation that catches previously invalid input
improving performance without changing output or ordering
adding new docs or examples
```

## Deprecation Policy

After `1.0.0`, public API removal should use a deprecation period:

```text
1. add a warning and replacement guidance
2. document the deprecation in release notes
3. keep tests for the deprecated path
4. remove only in the next compatible major release unless a security issue requires faster action
```

Before `1.0.0`, prefer explicit migration notes over silent churn.

## Python API Design

Python APIs should be:

```text
small and predictable
NumPy-compatible where arrays are involved
clear about ownership and copy behavior
explicit about backend selection
explicit about ordering and endianness
consistent in exception types
```

Public methods should avoid surprising mutation. Methods that return new operators should not mutate `self`. In-place operations must be named with an `_inplace` suffix if introduced later.

Backend selector values should be strings unless a stronger typed public enum is introduced:

```text
"cpu"
"cuda"
"hip"
"metal"
"auto"
```

`"metal"` is source-build-only behind `WOLFGANG_ENABLE_METAL=ON` and follows
`docs/architecture/apple_accelerator.md`. CPU-only, CUDA-only, and HIP-only
builds must reject explicit `"metal"` requests with rebuild guidance.

Unsupported backend requests should raise `RuntimeError` when the backend is unavailable and `ValueError` when the selector itself is invalid.

## C++ API And ABI

The C++ API exists for benchmarks and future non-Python bindings. It is not ABI-stable before a deliberate stable C++ release.

Policy:

```text
documented headers under include/wolfgang are source-compatible public API
binary ABI compatibility is not promised before explicit release support
public structs should avoid unnecessary layout churn once implementation begins
internal helpers stay under src/detail unless promoted through API review
```

Header placement is part of the API contract:

```text
include/wolfgang/*.hpp contains public C++ API headers only
src/detail/*.hpp contains private native implementation helpers
private helper headers are not installed, documented as user API, or included by downstream code
helper headers move into include/wolfgang only after an explicit public-API decision
```

When C++ API changes are necessary:

```text
update docs
update benchmarks
update bindings
update tests
state whether source compatibility changed
```

## CUDA API Compatibility

CUDA public behavior includes:

```text
WOLFGANG_ENABLE_CUDA build option
PauliSum.to_device(device=...)
DevicePauliSum.to_host()
DevicePauliSum.commutes_with_device(..., output=None)
DeviceCommutationMatrix dense uint8 ownership, to_host(), count_commuting(), conflict_degrees(), DLPack export, and CUDA-array-interface export
backend selector behavior
CUDA availability errors and skips
stream and synchronization semantics once documented
accepted device array protocols
```

Do not change CUDA synchronization or ownership semantics without updating `docs/architecture/cuda_backend.md`, tests, and user documentation.

Campaign 5 promotes dense device-output commutation as an experimental public
API. The promoted surface is `DeviceCommutationMatrix` plus
`DevicePauliSum.commutes_with_device()`, using one Wolfgang-owned dense
row-major `uint8` device buffer on the same CUDA device as the operands.
Compatibility for this experimental surface covers:

```text
class and method names
shape, dtype, row-major ordering, and 1/0 flag semantics
same-device validation
default-stream synchronize-before-return behavior
CPU-only import success with CUDA rebuild-guidance RuntimeError on use
CUDA-array-interface keys needed by downstream GPU consumers
compact-summary count semantics and uint64 host result dtype
benchmark schema fields that separate allocation, reuse, export, compact summaries, and host materialization
```

Campaign 6 extends this surface with `DeviceCommutationMatrix.count_commuting(...)`
after `docs/plans/cuda_commutation_consumer_api_review.md` retained compact
summary counts. Compatibility for that extension covers axis names, return
types, synchronous default-stream behavior, CPU-only rebuild-guidance behavior,
and the guarantee that only compact `uint64` count results are copied to host.

Campaign 9 extends the experimental CUDA surface with
`DeviceCommutationMatrix.conflict_degrees(axis=None|0|1)` and read-only
`DeviceCommutationMatrix.__dlpack__` / `__dlpack_device__`. Compatibility for
`conflict_degrees` covers anti-commuting complement semantics, axis names,
return types, synchronous default-stream behavior, and compact `uint64` host
materialization. Compatibility for DLPack covers shape `(rows, cols)`, dtype
`uint8`, row-major dense layout, device tuple `(kDLCUDA, device_ordinal)`,
single-consumer capsule behavior, owner lifetime retention by deleter context,
`copy=True` rejection, `stream=0` rejection, versioned capsule support for
positive `max_version`, legacy unversioned capsule rejection, and read-only
export status. The true
`DevicePauliSum.group_commuting_device(...)` API remains unavailable and is not
part of the compatibility promise.

Before `1.0.0`, this experimental API may still receive source-compatible
extensions or migration notes, but it must not silently change shape, dtype,
ordering, synchronization, ownership, compact-summary return types, or
documented exception categories.

Post-Phase 11 reusable workspace, bit-packed commutation, raw external
device-output buffers, stream-aware, and device-statevector prototypes remain
internal implementation or benchmark surfaces unless they are explicitly
promoted through API review. They must not be documented in README or
user-facing API docs as supported features, and bindings must not expose them
as public Python objects without:

```text
experimental API status and versioning note
absent-CUDA and CPU-only import behavior
device-ordinal compatibility rules
capacity ownership and reset/release behavior
synchronization and error-propagation semantics
tests for invalid device, invalid dtype/layout, moved-from objects, and public benchmark labels
```

CUDA Campaign 4 private hooks are intentionally excluded from the public API
compatibility promise. `_cuda_workspace_probe_for_testing()` is a test and
benchmark harness hook only; it may change or disappear before `1.0.0` without
deprecation. Environment variables prefixed `FASTPAULI_CUDA_BENCH_` select
benchmark experiments and must not be described as supported user-facing
configuration.

Existing CUDA-array-interface statevector support remains the public
device-resident statevector path for this campaign. A dedicated
device-statevector wrapper would be a new public API and therefore requires the
full API review checklist before exposure.

## ROCm/HIP API Compatibility

ROCm/HIP public behavior includes:

```text
WOLFGANG_ENABLE_HIP build option
WOLFGANG_HIP_ARCHITECTURES source-build target list
WOLFGANG_ENABLE_CUDA=ON with WOLFGANG_ENABLE_HIP=ON configure-time rejection
PauliSum.to_device(device=...) in HIP-only source builds
DevicePauliSum.backend == "hip"
DevicePauliSum.to_host()
DevicePauliSum.commutes_with(...)
DevicePauliSum.commutes_with_device(..., output=None)
DeviceCommutationMatrix dense uint8 ownership, to_host(), count_commuting(), and conflict_degrees()
DevicePauliSum.simplify()
HIP availability errors and skips
synchronous transfer and kernel-return behavior
documented absence of HIP DLPack, CUDA Array Interface, public streams, public workspaces, expectation, matmul, multi-GPU, ROCm wheel, broad portability, and simultaneous CUDA+HIP support
```

Do not change ROCm/HIP synchronization, ownership, backend reporting,
source-build flag behavior, or unsupported-surface errors without updating
`docs/architecture/rocm_backend.md`, tests, and user documentation.

Target-specific backend-neutral object-model changes must also update
`docs/architecture/backend_neutral_accelerators.md` and include compatibility
notes for:

```text
DevicePauliSum.backend
device ordinal behavior
cross-backend operation errors
_accelerator_status()
packaging extras and build flags
```

Until those compatibility notes and tests exist, simultaneous CUDA+HIP source
builds remain unavailable. Under the current target-specific policy, they are
not required for CUDA-target, HIP-target, or Apple Metal-target support.

The first ROCm/HIP public surface is source-build-only MI300X `gfx942`
evidence. It is not a binary wheel promise and not a support claim for every
AMD GPU.

Campaign 2 promotes HIP `DeviceCommutationMatrix` and
`DevicePauliSum.commutes_with_device()` for HIP-only builds. Compatibility for
this experimental surface covers shape, dtype, row-major ordering, 1/0 flag
semantics, same-device validation, synchronous return behavior, dense
`to_host()`, compact `count_commuting(axis=None|0|1)`, and compact
`conflict_degrees(axis=None|0|1)`.

Campaign 3 promotes HIP `DevicePauliSum.simplify()`. Compatibility covers
device-resident output, canonical ordering, coefficient summation,
empty/all-zero handling, inclusive tolerance filtering, invalid-tolerance
exception category, and explicit host materialization only through
`to_host()`.

ROCm Campaign 4 private simplify-hardening hooks are excluded from the public
API compatibility promise. Private HIP workspace, scratch-buffer, packed-key,
generic-reduction, and `FASTPAULI_HIP_BENCH_` experiment selectors must remain
benchmark or test harness details unless a separate API review promotes them.

## Compatibility Matrix

Phase 1 should record the initial supported matrix in project metadata and docs:

```text
Python >= 3.10
C++20
CMake >= 3.24
Ninja >= 1.10
CUDA disabled by default
hardware target claims follow docs/architecture/hardware_targets_and_testing.md
```

As support is validated, update:

```text
pyproject.toml
README.md
docs/user/installation.md when it exists
CI matrix
release notes
```

Do not claim platform, Python, compiler, CUDA, Qiskit, or OpenFermion support unless it is validated locally or in CI.

Do not claim CPU feature, SIMD, oneTBB, CUDA toolkit, CUDA architecture, Apple
Metal backend, MPSGraph adjunct, or wheel-platform support unless it satisfies
the evidence vocabulary in `docs/architecture/hardware_targets_and_testing.md`
and the Apple-specific boundary in `docs/architecture/apple_accelerator.md`
when Apple GPU behavior is involved.

## API Review Checklist

Before merging a public API change:

```text
is the API necessary for the current phase?
does it preserve semantic contracts?
does it expose backend behavior explicitly?
does it document ownership, ordering, dtype, and exceptions?
does it avoid import-time optional dependency requirements?
does it have tests for expected and invalid inputs?
does it require user-facing docs or migration notes?
does it affect benchmark or report compatibility?
has an API-focused review been completed for public API changes?
```

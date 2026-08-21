# Apple Metal Accelerator Architecture

This document defines the accepted architecture contract for the Wolfgang
Apple Silicon GPU target. Wolfgang now has a source-build-only Metal lane, but
Apple GPU wheel support and generic Apple GPU support claims remain unavailable.
Runtime support requires validation in a process where Metal exposes a usable
device.

## Decision

Wolfgang implements Apple Silicon GPU support as a target-specific Metal
source-build backend after the CPU, CUDA, ROCm/HIP, release-foundation, and
backend-neutral object-model work already in the repository.

The backend identity for this target is:

```text
metal
```

The source-build flag for this target is:

```text
WOLFGANG_ENABLE_METAL=ON
```

The normal build matrix remains target-specific:

```text
CPU-only default package
CUDA-target source build
HIP-target source build
Metal-target source build
```

`WOLFGANG_ENABLE_METAL=ON` must initially be mutually exclusive with CUDA and
HIP. A mixed CUDA+HIP+Metal runtime is not a goal for the Apple bring-up lane.

## Backend Naming

The public backend selector uses `"metal"` when the backend is compiled:

```python
op.to_device(backend="metal")
DeviceCommutationMatrix.empty(..., backend="metal")
```

Wolfgang should not expose `"mps"` or `"mpsgraph"` as object backend names.
Metal is the memory ownership and command-submission boundary. Metal
Performance Shaders and Metal Performance Shaders Graph may be implementation
helpers or benchmark baselines, but they do not define separate Wolfgang
device-object identities.

CPU-only, CUDA-only, and HIP-only builds reject explicit `"metal"` requests with
rebuild guidance.

## Technology Choice

The primary implementation path is custom Metal compute kernels over
Wolfgang-owned `MTLBuffer` storage, reached from C++ through either
Objective-C++ translation units or Apple's `metal-cpp` interface. This matches
Wolfgang's current C++20 extension architecture and keeps sparse Pauli
bit-parity, duplicate-reduction, and complex-accumulation kernels under direct
control.

Metal Performance Shaders is allowed only where its matrix or reduction
primitives map directly to a Wolfgang operation without changing semantics or
ownership. Metal Performance Shaders Graph is allowed as an evaluated adjunct
for dense tensor-style workloads or external baselines. It is not the default
bring-up path for sparse packed Pauli kernels because the first kernels need
explicit control over packed integer buffers, command buffers, and intermediate
lifetimes.

PyTorch `mps` is a useful external baseline for tensor workloads that can be
mapped exactly. It is not a Wolfgang backend and must not be used to implement
Wolfgang device objects.

## Public Header Boundary

CPU-only public headers must not include Metal, Foundation, Objective-C,
Objective-C++, MPS, or MPSGraph headers. Apple-specific declarations belong in
private source files or private headers under `src/metal/`.

The Python extension may expose Metal status through internal bindings only
after CPU-only imports on machines without Metal remain clean and deterministic.

## Ownership And Lifetime

A `DevicePauliSum` with backend `"metal"` owns:

```text
packed x buffers in MTLBuffer storage
packed z buffers in MTLBuffer storage
complex coefficient buffers in MTLBuffer storage
one MTLDevice
one MTLCommandQueue
one backend-local device ordinal
```

A `DeviceCommutationMatrix` with backend `"metal"` owns:

```text
dense row-major uint8 flag buffer in MTLBuffer storage
one MTLDevice
one MTLCommandQueue
one backend-local device ordinal
```

Initial implementation should use synchronous command-buffer completion for
public methods. Public async, user-provided command queue, command-buffer,
event, heap, or workspace APIs require a separate API plan.

Even before a public workspace API exists, private reusable accelerator scratch
and output buffers follow the cross-backend ownership/lifetime contract:

```text
private reusable accelerator scratch and output buffers remain move-only
workspace storage is tied to the same backend-local device ordinal as every operand it serves
reset retains the allocation for reuse
release returns the allocation to the runtime
capacity growth is monotonic until reset or release
must not expose raw device pointers or framework objects through the public API
```

## Memory Policy

The first Metal bring-up should use `MTLResourceStorageModeShared` on Apple
Silicon for transfer correctness, simple host materialization, and easier
debugging. Private storage, blit staging, heaps, and buffer reuse are
optimization topics that require A/B evidence.

Every benchmark report must label:

```text
storage mode
transfer-inclusive timing
device-resident timing
host materialization timing
buffer allocation and reuse boundary
command-buffer synchronization boundary
```

Wave 1D adds a retained commutation reuse evidence gate for local Apple-host
Metal prestaging. The benchmark and report must name the same-boundary
comparison explicitly rather than implying a single speedup number:

```text
retained reused-output boundary: device_output_reused
allocating boundary: device_output_allocating
transfer-inclusive boundary: transfer_inclusive
promotion metric: mean-of-medians across 3 independent reruns
small-row guard: reject and investigate any undocumented >5% reused-output regression versus the allocating boundary
```

The initial public API must not export raw Metal buffers, DLPack capsules,
PyTorch `mps` tensors, or any pretend CUDA Array Interface object. Interop is
future work until a real consumer contract prevents mutation hazards and
documents lifetime ownership.

## Status Reporting

Metal builds must add a `_metal_status()` internal binding
and extend `_accelerator_status()` and `_build_info()` with structured state:

```text
accelerator_build_mode == "metal_only"
compiled_accelerator_backends includes "metal"
runtime_visible_accelerator_backends includes "metal" only when a usable Metal device exists
compiled_backends includes "cpu" and "metal"
runtime_visible_backends includes "cpu" and optionally "metal"
metal runtime availability
macOS version
Xcode or Command Line Tools version
Metal device name
Metal feature family or capability summary
storage mode policy
```

Status checks must be safe on CPU-only machines and on macOS hosts without a
usable Metal device.

## Initial Operation Order

The first Apple accelerator implementation campaign should use this order:

```text
1. CPU-only safety and Metal source-build configuration
2. Metal runtime status and build metadata
3. PauliSum.to_device(backend="metal") and DevicePauliSum.to_host()
4. empty DevicePauliSum and DeviceCommutationMatrix cases
5. Metal pairwise commutation with CPU equivalence tests
6. DeviceCommutationMatrix.count_commuting(axis=None|0|1)
7. DeviceCommutationMatrix.conflict_degrees(axis=None|0|1)
8. Metal simplify only after commutation and compact consumers are correct
9. Metal statevector expectation only after transfer and synchronization boundaries are measured
10. Metal matmul only after simplify and expectation evidence exists
```

This order keeps correctness, ownership, and compact consumers ahead of more
complex kernels.

## Current Evidence

The first implementation report is
`docs/benchmarks/reports/apple_metal_bringup_2026-05-01.md`. It records:

```text
WOLFGANG_ENABLE_METAL=ON source build: passed
CPU-only source build and import safety: passed
Metal status/build metadata: implemented
transfer and pairwise commutation source code: implemented
runtime equivalence tests: passed under elevated Codex execution on Apple M4 Pro
non-elevated Codex sandbox: still exposes no device to MTLCreateSystemDefaultDevice()
Metal System Trace evidence: captured with Wolfgang python process present
profiler tooling remaining work: GPU counter profile and shader timeline require a deeper Instruments capture
```

The same host reports an Apple M4 Pro GPU with Metal support through
`system_profiler`. Metal runtime visibility therefore depends on the process
sandbox in this local Codex environment rather than on GPU utilization or a
source-build failure. Full Xcode, the downloadable Metal Toolchain component,
and `xctrace` are now available for source-build validation. The checked trace
summary records Metal command-buffer and GPU schema availability while omitting
device UUID, host display name, full process inventory, and local absolute
paths.

The first optimization and evidence refresh is
`docs/benchmarks/reports/apple_metal_optimization_campaign1_2026-05-05.md`,
planned in `docs/plans/apple_metal_optimization_campaign1_plan.md`. It expands
benchmark coverage beyond the original smoke case and refreshes the broad
README performance landscape with local Apple M4 Pro Metal rows.

The second optimization campaign is
`docs/benchmarks/reports/apple_metal_optimization_campaign2_2026-05-05.md`,
planned in `docs/plans/apple_metal_optimization_campaign2_plan.md`. It retains
the public source-build-only API boundary while replacing the single generic
flat-index commutation kernel with a two-dimensional dispatch lane that records
`[rhs_terms, lhs_terms, 1]` in benchmark metadata. Campaign 2 retains the
one-word specialized kernel as the default for one packed word, uses the
generic 2D kernel for two-word and larger inputs, and keeps the two-word
specialized kernel, legacy flat generic kernel, and explicit forced generic
rows as benchmark-only A/B candidates. Those selectors are not public APIs and
must not be used for support claims outside checked benchmark reports.

The third optimization campaign is
`docs/benchmarks/reports/apple_metal_optimization_campaign3_2026-05-06.md`,
planned in `docs/plans/apple_metal_optimization_campaign3_plan.md`. It adds
benchmark-only selectors for offline `.metallib` pipeline loading, private
output storage plus shared blit staging, and GPU compact-consumer reductions.
Campaign 3 keeps those paths as evidence tools only: the retained defaults
remain runtime source pipelines, shared storage for host-output commutation,
generic 2D for words >= 2, and CPU scans for compact consumers until broader
Apple Silicon evidence justifies a policy change.

The fourth optimization campaign is
`docs/benchmarks/reports/apple_metal_optimization_campaign4_2026-05-06.md`,
planned in `docs/plans/apple_metal_optimization_campaign4_plan.md`. It adds
larger Apple M4 Pro evidence for two-word commutation, larger compact-consumer
matrices, and a benchmark-only `gpu_parallel_total` compact-consumer selector
for `fp_count_commuting_total_block_sums`. Campaign 4 keeps two-word
specialization, private storage, offline `.metallib`, and compact GPU
reductions as evidence tools only. It also explicitly leaves PyPI publication,
Windows support, and older macOS compatibility outside this Apple Metal
optimization lane.

The fifth optimization campaign is
`docs/benchmarks/reports/apple_metal_optimization_campaign5_2026-05-06.md`,
planned in `docs/plans/apple_metal_optimization_campaign5_plan.md`. It is the
first Metal simplify bring-up slice after commutation and compact consumers.
It keeps the source-build-only backend boundary and implements
`DevicePauliSum.simplify(atol, rtol)` by transferring the Metal object to a
host `PauliSum`, running CPU simplify, and returning a Metal `DevicePauliSum`.
That retained path must be labeled `metal_simplify_transfer_reference` with
the `device_to_host_cpu_simplify_host_to_device` boundary. It is a correctness
bridge, not a device-resident GPU duplicate-reduction speedup.

The sixth optimization campaign is
`docs/benchmarks/reports/apple_metal_optimization_campaign6_2026-05-07.md`,
planned in `docs/plans/apple_metal_optimization_campaign6_plan.md`. It is
Apple Metal Campaign 6 device-resident simplify groundwork, not a public API
expansion. Campaign 6 keeps `DevicePauliSum.simplify(atol, rtol)` on the
Campaign 5 transfer-reference path while adding a private `MetalWorkspace`
scratch model, `WorkspaceTimingMode`, and `metal_simplify_workspace_probe`
benchmark rows. Those rows use the `status_only` boundary and must state that
the device-resident simplify candidate remains blocked until checked Metal
sort/prefix/reduce primitives exist. A future retained device-resident simplify
path must first provide correct Metal sort, prefix-sum, and reduce-by-key
primitives, prove deterministic canonical output order, and record
`device_resident` timing separately from transfer-reference timing.

The seventh optimization campaign is
`docs/benchmarks/reports/apple_metal_optimization_campaign7_2026-05-07.md`,
planned in `docs/plans/apple_metal_optimization_campaign7_plan.md`. It is
Apple Metal Campaign 7 checked device-resident simplify primitive stack work,
not a public API expansion. Campaign 7 adds a private benchmark-only one-word
Metal path with bitonic packed-key sort, uint32 prefix-sum, reduce-by-key, and
survivor compaction. The row identity is `metal_simplify_device_candidate` and
the boundary may be `device_resident` only when materialized candidate output
matches CPU simplify. Because this Apple Metal toolchain rejects `double`
arithmetic in kernels, the Campaign 7 candidate is limited to coefficients
exactly representable as signed fixed32 dyadic values whose accumulated sums and
tolerance threshold fit exact uint64 squared-magnitude comparison. Public
`DevicePauliSum.simplify(atol, rtol)` remains the Campaign 5 transfer-reference
path.

The eighth optimization campaign is
`docs/benchmarks/reports/apple_metal_optimization_campaign8_2026-05-07.md`,
planned in `docs/plans/apple_metal_optimization_campaign8_plan.md`. It is
Apple Metal Campaign 8 simplify performance-relevance evidence, not a public
API expansion. Campaign 8 rows include `timing_decomposition_seconds`,
`pipeline_cache`, `dispatch_counts`, and `performance_decision` metadata so the
report's timing decomposition can separate host preflight, scratch/output
allocation, command encoding, command execution, and output accounting. The
public `DevicePauliSum.simplify(atol, rtol)` remains the transfer-reference bridge
unless a later design proves a broader, correct, device-resident implementation
that beats same-host CPU simplify and the transfer-reference path.

The Wave 1D cross-backend prestage report is
`docs/benchmarks/reports/apple_metal_wave1d_2026-08-21.md`. It records the
required reused-output evidence gate for pairwise commutation on the local Apple
host: three independent reruns per case, mean-of-medians promotion metric, and
same-boundary comparisons between `device_output_reused`,
`device_output_allocating`, and `transfer_inclusive`. The checked Wave 1D rows
stay private to benchmark/report evidence and do not expand the public Metal
surface.

## Testing Ladder

Required local Apple Silicon validation before any support claim:

```text
CPU-only build and import on the same machine with WOLFGANG_ENABLE_METAL=OFF
source build with WOLFGANG_ENABLE_METAL=ON
runtime status on a named Apple Silicon SoC
transfer round trips for empty, one-term, multi-word, and duplicate-heavy operators
same-operation CPU/Metal equivalence tests
forced unsupported-backend errors on non-Metal builds
configure-time rejection for unsupported accelerator combinations
benchmark smoke with correctness checks
profiler or explicit profiler-blocker evidence
```

CI may run documentation, CPU-only, and configure-time safety checks. CI must
not claim Apple GPU support unless the runner exposes a real Metal device and
the release evidence records that device.

## Benchmark And Profiling Policy

Metal benchmark reports must follow `docs/benchmarks/protocol.md`. At minimum,
reports must include:

```text
Apple SoC name
GPU core count when available
macOS version
Xcode or Command Line Tools version
Metal device name
storage mode
threadgroup size and grid shape
selected Metal kernel name when multiple internal kernels implement the same public operation
CPU scalar, CPU default, and NEON baselines on the same host
transfer-inclusive and device-resident timings
host materialization timing when results leave the GPU
correctness checks and tolerance
profiler evidence from Xcode Instruments, Metal System Trace, xctrace, or a precise tooling blocker
```

Do not compare Metal timings against H100, A100, RTX PRO 6000 Blackwell, or
MI300X timings as a universal speedup claim. Cross-accelerator plots may include
Metal only as a labeled hardware-specific row with the same workload and timing
boundary.

## Packaging And Release Boundary

Metal support is source-build-only until a dedicated packaging plan accepts a
macOS arm64 accelerator wheel channel. CPU wheels must remain Metal-free unless
a later release plan proves that linking Apple frameworks into the wheel does
not change CPU-only import reliability or support claims.

Metal wheel support requires:

```text
macOS deployment target policy
Apple Silicon architecture policy
Xcode or Command Line Tools build policy
framework linkage policy for Metal, Foundation, MetalPerformanceShaders, and MetalPerformanceShadersGraph
clean-machine install and import evidence
runtime behavior on machines without a usable Metal device
support wording that distinguishes CPU wheels from Metal source builds
```

## External References

Refresh these references before changing this architecture:

```text
Apple Metal MTLDevice: https://developer.apple.com/documentation/metal/mtldevice
Apple Metal MTLBuffer: https://developer.apple.com/documentation/metal/mtlbuffer
Apple Metal MTLCommandBuffer: https://developer.apple.com/documentation/metal/mtlcommandbuffer
Apple metal-cpp: https://developer.apple.com/metal/cpp/
Apple Metal Performance Shaders: https://developer.apple.com/documentation/metalperformanceshaders
Apple Metal Performance Shaders Graph: https://developer.apple.com/documentation/metalperformanceshadersgraph
PyTorch MPS backend: https://docs.pytorch.org/docs/stable/notes/mps
```

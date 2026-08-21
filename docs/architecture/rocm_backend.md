# ROCm/HIP Backend Architecture

This document defines the first Wolfgang ROCm/HIP backend contract. It is a
source-build accelerator contract, not a release-wheel support claim.

## Scope And Non-Goals

ROCm/HIP is the second GPU backend after the CUDA foundation. The first
validated target is a single AMD Instinct MI300X exposed as LLVM target
`gfx942`.

In scope:

```text
source builds with ROCm/HIP enabled explicitly
runtime status reporting that works with and without a visible AMD GPU
host/device PauliSum transfers
the first retained HIP correctness kernel
HIP device-resident simplify on MI300X
HIP host-statevector expectation on MI300X
HIP matrix-product generation on MI300X
MI300X benchmark and profiling evidence
```

Out of scope:

```text
ROCm wheels
multi-GPU MI300X work
distributed ROCm communication
Metal/MPS
simultaneous CUDA and HIP runtime objects
public stream, graph, or external workspace APIs
```

## Build Flags And Source-Build Policy

HIP support is disabled by default:

```text
WOLFGANG_ENABLE_HIP=OFF
```

The first MI300X source-build target is:

```text
WOLFGANG_ENABLE_HIP=ON
WOLFGANG_HIP_ARCHITECTURES=gfx942
```

`WOLFGANG_ENABLE_HIP=ON` is a source-build path. It must not imply binary wheel
support, portable ROCm runtime availability, or support for every AMD GPU.

## Public API Compatibility

Existing CPU-only and CUDA builds keep their current public behavior. CPU-only
imports must not require ROCm libraries, ROCm devices, or ROCm environment
variables.

For the first MI300X campaign, HIP-only builds may reuse the existing
`PauliSum.to_device(device=0)` and `DevicePauliSum` public surface. That surface
must report its backend as `"hip"` when the object is HIP-backed and `"cuda"`
when the object is CUDA-backed.

## Target-Specific CUDA/HIP Build Rule

ROCm/HIP source builds remain target-specific and mutually exclusive with CUDA:

```text
WOLFGANG_ENABLE_CUDA=ON and WOLFGANG_ENABLE_HIP=ON is a configure-time error
```

This is now a deliberate packaging and validation boundary rather than a
temporary hardware-access blocker. Wolfgang shares backend-neutral API
semantics across CPU, CUDA-target, HIP-target, and Apple Metal-target builds,
but normal builds link at most one accelerator runtime. Simultaneous CUDA/HIP
builds require a separate accepted mixed-runtime architecture decision.

## Runtime Status Schema

The Python extension exposes separate status functions:

```text
_cuda_status()
_hip_status()
_accelerator_status()
```

`_hip_status()` reports:

```text
built
runtime_available
device_count
skip_reason
runtime_version
driver_version
toolkit_version when available
devices
```

Each HIP device entry reports:

```text
ordinal
name
gfx_target
total_memory_bytes
```

`_accelerator_status()` reports the active accelerator backend for the current
build:

```text
none
cuda
hip
```

Status calls must be safe on systems without ROCm devices. Missing or unusable
HIP runtime state is reported as `runtime_available=False` with a clear
`skip_reason`.

## Memory Ownership And Lifetime

HIP device objects own their allocations through RAII. Public headers must not
include ROCm or HIP headers. Implementation files under `src/hip/` may include
HIP runtime headers.

HIP allocations owned by `DevicePauliSum` include:

```text
packed x words
packed z words
complex coefficients
```

Moved-from objects are invalid. Methods on moved-from HIP objects must raise a
deterministic exception rather than dereferencing null device state.

Private reusable accelerator scratch and output buffers remain unavailable as
public API, but the internal ownership/lifetime contract is fixed for future
HIP prestaging:

```text
private reusable accelerator scratch and output buffers remain move-only
workspace storage is tied to the same backend-local device ordinal as every operand it serves
reset retains the allocation for reuse
release returns the allocation to the runtime
capacity growth is monotonic until reset or release
must not expose raw device pointers or framework objects through the public API
```

## Transfer Semantics

`PauliSum.to_device(device=0)` copies host packed storage to the selected HIP
device in HIP-only builds. `DevicePauliSum.to_host()` copies all owned device
storage back into a host `PauliSum`.

Empty operators are valid. Empty operators do not require non-null device
allocations, and round-tripping an empty operator preserves `num_qubits` and
zero terms.

Transfers are synchronous in the first campaign. Public asynchronous transfer or
stream ownership is out of scope.

## Kernel Implementation Rules

The first retained HIP kernel is pairwise commutation because it is bitwise,
maps directly to Wolfgang packed-word storage, and avoids a sorting or rocThrust
dependency during initial bring-up.

HIP kernels must:

```text
validate devices, qubit counts, sizes, and moved-from state before launch
translate HIP errors into runtime_error messages naming the failed operation
synchronize before returning host-visible results
preserve CPU scalar correctness exactly
avoid ROCm dependencies in CPU-only builds
```

## Post-Bring-Up Public Boundaries

The first accepted post-bring-up public HIP expansion is device-resident dense
commutation output. It was implemented and benchmarked by
`docs/plans/mi300x_rocm_optimization_campaign2_plan.md`, with checked evidence
in `docs/benchmarks/reports/rocm_mi300x_campaign2_2026-04-30.md`.

HIP `DeviceCommutationMatrix` must match the CUDA shape, dtype, device,
`to_host()`, `count_commuting(axis=None|0|1)`, and
`conflict_degrees(axis=None|0|1)` semantics where the backend supports those
operations. Differences must be documented before exposure and covered by HIP
tests.

HIP `DeviceCommutationMatrix` must not expose CUDA Array Interface semantics.
A HIP pointer returned under `__cuda_array_interface__` would be a correctness
bug because consumers would treat the pointer as CUDA memory.

HIP DLPack remains unavailable until a separate HIP DLPack contract accepts:

```text
kDLROCM device typing
producer and consumer ownership
read-only versus mutable export behavior
stream and synchronization semantics
consumer support for PyTorch ROCm, CuPy ROCm, or another named library
correctness and benchmark evidence for each retained consumer
```

The broader ROCm sequence is defined in
`docs/plans/rocm_next_waves_plan.md`.

The accepted Campaign 3 ROCm boundary is HIP `DevicePauliSum.simplify()`,
defined in `docs/plans/mi300x_rocm_optimization_campaign3_plan.md` and
reported in
`docs/benchmarks/reports/rocm_mi300x_campaign3_2026-04-30.md`. Campaign 3
also assigns a terminal status to each Campaign 2 residual item: HIP DLPack,
public streams, public workspaces, packed summaries, HIP expectation, HIP
matmul, ROCm portability, ROCm wheels, multi-GPU MI300X, and simultaneous
CUDA+HIP source builds.

## Campaign 3 Simplify Boundary

The Campaign 3 accepted HIP public operation is `DevicePauliSum.simplify()`.
HIP simplify returns a HIP-backed `DevicePauliSum` and must not implicitly copy
the simplified operator to host.

HIP simplify must match `PauliSum.simplify()` canonical ordering, coefficient
summation, and tolerance filtering. That includes empty input handling,
all-zero-output preservation of `num_qubits`, inclusive absolute/relative
tolerance thresholds, and the same invalid-tolerance public error class as the
host path where applicable.

rocThrust is the retained duplicate-reduction implementation path for Campaign
3. hipCUB and custom duplicate-reduction probes remain unavailable or rejected
until they pass CPU/HIP equivalence, allocation accounting, and rocprof
evidence in a separate campaign.

HIP DLPack, public streams, public workspaces, multi-GPU, ROCm wheels,
additional AMD GPU support claims, and simultaneous CUDA+HIP builds remain
unavailable in Campaign 3. HIP expectation and HIP matmul also remained
unavailable in Campaign 3 and were later promoted by Campaign 6. The Campaign 3
report must record terminal statuses and next triggers for those surfaces, but
accepting any of them requires a separate follow-on plan or architecture
decision.

## Campaign 4 Simplify Hardening Outcome

Campaign 4 is complete and remains a private HIP simplify hardening campaign,
not a public API expansion. The executable plan is
`docs/plans/mi300x_rocm_optimization_campaign4_plan.md`; the checked report is
`docs/benchmarks/reports/rocm_mi300x_campaign4_2026-04-30.md`.

Campaign 4 changed only private `DevicePauliSum.simplify()` implementation
details after MI300X correctness, benchmark, and profiler evidence. The
retained change is the parallel generic multi-word sorted-index
`reduce_by_key` path. The serial generic reducer remains private
benchmark-only fallback. Custom packed-key duplicate-reduction probes are
unavailable because no distinct lower-level implementation is retained or
timed, and rocPRIM/hipCUB scratch workspace probes are unavailable for the
current rocThrust boundary. These mechanisms stay private unless a
separate public API decision accepts their lifetime, ownership,
synchronization, and documentation contracts.

Campaign 4 must preserve:

```text
DevicePauliSum.simplify() returns a HIP-backed DevicePauliSum
simplify remains synchronous
canonical ordering, coefficient summation, and tolerance filtering match PauliSum.simplify()
public headers include no HIP or ROCm runtime headers
CPU-only and CUDA-only builds behave unchanged
WOLFGANG_ENABLE_CUDA=ON with WOLFGANG_ENABLE_HIP=ON remains a configure-time error
```

Public HIP DLPack, public streams, public workspaces, multi-GPU execution,
ROCm wheels, additional AMD GPU support claims, and simultaneous CUDA+HIP
builds remain unavailable in Campaign 4. HIP expectation and HIP matmul remained
unavailable in Campaign 4 and were later promoted by Campaign 6.

## Campaign 5 Interop And Execution-Control Outcome

Campaign 5 is complete in
`docs/benchmarks/reports/rocm_mi300x_campaign5_2026-04-30.md`. It was the first
ROCm wave allowed to change HIP Python interop behavior after resident
`DeviceCommutationMatrix` output and HIP simplify were retained.

Campaign 5 rejected HIP DLPack retention. A candidate versioned read-only
`DeviceCommutationMatrix` `uint8` export with DLPack `kDLROCM` device typing was
consumed by PyTorch ROCm on MI300X in a temporary candidate probe, but PyTorch
accepted mutation of the imported view. That violates Wolfgang's read-only
export contract, so HIP
`DeviceCommutationMatrix.__dlpack__` and `__dlpack_device__` remain
unavailable. HIP `__cuda_array_interface__` remains unavailable because HIP
device pointers must not be presented as CUDA memory.

Public HIP streams, graph execution, and workspaces remain unavailable because
Campaign 5 did not accept complete public API contracts, ownership and lifetime
rules, synchronization and error-propagation behavior, or measured acceptance
evidence. ROCm wheels, multi-GPU ROCm, additional AMD portability claims, and
simultaneous CUDA+HIP source builds remain separate campaigns. HIP expectation
and HIP matmul were later promoted by Campaign 6.

## Campaign 6 Expectation And Matmul Parity Boundary

Campaign 6 is complete in
`docs/benchmarks/reports/rocm_mi300x_campaign6_2026-04-30.md` and was executed
from `docs/plans/mi300x_rocm_optimization_campaign6_plan.md`. It promoted the
existing `DevicePauliSum.expectation_statevector()` and
`DevicePauliSum.matmul()` methods from HIP-unavailable to HIP-supported on the
MI300X `gfx942` source-build lane without adding new public Python methods or
arguments.

Campaign 6 HIP expectation is limited to host NumPy `complex64` and
`complex128` statevectors. External HIP device-pointer imports, DLPack
statevectors, CUDA Array Interface inputs, public stream arguments, and public
workspace arguments are not part of this boundary.

Campaign 6 HIP matmul returns a HIP-backed `DevicePauliSum`, enforces
`max_intermediate_terms` before allocation, preserves `simplify=False`
nested-loop product ordering, and uses retained HIP simplify for
`simplify=True`.

Campaign 6 preserves:

```text
public headers include no HIP or ROCm runtime headers
CPU-only and CUDA-only builds behave unchanged
WOLFGANG_ENABLE_CUDA=ON with WOLFGANG_ENABLE_HIP=ON remains a configure-time error
HIP DLPack remains unavailable after the Campaign 5 read-only consumer rejection
HIP CUDA Array Interface remains unavailable
public streams, graphs, and workspaces remain unavailable
ROCm wheels, multi-GPU ROCm, additional AMD support claims, and simultaneous CUDA+HIP remain separate campaigns
```

## Campaign 7 Release-Support Boundary

Campaign 7 is complete in
`docs/benchmarks/reports/rocm_mi300x_campaign7_2026-04-30.md` and was executed
from `docs/plans/mi300x_rocm_optimization_campaign7_plan.md`. It is a
release-support and portability-boundary campaign, not a new kernel or public
API campaign.

Campaign 7 retains:

```text
repeatable MI300X source-build release lane for WOLFGANG_ENABLE_HIP=ON and WOLFGANG_HIP_ARCHITECTURES=gfx942
ROCm release-runbook or release-lane script
benchmark and profiler schema for retained HIP operations
ROCm packaging policy that keeps wheels unavailable
README support wording that separates source-build evidence from wheel support and broad AMD portability claims
terminal statuses for portability, multi-GPU, and backend-neutral accelerator work
```

Campaign 7 does not retain:

```text
external HIP statevector device pointers
HIP DLPack
HIP CUDA Array Interface
public streams
public graph replay
public workspace handles
ROCm wheels
multi-GPU ROCm
simultaneous CUDA+HIP source builds
additional AMD GPU support claims without source-build and runtime evidence on that GPU
```

Additional AMD GPU support now requires all of:

```text
source build on that GPU architecture
runtime status capture
retained HIP operation tests
benchmark smoke
profiler availability status
README wording that distinguishes runtime-tested from performance-tested and release-supported
```

## Campaign 8 Architecture-Readiness Boundary

Campaign 8 is complete in
`docs/benchmarks/reports/rocm_campaign8_architecture_readiness_2026-05-01.md`
and was executed from `docs/plans/mi300x_rocm_optimization_campaign8_plan.md`.
It is a Wave 6 architecture-readiness campaign, not a new HIP kernel, wheel,
public API, or multi-GPU implementation campaign.

The backend-neutral CUDA/HIP object-model gate is
`docs/architecture/backend_neutral_accelerators.md`. Current source builds
remain target-specific CUDA-only or HIP-only under the policy validated by
`docs/plans/backend_neutral_accelerator_campaign9_plan.md`.

Campaign 8 defined gates for:

```text
backend-neutral accelerator object model
simultaneous CUDA+HIP source builds
multi-GPU ROCm execution
non-MI300X AMD portability claims
ROCm wheel packaging
rocprofv3 migration
external HIP statevector and HIP DLPack reconsideration
targeted ROCm performance reopening after profiler-backed bottleneck evidence
```

Backend-Neutral Accelerator Campaign 9 is complete for the target-specific
backend-neutral API gate, with report evidence in
`docs/benchmarks/reports/backend_neutral_accelerator_campaign9_2026-05-01.md`.
It covers the shared accelerator
status schema, object-local backend identity, explicit backend selectors for
initial device construction, omitted-backend ambiguity policy,
`DeviceCommutationMatrix.backend`, disjoint target source sets, mixed CUDA+HIP
configure-time rejection, same-backend same-device validation, CPU-only header
safety, CUDA-target and HIP-target regression lanes, and benchmark-boundary
reporting. It is not a ROCm wheel, non-MI300X AMD portability, HIP DLPack,
multi-GPU ROCm, Metal/MPS, mixed-runtime, or combined accelerator wheel support
claim.

Campaign 9 validation now includes local CPU-only, H100 CUDA-only, MI300X
HIP-only, and CUDA+HIP configure-rejection lanes. HIP source builds can use
`PauliSum.to_device(backend="hip")` and
`DeviceCommutationMatrix.empty(..., backend="hip")`, while CPU-only and
CUDA-only builds reject explicit HIP selection before allocation. Simultaneous
CUDA+HIP source builds remain unavailable by target-specific policy unless a
future accepted mixed-runtime plan reopens the design.

The Campaign 8 profiler migration decision is
`docs/plans/rocm_profiler_migration_campaign8_decision.md`. The Campaign 8 HIP
interop reconsideration decision is
`docs/plans/rocm_hip_interop_reconsideration_campaign8_decision.md`.

Until those gates are accepted and later implementation campaigns satisfy them,
the Campaign 7 support boundary remains unchanged: ROCm/HIP is source-build-only,
MI300X `gfx942` is the only runtime-tested AMD GPU claim, CUDA and HIP remain
mutually exclusive build modes, HIP DLPack remains unavailable, ROCm wheels
remain unavailable, public streams/graphs/workspaces remain unavailable, and
multi-GPU ROCm execution remains unavailable.

## Error Handling And Synchronization

HIP API failures are translated at the boundary where they occur. Error messages
must identify the operation, such as `hipMalloc device x words` or
`hipMemcpy device coefficients`.

First-campaign HIP operations synchronize before returning. A future async API
requires a separate lifetime and stream ownership contract.

## Python Interop And DLPack Policy

The first HIP campaign does not expose HIP DLPack or ROCm array-interface
interop. Existing CUDA DLPack behavior remains a CUDA-only claim.

Any future HIP Python interop must document:

```text
producer and consumer ownership
stream or synchronization model
device compatibility checks
read-only versus mutable export behavior
consumer-library versions and benchmark semantics
```

## Testing Ladder

HIP implementation proceeds through this ladder:

```text
CPU-only import and validation with WOLFGANG_ENABLE_HIP=OFF
configure-time rejection of WOLFGANG_ENABLE_CUDA=ON with WOLFGANG_ENABLE_HIP=ON
HIP source build on MI300X with WOLFGANG_HIP_ARCHITECTURES=gfx942
_hip_status() and _accelerator_status() checks with and without runtime availability
non-empty and empty host/device round-trip tests
deterministic pairwise commutation equivalence tests
randomized pairwise commutation equivalence tests with fixed seeds
HIP DeviceCommutationMatrix device-output, reused-output, dense to_host, compact count, and compact conflict equivalence tests
HIP simplify edge-case, tolerance, randomized, one-word, two-word, and generic multi-word equivalence tests
Campaign 4 private strategy and workspace tests that prove no public header, Python API, or unsupported-surface leak
Campaign 6 HIP expectation and matmul parity tests for retained operations
benchmark smoke with correctness checks enabled
```

## Benchmark And Profiling Evidence

ROCm benchmark reports follow `docs/benchmarks/protocol.md`. They must separate:

```text
CPU scalar timing
available optimized CPU selector timing
HIP transfer-inclusive timing
HIP device-resident timing when the operation supports it
profiler timing and counter evidence when tooling permits
```

MI300X reports must capture:

```text
ROCm version
HIP compiler version
driver/runtime versions
GPU model
LLVM target
VRAM
power and clock state when available
topology when available
command lines
git revision
raw benchmark artifacts
profiler artifacts or blocked-profiler diagnosis
```

No ROCm performance claim is allowed without transfer-inclusive and
device-resident evidence, or an explicit reason why a device-resident boundary
does not apply to the measured operation.

## Release And Packaging Boundaries

ROCm/HIP source-build evidence does not imply ROCm wheel support. ROCm wheels,
ROCm CI runners, manylinux policy, bundled runtime policy, and supported GPU
matrix claims require a separate release-packaging campaign.

README support language must distinguish:

```text
CPU wheel support
CUDA source-build evidence
ROCm/HIP source-build evidence
unsupported or exploratory accelerator candidates
```

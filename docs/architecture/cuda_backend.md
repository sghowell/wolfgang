# Wolfgang CUDA Backend Architecture

CUDA is a required Wolfgang backend milestone. CPU correctness still comes first, but the host implementation must preserve layout and semantics that can be mirrored on device without redesign.

## Backend Policy

Wolfgang ships a CPU backend for every supported build. CUDA support is a required product milestone implemented behind an explicit build option:

```text
WOLFGANG_ENABLE_CUDA=ON
```

CPU wheels remain the default distribution. CUDA wheels are a separate release artifact after source-build CUDA support is stable.

CUDA is the first GPU backend. ROCm/HIP and Metal/MPS are not part of the CUDA milestone and must not delay CUDA implementation. Any post-CUDA GPU backend needs a separate architecture document or an explicit extension to `docs/architecture/hardware_targets_and_testing.md`.

Backend-neutral accelerator APIs are governed by
`docs/architecture/backend_neutral_accelerators.md`. Current builds remain
target-specific: CUDA-only or HIP-only. `WOLFGANG_ENABLE_CUDA=ON` with
`WOLFGANG_ENABLE_HIP=ON` is deliberately rejected under the target-specific
accelerator policy unless a future accepted mixed-runtime plan changes that
boundary. The implementation handoff is complete in
`docs/plans/backend_neutral_accelerator_campaign9_plan.md`, with closeout
evidence in
`docs/benchmarks/reports/backend_neutral_accelerator_campaign9_2026-05-01.md`.
Campaign 9 added the explicit construction selector used by CUDA source builds:
`PauliSum.to_device(backend="cuda")` and
`DeviceCommutationMatrix.empty(..., backend="cuda")`.

## Build Matrix

Initial source-build CUDA support follows `docs/architecture/hardware_targets_and_testing.md`. The short version is:

```text
C++ standard: C++20
CUDA toolkit: CUDA 12.9.x or the current CUDA 12.x line first
CMake: >= 3.24
Python: >= 3.10
Host compilers: platform compilers supported by the selected CUDA toolkit
GPU architectures: documented compile/runtime targets from the hardware target policy
```

The first CUDA implementation must build with `WOLFGANG_ENABLE_CUDA=OFF` and `WOLFGANG_ENABLE_CUDA=ON` from the same source tree. CPU-only builds must not include CUDA headers in public CPU-only headers.

CUDA 13.x is a forward-compatibility lane after CUDA 12.x source builds are stable. Do not move the baseline to CUDA 13.x without updating the Volta `sm_70` policy in `docs/architecture/hardware_targets_and_testing.md`.

## Host And Device Types

The host type remains:

```cpp
struct PauliSum {
    std::size_t num_qubits;
    std::size_t words;
    std::size_t num_terms;
    std::vector<std::uint64_t> x;
    std::vector<std::uint64_t> z;
    std::vector<std::complex<double>> coeffs;
};
```

The CUDA type is a separate owning device mirror:

```cpp
struct DevicePauliSum {
    std::size_t num_qubits;
    std::size_t words;
    std::size_t num_terms;
    std::uint64_t* x;
    std::uint64_t* z;
    thrust::complex<double>* coeffs;
    int device_ordinal;
};
```

The initial CUDA backend must not use CUDA unified memory. Transfers are explicit and measurable.

## Python API Shape

CPU API remains usable without CUDA:

```python
h = PauliSum.from_labels(["XX", "ZZ"])
h2 = h.simplify()
```

CUDA support adds a device-resident object:

```python
dh = h.to_device(device=0, backend="cuda")
dh2 = dh.simplify()
h2 = dh2.to_host()
```

Operations that can run on either backend accept a backend selector once CUDA lands:

```python
h.simplify(backend="cpu")
h.simplify(backend="cuda")
h.simplify(backend="auto")
```

`backend="auto"` may choose CUDA only when data transfer and input size heuristics justify it. `backend="cuda"` raises `RuntimeError` if Wolfgang was built without CUDA.

## Ownership And Lifetime

`DevicePauliSum` owns its device buffers. Copying a device object is disabled unless an explicit `.copy()` API is implemented. Moving a device object transfers ownership.

`to_device()` copies host buffers to device. `to_host()` copies device buffers to a new host `PauliSum`.

Device operations return new device objects unless the method name ends in `_inplace`. No in-place device operations are part of the first CUDA milestone.

Post-Phase 11 optimization may introduce an internal CUDA workspace object for
H100 benchmark experiments. That object is not public Python API and is not an
installed C++ API until a later API review promotes it. Internal workspace
experiments must preserve the existing `DevicePauliSum` ownership model:

```text
workspace storage is tied to one CUDA device ordinal
workspace reuse is allowed only when every operand lives on that same device
capacity growth is monotonic for a run unless an explicit reset or release path is used
temporary-storage reuse must not change canonical ordering, zero-tolerance filtering, or coefficient tolerance semantics
benchmark labels must say whether workspace allocation and growth are inside or outside the timed boundary
```

Campaign 4 implements this as a private source-only `CudaWorkspace` under
`src/cuda/`. The only Python visibility is the underscored
`_cuda_workspace_probe_for_testing()` hook, which returns lifetime snapshots for
tests and benchmark validation without exposing raw device pointers. Benchmark
workspace timing is selected with `WOLFGANG_CUDA_BENCH_WORKSPACE_MODE` values
`absent`, `grow_inside_timing`, and `pre_reserved_outside_timing`; these labels
are evidence boundaries, not public API modes.

If H100 evidence justifies a public workspace API, the API must be documented
first as experimental and must include absent-CUDA, wrong-device, moved-from,
capacity-growth, reset/release, and CPU-only import tests before it ships.

The public C++ `DevicePauliSum::commutes_with_into()` method is the supported
direct-fill variant for integrations that already own host output storage. It
fills exactly `lhs.num_terms() * rhs.num_terms()` writable host bytes in
row-major order, after applying the same dense-output guardrail as
`commutes_with()`. The method writes `1` for commuting pairs and `0` for
anti-commuting pairs, requires both operands to live on the same CUDA device,
and raises the same moved-from, qubit-count, device-mismatch, output-size, and
max-entry exceptions as the vector-returning API.

Campaign 5 adds the experimental public device-output form for GPU-resident
consumers. `DeviceCommutationMatrix` owns a dense row-major `uint8` flag matrix
on one CUDA device. `DevicePauliSum::commutes_with_device(rhs, max_entries)`
allocates and returns a new matrix with shape `(lhs.num_terms(), rhs.num_terms())`;
`DevicePauliSum::commutes_with_device_into(rhs, output, max_entries)` fills a
caller-provided `DeviceCommutationMatrix` and returns no C++ value. Python
exposes the same behavior as `DevicePauliSum.commutes_with_device(rhs,
output=None, max_commutation_matrix_entries=...)`, returning the newly allocated
matrix or the same `output` object passed by the caller.

The device-output contract is:

```text
dtype: uint8, with 1 for commuting pairs and 0 for anti-commuting pairs
layout: dense C-contiguous row-major matrix over lhs terms, then rhs terms
shape: exactly (lhs.num_terms, rhs.num_terms)
device: one CUDA device ordinal recorded by the owner
ownership: Wolfgang owns the allocation; external users may view but not free it
synchronization: default-stream synchronize-before-return
interop: __cuda_array_interface__ version >= 3, typestr "|u1", read/write data pointer, stream None or 1
host materialization: DeviceCommutationMatrix.to_host() copies and returns a NumPy bool matrix in Python
compact summaries: DeviceCommutationMatrix.count_commuting(axis=None|0|1) and conflict_degrees(axis=None|0|1) reduce on device and copy only uint64 counts to host
DLPack interop: DeviceCommutationMatrix.__dlpack__ exports a read-only dense uint8 view with owner lifetime retained by the capsule deleter context; max_version must be >= (1, 0), the capsule version is min(consumer max_version, producer-supported 1.1), and legacy unversioned capsules are rejected because they cannot carry read-only flags
guardrail: commutes_with_device enforces max_commutation_matrix_entries before allocating its own output or filling caller output
empty allocation: DeviceCommutationMatrix.empty(shape, device) validates shape and allocation overflow, but does not take a max-entry policy
errors: absent CUDA, runtime-unavailable CUDA, moved-from operands/output, wrong device, wrong shape, qubit mismatch, max-entry overflow, or allocation failure
```

This API is not a drop-in replacement for host-output timing claims. Benchmarks
and reports must separate device-output allocation, device-output reuse,
CUDA-array-interface export, compact-summary reductions, and `to_host()`
materialization. Public bit-packed commutation output remains rejected until a
consumer layout contract is accepted.

Campaign 6 retains compact summary counts as the first supported downstream GPU
consumer for `DeviceCommutationMatrix`. `count_commuting(axis=None)` returns a
Python integer total count, `axis=0` returns `uint64` column counts, and
`axis=1` returns `uint64` row counts. All reductions execute on the matrix CUDA
device and copy only the compact count result to host. This is still a
synchronous public API: calls use the default stream and return only after the
count result is available on the host.

Campaign 9 adds `conflict_degrees(axis=None|0|1)` as the anti-commuting
complement of `count_commuting(...)` and adds read-only DLPack export for
`DeviceCommutationMatrix`. The true `DevicePauliSum.group_commuting_device(...)`
surface remains rejected with evidence and is not a public API.

Campaign 7 evaluates fused downstream graph and grouping-oriented consumers on
top of the same dense `DeviceCommutationMatrix` layout. The accepted public API
status at campaign start is conservative: anti-commutation CSR graph
construction, conflict-degree summaries, and grouping summaries are
benchmark-only private helpers unless
`docs/plans/cuda_fused_commutation_consumer_api_review.md` later accepts exact
return semantics, ownership, ordering, synchronization, CPU-only behavior, and
failure modes. Benchmark rows must carry `private_benchmark_only` labels and
decision fields for count-specialization, bit-packed output, and non-H100
portability. No Campaign 7 fused-consumer helper belongs in an installed public
header while the public API remains deferred.

Campaign 4 may benchmark private commutation materialization labels
`host_vector`, `caller_owned_host_bytes`, `caller_owned_device_bytes`, and
`bitpacked_device_words` through `WOLFGANG_CUDA_BENCH_COMMUTATION_OUTPUT`.
Only the first two correspond to supported public host-output paths in this
campaign. Device-byte and bit-packed rows must remain visibly labeled as
private prototypes. Campaign 5 supersedes the private dense device-byte
prototype with `DeviceCommutationMatrix`; private bit-packed rows remain
benchmark evidence only.

Campaign 7 leaves bit-packed public output deferred in
`docs/plans/cuda_bitpacked_commutation_campaign7_decision.md`. Dense
`DeviceCommutationMatrix` remains the device-resident representation because
the retained grouping summaries copy compact results without a dense-capacity
trigger, while CSR graph materialization is limited by exported edge-list size
and host transfer rather than by a proven packed-layout win.

Campaign 8 is complete on H100 in
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign8_2026-04-29.md`
and keeps the same public compatibility baseline. Its retained private
benchmark path uses device-resident graph and grouping consumers that avoid
full CSR edge-list host export for high-scale workflows. The private hook
`wolfgang_quantum._wolfgang_core._benchmark_cuda_device_resident_consumer` is limited
to benchmarks and CUDA-gated tests, is not re-exported from `wolfgang_quantum`, and
does not belong in installed public C++ headers. Public fused grouping APIs,
DLPack export, stream-aware execution, and CUDA Graph replay remain unavailable
because the Campaign 8 reviews did not accept exact return shapes, ownership,
lifetime, synchronization, error, CPU-only, and documentation semantics. CSR
scatter tuning is rejected for Campaign 8 because the retained compact graph
and grouping consumers no longer need full CSR scatter by default. Non-H100
NVIDIA portability remains blocked until a named second host validates the
retained boundary from source.

Campaign 9 is complete in
`docs/benchmarks/reports/cuda_deferred_headroom_campaign9_2026-04-29.md` and
closed those Campaign 8 deferred or blocked items without any final
`deferred` status. Privileged Nsight Compute evidence passed on H100. The true
public fused grouping API remains unavailable and is rejected with evidence;
the accepted public summary extension is
`DeviceCommutationMatrix.conflict_degrees(axis=None|0|1)`. Read-only
`DeviceCommutationMatrix.__dlpack__` and `__dlpack_device__` are supported for
the dense row-major `uint8` buffer and validated with CuPy. Public stream-aware
execution and CUDA Graph replay remain unavailable after Campaign 9 rejection
with evidence. CSR scatter tuning remains rejected because retained compact
consumers avoid full CSR scatter. Non-H100 NVIDIA portability is
`blocked_external` until a named non-H100 NVIDIA host with a working driver is
provisioned and validated from source.

## Streams And Synchronization

The first CUDA backend uses the default stream and synchronizes before returning to Python. This keeps public semantics simple and deterministic.

A later advanced API may accept an external stream handle, but stream-aware methods must document ownership, lifetime, and synchronization before implementation.

Post-Phase 11 H100 experiments may use stream-aware helpers only as private
benchmark prototypes after `docs/plans/cuda_async_stream_api_review.md` accepts
the lifetime and synchronization model. Public stream handles, asynchronous
return semantics, and stream-ordered workspace ownership remain out of scope
until an API document specifies synchronization, error reporting, object
lifetime, and interaction with Python exceptions. Campaign 6 explicitly defers
public async/stream APIs and rejects private stream/event timing probes for this
slice so benchmark evidence stays comparable to synchronous public behavior.
Campaign 7 rechecked the same surface in
`docs/plans/cuda_async_stream_campaign7_decision.md` and keeps the public
default-stream synchronize-before-return invariant intact. No stream handle,
event object, graph-capture contract, or async Python return object is added.

## Python GPU Array Interop

Initial CUDA expectation support must accept host NumPy arrays by copying them to device internally.

Device-resident statevectors are supported through `__cuda_array_interface__`
first. Campaign 9 adds read-only DLPack export for
`DeviceCommutationMatrix`; DLPack input support for statevectors remains a
separate future API decision.

The initial CUDA implementation rejects unsupported device array layouts with `TypeError`. Accepted device arrays must be:

```text
1-dimensional
contiguous
complex64 or complex128
length == 2 ** num_qubits
on the same CUDA device as DevicePauliSum
fully contained in the CUDA allocation reported for the advertised data pointer; checked length-by-itemsize arithmetic and allocation-offset arithmetic must reject truncated or overflowing views before synchronization or kernel launch
```

Wolfgang validates the full advertised byte extent with
`cudaMemGetAddressRange` after `cudaPointerGetAttributes`; recognizing the
first byte is not sufficient. This validation also applies to zero-term
operators so an early return cannot bypass malformed view metadata.

The DLPack C ABI is vendored from the official DLPack v1.1 release under
`third_party/dlpack/`, including its Apache-2.0 license. Source builds use this
pinned header offline rather than duplicating ABI structs in the binding or
fetching a build-time dependency.

Post-Phase 11 statevector optimization continues to use this
CUDA-array-interface path for user-visible device residency. A dedicated
device-statevector wrapper may be prototyped only behind benchmark or private
C++ helpers until ownership, dtype, contiguity, device-ordinal, and
synchronization semantics are specified as public API.

## Kernel Order

CUDA kernels are implemented in this order:

```text
1. Device transfer and equality tests
2. CUDA statevector expectation for device-resident or copied statevectors
3. CUDA simplify using CUB or Thrust sort/reduce primitives
4. CUDA pairwise commutation matrix with dense-output guardrails
5. CUDA multiplication product generation followed by CUDA simplify
```

Phase 11 implements this first CUDA kernel set for source builds:
statevector expectation, simplify, pairwise commutation, and matrix-product
generation followed by simplify. Later CUDA work may tune kernels and widen
benchmark coverage, but it must preserve the API, guardrails, and benchmark
separation in this document.

Statevector expectation and simplify both landed in Phase 11 after CPU benchmark
evidence identified expectation and duplicate reduction as CUDA-relevant hot
paths. Future CUDA kernels must continue to be selected by benchmark evidence.
Post-Phase 11 CUDA hillclimbing follows
`docs/plans/cuda_deep_optimization_plan.md`; that work may add reusable-output
or workspace APIs only when ownership, synchronization, correctness, and
benchmark boundaries stay explicit.

## Primitive Policy

Use CUB or Thrust for:

```text
sort
reduce_by_key
scan
compact
```

Custom kernels are allowed for:

```text
statevector expectation inner loops
product generation
pairwise commutation tiles
packing and unpacking helpers
```

Do not write a custom GPU sort/reduce in the initial CUDA milestone.

## Guardrails

CUDA methods must enforce the same semantic guardrails as CPU methods:

```text
max_intermediate_terms for multiplication
max_commutation_matrix_entries for dense commutation matrices
num_qubits and statevector length checks
tolerance semantics for simplify
```

Device allocations must check for overflow before multiplying dimensions such as `num_terms * words` or `lhs_terms * rhs_terms`.

## CPU/GPU Equivalence

Every CUDA operation must have CPU/GPU equivalence tests:

```text
same dense labels after to_host()
same coefficients within operation tolerance
same simplify canonical order
same multiplication phase behavior
same expectation result within dtype-specific tolerance
same error behavior for invalid shapes and sizes
```

CUDA tests are skipped when CUDA is not available, but the skip reason must say whether CUDA was absent at build time or no runtime device was found.

CUDA target, toolkit, driver, and hardware validation follows the testing ladder in `docs/architecture/hardware_targets_and_testing.md`.

## Benchmark Gates

CUDA implementation is not complete until benchmarks compare:

```text
CPU scalar
CPU oneTBB when available
CPU SIMD where available
CUDA including transfer time
CUDA excluding transfer time for device-resident workflows
GPU-library competitors when the workload maps cleanly to a documented accelerator library
```

The benchmark report must identify the input sizes where CUDA is slower than
CPU, faster than CPU, and transfer-bound. GPU-library competitor results must
be labeled as framework-level, transfer-inclusive primitive, or
device-resident primitive and must not be used for sparse-Pauli primitive
speedup claims unless the semantic mapping is exact.

When an optimized CPU selector is available for a CUDA benchmark operation, the
benchmark must time every available selector separately. Unavailable selectors
must be reported with reasons so plots and reports do not hide missing CPU
coverage.

## Post-Phase 11 Optimization Boundaries

H100 deep optimization campaigns after Phase 11 are allowed to retain
production code only when the public semantic boundary remains unchanged or the
affected public API is documented before implementation. The current
optimization-boundary decisions are:

```text
workspace ownership: internal C++/benchmark-only unless H100 evidence justifies an explicitly experimental public API
workspace lifetime: tied to one CUDA device ordinal; no cross-device reuse; monotonic capacity growth; explicit reset/release required before shrinking
statevector residency: public behavior stays on host NumPy copy and CUDA-array-interface device arrays
stream semantics: public API remains default stream and synchronize-before-return
stream semantics detail: existing public CUDA APIs remain default-stream and synchronize-before-return; Campaign 6 may benchmark private stream/event probes only when docs/plans/cuda_async_stream_api_review.md accepts their lifetime and synchronization model
consumer semantics: DeviceCommutationMatrix compact-summary APIs may be retained only when exact signatures, return types, synchronization, CPU-only behavior, and layout rules are documented before implementation; bit-packed public APIs remain deferred
result materialization: public commutation output includes vector return, caller-owned host bytes, dense DeviceCommutationMatrix output, compact DeviceCommutationMatrix count/conflict summaries, CUDA Array Interface export, and read-only DLPack export; bit-packed output and async copies remain private or deferred prototypes only
determinism: canonical sparse output ordering and simplify tolerances remain deterministic; floating reductions may use implementation-defined parallel order only within documented dtype tolerances
```

Any retained optimization must include same-boundary A/B evidence. A faster
prototype that changes transfer timing, allocation timing, stream behavior,
output ownership, or floating-point tolerance may appear in the campaign report
only as a rejected or deferred experiment unless the public contract above is
updated and reviewed in the same slice.

Campaign 3 retained one such unchanged-boundary production optimization: the
CUDA simplify implementation may use a packed 64-bit key for one-word operators
with at most 32 qubits. The packed key preserves canonical x-then-z ordering
and does not change public output ordering, tolerance, transfer, stream, or
ownership semantics.

Campaign 3 also added a private benchmark-only commutation output reuse path
behind `WOLFGANG_CUDA_BENCH_REUSE_COMMUTATION_DEVICE_OUTPUT=1`. This path is
not public API, does not expose device pointers, and exists only to quantify
device-output allocation and host-materialization overhead. Public commutation
output remains vector return or caller-owned host byte fill.

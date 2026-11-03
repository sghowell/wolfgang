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
target-specific: CUDA-only, HIP-only or Metal-only. The target-specific accelerator policy
deliberately rejects `WOLFGANG_ENABLE_CUDA=ON` with `WOLFGANG_ENABLE_HIP=ON`
unless a future accepted mixed-runtime plan changes that boundary. The
implementation handoff is complete in
`docs/plans/backend_neutral_accelerator_campaign9_plan.md`, with closeout
evidence in
`docs/benchmarks/reports/backend_neutral_accelerator_campaign9_2026-05-01.md`.
That backend-neutral Campaign 9 added the explicit construction selector used
by CUDA source builds: `PauliSum.to_device(backend="cuda")` and
`DeviceCommutationMatrix.empty(..., backend="cuda")`. Elsewhere in this
document, "Campaign N" means the numbered H100 CUDA optimization campaign:
Campaigns 2-9 are planned in `docs/plans/h100_deep_optimization_campaignN_plan.md`,
Campaign 10 in `docs/plans/cuda_cross_architecture_campaign10_plan.md` and
Campaign 11 in `docs/plans/cuda_residual_risk_campaign11_plan.md`.

## Build Matrix

Initial source-build CUDA support follows `docs/architecture/hardware_targets_and_testing.md`, summarized here:

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

Host `PauliSum` methods take no backend argument; they run on the CPU. The backend is chosen when an operator moves to a device:

```python
h.to_device(device=0, backend="auto")
h.to_device(device=0, backend="cuda")
```

`backend` may be `None`, `"auto"`, `"cuda"`, `"hip"` or `"metal"`. A build without the requested accelerator raises `RuntimeError` with rebuild guidance.

## Ownership And Lifetime

`DevicePauliSum` owns its device buffers. Copying a device object is disabled unless an explicit `.copy()` API is implemented. Moving a device object transfers ownership.

`to_device()` copies host buffers to device. `to_host()` copies device buffers to a new host `PauliSum`.

Device operations return new device objects unless the method name ends in `_inplace`. No in-place device operations are part of the first CUDA milestone.

Post-Phase 11 optimization may introduce an internal CUDA workspace object for
H100 benchmark experiments. That object is not public Python API and is not an
installed C++ API until a later API review promotes it. Internal workspace
experiments must preserve the existing `DevicePauliSum` ownership model and the
cross-backend private reusable accelerator scratch and output buffers contract:

```text
private reusable accelerator scratch and output buffers remain move-only
workspace storage is tied to the same backend-local device ordinal as every operand it serves
workspace reuse is allowed only when every operand lives on that same device
reset retains the allocation for reuse
release returns the allocation to the runtime
capacity growth is monotonic for a run unless an explicit reset or release path is used
temporary-storage reuse must not change canonical ordering, zero-tolerance filtering, or coefficient tolerance semantics
benchmark labels must say whether workspace allocation and growth are inside or outside the timed boundary
must not expose raw device pointers or framework objects through the public API
```

Campaign 4 implemented this as a private source-only `CudaWorkspace` under
`src/cuda/`. Python reaches it only through the private
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

Campaign 5 added the experimental public device-output form for GPU-resident
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

Campaign 6 retained compact summary counts as the first supported downstream
GPU consumer for `DeviceCommutationMatrix`. `count_commuting(axis=None)` returns
a Python integer total count, `axis=0` returns `uint64` column counts, and
`axis=1` returns `uint64` row counts. All reductions execute on the matrix's
CUDA device and copy only the compact count result to host. The API is
synchronous: calls use the default stream and return only after the count
result is available on the host.

Campaign 7 evaluated fused downstream graph and grouping-oriented consumers on
top of the same dense `DeviceCommutationMatrix` layout. At the start of that
campaign, anti-commutation CSR graph construction, conflict-degree summaries,
and grouping summaries were benchmark-only private helpers, pending acceptance
of exact return semantics, ownership, ordering, synchronization, CPU-only
behavior, and failure modes in
`docs/plans/cuda_fused_commutation_consumer_api_review.md`. Benchmark rows must
carry `private_benchmark_only` labels and decision fields for
count-specialization, bit-packed output, and non-H100 portability. No Campaign
7 fused-consumer helper belongs in an installed public header while the public
API remains deferred.

Campaign 4 could benchmark the private commutation materialization labels
`host_vector`, `caller_owned_host_bytes`, `caller_owned_device_bytes`, and
`bitpacked_device_words` through `WOLFGANG_CUDA_BENCH_COMMUTATION_OUTPUT`.
Only the first two corresponded to supported public host-output paths in that
campaign. Device-byte and bit-packed rows must remain visibly labeled as
private prototypes. Campaign 5 superseded the private dense device-byte
prototype with `DeviceCommutationMatrix`; private bit-packed rows remain
benchmark evidence only.

Campaign 7 left bit-packed public output deferred in
`docs/plans/cuda_bitpacked_commutation_campaign7_decision.md`. Dense
`DeviceCommutationMatrix` remains the device-resident representation because
the retained grouping summaries copy compact results without a dense-capacity
trigger, while CSR graph materialization is limited by exported edge-list size
and host transfer rather than by a proven packed-layout win.

Campaign 8 is complete on H100 in
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign8_2026-04-29.md`
and kept the same public compatibility baseline. Its retained private
benchmark path uses device-resident graph and grouping consumers that avoid
full CSR edge-list host export for high-scale workflows. The private hook
`wolfgang_quantum._wolfgang_core._benchmark_cuda_device_resident_consumer` is
limited to benchmarks and CUDA-gated tests, is not re-exported from
`wolfgang_quantum`, and does not belong in installed public C++ headers. Public
fused grouping APIs, DLPack export, stream-aware execution, and CUDA Graph
replay stayed unavailable after Campaign 8 because its reviews did not accept
exact return shapes, ownership, lifetime, synchronization, error, CPU-only, and
documentation semantics. Campaign 8 rejected CSR scatter tuning because the
retained compact graph and grouping consumers no longer need full CSR scatter
by default. Non-H100 NVIDIA portability remained blocked until a named second
host could validate the retained boundary from source.

Campaign 9 is complete in
`docs/benchmarks/reports/cuda_deferred_headroom_campaign9_2026-04-29.md` and
closed those Campaign 8 deferred or blocked items without any final
`deferred` status. Privileged Nsight Compute evidence passed on H100. Its
outcomes:

- `DeviceCommutationMatrix.conflict_degrees(axis=None|0|1)`, the
  anti-commuting complement of `count_commuting(...)`, is the accepted public
  summary extension.
- Read-only `DeviceCommutationMatrix.__dlpack__` and `__dlpack_device__` are
  supported for the dense row-major `uint8` buffer and validated with CuPy.
- The true public fused grouping API,
  `DevicePauliSum.group_commuting_device(...)`, remains unavailable and is
  rejected with evidence.
- Public stream-aware execution and CUDA Graph replay remain unavailable after
  Campaign 9 rejection with evidence.
- CSR scatter tuning remains rejected because retained compact consumers avoid
  full CSR scatter.
- Non-H100 NVIDIA portability was `blocked_external` until a named non-H100
  NVIDIA host with a working driver could be validated from source. Campaign 10
  resolved it with A100 `sm_80` and RTX PRO 6000 Blackwell `sm_120` source-build
  evidence.

## Streams And Synchronization

The first CUDA backend uses the default stream and synchronizes before returning to Python. This keeps public semantics simple and deterministic.

A later advanced API may accept an external stream handle, but stream-aware methods must document ownership, lifetime, and synchronization before implementation.

Post-Phase 11 H100 experiments may use stream-aware helpers only as private
benchmark prototypes after `docs/plans/cuda_async_stream_api_review.md` accepts
the lifetime and synchronization model. Public stream handles, asynchronous
return semantics, and stream-ordered workspace ownership remain out of scope
until an API document specifies synchronization, error reporting, object
lifetime, and interaction with Python exceptions. Campaign 6 deferred public
async/stream APIs and rejected private stream/event timing probes so benchmark
evidence stayed comparable to synchronous public behavior. Campaign 7
rechecked the same surface in
`docs/plans/cuda_async_stream_campaign7_decision.md` and kept the public
default-stream synchronize-before-return invariant. It added no stream handle,
event object, graph-capture contract, or async Python return object.

## Multiple Devices And Threads

Each `DevicePauliSum` and `DeviceCommutationMatrix` lives on one device.
Operations whose operands, output or statevector sit on different devices
raise `ValueError`. Every device switch goes through one guard
(`src/cuda/device_guard.cuh`), which makes the operand's device current for the
call and then switches back to the caller's device, unless that device has no
primary context yet. Since CUDA 12, `cudaSetDevice` creates the primary
context, about 500 MiB on an H100, so switching back to a device the process
never used would charge a process that only works on device 1 for a context on
device 0. In that case the calling thread stays on the operand's device, so
`cudaGetDevice` reports that device to every library on the thread afterwards.
PyTorch's device guard avoids the same context but keeps its own thread-local
device, so its `current_device()` does not change. A non-primary context made
current through the driver API is not restored, as before. HIP switches back
unconditionally: ROCm initializes its devices at startup, so switching back
creates nothing.

CUDA device calls release the GIL, so Python threads can keep several GPUs busy
at once. The host-side state they touch is per thread (workspaces, mapped
results) or locked (memory pools, pinned staging buffers). Compute calls also
take a per-device lock after releasing the GIL: concurrent small calls on one
device only contended for its default stream and launch path and lost about a
third of their throughput, so threads on one device run one after another, as
they did under the GIL, while threads on different devices overlap. Transfers
(`to_device`, `to_host`, `from_device_arrays`) skip the lock because most of
their time is host-side copying, which does overlap. Two threads driving two
H100s reach 1.88-1.98 times one thread's throughput
(`docs/benchmarks/reports/cuda_multi_gpu_h100_2026-11-03.md`). Concurrent calls
that write the same `DeviceCommutationMatrix` output race, as with any shared
buffer. HIP and Metal builds keep the GIL until the same is verified on their
hardware.

## Python GPU Array Interop

Initial CUDA expectation support must accept host NumPy arrays by copying them to device internally.

Device-resident statevectors are supported through `__cuda_array_interface__`
first. Campaign 9 added read-only DLPack export for
`DeviceCommutationMatrix`; DLPack input support for statevectors remains a
separate future API decision.

`DevicePauliSum.device_arrays()` exports the operator's x words, z words and
coefficients the same way: versioned DLPack capsules flagged read-only
(reusing the `DeviceCommutationMatrix` capsule context and deleters) and CUDA
Array Interface v3 dictionaries with `readonly` true and `stream` 1. The flag
is advisory, because CuPy and PyTorch import without enforcing it; writing
through a consumer array is undefined behavior and can break the tail-bit
invariant. PyTorch rejects read-only CUDA Array Interface input and reads the
views through `torch.from_dlpack`. Consumers must request versioned capsules
(`max_version >= (1, 0)`): CuPy 14 and PyTorch 2.15 were validated. The
capsule context holds a reference to the owning Python object, so a consumer
array keeps the operator alive.

Operator buffers come from the stream-ordered pool. An export first finishes
the legacy stream, because a pageable host-to-device copy from `to_device` can
return before its DMA lands, and then marks the operator. Freeing a marked
operator synchronizes the device before its blocks return to the pool, because
a consumer may still have work queued on another stream that reads them.
Unexported operators free without synchronizing. That synchronization runs
when the last reference goes away, possibly during garbage collection with the
GIL held. It waits for unrelated
work queued by other libraries, and a pending host callback that needs the GIL
(for example CuPy `launch_host_func`) would deadlock. Inside a CUDA graph
capture the synchronization fails and invalidates the capture; the error is
cleared and the free proceeds.

`DevicePauliSum.from_device_arrays` validates each input with
`cudaPointerGetAttributes` (device or managed memory, all three on one device)
and `cuMemGetAddressRange` in the owning device's context (the full advertised
extent lies inside one allocation). It then synchronizes the device, because a
CUDA Array Interface producer may have written the inputs on a non-blocking
stream that does not order with the legacy stream. It copies device to device
into pooled buffers, rejects set bits at or above `num_qubits` with a
device-side `thrust::count_if`, and synchronizes the legacy stream before
returning, so the caller may reuse the inputs and any stream may read the
result. Word counts use the same overflow-checked arithmetic as the host
path.
Evidence: `docs/benchmarks/reports/device_arrays_h100_2026-11-03.md`.

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

H100 deep optimization campaigns after Phase 11 may retain production code
only when the public semantic boundary remains unchanged or the affected public
API is documented before implementation. The current optimization-boundary
decisions are:

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
device-output allocation and host-materialization overhead. At the time, public
commutation output was a returned vector or a caller-owned host byte fill;
Campaign 5 later added the device-resident `DeviceCommutationMatrix`.

Operators that the fused path below does not handle run a multi-launch pipeline
(`src/detail/device_simplify_pipeline.cuh`, which builds under nvcc and hipcc)
that keeps every intermediate count on the device:

1. one CUB reduction computes the largest magnitude, NaN when a component is
   not finite, and the largest half magnitude;
2. a pack kernel writes the sort keys, and its first thread turns that
   reduction into the CPU drop rule;
3. one-word operators of at most 32 qubits radix-sort a packed key of
   `2 * num_qubits` bits; other one- and two-word operators sort word keys with
   `stable_sort_by_key`; wider operators sort term indices with an index
   tie-break. Equal terms keep input order on every path;
4. CUB reduce-by-key sums each duplicate run and leaves the run count in
   device memory;
5. an inclusive scan over "this run survives" flags assigns output positions
   over the whole input length, flagging runs past the device-side run count
   as 0;
6. one thread copies the counts and status flags to a per-thread pinned,
   device-mapped host buffer.

The host synchronizes once, allocates exactly the surviving terms, scatters
them with one kernel, and synchronizes again before returning. Non-finite input
raises `ValueError`, which takes precedence as on the CPU. Any non-finite sum
raises `OverflowError`, including the `inf + -inf = NaN` that a tree-ordered
reduction can form. Every path uses the CPU drop rule for finite coefficients
whose magnitude exceeds `DBL_MAX`: the threshold is `atol` alone when
`rtol == 0`, and magnitudes are compared in half units when
`atol + rtol * max_abs` overflows. Single-term inputs skip the pipeline and
apply only the drop rule, and zero-qubit operators sum their coefficients in
one reduction. The Thrust stages this pipeline replaced read the maximum, the
run count and the survivor count back between stages, eight synchronizations
per call; `simplify` on 4,096 terms dropped from 188 to 108 us on an H100.
Evidence: `docs/benchmarks/reports/cuda_device_counts_h100_2026-11-03.md`,
`docs/benchmarks/reports/cuda_simplify_paths_h100_2026-11-03.md`,
`docs/benchmarks/reports/hip_parity_mi300x_2026-11-03.md`.

Two benchmark-only baselines keep host-side counts for paired comparisons:
the CUB radix-sort path behind
`WOLFGANG_CUDA_BENCH_DUPLICATE_REDUCTION=cub_radix_sort_reduce`, and the
original one-thread reduction kernel for operators wider than two words behind
`WOLFGANG_CUDA_BENCH_GENERIC_MULTIWORD_REDUCTION=serial_kernel`. Their drop
predicate sets the same overflow flag, which the host reads after their
compaction synchronizes.

Small operators take a fused path shared with HIP
(`src/detail/small_simplify_kernel.cuh`). This covers operators with one to
four packed words (up to 256 qubits) and at most 1,024 terms (512 for three or
four words), single terms included. Each runs as one thread block, sized to
the next power of two of the term count (at least 64):

1. sort keys in shared memory;
2. gather coefficients into sorted order;
3. sum each run sequentially in input order;
4. apply the CPU drop rule;
5. compact survivors with a block scan.

The result comes back through pinned, device-mapped host memory after one
stream synchronization. Coefficients match CPU simplify bit for bit. Output
buffers are sized for every input term, and `num_terms` records the survivors.
More than 48 KiB of shared memory needs an opt-in, which is set per launch
because CUDA applies it per device; a launch-configuration error falls back to
the multi-launch path. The benchmark switch
`WOLFGANG_CUDA_BENCH_SMALL_SIMPLIFY=off` keeps the multi-launch path. Evidence:
`docs/benchmarks/reports/small_simplify_2026-11-03.md`.

`DevicePauliSum` buffers, Thrust algorithm temporaries, and simplify,
expectation and commutation scratch come from a Wolfgang-private stream-ordered
memory pool on the legacy default stream. Every operation still synchronizes
before returning, so a block freed into the pool is never in use when it is
reused; pooled Thrust algorithms use `par_nosync` for the same reason. The pool
caches up to 1 GiB of freed blocks per device (`kPoolReleaseThresholdBytes`);
above that it returns memory to the driver at the next synchronization. On an
out-of-memory failure, Wolfgang trims every pool and retries once before
raising `MemoryError` (non-pooled `cudaMalloc` allocations follow the same
rule), and it trims at process exit. Cached blocks are not available to other
libraries in the process until a trim. Devices without memory-pool support, or
runs with the benchmark switch `WOLFGANG_CUDA_BENCH_DISABLE_POOL=1`, use
`cudaMalloc`.

`DeviceCommutationMatrix` buffers and their count scratch also come from the
pool. A consumer on another stream could still read an exported block after
Wolfgang frees it, so handing out a matrix's pointer through DLPack or the CUDA
Array Interface marks it, and freeing a marked matrix synchronizes the device
first, as for `DevicePauliSum.device_arrays()`. The first export from a
device maps that device's pool into every peer GPU that can access it
(`cudaMemPoolSetAccess`, read-write), so a kernel on a peer reads exported
matrices and device arrays as it would `cudaMalloc` memory once peer access is
enabled. Before, such a read faulted with an illegal address that poisoned the
CUDA context. Freeing an exported buffer then synchronizes the mapped peers
that have a live context, as well as the owning device, because a peer kernel
may still be reading it. The mapping covers the pool's current and future
blocks; on two
NVLink-connected H100s it cost 26 ms once, created no context on the peer, and
left pool growth and per-call times unchanged
(`docs/benchmarks/reports/cuda_multi_gpu_h100_2026-11-03.md`). The pool is not
revalidated after `cudaDeviceReset`. Transfers
and output creation select the backend with a device count rather than a full
device-property query.
Evidence: `docs/benchmarks/reports/cuda_allocation_reuse_h100_2026-11-03.md`,
`docs/benchmarks/reports/cuda_commutation_pool_h100_2026-11-03.md`.

`DevicePauliSum.to_host()` copies the three buffers asynchronously on the
legacy stream into a pinned buffer borrowed from a process-wide pool
(`src/detail/staged_transfer.hpp`) and appends them to result vectors that are
reserved rather than zero-filled. One wait covers each batch, so an operator
that fits in half the buffer waits once; larger operators alternate between
the two halves, copying into one while appending the other. Buffers grow to at
most 16 MiB, the pool keeps at most four idle, and if pinned memory is
unavailable `to_host` copies directly. On Linux, result vectors of 4 MiB or more are
advised for transparent huge pages before their first touch, because on the
KVM hosts measured, faulting in fresh 4 KiB pages cost more than the copy
itself. `to_host` is 1.9-3.2x faster on an H100 from 1 term to 4M terms; a
direct copy into huge-page memory above 16 MiB measured slower end to end.
`to_device()` and statevector uploads keep pageable `cudaMemcpy`, which beat
staging up to 32 MB. Evidence:
`docs/benchmarks/reports/host_transfers_2026-11-03.md`.

# CUDA DLPack Interop Campaign 9 Contract

Status: final. Read-only `DeviceCommutationMatrix` DLPack producer
implemented and validated with CuPy on H100.

Campaign 9 revisits DLPack only for a complete ownership, stream, lifetime,
same-device, dtype, shape, mutability, and real-consumer test plan. DLPack may
not be accepted unless at least one real CUDA DLPack consumer is installed and
passes on the H100 host.

## Candidate Scope

```text
read-only DLPack producer for DeviceCommutationMatrix dense uint8 buffer
```

Accepted public methods:

```text
DeviceCommutationMatrix.__dlpack__(stream=None, max_version=(1, 0), copy=None)
DeviceCommutationMatrix.__dlpack_device__()
```

`__dlpack_device__` returns `(2, device_ordinal)`, where `2` is `kDLCUDA`.

## Default Rejections

Campaign 9 rejects these surfaces by default:

```text
mutable exports
cross-device copies
raw pointer exports
compact grouping metadata whose lifetime is not tied to a stable owning object
copy=True unless an explicit copy implementation is added
stream=0 because the Python Array API marks CUDA stream 0 ambiguous
```

## Acceptance Bar

DLPack is accepted only if all of these behaviors are specified and tested:

```text
__dlpack__(stream=None|1|2|>2, max_version=positive tuple, copy=None|False)
__dlpack_device__()
single-consumer capsule behavior
used_dltensor rename/deleter behavior
same-device enforcement
stream wait/event behavior
dtype "|u1", row-major compact strides, shape (rows, cols)
owner lifetime and moved-from behavior
CuPy and PyTorch consumer tests when both are installable
at least one real CUDA DLPack consumer if accepted
CPU-only BufferError or RuntimeError behavior
```

## Campaign 9 Final Decision

Campaign 9 accepts and implements the read-only dense
`DeviceCommutationMatrix` DLPack producer:

```text
DeviceCommutationMatrix.__dlpack__(stream=None, max_version=(1, 0), copy=None)
DeviceCommutationMatrix.__dlpack_device__()
```

The accepted surface is intentionally narrow:

```text
dense row-major uint8 buffer only
shape (rows, cols)
device tuple (kDLCUDA, device_ordinal)
single-consumer capsule behavior delegated to the consumer
owner lifetime retained by the capsule deleter context
copy=True rejected with BufferError
stream=0 rejected as ambiguous
legacy unversioned capsules rejected because they cannot carry read-only flags
positive versioned capsules required and supported
mutable exports and cross-device copies rejected
```

The H100 campaign installed and ran a real CuPy CUDA DLPack consumer. PyTorch
remains an optional additional consumer; the test is present and skips when
PyTorch CUDA is not installed. Benchmark evidence is recorded in
`docs/benchmarks/data/cuda_deferred_headroom_campaign9_2026-04-29/raw/dlpack_interop.json`.

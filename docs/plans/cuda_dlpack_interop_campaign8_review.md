# CUDA DLPack Interop Campaign 8 Review

Status: DLPack public API deferred; CUDA Array Interface consumers remain the
retained framework interop path.

Campaign 8 evaluates whether retained device outputs should be consumable by
frameworks without dense host materialization. The existing
`DeviceCommutationMatrix.__cuda_array_interface__` path remains available for
CuPy-style consumers. DLPack is not promoted to public API in this campaign.

## Decision

`dlpack_interop_status: deferred`

Reason: DLPack requires exact PyCapsule ownership, deleter, stream, mutability,
dtype, shape, and same-device semantics before exposing FastPauli-owned memory.
The allowed implementation is a private benchmark probe or an explicit
unavailable row. Campaign 8 records the unavailable DLPack reason while keeping
the CUDA Array Interface consumer evidence.

## Required Contract Before DLPack Acceptance

A future public or private DLPack probe must define and test:

```text
single-consumer PyCapsule ownership rules
deleter behavior and object lifetime
same-device enforcement
stream synchronization and producer/consumer ordering
dtype, shape, strides, and read-only or mutable semantics
use-after-release behavior
CPU-only error behavior
moved-from FastPauli object behavior
interaction with DeviceCommutationMatrix and future compact grouping metadata
documentation and API stability status
```

## Retained Interop Boundary

`DeviceCommutationMatrix.__cuda_array_interface__` remains the retained
framework interop surface. It exposes the dense row-major `uint8` matrix owned
by FastPauli. Public methods remain synchronous and default-stream; consumers
must respect the existing CUDA Array Interface metadata and keep the FastPauli
matrix object alive while using the view.

## Campaign 8 Benchmark Rows

Campaign 8 benchmark output distinguishes:

```text
cupy_cuda_array_interface_consumer
pytorch_dlpack_consumer or dlpack_unavailable
dlpack_interop_status
unavailable_reason
```

No user-facing DLPack documentation is added until the acceptance contract above
is satisfied.

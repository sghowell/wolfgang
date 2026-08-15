# ROCm Campaign 8 HIP Interop Reconsideration Decision

Campaign 8 defines the contract required before FastPauli reopens external HIP
statevector pointers or HIP DLPack exports. It does not implement interop.

## Current Decision

HIP DLPack and external HIP statevector pointer support remain unavailable.
Campaign 5 rejected HIP DLPack retention because PyTorch ROCm consumed the
candidate `kDLROCM` view but accepted mutation of a view FastPauli intended to
be read-only.

That rejection remains in force until a real consumer rejects mutation of a
read-only exported view or FastPauli accepts a deliberately mutable export with
a separate ownership and correctness contract.

HIP CUDA Array Interface remains unavailable. A HIP pointer must not be exposed
as CUDA memory.

## External HIP Statevector Contract

A future external HIP statevector API must define:

```text
accepted producer library and version
device type and device ordinal discovery
statevector dtype and layout
statevector length and qubit-count validation
ownership of the input allocation
stream or synchronization ownership
read-only versus mutable behavior
failure mode for wrong backend, wrong device, wrong dtype, non-contiguous layout, and moved-from operators
CPU oracle and tolerance
transfer-inclusive and device-resident timing boundaries
```

The first accepted consumer must be a named real ROCm library such as PyTorch
ROCm or another library with a documented HIP device-pointer or DLPack handoff.
A private fake pointer test is not enough to claim interop.

## HIP DLPack Producer Contract

A future HIP DLPack export must define:

```text
DLPack device type and versioned capsule behavior
producer object lifetime retained by the capsule deleter
read-only flag propagation and consumer behavior
stream handoff and synchronization behavior
same-device validation
consumer mutation test
consumer destruction test
benchmark boundary labels
```

The mutation test must show one of these outcomes:

```text
the consumer rejects mutation of a read-only FastPauli view
FastPauli explicitly accepts a mutable export and documents mutation effects
```

Without one of those outcomes, HIP DLPack remains rejected with evidence.

## Benchmark Boundaries

Interop reports must distinguish:

```text
consumer-only timing after a FastPauli device object already exists
transfer-inclusive timing that includes host-to-device setup
device-resident primitive timing that excludes host materialization
consumer materialization timing when data is copied back to host
```

Reports must not compare a consumer-only timing against a transfer-inclusive
FastPauli timing without making the boundary difference explicit.

## Terminal Statuses

Campaign 8 records:

```text
external_hip_statevector_contract: accepted_for_future_implementation
hip_dlpack_reconsideration_contract: accepted_for_future_implementation
hip_cuda_array_interface_policy: rejected_with_evidence
```

These statuses accept the gates for future work. They do not expose new Python
methods, device-pointer imports, DLPack capsules, or array-interface exports.

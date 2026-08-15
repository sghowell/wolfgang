# CUDA Async And Stream Campaign 7 Decision

Status: public async/stream API deferred.

Campaign 7 reconsidered streams because fused consumers can make multiple
device-resident steps visible: dense commutation fill, conflict reductions, CSR
scatter, compact host copies, and full dense materialization. The retained
public CUDA invariant remains simpler and safer: public methods use the default
stream and synchronize before returning to Python.

## Decision

No public stream-handle, event, future, or async return object is retained in
Campaign 7.

No private stream/event timing probe is retained as a public-facing benchmark
claim. Campaign 7 profiler evidence is collected with synchronous public
boundaries and Nsight kernel timelines so the timing remains comparable to the
current API.

## Required Contract Before Reconsideration

A future async or stream-aware API must define:

```text
stream owner and device ordinal
event owner and destruction behavior
which FastPauli objects are kept alive until completion
whether CUDA graph capture is supported or explicitly rejected
where deferred CUDA errors become Python exceptions
whether timing labels are enqueue-only, event-elapsed, synchronization-only, or end-to-end
interaction with DeviceCommutationMatrix and private workspace allocations
CPU-only behavior and error messages
moved-from object behavior
```

## Campaign 7 Evidence

The fused grouping-summary path already answers the retained downstream
question synchronously: it returns compact host summaries without dense host
materialization. The CSR graph path is dominated by the exported edge-list host
copy when the graph is materialized for validation/reporting. A stream-aware API
would not change public end-to-end semantics unless callers could also consume
the graph fully on device under a complete lifetime contract.

## Current Public Invariant

```text
DevicePauliSum CUDA methods: default stream, synchronize-before-return
DeviceCommutationMatrix.to_host(): synchronous host materialization
DeviceCommutationMatrix.count_commuting(axis=None|0|1): synchronous compact host result
private fused benchmark helpers: synchronize before host-visible summaries
```

This document does not change `docs/architecture/api_stability.md`; no public
async or stream API is added.

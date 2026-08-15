# CUDA Graphs And Stream Campaign 8 Decision

Status: public stream and CUDA Graph APIs deferred; private benchmark probe may
report unavailable status with evidence.

FastPauli's public CUDA methods continue to use the default stream and
synchronize before returning to Python. Campaign 8 does not add public stream
handles, event objects, graph capture handles, futures, or async return values.

## Decision

`stream_graph_status: deferred`

Reason: public streams and CUDA Graph capture require complete error
propagation, synchronization, lifetime, and Python ownership contracts. The
allowed Campaign 8 implementation is a private benchmark probe only if it can
preserve synchronous public APIs and report graph instantiation/replay timing
separately. The current retained result is an explicit unavailable row.

## Required Contract Before Promotion

A future stream-aware or CUDA Graph API must define:

```text
stream owner and device ordinal
event owner and destruction behavior
which FastPauli objects remain alive until completion
whether graph capture is supported, rejected, or benchmark-only
where deferred CUDA errors become Python exceptions
whether timing labels are enqueue-only, event-elapsed, synchronization-only, or end-to-end
workspace allocation and graph-capture safety
interaction with DeviceCommutationMatrix and compact consumers
CPU-only behavior and error messages
moved-from object behavior
```

## Current Public Invariant

```text
DevicePauliSum CUDA methods: default stream, synchronize-before-return
DeviceCommutationMatrix.to_host(): synchronous host materialization
DeviceCommutationMatrix.count_commuting(axis=None|0|1): synchronous compact host result
Campaign 8 private benchmark hooks: synchronize before host-visible summaries
```

## Benchmark Reporting

Campaign 8 rows must include `stream_graph_status`. If a graph probe is
unavailable, the row must record an explicit reason instead of silently omitting
the remaining-headroom item. Public documentation remains unchanged except for
the statement that graph/stream surfaces are deferred.

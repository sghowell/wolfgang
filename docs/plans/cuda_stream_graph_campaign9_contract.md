# CUDA Stream And Graph Campaign 9 Contract

Status: final. Public stream-aware execution and CUDA Graph replay
`rejected_with_evidence` for Campaign 9.

Public CUDA methods remain synchronous and default-stream compatible unless
this contract explicitly accepts a new public stream, event, or graph API.
Campaign 9 may accept a private benchmark-only CUDA Graph replay probe only for
fixed-shape retained compact graph/grouping consumers.

## Public Baseline

```text
existing public CUDA methods remain synchronous
existing public CUDA methods remain default-stream compatible
no public stream handle argument is accepted by default
no public event object is accepted by default
no public CUDA graph handle is accepted by default
```

## Private Probe Candidate

```text
replay the retained compact graph/grouping consumer as a benchmark-only CUDA Graph
fixed shapes only
stable workspace addresses only
no capture of pageable or pinned host allocation
no capture using cudaStreamLegacy as the capture stream
all CUDA errors become Python exceptions before returning from the private hook
```

## Acceptance Bar

Stream or graph replay is accepted only if the contract specifies:

```text
enqueue timing
event-elapsed timing
host synchronization
workspace lifetime
graph-capture safety
graph-update behavior
Python exception timing
CPU-only behavior
shape-change rejection behavior
```

Graph replay must improve a retained compact consumer by at least 5% end-to-end
or by at least 10% on repeated small/medium fixed-shape rows without changing
public synchronous semantics. Otherwise Campaign 9 records
`rejected_with_evidence`.

## Campaign 9 Final Decision

Campaign 9 keeps public CUDA methods synchronous and default-stream compatible.
It rejects public stream handles, public event objects, public CUDA Graph
handles, and private graph replay for this slice. The retained compact
consumers are already dominated by the commutation fill and compact reductions
at the measured H100 scale; adding a stream/graph surface would add ownership,
capture, error-propagation, and Python lifetime complexity without meeting the
required end-to-end improvement bar.

Evidence is recorded in
`docs/benchmarks/data/cuda_deferred_headroom_campaign9_2026-04-29/raw/stream_graph.json`.

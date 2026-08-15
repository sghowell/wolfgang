# CUDA Fused Grouping Public API Campaign 9 Contract

Status: final. True public grouping API `rejected_with_evidence`; compact
`DeviceCommutationMatrix.conflict_degrees(axis=None|0|1)` implemented.

Campaign 9 must close the Campaign 8 public fused grouping API headroom item
with either an implemented public grouping-returning API or a documented
`rejected_with_evidence` outcome. A compact summary API can be accepted only as
a separate surface; it does not close the true grouping-returning API item by
itself.

## Candidate Public Symbols

```text
DeviceCommutationMatrix.conflict_degrees(axis=None|0|1, *, top_k=None)
DeviceCommutationMatrix.top_conflicts(axis=1, *, k=8)
DevicePauliSum.group_commuting_device(mode="full", strategy="largest_first", max_terms_for_graph=50000)
```

## True Grouping API Acceptance Bar

`DevicePauliSum.group_commuting_device(...)` is accepted only if this contract
is updated before implementation to specify:

```text
exact return type and shape for groups
whether groups are returned as host PauliSum objects, host index arrays, or device-resident metadata
stable ordering relative to CPU group_commuting(mode="full", strategy="largest_first")
ownership and lifetime of every returned object
device and stream synchronization semantics
allocation guardrails and failure mode
CPU-only behavior and moved-from behavior
correctness oracle against CPU group_commuting
user docs and API-stability status
```

## Optional Compact Summary Semantics

If accepted separately, `DeviceCommutationMatrix.conflict_degrees(axis=None|0|1)`
has these semantics:

```text
axis=None returns a Python int equal to rows * cols - count_commuting()
axis=1 returns a uint64 NumPy array of length rows with per-row anti-commuting counts
axis=0 returns a uint64 NumPy array of length cols with per-column anti-commuting counts
ordering follows matrix row-major term order
the method is synchronous and default-stream compatible
CPU-only builds expose no DeviceCommutationMatrix object, matching existing CUDA behavior
moved-from matrices raise RuntimeError
bad axis values raise ValueError
```

## Campaign 9 Final Decision

Campaign 9 rejects the true grouping-returning
`DevicePauliSum.group_commuting_device(...)` surface. The decision is not a
performance rejection of fused grouping itself; it is an API-contract rejection.
The campaign did not accept exact group return shapes, stable ordering relative
to CPU grouping, ownership of returned host or device metadata, stream/lifetime
rules, or allocation guardrails for a public grouping object.

Campaign 9 does implement the compact summary surface
`DeviceCommutationMatrix.conflict_degrees(axis=None|0|1)`. That method is a
public complement to `count_commuting(...)`; it copies only compact `uint64`
counts to host and does not expose a grouping API. The benchmark row
`docs/benchmarks/data/cuda_deferred_headroom_campaign9_2026-04-29/raw/public_grouping_api.json`
records this split with final status `rejected_with_evidence` for the true
public grouping item and with timing evidence for `conflict_degrees`.

No installed public C++ declaration or Python export named
`group_commuting_device` is accepted by Campaign 9.

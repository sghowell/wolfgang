# CUDA Fused Grouping Public API Campaign 8 Review

Status: public fused grouping API deferred; private benchmark probes retained.

Campaign 8 evaluates device-resident graph and grouping consumers after
Campaign 7 showed that full CSR anti-commutation edge-list export is the wrong
high-scale public boundary. The initial Campaign 8 decision is conservative:
FastPauli may benchmark compact grouping metadata privately, but no new public
method is exposed until exact semantics are accepted.

## Candidate Surfaces

```text
compact grouping summary copied to host
device-resident grouping metadata exposed through CUDA Array Interface
DLPack-exported grouping metadata
full CSR anti-commutation graph export
device-resident graph consumer handle with no full edge-list host copy
```

## Decision

`public_grouping_api_status: deferred`

Reason: Campaign 8 must first prove that compact grouping metadata is useful,
stable, and ownership-safe without requiring full CSR host export. The retained
implementation is a private benchmark hook only:
`fastpauli._fastpauli_core._benchmark_cuda_device_resident_consumer`.

No declaration is added to installed public C++ headers and no symbol is
re-exported from `fastpauli.__init__`.

## Required Contract Before Promotion

A future public fused grouping API must define:

```text
exact Python method name and C++ method name
exact return type, shape, dtype, and ownership
commuting or anti-commuting convention
stable ordering of rows, columns, groups, or conflict metadata
device and stream synchronization semantics
host copy size and transfer boundary
CPU-only error behavior
moved-from object behavior
memory allocation limit and failure mode
correctness oracle against a CPU reference
benchmark labels for fill, device-resident consumer, compact host copy, and full to_host()
API stability status and documentation requirements
```

## Current Private Boundary

The Campaign 8 private grouping probe may return compact host metadata:

```text
top_row_indices, top_row_conflicts
top_col_indices, top_col_conflicts
row_conflict_sum, col_conflict_sum
compact_host_bytes
correctness_digest
```

Ordering is deterministic: descending conflict count, with the original row or
column index as the tie-breaker. The convention remains `0 == anti-commuting`
and `1 == commuting` for `DeviceCommutationMatrix`.

## Rejected Public Boundaries For This Campaign

Full CSR anti-commutation graph export is not accepted as the primary public
high-scale grouping API. It may remain a validation-only or baseline benchmark
row, because its host output is O(edges) and Campaign 7 evidence showed that
edge-list materialization dominates the retained graph-export shape.

Device-resident grouping handles, raw device pointers, and CUDA Array Interface
views over newly invented grouping metadata are deferred because the lifetime,
mutability, synchronization, and moved-from behavior are not yet accepted as a
stable public contract.

## CPU-Only Behavior

CPU-only benchmark callers receive `status="unavailable"` with CUDA rebuild
guidance. CUDA-required tests raise the existing FastPauli CUDA rebuild
`RuntimeError`. No CPU fallback is introduced for the private CUDA benchmark
hook.

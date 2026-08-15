# CUDA Bit-Packed Commutation Campaign 7 Decision

Status: deferred; no Campaign 7 bit-packed public output or benchmark prototype
is retained.

## Trigger Evidence

Campaign 7 tested dense-layout fused consumers first:

```text
anti-commutation CSR graph construction from DeviceCommutationMatrix
row and column conflict-degree summaries
grouping-oriented top-k conflict summaries
```

The retained grouping-oriented workflow copies only compact `uint64` summaries
to host and does not hit a dense-matrix capacity limit on the H100 stress rows.
The CSR graph workflow materializes a large edge list, but the evidence points
to graph-output size and host transfer as the dominant cost rather than a dense
`uint8` matrix capacity failure. A bit-packed matrix that immediately unpacks
into CSR or compact summaries would add layout complexity without proving an
end-to-end win.

## Decision Gate Result

Bit-packed output remains deferred because Campaign 7 did not prove any of the
required triggers:

```text
dense DeviceCommutationMatrix memory capacity prevents a target stress or extreme workload from running
dense fused CSR or grouping kernels are bandwidth-bound in a way a packed layout can reduce end-to-end time
dense host or device copy size is the measured bottleneck and packed layout avoids that copy without immediate unpacking
```

The benchmark status field for this campaign is:

```text
deferred_no_dense_capacity_or_bandwidth_trigger
```

## Future Layout Requirements If Reopened

Any retained packed layout must define:

```text
row-major uint64 words
word count per row
bit order inside each word
padding bits and their required value
bit meaning for commuting versus anti-commuting
CUDA Array Interface support or explicit rejection
host materialization behavior and synchronization
which fused consumer uses packed data without unpacking to dense flags
correctness oracle against dense DeviceCommutationMatrix
```

No public API changes are made by this decision.

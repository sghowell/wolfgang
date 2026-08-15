# CUDA CSR Scatter Campaign 9 Reopen Decision

Status: final. Privileged Nsight Compute evidence passed; CSR scatter reopen
`rejected_with_evidence`.

Campaign 9 reopens CSR scatter tuning only if a retained Campaign 9 consumer
exports or internally consumes full CSR edge lists and privileged Nsight
Compute evidence proves scatter is material to end-to-end retained-consumer
time.

## Reopen Condition

```text
a retained Campaign 9 consumer exports or internally consumes full CSR edge lists
Nsight Compute shows scatter kernels materially affect end-to-end retained-consumer time
```

## Reject Condition

```text
retained consumers remain compact and avoid full CSR scatter
scatter is not a top profiler bottleneck
scatter improves only an unretained full CSR baseline
```

## Material Threshold

Scatter tuning is worth implementation only if projected improvement is:

```text
at least 10% on one retained high-scale row
or at least 5% on the broad landscape row it affects
```

## Allowed Implementation Candidates If Reopened

```text
coalesced row-chunk scatter for dense high-conflict rows
CUB prefix-sum temporary-storage reuse if allocation appears material in Nsight Compute
vectorized index writes only if alignment and bounds are proven
```

Raw PTX, atomic-heavy scatter that changes deterministic ordering, and tuning
that improves only an unretained full CSR baseline remain rejected.

## Campaign 9 Final Decision

Privileged Nsight Compute capture succeeded on the H100 host for retained
compact graph and grouping consumers. The retained consumers still avoid full
CSR edge-list materialization; CSR scatter appears only in the unretained
Campaign 7 full-CSR export baseline. Campaign 9 therefore rejects CSR scatter
tuning for production because it would optimize a boundary that the retained
high-scale path deliberately avoids.

Checked evidence:

```text
docs/benchmarks/data/cuda_deferred_headroom_campaign9_2026-04-29/profiler/ncu_campaign9_compact_consumers_details.csv
docs/benchmarks/data/cuda_deferred_headroom_campaign9_2026-04-29/raw/privileged_ncu_compact_consumers.json
docs/benchmarks/data/cuda_deferred_headroom_campaign9_2026-04-29/raw/csr_scatter_ab.json
```

The binary Nsight Compute `.ncu-rep` is not checked in because it is too large
for normal source control; the CSV/stdout/stderr evidence is checked in and the
report records the remote binary path.

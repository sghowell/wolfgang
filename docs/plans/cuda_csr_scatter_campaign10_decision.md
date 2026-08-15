# CUDA CSR Scatter Campaign 10 Decision

Status: rejected with evidence for Campaign 10.

Campaign 10 rechecked whether CSR scatter tuning should be reopened after the
Campaign 8 and Campaign 9 compact-consumer work moved the retained path away
from full CSR edge-list materialization.

## Acceptance Contract

CSR scatter optimization is acceptable only when a retained public or private
consumer requires full CSR edge lists on device or host. A benchmark-only full
CSR export baseline is insufficient on its own.

Before reopening CSR scatter, the candidate must document:

```text
the retained consumer that needs full CSR edge lists
the required CSR layout and ownership boundary
the correctness oracle for row offsets and column indices
the expected memory footprint at stress and extreme scales
the profiler evidence showing scatter is dominant after required consumers are retained
```

## Campaign 10 Evidence

Campaign 10 retained the compact graph/grouping summaries and measured full CSR
export only as a baseline.

```text
A100 8192x8192 compact graph consumer: 0.001042617 s
A100 8192x8192 compact grouping consumer: 0.001056928 s
A100 8192x8192 full CSR export baseline: 0.101427140 s

RTX PRO 6000 Blackwell 8192x8192 compact graph consumer: 0.000558464 s
RTX PRO 6000 Blackwell 8192x8192 compact grouping consumer: 0.000574324 s
RTX PRO 6000 Blackwell 8192x8192 full CSR export baseline: 0.056712325 s
```

The full CSR path remains much more expensive than the retained compact
consumer boundary, and no Campaign 10 retained consumer requires full CSR edge
lists.

## Decision

Do not tune CSR scatter in Campaign 10.

Future CSR scatter work must start from a retained consumer that truly requires
full CSR edge-list materialization. Otherwise, optimization effort should stay
focused on compact device-resident summaries and host materialization
boundaries.

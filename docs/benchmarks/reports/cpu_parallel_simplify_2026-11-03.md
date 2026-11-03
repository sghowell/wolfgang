# CPU Parallel Simplify With oneTBB

Date: 2026-11-03

This report qualifies slice 6 of the usability and kernel campaign
(`docs/plans/usability_and_kernel_campaign_plan.md`). When Wolfgang is built
with oneTBB, `PauliSum.simplify` can run a parallel sort-and-reduce kernel:

- terms are sorted by (packed key, input index) with `tbb::parallel_sort`;
- each duplicate run is summed sequentially in input order;
- runs are reduced in parallel 4,096-entry chunks and concatenated in key order.

The result equals scalar simplify bit for bit. `auto` dispatch uses it from
8,192 terms, and `WOLFGANG_CPU_BACKEND=tbb` forces it. Release wheels built
without oneTBB keep the scalar path.

Before timing, every oneTBB and `auto` result was checked for structural
equality with the scalar result, and Qiskit's result for the same term set.

## Evidence

```text
summaries: docs/benchmarks/data/cpu_parallel_simplify_2026-11-03/
  summary_x86_64_26vcpu_onetbb2021_11.json      (Xeon host, sizes 2,000 to 1,000,000)
  summary_epyc9654_16vcpu_onetbb2021_5.json     (EPYC instance, sizes 2,000 to 1,000,000)
  summary_epyc9654_16vcpu_threshold_sweep.json  (EPYC instance, sizes 1,000 to 65,536)
per-run reports: private archive (docs/benchmarks/data/**/raw/ is not published)
benchmark: benchmarks/bench_cpu_parallel_simplify.py
```

| Host | CPU | oneTBB | Compiler | Python |
| --- | --- | --- | --- | --- |
| Xeon host | Intel Xeon Platinum 8480+, 26 vCPU, Linux 6.8 | 2021.11 | GCC 13.3 | 3.12.3 |
| EPYC instance | AMD EPYC 9654, 16 vCPU, Linux 5.15 | 2021.5 | GCC 11.4 | 3.10.12 |

Every run measured revision `e7cecd6`, the final kernel with the signed-zero
fix and the four-thread auto guard. Trees were clean, and each run's loaded
extension matched its checkout's native sources. Builds used the portable
scalar CPU baseline (no native flags) and uncontrolled oneTBB thread counts,
which is the default.

Commands (3 runs per host; 1 warmup and 7 timed repetitions; variants in
randomized order each repetition):

```text
python benchmarks/bench_cpu_parallel_simplify.py --seed <20261104+run> --output run_<n>.json
python benchmarks/bench_cpu_parallel_simplify.py --sizes 2000,4000,8000,10000,30000,100000,1000000 \
    --seed <20261104+run> --output run_<n>.json            # Xeon host
python benchmarks/bench_cpu_parallel_simplify.py --no-qiskit --sizes 1000,2000,4000,8000,16000,32000,65536 \
    --seed <20261204+run> --output run_<n>.json            # EPYC threshold sweep
python benchmarks/bench_cpu_parallel_simplify.py --summarize <dir> --output <summary>
```

Operators are random dense labels with complex normal coefficients at 24, 100
and 130 qubits, with duplicate fractions 0.0, 0.5 and 0.9. Qiskit is
`SparsePauliOp.simplify(atol=1e-12)` from Qiskit 1.4.5 on the same terms.

## Results

Ratios of medians of per-run medians, min-max over the 9 width and duplicate
combinations at each size:

| Terms | oneTBB vs scalar (Xeon) | oneTBB vs scalar (EPYC) | oneTBB vs Qiskit (Xeon) | oneTBB vs Qiskit (EPYC) |
| ---: | ---: | ---: | ---: | ---: |
| 2,000 | 0.94-1.36x | 0.50-0.95x | 1.59-2.59x | 1.35-2.04x |
| 10,000 | 1.75-2.58x | 1.61-2.09x | 1.99-2.98x | 1.69-2.62x |
| 30,000 | 2.33-3.34x | 2.03-2.71x | 2.33-3.80x | 1.90-2.81x |
| 100,000 | 3.17-4.36x | 2.62-3.37x | 2.97-4.86x | 2.31-3.57x |
| 1,000,000 | 4.01-5.66x | 2.89-3.66x | 4.67-7.58x | 2.11-4.66x |

`auto` tracks scalar below 8,192 terms and oneTBB above it on both hosts.

Representative absolute EPYC medians at one million terms, duplicate fraction
0.5:

| Qubits | Scalar | oneTBB | Qiskit |
| ---: | ---: | ---: | ---: |
| 24 | 124 ms | 41.0 ms | 173 ms |
| 100 | 129 ms | 43.7 ms | 204 ms |
| 130 | 242 ms | 71.4 ms | 225 ms |

The full matrices are in the summaries.

## Auto threshold

oneTBB / scalar speedup, min / median / max over the 9 combinations:

| Terms | EPYC (16 vCPU) | Xeon (26 vCPU) |
| ---: | ---: | ---: |
| 1,000 | 0.45 / 0.57 / 0.70 | not measured |
| 2,000 | 0.55 / 0.70 / 1.06 | 0.94 / 1.18 / 1.36 |
| 4,000 | 0.97 / 1.15 / 1.51 | 1.34 / 1.56 / 1.90 |
| 8,000 | 1.41 / 1.58 / 2.02 | 1.61 / 1.96 / 2.47 |
| 10,000 | not measured | 1.75 / 2.00 / 2.58 |
| 16,000 | 1.73 / 1.98 / 2.25 | not measured |
| 65,536 | 2.60 / 2.64 / 2.76 | not measured |

The EPYC crossover lies between 4,000 and 8,000 terms; the Xeon host crosses
near 2,000. `kAutoTbbSimplifyTermThreshold = 8192` is the smallest measured
size at which oneTBB wins every combination on both hosts. Auto also requires
at least four oneTBB threads (`kAutoTbbSimplifyMinConcurrency`), because with
one or two threads the parallel kernel does strictly more work than scalar.

Timings at 1,000-4,000 terms are 0.1-0.5 ms per call, so single-call medians
are noisy: on EPYC, `auto` and scalar run the same code at 2,000 terms yet their
medians differ by 0.72-1.35x. The threshold rests on the 8,000-term row, where
every combination is above 1.4x on both hosts.

## Correctness

| Check | Result |
| --- | --- |
| `tests/test_cpu_parallel_simplify.py` | Forced oneTBB and `auto` equal scalar structurally and in raw coefficient bits (signed zeros included, also on `-op`). Covers 3-130 qubits, 1-70,000 terms, duplicate fractions 0.0-0.95, drop rules, cancellation, overflowing magnitudes, a 30,000-entry run spanning several chunks, all-identical input and an overflow raised from the last chunk on a worker thread. Passes on both hosts. |
| Native contract | Includes duplicate runs that cross chunk boundaries. Passes under AddressSanitizer and UndefinedBehaviorSanitizer with oneTBB on the EPYC instance. |
| `python scripts/validate.py --profile core` | Passes on the EPYC instance with oneTBB detected (723 tests). |

Every CPU simplify path now seeds each duplicate run with its first
coefficient (the hash paths, fused matmul and CUDA already did). Before this,
the scalar sort path turned a lone `-0.0` component into `+0.0` while the hash
path kept it, so scalar simplify's signed zeros depended on which path its
heuristics picked.

## Limitations

- x86_64 only. oneTBB was not available on the Apple Silicon development
  machine, so ARM has no oneTBB evidence (deferral recorded in the plan).
- Release wheels do not bundle oneTBB. These speedups apply to source builds
  where CMake finds oneTBB.
- Parallel speedup depends on core count and memory bandwidth. Hosts with fewer
  cores than the two measured here may cross over later than 8,192 terms.
- Peak memory during the oneTBB kernel exceeds scalar's: sort keys, per-chunk
  (position, coefficient) outputs and the result coexist. Measured later on the
  EPYC host at 1M terms (24 and 130 qubits, 3 fresh processes each), the
  memory one simplify call adds was 1.34-1.41x scalar's with no or half
  duplicates (14-25 extra bytes per input term) and 1.10-1.17x with 90%
  duplicates (2-3 extra bytes per term)
  (`docs/benchmarks/data/cpu_parallel_simplify_2026-11-03/peak_memory_epyc.json`).
- Random dense operators only.

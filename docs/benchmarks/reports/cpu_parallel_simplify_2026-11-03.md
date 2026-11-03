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
  summary_x86_64_26vcpu_onetbb2021_11.json   (H100 host CPU, threshold 65536 at the time)
  summary_epyc9654_16vcpu_onetbb2021_5.json  (dedicated CPU instance, final threshold 8192)
  summary_epyc9654_16vcpu_threshold_sweep.json
per-run reports: private archive (docs/benchmarks/data/**/raw/ is not published)
benchmark: benchmarks/bench_cpu_parallel_simplify.py
```

| Host | CPU | oneTBB | Compiler | Python | Revision |
| --- | --- | --- | --- | --- | --- |
| H100 host | x86_64, 26 vCPU | 2021.11 | GCC 13.3 | 3.12.3 | `87af13e` |
| EPYC instance | AMD EPYC 9654, 16 vCPU | 2021.5 | GCC 11.4 | 3.10.12 | `3590e4a` |

Every run's loaded extension matched its checkout's native sources, and every
tree was clean. The `87af13e` and `3590e4a` kernels are identical; the later
revision only changes the auto threshold (65,536 to 8,192) and its comment.
The H100-host `auto` column therefore reflects the old threshold.

Commands (3 runs per host, seed `20261104 + run`; 1 warmup and 7 timed
repetitions; variants in randomized order each repetition):

```text
python benchmarks/bench_cpu_parallel_simplify.py --seed <s> --output run_<n>.json
python benchmarks/bench_cpu_parallel_simplify.py --no-qiskit --sizes 1000,2000,4000,8000,16000,32000,65536 --seed <20261204+run> ...
python benchmarks/bench_cpu_parallel_simplify.py --summarize <dir> --output <summary>
```

Operators are random dense labels with complex normal coefficients at 24, 100
and 130 qubits, with duplicate fractions 0.0, 0.5 and 0.9. Qiskit is
`SparsePauliOp.simplify(atol=1e-12)` from Qiskit 1.4.5 on the same terms.

## Results

Ratios of medians of per-run medians, min-max over the 9 width and duplicate
combinations at each size:

| Terms | oneTBB vs scalar (H100 host) | oneTBB vs scalar (EPYC) | oneTBB vs Qiskit (EPYC) |
| ---: | ---: | ---: | ---: |
| 2,000 | 0.84-1.32x | 0.52-0.88x | 1.15-2.01x |
| 10,000 | 1.47-2.47x | 1.37-1.88x | 1.47-2.51x |
| 30,000 | 1.96-3.11x | 1.73-2.39x | 1.68-2.55x |
| 100,000 | 2.53-4.27x | 2.16-2.88x | 2.06-3.13x |
| 1,000,000 | 3.29-5.63x | 2.39-3.87x | 2.38-3.88x |

Representative absolute EPYC medians at one million terms:

| Qubits | Duplicate fraction | Scalar | oneTBB | Qiskit |
| ---: | ---: | ---: | ---: | ---: |
| 24 | 0.5 | 125 ms | 47.7 ms | 158 ms |
| 130 | 0.5 | 258 ms | 71.2 ms | 224 ms |

The full matrices are in the summaries.

## Auto threshold

The EPYC sweep places the crossover between 4,000 and 8,000 terms:

| Terms | oneTBB / scalar speedup (min / median / max over 9 combinations) |
| ---: | ---: |
| 1,000 | 0.38 / 0.54 / 0.64 |
| 2,000 | 0.37 / 0.64 / 0.75 |
| 4,000 | 0.80 / 1.10 / 1.42 |
| 8,000 | 1.26 / 1.54 / 1.66 |
| 16,000 | 1.54 / 1.73 / 2.06 |
| 32,000 | 1.87 / 1.96 / 2.40 |
| 65,536 | 2.16 / 2.25 / 2.61 |

On the 26-vCPU host every combination is at least 1.47x faster from 10,000
terms. `kAutoTbbSimplifyTermThreshold = 8192` is the smallest size where both
hosts win in every combination. With the final threshold, EPYC `auto` medians
track scalar at 2,000 terms and oneTBB above the threshold.

## Correctness

| Check | Result |
| --- | --- |
| `tests/test_cpu_parallel_simplify.py` | Structural equality between forced oneTBB and scalar at 3, 24, 64, 100 and 130 qubits, 1-70,000 terms and duplicate fractions 0.0-0.95. Also drop rules, cancellation, overflowing magnitudes and accumulation overflow. Passes on both hosts. |
| Native contract | Includes duplicate runs that cross chunk boundaries. Passes under AddressSanitizer and UndefinedBehaviorSanitizer with oneTBB on the EPYC instance. |
| `python scripts/validate.py --profile core` | Passes on the EPYC instance with oneTBB detected (697 tests). |

## Limitations

- x86_64 only. oneTBB was not installed on the Apple Silicon development
  machine, so ARM has no oneTBB evidence.
- Release wheels do not bundle oneTBB. These speedups apply to source builds
  where CMake finds oneTBB.
- Parallel speedup depends on core count and memory bandwidth. Hosts with fewer
  cores than the two measured here may cross over later than 8,192 terms.
- Random dense operators only.

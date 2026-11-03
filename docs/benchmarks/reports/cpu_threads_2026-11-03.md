# CPU Thread Cap

Date: 2026-11-03

This report qualifies slice 4 of the accelerator follow-ups campaign
(`docs/plans/accelerator_followups_campaign_plan.md`). Release wheels bundle
oneTBB (`docs/benchmarks/reports/onetbb_wheels_2026-11-03.md`), so `auto`
runs parallel simplify and pairwise commutation on every core the process may
use, with no way to ask for fewer.

## What changed

- `wolfgang_quantum.set_num_threads(n)` caps the oneTBB kernels at `n`
  threads for the whole process, through a oneTBB `global_control`. A later
  call replaces the cap.
- `WOLFGANG_NUM_THREADS=n` sets the same cap without code changes. It is read
  the first time Wolfgang needs it; a value that is not a positive integer
  raises `ValueError` from `get_num_threads()` and from dispatch, as an invalid
  `WOLFGANG_CPU_BACKEND` does.
- `wolfgang_quantum.get_num_threads()` returns the threads the kernels may use:
  oneTBB's default concurrency (hardware threads, respecting the affinity
  mask) or the cap if it is lower; 1 without oneTBB or in a child forked after
  oneTBB ran.
- Auto dispatch's four-thread oneTBB floor now uses that count, so a cap below
  four sends simplify and pairwise commutation to the scalar or SIMD kernels.
  Before, it read the arena's concurrency, which a `global_control` cap does
  not change.
- A cap only installs a `global_control` when it is below oneTBB's default
  concurrency. A larger one would raise oneTBB's own worker limit, and oneTBB
  narrows the value to 32 bits internally, so caps near 2^31 and above made
  its kernels run serially or not finish; such caps now mean no cap.
- `_build_info()` reports `cpu_thread_limit` and `cpu_auto_tbb_eligible`.

## Evidence

```text
benchmarks: benchmarks/bench_cpu_threads.py --processes <4|8>
            benchmarks/probe_simplify_threads.py <cap> <calls>
data:       docs/benchmarks/data/cpu_threads_2026-11-03/shared_4_processes.json
            docs/benchmarks/data/cpu_threads_2026-11-03/shared_8_processes.json
            docs/benchmarks/data/cpu_threads_2026-11-03/quota_and_low_caps.json
```

| Item | Value |
| --- | --- |
| Host | Intel Xeon Platinum 8480+ (KVM guest), 26 vCPUs, Ubuntu 24.04, GCC 13.3 |
| oneTBB | 2021.11.0 (system package; release wheels bundle 2022.3.0) |
| Build | `c5ee325`, CUDA build with oneTBB, CPU timings only |
| Operator | 500,000 terms, 24 qubits, half of them duplicates |

## Results

One process, median simplify time by cap:

| Cap | Threads | Kernel | Median |
| --- | ---: | --- | ---: |
| 1 | 1 | scalar (below the floor) | 60.7 ms |
| 2 | 2 | scalar (below the floor) | 58.0 ms |
| 4 | 4 | oneTBB | 22.6 ms |
| 8 | 8 | oneTBB | 17.9 ms |
| 16 | 16 | oneTBB | 14.8 ms |
| none (cap 1,000,000) | 26 | oneTBB | 14.2 ms |

Inside a CPU quota smaller than the visible cores, the case the cap is for
(`systemd-run --scope -p CPUQuota=400%`, so 4 CPUs of 26), over 60 calls in two
runs each:

| | Median | Mean | 90th percentile | Max |
| --- | ---: | ---: | ---: | ---: |
| No cap (26 threads) | 14.5-14.6 ms | 27.8-28.4 ms | 59.4-61.4 ms | 70.6 ms |
| Cap 4 | 22.4-22.5 ms | 22.5-22.6 ms | 22.7-22.8 ms | 25.5 ms |

Uncapped, oneTBB starts 26 threads; calls that fit in the quota period run fast,
and the rest stall when the quota throttles the process. The cap trades the
best case for steady calls: the median is 1.55x slower, while mean throughput
is 1.25x better and the 90th percentile 2.6x better. Without a quota, capping at 4 costs what the scaling
table shows (22.7 against 14.2 ms in these runs).

Processes sharing the cores without a quota (each worker runs 10 calls; wall
time until the last worker finishes):

| Processes | No cap | Capped at cores / processes |
| ---: | ---: | ---: |
| 4 | 0.52 s (26 threads each) | 0.51 s (6 threads each) |
| 8 | 0.80 s (26 threads each) | 1.01 s (3 threads each, scalar) |

On this host oneTBB tolerated oversubscription: four uncapped processes ran as
fast as four capped ones. With eight processes the cap of 3 fell below the
floor and sent `auto` to the scalar kernel, which was slower than oversubscribed
oneTBB.

## The four-thread floor

The floor came from the CPU parallel simplify slice
(`docs/benchmarks/reports/cpu_parallel_simplify_2026-11-03.md`), which reasoned
that one or two oneTBB threads do strictly more work than scalar; it was not
measured. With the cap it can now be measured directly (forced
`WOLFGANG_CPU_BACKEND=tbb`, 30 calls, `quota_and_low_caps.json`):

| Threads | oneTBB | Scalar |
| ---: | ---: | ---: |
| 1 | 63.8 ms | 61.8 ms |
| 2 | 38.4 ms | |
| 3 | 26.0 ms | |
| 4 | 22.4 ms | |

At 500,000 terms, two and three oneTBB threads beat scalar by 1.6x and 2.4x,
and `auto` with a cap of 3 ran scalar (62.0 ms). The floor is unchanged in this
slice: it also governs pairwise commutation, where the alternative is a SIMD
kernel that was not measured here, and moving it needs measurements on more
than one host and at smaller sizes. Until then, processes capped below four
threads run the scalar or SIMD kernels; forcing `WOLFGANG_CPU_BACKEND=tbb` is
not a workaround, because operations without a oneTBB kernel then raise.

## Correctness

| Check | Result |
| --- | --- |
| `tests/test_cpu_threads.py` (new) | Exports and stubs; validation of `set_num_threads` (0, negative, oversized, float, string, `None`, `bool`) and NumPy integers; the cap, its effect on the oneTBB floor, caps at or above the default concurrency (up to 2^40) behaving as no cap; `WOLFGANG_NUM_THREADS` (valid, invalid, replaced by `set_num_threads`, invalid on a forced-oneTBB commutation); a child forked after oneTBB ran reports one thread; on Linux, a cap of 4 adds at most three threads; identical result bytes at 1, 2, 3, 4 and 64 threads under auto and forced oneTBB |
| Existing suites | `test_cpu_parallel_simplify.py`, `test_cpu_fork_safety.py`, `test_public_typing_and_capabilities.py` pass with oneTBB on the 26-vCPU host and without it on macOS |

## Limitations

- One host. Oversubscription cost and the floor's crossover depend on the
  CPU, the oneTBB version and the scheduler.
- oneTBB sizes itself from the affinity mask, not from cgroup CPU quotas, so in
  quota-limited containers the cap has to be set explicitly.
- The cap is process-wide; it cannot give different calls different thread
  counts.

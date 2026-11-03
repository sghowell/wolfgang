# HIP Simplify With Device-Side Counts And Cached Scratch

Date: 2026-11-03

This report qualifies slice 2 of the accelerator follow-ups campaign
(`docs/plans/accelerator_followups_campaign_plan.md`). Slice 1 moved CUDA
simplify beyond the fused kernel to a pipeline that keeps its counts on the
device (`docs/benchmarks/reports/cuda_device_counts_h100_2026-11-03.md`). The
pipeline lives in `src/detail/device_simplify_pipeline.cuh`, which builds with
hipCUB under hipcc, and HIP now uses it too. Two changes are measured
separately:

1. **Pipeline.** HIP simplify for operators the fused kernel does not take
   runs the shared pipeline instead of rocThrust stages that returned the
   input maximum, the run count and the survivor count through blocking
   copies.
2. **Per-thread block cache.** The pipeline's single scratch block, and
   rocThrust's algorithm temporaries elsewhere, come from a per-thread cache
   of up to two freed blocks (at most 256 MiB each) instead of a
   `hipMalloc`/`hipFree` pair per call.

The second change exists because the first, on its own, was slower than the
baseline at 50,000 terms (see below).

## Evidence

```text
benchmarks: benchmarks/bench_cuda_call_overhead.py --backend hip
            benchmarks/bench_cuda_simplify_paths.py --backend hip --suite all
profile:    rocprofv3 --hip-runtime-trace --kernel-trace --memory-copy-trace --stats,
            50 warmup + 500 profiled calls per operation
data:       docs/benchmarks/data/hip_device_counts_mi300x_2026-11-03/overhead_summary.json
            docs/benchmarks/data/hip_device_counts_mi300x_2026-11-03/paths_summary.json
            docs/benchmarks/data/hip_device_counts_mi300x_2026-11-03/api_breakdown.json
            docs/benchmarks/data/hip_device_counts_mi300x_2026-11-03/alloc_probe_50000.json
```

| Item | Value |
| --- | --- |
| Device | AMD Instinct MI300X VF, gfx942, HIP 7.16.26385, amdgpu 6.19.14 |
| Host | Intel Xeon Platinum 8568Y+ (KVM guest), Ubuntu 24.04.5, AMD clang 24.0.0 (ROCm), Python 3.12.16 |
| Baseline (`base`) | main at `ad6f7a8`, whose HIP sources match the slice's merge base `4ca82d5`; its checkout also held the candidate's `bench_cuda_simplify_paths.py` (the `--backend` option) |
| Pipeline only (`pipe`) | `9af4918` |
| Pipeline and cache (`cache`) | `4ca8655` |
| Build | `WOLFGANG_HIP_ARCHITECTURES=gfx942`, `hipMalloc` allocator (the optional pool off) |

The call-overhead suite ran 7 times and the simplify-path suite 3 times
(seeds 20261201-20261203), with the order of the three builds shuffled per
run.

## Results

Call overhead, device-resident 24-qubit operators. Medians of per-run medians,
with the min-max of per-run medians:

| Operation | Baseline | Pipeline only | Pipeline and cache | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `simplify`, 4,096 terms | 181.7 us (179.4-184.2) | 99.1 us (98.5-99.8) | 97.9 us (97.3-98.3) | 1.86x |
| `matmul`, 128 x 128 terms | 201.6 us (199.5-205.4) | 123.9 us (123.5-124.9) | 123.8 us (121.8-124.3) | 1.63x |

The other thirteen rows (fused, single-term and wide simplify below the
pipeline, transfers, expectation, commutation and the CPU control) stayed
within 1.00-1.01x.

Simplify paths, 50% duplicates, median of three runs (ms; speedup against the
baseline):

| Path | Terms | Timing | Baseline | Pipeline only | Pipeline and cache |
| --- | ---: | --- | ---: | ---: | ---: |
| One word, packed key (24 qubits) | 50,000 | device-resident | 0.270 | 0.310 (0.87x) | 0.158 (1.71x) |
| | | including transfers | 0.465 | 0.566 (0.82x) | 0.321 (1.45x) |
| | 1,000,000 | device-resident | 1.556 | 0.731 (2.13x) | 0.579 (2.69x) |
| | | including transfers | 6.209 | 7.041 (0.88x) | 3.557 (1.75x) |
| One word, two masks (60 qubits) | 50,000 | device-resident | 0.355 | 0.357 (0.99x) | 0.190 (1.86x) |
| | | including transfers | 0.490 | 0.658 (0.74x) | 0.363 (1.35x) |
| | 1,000,000 | device-resident | 2.080 | 1.468 (1.42x) | 1.031 (2.02x) |
| | | including transfers | 6.360 | 6.231 (1.02x) | 4.017 (1.58x) |
| Two words (100 qubits) | 50,000 | device-resident | 0.518 | 0.523 (0.99x) | 0.236 (2.19x) |
| | | including transfers | 0.690 | 0.939 (0.73x) | 0.448 (1.54x) |
| | 1,000,000 | device-resident | 2.503 | 1.970 (1.27x) | 1.558 (1.61x) |
| | | including transfers | 8.340 | 9.141 (0.91x) | 5.342 (1.56x) |

For operators wider than two words, on the rows that reach the pipeline (130
and 256 qubits from 4,096 terms, 1,024 qubits from 128 terms), the pipeline
and cache are 1.27-2.17x faster device-resident and 1.07-1.55x including
transfers. Rows that take the fused kernel or the single-term path stayed
within 0.96-1.02x of the baseline.

## Why the cache is needed

With the pipeline alone, simplify at 50,000 terms got slower even though its
GPU work shrank. The allocation probe at 50,000 terms
(`alloc_probe_50000.json`) shows why. The pipeline makes one scratch
allocation of about 3 MB per call instead of several smaller ones. `hipFree`
on that block took 95 us at the 90th percentile, against 19 us for the
baseline's smaller blocks. A full pass of 90 calls spent about as long in
`hipFree` with 498 calls (9.0 ms) as the baseline did with 1,308 (10.3 ms). ROCm
caches small blocks, which is why the earlier pool measurement
(`docs/benchmarks/reports/hip_parity_mi300x_2026-11-03.md`) favored plain
`hipMalloc`, but it does not cache blocks this large. Reusing the block per
thread removes the cost: at 50,000 terms the cache build is 1.7-2.2x faster
than the baseline where the pipeline alone was 0.87-0.99x.

## Where the time goes

Per call on 4,096 terms (24 qubits), from the rocprofv3 traces:

| | Baseline | Pipeline only | Pipeline and cache |
| --- | ---: | ---: | ---: |
| Blocking device-to-host copies | 3 (about 20 us each) | 0 | 0 |
| Stream and device synchronizations | 1 | 2 | 2 |
| `hipMalloc` / `hipFree` pairs | 13 | 4 | 3 |
| Kernel launches | 18 | 11 | 11 |
| GPU kernel time | 87.4 us | 76.1 us | 76.7 us |
| Wall time | 181.5 us | 99.8 us | 98.3 us |

The remaining allocations are the three output buffers. On HIP the radix
sort, reduce-by-key and scan come from rocPRIM through hipCUB, with fewer
launches than CUB on the H100 (11 against 17).

## Correctness

| Check | Result |
| --- | --- |
| `tests/test_device_simplify_pipeline.py` | All HIP cases pass with the fused kernel switched off (every key kind from 1 to 300 qubits, 100,000-term tile-spanning operators, overflow, non-finite precedence, half-unit rule, rounding, state between calls, threads, `matmul`) |
| `tests/test_hip_thread_cache.py` (new) | Scratch is reused across calls with no new block allocations; a request above the 256 MiB limit is not retained (about 5M terms); a thread's blocks are freed when it exits; with the optional pool enabled the cache is bypassed |
| Negative control | With the cache limit set to 0, the reuse and thread-exit tests fail; see the closeout |
| Existing suites | `test_accelerator_simplify_parity.py`, `test_accelerator_small_simplify.py`, `test_hip_pool_allocator.py` (including the pool's return-to-baseline test) pass |
| `WOLFGANG_VALIDATE_HIP=1 python scripts/validate.py --profile core` | Passed on the MI300X (945 tests in the HIP run) |

## Limitations

- One MI300X (a virtual function in a KVM guest) and one ROCm release.
  rocPRIM's algorithm choices and ROCm's allocator caching can change with the
  release.
- Each thread can keep up to 512 MiB of device memory in its cache (two
  blocks of 256 MiB) until it exits; in practice it holds the largest scratch
  it has needed, rounded up to a power of two.
- Allocation costs come from a KVM guest; bare-metal ROCm may show smaller
  `hipFree` costs for large blocks, and with them a smaller gain from the
  cache.

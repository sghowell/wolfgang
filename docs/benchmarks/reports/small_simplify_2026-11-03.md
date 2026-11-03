# Fused Small-Operator Simplify On CUDA And HIP

Date: 2026-11-03

This report qualifies slice 5 of the portability and interop campaign
(`docs/plans/portability_and_interop_campaign_plan.md`). After slices 1-2 of
this campaign and the earlier allocation work, a small device-resident
simplify still cost about 140 us on an H100 and 150 us on an MI300X. That cost
came from about ten kernel launches, several stream synchronizations and
thirteen allocations, not from the work itself.

Operators with one to four packed words (up to 256 qubits) and at most 1,024
terms (512 for three or four words) now run as one fused kernel, shared by both
backends (`src/detail/small_simplify_kernel.cuh`). The block size is the next
power of two of the term count, at least 64. The kernel:

1. loads the keys word-major into shared memory and reduces the largest
   magnitude, NaN-propagating for invalid input;
2. sorts (key, input index) with an all-ascending bitonic network; positions
   past the term count are sentinels;
3. gathers coefficients into sorted order and marks run heads once;
4. has each run head sum its run sequentially in input order and apply the CPU
   drop rule;
5. compacts survivors with a block scan.

The status and survivor count come back through pinned, device-mapped host
memory after one stream synchronization. The coefficients equal CPU simplify
bit for bit, signed zeros included. The multi-launch path's tree-ordered sums
match only to rounding. Survivor sets also match, except for coefficients
within the device `hypot`'s few-ulp error of the drop threshold.

The slice also makes every backend round `atol + rtol * max` as a separate
multiply and add. nvcc and HIP clang, and Apple clang and GCC on arm64 hosts,
otherwise contract it into a fused multiply-add. An FMA can move the threshold
by an ulp, so CPU simplify on Apple Silicon dropped a term that x86 kept, for
example `X: 1.3927415900879758, Z: 0.2785483180185952` with `rtol=0.2`.

## Evidence

```text
benchmark: benchmarks/bench_cuda_call_overhead.py --backend cuda|hip
summaries: docs/benchmarks/data/small_simplify_2026-11-03/summary_cuda_h100.json
           docs/benchmarks/data/small_simplify_2026-11-03/summary_hip_mi300x.json
           docs/benchmarks/data/small_simplify_2026-11-03/summary_api.json
per-run reports: private archive (docs/benchmarks/data/**/raw/ is not published)
```

| Item | CUDA | HIP |
| --- | --- | --- |
| GPU | NVIDIA H100 80GB HBM3, compute capability 9.0 | AMD Instinct MI300X VF, `gfx942` |
| Toolkit | CUDA 12.8.93, driver 580.105.08, GCC 13.3 host compiler | HIP 7.16.26385, AMD clang 24 |
| Build | `WOLFGANG_CUDA_ARCHITECTURES=90`, Release | `WOLFGANG_HIP_ARCHITECTURES=gfx942`, Release |
| Boundary | device-resident: operands already on the device, each call synchronizes before returning; `round_trip` and transfers are transfer-inclusive | same |

Commands: 7 runs per host, baseline and candidate in randomized order per run,
200 calls per batch and 7 batches per run.

```text
baseline (main after slice 2, d871133): python benchmarks/bench_cuda_call_overhead.py --backend <b> --label base --output <run>
candidate (b072c0a):                    python benchmarks/bench_cuda_call_overhead.py --backend <b> --label cand --output <run>
```

The baseline checkouts ran the candidate's benchmark script, which adds the
two capacity-edge variants, so their trees were dirty. Every run's loaded
extension matched its checkout's native sources.

## Results

Medians of per-run medians:

| Operation | H100 baseline | H100 candidate | Speedup | MI300X baseline | MI300X candidate | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| simplify, 128 terms, 24 qubits | 143.5 us | 18.4 us | 7.80x | 152.7 us | 26.9 us | 5.68x |
| simplify, 128 terms, 130 qubits | 117.3 us | 21.6 us | 5.44x | 171.1 us | 32.6 us | 5.25x |
| simplify, 1,024 terms, 24 qubits | 151.7 us | 35.4 us | 4.29x | 157.8 us | 44.6 us | 3.54x |
| simplify, 512 terms, 200 qubits | 126.2 us | 42.1 us | 3.00x | 186.8 us | 52.2 us | 3.58x |
| simplify, 1 term | 26.1 us | 17.2 us | 1.52x | 30.6 us | 30.5 us | 1.01x |
| round trip (to_device, simplify, to_host), 128 terms | 189.8 us | 63.6 us | 2.98x | 234.2 us | 104.4 us | 2.24x |
| simplify, 4,096 terms (not fused) | 188.7 us | 188.1 us | 1.00x | 178.2 us | 179.1 us | 0.99x |
| matmul, 128 x 128 (not fused) | 202.2 us | 202.3 us | 1.00x | 199.5 us | 199.4 us | 1.00x |
| to_device / to_host, commutes_with_device, expectation, CPU control | | | 0.99-1.00x | | | 0.99-1.01x |

Per-run ranges are in the summaries; every candidate row spans less than 2%.
HIP keeps its dedicated single-term path, which measured faster on the MI300X
than a one-term fused launch (31.3 against 32.8 us during development).

### Where the time went

API calls and time per simplify, 128 terms, 220 calls (`summary_api.json`):

| API | H100 baseline | H100 candidate | MI300X baseline | MI300X candidate |
| --- | --- | --- | --- | --- |
| kernel launches | 9 / 33.6 us | 1 / 3.7 us | 10 / 26.4 us | 1 / 2.9 us |
| stream synchronizations | 10 / 57.6 us | 1 / 12.9 us (waits for the kernel) | rocThrust value copies 3 / 43.2 us | 1 / 20.1 us (waits for the kernel) |
| other copies | 7 / 41.5 us | 0 | 2 / 5.4 us | 0 |
| allocations + frees | 26 / 22.9 us | 6 / 3.0 us | 26 / 59.9 us | 6 / 3.5 us |

What remains is mostly the kernel itself, which the single synchronization
waits for, plus three output allocations.

## Correctness

| Check | Result |
| --- | --- |
| `tests/test_accelerator_small_simplify.py` | Bit-identical to CPU simplify at 20-256 qubits, 2, 129 and capacity terms, duplicate fractions 0-0.97 and four tolerance modes; long runs; signed zeros; an all-Y key equal to the sentinel key; the capacity+1 boundary; the FMA threshold case; errors; the benchmark switch. Passes on both GPUs. |
| `tests/test_accelerator_simplify_parity.py` | Adds separately rounded threshold cases for CPU and every backend; passes on CUDA, HIP and Metal. |
| H100 | `WOLFGANG_VALIDATE_CUDA=1 python scripts/validate.py --profile core` at `b072c0a` passed (927 tests in the CUDA run). Compute Sanitizer memcheck, racecheck, synccheck and initcheck on the fused tests: 0 errors. |
| MI300X | `WOLFGANG_VALIDATE_HIP=1 python scripts/validate.py --profile core` at `b072c0a` passed (840 tests in the HIP run). |

An independent simulation of the sort network covered block sizes 2-1,024 and
widths 1-4, with duplicates and real all-ones keys, against a reference sort.

## Limitations

- One H100 and one MI300X. The kernel declares `__launch_bounds__(1024)`.
  Shared memory peaks at 56 KiB (two words, 1,024 terms), which needs the CUDA
  opt-in, set per launch. Any launch-configuration error falls back to the
  multi-launch path.
- Operators wider than four words or above capacity use the multi-launch path
  unchanged.
- One thread sums each duplicate run, so a single run of nearly 1,024 terms is
  summed serially from shared memory. That worst case was not benchmarked.
- The kernel is custom rather than CUB/hipCUB, an exception to the code
  standard's prefer-library rule. The library path is the multi-launch path
  measured here as the baseline.

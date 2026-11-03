# CUDA Allocation Reuse On H100

Date: 2026-11-03

This report measures slice 7 of the usability and kernel campaign
(`docs/plans/usability_and_kernel_campaign_plan.md`):

- `DevicePauliSum` buffers, Thrust device vectors, Thrust algorithm temporaries,
  and expectation and commutation scratch now come from a Wolfgang-private
  stream-ordered CUDA memory pool (`cudaMallocFromPoolAsync` /
  `cudaFreeAsync` on the legacy default stream) that keeps freed blocks cached;
- `to_device` and `commutes_with_device` select the backend with a device count
  instead of `cuda_status()`, which queried every device's properties per call;
- coefficients cross the host/device boundary without intermediate conversion
  vectors.

`DeviceCommutationMatrix` buffers, which can leave through DLPack and the CUDA
Array Interface, still use `cudaMalloc`.

## Evidence

```text
summaries: docs/benchmarks/data/cuda_allocation_reuse_h100_2026-11-03/summary_call_overhead.json
           docs/benchmarks/data/cuda_allocation_reuse_h100_2026-11-03/summary_simplify_paths.json
per-run reports: private archive (docs/benchmarks/data/**/raw/ is not published)
benchmarks: benchmarks/bench_cuda_call_overhead.py, benchmarks/bench_cuda_simplify_paths.py
```

Environment: NVIDIA H100 80GB HBM3 (compute capability 9.0, driver
580.105.08), CUDA toolkit 12.8.93, `sm_90`, x86_64 host with 26 logical CPUs,
GCC 13.3.0, Python 3.12.3. Every run's loaded extension matched its checkout's
native sources. The baseline checkout was dirty only because the overhead
benchmark script was copied into it.

The baseline is `main` at `23121d6` and the candidate is `98ef775`. The final
branch adds only an exit-time pool flush (`d1502d9`) that does not run during
timed calls.

```text
python benchmarks/bench_cuda_call_overhead.py --label <base|cand> --output ...      # 7 paired runs
python benchmarks/bench_cuda_simplify_paths.py --suite specialized --label <base|cand> --seed <20261120+run> --output ...  # 3 paired runs
```

The baseline/candidate order is randomized per run. The call-overhead benchmark
times 200 back-to-back calls per batch, 7 batches per variant, on small operands
so kernel time is negligible.

## Per-call overhead

Medians of per-run medians:

| Call | main | candidate | candidate / main |
| --- | ---: | ---: | ---: |
| `PauliSum.to_device()`, 1 term | 907.5 us | 16.4 us | 0.018 |
| `commutes_with_device`, 128 x 128 | 908.5 us | 64.3 us | 0.071 |
| round trip `to_device().simplify().to_host()`, 128 terms | 1,167.6 us | 258.8 us | 0.222 |
| `simplify`, 1 term (device-resident) | 33.0 us | 26.7 us | 0.809 |
| `expectation_statevector`, 64 terms, 12 qubits, host psi | 34.2 us | 30.1 us | 0.880 |
| `matmul`, 128 x 128, 24 qubits | 299.0 us | 281.1 us | 0.940 |
| `simplify`, 128 terms, 24 qubits | 224.9 us | 211.8 us | 0.942 |
| `simplify`, 128 terms, 130 qubits | 198.1 us | 187.7 us | 0.947 |
| `simplify`, 4096 terms, 24 qubits | 275.7 us | 265.4 us | 0.963 |
| `to_host()`, 1 term | 23.7 us | 23.8 us | 1.002 |
| CPU `simplify`, 128 terms (control) | 3.7 us | 3.8 us | 1.005 |

Device discovery was the cost behind `to_device` and `commutes_with_device`:
each call queried every device's properties. The device-resident `simplify`
floor of about 200 us barely moves, because it comes from Thrust's sequence of
kernel launches and implicit synchronizations (maximum magnitude, sort,
`reduce_by_key` and `copy_if` counts), not from allocation.

## Simplify at scale

Medians of per-run medians; random operators with duplicate fraction 0.5:

| Path | terms | device-resident main | candidate | ratio | transfer-incl. main | candidate | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| one-word packed key, 24 qubits | 50,000 | 0.527 ms | 0.333 ms | 0.63 | 2.05 ms | 0.70 ms | 0.34 |
| one-word packed key, 24 qubits | 1,000,000 | 1.58 ms | 0.64 ms | 0.41 | 15.2 ms | 5.76 ms | 0.38 |
| one-word two-mask key, 60 qubits | 50,000 | 0.611 ms | 0.309 ms | 0.51 | 2.00 ms | 0.68 ms | 0.34 |
| one-word two-mask key, 60 qubits | 1,000,000 | 1.87 ms | 0.86 ms | 0.46 | 12.0 ms | 5.96 ms | 0.50 |
| two words, 100 qubits | 50,000 | 0.631 ms | 0.345 ms | 0.55 | 2.25 ms | 0.89 ms | 0.39 |
| two words, 100 qubits | 1,000,000 | 2.52 ms | 1.40 ms | 0.56 | 16.8 ms | 9.43 ms | 0.56 |

The CPU simplify control moved between -0.7% and +4.6%.

**Device-resident.** At 50,000 to 1,000,000 terms, device-resident simplify is
1.6-2.4x faster. Allocating the large Thrust temporaries was a major share of
its time.

**Transfer-inclusive.** It is 1.8-2.9x faster, from cheaper allocation plus the
removed host conversion copies and device discovery.

## Correctness and sanitizers

- **Tests:** the CUDA foundation, kernel and generic-simplify suites pass (125
  passed, 4 skipped).
- **Clean runs:** Compute Sanitizer racecheck and synccheck report no errors on
  those suites. Memcheck with `--leak-check full` reports no errors and no
  leaks on the in-process generic-simplify tests.
- **Expected leak reports:** memcheck on `tests/test_phase11_cuda_kernels.py`
  reports leaks from subprocess tests whose device objects stay alive at
  interpreter exit. The same command on `main` with `cudaMalloc` reports them
  too (66 allocations there, 114 here, since the two allocators are tracked
  differently).

## Limitations

- **Hardware:** one H100 and one toolkit.
- **Default pool required:** devices without memory-pool support fall back to
  `cudaMalloc` and were not measured.
- **Retained memory:** the pool keeps freed blocks cached for the process
  lifetime, up to the largest working set seen, and trims at exit. Long-running
  processes that once handled a very large operator keep that memory reserved
  in Wolfgang's pool.
- **Simplify floor:** the remaining small-operator floor comes from simplify's
  multi-kernel Thrust pipeline. A fused small-operator kernel is the next lever,
  and it is not part of this slice.

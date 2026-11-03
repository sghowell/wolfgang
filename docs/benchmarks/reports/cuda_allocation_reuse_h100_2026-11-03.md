# CUDA Allocation Reuse And Call Overhead On H100

Date: 2026-11-03

This report measures slice 7 of the usability and kernel campaign
(`docs/plans/usability_and_kernel_campaign_plan.md`):

- **Pooled allocation.** `DevicePauliSum` buffers, Thrust algorithm temporaries,
  and simplify, expectation and commutation scratch come from a Wolfgang-private
  stream-ordered CUDA memory pool on the legacy default stream. The pool caches
  up to 1 GiB of freed blocks per device; it trims and retries once on
  out-of-memory and trims at exit.
- **Cheap device selection.** `to_device` and `commutes_with_device` pick the
  backend from a device count instead of `cuda_status()`, which queried every
  device's properties on every call.
- **Fewer syncs and fills in simplify.** It uses `thrust::cuda::par_nosync` and
  uninitialized pooled scratch, so Thrust no longer synchronizes after
  algorithms that return nothing to the host, or zero-fills buffers that are
  overwritten next.
- **No host conversion copies.** Coefficients cross the host/device boundary as
  raw bytes.

`DeviceCommutationMatrix` buffers can leave through DLPack and the CUDA Array
Interface, so they still use `cudaMalloc`.

## Evidence

```text
summaries: docs/benchmarks/data/cuda_allocation_reuse_h100_2026-11-03/summary_call_overhead.json
           docs/benchmarks/data/cuda_allocation_reuse_h100_2026-11-03/summary_simplify_paths.json
per-run reports and Nsight reports: private archive (not published)
benchmarks: benchmarks/bench_cuda_call_overhead.py, benchmarks/bench_cuda_simplify_paths.py
```

**Environment.** NVIDIA H100 80GB HBM3 (compute capability 9.0, driver
580.105.08), CUDA toolkit 12.8.93, `sm_90`, x86_64 host with 26 logical CPUs,
GCC 13.3.0, Python 3.12.3, Nsight Systems 2024.6.2.

**Revisions.** Baseline `main` at `23121d6`; candidate `6e48c14`. Every run's
loaded extension matched its checkout's native sources. The baseline checkout
was dirty only because the overhead benchmark script was copied into it.

```text
python benchmarks/bench_cuda_call_overhead.py --label <base|cand> --output ...      # 7 paired runs
python benchmarks/bench_cuda_simplify_paths.py --suite specialized --label <base|cand> --seed <20261130+run> --output ...  # 3 paired runs
nsys profile --trace=cuda python roundtrip.py   # 200 x to_device().simplify().to_host(), 128 terms
```

The baseline/candidate order is randomized per run. The call-overhead benchmark
times 200 back-to-back calls per batch, 7 batches per variant, on small operands
so kernel time is negligible. Timings are steady state, measured after the pool
has warmed. The first call into a fresh process also pays pool creation (about
1 ms, once).

## Per-call overhead

Medians of per-run medians:

| Call | main | candidate | speedup |
| --- | ---: | ---: | ---: |
| `PauliSum.to_device()`, 1 term | 907.5 us | 15.9 us | 57x |
| `commutes_with_device`, 128 x 128 | 907.3 us | 64.5 us | 14x |
| round trip `to_device().simplify().to_host()`, 128 terms | 1,167.8 us | 189.3 us | 6.2x |
| `simplify`, 128 terms, 130 qubits | 199.4 us | 117.3 us | 1.70x |
| `simplify`, 128 terms, 24 qubits | 225.5 us | 143.4 us | 1.57x |
| `matmul`, 128 x 128, 24 qubits | 301.2 us | 203.6 us | 1.48x |
| `simplify`, 4096 terms, 24 qubits | 276.0 us | 190.0 us | 1.45x |
| `simplify`, 1 term | 33.1 us | 26.3 us | 1.26x |
| `expectation_statevector`, 64 terms, host psi | 34.2 us | 30.1 us | 1.14x |
| `to_host()`, 1 term | 23.6 us | 23.7 us | 1.00x |
| CPU `simplify`, 128 terms (control) | 3.7 us | 3.8 us | 0.98x |

The `to_device` and `commutes_with_device` gains come from skipping device
discovery. The `commutes_with_device` output matrix still uses `cudaMalloc`.

## What Nsight Systems shows

CUDA API summary for 200 small round trips. Tracing adds overhead to each API
call, so these shares explain where time goes rather than giving untraced
costs.

| | main | candidate |
| --- | ---: | ---: |
| total traced CUDA API time | 388.8 ms | 162.3 ms |
| `cudaGetDeviceProperties` | 50.2% (220 calls) | 0 calls |
| `cudaMalloc` / `cudaMallocFromPoolAsync` | 31.4% (3,520 calls) | 71.6% (3,520 calls) |
| `cudaStreamSynchronize` calls | 4,400 | 2,201 |
| `cudaLaunchKernel` calls | 3,300 | 1,980 |

**Device discovery** was half of the traced API time on `main`, and the
candidate removes it entirely.

**Synchronization and launches** halve and fall 40%. The drop comes from
`par_nosync` and from no longer filling scratch buffers before overwriting them.

**Allocation** now dominates the remaining traced API time. Wall-clock timings
imply the untraced pool allocation cost is much lower. The 128-term simplify
makes about 11 allocations in 143 us total, far below the roughly 33 us per
call the trace records.

## Simplify at scale

Medians of per-run medians; random operators with duplicate fraction 0.5:

| Path | terms | device-resident main | candidate | speedup | transfer-incl. main | candidate | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| one-word packed key, 24 qubits | 50,000 | 0.537 ms | 0.243 ms | 2.2x | 2.06 ms | 0.60 ms | 3.4x |
| one-word packed key, 24 qubits | 1,000,000 | 1.54 ms | 0.52 ms | 2.9x | 14.7 ms | 5.47 ms | 2.7x |
| one-word two-mask key, 60 qubits | 50,000 | 0.617 ms | 0.224 ms | 2.8x | 2.03 ms | 0.59 ms | 3.4x |
| one-word two-mask key, 60 qubits | 1,000,000 | 1.85 ms | 0.72 ms | 2.6x | 12.1 ms | 5.65 ms | 2.1x |
| two words, 100 qubits | 50,000 | 0.639 ms | 0.255 ms | 2.5x | 2.32 ms | 0.80 ms | 2.9x |
| two words, 100 qubits | 1,000,000 | 2.46 ms | 1.17 ms | 2.1x | 16.7 ms | 8.76 ms | 1.9x |

The CPU simplify control moved between -1.5% and +3.8%. For large operators,
allocating and zero-filling big temporaries was most of simplify's time.

## Correctness and sanitizers

**Full validation.** `WOLFGANG_VALIDATE_CUDA=1 python scripts/validate.py
--profile core` passes at the candidate head. The CUDA foundation, kernel,
generic-simplify and pool suites pass with the pool, and also with the
`cudaMalloc` fallback forced by `WOLFGANG_CUDA_BENCH_DISABLE_POOL=1`.

**Pool tests.** `tests/test_cuda_pool_allocator.py` checks that every public
operation returns pool usage to its baseline, and that the fallback produces
identical results.

**Compute Sanitizer** on the generic-simplify and pool tests:

| Tool | Result |
| --- | --- |
| memcheck (`--leak-check full`) | 0 errors, 0 leaks |
| racecheck | 0 hazards |
| synccheck | 0 errors |
| initcheck | Reports only CUB `ReduceByKeyAgent` look-back reads, the same pattern recorded for `main` in `cuda_simplify_paths_h100_2026-11-03.md`. None are in Wolfgang kernels, so no uninitialized pooled scratch is ever read. |

Memcheck on the subprocess tests in `tests/test_phase11_cuda_kernels.py` reports
leaks of device objects still alive at interpreter exit. The same command on
`main` reports them too.

## Limitations

- **Hardware.** One H100 and one toolkit.
- **Devices without pool support.** These use `cudaMalloc`. That path is
  exercised by the forced-fallback test run but was not timed.
- **Retained memory.** The pool keeps up to 1 GiB of freed blocks per device
  until another Wolfgang allocation fails or the process exits. Other libraries
  in the same process cannot use that memory meanwhile.
- **Remaining floor.** About 120-145 us for device-resident simplify, made of
  the sort and reduction kernels plus the synchronizations that value-returning
  Thrust algorithms still need. A fused small-operator kernel is the next lever.

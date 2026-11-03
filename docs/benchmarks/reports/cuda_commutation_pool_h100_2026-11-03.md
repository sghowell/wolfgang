# Pooled CUDA Commutation Output

Date: 2026-11-03

This report qualifies the first slice 10 change of the portability and
interop campaign (`docs/plans/portability_and_interop_campaign_plan.md`),
found by the slice 9 profile
(`docs/benchmarks/reports/cuda_small_op_profile_h100_2026-11-03.md`).

`DeviceCommutationMatrix` kept its dense output in `cudaMalloc` memory because
it can be exported through DLPack and the CUDA Array Interface, and a consumer
on another stream could still read a pooled block after Wolfgang freed it. Its
count scratch (`count_commuting`, `conflict_degrees`) also used
`cudaMalloc`/`cudaFree` on every call. Now:

- the output and the scratch come from Wolfgang's stream-ordered pool;
- handing the pointer out through DLPack or the CUDA Array Interface marks the
  matrix, and freeing a marked matrix synchronizes the device first. This is
  the rule `DevicePauliSum.device_arrays()` introduced in slice 3;
- Wolfgang's own kernels read the matrix through a private accessor that does
  not mark it, so `conflict_degrees` does not force a synchronization on free.

## Evidence

```text
benchmark: benchmarks/bench_cuda_call_overhead.py
summary:   docs/benchmarks/data/cuda_commutation_pool_h100_2026-11-03/summary_call_overhead.json
per-run reports: private archive (docs/benchmarks/data/**/raw/ is not published)
```

| Item | Value |
| --- | --- |
| Device | NVIDIA H100 80GB HBM3, sm_90, CUDA runtime 12.8, driver 13.0 |
| Host | Intel Xeon Platinum 8480+, Ubuntu 24.04, GCC 13.3, Python 3.12.3 |
| Baseline | main at `002c013` (tree dirty only by the candidate's benchmark script) |
| Candidate | `3ce446b` (native sources identical to the built `5ac7121`), clean |

Command: 7 runs, baseline and candidate in randomized order per run, each
row 200 calls after warmup, 7 repetitions:

```text
python benchmarks/bench_cuda_call_overhead.py --backend cuda --label <base|cand> --output <run>
```

## Results

Medians of per-run medians, min-max of per-run medians, 24-qubit operators:

| Operation | Baseline | Candidate | Speedup |
| --- | ---: | ---: | ---: |
| `commutes_with_device`, 128 x 128 | 64.1 us (63.9-64.4) | 10.6 us (10.5-10.8) | 6.06x |
| `count_commuting()`, 128 x 128 matrix | 22.7 us (22.7-22.7) | 20.4 us (20.4-20.6) | 1.11x |
| `conflict_degrees(axis=1)`, 128 x 128 matrix | 23.8 us (23.8-24.1) | 21.7 us (21.6-21.8) | 1.10x |

Every other row (simplify at five sizes, `matmul`, expectation, transfers,
the round trip and the CPU control) stayed within 0.97-1.01x.

The baseline cost of `commutes_with_device` depends on what else the process
holds. When the output is the only `cudaMalloc` buffer, each free unmaps the
driver's chunk and the next allocation maps it again, about 55 us per call.
A first run of this comparison kept another matrix alive throughout, which
kept the chunk mapped. In that run the baseline took 12.1 us against 10.6 us
for the candidate. The count rows now run after every other row, so they no
longer change the baseline's allocation pattern; earlier campaign reports
measured the same 64 us baseline. The pooled path costs about 10.6 us in both
cases.

## Correctness

| Check | Result |
| --- | --- |
| `tests/test_cuda_pool_allocator.py` | The output and count scratch come from the pool and return to the baseline when released. An exported matrix (CUDA Array Interface and DLPack) outlives a consumer read queued behind a delay kernel on a non-blocking stream, even though a same-size matrix then reuses its block. |
| Negative control | With the export mark removed, both export tests fail. |
| `WOLFGANG_VALIDATE_CUDA=1 python scripts/validate.py --profile core` | Passed on the H100 (949 tests in the CUDA run). |
| `compute-sanitizer` | memcheck: 0 errors on the pool and CUDA kernel tests. initcheck and racecheck: 0 on the commutation tests. |

initcheck on the full pool and kernel test files also reports errors inside
Thrust's `reduce_by_key` under multi-launch simplify (reached through
`matmul`). main at `002c013` reports the same errors, so they predate this
change. They match a known CUB pattern: decoupled look-back reads tile slots
that are never written, for value types wider than 8 bytes such as
`complex<double>`.

## Limitations

- One H100.
- An exported matrix now synchronizes the device when it is freed, which
  waits for unrelated work and can happen during garbage collection; see
  `docs/architecture/cuda_backend.md`. Unexported matrices free without
  synchronizing.
- Cached pool blocks are unavailable to other libraries until a trim, as for
  every pooled allocation.

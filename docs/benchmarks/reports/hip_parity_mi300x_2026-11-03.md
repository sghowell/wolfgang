# Accelerator Simplify Parity And HIP Call Overhead

Date: 2026-11-03

This report qualifies slice 2 of the portability and interop campaign
(`docs/plans/portability_and_interop_campaign_plan.md`):

- CUDA and HIP simplify now follow the CPU numeric contract;
- HIP gains the CPU drop rule, `par_nosync` rocThrust execution, uninitialized
  scratch and a single-term fast path;
- HIP keeps `hipMalloc` as its allocator, after measurement showed a
  stream-ordered pool to be slower on ROCm.

## Correctness

`tests/test_accelerator_simplify_parity.py` runs the same checks on every
available accelerator against CPU simplify (69 cases per backend). It covers:

- equivalence across 5-300 qubits, 1-20,000 terms, duplicate fractions and
  tolerance modes;
- identity-only operators;
- `ValueError` for non-finite input;
- `OverflowError` for duplicate-sum overflow;
- finite coefficients whose magnitude exceeds the largest double.

| Backend | At `7c69edc` | After this slice |
| --- | --- | --- |
| Metal (M4 Pro) | 69 passed | 69 passed |
| CUDA (H100) | 16 failed: non-finite input accepted, overflow not raised | 69 passed |
| HIP (MI300X) | 20 failed: the CUDA failures plus overflowing-magnitude coefficients | 69 passed |

GPU reductions sum long duplicate runs in tree order, so overflowing partial
sums can meet as `inf + -inf = NaN`. Before this slice, the drop test then
discarded that term silently. The drop predicate now flags any non-finite
reduced sum, and every reduction path raises `OverflowError`. Coefficients
match CPU to rounding (`rtol = atol = 1e-11`), because the association order
differs from CPU's input-order sums.

## HIP call overhead

```text
benchmark: benchmarks/bench_cuda_call_overhead.py --backend hip
summaries: docs/benchmarks/data/hip_parity_mi300x_2026-11-03/summary_call_overhead.json
           docs/benchmarks/data/hip_parity_mi300x_2026-11-03/summary_pool_vs_hipmalloc.json
           docs/benchmarks/data/hip_parity_mi300x_2026-11-03/summary_hip_api.json
per-run reports: private archive (docs/benchmarks/data/**/raw/ is not published)
```

| Item | Value |
| --- | --- |
| GPU | AMD Instinct MI300X VF (`gfx942:sramecc+:xnack-`), 192 GB |
| HIP | 7.16.26385 (runtime and driver), AMD clang 24.0.0 |
| Host | 20 vCPU, Linux, Python 3.12 (uv-managed) |
| Build | `WOLFGANG_ENABLE_HIP=ON`, `WOLFGANG_HIP_ARCHITECTURES=gfx942`, Release |
| Boundary | Device-resident: each operation starts from operands already on the device and synchronizes before returning; `to_device`/`to_host`/round trip are transfer-inclusive |

Commands: 7 runs, baseline and candidate in randomized order per run, 200
calls per batch and 7 batches per run.

```text
baseline (main 7c69edc):   python benchmarks/bench_cuda_call_overhead.py --backend hip --label base --output <run>
candidate (d871133):       python benchmarks/bench_cuda_call_overhead.py --backend hip --label s2-cand --output <run>
summary:                   python benchmarks/bench_cuda_call_overhead.py --summarize <dir> --output <summary>
```

The baseline checkout ran the candidate's benchmark script (it predates
`--backend`), and an unrelated untracked test file was present, so its tree was
dirty. Both runs' loaded extensions matched their checkouts' native sources.

Per-call medians of per-run medians, with the per-run range:

| Operation | Baseline | Candidate | Speedup |
| --- | ---: | ---: | ---: |
| simplify, 1 term | 254.5 us (251.4-258.0) | 31.0 us (30.4-31.2) | 8.22x |
| simplify, 128 terms, 24 qubits | 255.2 us (252.4-260.1) | 155.2 us (154.4-156.8) | 1.64x |
| simplify, 4,096 terms, 24 qubits | 279.9 us (277.2-285.2) | 180.4 us (180.3-183.5) | 1.55x |
| simplify, 128 terms, 130 qubits | 252.5 us (250.1-255.4) | 172.1 us (171.3-174.9) | 1.47x |
| matmul, 128 x 128 terms | 299.8 us (295.8-302.4) | 200.1 us (198.2-204.2) | 1.50x |
| round trip (to_device, simplify, to_host), 128 terms | 336.8 us (332.4-339.4) | 236.6 us (233.5-240.0) | 1.42x |
| to_device, 1 term | 46.8 us | 46.5 us | 1.01x |
| to_host, 1 term | 39.4 us | 39.2 us | 1.01x |
| commutes_with_device, 128 x 128 | 13.3 us | 13.4 us | 0.99x |
| expectation, 64 terms, 12 qubits, host statevector | 44.7 us | 45.0 us | 0.99x |
| CPU simplify, 128 terms (control) | 6.6 us | 6.7 us | 0.98x |

### Where the time went

rocprofv3 HIP API statistics over 220 simplify calls of the 128-term operator
(20 warm-up, 200 timed; per-call means exclude each API's single largest call,
the first-use module load):

| HIP API | Baseline calls / time per simplify | Candidate calls / time per simplify |
| --- | --- | --- |
| `hipStreamSynchronize` | 12.0 / 110.5 us | 0 / 0 |
| `hipLaunchKernel` | 16.0 / 46.5 us | 10.0 / 26.6 us |
| `hipMemcpyWithStream` (rocThrust value returns) | 3.0 / 43.4 us | 3.0 / 43.3 us |
| `hipDeviceSynchronize` | 1.0 / 0.2 us | 1.0 / 8.4 us |
| `hipMalloc` + `hipFree` | 26.0 / 44.2 us | 26.0 / 60.2 us |

- **The main gain:** rocThrust's default policy synchronizes the stream after
  every algorithm. `par_nosync` removes those 12 synchronizations.
- **Fewer launches:** uninitialized scratch drops the fill kernels.
- **Frees cost more:** `hipFree` waits for outstanding work, so part of the
  removed waiting reappears in the frees and the closing synchronization.
- **The overflow flag:** it lives in pinned, device-mapped host memory and is
  read after the compaction's own synchronization, so it adds no API calls.

### Pool versus hipMalloc

An intermediate candidate (`7c950c6`) used a stream-ordered pool, a port of
the CUDA one, for every allocation. With everything else equal, plain `hipMalloc`/`hipFree`
was faster across the board (5 runs, randomized order;
`summary_pool_vs_hipmalloc.json`):

| Operation | Pooled | hipMalloc | Pooled / hipMalloc |
| --- | ---: | ---: | ---: |
| simplify, 1 term | 42.3 us | 30.8 us | 1.37 |
| simplify, 128 terms, 24 qubits | 191.5 us | 171.2 us | 1.12 |
| simplify, 4,096 terms, 24 qubits | 226.8 us | 196.0 us | 1.16 |
| matmul, 128 x 128 terms | 254.9 us | 214.6 us | 1.19 |
| expectation, host statevector | 45.9 us | 44.4 us | 1.03 |

ROCm already caches small `hipMalloc` blocks, so the pool only added overhead.
HIP therefore defaults to `hipMalloc`. The pool stays behind the benchmark-only
switch `WOLFGANG_HIP_BENCH_ENABLE_POOL=1`, so the comparison can be rerun on
later ROCm releases. CUDA keeps its pool, which removed most of the
small-operation cost on an H100
(`docs/benchmarks/reports/cuda_allocation_reuse_h100_2026-11-03.md`).

## Cost of the checks on CUDA

The NaN-aware max-reduction and the mapped overflow flag also run on CUDA.
Measured on the H100 (CUDA 12.8, `sm_90`) against a main-equivalent build
(`1041e08`, same CUDA sources as `7c69edc`), 7 paired runs:

- small device-resident simplify and matmul calls: 0.6-1.3% slower;
- single-term simplify, transfers, commutation and expectation: unchanged;
- the CPU control moved 2.3%, which is within run-to-run noise.

Data: `summary_cuda_check_cost.json`. An earlier version that copied the flag
back explicitly cost 3-5%.

## Validation

| Host | Check | Result |
| --- | --- | --- |
| MI300X | `WOLFGANG_VALIDATE_HIP=1 python scripts/validate.py --profile core` at `d871133` | passed (763 tests in the HIP run) |
| MI300X | HIP allocator, parity and ROCm foundation suites | 116 passed |
| H100 | `WOLFGANG_VALIDATE_CUDA=1 python scripts/validate.py --profile core` at `d871133` | passed (850 tests in the CUDA run) |
| H100 | Compute Sanitizer memcheck and racecheck on the parity suite | 0 errors, 0 hazards |
| M4 Pro | `WOLFGANG_VALIDATE_METAL=1 python scripts/validate.py --profile core`, quality profile | passed |

## Limitations

- One MI300X (a virtual function) on HIP 7.16; one H100.
- The pool comparison covers small operations only; very large or repeated
  large allocations were not measured.
- About 60 us of each small HIP simplify is now in `hipMalloc`/`hipFree`, and
  about 43 us in rocThrust's three synchronous value returns. A reused
  per-operation workspace and slice 5's fused small-operator kernel target
  these.

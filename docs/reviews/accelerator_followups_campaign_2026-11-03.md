# Accelerator Follow-Ups Campaign Review Closeout

Plan: `docs/plans/accelerator_followups_campaign_plan.md`.

## Slice 1: Device-side simplify counts (CUDA)

Branch `codex/device-counts`. The benchmarked candidate is `6ddf58b`; later
commits change documentation, tests, one comment and the temp-storage pointer
noted below.

- Scope: `src/detail/device_simplify_pipeline.cuh` (new),
  `src/cuda/simplify_cuda.cu`, `tests/test_device_simplify_pipeline.py` (new),
  `docs/architecture/cuda_backend.md`, `CHANGELOG.md`, the report
  `docs/benchmarks/reports/cuda_device_counts_h100_2026-11-03.md` and its data,
  the plan, the evidence index and `AGENTS.md`.
- Reviewer: independent agent (CUDA correctness, numerics, concurrency,
  benchmark and documentation roles), read-only, from the diff and the
  committed data.
- Findings: P0 0, P1 0, P2 3, P3 8.
- P2 resolutions:
  - the report's range for operators wider than two words now covers only rows
    that reach the pipeline (1.21-1.85x distinct, 1.11-1.37x with duplicates)
    and names the fused rows that were unchanged;
  - CHANGELOG, plan and index state the scope (operators the fused kernel does
    not take) and the synchronizations exactly: one wait before sizing the
    output instead of eight, plus the final synchronization both versions
    have; the 4,096-1M range is 1.08-1.74x;
  - this closeout exists, and the plan records the slice as merged only with
    the merge.
- P3 resolutions:
  - the generic-key comment no longer claims reduce-by-key keeps the
    lowest-index reference; any reference in a run names an equal term;
  - the CUB temp region is at least one byte, so a run call can never receive
    a null pointer and turn into a size query;
  - the thread test is renamed and says that `DevicePauliSum.simplify` holds
    the GIL, so it covers per-thread result buffers, not overlapping calls;
  - new 100,000-term cases at 24, 32, 40, 100 and 130 qubits reach multi-tile
    sort, reduce-by-key and scan stages, including the packed key at 64 bits;
    the `WIDTHS` comment names the key kinds correctly;
  - the report's per-kernel split is backed by `kernels_per_call` in the
    committed breakdown; the unrecorded select-versus-scan timing was dropped.
    The scan stays because CCCL 2.7's `DeviceSelect::If` takes an `int` item
    count (checked in the toolkit headers), while scan and reduce-by-key take
    64-bit counts;
  - "builds under hipcc" is verified: slice 2 compiles the same header with
    hipcc and hipCUB on the MI300X;
  - filler wording removed from the report.
- Deferred P3: dispatching CUB with 32-bit item counts when the input fits,
  which would give reduce-by-key and scan 32-bit tile state, needs a
  measurement first. A NaN-specific overflow test was not added: the
  pipeline's non-finite check treats NaN and infinity the same way, the
  duplicate-overflow tests reach it, and the parity suite's cancelling-overflow
  test (24,002 terms, so it takes this pipeline) accepts either an
  `OverflowError` or finite output, as tree order allows.
- Process exception: the plan asks for failing tests first. This slice changes
  cost, not behavior, so its new tests pin the existing contract and pass on
  the baseline as well; the evidence for the change is the profile and the
  paired timings.
- Validation:
  - H100, candidate `6ddf58b`:
    `WOLFGANG_VALIDATE_CUDA=1 python scripts/validate.py --profile core` passed
    (1,037 tests in the CUDA run), and the CUDA suites passed again with
    `WOLFGANG_CUDA_BENCH_DISABLE_POOL=1`;
  - compute-sanitizer racecheck and synccheck reported 0 on the new tests;
    memcheck and initcheck results are in the report;
  - local quality and docs profiles passed.
  - after the review fixes, the rebuilt branch (`ad14cfd`) passed the
    pipeline, parity, small-simplify, generic-simplify, pool and Phase 11 CUDA
    kernel suites on the H100: 367 passed, the rest skipped as HIP or Metal
    cases.
- Residual risk: one H100 and one CCCL release; CUB's kernel choices and
  tile sizes can change with the toolkit. Operators between the fused limit
  and roughly 64K terms still cost 100-120 us, set by sort latency and launch
  count.

## Slice 2: HIP pipeline and per-thread block cache

Branch `codex/hip-pipeline`. Benchmarked builds: `9af4918` (pipeline only) and
`4ca8655` (pipeline and cache), against main at `ad6f7a8`.

- Scope: `src/hip/simplify_hip.hip.cpp`, `src/hip/pool_allocator.hip.{hpp,cpp}`,
  `src/hip/pool_allocator_hooks.hpp`, the private
  `_hip_thread_cache_status_for_testing` hook,
  `tests/test_hip_thread_cache.py` (new), `--backend hip` in
  `benchmarks/bench_cuda_simplify_paths.py`, `docs/architecture/rocm_backend.md`,
  `CHANGELOG.md`, the report
  `docs/benchmarks/reports/hip_device_counts_mi300x_2026-11-03.md` and its data,
  the plan, the evidence index and `AGENTS.md`.
- Measured in two steps. The pipeline alone made HIP simplify 1.83x faster at
  4,096 terms but 0.73-0.99x at 50,000 terms, from the cost of freeing its
  multi-megabyte scratch; the per-thread block cache was added for that and
  measured as a third build.
- Tests: the four cache tests were written alongside the cache, before its
  first build, not ahead of it. As a negative control, a build with
  `kThreadCacheLimitBytes = 0` fails the reuse and thread-exit tests (the other
  two check that nothing is retained, which holds trivially).
- Validation: MI300X, `4ca8655`:
  `WOLFGANG_VALIDATE_HIP=1 python scripts/validate.py --profile core` passed
  (945 tests in the HIP run); the pipeline, parity, small-simplify, pool and
  cache suites passed (251 passed). The CPU-only smoke benchmark inside
  validation reports `correctness_passed: false` for its HIP case because that
  build has no HIP; earlier MI300X validation logs show the same.

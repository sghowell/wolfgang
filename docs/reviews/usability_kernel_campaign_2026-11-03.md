# Usability And Kernel Campaign Review — 2026-11-03

Plan: [usability and kernel campaign](../plans/usability_and_kernel_campaign_plan.md).
Each slice had an independent agent review against
[the review policy](../quality/code_review.md) before merge.

## Slice 1: CUDA simplify hardening

Branch `codex/cuda-simplify-hardening`, commits `c65553c`..`3198e0e`.

- Reviewer: independent agent (CUDA and performance roles), read-only.
- Findings: P0 0, P1 0, P2 3, P3 9.
- P2 resolutions:
  - Generic rows gained transfer-inclusive timings, and the CPU crossover claim
    now names its boundary.
  - Every CUDA path uses the CPU drop rule for finite coefficients whose
    magnitude overflows, with an oracle test. Previously a 0 * inf threshold
    dropped every term.
  - CPU tests now syntax-check the CUDA and HIP workspace headers, so the
    `ee4f95f` build break is caught without a GPU toolkit.
- P3 resolutions:
  - comment wording, test construction and long-run tile coverage;
  - selector validation order;
  - report accuracy and the changelog range;
  - selector updates moved out of timed regions;
  - doc drift;
  - initcheck and synccheck runs added.
- P3 deferred with reason: squashing the test fix into `c65553c` for
  bisectability. Rewriting history would change the revisions recorded in the
  benchmark evidence. Commits `c65553c`..`2a3053b` fail
  `test_backend_workspace_headers_share_private_contract_helpers`, which
  `4724bba` repairs.
- Validation after fixes (H100, CUDA 12.8, `sm_90`):
  - `WOLFGANG_VALIDATE_CUDA=1 python scripts/validate.py --profile core` passed;
  - CUDA foundation, kernel and generic-simplify tests: 130 passed;
  - Compute Sanitizer memcheck, racecheck and synccheck clean;
  - initcheck reports only CUB `ReduceByKeyAgent` look-back reads, identical
    (1,356) on the pre-change baseline;
  - local quality and docs profiles passed.
- Residual risk:
  - H100 is the only GPU with evidence;
  - HIP lacks the single-term fast path and the overflow drop rule;
  - CPU CI still compiles no `.cu` sources.

Evidence: [CUDA simplify paths report](../benchmarks/reports/cuda_simplify_paths_h100_2026-11-03.md).

## Slices 2 and 3: packed-array I/O and operator algebra

Branch `codex/host-array-io-and-algebra`, commits `7c64436`..`de87752`. The
branch is merged with a merge commit so the benchmark revisions stay reachable.

- Reviewer: independent agent (semantic/API and C++ safety roles), read-only.
- Findings: P0 0, P1 0, P2 6, P3 8.
- P2 resolutions:
  - Commutator coefficients are computed as `2 * (a * b)`, so doubling cannot
    overflow on its own.
  - Non-finite coefficients raise up front, regardless of which pairs cancel.
  - The bulk `from_labels` path is limited to NumPy arrays, so duck-typed
    arrays keep working.
  - The changelog records the `==`, hashing and `bool()` behavior change with a
    migration note.
  - The speed wording is backed by the host API report.
  - Zero-copy CUDA `DevicePauliSum` views moved to slice 7 in the plan.
- P3 resolutions:
  - `from_words` validates its own copy;
  - the binding policy documents the zero-copy view exception;
  - typed `self`, `op[None]` and guardrail messages fixed;
  - commutator output is sized exactly with a counting pass;
  - the NumPy object-array quirk and the `equiv` `rtol` scale are documented;
  - bool coefficients are accepted, and the byte-order message is clearer;
  - native contract tests cover `from_words`, `commutator`, `select_terms` and
    equality.
- Validation after fixes (Apple M4 Pro): `python scripts/validate.py --profile core`
  passed (598 Python tests, native CTest); the new test file passes 37 tests.
  The refreshed timings showed no commutator regression.
- Residual risk:
  - random operands only; Hamiltonians where most terms commute change the
    fused-vs-composed ratio;
  - x86_64 commutator rows predate the counting pass.

Evidence: [host API report](../benchmarks/reports/host_api_2026-11-03.md).

## Slices 4 and 5: statevector tools and measurement workflow

Branch `codex/statevector-and-measurement`, commits `1a12a7d`..`0519f97`.

- Reviewer: independent agent (semantic/API, C++ safety and testing roles),
  read-only. It checked against Qiskit oracles:
  - `to_matrix`, `apply_statevector` and `variance_statevector` against
    `SparsePauliOp.to_matrix`;
  - Clifford synthesis on 300 random commuting groups of 1-8 qubits and 20
    groups of 64-79 qubits, with 0 mismatches;
  - Y-basis counts at 400k shots.
- Findings: P0 0, P1 0, P2 2, P3 10.
- P2 resolutions:
  - Sparse export budgets its `2^n + 1` row pointers, so an empty operator can
    no longer allocate an unbounded index array.
  - The CI sanitizer job now runs the new test files with pinned SciPy, and
    native contracts cover `apply_statevector`, the dense and CSR matrices and
    `estimate_counts`.
- P3 resolutions:
  - the standard error is computed as `sqrt(pop / (N - 1))`, so huge counts no
    longer form inf / inf;
  - empty batches are validated;
  - error messages are generic;
  - the basis-rotation convention is documented and tested for X and Y and for
    the `qwc_basis` estimate path;
  - wide-group, identity, duplicate, empty-group and SciPy import-isolation
    tests added;
  - README and dependency-policy entries for the `scipy` extra;
  - the SciPy ImportError keeps its cause;
  - changelog wording and the documented 2-D behavior change;
  - variance uses the normalized state;
  - term grouping happens once per batch, and results are written into NumPy
    buffers directly.
- Validation after fixes (Apple M4 Pro): `python scripts/validate.py --profile core`
  passed (670 Python tests, native CTest); the quality profile passed with CI's
  pinned tools.
- Residual risk:
  - statevector tools are scalar CPU kernels of O(terms * 2^n), with no SIMD
    or oneTBB variant;
  - `scipy>=1.10` is a declared lower bound, while CI tests a newer pinned
    version.

## Slice 6: oneTBB simplify

Branch `codex/cpu-parallel-simplify`, commits `7450154`..`e7cecd6` plus the
final evidence. The benchmarked candidate is `e7cecd6`.

- Reviewer: independent agent (C++ concurrency, numerics and performance
  roles), read-only. oneTBB was not installed in the review environment, so the
  chunk logic was checked by simulation against the scalar run loop (120,000
  random sorted inputs, grains 1-64, no mismatches).
- Findings: P0 0, P1 1, P2 4, P3 5.
- P1 resolution (signed zeros): the oneTBB kernel started each run from
  `0.0 + 0.0i`, so `-0.0` components became `+0.0` where the scalar hash path
  kept them. Every CPU simplify path, scalar sort loops included, now seeds each
  run with its first coefficient. Tests compare raw coefficient bits, including
  on `-op`, in both the hash and sort regimes.
- P2 resolutions:
  - Apple Silicon timings: deferral recorded in the plan.
  - Auto now also requires at least four oneTBB threads.
  - Peak memory: the fixed-key kernel no longer keeps a separate index array or
    zero-fills its buffers, and chunks keep (position, coefficient) pairs. The
    remaining excess over scalar is listed as a limitation.
  - Documentation drift in `docs/user/performance.md` and the hardware doc
    fixed.
- P3 resolutions:
  - below the threshold, auto checks only the selector variable instead of
    building a backend report;
  - the comparator is a functor;
  - tests cover a run longer than a chunk, all-identical input, an overflow
    raised from the last chunk on a worker thread, and the reported thresholds;
    the phase 9 test that compared simplify with itself now compares against
    scalar bits;
  - the Xeon host was measured at 2,000-10,000 terms, the report states the
    sub-millisecond noise, and summaries record compiler flags, thread settings
    and the CPU vendor string;
  - the dispatch helpers stay in `include/wolfgang/cpu_backend.hpp`, next to
    the existing auto-dispatch policy they extend.
- Validation after fixes: on the EPYC instance with oneTBB,
  `python scripts/validate.py --profile core` passed (723 tests) and the native
  contract passed under AddressSanitizer and UndefinedBehaviorSanitizer; the
  oneTBB suites passed on the Xeon host.
- Deferred: Apple Silicon oneTBB timings; bundling oneTBB in release wheels.
- Residual risk:
  - x86_64 only;
  - peak memory above scalar was not measured;
  - the threshold rests on two hosts with 16 and 26 vCPUs.

## Slice 7: CUDA allocation reuse and call overhead

Branch `codex/cuda-allocation-reuse`, commits `781efb9`..`d998314`. The
benchmarked candidate is `6e48c14`.

- Reviewer: independent agent (CUDA, C++ safety and performance roles),
  read-only.
- Findings: P0 0, P1 1, P2 3, P3 several.
- P1 resolution (memory retention and out-of-memory recovery):
  - The pool now caches at most 1 GiB per device.
  - Every allocation path, pooled or `cudaMalloc`, trims the pools and retries
    once on out-of-memory, then raises `MemoryError`.
  - The changelog and `cuda_backend.md` document retention and its effect on
    other libraries.
- P2 resolutions:
  - Nsight Systems CUDA API breakdown recorded before and after the change.
  - Deferral of `_inplace` methods recorded in the plan.
  - Scratch value-initialization fills and redundant Thrust synchronizations
    removed (`par_nosync`, uninitialized pooled scratch), which the Nsight data
    attributes.
  - A test that the pool returns to baseline, and a forced-fallback switch with
    a test.
  - `git diff --check` clean.
- P3 resolutions:
  - the mutex lives for the whole process;
  - only self-raised CUDA errors are cleared;
  - buffers are freed on their known device;
  - byte copies replace typed casts;
  - stale comments and unused declarations removed;
  - out-of-memory raises `MemoryError`;
  - report and changelog wording fixed.
- Validation after fixes (H100):
  - `WOLFGANG_VALIDATE_CUDA=1 python scripts/validate.py --profile core` passed
    (720 tests in the CUDA run);
  - the suites pass with `WOLFGANG_CUDA_BENCH_DISABLE_POOL=1`;
  - memcheck, racecheck and synccheck clean;
  - initcheck gives 1,356 CUB look-back reports, identical to the baseline.
- Deferred: zero-copy CUDA `DevicePauliSum` views (needs HIP and Metal changes
  that cannot be validated here); public streams and graphs.
- Residual risk:
  - one GPU model;
  - multi-GPU and `cudaDeviceReset` behavior unexercised;
  - pool retention up to 1 GiB per device.

## Slice 8: Metal hybrid simplify

Branch `codex/metal-hybrid-simplify`, commits `a2b7917`..`cc7a9c7` plus the
final evidence. The benchmarked candidate is `cc7a9c7`.

- Reviewer: independent agent (Metal, Objective-C++ memory management,
  numerics and benchmark roles), read-only. It reviewed `e7cecd6..4f33861`,
  confirmed the report tables against the summary JSON, and ran a 72-case
  stress script that forced the hybrid at sizes 1-131,073.
- Findings: P0 0, P1 1, P2 3, P3 9.
- P1 resolution (public simplify read the build tree): the hybrid loaded
  `kernels/simplify.metal` from a path baked in at compile time, so sdist and
  `pip install git+...` builds and moved checkouts failed from 12,288 terms.
  - CMake now embeds the kernel source.
  - It compiles once per process, with the tile size passed from the host.
  - A pipeline failure keeps the host bridge.
  - A contract test pins the embedding, and a manual run with the source file
    moved away matched CPU simplify.
- P2 resolutions:
  - Signed zeros: the `-op` checks contained no `-0.0`. Tests now feed
    explicit `-0.0` components in single-term and duplicate runs, and the
    negation checks (here and in the slice 6 oneTBB test) zero imaginary parts
    first.
  - oneTBB Metal builds: recorded as a report limitation and a plan deferral.
  - Public docs: the `DevicePauliSum.simplify` and package docstrings, roadmap,
    provenance, support matrix, AGENTS.md and CHANGELOG now describe the hybrid.
- P3 resolutions:
  - a private run counter lets tests prove the public path takes the hybrid at
    the threshold and the bridge below it, for two-word operators and under the
    bridge selector;
  - one library build serves all pipelines; the first call (about 4 ms) is
    measured and reported;
  - the gather pipeline is fetched before encoding starts;
  - the pipeline cache is locked;
  - a failed default-device probe is retried;
  - operators past 2^31 terms keep the bridge;
  - scratch allocation failure raises `MemoryError`;
  - report and comment figures corrected (previous-design sizes, control
    drift, the duplicate-fraction claim, the 8,192-term comment);
  - unrecorded upload timings dropped from the report;
  - the summary records spread, output sizes, tolerances and build config;
  - the tile size is defined once, on the host.
- Validation after fixes:
  - Apple M4 Pro: `WOLFGANG_VALIDATE_METAL=1 python scripts/validate.py
    --profile core` passed (657 tests in the default run, 692 in the Metal
    run), and the quality and docs profiles passed.
  - H100: `WOLFGANG_VALIDATE_CUDA=1 python scripts/validate.py --profile core`
    passed (773 tests in the CUDA run), and the oneTBB simplify suite passed
    (52 tests).
- Deferred: hybrid timings on oneTBB Metal builds; a device-resident float64
  reduction.
- Residual risk:
  - one Apple GPU;
  - inputs above 1,048,577 terms unmeasured (single command buffer);
  - the first call in a process pays kernel compilation.

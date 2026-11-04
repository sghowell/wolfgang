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
- Tests: the cache and its tests were written together and first ran on the
  same build, so the tests did not fail first. As a negative control, a build
  with `kThreadCacheLimitBytes = 0` fails the reuse and thread-exit tests,
  which shows the cache is used; it does not separately test the thread-exit
  release (the other two tests check that nothing is retained, which holds
  trivially with the cache off).
- Validation: MI300X, `4ca8655`:
  `WOLFGANG_VALIDATE_HIP=1 python scripts/validate.py --profile core` passed
  (945 tests in the HIP run); the pipeline, parity, small-simplify, pool and
  cache suites passed (251 passed). The CPU-only smoke benchmark inside
  validation reports `correctness_passed: false` for its HIP case because that
  build has no HIP; earlier MI300X validation logs show the same. A follow-up
  could make that smoke report `not_applicable` instead.
- Reviewer: independent agent (HIP runtime, allocator, concurrency, benchmark
  and documentation roles), read-only.
- Findings: P0 0, P1 0, P2 2, P3 6.
- P2 resolutions:
  - an out-of-memory retry in any Wolfgang HIP allocation now releases the
    failing thread's idle cached blocks first, as the docs claimed; the docs
    now say that other threads keep their blocks and that a process can hold
    up to 512 MiB per thread that has run HIP simplify;
  - the allocation probe is committed (`benchmarks/probe_simplify_allocations.py`)
    and was rerun on three builds in history (`ad6f7a8`, `9af4918`, `0be167f`).
    The report now rests the cache's justification on the pipeline-only
    versus cache comparison (8.4 ms less `hipFree` and 3.6 ms less `hipMalloc`
    over 90 calls), presents the ROCm caching explanation as an inference,
    drops the p90 over all frees as evidence, and records the unexplained
    1M-term transfer-inclusive gain.
- P3 resolutions: header and doc comments now say the two blocks are shared
  by every call site, that an other-device block is freed only when evicted,
  that freeing relies on `hipFree` waiting for queued work, and that blocks
  are not revalidated after `hipDeviceReset`; the thread-exit comment gives the
  right reason; the `rocthrust_default` selector is documented as the current
  pipeline, the benchmark's strategy label says so, and the Campaign 3 and 4
  paragraphs in `rocm_backend.md` read as history plus current state; the
  pool-bypass test has a positive control and skips without pool support; the
  reuse test runs at 24, 60 and 100 qubits so scratch and a rocThrust
  temporary are held together; simplify-path summaries record the backend and
  refuse to mix CUDA and HIP reports; wording fixes in the closeout and
  CHANGELOG.
- Validation after fixes (MI300X, `0be167f`): the cache, pool, pipeline,
  parity, small-simplify, ROCm foundation and Campaign 3 asset suites passed
  (292 passed).

## Slice 3: Pinned transfers

Branch `codex/pinned-transfers`, merged with main (rather than rebased) so the
measured commits stay in history. Final benchmarked candidate: `9e667c3` on
both devices, against main at `4ca82d5` (H100) and `ad6f7a8` (MI300X, HIP
sources as in `4ca82d5`).

- Scope: `src/detail/staged_transfer.hpp` (new), `to_host` in
  `src/cuda/device_pauli_sum.cu` and `src/hip/device_pauli_sum.hip.cpp`, the
  native contract test and its include path in `CMakeLists.txt`,
  `tests/test_device_transfers.py` (new), `benchmarks/bench_host_transfers.py`,
  `benchmarks/probe_cold_to_host.py` and `benchmarks/native/host_copy_microbench.cu`
  (new), both backend contracts, `CHANGELOG.md`, the report
  `docs/benchmarks/reports/host_transfers_2026-11-03.md` and its data, the
  plan, the evidence index and `AGENTS.md`.
- Outcome against the plan: pinned staging was measurably faster only for
  device-to-host copies, so `to_device` and statevector uploads are unchanged.
  On the MI300X staging also lost from about 2 MB, so HIP stages only up to
  64 KiB.
- Finding beyond the plan: large `to_host` copies were dominated by page
  faults on fresh host memory (about 2 us per 4 KiB page on these KVM guests);
  advising huge pages on the reserved result storage cut them.
- Reviewer: independent agent (C++ safety, CUDA/HIP, performance and
  documentation roles), read-only.
- Findings: P0 0, P1 0, P2 3, P3 9.
- P2 resolutions:
  - the native test now runs with two fake streams, one landing copies at the
    next wait and one landing them when issued; a single-buffer mutant fails
    both, where the deferred-only test had passed it;
  - cold first calls were measured on fresh threads. They exposed a real
    regression on HIP (124 against 43 us for a thread's first small call,
    from an 80 us pinned allocation per thread), so per-thread buffers were
    replaced by a process-wide pool; the report states the cold numbers,
    including the one remaining case (a new thread's first 1M-term call on
    the H100);
  - the earlier HIP cutoff runs are committed
    (`hip_mi300x_earlier_cutoffs.json`, builds `1277015` and `1f1d9c3`).
- P3 resolutions: CUDA copies name `cudaStreamLegacy`; the microbenchmark
  marks its huge-page columns n/a below 4 MiB, the "halved" sentence is
  corrected, and a direct-copy-with-huge-pages column was added. That column
  favored copying directly above 16 MiB on CUDA; a build that did so
  (`58d63f7`) lost to staging end to end (9.9 against 5.4 ms at 1M terms), so
  CUDA stages at every size and the report explains the discrepancy. The
  CHANGELOG no longer claims HIP's direct path skips the zero-fill; the thread
  test is described as covering per-thread state, not overlapping calls; the
  MI300X 65,536-term row is described as a flat 3.0 ms wait, and the MI300X
  range no longer counts the within-noise 1M-term, 100-qubit row; the staging
  half is rounded to 16 bytes; a failed buffer allocation leaves the existing
  pool buffers usable; docs give HIP's 128 KiB buffer size and the
  device-reset and heap huge-page caveats; a failing copy waits for queued
  copies before rethrowing.
- Tests first: the native staged-copy test was written before the CUDA and
  HIP integration and run on the CPU first; the device round-trip tests guard
  behavior that did not change.
- Validation:
  - H100, `1277015`: `WOLFGANG_VALIDATE_CUDA=1 python scripts/validate.py
    --profile core` passed (1,051 tests in the CUDA run, native contracts
    included); compute-sanitizer memcheck with leak check, racecheck and
    synccheck reported 0 on the transfer tests;
  - after the review fixes, `9e667c3` passed the transfer, foundation,
    device-array and pipeline suites on the H100 (124 passed) and the
    transfer, pipeline, cache and pool suites on the MI300X (110 passed); the
    native staged-copy and pool tests pass under the host compiler.
- Residual risk: the direct-copy fallback when pinned memory cannot be had
  has not run on hardware; page-fault behavior and the huge-page benefit
  depend on the host.

## Slice 4: CPU thread cap

Branch `codex/cpu-threads`. Measured build: `c5ee325`; review fixes in
`a71b182`.

- Scope: `include/wolfgang/cpu_backend.hpp`, `src/cpu_backend.cpp`,
  `bindings/python/build_info.cpp`, the package exports and stubs,
  `tests/test_cpu_threads.py` (new), `benchmarks/bench_cpu_threads.py` and
  `benchmarks/probe_simplify_threads.py` (new), the report
  `docs/benchmarks/reports/cpu_threads_2026-11-03.md` and its data,
  `docs/guide/python-api.md`, `docs/user/performance.md`, `docs/api/index.md`,
  `docs/architecture/api_stability.md`, `README.md`, `CHANGELOG.md` and the
  plan.
- Public API: `set_num_threads`, `get_num_threads`, `WOLFGANG_NUM_THREADS`
  and the C++ pair, recorded in `docs/architecture/api_stability.md`.
- Tests first: `tests/test_cpu_threads.py` was committed before the
  implementation and failed on the missing functions.
- Reviewer: independent agent (semantic/API, C++ safety, performance, testing
  and documentation roles), read-only. To check oneTBB internals it built
  oneTBB 2022.3 from a source copy in the session scratchpad on macOS; the
  P1 below was then reproduced independently on the H100 host's system
  oneTBB 2021.11 (a cap of 2^31+1 gave no result within 40 s; 2^32+1 ran
  serially while reporting 26 threads).
- Findings: P0 0, P1 1, P2 3, P3 6.
- P1 resolution: a cap at or above oneTBB's default concurrency installs no
  `global_control` (it had raised oneTBB's own worker limit, and oneTBB
  narrows the value to 32 bits); a replacement control is built before the
  old one is released.
- P2 resolutions: CPU compute entry points (including commutation and
  grouping under a forced `WOLFGANG_CPU_BACKEND=tbb`) validate
  `WOLFGANG_NUM_THREADS`; the documented `WOLFGANG_CPU_BACKEND=tbb`
  workaround for small caps is withdrawn because operations without a oneTBB
  kernel raise under it; the user guidance now reports the measured
  trade-offs (no gain for four processes sharing cores, a slower result below
  the floor, a slower median under a quota) instead of advising caps for all
  pools.
- P3 resolutions: the state mutex is held across `fork()`; NumPy integers are
  accepted; the build report always carries `cpu_thread_limit_error`; stubs
  are sorted; the report uses 1.25x, drops an unrecorded 20,000-term check and
  names the cap-1,000,000 row; new tests cover huge caps, invalid values on a
  forced-oneTBB commutation, a forked child, the worker-thread bound on
  Linux, NumPy integers and identical result bytes at 1-64 threads under auto
  and forced oneTBB; `docs/architecture/api_stability.md` and the README
  describe the cap.
- Owner decision recorded as a follow-up: whether to move the four-thread
  oneTBB floor (the plan's follow-up list).
- Validation: H100 host (26 vCPUs, oneTBB 2021.11), `a71b182`: thread, CPU
  parallel simplify, fork-safety and typing suites, 87 passed; macOS without
  oneTBB: the same suites pass with the oneTBB cases skipped.

## Slice 5: oneTBB in the SBOM

Branch `codex/sbom-bundled`.

- Scope: `scripts/sbom_bundled_components.py` (new), the SBOM step of
  `.github/workflows/release-wheelhouse.yml`, the required-term list in
  `scripts/check_release_wheelhouse.py`, `tests/test_sbom_bundled_components.py`
  (new), `third_party/dlpack/README.md` (new),
  `docs/quality/security_and_supply_chain.md`, `CHANGELOG.md` and the plan.
- The SBOM generated by `cyclonedx-py requirements` listed only numpy. The
  release workflow now adds oneTBB (tag, commit and repository read and
  shape-checked from `scripts/build_onetbb.sh`; license
  `Apache-2.0 AND BSD-3-Clause`; license references to the notices at the
  pinned commit) and DLPack (version from the vendored `dlpack.h`, its
  SHA-256, Apache-2.0), adds both to the dependency graph, and validates the
  document against its CycloneDX schema before writing it.
- Tests: the component, merge and workflow-order tests were written with the
  script and run before the workflow change was pushed; they did not fail
  first.
- Reviewer: independent agent (release, supply-chain and documentation
  roles), read-only.
- Findings: P0 0, P1 0, P2 1, P3 6.
- P2 resolution: oneTBB's license is `Apache-2.0 AND BSD-3-Clause`. At the
  pinned commit, x86_64 builds define `__TBB_USE_ITT_NOTIFY` and compile the
  BSD-3-Clause ITT Notify static API into the runtime the manylinux wheel
  bundles; the SBOM said Apache-2.0 only.
- P3 resolutions: the purl is lowercase per the purl spec and derived from
  `ONETBB_REPOSITORY`; tag, commit and repository must match pinned shapes, so
  an override form such as `"${ONETBB_TAG:-...}"` fails instead of becoming
  the version (tested with a temporary tree); schema validation is tested
  when cyclonedx-python-lib is installed, and a missing `specVersion` gives a
  clear error; DLPack carries the header's SHA-256, its vendored tag is
  recorded in `third_party/dlpack/README.md`, and its property names the
  Python extension; both components join the dependency graph, and a rerun
  replaces components by name, whatever their version; the supply-chain doc
  and this closeout were reworded.
- Pre-existing gaps noted by the review, not addressed here: the SBOM has no
  `metadata.component`, numpy has no version, the SBOM is not attested, and
  one document covers the sdist and both wheels.
- Validation: the SBOM tests pass (7 without cyclonedx installed, 8 with
  cyclonedx-bom 7.1.0); the release supply-chain and wheelhouse tests and
  `scripts/check_release_wheelhouse.py` pass. Release workflow dry runs with
  `publish-target=none`: run 41698848188 produced the first merged SBOM, and
  run 41699199577 passed every job with the `--validate` step; the review
  fixes were validated against that run's SBOM with cyclonedx-bom 7.1.0, and
  run 41702026420, on the reviewed branch, passed every job.

## Slice 6: Multi-GPU

Branch `codex/multi-gpu`.

- Scope: `src/cuda/pool_allocator.{cu,cuh}`, `src/cuda/device_guard.cuh` (new),
  `src/cuda/device_pauli_sum.{cu,cuh}`, `src/cuda/device_commutation_matrix.cu`,
  `src/cuda/commutation_cuda.cu`, `src/cuda/workspace.cu`,
  `src/cuda/simplify_cuda.cu`, `src/detail/staged_transfer.hpp`,
  `bindings/python/pauli_sum_py.cpp`,
  `include/wolfgang/device_commutation_matrix.hpp`,
  `tests/test_cuda_multi_gpu.py` (new), `benchmarks/bench_cuda_multi_gpu.py`
  (new), the report and data, `docs/architecture/cuda_backend.md`,
  `docs/architecture/api_stability.md`, `docs/guide/python-api.md`,
  `docs/quality/python_binding_policy.md`, `CHANGELOG.md`, the plan, the roadmap
  and the evidence index.
- The first multi-GPU run (two NVLink-connected H100s) found three problems:
  peer kernels faulted reading exports (pool memory was not mapped into peers),
  two threads on two GPUs ran no faster than one (device calls held the GIL),
  and a process that used only GPU 1 held a 516 MiB context on GPU 0 (each
  device switch went back with `cudaSetDevice`). The slice maps the pool into
  peers on first export, releases the GIL around CUDA device calls with a
  per-device lock for compute calls, and routes every device switch through
  one guard that goes back only to a device with a primary context. Measuring
  the GIL change exposed a same-device regression (0.63-0.65 times one thread
  on small calls), which the per-device lock removed.
- Tests: the peer-read, GIL and second-device-only tests were written first and
  failed on main; the peer-free test, added for the review's P1, failed before
  its fix.
- Reviewer: independent agent (CUDA, concurrency, testing, documentation and
  benchmark roles), read-only.
- Findings: P0 0, P1 2, P2 4, P3 10.
- P1 resolutions: freeing an exported buffer synchronized only its own device,
  so a peer kernel could read a block the pool had reused (reproduced by the new
  test); it now also synchronizes every mapped peer with a live primary
  context. Validation and the sanitizer were rerun at the final head.
- P2 resolutions: the CHANGELOG and backend doc quoted 1.83-1.94 instead of the
  final 1.88-1.98, and the report's spread claim was too narrow; the no-lock
  data behind the 0.63-0.65 claim is committed (`threads_no_lock_summary.json`);
  `docs/quality/python_binding_policy.md` no longer says device methods keep the
  GIL in CUDA builds; the second-device-only test counts the GPUs that hold the
  process's memory by UUID and skips without `nvidia-smi` or in a separate PID
  namespace; the HIP exit-handler gap is recorded.
- P3 resolutions: peers are mapped one at a time and failures retried; a
  destructor uses the non-throwing guard; frees clear CUDA errors only on
  failure; both exit handlers skip in a forked child; the docs describe the
  PyTorch guard and HIP accurately and the guide covers peer reads; the plan's
  same-device numbers are exact; the pool-growth limitation is recorded.
  Not changed: restoring non-primary contexts with `cuCtxGetCurrent` (behavior
  as before), and daemon threads inside CUDA calls at interpreter exit
  (recorded as a limitation of releasing the GIL).
- Validation after the review fixes, at `9b0ec23` on the two-GPU host:
  `WOLFGANG_VALIDATE_CUDA=1 python scripts/validate.py --profile core` passed
  (1,099 tests in the CUDA run, 747 in the CPU-only run); the 14 multi-GPU tests
  passed; memcheck with leak checking reported 0 errors and 0 leaks over the
  in-process multi-GPU and transfer tests.

## Slice 7: Documentation and figures

Branch `codex/docs-refresh`.

- Scope: `scripts/render_readme_figures.py` (new), its three SVGs under
  `docs/benchmarks/plots/`, `tests/test_readme_figures.py` (new),
  `docs/javascripts/mermaid-init.js` and the two Mermaid tests; `README.md`,
  `docs/index.md`, `docs/user/performance.md` and `docs/benchmarks/index.md`;
  a prose edit of the getting-started, guide, user, API, accelerator,
  architecture, quality and release-process docs, the benchmark protocol,
  `docs/roadmap.md`, `docs/research/provenance.md`, `CONTRIBUTING.md`,
  `SECURITY.md`, `SUPPORT.md` and `GOVERNANCE.md`; "FastPauli" wording left in
  two binding docstrings, a HIP error message and a CUDA comment;
  `CHANGELOG.md` and the plan. Plans, benchmark reports, review closeouts and
  release ledgers were not edited for prose.
- Figures: hand-written SVG with light and dark styles, drawn from committed
  JSON. Simplify time per host, each accelerator (device-resident and with
  copies) beside the CPU of its own host; Wolfgang against Qiskit,
  OpenFermion and cuStateVec; per-call GPU times across the November
  campaigns. `tests/test_readme_figures.py` runs the renderer with `--check`.
- Diagrams: `pymdownx.superfences` emits `<div class="mermaid">`, but the
  initializer selected only `pre.mermaid`, so every diagram showed as source
  text on the site. It now selects both, and the integration test asserts
  it. The architecture diagram was redrawn and the agent-driven loop laid out
  top to bottom, so their text stays legible at the site's content width;
  both were checked in headless Chrome against a local build.
- Prose: three editors worked on disjoint file groups against
  `docs/quality/documentation_standards.md`, which gained a prose-patterns
  section, while checking every phrase a test asserts. Stale statements found
  on the way and fixed: a `simplify(backend=...)` selector that does not
  exist; Metal described as bring-up only; HIP expectation and `matmul`
  listed as absent after ROCm Campaign 6 shipped them; three native types
  instead of four; device-side counts and pinned transfers listed as future
  CUDA work; `SECURITY.md` described as not yet written; the kernel
  performance campaign called the current ranking handoff; ROCm Campaign 6
  named as the latest ROCm report.
- Tests: the figure freshness test was written with the renderer; the
  Mermaid selector assertion was added with the fix, and the layout tests now
  accept top-to-bottom flowcharts.
- Reviewer: independent agent (editorial, figure and factual accuracy),
  read-only.
- Findings: P0 0, P1 3, P2 3, P3 9.
- P1 resolutions: the README and guide credited device-side counts with the
  drop from the pre-pool baseline and now quote 188 to 108 us (H100) and 182
  to 98 us (MI300X); they credited pooled allocation with the MI300X's
  128-term gain, which came from the HIP parity changes (HIP has no pool); and
  the simplify figure set Metal and HIP against the H100 host's CPU under a
  "below CPU scalar means faster" subtitle. It now has one panel per host
  with that host's CPU rows (the M4 Pro's scalar and release-wheel oneTBB, the
  MI300X host's CPU control), and the prose says that oneTBB beats the Metal
  hybrid on the M4 Pro at 1M terms.
- P2 resolutions: the per-call figure describes a baseline and three changes,
  with only steps 2-4 as paired candidates; footnotes that clipped with the
  macOS system font are wrapped and the speedup bars rescaled; the simplify
  figure gained transfer-inclusive rows; the protocol's README-plot rules now
  describe the three figures (the broad per-host view first, external
  baselines second, a before/after figure only after them) and keep the
  accelerator landscape in the evidence index; provenance no longer calls
  the landscape the current README plot.
- P3 resolutions: small-operator timings are quoted at 128 terms; SIMD is no
  longer listed as covering simplify; 225 us is used throughout; the step-1
  footnote names both baselines; the H100 campaign-numbering note covers
  Campaigns 10 and 11; the README's API list includes `DeviceArrayView` and
  `measurement`; a dead branch in the renderer is gone; "Recent changes" is
  ordered by campaign; the Metal hybrid threshold names one-word operators.
- Not changed: two pairs of ledger lines in `docs/release/support_matrix.md`
  read as contradictory but are pinned verbatim by release tests, and
  `docs/release/history_sanitization.md` keeps FastPauli paths in its
  historical commands.
- Validation after the review fixes, on macOS arm64: `scripts/validate.py`
  with `--profile quality`, `--profile docs` (strict build) and
  `--profile core` (695 passed, 964 skipped for absent accelerators and
  optional libraries; native contracts, benchmark smokes and the sdist smoke)
  passed; `scripts/render_readme_figures.py --check` and `git diff --check`
  passed.

## Follow-up: support matrix package-index lines

Branch `codex/support-matrix-ledger`.

- Scope: the "Current Package Version" block of
  `docs/release/support_matrix.md`, `docs/release/README.md`,
  `tests/test_release_final_010.py` and `tests/test_release_v021_preparation.py`.
- The block held two "TestPyPI validation:" and two "PyPI publication:" lines
  that contradicted each other, each pinned verbatim by a different release
  test. One line each now states the whole status: 0.1.0 published its final
  dry run to TestPyPI (run 25462760923), its PyPI run (25462997972) failed
  with `invalid-publisher`, and v0.2.2 and v0.2.3 made no package-index run.
  The release index calls `0.1.0.md` the 0.1.0 final-release ledger rather
  than the current one.
- Tests: the updated assertions and a new test that each status key appears
  exactly once in the block, carrying both the 0.1.0 runs and v0.2.3, failed
  before the matrix was changed.
- Reviewer: independent agent (release documentation), read-only.
- Findings: P0 0, P1 0, P2 0, P3 4. Resolved: the PyPI line names the
  `invalid-publisher` failure and the remaining requirement; the uniqueness
  test scans only the block, strips lines and checks both fragments sit on
  one line; the release index wording. Left as written: the historical
  `0.2.1.md` and `0.2.2.md` status lines.
- Validation: `scripts/validate.py --profile core` (696 passed, 964 skipped),
  `scripts/check_release_readiness.py` and `scripts/check_release_wheelhouse.py`
  passed before the review fixes; after them, the release, identity and
  checkpoint tests (54 passed) and `scripts/check_release_readiness.py`.

## Slice 8: oneTBB thread floors

Branch `codex/thread-floor`.

- Scope: `include/wolfgang/cpu_backend.hpp`, `src/cpu_backend.cpp`,
  `src/commute.cpp`, `src/parallel/commute_kernels_tbb.cpp`,
  `src/simd/commute_kernels_{avx512,avx2,neon,scalar}.cpp`,
  `src/detail/commute_kernels.hpp`, `bindings/python/build_info.cpp`,
  `tests/test_cpu_tbb_commutation.py` (new), `tests/test_cpu_thread_floor.py`
  (new), `tests/test_cpu_threads.py`, `tests/test_cpu_parallel_simplify.py`,
  `benchmarks/bench_cpu_thread_floor.py` (new),
  `benchmarks/bench_cpu_commutation_kernels.py` (new),
  `benchmarks/bench_cpu_dispatch.py`, `benchmarks/bench_cpu_thresholds.py`, the
  report and data, `README.md`, `docs/user/performance.md`,
  `docs/guide/python-api.md`, `docs/architecture/hardware_targets_and_testing.md`,
  `docs/architecture/api_stability.md`, `CHANGELOG.md`, the plan, the roadmap and
  the evidence index.
- Measuring the floor showed that the oneTBB commutation kernel ran a scalar
  loop per thread (0.23-0.26 times one AVX-512 thread at four threads, which
  auto dispatch chose). The SIMD kernels became row-range kernels that oneTBB
  runs per chunk, which also sped up the single-threaded kernels (1.02-1.73x).
  Simplify runs on oneTBB from two threads; pairwise commutation from two, or
  three over AVX2 or AVX-512; one-word commutation over AVX2 or AVX-512 from
  2048 x 2048 entries.
- Tests: the floor and dispatch tests were written before each change and
  failed against it (the four-thread floor, the one-word threshold, the NEON
  floor's reported key); the equivalence tests guard the refactor.
- Reviewer: independent agent (CPU kernels, dispatch, testing, benchmark and
  documentation roles), read-only.
- Findings: P0 0, P1 1, P2 3, P3 7.
- P1 resolution: `README.md` and `docs/user/performance.md` still stated a
  four-thread floor; both now give the new floors.
- P2 resolutions: NEON's commutation floor is two (the M4 Pro wins at two
  threads, and Arm cores have no SMT), with the build-info key renamed
  `tbb_pairwise_min_concurrency_over_x86_simd`; the single-threaded kernel
  comparison is now produced by `benchmarks/bench_cpu_commutation_kernels.py`
  with its metadata and committed as a summary; validation was rerun at the
  head.
- P3 resolutions: the report states a 3% tie tolerance and that one-word
  commutation loses or ties below 2048 x 2048; the simplify table shows true
  medians; the before/after ranges are split by width; the header comment's
  range and wording are corrected; the plan's kernel range is exact; the builds
  table lists the EPYC first-kernel build; the dispatch and threshold
  benchmarks label auto's pairwise kernel with the new rules; the floor
  benchmark records its seed and describes the floors; the report notes that
  `cpu_auto_tbb_eligible` stays simplify's eligibility and that
  `tbb_auto_has_min_concurrency()` gained a defaulted parameter (source
  compatible, a different symbol, per `docs/architecture/api_stability.md`);
  the two-thread floor over scalar for narrow operators is a recorded
  limitation; the equivalence test compares every available SIMD selector with
  scalar.
- Validation after the review fixes, at `55d2088` on the Xeon host:
  `python scripts/validate.py --profile core` passed (777 passed), and the
  dispatch, equivalence, threads and threshold-benchmark suites passed pinned
  to two hyperthreads, two cores and four vCPUs, and unpinned.

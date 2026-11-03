# Portability And Interop Campaign Review Closeout

Plan: `docs/plans/portability_and_interop_campaign_plan.md`.

## Slice 1: GPU build health

Branch `codex/gpu-build-health`, draft PR #16 (used only to run the new
workflow before the local fast-forward merge).

- Scope: `.github/workflows/gpu-build.yml`, the campaign plan, and CI
  documentation in `docs/architecture/testing_and_ci.md` and
  `docs/architecture/hardware_targets_and_testing.md`.
- Reviewer: independent agent (CI, supply chain and build roles), read-only.
- Findings: P0 0, P1 0, P2 3, P3 5.
- P2 resolutions:
  - Undefined symbols: a shared object links with undefined references, so a
    device method missing from one backend would pass CI and fail only at
    import. Each lane now fails on undefined `wolfgang::` symbols (and when the
    module is missing), and the HIP lane imports the extension without a GPU.
  - CI documentation: the compile-only lanes, architectures and refresh policy
    are documented; AGENTS.md and the roadmap point at the new campaign.
  - Metal: a `macos-latest` lane builds and imports the Metal extension.
- P3 resolutions:
  - the CUDA lane also compiles oneTBB sources and `sm_80`;
  - Python build tools are pinned and the build runs without isolation;
  - the unused `hipcub-dev` package is dropped, and installed ROCm library
    versions are printed;
  - container digests are refreshed with toolkit bumps (documented);
  - plan slices 3, 4 and 7 name the Metal export rule, the CUDA-Q optional
    extra and adapter job, and the wheel licensing and SBOM work.
- Validation: GPU build workflow green on PR #16 (CUDA 12.8 `sm_80`/`sm_90`
  with oneTBB, ROCm 7.2 `gfx942` with import, Metal with import); all other
  PR checks green; `tests/test_release_supply_chain.py` and
  `tests/test_ci_quality_gates.py` pass. MI300X baseline at `7c69edc` (HIP
  7.16): `WOLFGANG_VALIDATE_HIP=1 python scripts/validate.py --profile core`
  passed (684 tests in the HIP run).
- Residual risk: the lanes run after the push to `main` under the repository's
  local-merge workflow, so they detect a break rather than block it; they make
  no runtime claims.

## Slice 2: Accelerator simplify parity and HIP call overhead

Branch `codex/accelerator-simplify-parity`, commits `926889b`..`d871133` plus
documentation. The benchmarked candidate is `d871133`.

- Reviewer: independent agent (CUDA/HIP numerics, memory management and
  benchmark roles), read-only. It audited every CUDA and HIP reduction path for
  a silently dropped non-finite sum and found none.
- Findings: P0 0, P1 0, P2 1, P3 9.
- P2 resolution (documentation): CHANGELOG entries for the user-visible
  behavior changes; the numeric contract in `cuda_backend.md` and
  `rocm_backend.md`; the HIP allocation policy; the plan row updated after the
  pool was measured slower.
- P3 resolutions:
  - the contract and the test docstring state that overflow near the largest
    double depends on each backend's summation order;
  - a cancelling-overflow test (one run of `[M, M, -M, -M]` repeats spanning
    reduction tiles) must raise `OverflowError` or return only finite values;
  - tests add a 33-64 qubit width, contract checks for every benchmark-only
    reduction path (CUDA CUB radix and serial kernel, HIP serial kernel), and
    keep transfers outside `pytest.raises`;
  - HIP validates its benchmark selectors before the single-term early return
    and gates the generic selector on width, as CUDA does;
  - trailing whitespace removed;
  - the overflow flag moved to pinned, device-mapped host memory, which removed
    its per-call copy (CUDA cost from 3-5% to 0.6-1.3% on small calls);
  - a local layout `static_assert` guards HIP's raw-byte coefficient copies;
  - the pool header's export rule is unconditional again;
  - the overhead summarizer refuses mixed CUDA/HIP directories.
- Measurement-driven change during the slice: the HIP pool (a port of CUDA's)
  measured 9-37% slower than `hipMalloc` on small calls, so HIP defaults to
  `hipMalloc` and the pool is a benchmark opt-in.
- Validation after fixes:
  - MI300X: `WOLFGANG_VALIDATE_HIP=1 python scripts/validate.py --profile core`
    passed (763 tests in the HIP run); allocator, parity and ROCm suites 116
    passed.
  - H100: `WOLFGANG_VALIDATE_CUDA=1 python scripts/validate.py --profile core`
    passed (850 tests in the CUDA run); memcheck and racecheck clean on the
    parity suite.
  - M4 Pro: Metal core profile and quality profile passed.
- Deferred: a reused per-operation scratch workspace for HIP (about 60 us of a
  small simplify is now in `hipMalloc`/`hipFree`), to the performance slice.
- Residual risk:
  - one MI300X (virtual function) and one H100;
  - the `inf + -inf` path is verified by inspection and by the
    raise-or-finite test, since no input forces NaN under every grouping.

## Slice 5: Fused small-operator simplify

Branch `codex/small-simplify`, commits `2ed099c`..`b072c0a` plus
documentation. The benchmarked candidate is `b072c0a`.

- Reviewer: independent agent (CUDA/HIP kernel, numerics and performance
  roles), read-only. It simulated the sort network at block sizes 2-1,024 and
  widths 1-4 against a reference sort, and checked barrier uniformity and the
  shared-memory layout.
- Findings: P0 0, P1 1, P2 2, P3 4.
- P1 resolution (per-device shared-memory opt-in): the opt-in was cached once
  per process, but CUDA applies it per device, so 56 KiB launches would fail
  on a second GPU, and a failed call stayed cached. The attribute is now set
  before each launch that needs more than 48 KiB. Any launch-configuration
  error falls back to the multi-launch path.
- P2 resolutions:
  - FMA contraction: the device threshold `atol + rtol * max` compiled to an
    FMA, and so did CPU simplify on arm64. On the M4 Pro, CPU dropped a term
    x86 kept. Every backend now rounds the product separately, with
    regression tests.
  - `__launch_bounds__(1024)` added.
- P3 resolutions:
  - the bit-identity claim now covers coefficients, with survivor sets equal
    except at ulp-level threshold ties from device `hypot`;
  - tests add an all-Y key equal to the sentinel key with fewer terms than the
    block, the capacity+1 boundary, and cases with both tolerances;
  - performance: word-major keys avoid bank conflicts, run heads are flagged
    once, the result returns through mapped host memory, and the block is
    sized to the input (with single terms fused on CUDA); together these took
    H100 small simplify from 39 to 18 us;
  - code-standards exception recorded: a custom kernel instead of CUB/hipCUB,
    justified by the measured multi-launch baseline in the report.
- Validation after fixes:
  - H100: `WOLFGANG_VALIDATE_CUDA=1 python scripts/validate.py --profile core`
    passed (927 tests in the CUDA run); memcheck, racecheck, synccheck and
    initcheck on the fused tests report 0 errors.
  - MI300X: `WOLFGANG_VALIDATE_HIP=1 python scripts/validate.py --profile core`
    passed (840 tests in the HIP run).
  - M4 Pro: the separately rounded threshold passes on CPU and Metal.
- Deferred: a multi-GPU check of the per-launch opt-in (one GPU per host
  here); a duplicate-heavy worst-case benchmark row.
- Residual risk:
  - one H100 and one MI300X;
  - device `hypot` can differ from host libm by a few ulps at the threshold.

## Slice 6: Metal follow-ups

Branch `codex/metal-followups`, commits `a26a756`..`c014c0f`, then a merge of
main (`ec7d9b0`), plus documentation. The benchmarked candidate is `ec7d9b0`.

- Reviewer: independent agent (Metal, numerics, build and packaging roles),
  read-only.
- Findings: P0 0, P1 1, P2 3, P3 6.
- P1 resolution (sdist audit): the embed script lived in a new top-level
  `cmake/` directory, which the source-distribution audit rejects. It moved to
  `src/metal/embed_binary.cmake`.
- P2 resolutions:
  - peak memory: the parallel path held the sort index and full-size outputs
    alongside the scratch buffers. When at least half the terms survive,
    chunks now compact in place and become the output. The sort index is
    released right after the GPU wait.
  - documentation and evidence were missing; they are now in
    `docs/architecture/apple_accelerator.md` and the report.
  - the precompiled-kernel test could pass even if the runtime source loaded.
    A private hook now reports the loaded library, and the test asserts it in
    fresh processes for both kinds.
- P3 resolutions:
  - with split command buffers, the first failed buffer is reported only after
    all committed buffers finish;
  - every global step gets its own buffer above 2^26 positions, so one buffer
    never covers a whole 2^28-position sort;
  - the CMake probe trial-compiles the kernels instead of only finding
    `xcrun metal`, and passes the deployment target;
  - chunk count is four per hardware thread for load balance;
  - tests assert the parallel and split paths through hybrid run counters.
- Validation after fixes (M4 Pro, at the documentation commit):
  `WOLFGANG_VALIDATE_METAL=1 python scripts/validate.py --profile core`,
  `--profile quality` and `--profile docs` passed; an independent stress
  script was bit-identical to CPU at chunk-boundary sizes.
- Deferred: Apple oneTBB timings (oneTBB is not installed on the M4 Pro);
  inputs near the 2^31-term limit.
- Residual risk:
  - one Apple GPU and one macOS version;
  - the GPU timeout depends on the display and driver; large inputs were run
    only on this Mac.

## Slice 3: Zero-copy device arrays

Branch `codex/device-arrays`, commits `0b4d296`..`01997f7` plus
documentation. The benchmarked candidate is `01997f7`.

- Reviewer: independent agent (CUDA interop, lifetime, security and
  documentation roles), read-only. It checked consumer behavior against CuPy
  and PyTorch sources.
- Findings: P0 1, P1 2, P2 6, P3 7.
- P0 resolution (stream safety): import ignored the producer's stream and
  copied on the legacy stream, which does not order with non-blocking streams
  (every PyTorch side stream). It also returned before the device-to-device
  copy finished. `from_device_arrays` now synchronizes the device before
  copying and the legacy stream before returning. Two tests use a delay
  kernel to keep work pending; both fail with the synchronizations removed.
- P1 resolutions:
  - an unchecked word count let `num_qubits = 2**64 - 1` wrap to zero words
    and read out of bounds on the device. Word counts now use the host path's
    checked arithmetic, and the binding rejects the value first, with a test;
  - the import timings excluded the copy (nothing synchronized at 64 qubits),
    and pending work skewed other rows. Every timed variant now ends with a
    device synchronization, a 100-qubit row exercises the tail check, and the
    report and CHANGELOG carry the re-measured numbers (import 84 us, not
    57 us, at 1M terms).
- P2 resolutions:
  - PyTorch rejects read-only CUDA Array Interface input; docs now say
    PyTorch reads the views through `torch.from_dlpack`, and a PyTorch 2.15
    test covers both;
  - read-only is advisory on every CUDA consumer. Decision recorded in the
    semantic contracts, CUDA backend doc, docstrings and header: the views are
    flagged read-only, consumers do not enforce it, and writing through them
    is undefined behavior. HIP stays disabled because no ROCm consumer was
    validated, not on a read-only argument;
  - `cuMemGetAddressRange` now runs in the owning device's context;
  - the pool-allocator comment describes the sync-on-free exception;
  - `api_stability.md` lists the experimental surface;
  - tests added for every listed gap that one GPU allows.
- P3 resolutions: the `copy=True` error names `DeviceArrayView`; builds
  without CUDA raise `RuntimeError` for `from_device_arrays` before parsing;
  device ordinals below -1 are rejected, and empty input uses the current
  device; masked inputs are rejected; producer attribute errors propagate
  (PyTorch's requires-grad error); `__dlpack__` accepts `dl_device`, and the
  minimum consumer versions are documented; the sync-on-free consequences
  (garbage collection, host callbacks, graph capture) are documented; strides
  are ignored for empty arrays.
- Also fixed: `thrust::any_of` returned padding bytes that initcheck flagged,
  so the tail check uses `thrust::count_if`.
- Validation after fixes:
  - H100 at `9e0194e`: `WOLFGANG_VALIDATE_CUDA=1 python scripts/validate.py
    --profile core` passed (944 tests in the CUDA run); memcheck, racecheck
    and synccheck report 0 errors. At `01997f7`: the device-array tests pass
    (16, with PyTorch 2.15), and initcheck and memcheck report 0 errors.
  - MI300X: the HIP build raises for both entry points, as documented.
- Deferred: multi-GPU tests (device inference, mixed-device rejection) need a
  host with two GPUs; HIP export with a validated ROCm consumer; DLPack input.
- Residual risk:
  - one H100; consumers may write through views despite the flag;
  - the device synchronization on free and on import waits for unrelated work.

## Slice 4: CUDA-Q interop

Branch `codex/cudaq-interop`, commits `c7ce299`..`09c17f5` plus
documentation.

- Reviewer: independent agent (adapter contract, numerics, packaging and
  documentation roles), read-only. It read the CUDA-Q 0.11.0 and current
  sources for the serialization, term merging and operator bindings, read the
  `cudaq` metapackage sdists, and ran the adapter against a pure-Python
  emulator of CUDA-Q's serialize and deserialize code.
- Findings: P0 0, P1 0, P2 5, P3 9.
- P2 resolutions:
  - the fast path rebuilt coefficients as `re + 1j * im`, which turns
    `(1, inf)` into `(nan, inf)` and loses signed zeros. It now assigns the
    real and imaginary parts separately, with tests;
  - the serialized test variant could silently run the fallback if the layout
    probe failed; it now asserts the probe accepts the installed CUDA-Q;
  - explicit identities and parameterized coefficients gained tests.
    Parameterized coefficients report no parameters in CUDA-Q and fail inside
    `serialize()`, so coefficient-evaluation failures now raise the documented
    `ValueError`;
  - the CI job asserts that the `cudaq` metapackage, the requested version and
    the compiled backend agree;
  - `to_cudaq` unpacked every word to one byte per qubit (about 8.5x the
    packed size); it now unpacks only words that hold a Pauli.
- P3 resolutions: finite-only exactness of `to_cudaq` documented; the
  term-API path walks `degrees`; a serialized operator that does not parse
  falls back to the term API; `num_qubits` accepts integer-like values and
  rejects `bool`; stub parameter names match the runtime for all three
  adapters; contract and guide drift fixed (construction order, constructor
  list, a Hermitian example, width wording); the unbenchmarked "slower" claim
  removed; installation notes state Linux-only wheels and Python 3.11 for
  CUDA-Q 0.16.
- Validation after fixes: on the H100, `tests/test_cudaq_adapter.py` passed
  (33 tests) with CUDA-Q 0.16.0 and with 0.11.0 (`cuda-quantum-cu12`
  0.11.0); the quality profile passed locally; CI runs the adapter on
  `cudaq==0.11.0` and the latest release.
- Deferred: a conversion benchmark; `DevicePauliSum` conversion without a host
  copy.
- Residual risk: CUDA-Q's serialized layout is not a documented promise; the
  probe and fallback cover a change, at a speed cost.

## Slice 8: Closeouts

Branch `codex/closeouts`, documentation and evidence only.

- Review: none required. The slice changes a benchmark report's limitations,
  two plans and an evidence file, none of which are review triggers in
  `docs/quality/code_review.md`.
- `_inplace` methods: closed as won't-do. Pooling removed per-call allocation,
  and the fused small simplify (slice 5) took H100 small simplify from about
  140 to 18 us, so in-place variants would add API surface without a cost to
  remove.
- oneTBB peak memory: measured on a 16-vCPU AMD EPYC 9654 VM (Ubuntu 22.04,
  oneTBB 2021.5) at `1041e08`, the parallel simplify candidate whose kernel
  main still carries. Fresh process per run, peak RSS reset before the call.
  At 1M terms the oneTBB kernel adds 1.34-1.41x scalar's peak with no or half
  duplicates and 1.10-1.17x with 90% duplicates.
- Validation: `python scripts/validate.py --profile docs` and `git diff
  --check`.
- Residual risk: one x86 host and two widths; ARM peak memory comes with the
  bundled-oneTBB wheels (slice 7).

## Slice 9: Streams and graphs

Branch `codex/streams-graphs-profile`, documentation and evidence only.

- Review: none required. The slice adds a profiling report, its data and plan
  status; it changes no contract, policy or code.
- Evidence: Nsight Systems on the H100 at `01997f7` (CUDA sources identical to
  main), 500 profiled calls per operation, per-call medians.
- Decision: no CUDA graphs and no public stream argument. Single-launch
  operations (fused simplify, commutation, expectation) have nothing to
  capture. Multi-launch simplify and `matmul` synchronize eight times per call
  on counts returned to the host, which also rules out capture. A stream
  argument would need asynchronous result objects for no measured gain.
- Follow-ups moved to slice 10: pool the exportable commutation output
  (55 us of a 64 us call is `cudaMalloc`/`cudaFree`); keep simplify's counts
  on the device; stage transfers through pinned memory.
- Validation: `python scripts/validate.py --profile docs` and `git diff
  --check`.
- Residual risk: one H100; HIP was not re-profiled.

## Slice 7: oneTBB in wheels

Branch `codex/onetbb-wheels`, commits `6d7f9de`..`9cb671c` plus
documentation. Wheel evidence: release-wheelhouse runs 41605223043 (first
candidate, timed), 41609630774 (after review fixes) and 41605250834 (main
baseline), all with `publish-target=none`.

- Reviewer: independent agent (supply chain, packaging, licensing, runtime
  behavior and documentation roles), read-only. It downloaded and inspected the
  CI wheels (rpaths, install names, minimum macOS, SONAMEs, RECORD), cloned
  oneTBB at the pinned commit, and ran the macOS wheels.
- Findings: P0 0, P1 1, P2 3, P3 8.
- P1 resolution (fork): with the CI macOS wheel, a child forked after a oneTBB
  simplify died inside oneTBB (libdispatch traps in forked children), and a
  `multiprocessing` fork pool hung; oneTBB does not support `fork()`. oneTBB
  kernels now record their use, and a `pthread_atfork` child handler marks the
  backend `unavailable_after_fork`, so `auto` runs the bit-identical scalar or
  SIMD kernels there and a forced `tbb` selector raises. A regression test
  runs on oneTBB builds, and the wheel smoke test forks after a oneTBB call.
  On Linux an older build's forked children ran oneTBB in five of five trials,
  so the guard is defensive there.
- P2 resolutions:
  - license metadata now reads `MIT AND Apache-2.0 AND BSD-3-Clause`, with
    oneTBB's notices (committed under `third_party/onetbb/` and compared with
    the pinned clone during the build) and DLPack's license as PEP 639
    license files;
  - the smoke test requires the notices and the pinned oneTBB version, and
    the wheelhouse checker requires the license files;
  - auto pairwise commutation needs four oneTBB threads, as simplify did.
- P3 resolutions: a stronger smoke probe (one-, two- and three-word operators,
  duplicate runs of one to five terms, mixed magnitudes, timeouts, stderr on
  failure); an explicit macOS deployment target; `LD_LIBRARY_PATH` dropped;
  the build prefix cleaned first; corrected wheel sizes and baseline note;
  documentation wording. Deferred: a CycloneDX component for the bundled
  runtime (documented instead).
- Validation after fixes: release-wheelhouse run 41609630774 passed every
  job, all six wheels passing the strengthened `--require-tbb` smoke test;
  on the EPYC VM (oneTBB 2021.5 source build) the fork test and the CPU
  suites passed; local core, quality and docs profiles passed.
- Residual risk: two hosts timed; oneTBB thread pools from other wheels can
  oversubscribe; the license expression also covers the sdist, which does not
  contain oneTBB.

## Slice 10: Further performance

Branch `codex/cuda-perf`, commits `aa9b583`..`3ce446b`, review fixes and
documentation. The benchmarked candidate is `3ce446b` (native sources of
`5ac7121`).

- Change: `DeviceCommutationMatrix` output and its count scratch come from the
  CUDA pool. Export marks the matrix, and freeing a marked matrix synchronizes
  the device first.
- Reviewer: independent agent (CUDA lifetime, pool semantics, benchmark
  methodology and documentation roles), read-only.
- Findings: P0 0, P1 0, P2 2, P3 9.
- P2 resolutions:
  - pool memory is not mapped for peer devices, so exported buffers (matrix
    and device arrays) are now documented as owner-device-only; granting peer
    access is a possible follow-up;
  - the CHANGELOG qualifies the 6x figure: it holds when the output is the
    process's only `cudaMalloc` buffer, and the gain is about 1.14x when
    another one keeps the driver's allocation mapped.
- P3 resolutions: the export test asserts the freed block was reused (skipping
  otherwise), creates its output on the consumer stream, drops a needless
  `gc.collect()` and requires only CUDA and CuPy; the pool test name matches
  what it checks; the report's accessor and "now" wording and the chunk
  explanation are corrected and hedged; header and Python docstrings document
  the sync-on-free and owner-device-only rules; the CUDA backend doc cites the
  report; a stale allocator comment is fixed; `empty()`'s unsynchronized
  allocation is documented. Kept as is: the destructor's error clearing,
  which mirrors `DevicePauliSum`.
- Outside the diff, the reviewer noted that `device_arrays()` right after
  `to_device()` could hand a non-blocking consumer a pageable copy still in
  flight. The export now finishes the legacy stream first.
- Validation after fixes: on the H100, the pool, CUDA kernel and device-array
  tests passed (61), with both export tests exercising block reuse; earlier
  at `5ac7121`, `WOLFGANG_VALIDATE_CUDA=1 python scripts/validate.py --profile
  core` passed (949 tests in the CUDA run), memcheck reported 0 errors,
  initcheck and racecheck 0 on the commutation tests, and the negative
  control failed both export tests without the mark.
- Not done in this slice: keeping simplify's intermediate counts on the
  device (the 4,096-term and `matmul` paths), pinned transfers, and a HIP
  per-operation workspace. They are recorded as follow-ups.
- Residual risk: one H100; initcheck errors in Thrust `reduce_by_key`
  predate this campaign and are attributed to a CUB pattern without proof.

## Slice 11: Documentation and site

Branch `codex/docs-refresh`, documentation, one chart and its renderer.

- Reviewer: independent agent (documentation accuracy and standards),
  read-only. It recomputed every chart and table figure from the cited
  reports, checked the README example against the bindings and stubs, resolved
  every link, and searched user-facing docs for claims the campaign made
  stale.
- Findings: P0 0, P1 1, P2 3, P3 9.
- P1 resolution: four places implied published wheels bundle oneTBB; no
  release does yet. The README, home page and accelerator overview now say
  release wheels built after `v0.2.3` bundle it, that source builds use
  oneTBB when CMake finds it, and give the four-thread condition.
- P2 resolutions: the Metal parallel-reduction row (a two-run spot check) left
  the chart, whose subtitle now cites each row's report medians; the evidence
  index lists all 52 reports, generated from their titles, and no longer
  overstates the provenance ledger; the performance guide's plot section
  documents the new chart and its renderer and labels the Campaign 10 plot as
  prior.
- P3 resolutions: qubit counts, hosts and device-resident boundaries on chart
  and table rows; both regimes for the pooled commutation figure; a runnable,
  dependency-labeled README interop example that states the read-only flag is
  advisory and links the CUDA backend contract; the CUDA-Q extra marked
  Linux-only; an accurate device-array row in the accelerator overview;
  current AGENTS.md pointers; the CUDA coverage list and future-work pointer
  in the performance guide; roadmap wording; a README license note for the
  vendored and bundled components.
- Site: the Documentation workflow already deploys the strict build to
  https://sghowell.github.io/wolfgang/ on every documentation change to main.
- Validation: `python scripts/validate.py --profile docs` (strict build),
  `--profile quality` and `--profile core`, and `git diff --check`.
- Residual risk: report descriptions in the index are titles, not summaries;
  the earlier accelerator landscape plot is no longer refreshed.

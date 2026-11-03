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

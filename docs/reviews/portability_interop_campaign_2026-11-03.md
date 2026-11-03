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

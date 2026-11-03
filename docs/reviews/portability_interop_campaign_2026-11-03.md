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

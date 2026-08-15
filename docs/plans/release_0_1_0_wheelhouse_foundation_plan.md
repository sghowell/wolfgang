# Release 0.1.0 Wheelhouse Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the final 0.1.0 CPU wheelhouse and package-index readiness path without expanding accelerator, Windows, or broad hardware support claims.

**Architecture:** Keep the release artifact lane CPU-only and target-specific. `pyproject.toml` owns static cibuildwheel configuration, `.github/workflows/release-wheelhouse.yml` owns manual artifact production and optional trusted-publishing gates, and small repo-local scripts own wheel smoke and checksum evidence.

**Tech Stack:** Python packaging, scikit-build-core, CMake, cibuildwheel, twine, GitHub Actions, PyPI trusted publishing.

---

## Scope

This plan prepares the final 0.1.0 CPU wheelhouse. It does not publish the final
release by itself and does not change runtime APIs, kernels, accelerator build
policy, or current package version.

In scope:

```text
manylinux x86_64 CPU wheels
macOS arm64 CPU wheels
Python 3.10, 3.11, and 3.12 wheel selectors
CPU-only cibuildwheel configuration
clean installed-wheel smoke for each produced wheel
source distribution metadata check
wheel metadata check with twine
sorted SHA256 checksum manifest
exact CPU wheelhouse completeness validation
manual GitHub Actions wheelhouse workflow
TestPyPI dry run gate
PyPI trusted publishing gate
release docs and support matrix routing
```

Out of scope:

```text
CUDA wheels remain unavailable
ROCm/HIP wheels remain unavailable
Metal wheels remain unavailable
combined accelerator wheels remain unavailable
Windows wheels remain unavailable
macOS x86_64 wheels remain unavailable
package-index publication is unavailable until the explicit publish gate succeeds
generic Apple GPU support remains unavailable
broad AMD GPU support remains unavailable
```

## Decisions

1. The final 0.1.0 wheelhouse starts with CPython 3.10, 3.11, and 3.12 because
   those versions match the current package metadata and existing CI support.
2. Linux starts at manylinux x86_64. Additional Linux architectures require
   separate CI evidence, artifact sizing, and clean install smokes.
3. macOS starts at arm64. macOS x86_64 remains unavailable until an explicit
   runner, wheel tag, and import-smoke lane is added.
4. Windows remains unavailable for 0.1.0 because there is no Windows build,
   import, or artifact evidence.
5. `publish-target=none` is the default workflow behavior. TestPyPI and PyPI
   publication require explicit manual selection and trusted-publishing
   environments. Package-index publication is additionally constrained to
   `v*` tag refs.
6. Accelerator wheels remain separate future packaging campaigns. The CPU
   wheelhouse must force CUDA, HIP, Metal, and native CPU tuning off through
   scikit-build-core config settings.
7. The checksum manifest remains part of the uploaded release artifact bundle,
   but it is not part of the package-index upload set. Publish jobs must prepare
   a separate `publish-dist/` containing only `.whl` and `.tar.gz` artifacts.

## Acceptance Criteria

The slice is complete only when:

```text
docs/plans/release_0_1_0_wheelhouse_foundation_plan.md is registered as a source-of-truth path
README, AGENTS, roadmap, release index, release standards, and support matrix route to this plan
pyproject.toml has cibuildwheel CPU-only selectors for CPython 3.10, 3.11, and 3.12
pyproject.toml forces FASTPAULI_ENABLE_CUDA=OFF, FASTPAULI_ENABLE_HIP=OFF, FASTPAULI_ENABLE_METAL=OFF, and FASTPAULI_ENABLE_NATIVE=OFF for cibuildwheel
pyproject.toml runs scripts/wheel_smoke.py as the cibuildwheel installed-wheel test command
.github/workflows/release-wheelhouse.yml builds source distribution and CPU wheels
.github/workflows/release-wheelhouse.yml runs twine check before artifact collection
.github/workflows/release-wheelhouse.yml writes and uploads a checksum manifest
.github/workflows/release-wheelhouse.yml validates the complete CPU wheelhouse shape before upload
.github/workflows/release-wheelhouse.yml can publish to TestPyPI or PyPI only through explicit manual input, a v* tag ref, and OIDC trusted publishing
.github/workflows/release-wheelhouse.yml publishes only the package artifact subset, not the checksum manifest
scripts/check_release_wheelhouse.py passes locally and through scripts/validate.py
scripts/write_release_checksums.py emits a sorted SHA256 manifest for one FastPauli version and can require one sdist plus six CPU wheels
scripts/prepare_publish_dist.py creates a package-index upload directory containing only .whl and .tar.gz artifacts
scripts/wheel_smoke.py verifies CPU-only, non-native, scalar-fallback-safe installed-wheel metadata
tests/test_release_wheelhouse_foundation.py passes
python scripts/validate.py passes
review requirements in docs/quality/code_review.md are satisfied before merge
```

## Execution Tasks

### Task 1: Add Wheelhouse Guard Tests

**Files:**
- Create: `tests/test_release_wheelhouse_foundation.py`

- [x] Add tests for plan registration, support-boundary language, cibuildwheel
  configuration, workflow gates, release-wheelhouse checker, wheel smoke, and
  checksum manifest generation.
- [x] Run `python -m pytest tests/test_release_wheelhouse_foundation.py -q`.
  Expected before implementation: failures for missing plan, workflow, checker,
  smoke script, checksum script, and cibuildwheel configuration.

### Task 2: Add CPU Wheelhouse Tooling

**Files:**
- Modify: `pyproject.toml`
- Create: `scripts/wheel_smoke.py`
- Create: `scripts/write_release_checksums.py`
- Create: `scripts/prepare_publish_dist.py`
- Create: `scripts/check_release_wheelhouse.py`

- [x] Configure cibuildwheel for CPython 3.10, 3.11, and 3.12.
- [x] Restrict the first wheelhouse to manylinux x86_64 and macOS arm64.
- [x] Force CUDA, HIP, Metal, and native CPU tuning off for cibuildwheel builds.
- [x] Add an installed-wheel smoke script that verifies CPU-only metadata and a
  simple `PauliSum.simplify()` behavior.
- [x] Add a checksum writer that records sorted SHA256 lines for the collected
  source distribution and wheels and can require the exact one-source-plus-six-
  wheel CPU wheelhouse shape.
- [x] Add a publish-directory preparer that excludes the checksum manifest from
  the package-index upload set.
- [x] Add a release-wheelhouse checker so drift fails locally and in CI.

### Task 3: Add Manual Wheelhouse Workflow

**Files:**
- Create: `.github/workflows/release-wheelhouse.yml`

- [x] Build a source distribution on Ubuntu.
- [x] Build CPU wheels through cibuildwheel.
- [x] Run `twine check` on artifacts before collection.
- [x] Generate the checksum manifest from collected artifacts.
- [x] Upload the complete release dist as a GitHub Actions artifact.
- [x] Keep publication disabled by default.
- [x] Gate TestPyPI and PyPI publication behind explicit manual input and
  trusted-publishing environments.
- [x] Require a `v*` tag ref for package-index publication.
- [x] Publish from `publish-dist/` so checksum text is retained as evidence but
  never uploaded as a Python distribution.

### Task 4: Route Docs And Validation

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/release/README.md`
- Modify: `docs/release/support_matrix.md`
- Modify: `docs/quality/release_and_packaging.md`
- Modify: `scripts/validate.py`

- [x] Register this plan as a source-of-truth path.
- [x] Route public release docs to the wheelhouse plan.
- [x] Run `python scripts/check_release_wheelhouse.py`.
- [x] Run `python -m pytest tests/test_release_wheelhouse_foundation.py -q`.
- [x] Run `python scripts/validate.py`.

## Publish Handoff

The final-release slice may bump from `0.1.0rc2` to `0.1.0` in pre-publication
status before package-index artifacts exist, but publication claims remain
blocked until the exact `v0.1.0` tag-ref wheelhouse workflow has produced
artifacts for every supported wheel target, the checksum manifest has been
retained, TestPyPI has been smoke-tested from a clean environment, PyPI trusted
publishing is configured, and the release ledger records concrete artifact
hashes and hosted CI evidence from the exact release revision.

Hosted dry-run evidence for the current `0.1.0rc2` metadata is recorded in
`docs/release/0.1.0-wheelhouse-dry-run.md`. That run validates the complete
one-source-plus-six-wheel CPU artifact shape and checksum-free publish upload
preparation with `publish-target=none`; it is not a substitute for final
`0.1.0` tag-ref package-index evidence.
The active final-release ledger is `docs/release/0.1.0.md`.

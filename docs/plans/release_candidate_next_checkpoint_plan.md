# Release Candidate Next Checkpoint Plan

## Goal

Prepare FastPauli for the next public release-candidate checkpoint by making
the support matrix, release gates, validation commands, and documentation
claims mechanically checkable before any version bump, tag, or publication
step.

Status: completed by the `0.1.0rc2` release-candidate finalization slice. The
original hardening scope below intentionally describes the pre-finalization
readiness slice; current package status is recorded in
`docs/release/0.1.0-rc2.md` and `docs/release/support_matrix.md`.

## Scope

This is a release-readiness and documentation-hardening slice. It does not
change runtime behavior, public APIs, kernel policies, benchmark data, package
version, git tags, GitHub releases, PyPI publication state, or wheel support
claims.

In scope:

```text
support matrix for CPU, CUDA, ROCm/HIP, Apple Metal, combined accelerator, and Windows claims
next-checkpoint release gates for deciding whether the next public checkpoint is 0.1.0rc2 or 0.1.0
mechanical release-readiness checks that run through scripts/validate.py
README, roadmap, AGENTS, release, and changelog routing to the new support matrix
explicit artifact and evidence requirements for the next public checkpoint
```

Out of scope:

```text
changing pyproject.toml or python/wolfgang_quantum/_version.py
creating v0.1.0rc2 or v0.1.0 tags
publishing GitHub or package-index artifacts
CUDA, ROCm/HIP, or Metal wheels
Windows wheels
new accelerator APIs, streams, graphs, workspaces, or interop surfaces
new kernel optimization campaigns
```

## Architecture

Release support is represented by a human-readable matrix in
`docs/release/support_matrix.md` and checked by
`scripts/check_release_readiness.py`. The checker is intentionally narrow: it
does not prove that release artifacts have been built, but it prevents drift
between README, roadmap, changelog, release standards, source-of-truth routing,
and the current support matrix.

`scripts/validate.py` runs the checker before build/install/test work so stale
release claims fail early in local validation and CI. The artifact builder
remains `scripts/validate_release_artifacts.py`; it continues to build and
smoke-test CPU-only artifacts from the current package version.

## Decisions

1. The hardening slice kept package version `0.1.0rc1` until a separate release
   finalization slice deliberately changed version metadata.
2. The next public checkpoint candidate was accepted as `0.1.0rc2`; final
   `0.1.0` remains a later release decision.
3. CPU artifacts remain the only wheel release lane at this checkpoint.
4. CUDA, ROCm/HIP, and Apple Metal remain source-build accelerator lanes.
5. A single mixed CUDA+HIP+Metal binary is not a release gate.
6. No package-index publication is claimed until a release slice records
   package-index credentials, artifact provenance, and clean install evidence.

## Acceptance

```text
docs/release/support_matrix.md exists and names current support status for CPU, CUDA, ROCm/HIP, Apple Metal, combined accelerator builds, Windows, and package-index publication
scripts/check_release_readiness.py verifies support matrix routing, source-build versus wheel claims, package-version consistency, and release-gate terms
scripts/validate.py runs scripts/check_release_readiness.py
README.md links the next-checkpoint plan and support matrix
AGENTS.md includes the next-checkpoint plan and support matrix in the read-first source list
docs/roadmap.md names release readiness as the next checkpoint and routes to the support matrix
docs/quality/release_and_packaging.md references the support matrix and release-readiness checker
CHANGELOG.md has an Unreleased section for post-0.1.0rc1 release-hardening work
tests cover the release-readiness checker and source-of-truth routing
python scripts/check_release_readiness.py passes
python -m pytest tests/test_release_next_checkpoint.py tests/test_release_candidate_foundation.py tests/test_release_artifact_validation.py passes
python scripts/validate.py passes
review requirements in docs/quality/code_review.md are satisfied before merge
```

## Next Public Checkpoint Gate

Before changing the version or publishing another public checkpoint, run:

```bash
python scripts/check_release_readiness.py
python scripts/validate.py
python scripts/validate_release_artifacts.py --output-dir /tmp/fastpauli-next-release-artifacts
```

The checkpoint was finalized as release candidate `0.1.0rc2` in
`pyproject.toml`, `python/wolfgang_quantum/_version.py`, `CHANGELOG.md`,
`docs/quality/release_and_packaging.md`, and
`docs/release/0.1.0-rc2.md`. If a later checkpoint becomes `0.1.0`, the same
surfaces must record the final version, final artifacts, CI runs, limitations,
and support matrix.

## Residual Risk

This slice improves release discipline but does not create release artifacts
from the eventual next tag. The next release finalization slice must still
produce tag-specific artifacts, hashes, CI evidence, and publication records
from the exact release revision.

# Wolfgang Release Evidence

This directory contains checked release evidence ledgers. A ledger is not a
published release announcement; it is the source-controlled record of commands,
artifacts, environments, support boundaries, and limitations required before a
release candidate can be published.

Release evidence must follow `docs/quality/release_and_packaging.md`.
The current release support boundary is `docs/release/support_matrix.md`.
The next public checkpoint plan is
`docs/plans/release_candidate_next_checkpoint_plan.md`.
The final `0.1.0` wheelhouse foundation plan is
`docs/plans/release_0_1_0_wheelhouse_foundation_plan.md`.
It defines the complete CPU wheelhouse shape, checksum evidence, and
tag-ref-gated PyPI package-index publication path.
The reusable cloud GPU qualification harness is
`docs/release/cloud_hardware_qualification_harness.md`; it defines the public/
private evidence split, fail-closed cleanup behavior, and local CPU dry-run for
future Hopper, Blackwell, and MI300X paid-instance validation.
The pending `0.2.3` GitHub-only successor ledger is `docs/release/0.2.3.md`; it
records the corrected capabilities fix-forward from
`bd550f4b91d575277508ca9880ec3695940c8c68`, promotes the active source version
to `0.2.3`, and preserves the immutable `v0.2.2` tag and its historical
provenance as read-only evidence rather than rewriting it.
The current final-release ledger is `docs/release/0.1.0.md`; it records the
corrected exact `v0.1.0` tag-ref CPU wheelhouse artifacts, successful TestPyPI
trusted publishing, clean TestPyPI install-smoke evidence, and the current PyPI
trusted-publisher blocker without claiming PyPI publication.
The current hosted wheelhouse dry-run evidence is recorded in
`docs/release/0.1.0-wheelhouse-dry-run.md`; it validates the complete
one-sdist-plus-six-wheel CPU artifact shape without claiming package-index
publication.
The current source version is `0.2.3`. The latest tagged release remains
`v0.2.2`; that immutable tag still points at the pre-fix candidate and is
preserved as historical provenance, not the publication target for corrected
assets. The pending release under finalization is `v0.2.3`; GitHub-only
publication is deferred for this PR and no TestPyPI or PyPI run is claimed.

Required CPU release evidence:

```text
clean git status
python scripts/validate.py
python scripts/check_release_readiness.py
python scripts/validate_release_artifacts.py --output-dir <artifact-dir>
Linux x86_64 CPU wheel smoke CI success
macOS arm64 CPU wheel smoke CI success
source distribution filename and hash
wheel filename and hash for each supported platform tag
clean-virtual-environment import smoke from each wheel
_build_info() proof for cpu_only, non-native, scalar-fallback-safe metadata
known limitations and accelerator support boundaries
```

Recommended local full-verification recipe (keeps packaging on a pristine tracked
snapshot instead of the working tree and avoids polluting the source tree with
MkDocs output):

```bash
snapshot_dir=/tmp/wolfgang-release-snapshot
artifact_dir=/tmp/wolfgang-release-artifacts
site_dir_outside_repo=/tmp/wolfgang-release-site

git worktree add --detach <snapshot-dir> HEAD
python <snapshot-dir>/scripts/validate.py
python <snapshot-dir>/scripts/check_release_readiness.py
mkdocs build --strict --site-dir <site-dir-outside-repo>
python <snapshot-dir>/scripts/validate_release_artifacts.py --output-dir <artifact-dir>
python <snapshot-dir>/scripts/audit_public_artifacts.py --sdist <artifact-dir>/*.tar.gz
python -m twine check <artifact-dir>/*.tar.gz <artifact-dir>/*.whl
git worktree remove <snapshot-dir>
```

Current checked ledgers:

```text
docs/release/0.1.0.md
docs/release/0.2.2.md
docs/release/0.2.3.md
docs/release/0.1.0-wheelhouse-dry-run.md
docs/release/0.1.0-rc2.md
docs/release/0.1.0-rc1.md
docs/release/cloud_hardware_qualification_harness.md
```

CUDA and ROCm/HIP support remain source-build lanes until a dedicated packaging
plan and release evidence ledger accept accelerator wheel support.
Apple Metal support remains a source-build evidence lane until a dedicated
packaging plan and release evidence ledger accept a Metal wheel channel.

Checked release ledgers are readiness records, not immutable artifact manifests.
Because the ledger itself is part of the source distribution payload, a final
sdist checksum cannot be recorded inside the same exact sdist without changing
the checksum. At publish time, generate checksums from the exact release tag and
publish the immutable checksum manifest outside the sdist payload. That external
manifest is the publish-time source of truth for final artifact hashes.

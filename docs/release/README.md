# Wolfgang Release Evidence

This directory holds checked release evidence ledgers. A ledger is not a
published release announcement. It is the source-controlled record of the
commands, artifacts, environments, support boundaries, and limitations required
before a release candidate can be published.

Release evidence must follow `docs/quality/release_and_packaging.md`.
The current release support boundary is `docs/release/support_matrix.md`.

The current source version is `0.2.3`. The latest tagged release is `v0.2.3`,
published as the GitHub-only successor release; no TestPyPI or PyPI run is
claimed. The immutable `v0.2.2` tag still points at the pre-fix candidate. It is
preserved as historical provenance and is not the publication target for
corrected assets.

The ledgers and plans that support these claims:

- `docs/release/0.2.3.md` is the published `0.2.3` GitHub-only successor
  ledger. It records the corrected capabilities fix-forward lineage from
  `bd550f4b91d575277508ca9880ec3695940c8c68`, promotes the active source
  version to `0.2.3`, and keeps the immutable `v0.2.2` tag and its historical
  provenance as read-only evidence instead of rewriting them.
- `docs/release/0.1.0.md` is the current final-release ledger. It records the
  corrected exact `v0.1.0` tag-ref CPU wheelhouse artifacts, successful TestPyPI
  trusted publishing, clean TestPyPI install-smoke evidence, and the current
  PyPI trusted-publisher blocker, without claiming PyPI publication.
- `docs/release/0.1.0-wheelhouse-dry-run.md` records the current hosted
  wheelhouse dry-run evidence. It validates the complete
  one-sdist-plus-six-wheel CPU artifact shape without claiming package-index
  publication.
- `docs/plans/release_candidate_next_checkpoint_plan.md` is the next public
  checkpoint plan.
- `docs/plans/release_0_1_0_wheelhouse_foundation_plan.md` is the final `0.1.0`
  wheelhouse foundation plan. It defines the complete CPU wheelhouse shape,
  checksum evidence, and tag-ref-gated PyPI package-index publication path.
- `docs/release/cloud_hardware_qualification_harness.md` is the reusable cloud
  GPU qualification harness. It defines the public/private evidence split,
  fail-closed cleanup behavior, and local CPU dry-run for future Hopper,
  Blackwell, and MI300X paid-instance validation.

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

The recommended local full verification packages from a pristine tracked
snapshot instead of the working tree, and writes MkDocs output outside the
source tree:

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
The ledger ships inside the source distribution, so it cannot record the final
sdist checksum without changing that checksum. At publish time, generate checksums from the exact release tag and
publish the immutable checksum manifest outside the sdist payload. That external
manifest is the publish-time source of truth for final artifact hashes.

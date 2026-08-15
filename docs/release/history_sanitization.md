# Git history sanitization before public visibility

This runbook removes sensitive raw benchmark/profiler objects from every reachable Git revision. It is intentionally separate from the normal feature pull request because history rewriting changes every affected commit identifier and requires coordinated remote administration.

Do not run this procedure merely to reduce repository size. Run it only after the public tree and source distribution pass `scripts/audit_public_artifacts.py`, all retained evidence is backed up privately, collaborators are notified, and repository visibility remains private.

## Preconditions

- The privacy-hardening branch is reviewed and merged into the private default branch.
- Raw evidence that must be retained has been copied to encrypted/private storage with checksums.
- A fresh clone builds, tests, packages, and passes the public-artifact audit.
- Branch protection, open pull requests, forks, releases, and collaborator recovery have been inventoried.
- A maintenance window and rollback owner are agreed.
- GitHub authentication has been verified; no long-lived credential is written to a command or file.

## 1. Create an immutable rollback bundle

From a fresh private clone:

```bash
git fetch --all --tags --prune
git bundle create ../FastPauli-before-public-sanitization.bundle --all
git bundle verify ../FastPauli-before-public-sanitization.bundle
shasum -a 256 ../FastPauli-before-public-sanitization.bundle
```

Store the bundle and checksum outside the working repository in access-controlled storage. Test restoration into a separate directory before rewriting anything.

## 2. Build the exact removal manifest

Generate the manifest from reviewed privacy-removal commits rather than from ad hoc shell globs. It should contain one repository-relative path per line and include every historical raw profiler database, host inventory, environment dump, private route/log, and other forbidden public artifact.

Review the manifest for paths that should be preserved as sanitized summaries. The manifest itself must not include credentials or copied file contents.

Typical forbidden classes include:

```text
docs/benchmarks/data/**/*.sqlite
docs/benchmarks/data/**/*.db
docs/benchmarks/data/**/*.nsys-rep
docs/benchmarks/data/**/*.ncu-rep
raw host/package/environment inventories
files containing private home paths, SSH endpoints, hostnames, or infrastructure identifiers
```

Use literal paths in the final `paths-to-remove.txt`; do not rely on shell glob expansion during the rewrite.

## 3. Rewrite a disposable mirror first

Install a reviewed `git-filter-repo` release in an isolated environment. Work from a new mirror clone, never from the maintainer's only checkout:

```bash
git clone --mirror <private-origin-url> FastPauli-sanitized.git
cd FastPauli-sanitized.git
git filter-repo --invert-paths --paths-from-file ../paths-to-remove.txt --force
```

If sensitive strings also appear in otherwise retained text files, use a reviewed replacement-expression file with `--replace-text`. Do not place real credentials in the replacement file; use exact non-secret infrastructure identifiers and replace them with neutral markers.

If the full-history audit still reports findings after path removal, do not keep
expanding an opaque rewrite indefinitely. A safer publication option is a new
repository with a single root commit created from the fully reviewed current
tree. Exclude private workspace-only files, run every gate below against that
one-commit repository, and retain the original history only in private archival
storage. This clean-history option intentionally sacrifices public commit
provenance to make the privacy boundary independently auditable.

## 4. Verify every reachable object

Create a normal clone from the rewritten mirror and run:

```bash
python scripts/audit_public_artifacts.py --tracked
python scripts/audit_public_artifacts.py --history
python scripts/validate.py
python scripts/validate_release_artifacts.py --output-dir /tmp/fastpauli-sanitized-release
```

Also run a dedicated full-history secret scanner such as Gitleaks or TruffleHog, review all findings, and scan the built sdist member-by-member. Search for known former paths, usernames, addresses, hostnames, environment-variable captures, GPU UUIDs, and profiler file signatures without printing discovered secret values.

Verify that:

- retained benchmark reports still link correctly;
- public plots and sanitized summaries remain reproducible;
- tags and release notes point to the intended rewritten source;
- package artifacts contain no removed history/data;
- repository and sdist sizes match the public artifact policy;
- a clean CPU wheel installs and executes the quickstart.

Any finding is an abort gate. Fix the removal/replacement manifest and restart from a fresh mirror.

## 5. Coordinate the remote rewrite

Only after local verification:

1. Temporarily prevent merges and package publication.
2. Export issue/release metadata if required for rollback.
3. Disable or adjust branch protection for the maintenance window.
4. Force-update all rewritten branches and tags from the verified mirror.
5. Restore branch protection and invalidate stale workflow approvals.
6. Ask collaborators to archive unpushed work and fresh-clone; do not encourage merging old history into the rewritten repository.
7. Delete cached release/source artifacts containing removed data.
8. Request cache/purge assistance from the hosting provider if sensitive objects were ever externally visible.

Do not publish or change visibility in the same command sequence as the force push. First clone the remote again and repeat all verification against the hosted result.

## 6. Post-rewrite gates

- Confirm the hosted default branch and every tag contain no forbidden object.
- Re-run hosted CI and release dry-run jobs without publishing.
- Verify GitHub private vulnerability reporting and security policy.
- Record the rewrite date, tool version, reviewed manifest checksum, verification commands, and responsible maintainer in the private release ledger.
- Rotate any credential discovered during the audit even if history removal succeeded.

## Rollback

If branch/tag integrity, build reproducibility, or privacy verification fails, stop publication and restore the private remote from the verified bundle during the maintenance window. Do not partially mix rewritten and original histories. Investigate in a disposable clone and repeat the complete procedure.

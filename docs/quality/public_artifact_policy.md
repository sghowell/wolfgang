# Public Artifact Policy

Wolfgang's public repository and release artifacts are products, not research
archives. Every tracked file must be safe for anonymous redistribution, and the
source distribution must contain only material needed to build and understand
the package.

## Public repository boundary

Allowed benchmark evidence is a reviewed report, plot, or derived JSON summary.
Public evidence may identify generic hardware and software versions but must not
identify a person, account, machine, network, or private filesystem layout.
Never track:

- personal home paths, usernames, hostnames, network addresses, or SSH targets;
- GPU/device UUIDs, serial numbers, unrestricted environment data, or process
  and filesystem inventories;
- raw benchmark captures, command logs, crash dumps, or remote access details;
- native profiler reports/databases or opaque profiler formats;
- credentials or secrets of any kind.

Redaction is deterministic: omit sensitive fields when collecting data. If an
older derived summary must retain a structural marker, use a category marker
such as `<private-path>` rather than a transformed identifier. Hashing personal
or infrastructure identifiers is not sanitization.

Raw evidence belongs in access-controlled research storage under the archive
rules in `../benchmarks/data/README.md`. Public conclusions remain in sanitized
summaries and reports.

## Source distribution boundary

The sdist contains only package/build inputs: project metadata, license/readme,
CMake inputs, native headers/sources, bindings, and Python package sources. It
excludes benchmark machinery and evidence, documentation plans, tests, internal
scripts, remote tools, repository automation, and other development material.
The public scanner enforces allowed roots, member count, unpacked size, privacy
patterns, and forbidden profiler formats.

## Required gate

Run all checks for a release candidate. The history check is mandatory before
repository visibility changes; it is expected to fail in a private development
repository until the reviewed history-sanitization procedure is complete.

```bash
python scripts/audit_public_artifacts.py --tracked
python scripts/audit_public_artifacts.py --history
python -m build --sdist --outdir dist
python scripts/audit_public_artifacts.py --sdist dist/wolfgang-quantum-*.tar.gz
```

Archive/no-`.git` tracked-tree audits remain fail-closed. Use either an explicit
`--tracked-manifest <path>` or the checked-in `scripts/tracked_files_manifest.txt`
exported with the archive. If neither Git metadata nor a tracked manifest exists,
`python scripts/audit_public_artifacts.py --tracked` must fail rather than guess.

The scanner deliberately prints only the affected path and rule. Do not paste a
matched private value into an issue, review, commit message, or sanitization
report. Treat a finding as potentially sensitive until reviewed locally.

Any exception requires a documented security review and a narrowly scoped test;
convenience and reproducibility alone are not exceptions.

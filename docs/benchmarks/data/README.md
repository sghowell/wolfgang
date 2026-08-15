# Public Benchmark Data

This directory contains **sanitized, derived summaries only**. These summaries
support the conclusions in `../reports/` without publishing infrastructure
identifiers or machine-generated evidence that is unsafe or impractical to
review.

## What belongs here

A public summary may contain:

- benchmark parameters, deterministic seeds, and operation names;
- aggregate timing and correctness results;
- generic CPU/GPU model, architecture, toolkit, compiler, and OS versions;
- the tested Git revision and documented limitations.

It must not contain usernames, home paths, hostnames, network addresses, SSH
targets, device UUIDs, unrestricted environment variables, process inventories,
raw command output, or profiler databases/reports. Metadata collection must use
an explicit allowlist.

## Private evidence archive

Raw benchmark JSON, validation logs, host inventories, profiler exports, and
profiler-native binaries belong in access-controlled research storage, not Git
and not a Python source distribution. Archive entries should record the public
summary/report they support, the revision, collection date, retention owner,
and a checksum. Access to that archive does not make an item suitable for later
publication: it must be reviewed and reduced to a new sanitized summary first.

Run the public gate before committing any benchmark evidence:

```bash
python scripts/audit_public_artifacts.py --tracked
```

The scanner reports only paths and policy rule names; it never echoes matched
content.

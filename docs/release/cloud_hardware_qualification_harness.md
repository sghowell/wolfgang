# Wolfgang cloud hardware qualification harness

This document defines the reusable, non-secret harness that future paid Hopper,
Blackwell, and MI300X release-qualification runs must use. It exists so the
repo can dry-run the structure locally on CPU, then execute the exact same
bundle shape on cloud hardware without publishing raw private evidence.

Canonical entrypoint:

```bash
python scripts/cloud_hardware_qualification_harness.py bundle --lane hopper --output-dir /tmp/wolfgang-hopper-harness
python scripts/cloud_hardware_qualification_harness.py bundle --lane blackwell --output-dir /tmp/wolfgang-blackwell-harness
python scripts/cloud_hardware_qualification_harness.py bundle --lane mi300x --output-dir /tmp/wolfgang-mi300x-harness
```

Archive/no-`.git` runs are supported only with explicit trusted source identity.
Either pass `--source-commit <40-hex>` (and optionally `--source-tree-state archive`)
or export from Git with `scripts/archive_source_identity.json` substituted via
`.gitattributes`. The harness never invents commit ids; if neither `.git` nor an
explicit/archive identity is available, it fails closed.

Each bundle contains:

```text
RUNBOOK.md
public/qualification_manifest.json
public/benchmark_policy.json
private/README.md
scripts/run_lane.sh
```

## Public/private boundary

`public/` is the checked-in or shareable surface. It may contain sanitized
derived evidence only:

- commit SHA and clean/dirty tree state
- compiler command and captured version string
- driver and CUDA/ROCm version strings
- device model and reported architecture
- build flags
- wheel/sdist hashes
- test totals/pass/fail/skip counts
- diagnostics summaries
- numerical parity summaries
- interop-check summaries
- benchmark medians, p95, and timing policy
- fail-closed cleanup outcome summaries

`private/` is for raw logs, verbose environment captures, profiler databases,
provider metadata, SSH details, and exact instance-routing information. Do not
commit `private/` contents. Derive only the public summary fields needed by the
manifest.

## Required gates

Every lane must record these gates before the support boundary changes:

1. Functional gate: clean source build, canonical import, compat import, and
   relevant test subset with no unexpected failures or silent CPU fallback.
2. Numerical parity gate: deterministic CPU-vs-device comparisons using the
   release thresholds from the hardware qualification plan.
3. Interop gate: host/device roundtrip, non-contiguous-input smoke, and the
   lane-specific adapter checks actually claimed in user-facing docs.
4. Diagnostics gate: compute-sanitizer on NVIDIA or rocprof/credible ROCm
   diagnostics on MI300X, with the public artifact reduced to a sanitized
   outcome summary.
5. Performance gate: fixed benchmark inputs with 10 warm-up and 30 timed
   iterations against the same architecture's frozen baseline only.
6. Reproducibility gate: two reruns on the same image and one fresh-provision
   rerun with identical test counts and bounded benchmark variance.

## Fail-closed cleanup policy

The generated `scripts/run_lane.sh` uses `set -euo pipefail` and a cleanup trap.
On any failing gate, the harness must stop and invoke the operator-provided
termination command rather than continuing with ambiguous evidence. A failed run
may still publish a sanitized failure summary, but it must not leave the cloud
instance running out of habit and must not convert partial evidence into a
support claim.

## Inventory capture policy

Remote inventory collectors are allowlisted:

- `tools/remote/collect_cuda_inventory.sh`
- `tools/remote/collect_rocm_inventory.sh`

They intentionally avoid raw environment dumps, `nvidia-smi -q`, `rocminfo`,
SSH targets, UUID queries, and other broad/private metadata. If a new field is
needed for release support, add the smallest possible allowlisted query and keep
it auditable.

## Local dry-run expectation

The repository-level dry run is CPU-safe and non-provisioning. It must only:

- generate the bundle for a lane,
- fill `public/qualification_manifest.json` with capture stubs plus current git
  commit/tree state,
- write the benchmark policy,
- render fail-closed instructions, and
- keep all live execution, provider-specific commands, and raw evidence capture
  out of the public tree.

For a fresh `git archive` validation, keep `scripts/archive_source_identity.json`,
`scripts/archive_portability.py`, and `scripts/tracked_files_manifest.txt` in the
exported tree. The identity file supplies sanitized commit provenance, while the
tracked-files manifest provides deterministic `--tracked` enumeration for
`scripts/audit_public_artifacts.py` when `.git` is absent.

## Release support boundary

This harness does not itself broaden support claims. The support boundary stays
where `docs/release/support_matrix.md` says it is until a reviewed release slice
runs this harness on named hardware and records sanitized derived evidence.

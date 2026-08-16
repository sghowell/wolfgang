# Wolfgang Release Support Matrix

This matrix is the current public support boundary for Wolfgang release
claims. It separates installable artifacts from source-build accelerator
evidence so docs and release notes do not overstate what users can install.

## Current Package Version

```text
Source version: 0.2.2
Next intended release: v0.2.2 (pending PR, not tagged or published)
Latest tagged release: v0.2.1
Latest tagged release PyPI status: publication pending trusted-publisher configuration
Release under finalization: v0.2.2 pending publication
Previous checkpoint: v0.1.0rc2 GitHub prerelease
0.1.0 wheelhouse foundation plan: docs/plans/release_0_1_0_wheelhouse_foundation_plan.md
0.1.0 wheelhouse dry-run evidence: docs/release/0.1.0-wheelhouse-dry-run.md
0.1.0 release ledger: docs/release/0.1.0.md
0.2.2 release-preparation ledger: docs/release/0.2.2.md
Reusable cloud hardware qualification harness: docs/release/cloud_hardware_qualification_harness.md
0.1.0 tag-ref wheelhouse run: 25452754832
Corrected 0.1.0 TestPyPI validation run: 25462760923
Latest 0.1.0 PyPI publication run: 25462997972
TestPyPI validation: historical 0.1.0 dry-run evidence only
TestPyPI validation: final dry run published and smoke-tested
PyPI publication: deferred for v0.2.2; no TestPyPI or PyPI run has been attempted
PyPI publication: unavailable pending trusted-publisher configuration
```

## Support Matrix

| Surface | Current support status | Evidence | Artifact status | Boundary |
| --- | --- | --- | --- | --- |
| CPU default package | CPU artifact target | `docs/release/0.2.2.md`, `docs/release/0.1.0.md`, `docs/release/0.1.0-rc2.md`, `docs/release/0.1.0-rc1.md`, `docs/release/0.1.0-wheelhouse-dry-run.md`, `docs/plans/release_0_1_0_wheelhouse_foundation_plan.md`, CI CPU wheel smoke, `scripts/validate_release_artifacts.py` | Source distribution and macOS arm64 CPU wheel are published for `v0.1.0rc2`; the corrected final `v0.1.0` tag-ref workflow produced one source distribution, six CPU wheels, checksum evidence, successful TestPyPI upload, and a clean TestPyPI install smoke; PyPI publication is blocked by PyPI trusted-publisher configuration | Portable scalar fallback required; native CPU tuning disabled for release wheels |
| CUDA accelerator | Source-build support | H100, A100, and RTX PRO 6000 Blackwell source-build reports plus CUDA validation lanes | CUDA wheels remain unavailable | CUDA support requires an explicit `WOLFGANG_ENABLE_CUDA=ON` source build and visible CUDA runtime |
| ROCm/HIP accelerator | Source-build support | MI300X `gfx942` bring-up, optimization, release-support, and architecture-readiness reports | ROCm/HIP wheels remain unavailable | Broader AMD GPU support remains unavailable without per-architecture evidence |
| Apple Metal accelerator | Source-build evidence | Apple M4 Pro source-build reports through Apple Metal Campaign 3 | Metal wheels remain unavailable | Generic Apple GPU support is unavailable; evidence is local Apple M4 Pro source-build evidence |
| Combined accelerator binary | Unsupported by policy | Backend-neutral accelerator Campaign 9 configure-time rejection evidence | Combined accelerator wheels remain unavailable | Normal builds target CPU-only, CUDA-only, HIP-only, or Metal-only modes |
| Windows | Unsupported release target | No Windows CI artifact evidence | Windows wheels remain unavailable | Windows support must not be claimed before clean build, import, and artifact evidence exists |
| TestPyPI validation | Published final dry run | `docs/release/0.1.0.md` records successful TestPyPI trusted publishing, artifact hashes, and clean TestPyPI install smoke evidence | Final `0.1.0` artifacts are present on TestPyPI as validation evidence | TestPyPI is not the production release channel and does not imply PyPI publication |
| PyPI final release | Not published | `docs/release/0.1.0.md` records final-release pre-publication status, exact-tag wheelhouse evidence, corrected tag strategy, successful TestPyPI validation, and the current PyPI trusted-publisher blocker; `docs/release/0.1.0-rc2.md` records GitHub prerelease publication only; `docs/plans/release_0_1_0_wheelhouse_foundation_plan.md` defines the final wheelhouse workflow guardrails | PyPI publication is not claimed | PyPI publication requires matching PyPI trusted-publisher configuration, complete CPU wheelhouse evidence from the release tag, TestPyPI install smoke evidence, and checksum-free package upload set |

## Next Checkpoint Requirements

A new public checkpoint must record:

```text
version metadata and changelog entry
clean git status at release revision
python scripts/check_release_readiness.py
python scripts/check_release_wheelhouse.py
python scripts/validate.py
python scripts/validate_release_artifacts.py --output-dir <artifact-dir>
manual release-wheelhouse workflow evidence from the exact release tag when PyPI artifacts are claimed
complete CPU wheelhouse checksum manifest from one sdist plus six CPU wheels
publish upload set containing only .whl and .tar.gz artifacts
hosted CI run for the release revision
source distribution filename and hash
wheel filename and hash for every release-supported wheel platform
clean virtual-environment install smoke for each produced wheel
support matrix review confirming no CUDA, ROCm/HIP, Metal, combined accelerator, Windows, or PyPI claim was added without evidence
benchmark report references for any performance claims
known limitations and unsupported surfaces
cloud GPU support claims routed through docs/release/cloud_hardware_qualification_harness.md with sanitized derived evidence only
release-preparation ledger for the pending version documenting deferrals, routing, and unchanged historical evidence
```

## Claim Rules

Release notes and README text may say:

```text
CPU wheels are the CPU artifact target
CUDA source-build support
ROCm/HIP source-build support on MI300X gfx942
Apple Metal source-build evidence on Apple M4 Pro
```

Release notes and README text must not say:

```text
CUDA wheels are available
ROCm/HIP wheels are available
Metal wheels are available
combined accelerator wheels are available
Windows wheels are available
PyPI publication is complete
generic Apple GPU support is available
broad AMD GPU support is available
```

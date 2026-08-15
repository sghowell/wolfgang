# Wolfgang Release Candidate Foundation Plan

## Goal

Prepare Wolfgang for a first CPU release-candidate lane without changing public
runtime behavior or adding accelerator wheel support claims.

This slice is release infrastructure only. CUDA and ROCm/HIP remain
source-build accelerator paths, CPU wheels remain the default packaging target,
and all accelerator packaging claims stay gated by
`docs/quality/release_and_packaging.md`.

## Scope

In scope:

```text
checked release-candidate plan
checked changelog
checked release evidence ledger
CPU source distribution build smoke
CPU wheel build, clean-venv install, import, and scalar-fallback smoke
GitHub Actions CPU wheel smoke on Linux x86_64 and macOS arm64
validation registration for release docs and artifact checks
README, roadmap, AGENTS, and release-standard routing updates
```

Out of scope:

```text
publishing a package
changing the public Python API
changing C++ or accelerator kernels
CUDA wheels
ROCm/HIP wheels
combined accelerator wheels
Windows wheels
Apple GPU implementation or Metal/MPS planning
```

## Architecture

The release candidate foundation keeps the packaging model target-specific:

```text
CPU wheel: default artifact, accelerator build mode cpu_only
CUDA source build: explicit WOLFGANG_ENABLE_CUDA=ON path
ROCm/HIP source build: explicit WOLFGANG_ENABLE_HIP=ON path
combined CUDA+HIP build: configure-time rejection under the current policy
```

The CPU artifact validator must build with:

```text
WOLFGANG_ENABLE_CUDA=OFF
WOLFGANG_ENABLE_HIP=OFF
WOLFGANG_ENABLE_NATIVE=OFF
```

The clean-install smoke must assert:

```text
import wolfgang_quantum succeeds from the produced wheel
wolfgang_quantum._wolfgang_core._build_info() reports accelerator_build_mode == "cpu_only"
cuda_enabled and hip_enabled are false
native_enabled is false
compiled_backends == ["cpu"]
the scalar CPU backend is compiled
a small PauliSum simplify operation returns the expected result
```

## Deliverables

```text
CHANGELOG.md
docs/release/README.md
docs/release/0.1.0-rc1.md
scripts/validate_release_artifacts.py
tests/test_release_candidate_foundation.py
tests/test_release_artifact_validation.py
.github/workflows/ci.yml CPU wheel smoke job
README.md release validation instructions
docs/roadmap.md release-candidate foundation checkpoint
AGENTS.md source-of-truth registration
scripts/validate.py source-of-truth registration
```

## Acceptance

This plan is complete when:

```text
release candidate foundation docs are source-of-truth registered
CHANGELOG.md records current user-facing status and limitations
release evidence ledger defines the required evidence for 0.1.0-rc1
scripts/validate_release_artifacts.py builds a source distribution and CPU wheel
scripts/validate_release_artifacts.py installs the produced wheel into a clean virtual environment
the wheel smoke verifies CPU-only, non-native, scalar-fallback-safe build metadata
CI has a Linux and macOS CPU wheel smoke job
artifact directories and generated wheels are ignored by .gitignore
tests cover plan registration, support boundaries, CI wiring, and validator command construction
python scripts/validate.py passes
an independent review stage is completed before merge
```

## Closeout Evidence

Record the following in `docs/release/0.1.0-rc1.md` when a release candidate is
actually cut:

```text
git revision
artifact names and hashes
validation commands
CI run URLs
supported Python versions
supported platform tags
CPU scalar fallback evidence
CUDA source-build status
ROCm/HIP source-build status
known limitations
benchmark report links for any performance statements
```

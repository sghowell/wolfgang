# Wolfgang Release And Packaging Standards

This document defines packaging, versioning, release, and distribution expectations.

## Packaging Goals

Wolfgang should be easy to install for CPU users and explicit for CUDA users.

Initial policy:

```text
CPU package is the default
CUDA is opt-in at build time until CUDA packaging is deliberately designed
source builds use scikit-build-core and CMake
wheels must not require CUDA unless they are explicitly CUDA-specific artifacts
source distributions contain build inputs only and follow docs/quality/public_artifact_policy.md
```

## Project Metadata

Phase 1 should define:

```text
project name: wolfgang-quantum
current development version: 0.2.2
latest tagged release: 0.2.1
requires-python: >=3.10
license metadata matching LICENSE
runtime dependency: numpy
optional extras for test, bench, qiskit, and openfermion as they land
```

Do not advertise optional extras until the corresponding adapter or benchmark support exists or is intentionally scaffolded with clear skips.

## Build Artifacts

Required artifacts by maturity:

```text
Phase 1: editable install and import smoke
early CPU implementation: source distribution build smoke
first CPU release candidate: CPU wheels for supported platforms
release candidate foundation: scripts/validate_release_artifacts.py builds source distribution and CPU wheel, installs the wheel into a clean virtual environment, and verifies CPU-only scalar fallback metadata
final 0.1.0 wheelhouse foundation: docs/plans/release_0_1_0_wheelhouse_foundation_plan.md defines cibuildwheel CPU wheels, checksum manifests, exact CPU wheelhouse completeness checks, publish-only artifact filtering, and tag-ref-gated TestPyPI/PyPI trusted-publishing gates
CUDA foundation: source CUDA build validation
future CUDA distribution: separate CUDA wheel strategy and compatibility matrix
```

CPU wheels should include a portable scalar baseline and may include runtime-dispatched optimized CPU paths once implemented.

CPU and CUDA hardware target claims must follow `docs/architecture/hardware_targets_and_testing.md`.

## Versioning And Changelog

Use semantic versioning after `1.0.0`. Before `1.0.0`, still document user-visible changes.

Wolfgang records user-visible changes in:

```text
CHANGELOG.md
```

Release notes should include:

```text
version
date
summary
user-facing changes
breaking changes
validated environments
known limitations
benchmark evidence when performance is mentioned
```

## Wheel And Platform Policy

Do not claim wheel support until it is built and tested.

Initial CPU wheel targets should be chosen by CI availability and user needs. Candidate targets:

```text
Linux x86_64
macOS arm64
macOS x86_64 if CI support is available
```

Windows support should not be claimed until validated.

Release wheels must not use native CPU tuning such as `-march=native`. Any runtime-dispatched optimized CPU code must keep a portable scalar fallback and must be tested on the release artifact.

The final `0.1.0` CPU wheelhouse readiness lane is
`docs/plans/release_0_1_0_wheelhouse_foundation_plan.md`. It keeps the first
package-index-ready wheelhouse to manylinux x86_64 and macOS arm64 for Python
3.10, 3.11, and 3.12, with CUDA, ROCm/HIP, Metal, Windows, combined
accelerator, macOS x86_64, and broad hardware support claims unavailable until
separate evidence accepts those surfaces.
Package-index publish jobs must run only from the exact `v${project.version}`
tag ref and must upload from a checksum-free package artifact directory such as
`publish-dist/`. Workflows must use immutable full-SHA action pins, read-only
default permissions, job-scoped OIDC, concurrency controls, dependency auditing,
a software bill of materials, and artifact attestations for the release payload.

CUDA wheel policy is deferred until after source CUDA support is stable. Any CUDA wheel plan must specify:

```text
CUDA toolkit/runtime compatibility
GPU architecture support
wheel naming strategy
host compiler constraints
test environment
artifact size expectations
CUDA driver, toolkit, and compute capability evidence
```

## Release Candidate Gate

A release candidate requires:

```text
clean git status
all required validation commands pass
CI green on required jobs
source distribution builds
CPU wheels build and import
scripts/validate_release_artifacts.py passes for every release-supported CPU wheel platform
CPU wheel scalar fallback is validated
release-relevant review findings are resolved or explicitly accepted
README installation and quickstart are current
API docs are current for public APIs
license metadata is correct
benchmark report is refreshed if performance claims are present
known limitations are documented
```

The reusable paid-instance source-build lane generator is
`docs/release/cloud_hardware_qualification_harness.md` with implementation in
`scripts/cloud_hardware_qualification_harness.py`. Use it for local CPU dry-runs
and for future Hopper, Blackwell, and MI300X qualification bundles so that raw
environment/profiler data stays in `private/` and the public tree only records
sanitized derived evidence.

CUDA release candidates additionally require:

```text
WOLFGANG_ENABLE_CUDA=ON source build evidence
CUDA runtime tests on at least one supported GPU
CUDA toolkit, driver, host compiler, and GPU architecture evidence matches claimed support
CPU-only build evidence without CUDA installed
CUDA availability errors and skips tested
CUDA-focused review completed for CUDA release candidates
```

## ROCm Source-Build Release-Support Policy

ROCm/HIP support is source-build-only until a separate packaging campaign
accepts a wheel channel. Campaign 7 is the first completed ROCm
release-support campaign, tracked by
`docs/plans/mi300x_rocm_optimization_campaign7_plan.md` and reported in
`docs/benchmarks/reports/rocm_mi300x_campaign7_2026-04-30.md`.
Campaign 8 is the latest completed packaging-facing ROCm gate, tracked by
`docs/plans/mi300x_rocm_optimization_campaign8_plan.md` and reported in
`docs/benchmarks/reports/rocm_campaign8_architecture_readiness_2026-05-01.md`.
It defines the wheel, CI hardware, clean-machine install, and support-matrix
gates that any later ROCm wheel campaign must satisfy before a ROCm wheel claim
can move out of unavailable status.

ROCm source-build release-support evidence requires:

```text
WOLFGANG_ENABLE_HIP=ON source build evidence
WOLFGANG_HIP_ARCHITECTURES value matching the validated GPU architecture
ROCm runtime, driver, toolkit, HIP compiler, GPU model, and gfx target evidence
retained HIP operation tests for transfers, commutation, device-output consumers, simplify, expectation, and matmul
CPU-only validation showing WOLFGANG_ENABLE_HIP=OFF does not require ROCm
WOLFGANG_ENABLE_CUDA=ON with WOLFGANG_ENABLE_HIP=ON configure-time rejection
target-specific build evidence rather than a mixed CUDA+HIP binary claim
benchmark smoke with transfer-inclusive and device-resident or compact-consumer boundaries where applicable
rocprof trace/stats evidence or a precise provider/tooling blocker
README support language that separates ROCm source-build evidence from wheel support and broad AMD portability claims
```

ROCm wheels remain unavailable until a dedicated packaging plan specifies:

```text
supported package channel decision
ROCm toolkit and runtime compatibility policy
GPU architecture support and evidence level for each architecture
Linux distribution and manylinux policy
runtime dependency policy for ROCm libraries, including bundling or external-runtime requirements
wheel naming strategy
artifact size expectations
CI hardware that can build and import the wheel
clean-machine install test for the produced artifact
support-matrix wording that distinguishes source-build evidence from wheel support
```

Campaign 8 keeps ROCm wheels unavailable but accepts this gate as the minimum
release-packaging contract for any later ROCm wheel campaign.

## Apple Metal Source-Build Packaging Policy

The Apple accelerator contract is `docs/architecture/apple_accelerator.md`.
The implementation handoff is
`docs/plans/apple_metal_mps_bringup_plan.md`.

Metal support is source-build-only until a dedicated packaging plan accepts a
macOS arm64 accelerator wheel channel. The CPU wheel must remain Metal-free
unless a release plan proves that Metal framework linkage, import behavior,
artifact size, and unsupported-device behavior satisfy the release evidence
standard.

The first bring-up report is
`docs/benchmarks/reports/apple_metal_bringup_2026-05-01.md`. It records a
successful `WOLFGANG_ENABLE_METAL=ON` source build and a local runtime blocker:
the host reports an Apple M4 Pro GPU with Metal support through
`system_profiler`, while `MTLCreateSystemDefaultDevice()` and
`MTLCopyAllDevices()` return no visible Metal device in the Codex execution
context. That is not sufficient for a Metal source-build support claim.

Release evidence for a Metal source-build claim must include:

```text
WOLFGANG_ENABLE_METAL=ON source build command
named Apple Silicon SoC and Metal device
macOS version
Xcode or Command Line Tools version
CPU-only validation with WOLFGANG_ENABLE_METAL=OFF on the same host
WOLFGANG_ENABLE_METAL=ON with CUDA or HIP configure-time rejection
retained Metal operation tests for transfers, commutation, and compact consumers
benchmark smoke with transfer-inclusive, device-resident, and host-materialized boundaries where applicable
Instruments, Metal System Trace, xctrace, or a precise profiler tooling blocker
README support language that separates Metal source-build evidence from Metal wheel support
```

Metal wheels remain unavailable until a dedicated packaging plan specifies:

```text
supported package channel decision
framework linkage policy for Metal, Foundation, MetalPerformanceShaders, and MetalPerformanceShadersGraph
macOS and Apple Silicon support matrix
runtime behavior on machines without a usable Metal device
clean-machine install test for the produced artifact
support-matrix wording that distinguishes CPU wheels from Metal source builds
```

## Release Evidence Template

Use `docs/release/README.md`, the pending release ledger
`docs/release/0.2.2.md`, the current release-preparation ledger,
`docs/release/0.1.0.md`, the current published final-release ledger,
`docs/release/0.1.0-wheelhouse-dry-run.md`, and historical ledgers such as
`docs/release/0.1.0-rc2.md` and `docs/release/0.1.0-rc1.md` for checked-in
release evidence. Release evidence must include:

```text
Version:
Git revision:
Date:
Artifacts:
Validation commands:
CI runs:
Supported Python versions:
Supported platforms:
CUDA status:
CPU target status:
Benchmark reports:
Known limitations:
```

The current claim boundary is `docs/release/support_matrix.md`. Before any
release candidate or final release is tagged, run
`scripts/check_release_readiness.py` to verify that README, roadmap, changelog,
release standards, source-of-truth routing, and the support matrix agree on
CPU wheels, CUDA source-build support, ROCm/HIP source-build support, Apple
Metal source-build evidence, unavailable accelerator wheels, unavailable
combined accelerator wheels, unsupported Windows wheels, and package-index
publication status. The next-checkpoint handoff is
`docs/plans/release_candidate_next_checkpoint_plan.md`; the final `0.1.0`
wheelhouse handoff is
`docs/plans/release_0_1_0_wheelhouse_foundation_plan.md`.

Do not publish a release from a dirty worktree.

## Packaging Review Checklist

Before changing packaging:

```text
does editable install still work?
does CPU-only build work without CUDA installed?
does the wheel avoid native CPU tuning unless it is a deliberately local source build?
does CPU dispatch preserve a validated scalar fallback?
are optional dependencies still optional?
does project metadata match README claims?
does the wheel include only intended packages and extension modules?
are build directories and artifacts ignored by .gitignore?
does CI run the same validation entrypoint as local development?
does scripts/check_release_readiness.py pass before artifacts are built?
does scripts/validate_release_artifacts.py build, install, import, and verify CPU-safe build metadata?
```

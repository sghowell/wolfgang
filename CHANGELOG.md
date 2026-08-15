# FastPauli Changelog

All user-visible FastPauli changes are recorded here before each release.

## Unreleased

Next version: 0.2.0

### Added

- Release supply-chain hardening with exact project-version tag binding,
  immutable GitHub Action pins, least-privilege workflow permissions,
  dependency auditing, SBOM generation, artifact attestations, and automated
  dependency updates.

- Apple Metal Campaign 8 simplify performance-relevance evidence with private
  timing decomposition, pipeline-cache boundary, dispatch-count metadata,
  Campaign 8 benchmark rows, and refreshed benchmark/report/README evidence.
  Public Metal `DevicePauliSum.simplify()` remains on the transfer-reference
  bridge unless a later design proves a broader retained device-resident
  implementation.
- Apple Metal Campaign 7 checked device-resident simplify primitive stack with
  a private benchmark-only one-word Metal sort, prefix-sum, reduce-by-key, and
  survivor-compaction path; a private validation hook; Campaign 7 benchmark
  rows; and refreshed benchmark/report/README evidence. The candidate remains
  benchmark-only and fixed-dyadic-coefficient limited because this Apple Metal
  toolchain rejects `double` arithmetic in kernels; accepted rows also require
  exact fixed32 accumulation and uint64 squared-magnitude tolerance comparison.
  Public Metal `DevicePauliSum.simplify()` stays on the transfer-reference
  bridge.
- Apple Metal Campaign 7 planning in
  `docs/plans/apple_metal_optimization_campaign7_plan.md` for checked
  device-resident simplify primitives while keeping public workspaces, public
  raw Metal exports, Metal statevector expectation, Metal matmul, Metal wheels,
  PyPI publication, Windows support, and older macOS compatibility out of
  scope.
- Apple Metal Campaign 6 device-resident simplify groundwork with a private
  `MetalWorkspace` model, `WorkspaceTimingMode`, Campaign 6 simplify benchmark
  cases, `metal_simplify_workspace_probe` status rows, Campaign 6 benchmark
  JSON, report, validation smoke, and README landscape evidence. The
  device-resident simplify candidate remains blocked until checked Metal
  sort/prefix/reduce primitives exist; the public Metal simplify behavior
  remains the Campaign 5 transfer-reference correctness bridge.
- Apple Metal Campaign 6 planning in
  `docs/plans/apple_metal_optimization_campaign6_plan.md` for
  source-build-only device-resident simplify groundwork while keeping public
  Metal workspaces, Metal statevector expectation, Metal matmul, Metal wheels,
  PyPI publication, Windows support, and older macOS compatibility out of
  scope.
- Apple Metal Campaign 5 source-build simplify behavior with
  `DevicePauliSum.simplify()` parity for empty, duplicate-heavy, cancellation,
  and multi-word operators; finite non-negative tolerance validation; Campaign
  5 benchmark JSON, report, validation smoke, and README landscape evidence.
  The retained Metal simplify path is recorded as
  `metal_simplify_transfer_reference` with the
  `device_to_host_cpu_simplify_host_to_device` boundary and is not a
  device-resident GPU duplicate-reduction speedup.
- Apple Metal Campaign 5 planning in
  `docs/plans/apple_metal_optimization_campaign5_plan.md` for
  source-build-only Metal `DevicePauliSum.simplify()` bring-up, including
  correctness, benchmark, validation, and documentation acceptance criteria
  while keeping PyPI publication, Windows support, older macOS support, Metal
  wheels, Metal statevector expectation, and Metal matmul out of scope.
- Apple Metal Campaign 4 benchmark evidence and documentation, including a
  benchmark-only parallel compact total-count selector, larger local Apple M4
  Pro benchmark cases, and refreshed broad accelerator landscape assets while
  keeping public Metal API support unchanged.

## 0.1.0

Status: final PyPI release under validation. `v0.1.0` tag-ref
artifacts have been built and validated, TestPyPI trusted publishing and clean
install smoke have passed, and PyPI publication is blocked by PyPI
trusted-publisher configuration for the observed `pypi` environment claims.
PyPI publication is not claimed.

### Added

- Final `0.1.0` CPU wheelhouse readiness foundation with cibuildwheel
  configuration, a manual release-wheelhouse workflow, installed-wheel smoke,
  checksum manifest generation, complete wheelhouse checks, checksum-free
  publish upload filtering, and tag-ref-gated TestPyPI/PyPI trusted-publishing
  gates while keeping accelerator and Windows wheel claims unavailable.
- Hosted release-wheelhouse dry-run evidence for the complete current CPU
  artifact shape: one source distribution, six CPU wheels, checksum manifest
  validation, merged-artifact metadata checks, and checksum-free package-index
  upload preparation with publication disabled.
- Exact-tag `v0.1.0` wheelhouse evidence for one source distribution, six CPU
  wheels, checksum validation, and checksum-free publish upload preparation; the
  previous TestPyPI blockers are cleared, TestPyPI upload and clean install
  smoke passed, and PyPI upload is blocked by missing matching PyPI
  trusted-publisher configuration.

## 0.1.0rc2

Status: published as GitHub prerelease `v0.1.0rc2` with source distribution,
macOS arm64 CPU wheel, and external checksum manifest artifacts.

### Added

- Release-readiness hardening for the next public checkpoint with a checked
  support matrix, release-readiness checker, and source-of-truth routing for
  CPU wheels, CUDA source-build support, ROCm/HIP source-build support, Apple
  Metal source-build evidence, unavailable accelerator wheels, unsupported
  Windows wheels, and package-index publication status.

### Support Boundaries

- CPU wheels remain the release-candidate artifact target.
- CUDA support remains source-build-only unless a later CUDA packaging plan adds
  wheel evidence.
- ROCm/HIP support remains source-build-only unless a later ROCm packaging plan
  adds wheel evidence.
- Apple Metal support remains source-build-only unless a later Apple
  accelerator packaging plan adds macOS arm64 wheel evidence.
- CUDA wheels remain unavailable.
- ROCm/HIP wheels remain unavailable.
- Metal wheels remain unavailable.
- Combined accelerator wheels remain unavailable.
- Windows wheels remain unavailable.
- PyPI or another package-index publication is not claimed.

## 0.1.0rc1

Status: published as GitHub prerelease `v0.1.0rc1` with source distribution,
macOS arm64 CPU wheel, and external checksum manifest artifacts.

### Added

- CPU package scaffold with C++20/nanobind extension and `fastpauli.PauliSum`.
- Dense-label and sparse-list construction/export with documented endianness.
- Optional Qiskit and OpenFermion adapters.
- Scalar simplify, arithmetic, multiplication, commutation, grouping, and
  expectation kernels.
- Runtime CPU backend metadata with portable scalar fallback and optional
  oneTBB/SIMD dispatch where compiled and runtime-available.
- CUDA source-build backend with explicit transfers, simplify, expectation,
  commutation, compact consumers, DLPack export for device commutation matrices,
  and matmul kernels.
- ROCm/HIP source-build backend with MI300X-evidenced transfers, commutation,
  compact consumers, simplify, expectation, and matmul kernels.
- Backend-neutral accelerator selectors for CPU-only, CUDA-target, and
  HIP-target builds.
- Apple Metal source-build backend with transfer, pairwise commutation,
  retained device-matrix, compact-consumer evidence, generic two-dimensional
  commutation dispatch, retained one-word specialization, and benchmarked
  baseline rows for common packed inputs including a benchmark-only two-word
  specialization candidate, offline `.metallib` loading, private-blit
  host-output staging, and GPU compact-reduction evidence.
- Release-candidate foundation artifacts: release evidence ledger and CPU wheel
  build/install/import validation.

### Support Boundaries

- CPU wheels are the first release-candidate artifact target.
- CUDA support remains source-build-only unless a later CUDA packaging plan adds
  wheel evidence.
- ROCm/HIP support remains source-build-only unless a later ROCm packaging plan
  adds wheel evidence.
- CUDA wheels remain unavailable.
- ROCm/HIP wheels remain unavailable.
- Combined accelerator wheels remain unavailable.
- Apple Metal support remains source-build-only unless a later Apple
  accelerator packaging plan adds macOS arm64 wheel evidence.

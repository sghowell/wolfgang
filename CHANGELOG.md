# Wolfgang Changelog

All user-visible Wolfgang changes are recorded here before each release.

## Unreleased

Next version: TBD

- CUDA: pool `DevicePauliSum` and scratch allocations in a Wolfgang-private
  stream-ordered memory pool and skip full device discovery on transfers. On
  H100, `to_device` of a small operator is 55x faster, `commutes_with_device`
  14x, and simplify of 50,000-1,000,000 terms 1.6-2.4x (device-resident).
- Restore the CUDA source build, broken since the shared workspace helper
  refactor, and pin the fully qualified aliases in the header contract test.
- CUDA simplify for operators wider than 128 qubits now reduces duplicates in
  parallel: 91-273x faster than the previous one-thread kernel on H100 at
  65,536 terms. Single-term simplify skips sort and reduction; one- and
  two-word paths use stable sorts with no measured cost.
- CUDA simplify no longer drops every term when a finite coefficient's
  magnitude overflows `DBL_MAX`; it now applies the CPU drop rule.
- Add `apply_statevector`, `variance_statevector` (of the normalized state),
  batched 2-D input for `expectation_statevector`, and `to_matrix` (dense, or
  SciPy CSR with `sparse=True` and the new `scipy` extra) with explicit size
  guards. Behavior change: a 2-D `psi` used to raise `ValueError`; it is now a
  batch, and the stubs return `complex | ndarray`.
- Add `estimate_counts(counts, basis)` for mean and standard error in any local
  measurement basis, and `wolfgang_quantum.measurement` with `qwc_basis`,
  `diagonalizing_clifford`, variance-weighted `allocate_shots` and
  `to_qiskit_circuit`.
- Add `PauliSum.from_arrays` / `to_arrays` for packed `uint64` mask and
  `complex128` coefficient exchange; `to_arrays` returns read-only zero-copy
  views. `from_labels` converts NumPy coefficient arrays in bulk.
- Add subtraction, exact negation, `len`, indexing and slicing, structural
  `==` (unhashable), `equiv`, `adjoint`, and fused `commutator` /
  `anticommutator` that skip term pairs which cancel (7-9x faster than
  composing two products in `docs/benchmarks/reports/host_api_2026-11-03.md`).
- Behavior change: `PauliSum == other` now compares structure (same terms,
  order and coefficients) instead of object identity, `PauliSum` is no longer
  hashable, and `bool(op)` follows `len(op)`, so an empty operator is falsy.
  Migration: use `op is other` for identity, `op.equiv(other)` for mathematical
  equality, `id(op)` as a dictionary key, and `op.num_terms > 0` where
  truthiness meant "not None".
- Exclude local worktrees and release staging from source distributions.
- Separate the native CMake library from Python, add standalone sanitizer
  contracts, bulk coefficient export and scoped CPU GIL release.
- Record loaded-extension/source fingerprints in benchmarks and keep requested
  revision labels separate from verified git state.
- Share core, quality, docs and all validation profiles with CI; add Python 3.13
  core coverage and reject unsupported forced streaming-grouping selectors.
- Preserve arbitrarily small nonzero coefficients when exporting to OpenFermion;
  duplicate export terms are combined before exact-zero removal.
- Harden CPU simplify at finite complex128 limits and normalize count weights
  without overflowing their total. Document the input-relative tolerance rule.
- Bound Metal temporary-object lifetimes on Python and worker threads.
- Make the Wave 1D promotion gate reject incomplete or invalid evidence and
  preserve CPU-only skips.
- Repair shipped-build GPU device discovery and expose immutable per-operation
  execution locations. Correct constructor/count typing and installation docs.
- Validate the agent source map and require a nonempty public-header scan.

## 0.2.3

Status: published as GitHub release `v0.2.3` with one source distribution, six
CPU wheels, and no TestPyPI or PyPI publication.

### Added

- Wolfgang `0.2.3` successor release evidence in `docs/release/0.2.3.md`, documenting the published GitHub-only patch path from the repaired `wolfgang_quantum.capabilities()` lineage while preserving the immutable `v0.2.2` tag as historical provenance.
- Active release/support surfaces now point at the published `v0.2.3` GitHub-only successor ledger while preserving the quarantined `v0.2.0` and `v0.2.1` tags/draft releases, the immutable `v0.2.2` checkpoint, and the published `0.1.0` evidence unchanged.
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

- CPU package scaffold with C++20/nanobind extension and `wolfgang_quantum.PauliSum`.
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

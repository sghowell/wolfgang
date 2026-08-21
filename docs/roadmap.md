# Wolfgang Roadmap

This roadmap tracks implementation order and the decisions that must stay stable across phases.

## Source Documents

```text
Agent guide:        AGENTS.md
Changelog:          CHANGELOG.md
Implementation plan: docs/plans/cpp_cuda_implementation_plan.md
Release candidate foundation plan: docs/plans/release_candidate_foundation_plan.md
Release candidate next checkpoint plan: docs/plans/release_candidate_next_checkpoint_plan.md
Release 0.1.0 wheelhouse foundation plan: docs/plans/release_0_1_0_wheelhouse_foundation_plan.md
Apple Metal/MPS bring-up plan: docs/plans/apple_metal_mps_bringup_plan.md
Apple Metal optimization Campaign 1 plan: docs/plans/apple_metal_optimization_campaign1_plan.md
Apple Metal optimization Campaign 2 plan: docs/plans/apple_metal_optimization_campaign2_plan.md
Apple Metal optimization Campaign 3 plan: docs/plans/apple_metal_optimization_campaign3_plan.md
Apple Metal optimization Campaign 4 plan: docs/plans/apple_metal_optimization_campaign4_plan.md
Apple Metal optimization Campaign 5 plan: docs/plans/apple_metal_optimization_campaign5_plan.md
Apple Metal optimization Campaign 6 plan: docs/plans/apple_metal_optimization_campaign6_plan.md
Apple Metal optimization Campaign 7 plan: docs/plans/apple_metal_optimization_campaign7_plan.md
Apple Metal optimization Campaign 8 plan: docs/plans/apple_metal_optimization_campaign8_plan.md
CUDA deep optimization plan: docs/plans/cuda_deep_optimization_plan.md
Latest CUDA campaign plan: docs/plans/cuda_residual_risk_campaign11_plan.md
Latest CUDA campaign report: docs/benchmarks/reports/cuda_residual_risk_campaign11_2026-04-29.md
Latest ROCm/HIP wave plan: docs/plans/rocm_next_waves_plan.md
Latest ROCm/HIP campaign plan: docs/plans/mi300x_rocm_optimization_campaign8_plan.md
First ROCm/HIP bring-up plan: docs/plans/mi300x_rocm_bringup_plan.md
Latest ROCm/HIP report: docs/benchmarks/reports/rocm_campaign8_architecture_readiness_2026-05-01.md
Backend-neutral accelerator plan: docs/plans/backend_neutral_accelerator_campaign9_plan.md
Cross-backend kernel performance campaign: docs/plans/wolfgang-kernel-performance-campaign.md
Semantic contracts: docs/architecture/semantic_contracts.md
CUDA architecture:  docs/architecture/cuda_backend.md
ROCm/HIP architecture: docs/architecture/rocm_backend.md
Apple Metal architecture: docs/architecture/apple_accelerator.md
Latest Apple Metal report: docs/benchmarks/reports/apple_metal_wave1d_2026-08-21.md
Hardware targets:   docs/architecture/hardware_targets_and_testing.md
Testing and CI:     docs/architecture/testing_and_ci.md
Adapter contracts: docs/architecture/adapter_contracts.md
Benchmark protocol: docs/benchmarks/protocol.md
Quality gates:     docs/quality/phase_quality_gates.md
Agent harness:      docs/quality/agent_harness.md
Code review:       docs/quality/code_review.md
Code standards:     docs/quality/code_standards.md
Documentation standards: docs/quality/documentation_standards.md
API stability:      docs/architecture/api_stability.md
Security and supply chain: docs/quality/security_and_supply_chain.md
Release and packaging: docs/quality/release_and_packaging.md
Release evidence index: docs/release/README.md
0.2.3 successor release evidence ledger: docs/release/0.2.3.md
0.2.2 historical provenance ledger: docs/release/0.2.2.md
0.1.0 release evidence ledger: docs/release/0.1.0.md
0.1.0 wheelhouse dry-run evidence: docs/release/0.1.0-wheelhouse-dry-run.md
0.1.0-rc2 release evidence ledger: docs/release/0.1.0-rc2.md
0.1.0-rc1 release evidence ledger: docs/release/0.1.0-rc1.md
Cloud hardware qualification harness: docs/release/cloud_hardware_qualification_harness.md
Release support matrix: docs/release/support_matrix.md
Contributing:       CONTRIBUTING.md
```

## Current Status

Wolfgang has completed the Phase 11 CUDA kernel slice:
C++20/scikit-build-core/nanobind
packaging, packed `x`/`z`/`coeffs` storage, dense-label and sparse-list
construction/export, explicit empty operators, optional Qiskit `SparsePauliOp`
conversion, optional OpenFermion `QubitOperator` conversion, scalar sort-based
simplify with canonical packed-word ordering, addition without implicit simplify,
scalar multiplication, phase-correct Pauli-sum
multiplication, native C++ sources split by representation, parse/export, simplify,
arithmetic, multiplication, commutation, and grouping responsibility, scalar/vector/
guarded-matrix commutation APIs, deterministic greedy QWC and full commuting grouping,
pytest semantic fixtures, simplify, multiply, grouping, and OpenFermion conversion
benchmark smokes, scalar CPU statevector and diagonal Z-count expectation kernels,
expectation benchmark smoke, runtime CPU backend dispatch metadata, forced scalar
execution through `WOLFGANG_CPU_BACKEND`, clear failures for uncompiled optimized
selectors, optional NEON/AVX2/AVX-512 commutation kernels, optional oneTBB
commutation kernels, operation-level auto dispatch for measured commutation and
full-grouping graph hot spots, CPU-dispatch benchmark smoke, repo-local validation,
core CPU CI, Qiskit adapter CI jobs, and OpenFermion adapter CI jobs. The CUDA
backend adds optional `WOLFGANG_ENABLE_CUDA=ON`
source builds, configurable `WOLFGANG_CUDA_ARCHITECTURES`, explicit
`PauliSum.to_device()` and `DevicePauliSum.to_host()` transfers, CUDA runtime
availability metadata, CPU-only stubs with build-time absence errors, CUDA
`DevicePauliSum.simplify()`, statevector expectation, dense pairwise
commutation, and matrix-product kernels, CUDA-array-interface statevector
interop, H100 equivalence validation, compute-sanitizer coverage, and CUDA
benchmark smoke/default evidence.
The latest CPU/CUDA performance hardening added H100 Nsight evidence, an
opt-in extreme CUDA scaling profile, direct NumPy CUDA commutation fills, large
host-output registration for dense CUDA commutation, and faster AVX2/AVX-512
commutation stores. Remaining high-leverage performance work is now expected to
involve device-resident result APIs, reusable CUDA/Thrust temporary storage, or
custom duplicate-reduction pipelines rather than small kernel instruction edits.
The CUDA deep optimization pass retained a host-statevector byte-copy
optimization for CUDA expectation, rejected an unstable one/two-word commutation
kernel specialization, added privileged Nsight Compute coverage for custom and
CCCL/Thrust-heavy paths, added a semantically matched cuStateVec expectation
competitor case, and checked in the comprehensive H100 report at
`docs/benchmarks/reports/cuda_deep_optimization_h100_2026-04-28.md`.
Remaining high-leverage CUDA work now requires new public lifetime boundaries
such as reusable workspaces or stream-aware/device-statevector APIs rather than
small instruction-level kernel edits.
The H100-first Campaign 3 plan in
`docs/plans/h100_deep_optimization_campaign3_plan.md` is now complete. It
retained the packed-key CUDA simplify path, quantified allocation and output
materialization boundaries, refreshed competitor baselines, and published the
checked report at
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign3_2026-04-28.md`.
The H100-first Campaign 4 plan in
`docs/plans/h100_deep_optimization_campaign4_plan.md` is now complete. It
implemented a private CUDA workspace and benchmark-only CUB/CCCL
scratch-boundary probes, rejected the narrow CUB radix-sort duplicate-reduction
prototype for production, deferred device-output commutation to API review,
refreshed CPU/CUDA/external H100 comparisons, and published the checked report
at `docs/benchmarks/reports/cuda_deep_optimization_h100_campaign4_2026-04-29.md`.
The cross-backend kernel performance campaign in
`docs/plans/wolfgang-kernel-performance-campaign.md` is the current ranking
handoff for the next user-visible CPU, CUDA, ROCm/HIP, and Apple Metal
optimization wave at baseline commit `d14b4960a5197485e41d81a5dc426af5fce7cbae`.
The H100-first Campaign 5 plan in
`docs/plans/h100_deep_optimization_campaign5_plan.md` is now complete. It
retained the experimental dense `DeviceCommutationMatrix` device-resident
commutation result boundary, refreshed broad CPU/CUDA/external H100
comparisons, and published the checked report at
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign5_2026-04-29.md`.
The H100-first Campaign 6 plan in
`docs/plans/h100_deep_optimization_campaign6_plan.md` is now complete. It
retained compact `DeviceCommutationMatrix.count_commuting(axis=None|0|1)`
consumers, added CuPy CUDA-array-interface consumer baselines, refreshed the
broad README CPU/CUDA/external landscape, and published the checked report at
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign6_2026-04-29.md`.
Public stream, async, and bit-packed CUDA APIs remain deferred until a separate
API plan accepts their lifetime, layout, and consumer contracts.
The H100-first Campaign 7 plan in
`docs/plans/h100_deep_optimization_campaign7_plan.md` is now complete. It
retained fused graph and grouping-oriented consumers as private benchmark-only
helpers, rejected count-reduction specialization for this slice as not
dominant, kept public async/stream and bit-packed output APIs deferred, recorded
the non-H100 NVIDIA portability blocker, refreshed the broad README
CPU/CUDA/external landscape, and published the checked report at
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign7_2026-04-29.md`.
The H100-first Campaign 8 plan in
`docs/plans/h100_deep_optimization_campaign8_plan.md` is now complete on H100.
It retained private benchmark-only compact device-resident graph and grouping
consumers that avoid full CSR edge-list host export for high-scale H100 rows,
kept public fused grouping, DLPack, and stream/CUDA Graph surfaces deferred,
rejected CSR scatter tuning because the retained consumer no longer needs full
CSR scatter, recorded the non-H100 NVIDIA portability blocker, refreshed the
broad README CPU/CUDA/external landscape, and published the checked report at
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign8_2026-04-29.md`.
The H100-first Campaign 9 plan in
`docs/plans/h100_deep_optimization_campaign9_plan.md` is now complete on H100
and published at
`docs/benchmarks/reports/cuda_deferred_headroom_campaign9_2026-04-29.md`.
It closes every Campaign 8 deferred or blocked surface with a non-deferred
status: named non-H100 NVIDIA portability is `blocked_external` after a
concrete access check, privileged Nsight Compute counter evidence passed, the
true public fused grouping API is `rejected_with_evidence`, compact
`DeviceCommutationMatrix.conflict_degrees(axis=None|0|1)` is implemented,
read-only `DeviceCommutationMatrix` DLPack export is implemented and validated
with CuPy, public stream/CUDA Graph surfaces are `rejected_with_evidence`, and
CSR scatter reopening is `rejected_with_evidence` because retained compact
consumers avoid full CSR edge-list materialization.
The cross-architecture Campaign 10 plan in
`docs/plans/cuda_cross_architecture_campaign10_plan.md` is now complete and
published at
`docs/benchmarks/reports/cuda_cross_architecture_campaign10_2026-04-29.md`.
It replaces the Campaign 9 non-H100 blocker with A100 `sm_80` and RTX PRO 6000
Blackwell `sm_120` source-build evidence, adds PyTorch CUDA DLPack coverage,
keeps a true public grouping API rejected while retaining compact conflict
degree summaries, rejects stream/CUDA Graph work because launch overhead is not
dominant, and keeps CSR scatter closed because no retained consumer requires
full CSR edge lists.
The residual-risk Campaign 11 slice in
`docs/plans/cuda_residual_risk_campaign11_plan.md` is now complete and
published at
`docs/benchmarks/reports/cuda_residual_risk_campaign11_2026-04-29.md`. It
closes the two in-scope Campaign 10 follow-ups with terminal statuses: RTX PRO
6000 Blackwell Nsight Compute counter capture passed, A100 counter capture is
blocked by host performance-counter permissions, and the nanobind
reference-leak diagnostics are classified as process-teardown diagnostics after
fresh sanitizer logs and clean lifecycle subprocesses. A10, L4, RTX 6000 Ada,
and other additional NVIDIA lanes remain explicitly out of scope for this
slice.
The first ROCm/HIP MI300X bring-up plan in
`docs/plans/mi300x_rocm_bringup_plan.md` is now complete and published at
`docs/benchmarks/reports/rocm_mi300x_bringup_2026-04-29.md`. It adds
source-build-only `WOLFGANG_ENABLE_HIP=ON` evidence for MI300X `gfx942`,
keeps CUDA and HIP mutually exclusive in this architecture slice, exposes HIP
runtime metadata and `DevicePauliSum.backend == "hip"`, validates non-empty
and empty HIP transfers, retains a HIP pairwise commutation kernel, captures
CPU/HIP benchmark comparisons, and records rocprof trace and counter evidence.
ROCm wheels, multi-GPU MI300X support, HIP streams, HIP DLPack, HIP workspace
APIs, and simultaneous CUDA+HIP builds remain out of scope until separate
architecture decisions accept those public boundaries.
The MI300X ROCm optimization Campaign 2 plan in
`docs/plans/mi300x_rocm_optimization_campaign2_plan.md` is now complete and
published at
`docs/benchmarks/reports/rocm_mi300x_campaign2_2026-04-30.md`. It adds the
HIP-backed `DeviceCommutationMatrix` dense device-output lifetime boundary,
reused-output execution, dense `to_host()` materialization, compact
`count_commuting(axis=None|0|1)` and `conflict_degrees(axis=None|0|1)`
consumers, MI300X CPU/HIP benchmark comparisons, and rocprof trace/counter
evidence. At the Campaign 2 boundary, HIP DLPack, public streams, public
workspaces, HIP simplify, HIP expectation, HIP matmul, ROCm wheels, multi-GPU
MI300X support, and simultaneous CUDA+HIP source builds remained out of scope
until separate plans accepted those public boundaries.
The MI300X ROCm optimization Campaign 3 plan in
`docs/plans/mi300x_rocm_optimization_campaign3_plan.md` is now complete and
published at
`docs/benchmarks/reports/rocm_mi300x_campaign3_2026-04-30.md`. It adds HIP
`DevicePauliSum.simplify()` with device-resident output, retains rocThrust as
the production duplicate-reduction path, validates CPU/HIP simplify parity on
edge, tolerance, randomized, duplicate-heavy, duplicate-light, one-word,
two-word, and generic multi-word cases, refreshes the broad README
CPU/CUDA/ROCm/external landscape, and gives every Campaign 2 remaining-headroom
item a terminal status. At the Campaign 3 boundary, HIP DLPack, public streams,
public workspaces, HIP expectation, HIP matmul, ROCm wheels, multi-GPU MI300X
support, portability claims beyond MI300X `gfx942`, and simultaneous CUDA+HIP
source builds remained out of scope until separate plans accepted those public
boundaries.
The MI300X ROCm optimization Campaign 4 plan in
`docs/plans/mi300x_rocm_optimization_campaign4_plan.md` is now complete and
published at
`docs/benchmarks/reports/rocm_mi300x_campaign4_2026-04-30.md`. It hardens HIP
`DevicePauliSum.simplify()` without adding public HIP APIs. The retained
private implementation replaces the Campaign 3 single-thread generic
multi-word reducer with a sorted-index parallel `reduce_by_key` path; the
130-qubit/4096-term generic A/B row improved from 0.005566 s to 0.000340 s
resident on MI300X. Custom packed-key probes for packed32, key1, and key2
rows are recorded as unavailable because no distinct lower-level rocPRIM or
hipCUB replacement was retained or timed. Private workspace/scratch probes are
recorded as unavailable for the current rocThrust boundary. HIP DLPack, public streams,
public workspaces, ROCm wheels, multi-GPU MI300X, broader AMD portability
claims, and simultaneous CUDA+HIP source builds remain separate follow-on
decisions; HIP expectation and HIP matmul were later addressed by Campaign 6.
The MI300X ROCm optimization Campaign 5 slice in
`docs/plans/mi300x_rocm_optimization_campaign5_plan.md` is complete and
published at
`docs/benchmarks/reports/rocm_mi300x_campaign5_2026-04-30.md`. It rejects
public HIP DLPack because PyTorch ROCm consumed the candidate versioned
`kDLROCM` capsule in a temporary candidate probe but accepted mutation of the
read-only view. HIP CUDA Array
Interface, public streams, public graph execution, and public workspaces are
rejected with evidence. Portability, ROCm wheels, multi-GPU ROCm, and
simultaneous CUDA+HIP source builds remain separate campaigns; HIP expectation
and HIP matmul were later addressed by Campaign 6.
The MI300X ROCm optimization Campaign 6 slice in
`docs/plans/mi300x_rocm_optimization_campaign6_plan.md` is complete and
published at
`docs/benchmarks/reports/rocm_mi300x_campaign6_2026-04-30.md`. It retains HIP
`DevicePauliSum.expectation_statevector()` for host NumPy `complex64` and
`complex128` statevectors and HIP `DevicePauliSum.matmul()` parity on MI300X
without adding new public API shape or reopening rejected HIP DLPack, HIP CUDA
Array Interface, public stream, graph, public workspace, ROCm wheel, multi-GPU
ROCm, broader portability, or simultaneous CUDA+HIP source-build surfaces.
The MI300X ROCm Campaign 7 slice in
`docs/plans/mi300x_rocm_optimization_campaign7_plan.md` is complete and
published at
`docs/benchmarks/reports/rocm_mi300x_campaign7_2026-04-30.md`. It moves ROCm to
Wave 5 release-support evidence: repeatable MI300X source-build validation,
CPU-only control validation, CUDA+HIP configure-time rejection, ROCm
release-runbook support, ROCm source-build-only packaging policy,
duplicate-pressure smoke rows for retained simplify/matmul operations,
portability gating, rocprof trace/stats evidence, and terminal statuses for
external HIP statevector interop, HIP DLPack, HIP CUDA Array Interface, public
streams, public graphs, public workspaces, multi-GPU ROCm, simultaneous
CUDA+HIP, and backend-neutral accelerator design. ROCm wheels remain
unavailable, broader AMD portability remains blocked until a non-MI300X AMD GPU
lane is available, and simultaneous CUDA+HIP remains unavailable under the
target-specific accelerator build policy unless a future mixed-runtime plan
reopens it.
The ROCm Campaign 8 plan is
`docs/plans/mi300x_rocm_optimization_campaign8_plan.md`, now published at
`docs/benchmarks/reports/rocm_campaign8_architecture_readiness_2026-05-01.md`.
It is a Wave 6 architecture-readiness campaign that converts Campaign 7 residual items into
explicit backend-neutral object-model, portability, packaging, profiler,
interop, multi-GPU, and targeted-performance gates before any new ROCm kernels,
public APIs, wheels, broad AMD support claims, or simultaneous CUDA+HIP builds
are attempted. It changes docs, contracts, tests, and evidence only; no HIP
kernel, public Python API, ROCm wheel, or multi-GPU runtime behavior changed.
The backend-neutral accelerator Campaign 9 plan is
`docs/plans/backend_neutral_accelerator_campaign9_plan.md`, with closeout
evidence in
`docs/benchmarks/reports/backend_neutral_accelerator_campaign9_2026-05-01.md`.
It completes the target-specific backend-neutral accelerator API trigger:
shared status schema, explicit device-construction backend selectors,
`DeviceCommutationMatrix.backend`, disjoint accelerator source-set declarations,
simulated backend/device validation tests, benchmark-boundary metadata, H100
CUDA-target validation, MI300X HIP-target validation, and the documented
CUDA+HIP configure-time rejection lane. Simultaneous CUDA+HIP source builds are
not a Campaign 9 completion gate; they require a future accepted mixed-runtime
design if the product ever needs one extension that links both accelerator
runtimes.
The release-candidate foundation plan in
`docs/plans/release_candidate_foundation_plan.md` completed the first packaging
readiness checkpoint. It added `CHANGELOG.md`, release evidence ledgers under
`docs/release/`, `scripts/validate_release_artifacts.py`, and a Linux/macOS CPU
wheel smoke CI lane. Wolfgang `0.1.0rc1` is now published as a GitHub
prerelease at `https://github.com/sghowell/Wolfgang/releases/tag/v0.1.0rc1`
with source distribution, macOS arm64 CPU wheel, and external checksum manifest
artifacts. This checkpoint does not change runtime behavior or add CUDA,
ROCm/HIP, combined accelerator, Windows, Apple GPU wheel, or package-index
publication claims.
The `0.1.0rc2` release-candidate checkpoint is tracked in
`docs/release/0.1.0-rc2.md`,
`docs/plans/release_candidate_next_checkpoint_plan.md`, and
`docs/release/support_matrix.md`. It publishes the next CPU source distribution
and macOS arm64 CPU wheel lane while preserving source-build-only CUDA,
ROCm/HIP, and Apple Metal accelerator support, unavailable combined
accelerator and Windows wheels, and unavailable PyPI publication
claims unless a later release finalization slice supplies evidence.
The final `0.1.0` CPU wheelhouse and PyPI readiness lane is tracked in
`docs/release/0.1.0.md` and
`docs/plans/release_0_1_0_wheelhouse_foundation_plan.md`. It adds CPU-only
cibuildwheel selectors, a manual release-wheelhouse workflow, installed-wheel
smokes, checksum manifest generation, exact CPU wheelhouse completeness checks,
and explicit tag-ref-gated TestPyPI/PyPI trusted publishing gates while keeping
accelerator and Windows wheel claims unavailable. The `v0.1.0` tag-ref workflow
has produced the complete CPU wheelhouse and checksum evidence. The corrected
tag-ref run passed TestPyPI upload and clean install smoke. PyPI publication
remains unavailable until PyPI trusted publishing is configured for the
observed `pypi` environment claims and the PyPI publish job succeeds.
The first hosted dry run of that workflow is recorded in
`docs/release/0.1.0-wheelhouse-dry-run.md`. It produced one source
distribution, six CPU wheels for manylinux x86_64 and macOS arm64 across
CPython 3.10, 3.11, and 3.12, and a checksum manifest with
`publish-target=none`; PyPI publication remains unavailable until the PyPI
trusted-publisher configuration accepts the exact `v0.1.0` tag-ref workflow
claims.
The published `0.2.3` GitHub-only successor slice is tracked in
`docs/release/0.2.3.md`, `docs/release/support_matrix.md`, and
`docs/release/README.md`. It bumps the active source version from the corrected
capabilities fix at `bd550f4b91d575277508ca9880ec3695940c8c68`, preserves the
immutable `v0.2.2` tag as historical provenance instead of rewriting it, keeps
release-quality CI in the checked path, preserves the quarantined `v0.2.0` and
`v0.2.1` draft releases, records the published GitHub release and exact-tag
wheelhouse evidence, and explicitly forbids TestPyPI/PyPI publication for this
successor slice.
The Apple Silicon accelerator implementation lane is active in
`docs/architecture/apple_accelerator.md`, with implementation handoff in
`docs/plans/apple_metal_mps_bringup_plan.md`. The source tree now includes the
`backend="metal"` selector, `WOLFGANG_ENABLE_METAL=ON` target-specific source
build flag, Metal status metadata, transfer code, and pairwise commutation
kernel. Metal target builds remain separate from CUDA and HIP target builds,
and MPS/MPSGraph remain optional implementation adjuncts or baselines rather
than Wolfgang backend identities. Local Apple Silicon runtime validation now
passes in an elevated Codex command context on Apple M4 Pro, including transfer
round trips, pairwise commutation, retained device-matrix materialization, and
compact count/conflict consumer equivalence. The non-elevated Codex sandbox
still reports no visible Metal device, so the report records the sandboxed and
elevated runtime distinction. Full Xcode and the Metal Toolchain component are
now installed, validation selects that toolchain explicitly, and the report
includes a short Metal System Trace TOC for the Wolfgang Metal benchmark.
Apple Metal optimization Campaign 1 extends this with a scaling benchmark
profile, explicit retained device-output and `to_host()` timing boundaries,
and README broad-landscape rows that keep Apple Metal visible next to CPU,
CUDA, ROCm/HIP, and external baseline evidence. Deeper Apple GPU counter and
shader-timeline work remains a later profiling task, not a bring-up blocker.
Apple Metal optimization Campaign 2 switches retained commutation launch
metadata to a two-dimensional dispatch grid, retains one-word specialization,
keeps two-word specialization as a benchmark-only candidate, records generic-2D
and legacy flat-generic A/B baseline rows, and refreshes the README broad
landscape with the latest Apple M4 Pro source-build rows.
Apple Metal optimization Campaign 3 tests offline `.metallib` loading, private
output storage with blit staging, GPU compact-consumer reductions, two-word
specialization policy, profiler availability, and exact MPSGraph/PyTorch MPS
baseline status without adding public Metal APIs or generic Apple GPU support
claims. It keeps those paths benchmark-only and retains generic 2D, shared
host-output storage, runtime source pipelines, and CPU compact scans as the
default policies.
Apple Metal optimization Campaign 4 pushes the Campaign 3 remaining headroom
with larger two-word and compact-consumer benchmark cases, a benchmark-only
parallel block-reduction compact total count, private device-boundary evidence,
and explicit deferral of PyPI publication, Windows support, and older macOS
compatibility. It preserves the source-build-only public boundary and keeps all
new Metal paths evidence tools until they beat retained defaults across more
workloads and hardware.
Apple Metal optimization Campaign 5 adds correct source-build-only
`DevicePauliSum.simplify(atol, rtol)` behavior for Metal builds as the first
operation bring-up after commutation and compact consumers. The retained path is
explicitly labeled `metal_simplify_transfer_reference` with the
`device_to_host_cpu_simplify_host_to_device` timing boundary; it is a
correctness bridge, not a device-resident GPU duplicate-reduction speedup.
Campaign 5 keeps Metal statevector expectation, Metal matmul, Metal wheels,
Windows, older macOS support, and PyPI publication outside the slice.
Apple Metal Campaign 6 device-resident simplify groundwork keeps the public
Metal simplify path unchanged while adding a private `MetalWorkspace` scratch
model, `WorkspaceTimingMode`, Campaign 6 simplify benchmark cases, and
`metal_simplify_workspace_probe` status rows. The device-resident simplify
candidate remains blocked until checked Metal sort/prefix/reduce primitives
exist, so Campaign 6 records workspace and benchmark evidence rather than a
retained GPU duplicate-reduction speedup.
Apple Metal Campaign 7 checked device-resident simplify primitive stack adds a
private benchmark-only one-word path with Metal bitonic key sort, prefix-sum,
reduce-by-key, and survivor compaction. `metal_simplify_device_candidate` rows
may use the `device_resident` boundary only after materialized output matches
CPU simplify. The candidate is limited to signed fixed32 dyadic coefficients
because the local Apple Metal toolchain rejects `double` arithmetic in kernels;
public Metal `DevicePauliSum.simplify()` remains the transfer-reference bridge.
Apple Metal Campaign 8 simplify performance-relevance evidence keeps that
public path unchanged while adding private timing decomposition,
pipeline-cache boundary, dispatch-count, and `performance_decision` metadata
to decide whether the Campaign 7 device candidate is performance-relevant or
must stay benchmark-only experimental.
Evidence is recorded in
`docs/benchmarks/reports/apple_metal_bringup_2026-05-01.md` and
`docs/benchmarks/reports/apple_metal_optimization_campaign1_2026-05-05.md` and
`docs/benchmarks/reports/apple_metal_optimization_campaign2_2026-05-05.md` and
`docs/benchmarks/reports/apple_metal_optimization_campaign3_2026-05-06.md` and
`docs/benchmarks/reports/apple_metal_optimization_campaign4_2026-05-06.md` and
`docs/benchmarks/reports/apple_metal_optimization_campaign5_2026-05-06.md` and
`docs/benchmarks/reports/apple_metal_optimization_campaign6_2026-05-07.md` and
`docs/benchmarks/reports/apple_metal_optimization_campaign7_2026-05-07.md` and
`docs/benchmarks/reports/apple_metal_optimization_campaign8_2026-05-07.md` and
`docs/benchmarks/reports/apple_metal_wave1d_2026-08-21.md`.

## Release Candidate Foundation Checkpoint

Goal: make a first CPU release-candidate lane mechanically checkable and publish
the first GitHub prerelease artifacts without overclaiming wheel coverage.

Acceptance:

```text
CHANGELOG.md records user-visible status and support boundaries
docs/release/README.md explains required release evidence
docs/release/0.1.0-rc1.md records the release-candidate evidence ledger
scripts/validate_release_artifacts.py builds a source distribution and CPU wheel
scripts/validate_release_artifacts.py installs the produced wheel into a clean virtual environment
wheel smoke verifies cpu_only, non-native, scalar-fallback-safe metadata
CI runs the wheel smoke on Linux x86_64 and macOS arm64
release artifact files and directories are ignored by .gitignore
v0.1.0rc1 GitHub prerelease publishes source, macOS arm64 CPU wheel, and checksum manifest
README, AGENTS, roadmap, validation, testing, and release-standard docs route to this checkpoint
CUDA and ROCm/HIP remain source-build-only support lanes unless later packaging plans add wheel evidence
```

## Phase 0: Planning And Architecture Lock

Goal: make implementation-start decisions explicit before writing package code.

Acceptance:

```text
semantic contracts documented
CUDA backend architecture documented
CPU and CUDA hardware target policy documented
testing and CI architecture documented
adapter contracts documented
benchmark protocol documented
phase quality gates documented
agent guide documented
agent harness documented
agent-driven review policy documented
code standards documented
documentation standards documented
API stability documented
security and supply-chain standards documented
release and packaging standards documented
contribution and review surface documented
implementation plan references the source-of-truth docs
first PR scope remains CPU-only but CUDA-compatible
```

## Phase 1: CPU Package Scaffold

Goal: create an installable Python package with a minimal C++/nanobind extension.

Acceptance:

```text
pip install -e . works
import wolfgang_quantum works
wolfgang_quantum.PauliSum exists
pytest runs
CPU-only build does not require CUDA headers or toolkit
portable CPU scalar baseline and build-option defaults follow docs/architecture/hardware_targets_and_testing.md
repo-local validation command exists
first mechanical harness checks exist
initial review-policy checks exist
initial code/documentation standard checks exist
initial API/security/release standard checks exist
Phase 1 gates in docs/quality/phase_quality_gates.md are satisfied
```

## Phase 2: Packed Representation And I/O

Goal: implement the packed Pauli representation and lock down endianness.

Acceptance:

```text
from_labels and to_labels work
from_sparse_list and to_sparse_list work
PauliSum.empty(num_qubits) works
construction order is preserved before simplify or sort
endianness tests pass
invalid input raises documented Python exceptions
Phase 2 gates in docs/quality/phase_quality_gates.md are satisfied
```

## Phase 3: Qiskit Adapter

Goal: round-trip Qiskit `SparsePauliOp` through Wolfgang without dense matrix materialization.

Acceptance:

```text
from_qiskit and to_qiskit work
Qiskit phases are folded into Wolfgang coefficients
small random operators round-trip
n <= 8 matrix comparisons pass
optional dependency tests skip cleanly when Qiskit is absent
Qiskit behavior follows docs/architecture/adapter_contracts.md
Phase 3 gates in docs/quality/phase_quality_gates.md are satisfied
```

## Phase 4: Simplify And Canonical Ordering

Goal: implement sort-based duplicate reduction and define benchmark baselines.

Acceptance:

```text
simplify combines duplicates
simplify applies documented atol and rtol semantics
simplify returns canonical order
simplify is idempotent
bench_simplify.py exists
benchmark behavior follows docs/benchmarks/protocol.md
Phase 4 gates in docs/quality/phase_quality_gates.md are satisfied
```

## Phase 5: Arithmetic And Multiplication

Goal: implement addition, scalar multiplication, and Pauli-sum multiplication with phase correctness.

Acceptance:

```text
addition concatenates without implicit simplify
scalar multiplication preserves terms
single-term multiplication examples match semantic contracts
PauliSum @ PauliSum enforces max_intermediate_terms
small dense matrix comparisons pass
Phase 5 gates in docs/quality/phase_quality_gates.md are satisfied
```

## Phase 6: Commutation And Grouping

Goal: implement pairwise commutation and deterministic greedy grouping.

Acceptance:

```text
commutes_with returns scalar, vector, or guarded matrix outputs
QWC grouping returns internally QWC-compatible groups
full grouping returns internally commuting groups
large pairwise commutation requests fail before unsafe allocation
bench_grouping.py exists
Phase 6 gates in docs/quality/phase_quality_gates.md are satisfied
```

## Phase 7: OpenFermion Adapter

Goal: round-trip OpenFermion `QubitOperator` through Wolfgang.

Acceptance:

```text
from_openfermion and to_openfermion work
num_qubits inference follows documented rules
round-trip tests pass
bench_openfermion_conversion.py exists
OpenFermion behavior follows docs/architecture/adapter_contracts.md
Phase 7 gates in docs/quality/phase_quality_gates.md are satisfied
```

## Phase 8: CPU Expectation Kernels

Goal: implement CPU statevector and Z-count expectation paths.

Acceptance:

```text
statevector expectation matches dense matrix for n <= 8
Z-count expectation matches direct Python computation
bitstring endianness follows dense label convention
bench_expectation.py exists
Phase 8 gates in docs/quality/phase_quality_gates.md are satisfied
```

## Phase 9: CPU Optimization

Goal: optimize measured hot paths without changing public semantics.

Acceptance:

```text
benchmarks report scalar and optimized-path availability
oneTBB and SIMD paths are optional, compiled only when dependencies/toolchains support them, and tested when available
runtime CPU dispatch works and rejects unavailable forced selectors
portable scalar baseline remains available
Apple Silicon CPU results are reported separately for Apple-specific performance claims
CPU target metadata and forced scalar/optimized path checks follow docs/architecture/hardware_targets_and_testing.md
Phase 9 gates in docs/quality/phase_quality_gates.md are satisfied
```

## CPU Performance Hardening Checkpoint

Goal: exhaust practical scalar CPU optimization and benchmark coverage before
Phase 10 CUDA work begins.

Acceptance:

```text
hot-path profiling covers simplify, multiplication, commutation/grouping, statevector expectation, and Z-count expectation
benchmarks include smoke, default, and stress-sized datasets that preserve deterministic seeds and exact correctness checks
benchmarks cover default simplified multiplication under duplicate pressure and all-diagonal statevector expectation
competitive baselines include Qiskit and OpenFermion comparisons when those optional libraries are installed
benchmarks report before/after results on Apple Silicon and x86_64 when both machines are available
benchmarks compare against direct Python references and optional Qiskit/OpenFermion baselines where those dependencies expose comparable operations
all CPU optimizations preserve public APIs, packed representation semantics, coefficient accuracy, ordering contracts, and documented exceptions
optimization claims include command, commit, hardware, compiler, dataset, median timing, and limitations
implemented oneTBB and SIMD kernels have separate forced-backend tests, runtime dispatch, packaging rules, and architecture-specific benchmark evidence
forced optimized selectors reject scalar-only operations instead of silently executing scalar code under an optimized label
oneTBB auto-dispatch thresholds are benchmarked and reported with Apple Silicon and x86_64 evidence before CUDA work begins
checked-in CPU evidence reports summarize Apple Silicon, x86_64, oneTBB, SIMD, threshold, and competitive-baseline results when available
future SIMD extensions and CUDA paths remain reported as unavailable until implemented, tested, and benchmarked
independent review verifies that no benchmark shortcuts, skipped correctness checks, or interface regressions were introduced
```

## CPU SIMD And oneTBB Dispatch Checkpoint

Goal: enable measured CPU dispatch backends before Phase 10 CUDA without
changing scalar semantics or release-wheel safety.

Acceptance:

```text
portable scalar backend remains compiled and forceable
NEON compiles and runs on Apple Silicon when WOLFGANG_ENABLE_ARM_NEON=auto or ON
AVX2 and AVX-512 compile into separate dispatched objects on x86_64 when the compiler supports the required flags
AVX-512 runtime availability checks require AVX-512F, AVX-512BW, AVX-512VL, and AVX-512 VPOPCNTDQ
oneTBB compiles only when CMake finds oneTBB, and WOLFGANG_ENABLE_TBB=ON fails clearly when it is missing
forced tbb, avx2, avx512, and neon selectors match forced scalar for covered commutation/grouping kernels
forced optimized selectors fail clearly for scalar-only operations
auto dispatch uses oneTBB only for large pairwise commutation and uses SIMD for covered packed-word full-grouping graph construction
bench_cpu_dispatch.py compares auto, forced scalar, and every available optimized selector with correctness checks
bench_cpu_thresholds.py characterizes the pairwise oneTBB auto-dispatch threshold on available CPU targets
benchmark reports record optimized kernel coverage, oneTBB version, CPU feature probes, and effective auto backend hints
```

## Phase 10: CUDA Backend Foundation

Goal: add the required CUDA backend without disrupting CPU-only builds.

Acceptance:

```text
WOLFGANG_ENABLE_CUDA=ON builds from source
WOLFGANG_ENABLE_CUDA=OFF still builds without CUDA installed
CUDA toolkit and architecture target handling follows docs/architecture/hardware_targets_and_testing.md
PauliSum.to_device and DevicePauliSum.to_host work
device transfer tests pass
CUDA availability skip reasons are explicit
Phase 10 gates in docs/quality/phase_quality_gates.md are satisfied
```

## Phase 11: CUDA Kernels

Goal: implement CUDA kernels where benchmarks justify device execution.

Status: completed for the first CUDA kernel set. Source builds now expose
device-resident simplify, statevector expectation, pairwise commutation, and
matrix-product generation followed by simplify. CUDA benchmark reports remain
source-build evidence, not wheel or release-speedup claims.

Acceptance:

```text
CUDA expectation or CUDA simplify lands first based on benchmark evidence
CPU/GPU equivalence tests pass
benchmarks include transfer-inclusive and device-resident measurements
CUDA benchmark reports include toolkit, driver, compiled architectures, device, and compute capability metadata
CUDA pairwise commutation and multiplication follow after expectation and simplify
CUDA benchmark reporting follows docs/benchmarks/protocol.md
Phase 11 gates in docs/quality/phase_quality_gates.md are satisfied
```

## CUDA Benchmarking And Reporting Checkpoint

Goal: make post-Phase 11 CUDA optimization evidence competitive, reproducible,
and easy to inspect before deeper CUDA hillclimbing.

Status: completed for the first H100 deep-optimization checkpoint and the
campaign-2, campaign-3, campaign-4, campaign-5, campaign-6, campaign-7,
campaign-8, campaign-9, campaign-10, and campaign-11 follow-ups. The latest
source-of-truth report is
`docs/benchmarks/reports/cuda_residual_risk_campaign11_2026-04-29.md`,
with raw JSON under
`docs/benchmarks/data/cuda_residual_risk_campaign11_2026-04-29/`.
Campaign 10 closed every Campaign 9 remaining-headroom item with final
non-deferred statuses, replaced the non-H100 blocker with A100 `sm_80` and RTX
PRO 6000 Blackwell `sm_120` source-build evidence, added PyTorch CUDA DLPack
coverage, rejected public grouping, stream/CUDA Graph, and CSR scatter reopen
work with fresh evidence, and keeps the README performance landscape as a
broad CPU/CUDA/external comparison generated from checked-in evidence.
Campaign 11 closes the immediate Campaign 10 residual-risk follow-up: RTX PRO
6000 Blackwell has checked Nsight Compute counter artifacts, A100 has a
terminal host-permission blocker for counter capture, and the nanobind
diagnostics are classified with fresh sanitizer and lifecycle evidence.
The campaign-10 source-of-truth report remains at
`docs/benchmarks/reports/cuda_cross_architecture_campaign10_2026-04-29.md`,
with raw JSON under
`docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29/` and
generated plots under `docs/benchmarks/plots/`.
The campaign-9 source-of-truth report remains at
`docs/benchmarks/reports/cuda_deferred_headroom_campaign9_2026-04-29.md`.
The campaign-8 source-of-truth report remains at
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign8_2026-04-29.md`.
The campaign-7 source-of-truth report remains at
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign7_2026-04-29.md`.
The campaign-6 source-of-truth report remains at
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign6_2026-04-29.md`.
The campaign-5 source-of-truth report remains at
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign5_2026-04-29.md`.
The campaign-4 source-of-truth report remains at
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign4_2026-04-29.md`.
The campaign-3 source-of-truth report remains at
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign3_2026-04-28.md`.
The campaign-2 source-of-truth report remains at
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign2_2026-04-28.md`.
The broader first H100 hillclimb report remains at
`docs/benchmarks/reports/cuda_deep_optimization_h100_2026-04-28.md`.
The completed cross-architecture checkpoint is
`docs/plans/cuda_cross_architecture_campaign10_plan.md`. The completed
residual-risk checkpoint is
`docs/plans/cuda_residual_risk_campaign11_plan.md`, which closes the in-scope
Campaign 10 residual risks without adding new NVIDIA host lanes. Future CUDA
checkpoints should start from Campaign 10 and Campaign 11 evidence and target
release packaging, separately approved additional portability lanes, or a
specific retained consumer with an accepted API and memory-ownership contract.

Acceptance:

```text
README includes a plot generated from checked-in benchmark evidence
README plot shows CPU scalar, every captured optimized CPU selector, CUDA transfer-inclusive, boundary-specific CUDA rows such as operator-resident or device-output where relevant, CUDA device-resident, and semantically comparable external package baselines where data exists
README plot records unavailable external baseline status or reasons when a planned comparable baseline cannot run
plot-rendering commands are documented and reproducible
H100 CUDA reports remain labeled as source-build hardware evidence, not portable wheel claims
CUDA benchmark producer times every available optimized CPU selector covered by the operation
CUDA benchmark producer reports unavailable optimized CPU selectors with reasons
CUDA scaling benchmark profiles cover multiple sizes for simplify, expectation, dense commutation, and matmul+simplify
Nsight Systems and Nsight Compute evidence is captured for CUDA hillclimb claims when NVIDIA profiling tools are available
Compute Sanitizer remains clean after CUDA performance changes
CUDA reports separate kernel limits from host allocation, synchronization, transfer, Python binding, and result-materialization overhead
the overnight CUDA/CPU hillclimb ends with a comprehensive checked-in performance optimization and profiling report, following docs/benchmarks/protocol.md
the final hillclimb report includes publication-quality measured plots plus architecture, hardware, kernel, and algorithm visuals
the final hillclimb report installs and benchmarks comparable open-source packages when platform-compatible, recording versions, install commands, semantic mappings, correctness checks, and unavailable reasons
cuQuantum cuStateVec is planned as a comparable baseline for statevector Pauli expectation workloads
cuQuantum cuPauliProp is planned only for Pauli-expansion workloads with exact semantic mapping
CUDA-Q is planned as an end-to-end spin-operator observe baseline, not a device-resident sparse-Pauli primitive baseline
Qiskit Aer GPU or Aer cuStateVec is planned as a framework-level circuit/statevector baseline when installed with GPU support
GPU-library benchmark reports record availability, version, GPU enablement, semantic mapping, timing boundary, correctness oracle, and unavailable reasons
future CUDA optimization claims refresh plots and checked-in reports rather than relying on chat-only or ad hoc timing output
```

## Post-CUDA Accelerator Candidates

Goal: evaluate non-CUDA GPU backends only after CUDA source-build support and at least one CUDA kernel are validated.

Current status: `docs/plans/mi300x_rocm_bringup_plan.md`,
`docs/plans/mi300x_rocm_optimization_campaign2_plan.md`,
`docs/plans/mi300x_rocm_optimization_campaign3_plan.md`, and
`docs/plans/mi300x_rocm_optimization_campaign4_plan.md`,
`docs/plans/mi300x_rocm_optimization_campaign5_plan.md`, and
`docs/plans/mi300x_rocm_optimization_campaign6_plan.md` are complete for the
first bounded ROCm/HIP campaigns on a 1x AMD Instinct MI300X. The latest
checked report is
`docs/benchmarks/reports/rocm_mi300x_campaign6_2026-04-30.md`, with Campaign 5
interop and execution-control evidence retained at
`docs/benchmarks/reports/rocm_mi300x_campaign5_2026-04-30.md`, Campaign 4
simplify-hardening evidence retained at
`docs/benchmarks/reports/rocm_mi300x_campaign4_2026-04-30.md`, Campaign 3
simplify evidence retained at
`docs/benchmarks/reports/rocm_mi300x_campaign3_2026-04-30.md`, Campaign 2
device-output evidence retained at
`docs/benchmarks/reports/rocm_mi300x_campaign2_2026-04-30.md` and the bring-up
foundation retained at
`docs/benchmarks/reports/rocm_mi300x_bringup_2026-04-29.md`. The accepted claim
is source-build MI300X evidence for HIP metadata, transfers, pairwise
commutation, HIP device-resident commutation matrices, dense host
materialization, compact count/conflict consumers, and HIP
`DevicePauliSum.simplify()` including the retained generic multi-word
`reduce_by_key` path, HIP `DevicePauliSum.expectation_statevector()` for host
NumPy complex statevectors, and HIP `DevicePauliSum.matmul()`. Campaign 5
rejects public HIP DLPack, HIP CUDA Array Interface, public streams, public
graphs, and public workspaces with evidence; Campaign 6 keeps external HIP
statevector device pointers unavailable. ROCm release wheels, multi-GPU ROCm,
broader AMD GPU support, and simultaneous CUDA+HIP remain separate
release-evidence and API-design work. The next ROCm wave plan is
`docs/plans/rocm_next_waves_plan.md`.

Acceptance for any post-CUDA backend:

```text
ROCm/HIP is the second source-build GPU backend, currently validated on MI300X gfx942
Metal is the planned Apple Silicon source-build backend identity after docs/architecture/apple_accelerator.md
MPS and MPSGraph are implementation adjuncts or baselines, not Wolfgang backend identities
backend contracts are documented before implementation begins
backend build flags, availability checks, transfer semantics, and Python interop are separate from CUDA
CPU/backend equivalence tests pass for every implemented operation
benchmarks follow docs/benchmarks/protocol.md and report transfer-inclusive and device-resident timings
release-supported claims follow docs/architecture/hardware_targets_and_testing.md
```

## Release Readiness

The first release candidate requires:

```text
semantic contracts fully covered by tests
README documents install and optional integrations
CPU wheels build in CI
source CUDA build is validated on at least one CUDA runner
release-supported platform, CPU, CUDA, and wheel claims match hardware target evidence
benchmarks cover Qiskit and OpenFermion baselines where dependencies are available
release evidence satisfies docs/quality/phase_quality_gates.md
```

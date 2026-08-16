# Research provenance and campaign ledger

This document preserves the detailed implementation, benchmark, profiler, and release routing that previously lived in the project landing page. It is an engineering provenance index, not a concise user guide or a claim that every historical surface is currently release-supported.

Before public visibility, sensitive removed evidence must also be purged from reachable Git history using the reviewed [history-sanitization runbook](../release/history_sanitization.md).

The current public support boundary remains [`docs/release/support_matrix.md`](../release/support_matrix.md). Reports retain the exact timing, hardware, toolchain, and negative-result context for each campaign. Raw infrastructure captures are intentionally excluded from the public artifact.
The reusable public/private cloud-run harness for future paid hardware
qualification is [`docs/release/cloud_hardware_qualification_harness.md`](../release/cloud_hardware_qualification_harness.md).

The canonical historical release registry is `docs/release/README.md`; the first
candidate record remains at `docs/release/0.1.0-rc1.md`.

---

# Historical FastPauli engineering ledger
High-performance accelerators for sparse sums of Pauli strings and Hamiltonians in C++20 with CUDA and ROCm/HIP source-build paths.

## Current Status

FastPauli has completed the Phase 11 CUDA kernel slice. The package builds and imports
with a C++20/nanobind extension, `wolfgang_quantum.PauliSum` supports dense-label and
sparse-list construction/export, Qiskit `SparsePauliOp` conversion is available
through the optional `qiskit` extra, OpenFermion `QubitOperator` conversion is
available through the optional `openfermion` extra, scalar `simplify()` combines
duplicate Pauli strings into canonical packed-word order, and Pauli-sum arithmetic
covers addition, scalar multiplication, phase-correct matrix-product multiplication,
guarded pairwise commutation, and deterministic greedy QWC/full commuting groups.
Scalar CPU expectation kernels cover statevectors and diagonal computational-basis
counts. CPU backend dispatch is observable through build metadata and can force
the portable scalar path with `WOLFGANG_CPU_BACKEND=scalar`. Optional oneTBB and
SIMD selectors are implemented for covered commutation/grouping kernels when
compiled and runtime-available; unavailable selectors and unsupported SIMD widths
fail clearly instead of silently relabeling scalar execution as optimized.
Source builds with `WOLFGANG_ENABLE_CUDA=ON` expose explicit
`PauliSum.to_device()` and `DevicePauliSum.to_host()` transfers plus CUDA
`DevicePauliSum.simplify()`, `DevicePauliSum.expectation_statevector()`,
`DevicePauliSum.commutes_with()`, `DevicePauliSum.commutes_with_device()`,
`DeviceCommutationMatrix.count_commuting(axis=None|0|1)`,
`DeviceCommutationMatrix.conflict_degrees(axis=None|0|1)`, read-only
`DeviceCommutationMatrix` DLPack export, and
`DevicePauliSum.matmul()` kernels. CUDA performance claims remain tied to
benchmark reports and source-build validation; CUDA wheels are still
deliberately deferred.
Source builds with `WOLFGANG_ENABLE_HIP=ON` now expose MI300X-evidenced
ROCm/HIP backend slices: explicit transfers, HIP runtime metadata,
`DevicePauliSum.backend == "hip"`, HIP pairwise commutation, HIP
`DevicePauliSum.commutes_with_device()`, and HIP
`DeviceCommutationMatrix.count_commuting(axis=None|0|1)` and
`conflict_degrees(axis=None|0|1)` compact consumers, plus HIP
`DevicePauliSum.simplify()` with device-resident output, HIP
`DevicePauliSum.expectation_statevector()` for host NumPy `complex64` and
`complex128` statevectors, and HIP `DevicePauliSum.matmul()`. HIP remains
source-build-only, mutually exclusive with CUDA in the current architecture,
and scoped to checked MI300X evidence rather than portable wheel support.
The Campaign 7 release-support lane now records repeatable MI300X source-build
validation, rocprof evidence, duplicate-pressure smoke rows, and terminal
statuses for ROCm packaging and portability boundaries.
The ROCm Campaign 8 architecture-readiness lane now converts the remaining
Campaign 7 portability, packaging, profiler, interop, multi-GPU, simultaneous
CUDA+HIP, and targeted-performance questions into explicit gates before new
ROCm public APIs or support claims are attempted.
The backend-neutral accelerator Campaign 9 target-specific closeout has landed:
`PauliSum.to_device()` and `DeviceCommutationMatrix.empty()` accept an
explicit `backend=None|"auto"|"cuda"|"hip"|"metal"` selector, `_accelerator_status()`
reports structured compiled and available backend sets, and
`DeviceCommutationMatrix.backend` mirrors the existing device-object backend
identity contract. The prep slice also adds disjoint CMake CUDA/HIP/stub source
sets, simulated backend/device validation tests, benchmark/build metadata for
Campaign 9 rows, H100 CUDA-only validation, MI300X HIP-only validation, and
documented CUDA+HIP configure-time rejection evidence. Campaign 9 follows a
target-specific accelerator build policy: CPU-only, CUDA-target, HIP-target,
and Apple Metal-target builds are the supported normal modes, while combining
CUDA, HIP, or Metal target flags remains a deliberate configure-time error.
The release-candidate foundation lane is tracked in
`docs/plans/release_candidate_foundation_plan.md`: it adds checked changelog and
release evidence surfaces plus CPU source-distribution and wheel
build/install/import validation, without changing runtime behavior or adding
accelerator wheel claims.
The `0.1.0rc2` release-candidate checkpoint is tracked in
`docs/release/0.1.0-rc2.md`,
`docs/plans/release_candidate_next_checkpoint_plan.md`, and
`docs/release/support_matrix.md`: it publishes the next CPU source distribution
and macOS arm64 CPU wheel lane while keeping CUDA, ROCm/HIP, and Apple Metal as
source-build lanes and keeping combined accelerator, Windows, and PyPI release
claims mechanically unavailable without later evidence.
The final `0.1.0` PyPI release lane is tracked in
`docs/release/0.1.0.md` and
`docs/plans/release_0_1_0_wheelhouse_foundation_plan.md`: it adds the CPU-only
cibuildwheel matrix, manual release-wheelhouse workflow, installed-wheel smoke,
checksum manifest, exact CPU wheelhouse completeness checks, and explicit
tag-ref-gated TestPyPI/PyPI trusted-publishing gates while keeping accelerator
and Windows wheel claims unavailable. The `v0.1.0` tag-ref workflow has
produced the complete CPU wheelhouse and checksum evidence. The corrected
tag-ref run has passed TestPyPI upload and clean install smoke; PyPI
publication remains unavailable until PyPI trusted publishing is configured for
the observed `pypi` environment claims and the PyPI publish job succeeds.
The published `0.2.3` GitHub-only successor checkpoint is tracked in
`docs/release/0.2.3.md`, `docs/release/README.md`, and
`docs/release/support_matrix.md`: it advances the active source version from the
corrected capabilities fix commit, preserves immutable `v0.2.2` tag provenance
without rewriting historical evidence, keeps the quarantined `v0.2.0` and
`v0.2.1` draft releases unchanged, records the published GitHub release and
exact-tag wheelhouse evidence, and keeps TestPyPI/PyPI out of scope for this
successor slice.
The Apple Silicon accelerator implementation lane is tracked in
`docs/architecture/apple_accelerator.md` and
`docs/plans/apple_metal_mps_bringup_plan.md`, with the first measured
optimization refresh tracked in
`docs/plans/apple_metal_optimization_campaign1_plan.md` and
`docs/plans/apple_metal_optimization_campaign2_plan.md` and
`docs/plans/apple_metal_optimization_campaign3_plan.md` and
`docs/plans/apple_metal_optimization_campaign4_plan.md` and
`docs/plans/apple_metal_optimization_campaign5_plan.md` and
`docs/plans/apple_metal_optimization_campaign6_plan.md` and
`docs/plans/apple_metal_optimization_campaign7_plan.md` and
`docs/plans/apple_metal_optimization_campaign8_plan.md`. It adds the
`backend="metal"` identity, `WOLFGANG_ENABLE_METAL=ON` source-build flag,
Metal status metadata, transfers, and pairwise commutation source code while
keeping Metal target builds separate from CUDA and HIP builds. Metal wheels,
MPSGraph-first sparse kernels, raw Metal buffer export, and generic Apple GPU
support claims remain unavailable. Local Apple Silicon runtime validation now
passes in an elevated Codex command context on an Apple M4 Pro, including
transfer, pairwise commutation, retained device matrix, and compact consumer
equivalence checks. The non-elevated Codex command sandbox still hides Metal
devices from `MTLCreateSystemDefaultDevice()`, so Apple Metal validation must
record whether the process is sandboxed or elevated. Full Xcode, automatic
selection of the installed Metal Toolchain component inside validation, and a
short Metal System Trace capture are recorded in
`docs/benchmarks/reports/apple_metal_bringup_2026-05-01.md`; this remains
source-build evidence rather than a Metal wheel or generic Apple GPU support
claim. Apple Metal Campaign 1 extends that evidence with scaling benchmark
rows, retained-output timing boundaries, and the broad README landscape refresh
recorded in
`docs/benchmarks/reports/apple_metal_optimization_campaign1_2026-05-05.md`.
Apple Metal Campaign 2 keeps the same public API boundary while adding
two-dimensional commutation dispatch, retained one-word specialization,
benchmark-only two-word specialization evidence, legacy flat-generic baseline
rows, and a refreshed broad landscape in
`docs/benchmarks/reports/apple_metal_optimization_campaign2_2026-05-05.md`.
Apple Metal Campaign 3 continues that evidence path with benchmark-only
offline `.metallib` loading, private output storage plus blit staging, GPU
compact-consumer reductions, two-word specialization policy evidence, profiler
availability notes, and exact MPSGraph/PyTorch MPS baseline status in
`docs/benchmarks/reports/apple_metal_optimization_campaign3_2026-05-06.md`.
Apple Metal Campaign 4 continues the local Apple M4 Pro optimization path with
larger two-word and compact-consumer evidence, a benchmark-only parallel
block-reduction total count, private device-boundary rows, and explicit
deferral of PyPI publication, Windows support, and older macOS compatibility
in
`docs/benchmarks/reports/apple_metal_optimization_campaign4_2026-05-06.md`.
Apple Metal Campaign 5 adds correct source-build-only
`DevicePauliSum.simplify()` behavior for Metal builds as a retained
transfer-reference path:
`Metal DevicePauliSum -> host PauliSum -> CPU simplify -> Metal DevicePauliSum`.
It records the path as `metal_simplify_transfer_reference` with the
`device_to_host_cpu_simplify_host_to_device` boundary and keeps Metal
statevector expectation, Metal matmul, Metal wheels, Windows, older macOS
support, and PyPI publication outside the slice. The retained Metal simplify
implementation is a correctness bridge, not a device-resident GPU
duplicate-reduction path. Evidence is recorded in
`docs/benchmarks/reports/apple_metal_optimization_campaign5_2026-05-06.md`.
Apple Metal Campaign 6 device-resident simplify groundwork keeps that public
behavior unchanged while adding a private `MetalWorkspace` model, Campaign 6
simplify benchmark cases, and `metal_simplify_workspace_probe` status rows.
The private MetalWorkspace remains an internal `src/metal` scratch contract,
not a public API.
The device-resident simplify candidate remains blocked until checked Metal
sort/prefix/reduce primitives exist, so Campaign 6 records groundwork evidence
rather than a retained GPU duplicate-reduction speedup. Evidence is recorded in
`docs/benchmarks/reports/apple_metal_optimization_campaign6_2026-05-07.md`.
Apple Metal Campaign 7 adds a checked device-resident simplify primitive stack
for benchmark-only one-word, fixed-dyadic-coefficient inputs. It includes a
private Metal bitonic key sort, prefix-sum, reduce-by-key, and
survivor-compaction path plus `metal_simplify_device_candidate` rows. This is
not a public simplify promotion: Apple Metal rejects `double` arithmetic in
kernels on this host, so the candidate is explicitly limited to coefficients
exactly representable as signed fixed32 dyadic values whose checked accumulated
sums and tolerance threshold fit exact uint64 squared-magnitude comparison;
public `DevicePauliSum.simplify()` remains the transfer-reference bridge.
Evidence is recorded in
`docs/benchmarks/reports/apple_metal_optimization_campaign7_2026-05-07.md`.
Apple Metal Campaign 8 makes the Campaign 7 candidate measurable enough to
decide whether it is performance-relevant or still experimental. It adds
private timing decomposition evidence through `timing_decomposition_seconds`,
`pipeline_cache`, dispatch counts, and `performance_decision` metadata for the
checked one-word simplify candidate while keeping the public Metal simplify API
on the transfer-reference bridge. The checked rows currently classify the
candidate as `experimental`. Evidence is recorded in
`docs/benchmarks/reports/apple_metal_optimization_campaign8_2026-05-07.md`.

```python
import numpy as np

from wolfgang_quantum import PauliSum

op = PauliSum.from_labels(["XYZ"], [1.0])
assert op.to_sparse_list() == [("ZYX", [0, 1, 2], 1.0 + 0.0j)]

combined = PauliSum.from_labels(["X", "X", "Z"], [1.0, -0.25, 2.0]).simplify()
assert combined.to_labels()[0] == ["Z", "X"]

lhs = PauliSum.from_labels(["X"], [2.0])
rhs = PauliSum.from_labels(["Y"], [3.0])
assert (lhs + rhs).to_labels()[0] == ["X", "Y"]
assert (0.5 * lhs).to_labels()[1][0] == 1.0 + 0.0j
product_labels, product_coeffs = (lhs @ rhs).to_labels()
assert product_labels == ["Z"]
assert product_coeffs[0] == 6.0j

assert lhs.commutes_with(rhs) is False
groups = PauliSum.from_labels(["XX", "XI", "IZ"], [1.0, 2.0, 3.0]).group_commuting(mode="qwc")
assert [group.to_labels()[0] for group in groups] == [["XX", "XI"], ["IZ"]]

z_op = PauliSum.from_labels(["ZI", "IZ"], [1.0, -0.5])
psi = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
assert z_op.expectation_statevector(psi) == 0.5 + 0.0j
assert (
    PauliSum.from_labels(["IZ"], [1.0]).expectation_z_counts({"00": 3, "01": 1})
    == 0.5 + 0.0j
)

empty = PauliSum.empty(num_qubits=3)
assert empty.num_terms == 0
```

```python
from qiskit.quantum_info import SparsePauliOp

op = PauliSum.from_qiskit(SparsePauliOp(["XYZ"], coeffs=[1.0]))
assert op.to_qiskit().paulis.to_labels() == ["XYZ"]
```

```python
from openfermion.ops import QubitOperator

op = PauliSum.from_openfermion(QubitOperator("X0 Y2", 1.0))
assert op.to_sparse_list() == [("XY", [0, 2], 1.0 + 0.0j)]
assert op.to_openfermion().terms == {((0, "X"), (2, "Y")): 1.0 + 0.0j}
```

Optional oneTBB and explicit SIMD commutation kernels are part of the CPU
dispatch surface. CUDA transfer support is source-build-only and disabled by
default for CPU wheels.

## Performance Landscape

The latest checked-in accelerator reports are source-build evidence on MI300X
ROCm/HIP, Apple M4 Pro Metal, A100, RTX PRO 6000 Blackwell, and H100. They are
not portable wheel claims and not promises for every CPU or GPU. The current
README plot is the tracked across-the-board performance landscape view: it
includes FastPauli CPU scalar, default, oneTBB, AVX2, and AVX-512 selectors
where captured; CUDA transfer-inclusive and device-resident paths; CUDA operator-resident,
device-output, compact-consumer, and CSR-baseline rows where relevant; CuPy and
PyTorch DLPack consumer rows; ROCm/HIP transfer-inclusive, device-resident,
compact-consumer, and explicit `to_host()` rows; Apple Metal transfer-inclusive,
device-resident host-output, retained device-matrix allocation, reused-output,
explicit `to_host()`, compact-consumer, benchmark-only private-blit,
benchmark-only offline `.metallib`, and benchmark-only GPU compact-reduction
rows; Apple Metal simplify transfer-reference rows; benchmark-only Apple Metal
simplify device-candidate rows; and semantically
comparable external package baselines where available.
Apple Metal rows are included as local Apple M4 Pro source-build evidence, not
generic Apple GPU or wheel support claims.

![FastPauli accelerator performance landscape](../benchmarks/plots/accelerator_landscape_with_rocm.svg)

Latest source reports:
[Apple Metal Campaign 8](../benchmarks/reports/apple_metal_optimization_campaign8_2026-05-07.md),
[Apple Metal Campaign 7](../benchmarks/reports/apple_metal_optimization_campaign7_2026-05-07.md),
[Apple Metal Campaign 6](../benchmarks/reports/apple_metal_optimization_campaign6_2026-05-07.md),
[Apple Metal Campaign 5](../benchmarks/reports/apple_metal_optimization_campaign5_2026-05-06.md),
[Apple Metal Campaign 4](../benchmarks/reports/apple_metal_optimization_campaign4_2026-05-06.md),
[Apple Metal Campaign 3](../benchmarks/reports/apple_metal_optimization_campaign3_2026-05-06.md),
[Apple Metal Campaign 2](../benchmarks/reports/apple_metal_optimization_campaign2_2026-05-05.md),
[Apple Metal Campaign 1](../benchmarks/reports/apple_metal_optimization_campaign1_2026-05-05.md),
[Apple Metal bring-up](../benchmarks/reports/apple_metal_bringup_2026-05-01.md),
[CUDA cross-architecture Campaign 10](../benchmarks/reports/cuda_cross_architecture_campaign10_2026-04-29.md)
and
[CUDA residual-risk Campaign 11](../benchmarks/reports/cuda_residual_risk_campaign11_2026-04-29.md).
The latest ROCm planning evidence is the
[ROCm Campaign 8 architecture-readiness report](../benchmarks/reports/rocm_campaign8_architecture_readiness_2026-05-01.md),
which records that no HIP kernel, public Python API, ROCm wheel, or multi-GPU
runtime behavior changed, and it retains the MI300X source-build release lane
while accepting explicit future gates for backend-neutral accelerator design,
ROCm packaging, non-MI300X AMD portability, rocprofv3 migration, HIP interop
reconsideration, and targeted performance reopening. The latest ROCm
source-build runtime evidence remains the
[ROCm MI300X Campaign 7 report](../benchmarks/reports/rocm_mi300x_campaign7_2026-04-30.md),
which converts retained MI300X HIP operation evidence into a repeatable
source-build release-support lane. Campaign 7 validates CPU-only control,
MI300X HIP source build, retained HIP operations, CUDA+HIP configure rejection,
rocprof trace/stats, duplicate-pressure simplify and matmul smoke rows, and
explicit terminal statuses for ROCm wheels, portability, external HIP
statevectors, HIP DLPack, HIP `__cuda_array_interface__`, public streams,
graphs, workspaces, multi-GPU ROCm, simultaneous CUDA+HIP, and backend-neutral
accelerator design. ROCm wheels remain unavailable, and broader AMD portability
remains blocked until a non-MI300X AMD GPU lane is available. The previous
[ROCm MI300X Campaign 6 report](../benchmarks/reports/rocm_mi300x_campaign6_2026-04-30.md)
retains HIP `DevicePauliSum.expectation_statevector()` for host NumPy
`complex64` and `complex128` statevectors and HIP `DevicePauliSum.matmul()` for
`simplify=True` and `simplify=False` on MI300X. The previous
[ROCm MI300X Campaign 5 report](../benchmarks/reports/rocm_mi300x_campaign5_2026-04-30.md)
rejects public HIP DLPack after PyTorch ROCm consumed the candidate versioned
`kDLROCM` capsule in a temporary candidate probe but accepted mutation of the
read-only view. The previous
[ROCm MI300X Campaign 4 report](../benchmarks/reports/rocm_mi300x_campaign4_2026-04-30.md)
hardens HIP `DevicePauliSum.simplify()` without adding public HIP APIs. It
retains a parallel generic multi-word `reduce_by_key` path, records a 16.4x
resident A/B speedup over the Campaign 3 serial generic fallback on the
130-qubit/4096-term row, records custom packed-key probes as unavailable
because no distinct lower-level rocPRIM/hipCUB implementation was retained, and
records rocPRIM/hipCUB scratch workspace probes as unavailable for the current
rocThrust implementation boundary. The previous
[ROCm MI300X Campaign 3 report](../benchmarks/reports/rocm_mi300x_campaign3_2026-04-30.md)
validates HIP `DevicePauliSum.simplify()` with device-resident output,
CPU/HIP benchmark comparisons, rocThrust duplicate reduction, rocprof
trace/counter evidence, and a terminal-status table for the remaining Campaign
2 headroom items. The earlier
[ROCm MI300X Campaign 2 report](../benchmarks/reports/rocm_mi300x_campaign2_2026-04-30.md)
validates HIP device-resident commutation output, dense `to_host()`
materialization, and compact count/conflict consumers on `gfx942`. The earlier
[ROCm MI300X bring-up report](../benchmarks/reports/rocm_mi300x_bringup_2026-04-29.md)
remains the source-build foundation evidence for HIP metadata, transfers, and
host-output pairwise commutation. Future ROCm work is tracked in the
[ROCm next waves plan](../plans/rocm_next_waves_plan.md), with the next
completed architecture-readiness plan in
[ROCm Campaign 8 architecture readiness](../plans/mi300x_rocm_optimization_campaign8_plan.md).
The accepted implementation plan for backend-neutral target-specific accelerator
builds is
[Backend-neutral accelerator Campaign 9](../plans/backend_neutral_accelerator_campaign9_plan.md).
The checked closeout report is
[Backend-neutral accelerator Campaign 9 closeout](../benchmarks/reports/backend_neutral_accelerator_campaign9_2026-05-01.md).
Campaign 9 is not a wheel-support, broader AMD portability, HIP DLPack,
multi-GPU, Metal/MPS, or simultaneous CUDA+HIP source-build support claim.
Under the current policy, simultaneous CUDA+HIP source builds are a future
mixed-runtime design topic, not a Campaign 9 completion requirement.
CUDA Campaign 10 closes every H100 CUDA Campaign 9 remaining-headroom item with
a non-deferred status. A100 `sm_80` and RTX PRO 6000 Blackwell `sm_120` source
builds compile and run, PyTorch CUDA consumes the read-only
`DeviceCommutationMatrix.__dlpack__` export, the true public grouping API is
rejected while `DeviceCommutationMatrix.conflict_degrees(axis=None|0|1)`
remains the accepted compact public summary API, public stream-aware execution
and CUDA Graph replay are rejected because launch overhead is not dominant, and
CSR scatter tuning remains rejected because retained compact consumers avoid
full CSR edge-list materialization.
Campaign 11 closes the immediate Campaign 10 residual-risk follow-up: RTX PRO
6000 Blackwell Nsight Compute counters were captured, A100 counter capture is
terminally blocked by host permissions, and the nanobind sanitizer messages are
classified as process-teardown diagnostics after clean targeted lifecycle
subprocesses.
The previous
[H100 CUDA deferred-headroom campaign 9](../benchmarks/reports/cuda_deferred_headroom_campaign9_2026-04-29.md)
captured privileged Nsight Compute counters on H100 and implemented read-only
CuPy DLPack coverage.
The previous
[H100 CUDA deep optimization campaign 8](../benchmarks/reports/cuda_deep_optimization_h100_campaign8_2026-04-29.md)
report retains private benchmark-only compact device-resident graph and
grouping consumers that avoid full CSR edge-list host export on high-scale H100
rows.
The previous
[H100 CUDA deep optimization campaign 7](../benchmarks/reports/cuda_deep_optimization_h100_campaign7_2026-04-29.md)
report adds private benchmark-only fused consumers for anti-commutation CSR
graph export, conflict-degree summaries, and grouping-oriented summaries over
`DeviceCommutationMatrix`. The previous
[H100 CUDA deep optimization campaign 6](../benchmarks/reports/cuda_deep_optimization_h100_campaign6_2026-04-29.md)
report retains `DeviceCommutationMatrix.count_commuting(axis=None|0|1)` as
a compact GPU-resident consumer boundary and adds CuPy consumer baselines
through the CUDA Array Interface. The previous
[H100 CUDA deep optimization campaign 5](../benchmarks/reports/cuda_deep_optimization_h100_campaign5_2026-04-29.md)
report retains the experimental dense `DeviceCommutationMatrix` device-output
commutation API and validates that boundary with H100 correctness, sanitizer,
benchmark, and profiler evidence. The
[H100 CUDA deep optimization campaign 4](../benchmarks/reports/cuda_deep_optimization_h100_campaign4_2026-04-29.md)
report implements a private CUDA workspace and benchmark-only CUB/CCCL
scratch-boundary probes, rejects the narrow CUB radix-sort duplicate-reduction
prototype for production, and keeps public CUDA ownership and stream semantics
unchanged. The
[H100 CUDA deep optimization campaign 3](../benchmarks/reports/cuda_deep_optimization_h100_campaign3_2026-04-28.md)
report retains a packed-key CUDA simplify path for one-word operators with at
most 32 qubits and adds allocation/materialization evidence. The
[H100 CUDA deep optimization campaign 2](../benchmarks/reports/cuda_deep_optimization_h100_campaign2_2026-04-28.md)
report retains the fused statevector expectation accumulator and size-gated
launch policy. The broader first
[CUDA H100 deep optimization evidence](../benchmarks/reports/cuda_deep_optimization_h100_2026-04-28.md)
remains the initial H100 hillclimb report. These reports include
smoke/default/stress/extreme H100 scaling, privileged Nsight Compute passes,
Nsight Systems traces, Compute Sanitizer, cuobjdump PTX/SASS inventory,
retained and rejected A/B experiments, and open-source competitor baselines
including a cuStateVec statevector expectation comparison.
Further CUDA hillclimbing should follow the
[CUDA deep optimization plan](../plans/cuda_deep_optimization_plan.md).
Use the repo-local profiling ladder:

```bash
python scripts/cuda_deep_profile.py --dry-run --json --profile stress
```

The campaign-7 report assets are generated with:

```bash
python scripts/render_cuda_campaign7_assets.py \
  --data-dir docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29 \
  --plot-dir docs/benchmarks/plots
```

The Campaign 10 CUDA landscape is generated with:

```bash
python scripts/render_cuda_campaign10_assets.py \
  --data-dir docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29 \
  --plot-dir docs/benchmarks/plots
```

Campaign 10 kept its plot as a broad CPU/CUDA/external performance landscape
rather than a narrow single-surface view; the current README landscape extends
that policy with checked ROCm and Apple Metal rows.

The Apple Metal Campaign 8 assets and README landscape refresh are generated
with:

```bash
python scripts/render_apple_metal_assets.py \
  --data-dir docs/benchmarks/data/apple_metal_optimization_campaign8_2026-05-07 \
  --plot-dir docs/benchmarks/plots
```

The ROCm Campaign 2 report-local assets are generated with:

```bash
python scripts/render_rocm_campaign2_assets.py \
  --data-dir docs/benchmarks/data/rocm_mi300x_campaign2_2026-04-30 \
  --plot-dir docs/benchmarks/plots
```

The ROCm Campaign 5 report-local assets and README landscape are generated
with:

```bash
python scripts/render_rocm_campaign5_assets.py \
  --data-dir docs/benchmarks/data/rocm_mi300x_campaign5_2026-04-30 \
  --plot-dir docs/benchmarks/plots
```

Campaign 5 did not add a retained comparable DLPack performance row, so the
renderer preserves the broad CPU/CUDA/ROCm/external README landscape and emits
a report-local interop plot. The ROCm Campaign 7 report-local assets and README
landscape are generated with:

```bash
python scripts/render_rocm_campaign7_assets.py \
  --data-dir docs/benchmarks/data/rocm_mi300x_campaign7_2026-04-30 \
  --plot-dir docs/benchmarks/plots
```

The ROCm Campaign 6 report-local assets and README
landscape are generated with:

```bash
python scripts/render_rocm_campaign6_assets.py \
  --data-dir docs/benchmarks/data/rocm_mi300x_campaign6_2026-04-30 \
  --plot-dir docs/benchmarks/plots
```

The ROCm Campaign 4 report-local assets and README
landscape are generated with:

```bash
python scripts/render_rocm_campaign4_assets.py \
  --data-dir docs/benchmarks/data/rocm_mi300x_campaign4_2026-04-30 \
  --plot-dir docs/benchmarks/plots
```

The ROCm Campaign 3 report-local assets and README landscape are generated
with:

```bash
python scripts/render_rocm_campaign3_assets.py \
  --data-dir docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30 \
  --plot-dir docs/benchmarks/plots
```

The README plot must remain a broad CPU/CUDA/ROCm/Apple Metal/external
landscape rather than a narrow single-surface view.

The campaign-6 report assets are generated with:

```bash
python scripts/render_cuda_campaign6_assets.py \
  --data-dir docs/benchmarks/data/cuda_deep_optimization_h100_campaign6_2026-04-29 \
  --plot-dir docs/benchmarks/plots
```

The following campaign-3, campaign-2, and initial H100 commands are historical
reproduction recipes. They require the separately retained private raw benchmark
archive and do not run from a public clone alone.

```bash
python scripts/render_cuda_campaign3_assets.py \
  --raw-dir docs/benchmarks/data/cuda_deep_optimization_h100_campaign3_2026-04-28/raw \
  --summary-output docs/benchmarks/data/cuda_deep_optimization_h100_campaign3_2026-04-28/summary.json \
  --plot-dir docs/benchmarks/plots
```

The campaign-2 historical recipe is:

```bash
python scripts/render_cuda_campaign2_assets.py \
  --summary docs/benchmarks/data/cuda_deep_optimization_h100_campaign2_2026-04-28/summary.json \
  --raw-dir docs/benchmarks/data/cuda_deep_optimization_h100_campaign2_2026-04-28/raw \
  --plot-dir docs/benchmarks/plots
```

The first H100 deep-optimization historical recipe is:

```bash
python scripts/render_cuda_deep_report_assets.py \
  --raw-dir docs/benchmarks/data/cuda_deep_optimization_h100_2026-04-28/raw \
  --summary-output docs/benchmarks/data/cuda_deep_optimization_h100_2026-04-28/summary.json \
  --plot-dir docs/benchmarks/plots
```

GPU-library baselines such as NVIDIA cuQuantum, CUDA-Q, and Qiskit Aer GPU are
tracked in the benchmark protocol and should be added only where the workload
semantics are genuinely comparable.

## Local Validation

```bash
python -m pip install -e ".[test]" \
  --config-settings=cmake.define.WOLFGANG_ENABLE_INTERNAL_BINDINGS=ON
python scripts/check_release_readiness.py
python scripts/validate.py
python scripts/validate_release_artifacts.py --output-dir /tmp/fastpauli-release-artifacts
python benchmarks/bench_simplify.py --smoke --repeat 1
python benchmarks/bench_multiply.py --smoke --repeat 1
python benchmarks/bench_grouping.py --smoke --repeat 1
python benchmarks/bench_expectation.py --smoke --repeat 1
python benchmarks/bench_cuda_kernels.py --smoke --repeat 1
python benchmarks/bench_cuda_scaling.py --profile smoke --repeat 1 --json
python benchmarks/bench_cpu_dispatch.py --smoke --repeat 1
python benchmarks/bench_cpu_thresholds.py --smoke --repeat 1

python -m pip install -e ".[test,qiskit]"
python -m pytest tests/test_qiskit_adapter.py

python -m pip install -e ".[test,openfermion]"
python -m pytest tests/test_openfermion_adapter.py
python benchmarks/bench_openfermion_conversion.py --smoke --repeat 1
```

CUDA validation requires a CUDA toolkit, `nvcc` on `PATH`, and a visible NVIDIA
device:

```bash
WOLFGANG_VALIDATE_CUDA=1 WOLFGANG_CUDA_ARCHITECTURES=90 python scripts/validate.py
```

## Planning Sources

- [Agent guide](https://github.com/sghowell/wolfgang/blob/main/AGENTS.md)
- [Changelog](https://github.com/sghowell/wolfgang/blob/main/CHANGELOG.md)
- [Implementation plan](../plans/cpp_cuda_implementation_plan.md)
- [Release candidate foundation plan](../plans/release_candidate_foundation_plan.md)
- [Release candidate next checkpoint plan](../plans/release_candidate_next_checkpoint_plan.md)
- [0.2.3 successor release evidence ledger](../release/0.2.3.md)
- [0.2.2 historical provenance ledger](../release/0.2.2.md)
- [Release 0.1.0 wheelhouse foundation plan](../plans/release_0_1_0_wheelhouse_foundation_plan.md)
- [Apple Metal/MPS bring-up plan](../plans/apple_metal_mps_bringup_plan.md)
- [Apple Metal optimization Campaign 1 plan](../plans/apple_metal_optimization_campaign1_plan.md)
- [Apple Metal optimization Campaign 2 plan](../plans/apple_metal_optimization_campaign2_plan.md)
- [Apple Metal optimization Campaign 3 plan](../plans/apple_metal_optimization_campaign3_plan.md)
- [Apple Metal optimization Campaign 4 plan](../plans/apple_metal_optimization_campaign4_plan.md)
- [Apple Metal optimization Campaign 5 plan](../plans/apple_metal_optimization_campaign5_plan.md)
- [Apple Metal optimization Campaign 6 plan](../plans/apple_metal_optimization_campaign6_plan.md)
- [Apple Metal optimization Campaign 7 plan](../plans/apple_metal_optimization_campaign7_plan.md)
- [Apple Metal optimization Campaign 8 plan](../plans/apple_metal_optimization_campaign8_plan.md)
- [CUDA deep optimization plan](../plans/cuda_deep_optimization_plan.md)
- [H100 deep optimization Campaign 2 plan](../plans/h100_deep_optimization_campaign2_plan.md)
- [H100 deep optimization Campaign 3 plan](../plans/h100_deep_optimization_campaign3_plan.md)
- [H100 deep optimization Campaign 4 plan](../plans/h100_deep_optimization_campaign4_plan.md)
- [H100 deep optimization Campaign 5 plan](../plans/h100_deep_optimization_campaign5_plan.md)
- [H100 deep optimization Campaign 6 plan](../plans/h100_deep_optimization_campaign6_plan.md)
- [H100 deep optimization Campaign 7 plan](../plans/h100_deep_optimization_campaign7_plan.md)
- [H100 deep optimization Campaign 8 plan](../plans/h100_deep_optimization_campaign8_plan.md)
- [H100 deep optimization Campaign 9 plan](../plans/h100_deep_optimization_campaign9_plan.md)
- [CUDA cross-architecture Campaign 10 plan](../plans/cuda_cross_architecture_campaign10_plan.md)
- [CUDA residual-risk Campaign 11 plan](../plans/cuda_residual_risk_campaign11_plan.md)
- [MI300X ROCm/HIP bring-up plan](../plans/mi300x_rocm_bringup_plan.md)
- [ROCm next waves plan](../plans/rocm_next_waves_plan.md)
- [MI300X ROCm optimization Campaign 2 plan](../plans/mi300x_rocm_optimization_campaign2_plan.md)
- [MI300X ROCm optimization Campaign 3 plan](../plans/mi300x_rocm_optimization_campaign3_plan.md)
- [MI300X ROCm optimization Campaign 4 plan](../plans/mi300x_rocm_optimization_campaign4_plan.md)
- [MI300X ROCm optimization Campaign 5 plan](../plans/mi300x_rocm_optimization_campaign5_plan.md)
- [MI300X ROCm optimization Campaign 6 plan](../plans/mi300x_rocm_optimization_campaign6_plan.md)
- [MI300X ROCm optimization Campaign 7 plan](../plans/mi300x_rocm_optimization_campaign7_plan.md)
- [Backend-neutral accelerator Campaign 9 plan](../plans/backend_neutral_accelerator_campaign9_plan.md)
- [Backend-neutral accelerator Campaign 9 closeout report](../benchmarks/reports/backend_neutral_accelerator_campaign9_2026-05-01.md)
- [Apple Metal bring-up report](../benchmarks/reports/apple_metal_bringup_2026-05-01.md)
- [Apple Metal optimization Campaign 1 report](../benchmarks/reports/apple_metal_optimization_campaign1_2026-05-05.md)
- [Apple Metal optimization Campaign 2 report](../benchmarks/reports/apple_metal_optimization_campaign2_2026-05-05.md)
- [Apple Metal optimization Campaign 3 report](../benchmarks/reports/apple_metal_optimization_campaign3_2026-05-06.md)
- [Apple Metal optimization Campaign 4 report](../benchmarks/reports/apple_metal_optimization_campaign4_2026-05-06.md)
- [Apple Metal optimization Campaign 5 report](../benchmarks/reports/apple_metal_optimization_campaign5_2026-05-06.md)
- [Apple Metal optimization Campaign 6 report](../benchmarks/reports/apple_metal_optimization_campaign6_2026-05-07.md)
- [Apple Metal optimization Campaign 7 report](../benchmarks/reports/apple_metal_optimization_campaign7_2026-05-07.md)
- [Apple Metal optimization Campaign 8 report](../benchmarks/reports/apple_metal_optimization_campaign8_2026-05-07.md)
- [CUDA cross-architecture Campaign 10 report](../benchmarks/reports/cuda_cross_architecture_campaign10_2026-04-29.md)
- [CUDA residual-risk Campaign 11 report](../benchmarks/reports/cuda_residual_risk_campaign11_2026-04-29.md)
- [ROCm MI300X Campaign 7 report](../benchmarks/reports/rocm_mi300x_campaign7_2026-04-30.md)
- [ROCm MI300X Campaign 6 report](../benchmarks/reports/rocm_mi300x_campaign6_2026-04-30.md)
- [ROCm MI300X Campaign 5 report](../benchmarks/reports/rocm_mi300x_campaign5_2026-04-30.md)
- [ROCm MI300X Campaign 4 report](../benchmarks/reports/rocm_mi300x_campaign4_2026-04-30.md)
- [ROCm MI300X Campaign 3 report](../benchmarks/reports/rocm_mi300x_campaign3_2026-04-30.md)
- [ROCm MI300X Campaign 2 report](../benchmarks/reports/rocm_mi300x_campaign2_2026-04-30.md)
- [ROCm MI300X bring-up report](../benchmarks/reports/rocm_mi300x_bringup_2026-04-29.md)
- [Semantic contracts](../architecture/semantic_contracts.md)
- [CUDA backend architecture](../architecture/cuda_backend.md)
- [ROCm/HIP backend architecture](../architecture/rocm_backend.md)
- [Apple Metal accelerator architecture](../architecture/apple_accelerator.md)
- [Hardware targets and testing](../architecture/hardware_targets_and_testing.md)
- [Testing and CI architecture](../architecture/testing_and_ci.md)
- [Adapter contracts](../architecture/adapter_contracts.md)
- [Benchmark protocol](../benchmarks/protocol.md)
- [Phase quality gates](../quality/phase_quality_gates.md)
- [Agent harness](../quality/agent_harness.md)
- [Agent-driven code review](../quality/code_review.md)
- [Code standards](../quality/code_standards.md)
- [Documentation standards](../quality/documentation_standards.md)
- [API stability and compatibility](../architecture/api_stability.md)
- [Security and supply chain](../quality/security_and_supply_chain.md)
- [Release and packaging](../quality/release_and_packaging.md)
- [Release evidence index](../release/README.md)
- [0.1.0 release evidence ledger](../release/0.1.0.md)
- [0.1.0 wheelhouse dry-run evidence](../release/0.1.0-wheelhouse-dry-run.md)
- [0.1.0-rc2 release evidence ledger](../release/0.1.0-rc2.md)
- [0.1.0-rc1 release evidence ledger](../release/0.1.0-rc1.md)
- [Release support matrix](../release/support_matrix.md)
- [Expectation values guide](../user/expectation_values.md)
- [Performance guide](../user/performance.md)
- [Contributing](https://github.com/sghowell/wolfgang/blob/main/CONTRIBUTING.md)
- [Roadmap](../roadmap.md)
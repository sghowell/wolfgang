# ROCm Next Waves Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the first MI300X ROCm/HIP bring-up into a measured, production-quality ROCm development track with explicit waves for resident results, HIP algorithms, interop, portability, and release readiness.

**Architecture:** ROCm/HIP remains source-build-only and mutually exclusive with CUDA under the accepted target-specific accelerator policy. Backend-Neutral Accelerator Campaign 9 implements and validates shared API semantics across separate CPU, CUDA, HIP, and future Apple targets rather than requiring a mixed CUDA+HIP binary. Each wave must preserve CPU-only import/build behavior, reuse scalar CPU semantics as the oracle, and add public API only after its lifetime, synchronization, ownership, benchmark, and documentation contracts are accepted. MI300X ROCm optimization Campaign 7 is complete and retains a repeatable MI300X source-build release-support lane; ROCm Campaign 8 is complete as the latest architecture-readiness campaign and turns the remaining portability, packaging, profiler, interop, multi-GPU, simultaneous CUDA+HIP, and targeted-performance questions into explicit gates. Backend-Neutral Accelerator Campaign 9 is planned as the implementation handoff for the target-specific backend-neutral accelerator API trigger.

**Tech Stack:** C++20, nanobind, scikit-build-core, CMake HIP language support, ROCm/HIP, AMD Instinct MI300X `gfx942`, rocprof, pytest, NumPy, existing FastPauli CPU/CUDA benchmark-report infrastructure.

---

## Baseline

The first six MI300X ROCm/HIP campaigns are complete:

```text
bring-up plan: docs/plans/mi300x_rocm_bringup_plan.md
bring-up report: docs/benchmarks/reports/rocm_mi300x_bringup_2026-04-29.md
campaign 2 plan: docs/plans/mi300x_rocm_optimization_campaign2_plan.md
campaign 2 report: docs/benchmarks/reports/rocm_mi300x_campaign2_2026-04-30.md
campaign 3 plan: docs/plans/mi300x_rocm_optimization_campaign3_plan.md
campaign 3 report: docs/benchmarks/reports/rocm_mi300x_campaign3_2026-04-30.md
campaign 4 plan: docs/plans/mi300x_rocm_optimization_campaign4_plan.md
campaign 4 report: docs/benchmarks/reports/rocm_mi300x_campaign4_2026-04-30.md
campaign 5 plan: docs/plans/mi300x_rocm_optimization_campaign5_plan.md
campaign 5 report: docs/benchmarks/reports/rocm_mi300x_campaign5_2026-04-30.md
campaign 6 plan: docs/plans/mi300x_rocm_optimization_campaign6_plan.md
campaign 6 report: docs/benchmarks/reports/rocm_mi300x_campaign6_2026-04-30.md
campaign 7 plan: docs/plans/mi300x_rocm_optimization_campaign7_plan.md
campaign 7 report: docs/benchmarks/reports/rocm_mi300x_campaign7_2026-04-30.md
campaign 8 plan: docs/plans/mi300x_rocm_optimization_campaign8_plan.md
campaign 8 report: docs/benchmarks/reports/rocm_campaign8_architecture_readiness_2026-05-01.md
backend-neutral campaign 9 plan: docs/plans/backend_neutral_accelerator_campaign9_plan.md
backend contract: docs/architecture/rocm_backend.md
accepted claim: source-build MI300X evidence for HIP metadata, transfers, pairwise commutation, HIP device-resident commutation matrices, dense host materialization, compact count/conflict consumers, HIP DevicePauliSum.simplify() including retained generic multi-word reduce_by_key, HIP DevicePauliSum.expectation_statevector() for host NumPy complex statevectors, and HIP DevicePauliSum.matmul()
```

Campaign 3 is the first retained sparse-output HIP Pauli operation. It added
HIP `DevicePauliSum.simplify()` with CPU-equivalent canonical semantics,
MI300X benchmark/profiler evidence, and a terminal-status table for Campaign 2
deferred items instead of leaving DLPack, streams, workspaces, packed summaries,
expectation, matmul, portability, packaging, or simultaneous CUDA+HIP as
informal follow-up.

Campaign 4 completed the private HIP simplify performance-hardening pass. It
retains the parallel generic multi-word sorted-index `reduce_by_key` path,
records custom packed-key one-word/two-word probes as unavailable because no
distinct lower-level implementation was retained or timed, records
rocPRIM/hipCUB workspace scratch probes as unavailable for the current
rocThrust boundary, and does not add public HIP APIs.

Campaign 5 completed the first Wave 4 public-boundary pass. It rejected public
HIP DLPack because PyTorch ROCm consumed the candidate versioned `kDLROCM`
capsule in a temporary candidate probe but accepted mutation of the read-only
view. It also rejected HIP CUDA
Array Interface, public streams, public graphs, and public workspaces with
evidence, and assigned terminal next-campaign or out-of-scope statuses to HIP
expectation, HIP matmul, portability, ROCm wheels, multi-GPU ROCm, and
simultaneous CUDA+HIP source builds at the Campaign 5 boundary.

Campaign 6 completed the Wave 4 follow-up for retained operations. It promotes
the existing HIP `DevicePauliSum.expectation_statevector()` and
`DevicePauliSum.matmul()` public methods on MI300X without reopening HIP
DLPack, HIP CUDA Array Interface, public streams, graphs, workspaces, ROCm
wheels, multi-GPU ROCm, portability, or simultaneous CUDA+HIP source-build
surfaces.

Campaign 7 completed the Wave 5 MI300X release-support pass. It keeps ROCm/HIP
source-build-only, validates CPU-only control plus MI300X `gfx942` HIP source
builds, retains the release-runbook command lane, captures rocprof trace/stats
for representative retained operations, records duplicate-pressure simplify and
matmul rows without accepting new kernels or public APIs, keeps ROCm wheels
unavailable, and blocks broader AMD portability until a non-MI300X AMD GPU lane
is available.

Campaign 8 is complete as the first Wave 6 architecture-readiness pass. It did
not add new HIP kernels, public APIs, ROCm wheels, multi-GPU runtime behavior,
or simultaneous CUDA+HIP source builds. It defines the
backend-neutral accelerator object model, non-MI300X AMD portability gate, ROCm
wheel packaging gate, rocprofv3 migration lane, external HIP statevector and
DLPack reconsideration contracts, multi-GPU ROCm gate, simultaneous CUDA+HIP
gate, and profiler-backed targeted-performance reopen gate.

Backend-Neutral Accelerator Campaign 9 completes the first implementation
handoff from the Campaign 8 backend-neutral trigger, with closeout evidence in
`docs/benchmarks/reports/backend_neutral_accelerator_campaign9_2026-05-01.md`.
It covers shared accelerator status, explicit initial device-construction
selectors, omitted-backend ambiguity policy, `DeviceCommutationMatrix.backend`,
disjoint CUDA/HIP/stub source-set declarations, simulated backend/device
validation, benchmark-boundary metadata, H100 CUDA-only validation, MI300X
HIP-only validation, and configure-time rejection of mixed CUDA+HIP requests.
It does not claim ROCm wheels, non-MI300X AMD portability, HIP DLPack, HIP CUDA
Array Interface, multi-GPU ROCm, Metal/MPS, combined accelerator wheels, or a
normal build that links both CUDA and HIP runtimes.

## Wave Map

```text
Wave 2: MI300X resident commutation outputs and compact consumers
Wave 3: HIP simplify and duplicate-reduction algorithms
Wave 4: HIP interop, reusable workspaces, and stream decisions
Wave 5: ROCm portability, CI, packaging, and release-support evidence
Wave 6: backend-neutral multi-accelerator design, multi-GPU ROCm, and Metal/MPS handoff
```

Wave numbers continue from the completed ROCm/HIP bring-up. A wave may be
split into several implementation campaigns when a public API or performance
claim would otherwise become too broad for one reviewable slice.

## Cross-Wave Rules

Every ROCm wave must satisfy these rules:

```text
CPU-only builds keep FASTPAULI_ENABLE_HIP=OFF and do not require ROCm headers, libraries, devices, or environment variables
CUDA behavior remains unchanged unless a wave explicitly updates the shared accelerator contract and validates CUDA
FASTPAULI_ENABLE_CUDA=ON with FASTPAULI_ENABLE_HIP=ON remains a configure-time error under the target-specific accelerator build policy
public headers do not include HIP or ROCm runtime headers
HIP public APIs either match existing CUDA API semantics exactly or document a narrower HIP-specific contract before exposure
every implemented HIP operation has deterministic CPU/HIP equivalence tests and randomized fixed-seed equivalence tests
benchmark reports include CPU scalar, available optimized CPU selectors, CUDA where available, HIP transfer-inclusive rows, HIP device-resident rows, and unavailable external-baseline reasons
rocprof trace, stats, counters, or an explicit provider/tooling diagnosis is captured for every HIP performance claim
README and user docs distinguish source-build evidence from wheel or release-support claims
```

## Wave 2: Resident Commutation Outputs

Status:

```text
complete
docs/plans/mi300x_rocm_optimization_campaign2_plan.md
docs/benchmarks/reports/rocm_mi300x_campaign2_2026-04-30.md
```

Purpose:

```text
remove dense host output materialization from the hot HIP commutation path when callers need device-resident or compact summary results
```

Expected retained surfaces:

```text
HIP-backed DeviceCommutationMatrix allocation and RAII lifetime
DevicePauliSum.commutes_with_device() on HIP-only builds
DevicePauliSum.commutes_with_device(..., output=existing_matrix) reuse path on HIP-only builds
DeviceCommutationMatrix.to_host() on HIP-only builds
DeviceCommutationMatrix.count_commuting(axis=None|0|1) on HIP-only builds
DeviceCommutationMatrix.conflict_degrees(axis=None|0|1) on HIP-only builds
```

Out of scope for Wave 2:

```text
HIP DLPack or ROCm array interop
public HIP stream handles
public external workspace handles
HIP simplify, expectation, or matmul kernels
multi-GPU MI300X execution
ROCm wheels
simultaneous CUDA+HIP builds
```

Exit evidence:

```text
MI300X source build succeeds with FASTPAULI_ENABLE_HIP=ON and FASTPAULI_HIP_ARCHITECTURES=gfx942
local CPU-only validation passes with HIP disabled
HIP device-output commutation tests pass on edge, randomized, empty, invalid-device, wrong-shape, wrong-device, and reuse cases
rocprof evidence shows separate kernel fill, compact consumer, and host materialization boundaries
benchmark tables compare host-output HIP, device-output allocating HIP, device-output reused HIP, compact HIP consumers, CPU scalar, optimized CPU selectors, and latest relevant CUDA rows
```

## Wave 3: HIP Simplify And Duplicate Reduction

Executable plan:

```text
docs/plans/mi300x_rocm_optimization_campaign3_plan.md
```

Executable follow-up plan:

```text
docs/plans/mi300x_rocm_optimization_campaign4_plan.md
```

Status:

```text
complete
docs/benchmarks/reports/rocm_mi300x_campaign3_2026-04-30.md
docs/benchmarks/reports/rocm_mi300x_campaign4_2026-04-30.md
```

Purpose:

```text
add the first sparse-output HIP Pauli operation, decide whether rocThrust, hipCUB, or a custom packed-key duplicate-reduction path is justified, and give every Campaign 2 residual risk a terminal status
```

Required decisions before implementation:

```text
temporary storage ownership model for HIP sort/reduction scratch
packed-key support bounds for one-word and multi-word operators
tolerance and zero-coefficient filtering location
library dependency policy for rocThrust or hipCUB
benchmark dimensions that separate duplicate-heavy, duplicate-light, and wide-qubit regimes
```

Candidate retained surfaces:

```text
DevicePauliSum.simplify() on HIP-only builds
private scratch probes for rocThrust and explicit unavailable status for non-retained custom reduction
optional private HIP workspace if allocation pressure dominates and public workspace API remains deferred
README broad performance landscape refresh with ROCm rows preserved alongside CPU, CUDA, and external baseline rows
```

Retained surfaces:

```text
HIP DevicePauliSum.simplify() with device-resident output
rocThrust duplicate reduction as the production path
parallel generic multi-word reduce_by_key for HIP simplify
Campaign 4 README landscape with CPU, CUDA, ROCm, and external rows
```

Exit evidence:

```text
CPU/HIP simplify equivalence across canonical ordering, tolerance, empty output, one-word, two-word, duplicate-heavy, and randomized cases
benchmark evidence against scalar CPU, optimized CPU where relevant, CUDA simplify evidence where comparable, and unavailable ROCm external baselines
rocprof counters for retained kernels and library calls, including allocation/materialization attribution
rejected-experiment table for non-retained rocThrust, hipCUB, or custom variants
terminal-status table covering DLPack, streams, workspaces, packed summaries, expectation, matmul, portability, ROCm wheels, multi-GPU, and simultaneous CUDA+HIP source builds
```

Remaining measured headroom after Wave 3 and Campaign 4, before Campaign 5
interop decisions and Campaign 6 parity retention:

```text
public HIP DLPack/consumer interop only with a named PyTorch ROCm or CuPy ROCm consumer and accepted ownership/stream contracts
public HIP stream or graph execution only after explicit lifetime, error-propagation, and synchronization contracts
public HIP workspace handles only after an API design shows measurable benefit beyond the rejected private rocThrust scratch probes
HIP expectation and HIP matmul only after CPU/CUDA parity fixtures are promoted to HIP; this item was closed by Campaign 6
ROCm portability, CI, and packaging evidence beyond single-host MI300X gfx942 source builds
backend-neutral multi-accelerator design before simultaneous CUDA+HIP builds or multi-GPU ROCm claims
custom lower-level rocPRIM/hipCUB duplicate reduction only if future profiling identifies an explicit temporary-storage bottleneck not solved by the retained generic reduce_by_key path
```

Campaign 4 closed the private simplify performance items, Campaign 5 closed
the immediate interop and execution-control decisions, and Campaign 6 closed
the HIP expectation and matmul parity item. Remaining ROCm work now needs
explicit release-support, portability, or backend-neutral API plans before
exposure.

## Wave 4: HIP Interop, Workspaces, And Streams

Executable plan:

```text
docs/plans/mi300x_rocm_optimization_campaign5_plan.md
```

Status:

```text
complete
```

Purpose:

```text
decide whether ROCm Python interop and execution-control APIs are worth exposing after resident HIP outputs exist
```

Required decision documents:

```text
HIP DLPack ownership and stream contract
HIP Python consumer baseline contract for PyTorch ROCm and CuPy ROCm when available
HIP reusable workspace contract or explicit rejection
HIP stream/graph contract or explicit rejection
```

Candidate retained surfaces:

```text
read-only DeviceCommutationMatrix.__dlpack__ for HIP with kDLROCM when consumer support is validated
PyTorch ROCm DLPack consumer benchmark rows
CuPy ROCm DLPack consumer benchmark rows when the package is available on the host
private or public HIP workspace reuse only after allocation costs are measured as dominant
```

Exit evidence:

```text
PyTorch ROCm consumer availability plus rejected read-only mutation evidence
HIP DLPack unavailable tests because no real consumer enforced read-only mutation rejection
benchmark rows labeled framework-consumer, transfer-inclusive primitive, device-resident primitive, or compact-consumer boundary
explicit keep-or-reject status for streams, graph replay, and public workspace exposure
```

Campaign 5 report:

```text
docs/benchmarks/reports/rocm_mi300x_campaign5_2026-04-30.md
```

Campaign 6 follow-up:

```text
complete
docs/plans/mi300x_rocm_optimization_campaign6_plan.md
docs/benchmarks/reports/rocm_mi300x_campaign6_2026-04-30.md
```

## Wave 4 Follow-Up: HIP Expectation And Matmul Parity

Executable plan:

```text
docs/plans/mi300x_rocm_optimization_campaign6_plan.md
```

Status:

```text
complete
docs/benchmarks/reports/rocm_mi300x_campaign6_2026-04-30.md
```

Purpose:

```text
promote existing DevicePauliSum.expectation_statevector() and DevicePauliSum.matmul() methods from HIP-unavailable to HIP-supported on MI300X without adding new public API shape
```

In-scope retained surfaces:

```text
DevicePauliSum.expectation_statevector(host NumPy complex64 or complex128) on HIP-only builds
DevicePauliSum.matmul(rhs, simplify=True|False, max_intermediate_terms=...) on HIP-only builds
```

Out of scope for this follow-up:

```text
HIP external statevector device pointers
HIP DLPack
HIP CUDA Array Interface
public streams
public graphs
public workspaces
ROCm wheels
multi-GPU ROCm
additional AMD GPU support claims
simultaneous CUDA+HIP source builds
```

Exit evidence:

```text
CPU/HIP expectation equivalence for complex64, complex128, empty, identity, diagonal, off-diagonal, duplicate, randomized, and invalid-input cases
CPU/HIP matmul equivalence for simplify=True, simplify=False, one-word, multi-word, empty, phase, guardrail, and randomized cases
MI300X benchmark rows for expectation and matmul with CPU scalar, available optimized CPU selectors, HIP transfer-inclusive timing, HIP device-resident timing where measurable, and explicit materialization boundaries
rocprof trace/stats evidence for retained HIP expectation and matmul kernels
README broad performance landscape refreshed only as a CPU/CUDA/ROCm/external view
terminal-status table covering external statevector pointers, DLPack, CUDA Array Interface guard, streams, graphs, workspaces, portability, ROCm wheels, multi-GPU, and simultaneous CUDA+HIP
```

Result:

```text
complete
HIP expectation and HIP matmul retained on MI300X gfx942
external statevector device pointers remain unavailable
HIP DLPack, HIP CUDA Array Interface, public streams, graphs, and workspaces remain rejected with evidence
portability, ROCm wheels, multi-GPU ROCm, and simultaneous CUDA+HIP remain Wave 5 or Wave 6 work
```

## Wave 5: ROCm Portability, CI, Packaging, And Release Claims

Executable plan:

```text
docs/plans/mi300x_rocm_optimization_campaign7_plan.md
```

Status:

```text
complete
docs/benchmarks/reports/rocm_mi300x_campaign7_2026-04-30.md
```

Purpose:

```text
convert single-host source-build evidence into a responsible support matrix without overstating ROCm wheel or GPU coverage
```

Required decisions before release-support wording changes:

```text
minimum ROCm toolkit line and compiler baseline
supported AMD GPU architecture list and evidence level for each architecture
Linux distribution and kernel assumptions
source distribution build policy for HIP users
ROCm CI runner strategy or documented local release-lane substitute
wheel policy, including whether ROCm wheels are rejected, deferred, or scoped to a separate packaging channel
```

Candidate hardware lanes:

```text
MI300X gfx942 repeatability lane
MI250X or another gfx90a-class lane when access exists
another MI300-series lane when release wording needs more than one gfx942-style host
```

Exit evidence:

```text
per-host inventory, source build, runtime tests, benchmark smoke, and profiler availability status
docs/quality/release_and_packaging.md updated with ROCm release wording
README support table distinguishes runtime-tested, performance-tested, and release-supported ROCm states
CI or release-runbook instructions state exactly how ROCm evidence is refreshed
terminal statuses are recorded for external HIP statevector interop, HIP DLPack, HIP CUDA Array Interface, public streams, public graphs, public workspaces, multi-GPU ROCm, simultaneous CUDA+HIP, and backend-neutral accelerator design
```

Result:

```text
complete
MI300X gfx942 source-build release-support lane retained
ROCm wheels remain unavailable
additional AMD GPU portability remains blocked_external until hardware access exists
external HIP statevector interop, HIP DLPack, HIP CUDA Array Interface, public streams, public graphs, public workspaces, multi-GPU ROCm, simultaneous CUDA+HIP, and backend-neutral accelerator design remain rejected, unavailable, or out of scope with terminal statuses
```

## Wave 6: Backend-Neutral And Long-Horizon Accelerator Work

Executable plan:

```text
docs/plans/mi300x_rocm_optimization_campaign8_plan.md
docs/plans/backend_neutral_accelerator_campaign9_plan.md
```

Status:

```text
campaign 8 complete
campaign 9 planned
docs/benchmarks/reports/rocm_campaign8_architecture_readiness_2026-05-01.md
```

Purpose:

```text
unlock work that should not be forced into the early HIP source-build track
```

Candidate design tracks:

```text
backend-neutral DevicePauliSum and DeviceCommutationMatrix model for target-specific accelerator builds
multi-GPU ROCm execution and distributed MI300X workflows
Metal/MPS design for Apple GPU exploration
cross-backend benchmark dashboards that show CPU, CUDA, ROCm, and external baselines from checked evidence
ROCm wheel packaging design and non-MI300X AMD portability support-matrix gates
rocprofv3 migration from legacy rocprof evidence capture
external HIP statevector and HIP DLPack reconsideration contracts
targeted ROCm performance reopen only after profiler-backed retained-operation bottlenecks
```

This wave changes object-model and packaging assumptions. Campaign 8 started it
with architecture decision documents, tests, dry-run evidence lanes, and
terminal statuses before implementation claims. Later executable ROCm campaigns
must satisfy one of the Campaign 8 trigger gates before adding public APIs,
packaging claims, portability claims, multi-GPU behavior, simultaneous CUDA+HIP
source-build support, or targeted ROCm performance work.

## Planning Acceptance

This multi-wave plan is accepted only when:

```text
docs/roadmap.md points to this document as the latest ROCm plan
docs/architecture/rocm_backend.md names the wave progression and public-boundary rules
docs/benchmarks/protocol.md defines ROCm device-output, simplify, and simplify-hardening timing fields
scripts/validate.py treats this document and the latest completed ROCm readiness or executable campaign as source-of-truth docs
README planning sources link the new plan
git diff --check passes
```

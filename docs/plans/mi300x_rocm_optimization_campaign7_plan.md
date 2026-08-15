# MI300X ROCm Optimization Campaign 7 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Convert the MI300X ROCm/HIP source-build lane from single-host kernel evidence into reproducible release-support evidence without claiming ROCm wheels, broad AMD portability, external HIP statevector interop, multi-GPU ROCm, or simultaneous CUDA+HIP support.

**Architecture:** Campaign 7 is a Wave 5 campaign. It should not add new public runtime APIs or new HIP kernels unless a validation bug is found while building the release-support lane. The retained work is a repeatable ROCm source-build runbook, benchmark/profile evidence schema, packaging policy, README support wording, and terminal statuses for portability and long-horizon accelerator items.

**Tech Stack:** C++20, nanobind, scikit-build-core, CMake HIP language support, ROCm/HIP, AMD Instinct MI300X `gfx942`, rocprof, pytest, NumPy, existing FastPauli CPU/CUDA/ROCm benchmark-report infrastructure.

---

## Status

```text
complete
wave: Wave 5 ROCm portability, CI, packaging, and release-support evidence
previous campaign report: docs/benchmarks/reports/rocm_mi300x_campaign6_2026-04-30.md
evidence root: docs/benchmarks/data/rocm_mi300x_campaign7_2026-04-30/
report: docs/benchmarks/reports/rocm_mi300x_campaign7_2026-04-30.md
```

If execution occurs on a later date, update the evidence root, report path, and
all references in the same implementation slice before running the campaign.

## Campaign 7 Decision

Campaign 7 closes the Campaign 6 remaining-headroom list by deciding what
FastPauli can responsibly claim for ROCm/HIP release support after MI300X
kernel parity is retained.

In scope:

```text
repeatable MI300X source-build validation lane for FASTPAULI_ENABLE_HIP=ON and FASTPAULI_HIP_ARCHITECTURES=gfx942
local CPU-only control validation with FASTPAULI_ENABLE_HIP=OFF
ROCm release-runbook or release-lane script that captures commands, environment, artifacts, and terminal statuses
ROCm packaging policy stating that ROCm wheels are not supported until a separate packaging channel is designed and tested
README support table wording that separates source-build evidence from wheel support and broad AMD portability claims
benchmark smoke rows for retained HIP transfers, commutation, compact consumers, simplify, expectation, and matmul
profiler availability and rocprof evidence for the representative retained HIP operations
larger duplicate-pressure simplify and matmul rows that determine whether retained Campaign 6 operations expose a new bottleneck
alternate AMD GPU portability lane only when an AMD GPU outside the current MI300X host is available
terminal statuses for external HIP statevector interop, portability, ROCm wheels, multi-GPU ROCm, simultaneous CUDA+HIP, and backend-neutral accelerator design
```

Hard out of scope:

```text
new public Python methods or arguments
HIP external statevector device-pointer support
HIP DLPack retention
HIP CUDA Array Interface exposure
public HIP streams
public HIP graph replay
public HIP workspace handles
ROCm binary wheels
multi-GPU ROCm execution
simultaneous CUDA+HIP source builds
Metal/MPS implementation
new AMD GPU support claims without source-build and runtime evidence from that GPU
```

## Terminal Status Model

The Campaign 7 report must use these final-status values:

```text
passed
retained
rejected_with_evidence
blocked_external
unavailable
out_of_scope_with_next_trigger
```

Required terminal-status keys:

```text
mi300x_repeatability
cpu_only_control
rocm_source_build_runbook
rocm_ci_or_release_lane
rocm_packaging_policy
rocm_wheel_support
alternate_amd_gpu_portability
profiler_availability
duplicate_pressure_simplify
duplicate_pressure_matmul
external_statevector_interop
hip_dlpack
hip_cuda_array_interface
public_streams
public_graphs
public_workspaces
multi_gpu_rocm
simultaneous_cuda_hip
backend_neutral_accelerator_design
```

The report may not use a soft "deferred" status. If hardware or provider
access is missing, use `blocked_external` and include the command or access
attempt that proved the blocker.

## Evidence Layout

Campaign 7 execution must write evidence under one root:

```text
docs/benchmarks/data/rocm_mi300x_campaign7_2026-04-30/
docs/benchmarks/data/rocm_mi300x_campaign7_2026-04-30/logs/
docs/benchmarks/data/rocm_mi300x_campaign7_2026-04-30/raw/
docs/benchmarks/data/rocm_mi300x_campaign7_2026-04-30/profiler/
docs/benchmarks/data/rocm_mi300x_campaign7_2026-04-30/summary.json
docs/benchmarks/reports/rocm_mi300x_campaign7_2026-04-30.md
docs/benchmarks/plots/rocm_mi300x_campaign7_release_support.svg
docs/benchmarks/plots/accelerator_landscape_with_rocm.svg
```

The report must state the exact commit used for MI300X execution. If later
documentation commits do not change HIP implementation code, the report must
say so explicitly.

## Acceptance Criteria

Campaign 7 is complete only when every item below has evidence or a terminal
status:

```text
local CPU-only validation passes with FASTPAULI_ENABLE_HIP=OFF
MI300X HIP source build succeeds with FASTPAULI_ENABLE_HIP=ON and FASTPAULI_HIP_ARCHITECTURES=gfx942
CUDA+HIP configure-time rejection remains validated
public headers still include no HIP or ROCm runtime headers
retained HIP operation tests pass on MI300X for transfers, commutation, device-output consumers, simplify, expectation, and matmul
Campaign 7 benchmark JSON includes release-support protocol fields
representative retained HIP benchmark rows include transfer-inclusive and resident or compact-consumer boundaries where applicable
duplicate-pressure simplify and matmul rows either identify a concrete next bottleneck or close with rejected_with_evidence
rocprof trace/stats are captured, or profiler unavailability has a precise provider/tooling blocker
ROCm release-runbook or release-lane script records exact commands, environment, artifacts, and expected outputs
release and packaging docs state the source-build-only ROCm policy and wheel policy
README support wording separates CPU wheels, CUDA source-build evidence, ROCm source-build evidence, and unsupported accelerator candidates
alternate AMD GPU portability is passed only with real source-build/runtime evidence; otherwise it is blocked_external with evidence
external statevector interop, HIP DLPack, HIP CUDA Array Interface, streams, graphs, workspaces, multi-GPU ROCm, simultaneous CUDA+HIP, and backend-neutral accelerator design have terminal statuses
README broad performance landscape remains a CPU/CUDA/ROCm/external view
independent review is recorded before merge
```

## Task 1: Protocol, Source-Of-Truth, And Red Plan Tests

**Files:**
- Modify: `docs/benchmarks/protocol.md`
- Modify: `scripts/validate.py`
- Create: `tests/test_rocm_campaign7_plan.py`
- Test: `uv run python -m pytest tests/test_rocm_campaign7_plan.py tests/test_validate_entrypoint.py -q`

- [x] **Step 1: Add the Campaign 7 benchmark protocol section**

Add a `ROCm Campaign 7 release-support rows` section to
`docs/benchmarks/protocol.md` requiring these fields:

```text
campaign: rocm_mi300x_campaign7
operation: release_source_build, runtime_validation, retained_operation_smoke, profiler_smoke, duplicate_pressure_probe, portability_lane, packaging_decision, ci_runbook, or backend_neutral_decision
backend: hip, cpu, cuda, or none
mode: mi300x_repeatability, cpu_only_control, cuda_hip_rejection, retained_transfer, retained_commutation, retained_device_consumers, retained_simplify, retained_expectation, retained_matmul, simplify_duplicate_pressure, matmul_duplicate_pressure, rocprof_availability, alternate_amd_gpu_probe, rocm_wheel_policy, release_runbook, external_statevector_decision, multi_gpu_decision, or simultaneous_cuda_hip_decision
status: ok, passed, retained, rejected_with_evidence, blocked_external, unavailable, or out_of_scope_with_next_trigger
final_status: passed, retained, rejected_with_evidence, blocked_external, unavailable, or out_of_scope_with_next_trigger
host_role: local_cpu_control, primary_mi300x, alternate_amd_gpu, or decision_only
rocm_runtime_version
rocm_toolkit_version
hip_compiler_version
gpu_name
gfx_target
build_command
validation_command
profiler_command
timing_boundary: transfer_inclusive, device_resident, compact_consumer, device_output_to_host, source_build, validation_only, profiler_only, decision_only, or benchmark_only
correctness_digest
campaign7_terminal_statuses
```

- [x] **Step 2: Add Campaign 7 to source-of-truth validation**

Add `docs/plans/mi300x_rocm_optimization_campaign7_plan.md` to
`SOURCE_OF_TRUTH_PATHS` in `scripts/validate.py`.

- [x] **Step 3: Add red plan tests**

Create `tests/test_rocm_campaign7_plan.py` with checks that the plan is wired
into the roadmap, README, ROCm wave plan, backend architecture, release policy,
benchmark protocol, and validation source list. The test should assert the
required terminal-status keys so future docs cannot silently drop the Campaign
6 residuals.

- [x] **Step 4: Run targeted tests**

Run:

```bash
uv run python -m pytest tests/test_rocm_campaign7_plan.py tests/test_validate_entrypoint.py -q
```

Expected result:

```text
all selected tests pass
```

## Task 2: Release Lane Harness And Benchmark Profiles

**Files:**
- Create: `scripts/run_rocm_release_support_lane.py`
- Modify: `benchmarks/bench_rocm_kernels.py`
- Test: `uv run python -m pytest tests/test_phase12_rocm_foundation.py tests/test_rocm_campaign7_plan.py -q`

- [x] **Step 1: Add a release-support dry-run mode**

Create `scripts/run_rocm_release_support_lane.py` with a dry-run mode that
prints the exact MI300X commands without requiring ROCm locally. The dry-run
must include:

```text
host inventory capture
Python and package metadata capture
CPU-only control validation command
HIP source-build command
HIP pytest command
CUDA+HIP configure-time rejection command
Campaign 7 benchmark commands
rocprof command
asset renderer command
report validation command
```

- [x] **Step 2: Add Campaign 7 benchmark profiles**

Add profiles to `benchmarks/bench_rocm_kernels.py`:

```text
campaign7-release-smoke
campaign7-duplicate-pressure
campaign7-profiler
```

The release-smoke profile should cover retained HIP transfers, pairwise
commutation, device-output compact consumers, simplify, expectation, and
matmul. The duplicate-pressure profile should include one simplify row and one
matmul row large enough to stress the retained rocThrust simplify path without
requiring a full stress run in local validation.

- [x] **Step 3: Add dry-run and profile tests**

Extend existing ROCm tests so CPU-only local validation proves the new profiles
are registered and the release lane can emit commands without a HIP runtime.

- [x] **Step 4: Run targeted tests**

Run:

```bash
uv run python -m pytest tests/test_phase12_rocm_foundation.py tests/test_rocm_campaign7_plan.py -q
```

Expected result on a CPU-only host:

```text
tests for unavailable HIP runtime skip cleanly
profile and dry-run tests pass
```

## Task 3: MI300X Campaign Execution

**Files:**
- Create: `docs/benchmarks/data/rocm_mi300x_campaign7_2026-04-30/`
- Test: MI300X commands from `scripts/run_rocm_release_support_lane.py --print-commands`

- [x] **Step 1: Prepare a clean MI300X checkout**

Use the normal repository checkout on the MI300X host or a clean temporary
checkout under `/tmp`. Record:

```text
git revision
hostname
OS release
Python version
ROCm runtime and driver versions
ROCm toolkit version
HIP compiler version
GPU model and gfx target
CPU model
visible device count
```

- [x] **Step 2: Run the HIP source build and validation lane**

Run the generated HIP source-build and validation commands with:

```text
FASTPAULI_ENABLE_HIP=ON
FASTPAULI_HIP_ARCHITECTURES=gfx942
PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
```

The HIP test command must include `tests/test_phase12_rocm_foundation.py`.

- [x] **Step 3: Run Campaign 7 benchmarks**

Run the release-smoke, duplicate-pressure, and profiler profiles with fixed
warmup and repeat counts. Store raw JSON under:

```text
docs/benchmarks/data/rocm_mi300x_campaign7_2026-04-30/raw/
```

- [x] **Step 4: Run rocprof**

Run `rocprof --hip-trace --stats` over the Campaign 7 profiler profile. Store
CSV, JSON, database, and system-info artifacts under:

```text
docs/benchmarks/data/rocm_mi300x_campaign7_2026-04-30/profiler/
```

If provider permissions block profiler output, record the failed command,
stderr, exit code, and provider limitation in the report.

## Task 4: Portability, Packaging, And Long-Horizon Decisions

**Files:**
- Modify: `docs/quality/release_and_packaging.md`
- Modify: `docs/architecture/rocm_backend.md`
- Modify: `docs/plans/rocm_next_waves_plan.md`
- Test: `git diff --check`

- [x] **Step 1: Lock ROCm packaging policy**

Update `docs/quality/release_and_packaging.md` so ROCm support is described as
source-build evidence only. ROCm wheel support remains unavailable until a
separate package channel documents runtime bundling, supported ROCm versions,
GPU architecture coverage, CI hardware, artifact size, and installation tests.

- [x] **Step 2: Record portability evidence rules**

Update `docs/architecture/rocm_backend.md` and
`docs/plans/rocm_next_waves_plan.md` so additional AMD GPU support requires:

```text
source build on that GPU architecture
runtime status capture
retained HIP operation tests
benchmark smoke
profiler availability status
README wording that distinguishes runtime-tested from performance-tested and release-supported
```

- [x] **Step 3: Close long-horizon items**

The Campaign 7 report must give terminal statuses to:

```text
external HIP statevector interop
HIP DLPack
HIP CUDA Array Interface
public streams
public graphs
public workspaces
multi-GPU ROCm
simultaneous CUDA+HIP
backend-neutral accelerator design
```

These items may not move to retained scope in Campaign 7.

## Task 5: Report, Plots, README, And Roadmap

**Files:**
- Create: `scripts/render_rocm_campaign7_assets.py`
- Create: `docs/benchmarks/reports/rocm_mi300x_campaign7_2026-04-30.md`
- Create: `docs/benchmarks/plots/rocm_mi300x_campaign7_release_support.svg`
- Modify: `docs/benchmarks/plots/accelerator_landscape_with_rocm.svg`
- Modify: `README.md`
- Modify: `docs/roadmap.md`
- Test: `uv run python -m pytest tests/test_rocm_campaign7_plan.py -q`

- [x] **Step 1: Add the Campaign 7 renderer**

Create a renderer that reads Campaign 7 raw JSON and writes:

```text
docs/benchmarks/data/rocm_mi300x_campaign7_2026-04-30/summary.json
docs/benchmarks/plots/rocm_mi300x_campaign7_release_support.svg
docs/benchmarks/plots/accelerator_landscape_with_rocm.svg
```

The README landscape must remain broad across CPU, CUDA, ROCm, and external
baselines. The Campaign 7 report-local plot may focus on release lane coverage
and terminal statuses.

- [x] **Step 2: Write the Campaign 7 report**

The report must include:

```text
scope
host and build inventory
validation commands and observed results
benchmark commands and observed results
profiler artifacts or precise profiler blocker
release and packaging policy outcome
portability lane outcome
duplicate-pressure outcome
terminal-status table
residual risk and next work
```

- [x] **Step 3: Update README and roadmap**

Update README and `docs/roadmap.md` so:

```text
Campaign 7 is the latest ROCm campaign plan and checked ROCm benchmark evidence
Campaign 6 remains linked as the prior retained-operation parity campaign
ROCm support wording says source-build evidence, not wheel support or broad AMD portability
the README broad landscape plot remains visible
```

## Task 6: Review, Validation, Merge, Push, And CI

**Files:**
- Review all modified files in the branch.

- [x] **Step 1: Run local validation**

Run:

```bash
git diff --check
uv run python -m pytest tests/test_rocm_campaign7_plan.py tests/test_validate_entrypoint.py -q
uv run python scripts/validate.py
```

- [x] **Step 2: Request independent review**

Give the reviewer:

```text
branch name
commit list
diff summary
Campaign 7 goal
validation output
known limitations: single MI300X lane, ROCm source-build evidence only, no ROCm wheels, no non-MI300X AMD portability lane
relevant docs: this plan, rocm_next_waves_plan.md, rocm_backend.md, release_and_packaging.md, benchmark protocol
```

- [x] **Step 3: Resolve findings and revalidate**

Resolve every P0/P1 finding. Fix or record P2 findings with a follow-up scope.
Rerun validation after fixes.

- [ ] **Step 4: Complete repository closeout**

Use the standard FastPauli closeout:

```text
stage and commit in a concise docs/planning commit
fast-forward merge to main when possible
validate merged main
push main
confirm CI is green
delete the merged feature branch locally and remotely if it was pushed
```

# ROCm Campaign 8 Architecture-Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the Campaign 7 residual ROCm items into explicit architecture, packaging, portability, profiler, and interop gates before any new ROCm public APIs, wheels, multi-GPU claims, or simultaneous CUDA+HIP builds are attempted.

**Architecture:** Campaign 8 is a Wave 6 planning-and-harness campaign. It should not add HIP kernels, public runtime methods, ROCm wheels, multi-GPU execution, or simultaneous CUDA+HIP builds. The retained output is a set of enforceable decision contracts, dry-run lanes, tests, and documentation updates that define when the next ROCm implementation campaign is allowed to start.

**Tech Stack:** C++20, nanobind, scikit-build-core, CMake CUDA/HIP language configuration, ROCm/HIP, AMD Instinct MI300X `gfx942`, optional non-MI300X AMD GPU lanes, rocprof and rocprofv3, pytest, NumPy, existing FastPauli CPU/CUDA/ROCm benchmark-report infrastructure.

---

## Status

```text
complete
wave: Wave 6 backend-neutral and long-horizon accelerator work
previous campaign report: docs/benchmarks/reports/rocm_mi300x_campaign7_2026-04-30.md
evidence root: docs/benchmarks/data/rocm_campaign8_architecture_readiness_2026-05-01/
report: docs/benchmarks/reports/rocm_campaign8_architecture_readiness_2026-05-01.md
```

If execution occurs on a later date, update the evidence root, report path, and
all references in the same implementation slice before running the campaign.

## Campaign 8 Decision

Campaign 8 should address every Campaign 7 residual-risk item that can be
resolved by architecture, packaging, portability, profiler, or interop
contracts. It is not a same-host MI300X performance rerun.

In scope:

```text
backend-neutral accelerator object-model decision for CUDA-only, HIP-only, and future multi-backend builds
explicit decision on whether simultaneous CUDA+HIP source builds remain unavailable or move to a designed implementation campaign
ROCm multi-GPU decision contract covering device ordinals, same-device guards, cross-device rejection, and future peer-copy ownership
non-MI300X AMD GPU portability lane specification, including exact evidence required when a host is available
ROCm wheel packaging design gate covering package channels, runtime policy, clean-machine install tests, and CI hardware requirements
rocprofv3 migration lane with side-by-side legacy rocprof evidence when ROCm 7.x tooling is available
external HIP statevector and HIP DLPack reconsideration contracts, including ownership, read-only behavior, streams, and accepted consumers
targeted ROCm performance reopen gate that requires profiler evidence before kernel work resumes
dry-run command inventory for all Campaign 8 lanes so local validation can verify command shape without ROCm hardware
terminal statuses for all Campaign 7 residual items
```

Hard out of scope:

```text
new public Python methods or arguments
new HIP kernels
HIP external statevector device-pointer implementation
HIP DLPack implementation
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

The Campaign 8 report must use these final-status values:

```text
accepted_for_future_implementation
retained
rejected_with_evidence
blocked_external
unavailable
out_of_scope_with_next_trigger
```

Required terminal-status keys:

```text
backend_neutral_object_model
simultaneous_cuda_hip_source_builds
multi_gpu_rocm_execution
non_mi300x_amd_portability
rocm_wheel_packaging_design
rocm_ci_hardware_policy
rocm_clean_machine_install_tests
rocprofv3_migration
legacy_rocprof_retention
external_hip_statevector_contract
hip_dlpack_reconsideration_contract
hip_cuda_array_interface_policy
public_streams_policy
public_graphs_policy
public_workspaces_policy
targeted_rocm_performance_reopen
source_build_release_lane_retention
```

The report may not use a soft unresolved status. If hardware, packaging
infrastructure, or provider support is missing, use `blocked_external` and
record the command, access attempt, or policy dependency that proves the
blocker.

## Evidence Layout

Campaign 8 execution must write evidence under one root:

```text
docs/benchmarks/data/rocm_campaign8_architecture_readiness_2026-05-01/
docs/benchmarks/data/rocm_campaign8_architecture_readiness_2026-05-01/logs/
docs/benchmarks/data/rocm_campaign8_architecture_readiness_2026-05-01/raw/
docs/benchmarks/data/rocm_campaign8_architecture_readiness_2026-05-01/profiler/
docs/benchmarks/data/rocm_campaign8_architecture_readiness_2026-05-01/summary.json
docs/benchmarks/reports/rocm_campaign8_architecture_readiness_2026-05-01.md
docs/benchmarks/plots/accelerator_landscape_with_rocm.svg
```

The report must state whether Campaign 8 changed runtime code. If the campaign
only adds contracts, scripts, tests, and documentation, the report must say that
no HIP kernel, public Python API, wheel, or multi-GPU behavior changed.

## Acceptance Criteria

Campaign 8 is complete only when every item below has evidence or a terminal
status:

```text
local CPU-only validation passes with FASTPAULI_ENABLE_HIP=OFF
CUDA+HIP configure-time rejection remains validated or a new backend-neutral design is explicitly accepted for a later implementation campaign
public headers still include no HIP or ROCm runtime headers
backend-neutral object-model decision covers ownership, backend identity, device ordinal semantics, status reporting, error classes, and packaging impact
ROCm multi-GPU decision covers same-device enforcement, cross-device rejection, copy semantics, and benchmark boundaries
non-MI300X AMD portability lane is executable when hardware exists and blocked_external with evidence when hardware does not exist
ROCm wheel packaging design states package channel, runtime dependency policy, CI hardware, clean-machine install tests, manylinux policy, and support-matrix wording
rocprofv3 migration lane captures side-by-side command shape and either evidence artifacts or a precise tooling blocker
external HIP statevector and HIP DLPack reconsideration contracts state ownership, stream, read-only, consumer-library, and mutation-test requirements
targeted ROCm performance reopen gate requires profiler evidence tied to a retained operation and rejects same-host reruns without a concrete bottleneck
README, roadmap, ROCm backend architecture, release packaging docs, and benchmark protocol link the Campaign 8 plan
scripts/validate.py treats this plan as a source-of-truth document
independent review is recorded before merge
```

## Task 1: Campaign 8 Source-Of-Truth And Red Plan Tests

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/plans/rocm_next_waves_plan.md`
- Modify: `docs/architecture/rocm_backend.md`
- Modify: `docs/benchmarks/protocol.md`
- Modify: `docs/quality/release_and_packaging.md`
- Modify: `scripts/validate.py`
- Create: `tests/test_rocm_campaign8_plan.py`
- Test: `uv run python -m pytest tests/test_rocm_campaign8_plan.py tests/test_validate_entrypoint.py -q`

- [x] **Step 1: Register Campaign 8 as the latest ROCm plan**

  Update the latest ROCm campaign references to:

  ```text
  docs/plans/mi300x_rocm_optimization_campaign8_plan.md
  ```

  Keep Campaign 7 as the latest completed ROCm report until Campaign 8
  execution publishes a report.

- [x] **Step 2: Add Campaign 8 to validation source documents**

  Add this path to `SOURCE_OF_TRUTH_PATHS` in `scripts/validate.py`:

  ```text
  docs/plans/mi300x_rocm_optimization_campaign8_plan.md
  ```

- [x] **Step 3: Add red plan tests**

  Create `tests/test_rocm_campaign8_plan.py` with checks for:

  ```python
  REQUIRED_TERMINAL_KEYS = {
      "backend_neutral_object_model",
      "simultaneous_cuda_hip_source_builds",
      "multi_gpu_rocm_execution",
      "non_mi300x_amd_portability",
      "rocm_wheel_packaging_design",
      "rocm_ci_hardware_policy",
      "rocm_clean_machine_install_tests",
      "rocprofv3_migration",
      "legacy_rocprof_retention",
      "external_hip_statevector_contract",
      "hip_dlpack_reconsideration_contract",
      "hip_cuda_array_interface_policy",
      "public_streams_policy",
      "public_graphs_policy",
      "public_workspaces_policy",
      "targeted_rocm_performance_reopen",
      "source_build_release_lane_retention",
  }
  ```

  The test must assert that each key appears in this plan and in
  `docs/benchmarks/protocol.md`, and that README, AGENTS, roadmap, ROCm wave
  plan, ROCm backend architecture, release packaging docs, and
  `scripts/validate.py` point at the plan.

- [x] **Step 4: Run targeted tests**

  Run:

  ```bash
  uv run python -m pytest tests/test_rocm_campaign8_plan.py tests/test_validate_entrypoint.py -q
  ```

  Expected result:

  ```text
  all selected tests pass
  ```

- [x] **Step 5: Commit source-of-truth plan wiring**

  Run:

  ```bash
  git add README.md AGENTS.md docs/roadmap.md docs/plans/rocm_next_waves_plan.md docs/architecture/rocm_backend.md docs/benchmarks/protocol.md docs/quality/release_and_packaging.md scripts/validate.py tests/test_rocm_campaign8_plan.py docs/plans/mi300x_rocm_optimization_campaign8_plan.md
  git commit -m "docs: plan ROCm campaign 8"
  ```

## Task 2: Backend-Neutral Accelerator Contract

**Files:**
- Create: `docs/architecture/backend_neutral_accelerators.md`
- Modify: `docs/architecture/rocm_backend.md`
- Modify: `docs/architecture/cuda_backend.md`
- Modify: `docs/architecture/api_stability.md`
- Test: `uv run python -m pytest tests/test_rocm_campaign8_plan.py -q`

- [x] **Step 1: Write the backend-neutral decision document**

  Create `docs/architecture/backend_neutral_accelerators.md` with sections for:

  ```text
  backend identity: cpu, cuda, hip
  build modes: CPU-only, CUDA-only, HIP-only, future multi-backend
  DevicePauliSum ownership and moved-from behavior
  DeviceCommutationMatrix ownership and read-only export policy
  device ordinal semantics and same-device guards
  status reporting through _accelerator_status()
  error classes and message requirements
  packaging and wheel implications
  required tests before simultaneous CUDA+HIP can move from unavailable to accepted
  ```

  The document must keep the current result:

  ```text
  FASTPAULI_ENABLE_CUDA=ON with FASTPAULI_ENABLE_HIP=ON remains unavailable until a later implementation campaign accepts and validates the backend-neutral build model.
  ```

- [x] **Step 2: Link CUDA and ROCm architecture docs**

  Add a short section in both accelerator architecture docs pointing to
  `docs/architecture/backend_neutral_accelerators.md` and stating that the
  current builds remain CUDA-only or HIP-only.

- [x] **Step 3: Add API stability wording**

  Update `docs/architecture/api_stability.md` so any future multi-backend
  object changes require compatibility notes for:

  ```text
  DevicePauliSum.backend
  device ordinal behavior
  cross-backend operation errors
  _accelerator_status()
  packaging extras and build flags
  ```

- [x] **Step 4: Run targeted tests**

  Run:

  ```bash
  uv run python -m pytest tests/test_rocm_campaign8_plan.py -q
  ```

  Expected result:

  ```text
  all selected tests pass
  ```

## Task 3: Portability, Packaging, And Release Gates

**Files:**
- Create: `scripts/run_rocm_campaign8_readiness_lane.py`
- Modify: `docs/quality/release_and_packaging.md`
- Modify: `docs/architecture/hardware_targets_and_testing.md`
- Modify: `docs/benchmarks/protocol.md`
- Test: `uv run python -m pytest tests/test_rocm_campaign8_plan.py -q`

- [x] **Step 1: Add a dry-run readiness lane**

  Create `scripts/run_rocm_campaign8_readiness_lane.py` with `--print-commands`
  output for these command labels:

  ```text
  host-inventory
  cpu-only-control
  cuda-hip-rejection
  hip-source-build-mi300x
  hip-source-build-alternate-amd
  hip-retained-operation-tests
  rocm-release-smoke
  rocprof-legacy
  rocprofv3
  clean-machine-sdist-install
  packaging-policy-check
  render-assets
  report-validation
  ```

  The alternate-AMD command must be clearly labeled as requiring a real
  non-MI300X AMD GPU host before a support claim is made.

- [x] **Step 2: Document the ROCm wheel gate**

  Update `docs/quality/release_and_packaging.md` so ROCm wheels require:

  ```text
  supported package channel decision
  runtime dependency policy for ROCm libraries
  CI hardware that can build and import the wheel
  clean-machine install test for the produced artifact
  support-matrix wording that distinguishes source-build evidence from wheel support
  ```

- [x] **Step 3: Document the portability gate**

  Update `docs/architecture/hardware_targets_and_testing.md` so a new AMD GPU
  architecture requires:

  ```text
  source build on that architecture
  runtime status capture
  retained HIP operation tests
  benchmark smoke with correctness checks
  profiler availability status
  README support wording update
  ```

- [x] **Step 4: Run targeted tests**

  Run:

  ```bash
  uv run python -m pytest tests/test_rocm_campaign8_plan.py -q
  ```

  Expected result:

  ```text
  all selected tests pass
  ```

## Task 4: Profiler, Interop, And Performance-Reopen Contracts

**Files:**
- Create: `docs/plans/rocm_profiler_migration_campaign8_decision.md`
- Create: `docs/plans/rocm_hip_interop_reconsideration_campaign8_decision.md`
- Modify: `docs/architecture/rocm_backend.md`
- Modify: `docs/benchmarks/protocol.md`
- Test: `uv run python -m pytest tests/test_rocm_campaign8_plan.py -q`

- [x] **Step 1: Write the profiler migration decision**

  Create `docs/plans/rocm_profiler_migration_campaign8_decision.md` with:

  ```text
  legacy rocprof remains accepted when it produces HIP trace and stats artifacts
  rocprofv3 is preferred for ROCm 7.x and later when installed on the host
  side-by-side command shape is required before replacing legacy rocprof in reports
  profiler unavailability must record the exact missing binary, permission, or provider blocker
  ```

- [x] **Step 2: Write the HIP interop reconsideration decision**

  Create `docs/plans/rocm_hip_interop_reconsideration_campaign8_decision.md`
  with acceptance requirements for:

  ```text
  external HIP statevector device pointers
  HIP DLPack producer ownership
  consumer-library stream ownership
  read-only mutation rejection
  PyTorch ROCm or another named real consumer
  benchmark boundaries for consumer-only, transfer-inclusive, and device-resident rows
  ```

  The decision must retain the Campaign 5 rejection until a consumer rejects
  mutation of a read-only exported view.

- [x] **Step 3: Add the targeted performance reopen gate**

  Update `docs/benchmarks/protocol.md` so a new ROCm performance campaign must
  name:

  ```text
  retained operation
  profiler artifact
  measured bottleneck
  proposed implementation
  correctness oracle
  A/B timing boundary
  rejection criteria
  ```

- [x] **Step 4: Run targeted tests**

  Run:

  ```bash
  uv run python -m pytest tests/test_rocm_campaign8_plan.py -q
  ```

  Expected result:

  ```text
  all selected tests pass
  ```

## Task 5: Campaign 8 Execution Evidence And Report

**Files:**
- Create: `scripts/render_rocm_campaign8_assets.py`
- Create: `tests/test_rocm_campaign8_assets.py`
- Create: `docs/benchmarks/data/rocm_campaign8_architecture_readiness_2026-05-01/summary.json`
- Create: `docs/benchmarks/reports/rocm_campaign8_architecture_readiness_2026-05-01.md`
- Modify: `README.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/plans/rocm_next_waves_plan.md`
- Test: `uv run python -m pytest tests/test_rocm_campaign8_plan.py tests/test_rocm_campaign8_assets.py -q`

- [x] **Step 1: Add the Campaign 8 renderer**

  Create a renderer that reads `summary.json`, verifies the exact
  `campaign8_terminal_statuses` key set, and preserves the broad
  `accelerator_landscape_with_rocm.svg` README plot. Campaign 8 should not
  replace the broad landscape with a narrow architecture-only image.

- [x] **Step 2: Capture execution evidence**

  Run the readiness lane and write evidence under:

  ```text
  docs/benchmarks/data/rocm_campaign8_architecture_readiness_2026-05-01/
  ```

  The summary must record:

  ```text
  command
  git revision
  local CPU-only validation result
  CUDA+HIP configure rejection result
  alternate AMD GPU availability result
  packaging gate status
  rocprofv3 status
  interop reconsideration status
  targeted performance reopen status
  campaign8_terminal_statuses
  ```

- [x] **Step 3: Write the report**

  Create `docs/benchmarks/reports/rocm_campaign8_architecture_readiness_2026-05-01.md`
  with:

  ```text
  scope and non-goals
  summary of terminal statuses
  architecture decision outcomes
  portability and packaging gate outcomes
  profiler migration outcome
  interop reconsideration outcome
  targeted performance reopen outcome
  validation commands and observed results
  residual risk and next triggers
  ```

- [x] **Step 4: Update public docs after evidence exists**

  Update README, roadmap, and ROCm next waves only after the report exists.
  The wording must distinguish planned Campaign 8 from completed Campaign 8
  until the report and validation are checked in.

- [x] **Step 5: Run targeted tests**

  Run:

  ```bash
  uv run python -m pytest tests/test_rocm_campaign8_plan.py tests/test_rocm_campaign8_assets.py -q
  git diff --check
  ```

  Expected result:

  ```text
  all selected tests pass
  git diff --check passes
  ```

## Task 6: Review, Merge, Push, And CI Closeout

**Files:**
- Review all files changed by Tasks 1-5.
- Test: `uv run python scripts/validate.py`

- [ ] **Step 1: Request independent review**

  Request review with scope:

  ```text
  source docs: Campaign 8 plan, backend-neutral architecture decision, ROCm backend architecture, release packaging docs, benchmark protocol
  tests: Campaign 8 plan/assets tests and validation source-of-truth checks
  risk focus: accidental support claims, missing terminal statuses, stale Campaign 7 latest-plan routing, unsupported ROCm wheel claims, and architecture decisions recorded as completed before evidence exists
  ```

- [ ] **Step 2: Resolve blocking findings**

  For each blocking finding:

  ```text
  patch the source document or test
  rerun the narrow failing test
  record the resolution in the closeout
  ```

- [ ] **Step 3: Run full validation**

  Run:

  ```bash
  uv run python scripts/validate.py
  ```

  Expected result:

  ```text
  validation passes
  ```

- [ ] **Step 4: Commit final evidence and docs**

  Run:

  ```bash
  git add README.md AGENTS.md docs scripts tests
  git commit -m "docs: publish ROCm campaign 8 readiness plan"
  ```

- [ ] **Step 5: Merge, push, confirm CI, and clean up**

  Run:

  ```bash
  git switch main
  git merge --ff-only <campaign8-branch>
  uv run python scripts/validate.py
  git push origin main
  gh run list --branch main --limit 5
  gh run watch <run-id> --exit-status
  git branch -d <campaign8-branch>
  ```

  Expected result:

  ```text
  main is pushed
  CI is green
  local feature branch is deleted
  worktree is clean
  ```

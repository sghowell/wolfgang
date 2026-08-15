# CUDA Residual-Risk Campaign 11 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development or superpowers:executing-plans to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Goal:** Close the two Campaign 10 residual-risk items that are in scope now:
install or enable Nsight Compute CLI on the current non-H100 hosts and
investigate the nanobind reference-leak diagnostics reported by Compute
Sanitizer. This campaign must not broaden FastPauli's NVIDIA portability claims
or reopen rejected Campaign 10 surfaces.

**Architecture:** Campaign 11 is a residual-risk closure campaign. It adds
missing profiler-counter evidence for the already validated A100 and RTX PRO
6000 Blackwell source-build lanes, and hardens Python binding lifecycle
confidence without changing public CUDA semantics unless a real ownership bug
is proven.

**Tech Stack:** C++20, CUDA C++ 12.x or the installed host toolkit, nanobind,
pytest, Compute Sanitizer, Nsight Compute CLI (`ncu`), existing CUDA benchmark
and profiling scripts, and checked JSON/Markdown benchmark artifacts.

---

## Status

Status: completed on 2026-04-29.

Checked output:

```text
plan: docs/plans/cuda_residual_risk_campaign11_plan.md
report: docs/benchmarks/reports/cuda_residual_risk_campaign11_2026-04-29.md
summary: docs/benchmarks/data/cuda_residual_risk_campaign11_2026-04-29/summary.json
raw data: docs/benchmarks/data/cuda_residual_risk_campaign11_2026-04-29/raw/
logs: docs/benchmarks/data/cuda_residual_risk_campaign11_2026-04-29/logs/
profiler artifacts: docs/benchmarks/data/cuda_residual_risk_campaign11_2026-04-29/profiler/
```

Completion evidence:

```text
local macOS CPU validation: passed on f9e9e46 with 195 passed / 59 skipped plus benchmark and sdist smokes
A100 full CUDA validation: passed on f9e9e46 with FASTPAULI_CUDA_ARCHITECTURES=80
RTX PRO 6000 Blackwell full CUDA validation: passed on f9e9e46 with FASTPAULI_CUDA_ARCHITECTURES=120
A100 CUDA subset validation: passed with FASTPAULI_CUDA_ARCHITECTURES=80
RTX PRO 6000 Blackwell CUDA subset validation: passed with FASTPAULI_CUDA_ARCHITECTURES=120
A100 ncu: blocked_permissions after ERR_NVGPUCTRPERM and sudo InterprocessLockFailed evidence
RTX PRO 6000 Blackwell ncu: passed with .ncu-rep and text export checked in
nanobind diagnostics: rejected_with_evidence as sanitizer/nanobind teardown diagnostics after clean lifecycle subprocesses
```

Campaign 11 starts from:

```text
plan: docs/plans/cuda_cross_architecture_campaign10_plan.md
report: docs/benchmarks/reports/cuda_cross_architecture_campaign10_2026-04-29.md
summary: docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29/summary.json
ncu gaps: docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29/logs/ncu_unavailable_a100.log
ncu gaps: docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29/logs/ncu_unavailable_rtxpro6000blackwell.log
memcheck logs: docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29/logs/compute_sanitizer_memcheck_a100.log
memcheck logs: docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29/logs/compute_sanitizer_memcheck_rtxpro6000blackwell.log
```

The final Campaign 11 summary must give every in-scope item a terminal status.
Allowed terminal outcomes are:

```text
passed
fixed
rejected_with_evidence
blocked_toolchain
blocked_permissions
blocked_dependency
blocked_external
```

The final Campaign 11 summary must not contain `final_status: "deferred"`.

## Scope Freeze

In scope:

```text
A100 host only for the sm_80 non-H100 datacenter lane
RTX PRO 6000 Blackwell host only for the sm_120 workstation/newer-architecture lane
installing or enabling Nsight Compute CLI on those two hosts
capturing non-H100 Nsight Compute counter evidence when installation and permissions allow
recording exact ncu versions, install commands, permission status, and profiler command lines
investigating nanobind reference-leak diagnostics from Campaign 10 Compute Sanitizer logs
fixing binding lifecycle bugs only if the investigation proves a real ownership issue
adding targeted tests, report assets, and documentation required to make the residual-risk status durable
independent review before merge because this changes CUDA benchmark and target-policy planning surfaces
```

Out of scope:

```text
A10, L4, RTX 6000 Ada, sm_86, sm_89, or any other additional NVIDIA host
CUDA wheel release claims or packaging work
new public CUDA APIs, stream/event APIs, async APIs, raw pointer APIs, or ownership surfaces
reopening public grouping, CUDA Graph, stream-aware execution, CSR scatter, or raw PTX work
performance hillclimbing beyond profiler evidence needed to close the residual-risk items
broad README performance-claim changes unless new checked evidence materially changes the landscape
```

## Hardware And Access

Use the existing Campaign 10 hosts:

```text
A100: ssh ubuntu@<private-address>
RTX PRO 6000 Blackwell: ssh root@<private-address> -p 22
```

Before running profiler or sanitizer commands, record:

```text
hostname
uname -a
nvidia-smi
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
nvcc --version
python --version
git rev-parse HEAD
```

Use `FASTPAULI_CUDA_ARCHITECTURES=80` on A100 and
`FASTPAULI_CUDA_ARCHITECTURES=120` on RTX PRO 6000 Blackwell unless host
toolchain discovery proves that a different explicit value is required. Do not
silently compile a lower architecture while reporting the host as validated for
its native compute capability.

## Source Inputs

Read these before editing code, tests, or benchmark logic:

```text
README.md
docs/roadmap.md
docs/plans/cuda_deep_optimization_plan.md
docs/plans/cuda_cross_architecture_campaign10_plan.md
docs/benchmarks/reports/cuda_cross_architecture_campaign10_2026-04-29.md
docs/architecture/cuda_backend.md
docs/architecture/hardware_targets_and_testing.md
docs/architecture/api_stability.md
docs/benchmarks/protocol.md
docs/quality/code_review.md
docs/quality/code_standards.md
docs/quality/documentation_standards.md
bindings/python/pauli_sum_py.cpp
include/fastpauli/device_commutation_matrix.hpp
include/fastpauli/device_pauli_sum.hpp
src/cuda/commutation_cuda.cu
src/cuda/device_commutation_matrix.cu
src/cuda/device_commutation_matrix.cuh
src/cuda/device_pauli_sum.cu
src/cuda/workspace.cu
src/cuda/workspace.cuh
benchmarks/bench_cuda_scaling.py
benchmarks/bench_cuda_kernels.py
scripts/cuda_deep_profile.py
scripts/render_cuda_campaign10_assets.py
tests/test_phase11_cuda_kernels.py
tests/test_cuda_deep_report_assets.py
```

## Execution Plan

### 1. Lock The Evidence Schema

- [x] Add or update report-asset tests so Campaign 11 summary evidence records
  both residual-risk items:

  ```text
  non_h100_ncu_counters
  nanobind_refleak_investigation
  ```

- [x] Require each item to use one of the allowed terminal statuses and reject
  any checked summary that contains a deferred status.
- [x] Require host-level records for A100 and RTX PRO 6000 Blackwell, including
  `ncu_version`, `ncu_install_status`, `profiler_permission_status`, command,
  output artifact paths, and limitations.
- [x] Keep the renderer or summary helper small. Do not create a new benchmark
  framework for this residual-risk slice.

### 2. Install Or Enable Nsight Compute On Non-H100 Hosts

- [x] On each host, record current `ncu` discovery:

  ```bash
  command -v ncu || true
  ncu --version || true
  apt-cache search '^nsight-compute' || true
  apt-cache policy nsight-compute || true
  ```

- [x] If `ncu` is missing, install the package exposed by the configured CUDA
  apt repository. Prefer the repository package that supplies the CLI instead
  of downloading ad hoc binaries.

  ```bash
  sudo apt-get update
  sudo apt-get install -y nsight-compute
  ```

  On the RTX host, use root-equivalent commands without `sudo` when the login
  user is root. If the package name differs, record the discovered package name
  and exact install command.

- [x] Verify:

  ```bash
  command -v ncu
  ncu --version
  ncu --query-metrics
  ```

- [x] If installation is blocked by repository state, package absence, network,
  or host permissions, record `blocked_toolchain` or `blocked_external` with the
  exact command output. Do not substitute another host.

### 3. Capture Non-H100 Nsight Compute Counter Evidence

- [x] Build and validate the current repository revision on each host before
  profiler capture.
- [x] Run a targeted non-H100 Nsight Compute pass for retained compact CUDA
  consumers, not rejected full-CSR or stream/CUDA Graph experiments.
- [x] Prefer profiling command shapes that match Campaign 10 retained evidence:

  ```text
  compact graph consumer over DeviceCommutationMatrix
  compact grouping/conflict-degree consumer over DeviceCommutationMatrix
  PyTorch and CuPy DLPack consumer smoke only when the installed environment supports them
  ```

- [x] Capture both `.ncu-rep` files and text exports when `ncu` supports them.
- [x] If `ncu` reports `ERR_NVGPUCTRPERM`, retry only with available root or
  sudo privileges on the same host. If counters remain blocked, record
  `blocked_permissions` with the exact message and do not change kernel module
  settings without a separate owner-approved infrastructure action.
- [x] Compare A100 and RTX PRO 6000 Blackwell counter shape against the retained
  H100 Nsight Compute evidence from Campaign 9. The report should identify
  whether the non-H100 evidence supports the same bottleneck model or exposes a
  host-specific limitation.

### 4. Investigate Nanobind Reference-Leak Diagnostics

- [x] Reproduce the Campaign 10 Compute Sanitizer memcheck command on both
  hosts and retain fresh logs.
- [x] Add a narrow binding-lifecycle test if the existing test suite does not
  isolate wrapper construction/destruction. The test should create, transfer,
  consume, and destroy:

  ```text
  PauliSum
  DevicePauliSum
  DeviceCommutationMatrix
  read-only DLPack consumers when optional dependencies are installed
  ```

- [x] Inspect nanobind ownership and return policies in
  `bindings/python/pauli_sum_py.cpp`, especially device-returning methods,
  matrix-returning methods, DLPack capsules, and objects that own CUDA memory.
- [x] Classify the diagnostic with evidence:

  ```text
  fixed: a real reference or ownership bug was found, patched, tested, and sanitizer-cleaned
  rejected_with_evidence: the diagnostic is sanitizer/process-teardown noise with no reachable runtime leak under targeted lifecycle tests
  blocked_dependency: nanobind or sanitizer limitations prevent a definitive classification after bounded reproduction
  ```

- [x] If a real bug is fixed, rerun CUDA correctness tests, the targeted
  lifecycle test, Compute Sanitizer memcheck, and at least one retained CUDA
  benchmark smoke on the affected host.

### 5. Produce Checked Report Artifacts

- [x] Write the Campaign 11 report at
  `docs/benchmarks/reports/cuda_residual_risk_campaign11_2026-04-29.md`.
- [x] Include a concise table for each residual-risk item:

  ```text
  item
  host
  terminal status
  command
  artifact path
  limitation
  decision
  ```

- [x] Include a short comparison to Campaign 10 and the H100 Nsight Compute
  baseline. Make clear that Campaign 11 closes residual risk only for the
  current A100 and RTX PRO 6000 Blackwell lanes.
- [x] Update Campaign 10 follow-up references in `README.md`, `docs/roadmap.md`,
  and `docs/plans/cuda_deep_optimization_plan.md` only to the extent needed to
  point to the new checked report. Do not rewrite the performance landscape
  unless the new evidence changes benchmark results.

### 6. Review, Validate, Merge, Push, And Cleanup

- [x] Run local validation on the feature branch:

  ```bash
  python scripts/validate.py
  ```

- [x] Run CUDA validation on each affected GPU host:

  ```bash
  FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=80 python scripts/validate.py
  FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=120 python scripts/validate.py
  ```

- [x] Commit in sensible chunks: schema/test changes, implementation or
  lifecycle fixes, and checked report/docs updates.
- [x] Request independent agent-driven review before merge, with branch diff,
  validation output, GPU host evidence, and known limitations.
- [x] Resolve P0/P1 findings, rerun validation after any fix, then fast-forward
  merge locally to `main`.
- [ ] Validate merged `main`, push, confirm CI green, and delete the merged
  local branch.

## Acceptance Criteria

Campaign 11 is complete only when:

```text
Campaign 11 summary exists and contains no deferred residual-risk item
A100 ncu status is terminal: counters captured or explicit toolchain/permission blocker recorded
RTX PRO 6000 Blackwell ncu status is terminal: counters captured or explicit toolchain/permission blocker recorded
non-H100 profiler evidence, if captured, has command lines, ncu versions, raw artifacts, and text summaries checked in
nanobind reference-leak diagnostics have a terminal classification with fresh reproduction evidence
any real binding lifecycle bug has a targeted regression test and sanitizer evidence after the fix
the report explicitly excludes A10, L4, RTX 6000 Ada, and other additional NVIDIA hosts
no CUDA wheel, stream, graph, CSR scatter, raw PTX, or new public API claim is introduced
docs and README references point to the Campaign 11 plan/report without overstating portability
required validation and independent review evidence are recorded in closeout
```

## Exhaustion Criteria

Stop Campaign 11 only when all in-scope residual-risk items have terminal
evidence:

```text
ncu installed and profiler artifacts captured on both hosts
or ncu installation/counter access is blocked on a host with exact command evidence
and the nanobind diagnostic is either fixed or bounded to non-runtime-risk/noise with targeted lifecycle and sanitizer evidence
```

If a new correctness, ownership, or sanitizer error appears during this work,
stop treating the slice as a small residual-risk closure and write a focused
bug-fix plan before broadening scope.

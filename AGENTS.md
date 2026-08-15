# FastPauli Agent Guide

This repository is intended to be implemented by Codex agents. Keep this file short: it is the entrypoint map, not the full manual.

## Read First

Before implementation work, read the relevant source-of-truth docs:

```text
README.md
docs/roadmap.md
CHANGELOG.md
docs/plans/fastpauli_cpp_cuda_implementation_plan.md
docs/plans/release_candidate_foundation_plan.md
docs/plans/release_candidate_next_checkpoint_plan.md
docs/plans/release_0_1_0_wheelhouse_foundation_plan.md
docs/plans/apple_metal_mps_bringup_plan.md
docs/plans/apple_metal_optimization_campaign1_plan.md
docs/plans/apple_metal_optimization_campaign2_plan.md
docs/plans/apple_metal_optimization_campaign3_plan.md
docs/plans/apple_metal_optimization_campaign4_plan.md
docs/plans/apple_metal_optimization_campaign5_plan.md
docs/plans/apple_metal_optimization_campaign6_plan.md
docs/plans/apple_metal_optimization_campaign7_plan.md
docs/plans/apple_metal_optimization_campaign8_plan.md
docs/plans/cuda_deep_optimization_plan.md
docs/plans/h100_deep_optimization_campaign2_plan.md
docs/plans/h100_deep_optimization_campaign3_plan.md
docs/plans/h100_deep_optimization_campaign8_plan.md
docs/plans/h100_deep_optimization_campaign9_plan.md
docs/plans/cuda_cross_architecture_campaign10_plan.md
docs/plans/cuda_residual_risk_campaign11_plan.md
docs/plans/mi300x_rocm_bringup_plan.md
docs/plans/rocm_next_waves_plan.md
docs/plans/mi300x_rocm_optimization_campaign2_plan.md
docs/plans/mi300x_rocm_optimization_campaign3_plan.md
docs/plans/mi300x_rocm_optimization_campaign4_plan.md
docs/plans/mi300x_rocm_optimization_campaign5_plan.md
docs/plans/mi300x_rocm_optimization_campaign6_plan.md
docs/plans/mi300x_rocm_optimization_campaign7_plan.md
docs/plans/mi300x_rocm_optimization_campaign8_plan.md
docs/plans/backend_neutral_accelerator_campaign9_plan.md
docs/benchmarks/reports/backend_neutral_accelerator_campaign9_2026-05-01.md
docs/benchmarks/reports/cuda_cross_architecture_campaign10_2026-04-29.md
docs/benchmarks/reports/cuda_residual_risk_campaign11_2026-04-29.md
docs/benchmarks/reports/rocm_mi300x_bringup_2026-04-29.md
docs/benchmarks/reports/rocm_mi300x_campaign2_2026-04-30.md
docs/benchmarks/reports/rocm_mi300x_campaign3_2026-04-30.md
docs/benchmarks/reports/rocm_mi300x_campaign4_2026-04-30.md
docs/benchmarks/reports/rocm_mi300x_campaign5_2026-04-30.md
docs/benchmarks/reports/rocm_mi300x_campaign6_2026-04-30.md
docs/benchmarks/reports/rocm_mi300x_campaign7_2026-04-30.md
docs/benchmarks/reports/rocm_campaign8_architecture_readiness_2026-05-01.md
docs/benchmarks/reports/apple_metal_bringup_2026-05-01.md
docs/benchmarks/reports/apple_metal_optimization_campaign1_2026-05-05.md
docs/benchmarks/reports/apple_metal_optimization_campaign2_2026-05-05.md
docs/benchmarks/reports/apple_metal_optimization_campaign3_2026-05-06.md
docs/benchmarks/reports/apple_metal_optimization_campaign4_2026-05-06.md
docs/benchmarks/reports/apple_metal_optimization_campaign5_2026-05-06.md
docs/benchmarks/reports/apple_metal_optimization_campaign6_2026-05-07.md
docs/benchmarks/reports/apple_metal_optimization_campaign7_2026-05-07.md
docs/benchmarks/reports/apple_metal_optimization_campaign8_2026-05-07.md
docs/architecture/semantic_contracts.md
docs/architecture/cuda_backend.md
docs/architecture/rocm_backend.md
docs/architecture/apple_accelerator.md
docs/architecture/backend_neutral_accelerators.md
docs/architecture/hardware_targets_and_testing.md
docs/architecture/testing_and_ci.md
docs/architecture/adapter_contracts.md
docs/benchmarks/protocol.md
docs/quality/phase_quality_gates.md
docs/quality/agent_harness.md
docs/quality/code_review.md
docs/quality/code_standards.md
docs/quality/documentation_standards.md
docs/architecture/api_stability.md
docs/quality/security_and_supply_chain.md
docs/quality/release_and_packaging.md
docs/release/README.md
docs/release/0.1.0.md
docs/release/0.1.0-wheelhouse-dry-run.md
docs/release/0.1.0-rc2.md
docs/release/0.1.0-rc1.md
docs/release/support_matrix.md
CONTRIBUTING.md
```

Use progressive disclosure. Start with `README.md` and `docs/roadmap.md`, then load only the phase-specific docs needed for the task.
Apple Metal Campaign 8 is the latest Apple Metal simplify performance-relevance evidence slice.

## Operating Rules

Work in short-lived feature branches:

```text
1. Create a branch with the codex/ prefix.
2. Keep changes scoped to the requested phase or slice.
3. Validate the feature branch with the checks available for the slice.
4. Stage and commit in sensible chunks with concise descriptive messages.
5. Complete the review stage required by docs/quality/code_review.md.
6. Resolve blocking review findings and revalidate when fixes are made.
7. Merge locally to main with a fast-forward merge when possible.
8. Validate the merged result.
9. Push main.
10. Confirm CI is green when CI exists.
11. Delete the merged local feature branch.
```

Never claim a phase is complete without fresh validation evidence.

## Phase Completion

A phase is complete only when:

```text
the roadmap acceptance criteria are satisfied
the relevant section of docs/quality/phase_quality_gates.md is satisfied
public behavior matches docs/architecture/semantic_contracts.md
CUDA-relevant choices remain compatible with docs/architecture/cuda_backend.md
ROCm/HIP-relevant choices remain compatible with docs/architecture/rocm_backend.md
Apple Metal-relevant choices remain compatible with docs/architecture/apple_accelerator.md
CPU, CUDA, ROCm/HIP, compiler, dispatch, and benchmark-target choices follow docs/architecture/hardware_targets_and_testing.md
review requirements in docs/quality/code_review.md are satisfied or an explicit exception is recorded
code quality follows docs/quality/code_standards.md
user-facing docs and API docs follow docs/quality/documentation_standards.md
public API changes follow docs/architecture/api_stability.md
security and dependency choices follow docs/quality/security_and_supply_chain.md
packaging and release changes follow docs/quality/release_and_packaging.md
docs, tests, and benchmarks are updated in the same slice when affected
```

Phase 1 must introduce the repo-local validation entrypoint and initial CPU CI surface described in `docs/architecture/testing_and_ci.md` and `docs/quality/agent_harness.md`.

## Implementation Discipline

Prefer:

```text
C++20 baseline
CPU correctness before CUDA kernels
small exact tests before broad randomized tests
deterministic datasets and seeds
clear guardrails before large allocations
structured APIs and parsers over ad hoc string manipulation
scalar correctness paths before oneTBB, SIMD, or CUDA optimization
portable CPU wheel defaults before native CPU tuning
expert-level comments that explain invariants, formulas, edge cases, and performance assumptions
user-facing docs that are accurate, example-driven, and evidence-based
```

Do not:

```text
import optional Qiskit/OpenFermion dependencies at package import time
make performance claims without benchmark evidence from docs/benchmarks/protocol.md
add CUDA header dependencies to CPU-only public headers
add HIP or ROCm header dependencies to CPU-only public headers
add Metal, Foundation, MPS, or MPSGraph header dependencies to CPU-only public headers
compile release wheels with native CPU flags or import-time CPU feature requirements
skip documented phase gates because a local test happened to pass
merge implementation work without the required review stage
leave accepted decisions only in chat
ship public APIs without docstrings or API documentation once exposed beyond scaffold
add noisy comments that restate obvious syntax
```

## Validation

Until Phase 1 creates `scripts/validate.py`, use the checks available for the current slice and report exactly what ran.

For docs-only changes, run at minimum:

```bash
git diff --check
```

Also scan `README.md`, `AGENTS.md`, and `docs/` for stale markers or obsolete planning language.

After Phase 1, use:

```bash
python scripts/validate.py
```

CI should run the same validation entrypoint.

## Review

Implementation slices require independent agent-driven review before merge. Docs-only review triggers are defined in `docs/quality/code_review.md`; use that policy as the source of truth instead of maintaining a shorter list here.

Follow `docs/quality/code_review.md`. Record review scope, findings, resolutions, deferrals, validation after fixes, and residual risk in the closeout.

## Benchmarks

Benchmarks must be reproducible and evidence-bearing:

```text
record command, git revision, dataset, environment, baseline, result, and limitations
record active CPU backend, CPU feature set, compiler flags, and thread settings
record CUDA toolkit, compiled architectures, driver, device, and compute capability when CUDA is used
record ROCm toolkit, HIP architecture, driver/runtime, device, and gfx target when HIP is used
record macOS, Xcode or Command Line Tools, Metal device, storage mode, and command-buffer boundary when Metal is used
report transfer-inclusive and device-resident timings separately for CUDA, ROCm/HIP, and Metal
label Apple Metal Campaign 7 fixed-dyadic checked primitive rows as benchmark-only evidence
do not compare an optimized path only against itself
```

Follow `docs/benchmarks/protocol.md`.

## Cleanup

If a rule becomes repeatedly relevant, encode it in docs, validation scripts, tests, or CI. The repo should become more legible to future agents after every slice.

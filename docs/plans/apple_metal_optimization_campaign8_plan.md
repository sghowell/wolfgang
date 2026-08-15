# Apple Metal Campaign 8 Simplify Performance Relevance Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the Campaign 7 checked one-word Metal simplify candidate can become performance-relevant, or record evidence that it should remain benchmark-only experimental work.

**Architecture:** Campaign 8 keeps public Metal `DevicePauliSum.simplify(atol, rtol)` on the Campaign 5 transfer-reference path. It adds benchmark-only timing decomposition for the private one-word fixed-dyadic simplify candidate, makes the pipeline/library cache boundary explicit, records dispatch-count and scratch-allocation evidence, and publishes a decision report. No public API is promoted unless same-host evidence shows a correct retained Metal path beating CPU simplify and the transfer-reference bridge.

**Tech Stack:** C++20, Objective-C++ private Metal translation units, Metal Shading Language, nanobind private test hooks, pytest, deterministic benchmark JSON, checked Markdown evidence, and the existing Apple Metal asset renderer.

---

## Scope

Campaign 8 covers:

```text
private timing decomposition for the checked one-word Metal simplify candidate
explicit pipeline/library cache boundary reporting
benchmark rows that separate host preflight, scratch/output allocation, command encoding, command execution, and output accounting
dispatch-count evidence for the bitonic sort, prefix-sum, reduce-by-key, and compaction stack
Campaign 8 benchmark profile, raw JSON, summary JSON, README landscape refresh, and evidence report
decision criteria for keeping the candidate experimental or opening a later retained-public-path design
```

Campaign 8 does not cover:

```text
public promotion of Metal simplify
general FP64 Metal simplify
multi-word Metal simplify
Metal wheels or PyPI publication
MPSGraph or PyTorch MPS sparse-Pauli substitutes
raw Metal buffer or command-queue public APIs
```

## Acceptance Criteria

Campaign 8 is complete only when:

```text
docs/plans/apple_metal_optimization_campaign8_plan.md is registered in README, roadmap, AGENTS, and scripts/validate.py
tests prove the Campaign 8 profile exists and has deterministic simplify cases
private candidate reports include timing_decomposition_seconds
private candidate reports include pipeline_cache, dispatch_counts, and performance_decision metadata
benchmark rows include the new decomposition fields for ok device-candidate rows
status-only/rejected rows do not claim command buffers or kernel stacks
Campaign 8 raw JSON and summary JSON are checked in under docs/benchmarks/data/apple_metal_optimization_campaign8_2026-05-07
the broad README performance landscape remains CPU/CUDA/ROCm/HIP/Apple Metal/external rather than Metal-only
the Campaign 8 report states whether the candidate is performance-relevant or should remain experimental, based on checked same-host data
public DevicePauliSum.simplify() remains the transfer-reference bridge
default validation passes
FASTPAULI_VALIDATE_METAL=1 validation passes on the local Apple Silicon host
independent review finds no blocking issues, or all blocking issues are fixed before merge
```

## Implementation Tasks

### Task 1: Add Plan And Failing Asset Tests

**Files:**

```text
Create: tests/test_apple_metal_campaign8_plan.py
Create: tests/test_apple_metal_campaign8_assets.py
Modify: docs/plans/apple_metal_optimization_campaign8_plan.md
```

- [x] Write tests that require Campaign 8 plan routing in README, roadmap, AGENTS, architecture, protocol, and validation.
- [x] Write tests that require a `campaign8` benchmark profile and checked Campaign 8 assets.
- [x] Run the new tests and confirm they fail because Campaign 8 code and assets do not exist yet.

### Task 2: Add Private Timing Decomposition

**Files:**

```text
Modify: src/metal/simplify_metal.hpp
Modify: src/metal/simplify_metal.mm
Modify: bindings/python/module.cpp
Modify: tests/test_apple_metal_campaign8.py
```

- [x] Extend `MetalSimplifyCandidateResult` with timing decomposition, dispatch counts, pipeline-cache boundary, and performance-decision fields.
- [x] Measure host preflight, scratch/output allocation, command encoding, command execution, and output accounting with `std::chrono::steady_clock`.
- [x] Keep the timing fields diagnostic only; benchmark medians continue to come from the Python benchmark loop.
- [x] Record the static pipeline cache boundary as `prewarmed_static_pipeline_cache` when the benchmark calls the hook after correctness prewarm.
- [x] Preserve all Campaign 7 rejection behavior and exact fixed32 dyadic correctness checks.

### Task 3: Add Campaign 8 Benchmark Profile

**Files:**

```text
Modify: benchmarks/bench_metal_kernels.py
Modify: scripts/validate.py
Modify: tests/test_apple_metal_campaign8_assets.py
```

- [x] Add deterministic `campaign8` simplify cases that include duplicate-heavy, duplicate-light, cancellation, and larger one-word workloads.
- [x] Include same-host CPU default/scalar, Metal transfer-reference, and private device-candidate rows.
- [x] Add timing decomposition, dispatch counts, pipeline-cache boundary, and decision metadata to ok device-candidate rows.
- [x] Register a Campaign 8 Metal benchmark smoke in `scripts/validate.py`.

### Task 4: Render Evidence And Update Docs

**Files:**

```text
Modify: scripts/render_apple_metal_assets.py
Create: docs/benchmarks/data/apple_metal_optimization_campaign8_2026-05-07/raw/metal_benchmark_campaign8.json
Create: docs/benchmarks/data/apple_metal_optimization_campaign8_2026-05-07/summary.json
Create: docs/benchmarks/reports/apple_metal_optimization_campaign8_2026-05-07.md
Modify: docs/benchmarks/plots/accelerator_landscape_with_rocm.svg
Modify: README.md
Modify: docs/roadmap.md
Modify: docs/architecture/apple_accelerator.md
Modify: docs/benchmarks/protocol.md
Modify: CHANGELOG.md
Modify: AGENTS.md
```

- [x] Teach the renderer about `apple_metal_optimization_campaign8`.
- [x] Render Campaign 8 summary JSON and refresh the broad README landscape plot.
- [x] Write the Campaign 8 report with commands, environment, row interpretation, and the retain-or-experimental decision.
- [x] Update docs to keep Campaign 8 as the latest Apple Metal source-build evidence.

### Task 5: Validate, Review, And Close Out

**Files:**

```text
All changed files
```

- [x] Run focused Campaign 8 tests.
- [x] Run `git diff --check`.
- [x] Run `python scripts/validate.py`.
- [x] Run `FASTPAULI_VALIDATE_METAL=1 python scripts/validate.py`.
- [x] Request independent review of the branch diff.
- [x] Fix blocking findings and revalidate affected tests.
- [ ] Commit, merge to `main`, validate merged result, push `main`, confirm CI green, and delete the merged local feature branch.

## Decision Rule

The Campaign 8 report must use this rule:

```text
performance-relevant: the checked device-candidate median beats same-host CPU default and Metal transfer-reference on at least one duplicate-heavy or cancellation workload, with correctness true and no unsupported domain broadening
experimental: the checked device-candidate remains slower than same-host CPU default on all checked workloads, or the timing decomposition shows that public promotion would require new lifetime/output APIs or a lower-pass sort design
```

If the result is `experimental`, the report must name the smallest credible next design: lower-pass deterministic Metal sort, reusable scratch/output ownership boundary, or CPU simplify optimization.

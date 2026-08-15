# Apple Metal Campaign 6 Device-Resident Simplify Groundwork Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the private workspace, benchmark metadata, and evidence surfaces needed before any Metal device-resident simplify implementation can be retained.

**Architecture:** Campaign 6 does not change public API behavior. `DevicePauliSum.simplify(atol, rtol)` remains the Campaign 5 transfer-reference correctness bridge. Campaign 6 adds a private MetalWorkspace model under `src/metal`, Campaign 6 benchmark cases, and status-only `metal_simplify_workspace_probe` rows that record the scratch-space shape needed for a future device-resident duplicate-reduction candidate. The device-resident simplify candidate remains blocked until checked Metal sort, prefix-sum, and reduce-by-key primitives exist.

**Tech Stack:** C++20, Objective-C++, Metal `MTLBuffer` scratch reservation, nanobind source builds, pytest, `benchmarks/bench_metal_kernels.py`, `scripts/render_apple_metal_assets.py`, and `scripts/validate.py`.

**Status:** Completed in `docs/benchmarks/reports/apple_metal_optimization_campaign6_2026-05-07.md`.

---

Date: 2026-05-07

This plan follows
`docs/benchmarks/reports/apple_metal_optimization_campaign5_2026-05-06.md`.
Campaign 5 made Metal simplify correct through a transfer-reference path.
Campaign 6 addresses the first remaining-headroom item from that report:
designing the private scratch/workspace model before a device-resident simplify
candidate is attempted.
Apple Metal Campaign 6 device-resident simplify groundwork is therefore a
planning, workspace, benchmark, and evidence slice rather than a new retained
GPU duplicate-reduction implementation.

## Scope

Campaign 6 covers:

```text
private MetalWorkspace class with WorkspaceTimingMode and WorkspaceSnapshot
FASTPAULI_METAL_BENCH_WORKSPACE_TIMING benchmark timing-mode vocabulary
FASTPAULI_EXPERIMENTAL_METAL_SIMPLIFY_STRATEGY benchmark selector vocabulary
campaign6 simplify benchmark cases that mirror Campaign 5 duplicate pressure
status-only metal_simplify_workspace_probe rows with status_only boundary
retained transfer-reference Metal simplify rows on the Campaign 6 cases
Campaign 6 raw JSON, summary JSON, report, and README landscape refresh
validation entrypoint coverage for the Campaign 6 benchmark smoke
```

Out of scope:

```text
public Metal workspaces, queues, command buffers, heaps, events, or async APIs
raw Metal buffer export, DLPack, PyTorch MPS tensor export, or Metal Array Interface-style objects
retaining a device-resident simplify implementation without a checked primitive stack
claiming the transfer-reference path as a device_resident GPU speedup
Metal statevector expectation
Metal matmul
Metal wheels, PyPI publication, Windows support, or older macOS compatibility
generic Apple GPU support claims from one Apple M4 Pro host
MPSGraph or PyTorch MPS as FastPauli backend identities
```

## Accepted Campaign 6 Boundary

The retained public behavior is unchanged:

```text
Metal DevicePauliSum -> host PauliSum -> CPU PauliSum.simplify() -> Metal DevicePauliSum
```

The new groundwork row is intentionally not timed as a Metal kernel:

```text
variant: metal_simplify_workspace_probe
operation: simplify
transfer_boundary: status_only
metal_simplify_strategy: device_candidate
metal_simplify_strategy_status: rejected_with_evidence
metal_simplify_workspace_model.status: retained_private_model
```

The row must state that the device-resident simplify candidate remains blocked
until FastPauli has checked Metal sort, prefix-sum, and reduce-by-key
primitives. It may record the estimated private scratch bytes needed for a
future key-sort and reduce pipeline, but it must not be described as
`device_resident` execution.

## File Map

Create:

```text
docs/plans/apple_metal_optimization_campaign6_plan.md
tests/test_apple_metal_campaign6_plan.py
tests/test_apple_metal_campaign6_assets.py
docs/benchmarks/reports/apple_metal_optimization_campaign6_2026-05-07.md
docs/benchmarks/data/apple_metal_optimization_campaign6_2026-05-07/raw/metal_benchmark_campaign6.json
docs/benchmarks/data/apple_metal_optimization_campaign6_2026-05-07/summary.json
```

Modify:

```text
AGENTS.md
CHANGELOG.md
README.md
docs/architecture/apple_accelerator.md
docs/benchmarks/protocol.md
docs/roadmap.md
src/metal/workspace_metal.hpp
src/metal/workspace_metal.mm
benchmarks/bench_metal_kernels.py
scripts/render_apple_metal_assets.py
scripts/validate.py
docs/benchmarks/plots/accelerator_landscape_with_rocm.svg
```

## Task 1: Planning Contract Registration

- [x] Add tests that require Campaign 6 plan registration in `scripts/validate.py`,
      `AGENTS.md`, `README.md`, `docs/roadmap.md`,
      `docs/architecture/apple_accelerator.md`, and `CHANGELOG.md`.
- [x] Register this plan in every source-of-truth index.
- [x] Keep the plan explicit that Campaign 6 is device-resident simplify
      groundwork, not a retained device-resident simplify implementation.

## Task 2: Private Metal Workspace Model

- [x] Replace the previous inert workspace reservation with a private
      `MetalWorkspace` class, `WorkspaceTimingMode`, and `WorkspaceSnapshot`.
- [x] Keep all Metal framework types inside `src/metal`; CPU-only public headers
      must not include Metal, Foundation, MPS, or MPSGraph headers.
- [x] Implement `reserve_bytes`, `reset`, `release`, and snapshot accounting for
      reserved bytes, high-watermark bytes, allocation count, and growth count.
- [x] Add `workspace_timing_mode_from_env()` for
      `FASTPAULI_METAL_BENCH_WORKSPACE_TIMING` with accepted values
      `absent`, `grow_inside_timing`, and `pre_reserved_outside_timing`.

## Task 3: Benchmark Groundwork Rows

- [x] Add `campaign6` simplify cases matching the Campaign 5 duplicate-heavy,
      duplicate-light, two-word, generic multi-word, and cancellation pressure.
- [x] Keep CPU default, CPU scalar, skipped CPU NEON, and retained
      `metal_simplify_transfer_reference` rows.
- [x] Add one `metal_simplify_workspace_probe` status row for each Campaign 6
      case, with the `status_only` boundary and a private workspace byte model.
- [x] Add `FASTPAULI_EXPERIMENTAL_METAL_SIMPLIFY_STRATEGY` as benchmark
      vocabulary for future candidate selectors. No selector may retain a
      device-resident implementation in Campaign 6.

## Task 4: Evidence, Docs, And Assets

- [x] Update `docs/benchmarks/protocol.md` with Campaign 6 row requirements.
- [x] Generate Campaign 6 benchmark JSON on the local Apple M4 Pro Metal source
      build with `FASTPAULI_VALIDATE_METAL=1`.
- [x] Render Campaign 6 summary JSON and the broad README landscape plot.
- [x] Publish a Campaign 6 report with commands, environment, benchmark rows,
      decision, and remaining headroom.
- [x] Update README, roadmap, architecture, changelog, and AGENTS routing.

## Task 5: Validation And Closeout

- [x] Run Campaign 6 focused tests.
- [x] Run `python scripts/validate.py` on the CPU/default environment.
- [x] Run `FASTPAULI_VALIDATE_METAL=1 python scripts/validate.py` from an
      elevated command context on Apple Silicon.
- [ ] Complete independent review, resolve blocking findings, revalidate,
      commit, merge locally to `main`, validate merged result, push, confirm CI,
      and delete the merged local feature branch.

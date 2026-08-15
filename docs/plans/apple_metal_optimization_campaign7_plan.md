# Apple Metal Campaign 7 Checked Device-Resident Simplify Primitive Stack Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate Apple Metal Campaign 7 checked device-resident simplify primitive stack behavior with a benchmark-only one-word simplify primitive stack covering checked sort, prefix-sum, reduce-by-key, and survivor compaction.

**Architecture:** Campaign 7 keeps the public Metal `DevicePauliSum.simplify(atol, rtol)` behavior on the Campaign 5 transfer-reference bridge. It adds a private benchmark-only hook under `src/metal` that returns a Metal `DevicePauliSum` from a device-resident one-word primitive stack when the input coefficients are exactly representable as signed fixed32 dyadic values and the fixed32 accumulated coefficient sums plus the squared magnitude threshold fit exact uint64 comparison. Apple Metal does not support `double` arithmetic in kernels on this host, so Campaign 7 must label the coefficient-domain limitation and must not promote this candidate as a general retained Metal simplify implementation.

**Tech Stack:** C++20, Objective-C++, Metal Shading Language, shared `MTLBuffer` storage, nanobind private test hooks, pytest, `benchmarks/bench_metal_kernels.py`, `scripts/render_apple_metal_assets.py`, and `scripts/validate.py`.

**Status:** Completed.

---

Date: 2026-05-07

Campaign 6 retained the private workspace and benchmark vocabulary needed before
a Metal device-resident simplify attempt. Campaign 7 addresses the next
remaining-headroom item: checked Metal sort, prefix-sum, and reduce-by-key
primitives. The campaign is intentionally narrow and evidence-driven.

## Scope

Campaign 7 covers:

```text
benchmark-only one-word packed-key sort primitive
prefix-sum primitive for survivor compaction
reduce-by-key primitive for duplicate coefficient summation
deterministic canonical output order for checked one-word candidate rows
private nanobind hook for benchmark and validation only
campaign7 simplify benchmark cases with fixed-dyadic coefficients
unavailable row for multi-word inputs
Campaign 7 raw JSON, summary JSON, report, and README landscape refresh
validation entrypoint coverage for the Campaign 7 benchmark smoke
```

Out of scope:

```text
public Metal workspaces, queues, command buffers, heaps, events, or async APIs
public raw Metal buffer export, DLPack, PyTorch MPS tensor export, or Metal Array Interface-style objects
changing public DevicePauliSum.simplify() behavior unless the candidate is correct, general enough, and faster with evidence
claiming fixed-dyadic benchmark-only rows as a general FP64 Metal simplify implementation
generic multi-word Metal simplify
Metal statevector expectation
Metal matmul
Metal wheels, PyPI publication, Windows support, or older macOS compatibility
generic Apple GPU support claims from one Apple M4 Pro host
```

## Accepted Campaign 7 Boundary

The public behavior remains:

```text
Metal DevicePauliSum -> host PauliSum -> CPU PauliSum.simplify() -> Metal DevicePauliSum
```

The new benchmark-only candidate row is:

```text
variant: metal_simplify_device_candidate
operation: simplify
transfer_boundary: device_resident
metal_simplify_strategy: device_candidate
metal_simplify_strategy_status: benchmark_only
metal_simplify_coefficient_domain: signed_fixed32_dyadic_coefficients_only
```

The candidate may only report `status: ok` when it returns a Metal
`DevicePauliSum`, the host materialized result matches CPU `PauliSum.simplify()`
with deterministic canonical output order, and the row records the primitive
stack. Multi-word inputs and non-fixed-dyadic coefficients must be reported as
unavailable or rejected with evidence. Inputs whose worst-case fixed32 duplicate
sum could overflow signed accumulation, or whose nonzero tolerance comparison
cannot be represented as an exact uint64 squared-magnitude comparison, must also
be rejected with evidence.

do not change public DevicePauliSum.simplify() behavior unless a later slice
proves a device-resident path that is correct for the public coefficient domain,
beats the transfer-reference boundary, and has review approval.

## File Map

Create:

```text
docs/plans/apple_metal_optimization_campaign7_plan.md
src/metal/simplify_metal.hpp
src/metal/simplify_metal.mm
src/metal/kernels/simplify.metal
tests/test_apple_metal_campaign7_plan.py
tests/test_apple_metal_campaign7.py
tests/test_apple_metal_campaign7_assets.py
docs/benchmarks/reports/apple_metal_optimization_campaign7_2026-05-07.md
docs/benchmarks/data/apple_metal_optimization_campaign7_2026-05-07/raw/metal_benchmark_campaign7.json
docs/benchmarks/data/apple_metal_optimization_campaign7_2026-05-07/summary.json
```

Modify:

```text
AGENTS.md
CHANGELOG.md
README.md
docs/architecture/apple_accelerator.md
docs/benchmarks/protocol.md
docs/roadmap.md
CMakeLists.txt
bindings/python/module.cpp
include/fastpauli/device_pauli_sum.hpp
benchmarks/bench_metal_kernels.py
scripts/render_apple_metal_assets.py
scripts/validate.py
docs/benchmarks/plots/accelerator_landscape_with_rocm.svg
```

## Task 1: Planning Contract Registration

- [x] Add tests requiring Campaign 7 plan registration in `scripts/validate.py`,
      `AGENTS.md`, `README.md`, `docs/roadmap.md`,
      `docs/architecture/apple_accelerator.md`, and `CHANGELOG.md`.
- [x] Register this plan in every source-of-truth index.
- [x] Keep the plan explicit that Campaign 7 is benchmark-only until a general
      retained public Metal simplify path is justified.

## Task 2: Checked Metal Primitive Stack

- [x] Add private `src/metal/simplify_metal.hpp`,
      `src/metal/simplify_metal.mm`, and `src/metal/kernels/simplify.metal`.
- [x] Implement one-word key initialization and bitonic packed-key sorting.
- [x] Implement a Hillis-Steele inclusive prefix-sum primitive for uint32
      survivor/head flags.
- [x] Implement head-parallel reduce-by-key for duplicate coefficient summation.
- [x] Implement prefix-compacted survivor output with canonical sorted keys.
- [x] Keep all Metal framework types inside `src/metal`; CPU-only public headers
      must not include Metal, Foundation, MPS, or MPSGraph headers.

## Task 3: Private Benchmark Hook

- [x] Add a private nanobind hook:
      `_metal_simplify_words1_candidate_for_testing`.
- [x] Return a Metal `DevicePauliSum` when `include_output=True` and the
      candidate is available.
- [x] Report unavailable for multi-word inputs.
- [x] Report the fixed-dyadic coefficient-domain limitation because this Apple
      Metal toolchain rejects `double` in kernels.

## Task 4: Evidence, Docs, And Assets

- [x] Add `campaign7` benchmark cases and candidate rows.
- [x] Update `docs/benchmarks/protocol.md` with Campaign 7 row requirements.
- [x] Generate Campaign 7 benchmark JSON on the local Apple M4 Pro Metal source
      build with `FASTPAULI_VALIDATE_METAL=1`.
- [x] Render Campaign 7 summary JSON and the broad README landscape plot.
- [x] Publish a Campaign 7 report with commands, environment, benchmark rows,
      decision, and remaining headroom.
- [x] Update README, roadmap, architecture, changelog, and AGENTS routing.

## Task 5: Validation And Closeout

- [x] Run Campaign 7 focused tests.
- [x] Run `python scripts/validate.py` on the CPU/default environment.
- [x] Run `FASTPAULI_VALIDATE_METAL=1 python scripts/validate.py` from an
      elevated command context on Apple Silicon.
- [x] Complete independent review, resolve blocking findings, revalidate,
      commit, merge locally to `main`, validate merged result, push, confirm CI,
      and delete the merged local feature branch.

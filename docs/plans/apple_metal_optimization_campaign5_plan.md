# Apple Metal Campaign 5 Metal Simplify Bring-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first correct source-build-only Apple Metal `DevicePauliSum.simplify()` lane, benchmark it honestly, and record whether it is a performance path or only a correctness bridge.

**Architecture:** Campaign 5 extends the existing `backend="metal"` object model without adding new public APIs. The accepted baseline is a synchronous Metal-build implementation that preserves `DevicePauliSum.simplify(atol, rtol)` semantics and returns a Metal `DevicePauliSum`; any transfer-assisted implementation must be labeled as such and must not be reported as a device-resident speedup. Device-resident Metal sort/reduce work stays benchmark-only until it proves correctness and performance against the retained baseline.

**Tech Stack:** C++20, Objective-C++, Metal shared `MTLBuffer` storage, nanobind, pytest, `benchmarks/bench_metal_kernels.py`, `scripts/render_apple_metal_assets.py`, and `scripts/validate.py`.

**Status:** Completed in
`docs/benchmarks/reports/apple_metal_optimization_campaign5_2026-05-06.md`.
This plan remains as execution provenance; use the report, checked benchmark
JSON, and current source code for Campaign 5 closeout evidence.

---

Date: 2026-05-06

This plan follows
`docs/benchmarks/reports/apple_metal_optimization_campaign4_2026-05-06.md`.
Campaign 4 hardened commutation and compact consumers; Campaign 5 moves to the
next operation in `docs/architecture/apple_accelerator.md`: Metal simplify.

## Scope

Campaign 5 covers:

```text
Metal source-build DevicePauliSum.simplify(atol, rtol) behavior
CPU/Metal semantic parity for empty, single-term, duplicate-heavy, cancellation, one-word, two-word, and generic multi-word operators
finite non-negative tolerance validation that matches CPU, CUDA, and HIP behavior
honest timing boundaries for any host-assisted simplify path
benchmark-only feasibility rows for device-resident simplify candidates only when they are correct
Campaign 5 raw JSON, summary JSON, report, and README landscape refresh
validation entrypoint coverage for the new Campaign 5 Metal smoke
```

Out of scope:

```text
PyPI publication, Windows support, and older macOS compatibility
Metal wheels
public Metal queues, command buffers, events, heaps, graphs, streams, or workspaces
raw Metal buffer export, DLPack, PyTorch MPS tensor export, or Metal Array Interface-style objects
Metal statevector expectation
Metal matmul
generic Apple GPU support claims from one Apple M4 Pro host
promoting a device-resident custom sort/reduce implementation without benchmark evidence
MPSGraph or PyTorch MPS as FastPauli backend identities
```

## Accepted Simplify Semantics

Metal simplify must match `PauliSum.simplify()`:

```text
combine duplicate packed Pauli keys by summing complex128 coefficients
drop terms where abs(summed_coeff) <= atol + rtol * max_abs_input_coeff
return deterministic canonical packed-key order
return an empty PauliSum with the original num_qubits when every term is dropped
preserve backend="metal" on the returned DevicePauliSum
raise invalid_argument for negative, NaN, or infinite atol or rtol
raise moved-from runtime errors consistently with other DevicePauliSum methods
```

The first retained Campaign 5 implementation may be transfer-assisted:

```text
Metal DevicePauliSum -> host PauliSum -> CPU PauliSum.simplify() -> Metal DevicePauliSum
```

If this path is retained, every benchmark row and report paragraph must call it
`metal_simplify_transfer_reference` and record the transfer boundary
`device_to_host_cpu_simplify_host_to_device`. It must not be described as a GPU
sort/reduce or device-resident simplify path.

## File Map

Create:

```text
docs/plans/apple_metal_optimization_campaign5_plan.md
tests/test_apple_metal_campaign5_plan.py
```

Modify during the planning slice:

```text
AGENTS.md
CHANGELOG.md
README.md
docs/architecture/apple_accelerator.md
docs/roadmap.md
scripts/validate.py
```

Modify during the implementation slice:

```text
src/metal/device_pauli_sum_metal.mm
src/metal/device_pauli_sum_metal.hpp
tests/test_apple_metal_foundation.py
tests/test_apple_metal_campaign5.py
benchmarks/bench_metal_kernels.py
benchmarks/_benchmark_metadata.py
scripts/render_apple_metal_assets.py
scripts/validate.py
README.md
docs/architecture/apple_accelerator.md
docs/benchmarks/protocol.md
docs/roadmap.md
docs/benchmarks/reports/apple_metal_optimization_campaign5_2026-05-06.md
docs/benchmarks/data/apple_metal_optimization_campaign5_2026-05-06/raw/metal_benchmark_campaign5.json
docs/benchmarks/data/apple_metal_optimization_campaign5_2026-05-06/summary.json
docs/benchmarks/plots/accelerator_landscape_with_rocm.svg
```

## Task 1: Planning Contract Registration

**Files:**

```text
Create: docs/plans/apple_metal_optimization_campaign5_plan.md
Create: tests/test_apple_metal_campaign5_plan.py
Modify: AGENTS.md
Modify: README.md
Modify: docs/roadmap.md
Modify: docs/architecture/apple_accelerator.md
Modify: CHANGELOG.md
Modify: scripts/validate.py
```

- [ ] **Step 1: Write the plan-registration test**

Add `tests/test_apple_metal_campaign5_plan.py`:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = "docs/plans/apple_metal_optimization_campaign5_plan.md"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def normalized(path: str) -> str:
    return " ".join(read(path).split())


def test_campaign5_plan_is_registered_as_source_of_truth() -> None:
    validate_path = ROOT / "scripts" / "validate.py"
    spec = importlib.util.spec_from_file_location("fastpauli_validate", validate_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert (ROOT / PLAN_PATH).exists()
    assert PLAN_PATH in module.SOURCE_OF_TRUTH_PATHS
    assert PLAN_PATH in read("README.md")
    assert PLAN_PATH in read("AGENTS.md")
    assert PLAN_PATH in read("docs/roadmap.md")

    plan = normalized(PLAN_PATH)
    for required in (
        "Metal source-build DevicePauliSum.simplify(atol, rtol) behavior",
        "metal_simplify_transfer_reference",
        "device_to_host_cpu_simplify_host_to_device",
        "finite non-negative tolerance validation",
        "benchmark-only feasibility rows for device-resident simplify candidates",
        "PyPI publication, Windows support, and older macOS compatibility",
        "Metal statevector expectation",
        "Metal matmul",
    ):
        assert required in plan
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/test_apple_metal_campaign5_plan.py -q
```

Expected result before registration:

```text
FAILED tests/test_apple_metal_campaign5_plan.py::test_campaign5_plan_is_registered_as_source_of_truth
```

- [ ] **Step 3: Register the plan in source-of-truth docs**

Add `docs/plans/apple_metal_optimization_campaign5_plan.md` to:

```text
scripts/validate.py SOURCE_OF_TRUTH_PATHS
AGENTS.md Read First list
README.md Apple Metal plan list and implementation-lane paragraph
docs/roadmap.md Source Documents and Current Status
CHANGELOG.md Unreleased Added
```

- [ ] **Step 4: Run the new test and docs checks**

Run:

```bash
.venv/bin/python -m pytest tests/test_apple_metal_campaign5_plan.py -q
git diff --check
```

Expected result:

```text
1 passed
git diff --check exits 0
```

## Task 2: Metal Simplify Semantic Tests

**Files:**

```text
Modify: tests/test_apple_metal_foundation.py
Create: tests/test_apple_metal_campaign5.py
```

- [ ] **Step 1: Add runtime-gated simplify parity tests**

Add cases that use existing helpers from `tests/test_apple_metal_foundation.py`
or duplicate their local equivalents in `tests/test_apple_metal_campaign5.py`:

```python
@pytest.mark.parametrize(
    ("labels", "coeffs", "atol", "rtol"),
    [
        ([], [], 1.0e-12, 0.0),
        (["XYZ"], [1.0 + 0.0j], 1.0e-12, 0.0),
        (["XYZ", "XYZ", "III"], [1.0, 2.0, 0.0], 1.0e-12, 0.0),
        (["XII", "XII", "ZII"], [1.0, -1.0, 0.25], 1.0e-12, 0.0),
        ([_multiword_label(130, {0: "X", 65: "Z"}), _multiword_label(130, {0: "X", 65: "Z"})], [1.0, 3.0j], 1.0e-12, 0.0),
        (["XYZ", "XYZ"], [1.0e-8, -0.5e-8], 1.0e-9, 0.5),
    ],
)
def test_metal_simplify_matches_cpu_when_available(labels, coeffs, atol, rtol):
    _require_metal_runtime()
    host = fastpauli.PauliSum.empty(num_qubits=7) if not labels else fastpauli.PauliSum.from_labels(labels, coeffs)
    expected = host.simplify(atol=atol, rtol=rtol)
    actual = host.to_device(backend="metal").simplify(atol=atol, rtol=rtol).to_host()
    _assert_same_operator(actual, expected)
```

- [ ] **Step 2: Add tolerance validation tests**

Use the same invalid tolerance set covered by CUDA and HIP:

```python
@pytest.mark.parametrize("bad_value", [-1.0, float("nan"), float("inf")])
def test_metal_simplify_rejects_invalid_tolerances_when_available(bad_value):
    _require_metal_runtime()
    device_op = fastpauli.PauliSum.from_labels(["X"], [1.0]).to_device(backend="metal")
    with pytest.raises(ValueError, match="simplify tolerances must be non-negative finite values"):
        device_op.simplify(atol=bad_value)
    with pytest.raises(ValueError, match="simplify tolerances must be non-negative finite values"):
        device_op.simplify(rtol=bad_value)
```

- [ ] **Step 3: Verify the tests fail before implementation**

Run with a Metal source build:

```bash
FASTPAULI_VALIDATE_METAL=1 .venv/bin/python -m pytest tests/test_apple_metal_campaign5.py -q
```

Expected failure:

```text
FastPauli Metal backend does not implement simplify
```

## Task 3: Retained Metal Simplify Behavior

**Files:**

```text
Modify: src/metal/device_pauli_sum_metal.hpp
Modify: src/metal/device_pauli_sum_metal.mm
```

- [ ] **Step 1: Add Metal simplify tolerance validation**

Add this helper to `namespace metal_detail` in `src/metal/device_pauli_sum_metal.hpp`:

```cpp
inline void validate_simplify_tolerances(double atol, double rtol) {
  if (atol < 0.0 || rtol < 0.0 || !std::isfinite(atol) || !std::isfinite(rtol)) {
    throw std::invalid_argument("simplify tolerances must be non-negative finite values");
  }
}
```

Also add:

```cpp
#include <cmath>
#include <stdexcept>
```

- [ ] **Step 2: Implement the retained transfer-reference simplify**

Replace the unsupported implementation in `src/metal/device_pauli_sum_metal.mm`:

```cpp
DevicePauliSum DevicePauliSum::simplify(double atol, double rtol) const {
  if (!impl_) {
    throw std::runtime_error("DevicePauliSum is empty or moved-from");
  }
  metal_detail::validate_simplify_tolerances(atol, rtol);

  PauliSum simplified = to_host().simplify(atol, rtol);
  return DevicePauliSum::from_host(simplified, AcceleratorBackend::Metal, impl_->device_ordinal);
}
```

This is intentionally a correctness bridge. It keeps the returned object on the
Metal backend, but its benchmark boundary is transfer-assisted, not
device-resident GPU simplify.

- [ ] **Step 3: Run the Metal simplify tests**

Run:

```bash
FASTPAULI_VALIDATE_METAL=1 .venv/bin/python -m pytest tests/test_apple_metal_campaign5.py tests/test_apple_metal_foundation.py -q
```

Expected result:

```text
all non-skipped Metal simplify and foundation tests pass
```

## Task 4: Campaign 5 Benchmark Profile

**Files:**

```text
Modify: benchmarks/bench_metal_kernels.py
Modify: benchmarks/_benchmark_metadata.py
```

- [ ] **Step 1: Add the transfer boundary label**

Add this boundary to `benchmarks/_benchmark_metadata.py`:

```python
"device_to_host_cpu_simplify_host_to_device",
```

- [ ] **Step 2: Add Campaign 5 simplify cases**

Add a `campaign5` profile in `benchmarks/bench_metal_kernels.py` with these
case names and dimensions:

```text
metal_campaign5_simplify_words1_duplicate_heavy_8192_terms: 64 qubits, 8192 terms, 1 packed word, duplicate_rate 0.85
metal_campaign5_simplify_words1_duplicate_light_8192_terms: 64 qubits, 8192 terms, 1 packed word, duplicate_rate 0.05
metal_campaign5_simplify_words2_duplicate_heavy_4096_terms: 96 qubits, 4096 terms, 2 packed words, duplicate_rate 0.70
metal_campaign5_simplify_generic_multiword_2048_terms: 130 qubits, 2048 terms, 3 packed words, duplicate_rate 0.50
metal_campaign5_simplify_cancellation_4096_terms: 64 qubits, 4096 terms, coefficients chosen so many duplicate sums cancel
```

Each row must include:

```text
cpu_default
cpu_neon when available
metal_simplify_transfer_reference
```

If a device-resident prototype is added, it must use a separate variant name:

```text
metal_simplify_device_candidate
```

That row remains benchmark-only unless it is correct and beats the transfer
reference and same-host CPU baselines on the planned cases.

- [ ] **Step 3: Add correctness checks to the benchmark**

For every simplify row:

```python
expected = op.simplify()
actual = device_op.simplify().to_host()
assert_same_operator(actual, expected)
```

The benchmark must fail loudly on semantic mismatch before recording timing.

- [ ] **Step 4: Run smoke and evidence benchmarks**

Run:

```bash
FASTPAULI_VALIDATE_METAL=1 .venv/bin/python benchmarks/bench_metal_kernels.py --profile campaign5 --repeat 1 --json
FASTPAULI_VALIDATE_METAL=1 .venv/bin/python benchmarks/bench_metal_kernels.py --profile campaign5 --repeat 10 --json --output docs/benchmarks/data/apple_metal_optimization_campaign5_2026-05-06/raw/metal_benchmark_campaign5.json
```

Expected result:

```text
profile == "campaign5"
status == "ok"
every simplify row records correct == true
```

## Task 5: Report, Plot, And Validation Wiring

**Files:**

```text
Modify: scripts/render_apple_metal_assets.py
Modify: scripts/validate.py
Modify: README.md
Modify: docs/roadmap.md
Modify: docs/architecture/apple_accelerator.md
Modify: docs/benchmarks/protocol.md
Create: docs/benchmarks/reports/apple_metal_optimization_campaign5_2026-05-06.md
Create: docs/benchmarks/data/apple_metal_optimization_campaign5_2026-05-06/summary.json
Modify: docs/benchmarks/plots/accelerator_landscape_with_rocm.svg
```

- [ ] **Step 1: Extend the Apple Metal renderer**

Add Campaign 5 recognition to `scripts/render_apple_metal_assets.py`:

```python
LATEST_APPLE_METAL_CAMPAIGN = "apple_metal_optimization_campaign5"
```

Add a `CAMPAIGN_CONFIGS["apple_metal_optimization_campaign5"]` entry with:

```python
{
    "date": "2026-05-06",
    "label": "Apple Metal Campaign 5",
    "limitations": [
        "Campaign 5 keeps the source-build-only Metal API boundary.",
        "Campaign 5 retained simplify is a transfer-reference correctness bridge unless a device-resident candidate is proven correct and faster.",
        "Metal statevector expectation, Metal matmul, Metal wheels, PyPI publication, Windows support, and older macOS compatibility are out of scope.",
    ],
    "profiler": {
        "status": "not_recaptured",
        "source": "docs/benchmarks/data/apple_metal_optimization_campaign4_2026-05-06/profiler/metal_campaign4_profiler_evidence.json",
        "remaining": "Campaign 5 simplify profiling should focus on timing-boundary evidence before shader-counter capture.",
    },
}
```

Extend `infer_campaign()`:

```python
if "apple_metal_optimization_campaign5" in name:
    return "apple_metal_optimization_campaign5"
```

Add variant labels:

```python
"metal_simplify_transfer_reference": "Apple Metal simplify transfer reference",
"metal_simplify_device_candidate": "Apple Metal simplify device candidate",
```

Add a Campaign 5 required landscape series set:

```python
CAMPAIGN5_EXTRA_LANDSCAPE_SERIES = {
    *CAMPAIGN4_EXTRA_LANDSCAPE_SERIES,
    "Apple Metal simplify transfer reference",
}
```

Update `required_landscape_series()` so Campaign 5 requires
`CAMPAIGN5_EXTRA_LANDSCAPE_SERIES`.

Add simplify-specific scale formatting in `apple_landscape_rows()` instead of
using pairwise-only `lhs_terms x rhs_terms` text for every operation:

```python
def apple_case_scale(case: dict[str, Any], row: dict[str, Any]) -> str:
    if row.get("operation") == "simplify":
        return (
            f'{case.get("num_terms")} terms before simplify, '
            f'{case.get("output_terms", "unknown")} survivor terms, '
            f'duplicate_rate={case.get("duplicate_rate", "unknown")}, '
            f'{case.get("num_qubits")} qubits, {case.get("packed_words")} words'
        )
    return (
        f'{case.get("lhs_terms")}x{case.get("rhs_terms")} terms, '
        f'{case.get("num_qubits")} qubits, {case.get("packed_words")} words'
    )
```

Use `apple_case_scale(case, row)` when creating the landscape item. If Campaign
5 includes `metal_simplify_device_candidate`, keep it out of the required
series set until the candidate is correct and retained.

- [ ] **Step 2: Add the Campaign 5 validation smoke**

Add a Metal validation command in `scripts/validate.py` after Campaign 4 smoke:

```python
run_command(
    "Apple Metal Campaign 5 simplify benchmark smoke",
    [
        str(python),
        "benchmarks/bench_metal_kernels.py",
        "--profile",
        "campaign5",
        "--repeat",
        "1",
        "--json",
    ],
    env=metal_env,
)
```

- [ ] **Step 3: Update the benchmark protocol**

Add an Apple Metal Campaign 5 subsection to `docs/benchmarks/protocol.md` that
requires:

```text
operation: simplify
variant: cpu_default, cpu_neon, metal_simplify_transfer_reference, or metal_simplify_device_candidate
transfer_boundary: host_materialized, device_to_host_cpu_simplify_host_to_device, or device_resident
metal_simplify_strategy: cpu_reference, transfer_reference, or device_candidate
metal_simplify_strategy_status: retained, benchmark_only, rejected_with_evidence, or unavailable
num_terms
output_terms
duplicate_rate
atol
rtol
correct
timing median/min/max seconds
```

The protocol text must state that
`metal_simplify_transfer_reference` uses the
`device_to_host_cpu_simplify_host_to_device` boundary and may not be reported as
a device-resident GPU duplicate-reduction speedup.

- [ ] **Step 4: Write the Campaign 5 report**

The report must include:

```text
commands
environment
semantic coverage
retained implementation boundary
CPU default and CPU NEON timings
Metal transfer-reference timings
any device-resident candidate timing and decision
README landscape plot
remaining headroom
```

The report must contain this sentence when the transfer-reference path is
retained:

```text
The retained Metal simplify implementation is a correctness bridge, not a device-resident GPU duplicate-reduction path.
```

- [ ] **Step 5: Generate summary and plot assets**

Run:

```bash
.venv/bin/python scripts/render_apple_metal_assets.py \
  --data-dir docs/benchmarks/data/apple_metal_optimization_campaign5_2026-05-06 \
  --plot-dir docs/benchmarks/plots
```

Expected result:

```text
docs/benchmarks/data/apple_metal_optimization_campaign5_2026-05-06/summary.json exists
docs/benchmarks/plots/accelerator_landscape_with_rocm.svg includes Apple Metal simplify transfer reference
```

## Task 6: Final Validation And Review

**Files:**

```text
All files touched by Campaign 5
```

- [ ] **Step 1: Run local validation**

Run:

```bash
git diff --check
.venv/bin/python scripts/validate.py
FASTPAULI_VALIDATE_METAL=1 .venv/bin/python scripts/validate.py
```

Expected result:

```text
all commands exit 0
```

- [ ] **Step 2: Request independent review**

Review is required because Campaign 5 changes architecture, user-facing
behavior claims, benchmark protocol surfaces, native Objective-C++ behavior,
and performance evidence.

Reviewer scope:

```text
Metal simplify semantics and tolerance behavior
correctness of transfer-reference implementation
honesty of benchmark timing boundaries
absence of unsupported Metal wheel or generic Apple GPU claims
source-of-truth doc routing
CPU-only validation safety
```

- [ ] **Step 3: Merge, push, and confirm CI**

Use the normal FastPauli closeout:

```bash
git switch main
git merge --ff-only codex/apple-metal-campaign5
.venv/bin/python scripts/validate.py
FASTPAULI_VALIDATE_METAL=1 .venv/bin/python scripts/validate.py
git push origin main
gh run watch <run-id> --exit-status
git branch -d codex/apple-metal-campaign5
```

Expected result:

```text
main is pushed
CI is green
the feature branch is deleted
the worktree is clean
```

## Remaining Headroom After Campaign 5

Campaign 5 should leave only evidence-backed next work:

```text
custom device-resident Metal duplicate reduction only if transfer-reference evidence shows simplify is worth optimizing
device-resident sort/scan primitives only after their lifetime and scratch-buffer boundaries are designed
Metal statevector expectation after simplify is correct and the source-build lane stays stable
Metal matmul after simplify and expectation evidence exists
additional Apple Silicon generation validation before changing default selector policies
sanitized shader-counter exports when Instruments can emit narrow value CSVs without raw trace retention
```

# MI300X ROCm Optimization Campaign 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and benchmark HIP `DevicePauliSum.simplify()` on MI300X while giving every Campaign 2 remaining-headroom and correctness-risk item a recorded terminal status.

**Architecture:** Campaign 3 keeps ROCm/HIP source-build-only, single-backend, and mutually exclusive with CUDA. The only retained public expansion allowed in this campaign is HIP `DevicePauliSum.simplify()` with CPU-equivalent canonical ordering, tolerance filtering, and device-resident output. HIP DLPack, public streams, public workspaces, expectation, matmul, multi-GPU, ROCm wheels, additional AMD GPU support claims, and simultaneous CUDA+HIP builds remain outside implementation scope; Campaign 3 may only give those items terminal statuses and next triggers for separate follow-on plans. The implementation should mirror the proven CUDA simplify structure where it fits HIP/rocThrust, but every retained HIP path must be justified with MI300X correctness, benchmark, and rocprof evidence.

**Tech Stack:** C++20, nanobind, CMake HIP language support, ROCm/HIP runtime, rocThrust, optional hipCUB benchmark probes when available, AMD Instinct MI300X `gfx942`, rocprof, pytest, NumPy, existing FastPauli CPU/CUDA/ROCm benchmark-report infrastructure.

---

## Status

```text
complete
report: docs/benchmarks/reports/rocm_mi300x_campaign3_2026-04-30.md
evidence: docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30/
plots: docs/benchmarks/plots/rocm_mi300x_campaign3_simplify.svg
       docs/benchmarks/plots/accelerator_landscape_with_rocm.svg
```

## Baseline

Campaign 2 is complete:

```text
plan: docs/plans/mi300x_rocm_optimization_campaign2_plan.md
report: docs/benchmarks/reports/rocm_mi300x_campaign2_2026-04-30.md
accepted HIP public surface: transfers, backend metadata, pairwise commutation, device-resident commutation matrices, dense to_host, compact count/conflict consumers
```

Campaign 2 retained these remaining-headroom items:

```text
define and validate HIP DLPack only if a named ROCm consumer and stream/ownership contract are accepted
add HIP simplify after deciding whether rocThrust, hipCUB, or custom duplicate reduction is the retained implementation path
add HIP expectation and matmul after CPU/CUDA semantic parity tests are promoted to HIP
profile packed or bitset commutation summaries only if a public consumer can use them without forcing dense host materialization
refresh the broad README performance landscape once ROCm rows can be added without replacing the cross-path CPU/CUDA/external view
evaluate additional AMD GPUs only when portability evidence is needed for a release or support claim
```

Campaign 3 must not leave these as informal chat state. The report must give
each item a terminal status:

```text
accepted
rejected_with_evidence
deferred_with_blocker
out_of_scope_with_next_trigger
blocked_external
```

`accepted` is valid only for retained Campaign 3 scope such as HIP simplify,
strategy selection, and the README landscape refresh. Hard out-of-scope items
must use `rejected_with_evidence`, `deferred_with_blocker`,
`out_of_scope_with_next_trigger`, or `blocked_external`.

## Campaign 3 Scope

In scope:

```text
HIP DevicePauliSum.simplify(atol=1e-12, rtol=0.0)
device-resident HIP simplified output returned as DevicePauliSum
empty input and all-zero output preservation of num_qubits
negative, NaN, and infinite tolerance rejection
canonical packed-word ordering identical to PauliSum.simplify()
inclusive tolerance threshold matching CPU semantics
one-word <=32-qubit packed-key path
one-word wider-than-32-qubit key path
two-word key path
generic multi-word fallback path
rocThrust retained implementation path unless hipCUB/custom experiments prove a safer faster retained path
benchmark-only hipCUB/custom duplicate-reduction probes when available and bounded
MI300X benchmark rows for transfer-inclusive simplify, device-resident simplify, to_host materialization, duplicate-heavy, duplicate-light, wide-qubit, and all-zero regimes
rocprof trace/stats/counter evidence for retained simplify kernels and library calls
README broad performance landscape refresh that adds ROCm rows without replacing CPU/CUDA/external comparisons
report terminal statuses for HIP DLPack, streams, workspaces, packed summaries, expectation, matmul, multi-GPU, portability, ROCm wheels, and simultaneous CUDA+HIP source builds
```

Hard out of implementation scope for Campaign 3:

```text
public HIP DLPack or __dlpack_device__
CUDA Array Interface exposure from HIP objects
public HIP stream parameters
public HIP graph/capture APIs
public HIP workspace handles
HIP expectation kernels
HIP matmul kernels
multi-GPU MI300X execution
ROCm binary wheels
additional AMD GPU support claims
simultaneous CUDA+HIP source builds
```

Campaign 3 may record evidence, blockers, rejected reasons, and next triggers
for the hard out-of-scope items above. It must not retain public APIs,
packaging claims, portability claims, or backend-object-model changes for them.

## Evidence Layout

Use this evidence root:

```text
docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30/
docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30/logs/
docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30/raw/
docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30/profiler/
docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30/summary.json
docs/benchmarks/reports/rocm_mi300x_campaign3_2026-04-30.md
docs/benchmarks/plots/rocm_mi300x_campaign3_simplify.svg
docs/benchmarks/plots/accelerator_landscape_with_rocm.svg
```

If execution crosses UTC midnight, keep this evidence root only if the report
states the exact execution dates and why the root remains named `2026-04-30`.

## Acceptance Criteria

Campaign 3 is complete only when every applicable item has a terminal status in
the Campaign 3 report:

```text
local CPU-only validation passes with FASTPAULI_ENABLE_HIP=OFF
HIP source build succeeds on MI300X with FASTPAULI_ENABLE_HIP=ON and FASTPAULI_HIP_ARCHITECTURES=gfx942
CUDA+HIP configure-time rejection still passes
public headers still contain no HIP or ROCm runtime headers
HIP DevicePauliSum.simplify() matches CPU simplify for empty, all-zero, scalar, one-word <=32-qubit, one-word >32-qubit, two-word, multi-word, duplicate-heavy, duplicate-light, tolerance-boundary, and randomized cases
HIP DevicePauliSum.simplify() rejects negative, NaN, and infinite tolerances with the same public error class as CUDA/CPU where applicable
HIP simplified output stays device-resident until to_host() is explicitly called
HIP build metadata lists simplify in hip_kernels only when FASTPAULI_ENABLE_HIP=ON
HIP expectation, matmul, DLPack, streams, workspaces, multi-GPU, portability, ROCm wheel, and simultaneous CUDA+HIP claims remain unavailable and receive terminal statuses with named next triggers
benchmark JSON separates transfer-inclusive, device-resident, and to_host materialization boundaries
benchmark JSON records retained duplicate-reduction strategy and rejected strategy reasons
rocprof trace/stats/counter evidence exists for the retained simplify path, or an exact tool/provider diagnosis is checked in
README broad performance landscape includes ROCm rows without dropping CPU, CUDA, and external rows
independent review is recorded before merge
```

## Task 1: Contracts, Decision Gates, And Failing Tests

**Files:**
- Modify: `docs/architecture/rocm_backend.md`
- Modify: `docs/benchmarks/protocol.md`
- Modify: `tests/test_phase12_rocm_foundation.py`
- Test: `python -m pytest tests/test_phase12_rocm_foundation.py -q`

- [ ] **Step 1: Add Campaign 3 public-boundary contract**

Add a `Campaign 3 Simplify Boundary` section to
`docs/architecture/rocm_backend.md` with these decisions:

```text
The next accepted HIP public operation is DevicePauliSum.simplify().
HIP simplify must return a HIP-backed DevicePauliSum and must not implicitly copy the simplified operator to host.
HIP simplify must match PauliSum.simplify() canonical ordering, coefficient summation, and tolerance filtering.
rocThrust is the default retained duplicate-reduction implementation path unless Campaign 3 evidence accepts hipCUB or a custom path.
hipCUB and custom duplicate-reduction probes are benchmark-only until they pass CPU/HIP equivalence, allocation accounting, and rocprof evidence.
HIP DLPack, public streams, public workspaces, HIP expectation, HIP matmul, multi-GPU, ROCm wheels, additional AMD GPU support claims, and simultaneous CUDA+HIP builds remain unavailable in Campaign 3. The report must record terminal statuses and next triggers for them, but accepting any of those surfaces requires a separate follow-on plan or architecture decision.
```

- [ ] **Step 2: Add benchmark protocol fields**

Add a `ROCm Campaign 3 simplify fields` subsection to
`docs/benchmarks/protocol.md` requiring these JSON fields when the row status
is `ok`:

```text
operation: simplify
backend: hip
hip_simplify_transfer_seconds and p10/p90/min/max variants
hip_simplify_device_resident_seconds and p10/p90/min/max variants
hip_simplify_to_host_seconds and p10/p90/min/max variants
hip_simplify_strategy: rocthrust_default, hipcub_radix_sort_reduce, custom_packed_key, or unavailable
hip_simplify_strategy_status: retained, rejected_with_evidence, benchmark_only, or unavailable
hip_simplify_output_terms
hip_simplify_output_words
result_materialization_target: device_pauli_sum or host_pauli_sum
timing_boundary: transfer_inclusive, device_resident, or device_output_to_host
correctness_digest with input_terms, output_terms, coefficient_l1, and canonical_label_hash
campaign3_headroom_statuses with terminal status for DLPack, streams, workspaces, packed summaries, expectation, matmul, portability, ROCm wheels, multi-GPU, and simultaneous CUDA+HIP
```

- [ ] **Step 3: Add failing HIP simplify tests**

Append tests to `tests/test_phase12_rocm_foundation.py` using these helper
shapes. Keep the test names exact so remote evidence can target them:

```python
def _assert_hip_simplify_matches_cpu(
    op: fastpauli.PauliSum,
    *,
    atol: float = 1.0e-12,
    rtol: float = 0.0,
) -> None:
    expected = op.simplify(atol=atol, rtol=rtol)
    actual = op.to_device().simplify(atol=atol, rtol=rtol).to_host()
    assert _labels_and_coeffs(actual) == _labels_and_coeffs(expected)


def test_hip_simplify_matches_cpu_for_edge_cases_when_available() -> None:
    _require_hip_runtime()

    cases = [
        fastpauli.PauliSum.empty(num_qubits=5),
        fastpauli.PauliSum.from_labels(["I", "I"], [0.25, -0.25]),
        fastpauli.PauliSum.from_labels(["X", "X", "Z"], [1.0, -0.5, 2.0]),
        fastpauli.PauliSum.from_sparse_list(
            [("X", [33], 1.0), ("X", [33], 2.0), ("Z", [0], -1.0)],
            num_qubits=64,
        ),
        fastpauli.PauliSum.from_sparse_list(
            [("X", [64], 1.0), ("X", [64], -0.25), ("YZ", [1, 64], 0.5j)],
            num_qubits=65,
        ),
        fastpauli.PauliSum.from_sparse_list(
            [("XZ", [0, 129], 1.0), ("XZ", [0, 129], -2.0), ("Y", [128], 3.0)],
            num_qubits=130,
        ),
    ]

    for op in cases:
        _assert_hip_simplify_matches_cpu(op, atol=0.0, rtol=0.0)


def test_hip_simplify_tolerance_matches_cpu_when_available() -> None:
    _require_hip_runtime()

    op = fastpauli.PauliSum.from_labels(
        ["X", "X", "Z", "Z"],
        [1.0, -0.95, 2.0, -1.79],
    )
    _assert_hip_simplify_matches_cpu(op, atol=0.0, rtol=0.1)


def test_hip_simplify_randomized_matches_cpu_when_available() -> None:
    _require_hip_runtime()

    rng = np.random.default_rng(3942)
    alphabet = np.asarray(["I", "X", "Y", "Z"])
    for num_qubits, terms in [(9, 32), (70, 24), (130, 20)]:
        labels = ["".join(rng.choice(alphabet, size=num_qubits).tolist()) for _ in range(terms)]
        labels.extend(labels[: terms // 4])
        coeffs = [
            complex(float(rng.normal()), float(rng.normal()))
            for _ in range(len(labels))
        ]
        _assert_hip_simplify_matches_cpu(
            fastpauli.PauliSum.from_labels(labels, coeffs),
            atol=1.0e-11,
            rtol=1.0e-12,
        )


@pytest.mark.parametrize("bad_value", [-1.0, float("nan"), float("inf")])
def test_hip_simplify_rejects_invalid_tolerances_when_available(bad_value: float) -> None:
    _require_hip_runtime()

    op = fastpauli.PauliSum.from_labels(["X"], [1.0]).to_device()
    with pytest.raises(ValueError, match="tolerances"):
        op.simplify(atol=bad_value)
    with pytest.raises(ValueError, match="tolerances"):
        op.simplify(rtol=bad_value)


def test_hip_deferred_surfaces_remain_unavailable_after_simplify_when_available() -> None:
    _require_hip_runtime()

    device_op = fastpauli.PauliSum.from_labels(["X"], [1.0]).to_device()
    simplified = device_op.simplify()
    assert simplified.backend == "hip"

    with pytest.raises(RuntimeError, match="HIP expectation_statevector is not implemented yet"):
        simplified.expectation_statevector(np.asarray([1.0, 0.0], dtype=np.complex128))
    with pytest.raises(RuntimeError, match="HIP matmul is not implemented yet"):
        simplified.matmul(simplified)
```

- [ ] **Step 4: Verify the red step**

Run on the MI300X HIP build before implementation:

```bash
PATH=/opt/rocm/bin:$PATH .venv/bin/python -m pytest \
  tests/test_phase12_rocm_foundation.py::test_hip_simplify_matches_cpu_for_edge_cases_when_available \
  tests/test_phase12_rocm_foundation.py::test_hip_simplify_tolerance_matches_cpu_when_available \
  tests/test_phase12_rocm_foundation.py::test_hip_simplify_randomized_matches_cpu_when_available \
  tests/test_phase12_rocm_foundation.py::test_hip_simplify_rejects_invalid_tolerances_when_available \
  tests/test_phase12_rocm_foundation.py::test_hip_deferred_surfaces_remain_unavailable_after_simplify_when_available \
  -q
```

Expected: fails with `HIP simplify is not implemented yet` for the positive
simplify tests, while pre-existing HIP tests still pass.

- [ ] **Step 5: Commit**

```bash
git add docs/architecture/rocm_backend.md docs/benchmarks/protocol.md tests/test_phase12_rocm_foundation.py
git commit -m "test: specify HIP simplify behavior"
```

## Task 2: HIP Simplify Build Integration And Shared Helpers

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `scripts/validate.py`
- Modify: `bindings/python/module.cpp`
- Modify: `src/hip/device_pauli_sum.hip.hpp`
- Create: `src/hip/simplify_hip.hip.hpp`
- Create: `src/hip/simplify_hip.hip.cpp`
- Test: `python scripts/validate.py`

- [ ] **Step 1: Add HIP simplify sources to CMake**

Add `src/hip/simplify_hip.hip.cpp` to the HIP source list and to
`set_source_files_properties(... LANGUAGE HIP)`.

- [ ] **Step 2: Add HIP source inventory validation**

Add the new files to `HIP_FOUNDATION_SOURCES` in `scripts/validate.py`:

```python
"src/hip/simplify_hip.hip.cpp",
"src/hip/simplify_hip.hip.hpp",
```

- [ ] **Step 3: Report the retained kernel in build metadata**

Update `bindings/python/module.cpp` so HIP builds append `"simplify"` to
`hip_kernels` only when `FASTPAULI_BUILD_HIP_ENABLED` is true.

- [ ] **Step 4: Move tolerance validation into HIP helpers**

Add this helper to `src/hip/device_pauli_sum.hip.hpp`:

```cpp
inline void validate_simplify_tolerances(double atol, double rtol) {
  if (!std::isfinite(atol) || !std::isfinite(rtol) || atol < 0.0 || rtol < 0.0) {
    throw std::invalid_argument("simplify tolerances must be non-negative finite values");
  }
}
```

Include `<cmath>` in that header.

- [ ] **Step 5: Create HIP simplify source files**

Create `src/hip/simplify_hip.hip.hpp`:

```cpp
#pragma once

#include "device_pauli_sum.hip.hpp"

namespace fastpauli::hip_detail {

enum class DuplicateReductionStrategy {
  kRocThrustDefault,
  kHipCubRadixSortReduce,
  kCustomPackedKey,
};

DuplicateReductionStrategy duplicate_reduction_strategy_from_env();

}  // namespace fastpauli::hip_detail
```

Create `src/hip/simplify_hip.hip.cpp` with:

```cpp
#include "simplify_hip.hip.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace fastpauli {

namespace hip_detail {

DuplicateReductionStrategy duplicate_reduction_strategy_from_env() {
  const char* value = std::getenv("FASTPAULI_HIP_BENCH_DUPLICATE_REDUCTION");
  if (value == nullptr || std::string(value).empty() || std::string(value) == "rocthrust_default") {
    return DuplicateReductionStrategy::kRocThrustDefault;
  }
  const std::string setting(value);
  if (setting == "hipcub_radix_sort_reduce") {
    return DuplicateReductionStrategy::kHipCubRadixSortReduce;
  }
  if (setting == "custom_packed_key") {
    return DuplicateReductionStrategy::kCustomPackedKey;
  }
  throw std::invalid_argument(
      "FASTPAULI_HIP_BENCH_DUPLICATE_REDUCTION must be rocthrust_default, "
      "hipcub_radix_sort_reduce, or custom_packed_key");
}

}  // namespace hip_detail

DevicePauliSum DevicePauliSum::simplify(double atol, double rtol) const {
  if (!impl_) {
    throw std::runtime_error("DevicePauliSum is empty or moved-from");
  }
  hip_detail::validate_simplify_tolerances(atol, rtol);
  throw std::runtime_error("HIP simplify implementation is not wired yet");
}

}  // namespace fastpauli
```

Remove the old `DevicePauliSum::simplify(double, double)` stub from
`src/hip/device_pauli_sum.hip.cpp` so the symbol is defined only once.

- [ ] **Step 6: Validate CPU-only and HIP source inventory**

Run locally:

```bash
python scripts/validate.py
```

Expected: local CPU-only validation passes and source inventory accepts the new
HIP files.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt scripts/validate.py bindings/python/module.cpp src/hip/device_pauli_sum.hip.hpp src/hip/device_pauli_sum.hip.cpp src/hip/simplify_hip.hip.hpp src/hip/simplify_hip.hip.cpp
git commit -m "build: add HIP simplify source boundary"
```

## Task 3: Retained rocThrust HIP Simplify

**Files:**
- Modify: `src/hip/simplify_hip.hip.cpp`
- Test: remote MI300X pytest targets from Task 1

- [ ] **Step 1: Implement complex and key helpers**

In `src/hip/simplify_hip.hip.cpp`, add local helpers equivalent to the CUDA
ones, using HIP-compatible `double2` arithmetic:

```cpp
struct HipComplex {
  double real;
  double imag;
};

__host__ __device__ HipComplex to_hip_complex(double2 value) noexcept {
  return {value.x, value.y};
}

__host__ __device__ double2 to_double2(HipComplex value) noexcept {
  return double2{value.real, value.imag};
}

__host__ __device__ HipComplex operator+(HipComplex lhs, HipComplex rhs) noexcept {
  return {lhs.real + rhs.real, lhs.imag + rhs.imag};
}

__host__ __device__ double hip_abs(HipComplex value) noexcept {
  return sqrt(value.real * value.real + value.imag * value.imag);
}
```

Use `thrust::complex<double>` instead if MI300X compilation proves it is
trivially copy-compatible with `double2`; record the decision in the report.

- [ ] **Step 2: Implement empty and zero-word paths**

Retain the CPU/CUDA semantics:

```text
empty input returns empty DevicePauliSum with the same num_qubits and device
words == 0 reduces all coefficients to one identity term unless tolerance drops it
all-zero output returns empty DevicePauliSum preserving num_qubits
```

- [ ] **Step 3: Implement one-word <=32-qubit packed-key path**

Use a packed `uint64_t` key:

```text
key = (x_word << 32) | (z_word & 0xffffffff)
sort keys and coefficients by key
reduce equal keys by complex addition
drop coefficients with abs(coeff) <= atol + rtol * max_abs_input
unpack survivor keys into x and z output buffers
```

This is the expected retained path for many near-term workloads and is the
first path to benchmark.

- [ ] **Step 4: Implement one-word wide and two-word paths**

Add struct-key paths equivalent to CUDA:

```text
HipKey1: x, z
HipKey2: x0, z0, x1, z1
sort_by_key with lexicographic comparators
reduce_by_key with equality comparators
copy survivors into output DevicePauliSum
```

- [ ] **Step 5: Implement generic multi-word fallback**

Use a sorted-index fallback for `words > 2`:

```text
create sorted_indices = [0, 1, ..., num_terms - 1]
sort indices with comparator that compares packed x words then z words lexicographically
launch one deterministic reduction kernel over sorted indices
copy survivor count to host
allocate exactly survivor_count output terms
copy survivor packed words and coefficients into output buffers
```

This fallback prioritizes correctness and coverage over peak performance.
Campaign 3 benchmarks must label generic fallback rows separately.

- [ ] **Step 6: Reject unsupported strategy values cleanly**

If `FASTPAULI_HIP_BENCH_DUPLICATE_REDUCTION` requests a strategy that is not
compiled or not supported for the current shape, raise a `RuntimeError` naming
the strategy, the shape, and the supported shapes.

- [ ] **Step 7: Run targeted MI300X validation**

Run:

```bash
PATH=/opt/rocm/bin:$PATH .venv/bin/python -m pytest \
  tests/test_phase12_rocm_foundation.py::test_hip_simplify_matches_cpu_for_edge_cases_when_available \
  tests/test_phase12_rocm_foundation.py::test_hip_simplify_tolerance_matches_cpu_when_available \
  tests/test_phase12_rocm_foundation.py::test_hip_simplify_randomized_matches_cpu_when_available \
  tests/test_phase12_rocm_foundation.py::test_hip_simplify_rejects_invalid_tolerances_when_available \
  tests/test_phase12_rocm_foundation.py::test_hip_deferred_surfaces_remain_unavailable_after_simplify_when_available \
  -q
```

Expected: all targeted tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/hip/simplify_hip.hip.cpp tests/test_phase12_rocm_foundation.py
git commit -m "feat: add HIP simplify"
```

## Task 4: Benchmark And Strategy Experiments

**Files:**
- Modify: `benchmarks/bench_rocm_kernels.py`
- Create: `scripts/render_rocm_campaign3_assets.py`
- Test: `python -m pytest tests/test_benchmark_metadata.py tests/test_cuda_deep_report_assets.py -q`

- [ ] **Step 1: Add Campaign 3 profiles**

Add these profiles to `benchmarks/bench_rocm_kernels.py`:

```text
simplify-smoke
simplify-duplicate-pressure
simplify-wide-qubit
simplify-campaign3-profiler
simplify-strategy-ab
```

Datasets must include:

```text
campaign3_smoke_one_word: num_qubits=8, num_terms=128, duplicate_rate=0.25
campaign3_duplicate_heavy: num_qubits=24, num_terms=32768, duplicate_rate=0.875
campaign3_duplicate_light: num_qubits=24, num_terms=32768, duplicate_rate=0.0625
campaign3_wide_two_word: num_qubits=70, num_terms=8192, duplicate_rate=0.25
campaign3_generic_multiword: num_qubits=130, num_terms=4096, duplicate_rate=0.25
campaign3_all_zero: num_qubits=24, num_terms=4096, duplicate_rate=1.0, coefficients cancel to zero
```

- [ ] **Step 2: Add timing boundaries**

For each row, measure:

```text
cpu_scalar_seconds: op.simplify()
available_cpu_selector_seconds: every compiled CPU selector that can run simplify or "not_applicable" if simplify has no selector
hip_simplify_transfer_seconds: op.to_device().simplify().to_host()
hip_simplify_device_resident_seconds: device_op.simplify()
hip_simplify_to_host_seconds: simplified_device_op.to_host()
```

Correctness checks must compare `device_op.simplify().to_host()` to
`op.simplify()` with exact labels and coefficients.

- [ ] **Step 3: Add strategy metadata**

Every row must record:

```text
hip_simplify_strategy
hip_simplify_strategy_status
hip_simplify_strategy_unavailable_reason
hip_simplify_output_terms
hip_simplify_output_words
timing_boundary
result_materialization_target
```

When an A/B strategy is requested through
`FASTPAULI_HIP_BENCH_DUPLICATE_REDUCTION`, benchmark rows must state whether it
was retained, rejected, benchmark-only, unsupported for the shape, unavailable
because headers/libraries are missing, or failed correctness.

- [ ] **Step 4: Add report asset renderer**

Create `scripts/render_rocm_campaign3_assets.py` that reads Campaign 3 raw JSON
files, writes `summary.json`, and renders:

```text
docs/benchmarks/plots/rocm_mi300x_campaign3_simplify.svg
docs/benchmarks/plots/accelerator_landscape_with_rocm.svg
```

The first plot is report-local. The second plot must be broad: CPU scalar,
available CPU selectors, CUDA rows from checked Campaign 10 evidence, ROCm
Campaign 2 commutation rows, ROCm Campaign 3 simplify rows, and external
baseline rows where checked evidence exists.

- [ ] **Step 5: Run local benchmark smoke**

Run locally without HIP:

```bash
.venv/bin/python benchmarks/bench_rocm_kernels.py --profile simplify-smoke --repeat 1 --warmup 0 --json
```

Expected: the row is skipped or reports HIP unavailable with a clear
`unavailable_reason`; no import-time ROCm dependency is required.

- [ ] **Step 6: Run MI300X benchmark profiles**

Run on MI300X:

```bash
PATH=/opt/rocm/bin:$PATH FASTPAULI_BENCHMARK_GIT_COMMIT=$(git rev-parse --short HEAD) \
  .venv/bin/python benchmarks/bench_rocm_kernels.py \
  --profile simplify-duplicate-pressure --repeat 5 --warmup 2 --json \
  --output docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30/raw/rocm_simplify_duplicate_pressure_mi300x.json

PATH=/opt/rocm/bin:$PATH FASTPAULI_BENCHMARK_GIT_COMMIT=$(git rev-parse --short HEAD) \
  .venv/bin/python benchmarks/bench_rocm_kernels.py \
  --profile simplify-wide-qubit --repeat 5 --warmup 2 --json \
  --output docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30/raw/rocm_simplify_wide_qubit_mi300x.json

PATH=/opt/rocm/bin:$PATH FASTPAULI_HIP_BENCH_DUPLICATE_REDUCTION=hipcub_radix_sort_reduce \
  FASTPAULI_BENCHMARK_GIT_COMMIT=$(git rev-parse --short HEAD) \
  .venv/bin/python benchmarks/bench_rocm_kernels.py \
  --profile simplify-strategy-ab --repeat 5 --warmup 2 --json \
  --output docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30/raw/rocm_simplify_strategy_hipcub_mi300x.json
```

- [ ] **Step 7: Capture rocprof evidence**

Run:

```bash
PATH=/opt/rocm/bin:$PATH FASTPAULI_BENCHMARK_GIT_COMMIT=$(git rev-parse --short HEAD) \
  rocprof -d docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30/profiler \
  --hip-trace --stats \
  .venv/bin/python benchmarks/bench_rocm_kernels.py \
  --profile simplify-campaign3-profiler --repeat 1 --warmup 0 --json \
  --output docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30/raw/rocm_simplify_profiler_mi300x.json
```

Then capture at least these counters when rocprof supports them:

```text
SQ_WAVES
GRBM_GUI_ACTIVE
FETCH_SIZE
WRITE_SIZE
VALUUtilization
VALUBusy
```

If counters fail because of host permissions or provider limits, check in the
exact command output and mark the counter lane `blocked_external`.

- [ ] **Step 8: Commit**

```bash
git add benchmarks/bench_rocm_kernels.py scripts/render_rocm_campaign3_assets.py docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30 docs/benchmarks/plots
git commit -m "bench: add ROCm simplify campaign 3"
```

## Task 5: Report, README Landscape, Review, And Closeout

**Files:**
- Create: `docs/benchmarks/reports/rocm_mi300x_campaign3_2026-04-30.md`
- Modify: `README.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/plans/rocm_next_waves_plan.md`
- Test: `python scripts/validate.py`

- [ ] **Step 1: Write the Campaign 3 report**

The report must include:

```text
scope and non-goals
host and build inventory
implementation outcome table
validation commands and outcomes
benchmark command table
benchmark result table with CPU scalar, CPU selector, HIP transfer-inclusive, HIP device-resident, and HIP to_host rows
strategy decision table for rocThrust, hipCUB, and custom duplicate reduction
rocprof trace and counter interpretation
headroom terminal-status table covering DLPack, streams, workspaces, packed summaries, expectation, matmul, portability, ROCm wheels, multi-GPU, and simultaneous CUDA+HIP
retained decisions
rejected or deferred decisions
review findings and resolutions
remaining ROCm headroom
release claim and rejected claims
```

- [ ] **Step 2: Update README performance landscape**

Replace or supplement the current README performance plot only with the broad
`accelerator_landscape_with_rocm.svg` plot generated from checked evidence.
The README must continue to show CPU, CUDA, external baseline, and ROCm rows in
one landscape view.

- [ ] **Step 3: Update roadmap and wave routing**

Update:

```text
docs/roadmap.md latest ROCm campaign plan/report references
docs/plans/rocm_next_waves_plan.md Wave 3 status and next executable wave
docs/architecture/rocm_backend.md retained Campaign 3 public boundary
```

- [ ] **Step 4: Run local validation**

Run locally with HIP disabled:

```bash
python scripts/validate.py
```

Expected: pass.

- [ ] **Step 5: Run MI300X validation**

Run on the MI300X HIP build:

```bash
PATH=/opt/rocm/bin:$PATH .venv/bin/python -m pytest
PATH=/opt/rocm/bin:$PATH .venv/bin/python -m pytest tests/test_phase12_rocm_foundation.py tests/test_phase4_simplify.py -q
PATH=/opt/rocm/bin:$PATH .venv/bin/python benchmarks/bench_rocm_kernels.py --profile simplify-smoke --repeat 3 --warmup 1 --json
```

Expected: pytest passes, with only explicitly documented optional dependency
or unavailable-hardware skips, and benchmark smoke passes with correctness
checks enabled.

- [ ] **Step 6: Complete independent review**

Request independent review covering:

```text
HIP correctness and lifetime
canonical ordering and tolerance parity
CUDA regression risk
public API wording and docstrings
benchmark timing-boundary honesty
profiler evidence freshness
README landscape breadth
headroom terminal statuses
release-claim wording
```

Resolve P0/P1 findings, rerun affected validation, and record P2 deferrals
with named follow-up scope.

- [ ] **Step 7: Commit**

```bash
git add README.md docs/roadmap.md docs/plans/rocm_next_waves_plan.md docs/architecture/rocm_backend.md docs/benchmarks/reports/rocm_mi300x_campaign3_2026-04-30.md docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30 docs/benchmarks/plots
git commit -m "docs: report ROCm MI300X campaign 3"
```

- [ ] **Step 8: Merge, validate, push, and confirm CI**

Follow the repository closeout flow:

```bash
git switch main
git merge --ff-only codex/rocm-campaign3
python scripts/validate.py
git push origin main
gh run watch <run-id> --exit-status
git branch -d codex/rocm-campaign3
```

Expected: merged `main` validates locally, pushed CI is green, and the feature
branch is deleted after merge.

## Remaining Work After Campaign 3

Campaign 3 should leave a smaller, explicit ROCm backlog:

```text
HIP DLPack only if PyTorch ROCm, CuPy ROCm, or another named consumer is installed and a kDLROCM stream/ownership contract is accepted
HIP expectation only after statevector input ownership and ROCm Python consumer support are decided
HIP matmul only after simplify evidence proves duplicate-reduction output is reliable enough for product generation
HIP public workspace only if Campaign 3 allocation attribution shows allocation dominates retained workloads
HIP streams/graphs only if profiler evidence shows launch/synchronization dominates a retained public workload
additional AMD GPUs only when release wording needs portability evidence beyond MI300X gfx942
ROCm wheels only through a release-packaging plan with manylinux, runtime, and CI policy
simultaneous CUDA+HIP only through a backend-neutral object-model architecture decision
```

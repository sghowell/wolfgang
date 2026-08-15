# MI300X ROCm Optimization Campaign 6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retain HIP `DevicePauliSum.expectation_statevector()` and HIP `DevicePauliSum.matmul()` parity on MI300X without adding new public API shape or reopening rejected HIP interop surfaces.

**Architecture:** Campaign 6 is a Wave 4 to Wave 5 parity bridge. The existing public Python methods already exist for CUDA and already raise clear HIP unsupported-operation errors, so this campaign promotes those methods to HIP-backed implementations where CPU/CUDA contracts already define behavior. ROCm/HIP remains source-build-only, MI300X `gfx942` evidenced, mutually exclusive with CUDA, synchronous at public method boundaries, and unavailable for HIP DLPack, HIP CUDA Array Interface, public streams, graphs, public workspaces, multi-GPU ROCm, ROCm wheels, broader AMD portability claims, and simultaneous CUDA+HIP source builds.

**Tech Stack:** C++20, nanobind, CMake HIP language support, ROCm/HIP runtime, AMD Instinct MI300X `gfx942`, rocThrust for existing simplify reuse, pytest, NumPy, rocprof, existing FastPauli CPU/CUDA/ROCm benchmark and report infrastructure.

---

## Status

```text
complete
report: docs/benchmarks/reports/rocm_mi300x_campaign6_2026-04-30.md
evidence: docs/benchmarks/data/rocm_mi300x_campaign6_2026-04-30/
```

## Baseline

Campaign 5 is complete:

```text
plan: docs/plans/mi300x_rocm_optimization_campaign5_plan.md
report: docs/benchmarks/reports/rocm_mi300x_campaign5_2026-04-30.md
evidence: docs/benchmarks/data/rocm_mi300x_campaign5_2026-04-30/
retained public HIP surfaces: DevicePauliSum.to_device(), DevicePauliSum.to_host(), DevicePauliSum.commutes_with(), DevicePauliSum.commutes_with_device(), DeviceCommutationMatrix.to_host(), DeviceCommutationMatrix.count_commuting(), DeviceCommutationMatrix.conflict_degrees(), DevicePauliSum.simplify()
rejected public HIP surfaces: DeviceCommutationMatrix.__dlpack__(), DeviceCommutationMatrix.__dlpack_device__(), DeviceCommutationMatrix.__cuda_array_interface__, public streams, graph replay, public workspaces
```

The HIP methods retained here were already present on `DevicePauliSum`:

```text
DevicePauliSum.expectation_statevector(psi)
DevicePauliSum.matmul(rhs, simplify=True, max_intermediate_terms=50000000)
```

Pre-campaign HIP behavior was intentionally unavailable:

```text
HIP expectation_statevector is not implemented yet
HIP matmul is not implemented yet
```

Campaign 6 replaced those unsupported-operation paths with validated HIP
implementations while preserving the public method signatures, CPU-only build
behavior, CUDA behavior, and all Campaign 5 rejections.

## Campaign 6 Scope

In scope:

```text
HIP host NumPy complex64 and complex128 statevector expectation
HIP expectation parity for empty operators, identity terms, diagonal terms, off-diagonal X/Y/Z terms, duplicate terms, randomized small systems, and invalid host arrays
HIP expectation rejection for external device-pointer inputs until a separate HIP interop contract exists
HIP matmul product generation for one-word and multi-word Pauli operators
HIP matmul simplify=True path that reuses the retained HIP simplify implementation
HIP matmul simplify=False path that preserves CPU nested-loop product ordering
HIP matmul guardrails for same-device, same-num-qubits, moved-from, max_intermediate_terms, and overflow behavior
MI300X benchmark profiles for expectation and matmul parity rows with CPU scalar, available optimized CPU selectors, HIP transfer-inclusive timing, HIP device-resident kernel timing where measurable, and explicit materialization boundaries
rocprof trace/stats evidence for retained expectation and matmul kernels
README, roadmap, ROCm wave plan, architecture, benchmark protocol, and report updates after execution
```

Hard out of scope:

```text
HIP DLPack or HIP __dlpack_device__
HIP CUDA Array Interface
HIP external device-pointer statevector interop
public HIP streams
public HIP graph replay
public HIP workspace handles
multi-GPU ROCm execution
ROCm binary wheels
additional AMD GPU support claims beyond MI300X gfx942 evidence
simultaneous CUDA+HIP source builds
new public Python methods or arguments
raw PTX, GCN assembly, or non-HIP DSL rewrites
```

## Retention Gates

HIP expectation may be retained only when every gate passes:

```text
host NumPy complex128 and complex64 inputs match CPU expectation within dtype-appropriate tolerances
psi must be one-dimensional, C-contiguous, complex64 or complex128, and length 2 ** num_qubits
invalid input exception class and message intent match the existing CPU/CUDA binding behavior
empty operators return 0j after validating psi metadata
identity and zero-qubit identity behavior matches CPU
num_qubits > 63 raises a deterministic ValueError before any statevector-size shift overflow
HIP expectation rejects external device-pointer paths with a HIP/ROCm-specific unavailable message
HIP implementation synchronizes before returning the Python scalar
result accumulation error is documented and tested against CPU with explicit tolerances
public headers include no HIP or ROCm runtime headers
CPU-only and CUDA-only builds behave unchanged
FASTPAULI_ENABLE_CUDA=ON with FASTPAULI_ENABLE_HIP=ON remains a configure-time error
```

HIP matmul may be retained only when every gate passes:

```text
simplify=True matches PauliSum.matmul(rhs, simplify=True) canonical output
simplify=False matches PauliSum.matmul(rhs, simplify=False) nested-loop output order exactly
phase exponent handling matches CPU/CUDA fixtures for X, Y, Z, I combinations
one-word and multi-word packed layouts match CPU labels and coefficients
empty lhs, empty rhs, and empty output behavior matches CPU
same-device and same-num-qubits guardrails raise deterministic ValueError before kernel launch
max_intermediate_terms guardrail runs before allocation
checked arithmetic prevents size_t overflow in product-term and packed-word counts
simplify=True uses the retained HIP simplify path rather than copying to host for simplify
public API shape and docstrings stay unchanged except replacing HIP unsupported wording with supported-HIP wording
```

## Evidence Layout

Use this evidence root:

```text
docs/benchmarks/data/rocm_mi300x_campaign6_2026-04-30/
docs/benchmarks/data/rocm_mi300x_campaign6_2026-04-30/logs/
docs/benchmarks/data/rocm_mi300x_campaign6_2026-04-30/raw/
docs/benchmarks/data/rocm_mi300x_campaign6_2026-04-30/profiler/
docs/benchmarks/data/rocm_mi300x_campaign6_2026-04-30/summary.json
docs/benchmarks/reports/rocm_mi300x_campaign6_2026-04-30.md
docs/benchmarks/plots/rocm_mi300x_campaign6_parity.svg
docs/benchmarks/plots/accelerator_landscape_with_rocm.svg
```

If execution crosses UTC midnight, keep this evidence root only if the report
states the exact execution dates and why the root remains named
`rocm_mi300x_campaign6_2026-04-30`.

## Acceptance Criteria

Campaign 6 is complete only when every applicable item has a terminal status in
the Campaign 6 report:

```text
local CPU-only validation passes with FASTPAULI_ENABLE_HIP=OFF
HIP source build succeeds on MI300X with FASTPAULI_ENABLE_HIP=ON and FASTPAULI_HIP_ARCHITECTURES=gfx942
CUDA+HIP configure-time rejection still passes
public headers still contain no HIP or ROCm runtime headers
HIP expectation_statevector matches CPU for complex64, complex128, empty, identity, diagonal, off-diagonal, duplicate, randomized, and invalid-input cases
HIP expectation external device-pointer interop remains unavailable with an explicit HIP/ROCm message
HIP matmul matches CPU for simplify=True, simplify=False, one-word, multi-word, empty, phase, guardrail, and randomized cases
HIP matmul simplify=True stays device-resident through HIP simplify
benchmark JSON includes Campaign 6 expectation and matmul protocol fields
rocprof trace/stats or a precise tooling blocker is checked in for retained HIP expectation and matmul rows
README broad performance landscape remains CPU/CUDA/ROCm/external rather than a narrow ROCm-only plot
Campaign 5 rejected surfaces remain rejected unless a separate plan accepts them
portability, ROCm wheels, multi-GPU ROCm, and simultaneous CUDA+HIP retain explicit next-wave statuses
independent review is recorded before merge
```

## Task 1: Contracts, Benchmark Schema, And Red Tests

**Files:**
- Modify: `docs/architecture/rocm_backend.md`
- Modify: `docs/benchmarks/protocol.md`
- Modify: `tests/test_phase12_rocm_foundation.py`
- Test: `python -m pytest tests/test_phase12_rocm_foundation.py -q`

- [ ] **Step 1: Add the Campaign 6 parity boundary**

Add a `Campaign 6 Expectation And Matmul Parity Boundary` section to
`docs/architecture/rocm_backend.md` with these decisions:

```text
Campaign 6 may promote existing DevicePauliSum.expectation_statevector() and DevicePauliSum.matmul() methods from HIP-unavailable to HIP-supported.
Campaign 6 may not add new public Python methods, stream arguments, workspace arguments, DLPack statevector imports, or external HIP pointer imports.
HIP expectation accepts host NumPy complex64 and complex128 inputs only.
HIP matmul returns a HIP-backed DevicePauliSum and uses HIP simplify for simplify=True.
HIP DLPack, HIP CUDA Array Interface, public streams, public graphs, public workspaces, ROCm wheels, multi-GPU ROCm, additional AMD GPU support claims, and simultaneous CUDA+HIP source builds remain unavailable or separate campaigns.
```

- [ ] **Step 2: Add Campaign 6 benchmark fields**

Add a `ROCm Campaign 6 expectation and matmul parity fields` subsection to
`docs/benchmarks/protocol.md` requiring these JSON fields:

```text
campaign: rocm_mi300x_campaign6
operation: expectation_statevector or matmul
backend: hip
mode: host_complex128, host_complex64, matmul_simplify_true, matmul_simplify_false, external_device_pointer_guard, profiler_expectation, profiler_matmul, portability_decision, packaging_decision, multi_gpu_decision, or simultaneous_cuda_hip_decision
status: ok, retained, rejected_with_evidence, blocked_external, unavailable, or out_of_scope_with_next_trigger
final_status: retained, rejected_with_evidence, unavailable, or out_of_scope_with_next_trigger
timing_boundary: transfer_inclusive, device_resident_kernel, device_output_to_host, decision_only, or benchmark_only
hip_expectation_input_dtype
hip_expectation_state_size
hip_expectation_num_terms
hip_expectation_words
hip_expectation_transfer_seconds and p10/p90/min/max variants
hip_expectation_device_resident_seconds and p10/p90/min/max variants
hip_expectation_result_copy_seconds and p10/p90/min/max variants
hip_matmul_lhs_terms
hip_matmul_rhs_terms
hip_matmul_output_terms
hip_matmul_words
hip_matmul_simplify_output
hip_matmul_transfer_seconds and p10/p90/min/max variants
hip_matmul_device_resident_seconds and p10/p90/min/max variants
hip_matmul_to_host_seconds and p10/p90/min/max variants
correctness_digest with operation-specific label hash, coefficient_l1, and result summary
campaign6_terminal_statuses for expectation, matmul, external device pointers, DLPack, CUDA Array Interface guard, streams, graphs, workspaces, portability, ROCm wheels, multi-GPU, and simultaneous CUDA+HIP
```

- [ ] **Step 3: Add red HIP expectation tests**

Replace the expectation part of
`test_hip_deferred_surfaces_remain_unavailable_after_simplify_when_available`
with retained-HIP tests and keep the unavailable external-pointer guard:

```python
def test_hip_expectation_statevector_matches_cpu_for_complex_dtypes_when_available() -> None:
    _require_hip_runtime()

    op = fastpauli.PauliSum.from_labels(
        ["ZI", "IZ", "XX", "YY", "XY"],
        [1.0, -0.5, 0.25, 0.75j, -0.125 + 0.5j],
    )
    raw = np.asarray([1.0 + 0.25j, -0.5j, 0.75, -0.125 + 0.5j], dtype=np.complex128)
    psi128 = raw / np.linalg.norm(raw)
    psi64 = psi128.astype(np.complex64)
    device_op = op.to_device()

    np.testing.assert_allclose(
        device_op.expectation_statevector(psi128),
        op.expectation_statevector(psi128),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        device_op.expectation_statevector(psi64),
        op.expectation_statevector(psi64),
        rtol=1.0e-5,
        atol=1.0e-5,
    )
```

Add edge and guardrail coverage:

```python
def test_hip_expectation_statevector_edge_cases_when_available() -> None:
    _require_hip_runtime()

    empty = fastpauli.PauliSum.empty(2)
    assert empty.to_device().expectation_statevector(
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    ) == 0j

    identity = fastpauli.PauliSum.from_sparse_list([("", [], 2.5 - 0.75j)], num_qubits=0)
    assert identity.to_device().expectation_statevector(
        np.asarray([1.0 + 0.0j], dtype=np.complex128)
    ) == pytest.approx(2.5 - 0.75j, abs=1.0e-12)


def test_hip_expectation_statevector_rejects_invalid_host_arrays_when_available() -> None:
    _require_hip_runtime()

    device_op = fastpauli.PauliSum.from_labels(["ZI", "IZ"], [1.0, -0.5]).to_device()
    with pytest.raises(TypeError, match="complex64 or complex128"):
        device_op.expectation_statevector(np.ones(4, dtype=np.float64))
    with pytest.raises(TypeError, match="C-contiguous"):
        device_op.expectation_statevector(np.ones(8, dtype=np.complex128)[::2])
    with pytest.raises(ValueError, match="len\\(psi\\) == 2 \\*\\* num_qubits"):
        device_op.expectation_statevector(np.ones(2, dtype=np.complex128))


def test_hip_expectation_external_device_pointer_remains_unavailable_when_available() -> None:
    _require_hip_runtime()

    class FakeCudaArray:
        def __init__(self, interface: dict[str, object]) -> None:
            self.__cuda_array_interface__ = interface

    device_op = fastpauli.PauliSum.from_labels(["ZI", "IZ"], [1.0, -0.5]).to_device()
    with pytest.raises((BufferError, RuntimeError, ValueError), match="HIP|ROCm|device pointer"):
        device_op.expectation_statevector(
            FakeCudaArray({"shape": (4,), "typestr": "<c16", "data": (1, False), "version": 3})
        )
```

- [ ] **Step 4: Add duplicate and randomized HIP expectation tests**

Add explicit duplicate-term and randomized expectation fixtures so Campaign 6
cannot satisfy only the fixed two-qubit examples:

```python
def test_hip_expectation_statevector_duplicate_terms_when_available() -> None:
    _require_hip_runtime()

    op = fastpauli.PauliSum.from_labels(
        ["ZI", "ZI", "XX", "XX", "YY"],
        [1.0, -0.25, 0.5j, -0.125j, 0.75],
    )
    raw = np.asarray([1.0, 0.25j, -0.5, 0.125 + 0.5j], dtype=np.complex128)
    psi = raw / np.linalg.norm(raw)

    np.testing.assert_allclose(
        op.to_device().expectation_statevector(psi),
        op.expectation_statevector(psi),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_hip_expectation_statevector_randomized_small_systems_when_available() -> None:
    _require_hip_runtime()

    rng = np.random.default_rng(69451)
    alphabet = np.asarray(["I", "X", "Y", "Z"])
    for num_qubits, terms in [(3, 12), (5, 24)]:
        labels = [
            "".join(rng.choice(alphabet, size=num_qubits).tolist())
            for _ in range(terms)
        ]
        coeffs = [
            complex(float(rng.normal()), float(rng.normal()))
            for _ in range(terms)
        ]
        psi = rng.normal(size=2**num_qubits) + 1j * rng.normal(size=2**num_qubits)
        psi = np.asarray(psi / np.linalg.norm(psi), dtype=np.complex128)
        op = fastpauli.PauliSum.from_labels(labels, coeffs)

        np.testing.assert_allclose(
            op.to_device().expectation_statevector(psi),
            op.expectation_statevector(psi),
            rtol=1.0e-11,
            atol=1.0e-11,
        )
```

- [ ] **Step 5: Add red HIP matmul tests**

Replace the matmul part of
`test_hip_deferred_surfaces_remain_unavailable_after_simplify_when_available`
with retained-HIP tests:

```python
def _assert_same_operator(lhs: fastpauli.PauliSum, rhs: fastpauli.PauliSum) -> None:
    lhs_labels, lhs_coeffs = _labels_and_coeffs(lhs)
    rhs_labels, rhs_coeffs = _labels_and_coeffs(rhs)
    assert lhs_labels == rhs_labels
    np.testing.assert_allclose(lhs_coeffs, rhs_coeffs, rtol=1.0e-12, atol=1.0e-12)


def test_hip_matmul_matches_cpu_and_keeps_guardrails_when_available() -> None:
    _require_hip_runtime()

    lhs = fastpauli.PauliSum.from_labels(["X", "Y", "Z"], [2.0, -0.5j, 1.25])
    rhs = fastpauli.PauliSum.from_labels(["Y", "Z"], [3.0, 0.25j])

    expected = lhs.matmul(rhs, simplify=True)
    actual = lhs.to_device().matmul(rhs.to_device(), simplify=True).to_host()
    _assert_same_operator(actual, expected)

    raw_expected = lhs.matmul(rhs, simplify=False)
    raw_actual = lhs.to_device().matmul(rhs.to_device(), simplify=False).to_host()
    _assert_same_operator(raw_actual, raw_expected)

    with pytest.raises(ValueError, match="matmul intermediate term count exceeds"):
        lhs.to_device().matmul(rhs.to_device(), max_intermediate_terms=5)
```

Add multi-word and device-guard coverage:

```python
def test_hip_matmul_multiword_and_empty_cases_when_available() -> None:
    _require_hip_runtime()

    cases = [
        (
            fastpauli.PauliSum.empty(70),
            fastpauli.PauliSum.from_sparse_list([("X", [64], 1.0)], num_qubits=70),
        ),
        (
            fastpauli.PauliSum.from_sparse_list([("XY", [0, 70], 1.0j), ("Z", [69], -2.0)], num_qubits=72),
            fastpauli.PauliSum.from_sparse_list([("YZ", [1, 70], -0.5), ("X", [69], 3.0)], num_qubits=72),
        ),
    ]
    for lhs, rhs in cases:
        _assert_same_operator(
            lhs.to_device().matmul(rhs.to_device(), simplify=True).to_host(),
            lhs.matmul(rhs, simplify=True),
        )
        _assert_same_operator(
            lhs.to_device().matmul(rhs.to_device(), simplify=False).to_host(),
            lhs.matmul(rhs, simplify=False),
        )


def test_hip_matmul_rejects_mismatched_devices_when_available() -> None:
    status = _require_hip_runtime()
    if int(status["device_count"]) < 2:
        pytest.skip("different-device HIP matmul check requires at least two visible HIP devices")

    lhs = fastpauli.PauliSum.from_labels(["X"], [1.0]).to_device(device=0)
    rhs = fastpauli.PauliSum.from_labels(["Y"], [1.0]).to_device(device=1)
    with pytest.raises(ValueError, match="same device"):
        lhs.matmul(rhs)
```

- [ ] **Step 6: Add randomized HIP matmul tests**

Add a randomized matmul fixture that validates both simplified and raw output:

```python
def test_hip_matmul_randomized_matches_cpu_when_available() -> None:
    _require_hip_runtime()

    rng = np.random.default_rng(69461)
    alphabet = np.asarray(["I", "X", "Y", "Z"])
    for num_qubits, lhs_terms, rhs_terms in [(4, 8, 7), (70, 5, 6)]:
        lhs_labels = [
            "".join(rng.choice(alphabet, size=num_qubits).tolist())
            for _ in range(lhs_terms)
        ]
        rhs_labels = [
            "".join(rng.choice(alphabet, size=num_qubits).tolist())
            for _ in range(rhs_terms)
        ]
        lhs_coeffs = [
            complex(float(rng.normal()), float(rng.normal()))
            for _ in range(lhs_terms)
        ]
        rhs_coeffs = [
            complex(float(rng.normal()), float(rng.normal()))
            for _ in range(rhs_terms)
        ]
        lhs = fastpauli.PauliSum.from_labels(lhs_labels, lhs_coeffs)
        rhs = fastpauli.PauliSum.from_labels(rhs_labels, rhs_coeffs)

        _assert_same_operator(
            lhs.to_device().matmul(rhs.to_device(), simplify=True).to_host(),
            lhs.matmul(rhs, simplify=True),
        )
        _assert_same_operator(
            lhs.to_device().matmul(rhs.to_device(), simplify=False).to_host(),
            lhs.matmul(rhs, simplify=False),
        )
```

- [ ] **Step 7: Run the red/guard step**

Run locally:

```bash
python -m pytest tests/test_phase12_rocm_foundation.py -q
```

Expected on macOS CPU-only builds: HIP-specific tests skip and existing local
tests pass. Run on MI300X before implementation:

```bash
PATH=/opt/rocm/bin:$PATH .venv/bin/python -m pytest tests/test_phase12_rocm_foundation.py -q
```

Expected on the current Campaign 5 HIP build: the new expectation and matmul
tests fail with the current unsupported-operation messages. Do not weaken the
tests to make the unsupported implementation pass.

## Task 2: HIP Statevector Expectation Implementation

**Files:**
- Create: `src/hip/expectation_hip.hip.cpp`
- Modify: `src/hip/device_pauli_sum.hip.cpp`
- Modify: `src/hip/device_pauli_sum.hip.hpp`
- Modify: `CMakeLists.txt`
- Modify: `bindings/python/pauli_sum_py.cpp`
- Modify: `scripts/validate.py`
- Modify: `tests/test_validate_entrypoint.py`
- Test: `python -m pytest tests/test_phase12_rocm_foundation.py tests/test_validate_entrypoint.py -q`

- [ ] **Step 1: Add the HIP expectation source to CMake**

Add `src/hip/expectation_hip.hip.cpp` to both HIP source lists in
`CMakeLists.txt`:

```cmake
src/hip/expectation_hip.hip.cpp
```

Set its source language with the other HIP files:

```cmake
set_source_files_properties(
  src/hip/commutation_hip.hip.cpp
  src/hip/device_commutation_matrix.hip.cpp
  src/hip/device_pauli_sum.hip.cpp
  src/hip/expectation_hip.hip.cpp
  src/hip/simplify_hip.hip.cpp
  src/hip/workspace_hip.hip.cpp
  PROPERTIES LANGUAGE HIP)
```

- [ ] **Step 2: Add the HIP expectation source to validation layout checks**

Add `src/hip/expectation_hip.hip.cpp` to `HIP_FOUNDATION_SOURCES` in
`scripts/validate.py`. `tests/test_validate_entrypoint.py` already builds its
fixture file list from `validate.HIP_FOUNDATION_SOURCES`; after this addition,
the existing `test_native_source_layout_check_rejects_top_level_backend_sources`
fixture must still pass and must include the new HIP expectation source in the
generated `CMakeLists.txt` content.

Run:

```bash
python -m pytest tests/test_validate_entrypoint.py::test_native_source_layout_check_rejects_top_level_backend_sources -q
```

- [ ] **Step 3: Add shared HIP statevector helpers**

Add these helpers to `src/hip/device_pauli_sum.hip.hpp` in
`namespace hip_detail`:

```cpp
constexpr std::size_t kMaxHipStatevectorQubits = 63;

inline std::size_t expected_statevector_length(std::size_t num_qubits) {
  if (num_qubits > kMaxHipStatevectorQubits) {
    throw std::invalid_argument("expectation_statevector requires num_qubits <= 63");
  }
  return std::size_t{1} << num_qubits;
}

inline void validate_statevector_length(std::size_t num_qubits, std::size_t actual_size) {
  const std::size_t expected_size = expected_statevector_length(num_qubits);
  if (actual_size != expected_size) {
    throw std::invalid_argument("expectation_statevector requires len(psi) == 2 ** num_qubits");
  }
}

inline void copy_bytes_to_device(void* dst, const void* src, std::size_t bytes, const char* name) {
  if (bytes == 0) {
    return;
  }
  (void)name;
  check_hip(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice), "copy host buffer to device");
}
```

- [ ] **Step 4: Move unsupported stubs out of `device_pauli_sum.hip.cpp`**

Delete these HIP method definitions from `src/hip/device_pauli_sum.hip.cpp` so
the new source owns them:

```cpp
std::complex<double> DevicePauliSum::expectation_statevector_complex128(
    std::span<const std::complex<double>>) const {
  throw_kernel_not_implemented("expectation_statevector");
}

std::complex<double> DevicePauliSum::expectation_statevector_complex64(
    std::span<const std::complex<float>>) const {
  throw_kernel_not_implemented("expectation_statevector");
}

std::complex<double> DevicePauliSum::expectation_statevector_device_pointer(
    std::uintptr_t,
    DeviceStatevectorDtype,
    std::size_t) const {
  throw_kernel_not_implemented("expectation_statevector");
}
```

Keep `throw_kernel_not_implemented()` until matmul is moved in Task 3.

- [ ] **Step 5: Implement host statevector expectation**

Create `src/hip/expectation_hip.hip.cpp` by adapting the CUDA expectation
kernel structure to HIP. The retained source must include:

```cpp
#include "device_pauli_sum.hip.hpp"

#include <hip/hip_runtime.h>
#include <thrust/complex.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
```

Use the same storage guards as CUDA:

```cpp
static_assert(std::is_trivially_copyable_v<std::complex<double>>);
static_assert(std::is_trivially_copyable_v<std::complex<float>>);
static_assert(sizeof(std::complex<double>) == sizeof(thrust::complex<double>));
static_assert(sizeof(std::complex<float>) == sizeof(thrust::complex<float>));
```

Implement:

```cpp
template <typename StateComplex>
__global__ void expectation_statevector_terms_kernel(...);
```

with the same algorithm as CUDA:

```text
one block per Pauli term
shared-memory reduction within each term
target basis index is basis ^ x_mask
Z parity is popcount(z_mask & basis) & 1
Y phase uses popcount(x_mask & z_mask)
atomicAdd accumulates real and imaginary double result parts
```

Implement host methods:

```cpp
std::complex<double> DevicePauliSum::expectation_statevector_complex128(
    std::span<const std::complex<double>> psi) const;

std::complex<double> DevicePauliSum::expectation_statevector_complex64(
    std::span<const std::complex<float>> psi) const;
```

Both methods must:

```text
validate moved-from state
validate statevector length before allocation
copy host bytes into hipMalloc storage without an intermediate conversion vector
call a private expectation_statevector_device_pointer_impl(...)
return a host std::complex<double>
```

Implement the public external-pointer method as an explicit rejection:

```cpp
std::complex<double> DevicePauliSum::expectation_statevector_device_pointer(
    std::uintptr_t,
    DeviceStatevectorDtype,
    std::size_t) const {
  throw std::runtime_error(
      "HIP expectation_statevector does not accept external device pointers; "
      "ROCm/HIP statevector interop requires a separate ownership and stream contract");
}
```

- [ ] **Step 6: Update HIP docstring wording**

In `bindings/python/pauli_sum_py.cpp`, update the `DevicePauliSum.simplify`,
`DevicePauliSum.expectation_statevector`, and `DevicePauliSum.matmul`
docstrings so they accurately state:

```text
CUDA and HIP support simplify.
CUDA and HIP support host NumPy complex64 and complex128 statevector expectation.
Only CUDA supports CUDA-array-interface statevector inputs.
CUDA supports matmul; HIP support is added by Campaign 6 after Task 3.
```

Do not add a new Python argument or backend selector.

- [ ] **Step 7: Validate expectation locally and on MI300X**

Run locally:

```bash
python -m pytest tests/test_phase12_rocm_foundation.py -q
```

Run on MI300X after rebuilding the HIP editable install:

```bash
PATH=/opt/rocm/bin:$PATH .venv/bin/python -m pip install -e ".[test]" \
  --config-settings=cmake.define.FASTPAULI_ENABLE_HIP=ON \
  --config-settings=cmake.define.FASTPAULI_HIP_ARCHITECTURES=gfx942
PATH=/opt/rocm/bin:$PATH .venv/bin/python -m pytest tests/test_phase12_rocm_foundation.py -q
```

Expected after Task 2 and before Task 3: expectation tests pass, matmul tests
still fail with the unsupported HIP matmul message.

## Task 3: HIP Matmul Implementation

**Files:**
- Create: `src/hip/matmul_hip.hip.cpp`
- Modify: `src/hip/device_pauli_sum.hip.cpp`
- Modify: `CMakeLists.txt`
- Modify: `bindings/python/pauli_sum_py.cpp`
- Modify: `scripts/validate.py`
- Modify: `tests/test_validate_entrypoint.py`
- Test: `python -m pytest tests/test_phase12_rocm_foundation.py tests/test_validate_entrypoint.py -q`

- [ ] **Step 1: Add the HIP matmul source to CMake**

Add `src/hip/matmul_hip.hip.cpp` to both HIP source lists and the HIP language
source-properties list in `CMakeLists.txt`.

- [ ] **Step 2: Add the HIP matmul source to validation layout checks**

Add `src/hip/matmul_hip.hip.cpp` to `HIP_FOUNDATION_SOURCES` in
`scripts/validate.py`. Keep `tests/test_validate_entrypoint.py` passing so the
layout check proves both HIP parity sources are present and listed in CMake.

Run:

```bash
python -m pytest tests/test_validate_entrypoint.py::test_native_source_layout_check_rejects_top_level_backend_sources -q
```

- [ ] **Step 3: Move the unsupported matmul stub out of `device_pauli_sum.hip.cpp`**

Delete this method from `src/hip/device_pauli_sum.hip.cpp`:

```cpp
DevicePauliSum DevicePauliSum::matmul(
    const DevicePauliSum&,
    bool,
    std::size_t) const {
  throw_kernel_not_implemented("matmul");
}
```

If no unsupported HIP methods remain in that file, delete
`throw_kernel_not_implemented()`.

- [ ] **Step 4: Implement HIP matmul product generation**

Create `src/hip/matmul_hip.hip.cpp` by adapting the CUDA matmul kernel to HIP.
The retained source must include:

```cpp
#include "device_pauli_sum.hip.hpp"

#include <hip/hip_runtime.h>
#include <thrust/complex.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
```

Implement:

```cpp
__device__ thrust::complex<double> multiply_by_phase_exponent_device(
    thrust::complex<double> value,
    std::int64_t exponent);

__global__ void matmul_product_kernel(...);
```

Use the established CPU/CUDA phase formula:

```text
out_x = lhs_x ^ rhs_x
out_z = lhs_z ^ rhs_z
lhs_y = popcount(lhs_x & lhs_z)
rhs_y = popcount(rhs_x & rhs_z)
out_y = popcount(out_x & out_z)
lhs_rhs_cross = popcount(lhs_x & rhs_z)
phase_exponent = out_y - lhs_y - rhs_y + 2 * lhs_rhs_cross
out_coeff = lhs_coeff * rhs_coeff * i ** phase_exponent
```

- [ ] **Step 5: Implement `DevicePauliSum::matmul` for HIP**

The method must:

```text
reject moved-from operands
reject different HIP devices with "HIP matmul requires both operands on the same device"
reject different num_qubits with "PauliSum matmul requires the same num_qubits"
call detail::checked_matmul_intermediate_terms before allocation
return an empty HIP-backed DevicePauliSum for zero output terms
allocate out x, z, and coeff buffers on the same HIP device
launch matmul_product_kernel
return product.simplify() when simplify_output is true
synchronize before returning when simplify_output is false
```

Use `hip_detail::checked_launch_blocks`, `hip_detail::hip_allocate`,
`hip_detail::check_hip`, and `hip_detail::ScopedHipDevice`.

- [ ] **Step 6: Update final docstring wording**

In `bindings/python/pauli_sum_py.cpp`, update the `DevicePauliSum.matmul`
docstring to state:

```text
CUDA and HIP support this operation.
rhs acts first and self acts second.
max_intermediate_terms is enforced before allocating product buffers.
```

- [ ] **Step 7: Validate matmul locally and on MI300X**

Run locally:

```bash
python -m pytest tests/test_phase12_rocm_foundation.py -q
```

Run on MI300X:

```bash
PATH=/opt/rocm/bin:$PATH .venv/bin/python -m pip install -e ".[test]" \
  --config-settings=cmake.define.FASTPAULI_ENABLE_HIP=ON \
  --config-settings=cmake.define.FASTPAULI_HIP_ARCHITECTURES=gfx942
PATH=/opt/rocm/bin:$PATH .venv/bin/python -m pytest tests/test_phase12_rocm_foundation.py -q
```

Expected after Task 3: all HIP foundation tests pass except skips that require
two visible HIP devices when the host exposes only one device.

## Task 4: Campaign 6 Benchmarks And Profiling

**Files:**
- Modify: `benchmarks/bench_rocm_kernels.py`
- Modify: `tests/test_phase12_rocm_foundation.py`
- Create: `scripts/render_rocm_campaign6_assets.py`
- Create: `tests/test_rocm_campaign6_assets.py`
- Test: `python -m pytest tests/test_phase12_rocm_foundation.py tests/test_rocm_campaign6_assets.py -q`

- [ ] **Step 1: Add Campaign 6 benchmark profiles**

In `benchmarks/bench_rocm_kernels.py`, add these profile case lists:

```python
CAMPAIGN6_EXPECTATION_CASES = [
    {
        "name": "campaign6_expectation_two_qubit_complex128",
        "num_qubits": 2,
        "num_terms": 5,
        "term_weight": 2,
        "statevector_dtype": "complex128",
        "random_seed": 69421,
    },
    {
        "name": "campaign6_expectation_ten_qubit_complex64",
        "num_qubits": 10,
        "num_terms": 256,
        "term_weight": 4,
        "statevector_dtype": "complex64",
        "random_seed": 69422,
    },
]

CAMPAIGN6_MATMUL_CASES = [
    {
        "name": "campaign6_matmul_one_word_simplify_true",
        "num_qubits": 24,
        "lhs_terms": 256,
        "rhs_terms": 256,
        "term_weight": 4,
        "duplicate_rate": 0.25,
        "simplify_output": True,
        "random_seed": 69431,
    },
    {
        "name": "campaign6_matmul_two_word_simplify_false",
        "num_qubits": 70,
        "lhs_terms": 64,
        "rhs_terms": 64,
        "term_weight": 6,
        "duplicate_rate": 0.0,
        "simplify_output": False,
        "random_seed": 69432,
    },
]

CAMPAIGN6_PROFILER_CASES = [
    {
        "name": "campaign6_profiler_expectation",
        "num_qubits": 14,
        "num_terms": 1024,
        "term_weight": 6,
        "statevector_dtype": "complex128",
        "random_seed": 69441,
    },
    {
        "name": "campaign6_profiler_matmul",
        "num_qubits": 70,
        "lhs_terms": 128,
        "rhs_terms": 128,
        "term_weight": 6,
        "duplicate_rate": 0.25,
        "simplify_output": True,
        "random_seed": 69442,
    },
]
```

Register profiles:

```python
"campaign6-expectation-parity": CAMPAIGN6_EXPECTATION_CASES,
"campaign6-matmul-parity": CAMPAIGN6_MATMUL_CASES,
"campaign6-profiler": CAMPAIGN6_PROFILER_CASES,
```

- [ ] **Step 2: Add Campaign 6 row builders**

Add `CAMPAIGN6_TERMINAL_STATUSES`:

```python
CAMPAIGN6_TERMINAL_STATUSES = {
    "expectation": "retained",
    "matmul": "retained",
    "external device pointers": "unavailable",
    "DLPack": "rejected_with_evidence",
    "CUDA Array Interface guard": "rejected_with_evidence",
    "streams": "rejected_with_evidence",
    "graphs": "rejected_with_evidence",
    "workspaces": "rejected_with_evidence",
    "portability": "out_of_scope_with_next_trigger",
    "ROCm wheels": "out_of_scope_with_next_trigger",
    "multi-GPU": "out_of_scope_with_next_trigger",
    "simultaneous CUDA+HIP": "unavailable",
}
```

Add helpers that emit rows with:

```text
campaign: rocm_mi300x_campaign6
operation: expectation_statevector or matmul
backend: hip
final_status: retained when HIP runtime is available and correctness passes
status: unavailable with unavailable_reason when no HIP runtime is visible
campaign6_terminal_statuses: CAMPAIGN6_TERMINAL_STATUSES
correctness_passed: true only after CPU/HIP comparison
```

For expectation, measure:

```text
CPU scalar PauliSum.expectation_statevector(psi)
HIP transfer-inclusive op.to_device().expectation_statevector(psi)
HIP device-resident timing using a pre-transferred DevicePauliSum and host psi copy inside the method
explicit result copy timing if separately measurable, else null with timing_boundary explaining transfer-inclusive method boundary
```

For matmul, measure:

```text
CPU scalar lhs.matmul(rhs, simplify=case["simplify_output"])
HIP transfer-inclusive lhs.to_device().matmul(rhs.to_device(), simplify=case["simplify_output"]).to_host()
HIP device-resident timing using pre-transferred lhs/rhs and leaving the result on device
HIP to_host timing for output materialization
```

- [ ] **Step 3: Add benchmark protocol tests**

Add a new test to `tests/test_phase12_rocm_foundation.py`:

```python
def test_rocm_campaign6_benchmark_reports_parity_fields() -> None:
    script = ROOT / "benchmarks" / "bench_rocm_kernels.py"
    for profile, operation in [
        ("campaign6-expectation-parity", "expectation_statevector"),
        ("campaign6-matmul-parity", "matmul"),
    ]:
        completed = subprocess.run(
            [sys.executable, str(script), "--profile", profile, "--repeat", "1", "--warmup", "0", "--json"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        report = json.loads(completed.stdout)
        assert report["benchmark"] == "rocm_kernels"
        assert report["profile"] == profile
        assert report["cases"]
        row = report["cases"][0]
        assert row["campaign"] == "rocm_mi300x_campaign6"
        assert row["operation"] == operation
        assert row["backend"] == "hip"
        assert {
            "expectation",
            "matmul",
            "external device pointers",
            "DLPack",
            "CUDA Array Interface guard",
            "streams",
            "graphs",
            "workspaces",
            "portability",
            "ROCm wheels",
            "multi-GPU",
            "simultaneous CUDA+HIP",
        } <= set(row["campaign6_terminal_statuses"])
```

- [ ] **Step 4: Add Campaign 6 asset renderer**

Create `scripts/render_rocm_campaign6_assets.py` following the Campaign 5
renderer pattern. It must:

```text
read raw Campaign 6 expectation and matmul JSON files
write docs/benchmarks/data/rocm_mi300x_campaign6_2026-04-30/summary.json
write docs/benchmarks/plots/rocm_mi300x_campaign6_parity.svg
preserve or refresh docs/benchmarks/plots/accelerator_landscape_with_rocm.svg with broad CPU/CUDA/ROCm/external rows
reject missing retained rows unless the report explicitly records a blocked HIP runtime
```

Add `tests/test_rocm_campaign6_assets.py` with parser-level tests that use
small fixture dictionaries and assert:

```text
summary contains expectation and matmul retained rows
plot SVG contains Campaign 6 labels
missing retained Campaign 6 rows fail with a clear ValueError
```

- [ ] **Step 5: Run benchmark smokes locally**

Run:

```bash
python benchmarks/bench_rocm_kernels.py --profile campaign6-expectation-parity --repeat 1 --warmup 0 --json
python benchmarks/bench_rocm_kernels.py --profile campaign6-matmul-parity --repeat 1 --warmup 0 --json
python -m pytest tests/test_phase12_rocm_foundation.py tests/test_rocm_campaign6_assets.py -q
```

Expected locally without HIP runtime: benchmark rows report HIP unavailable
without crashing, and tests pass.

## Task 5: MI300X Evidence, Report, And README Refresh

**Files:**
- Create: `docs/benchmarks/reports/rocm_mi300x_campaign6_2026-04-30.md`
- Create: `docs/benchmarks/data/rocm_mi300x_campaign6_2026-04-30/summary.json`
- Create: `docs/benchmarks/data/rocm_mi300x_campaign6_2026-04-30/raw/*.json`
- Create: `docs/benchmarks/data/rocm_mi300x_campaign6_2026-04-30/logs/*.log`
- Create: `docs/benchmarks/data/rocm_mi300x_campaign6_2026-04-30/profiler/*`
- Create: `docs/benchmarks/plots/rocm_mi300x_campaign6_parity.svg`
- Modify: `docs/benchmarks/plots/accelerator_landscape_with_rocm.svg`
- Modify: `README.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/plans/rocm_next_waves_plan.md`
- Modify: `docs/architecture/rocm_backend.md`
- Test: `uv run python scripts/validate.py`

- [ ] **Step 1: Capture MI300X build and test evidence**

On the MI300X host:

```bash
git fetch origin
git checkout codex/rocm-campaign6
mkdir -p \
  docs/benchmarks/data/rocm_mi300x_campaign6_2026-04-30/logs \
  docs/benchmarks/data/rocm_mi300x_campaign6_2026-04-30/raw \
  docs/benchmarks/data/rocm_mi300x_campaign6_2026-04-30/profiler
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
PATH=/opt/rocm/bin:$PATH .venv/bin/python -m pip install -e ".[test]" \
  --config-settings=cmake.define.FASTPAULI_ENABLE_HIP=ON \
  --config-settings=cmake.define.FASTPAULI_HIP_ARCHITECTURES=gfx942
PATH=/opt/rocm/bin:$PATH .venv/bin/python -m pytest tests/test_phase12_rocm_foundation.py -q \
  | tee docs/benchmarks/data/rocm_mi300x_campaign6_2026-04-30/logs/pytest_phase12_rocm_mi300x.log
```

- [ ] **Step 2: Capture benchmark JSON**

Run on MI300X:

```bash
PATH=/opt/rocm/bin:$PATH .venv/bin/python benchmarks/bench_rocm_kernels.py \
  --profile campaign6-expectation-parity --repeat 5 --warmup 2 --json \
  > docs/benchmarks/data/rocm_mi300x_campaign6_2026-04-30/raw/rocm_campaign6_expectation_mi300x.json
PATH=/opt/rocm/bin:$PATH .venv/bin/python benchmarks/bench_rocm_kernels.py \
  --profile campaign6-matmul-parity --repeat 5 --warmup 2 --json \
  > docs/benchmarks/data/rocm_mi300x_campaign6_2026-04-30/raw/rocm_campaign6_matmul_mi300x.json
PATH=/opt/rocm/bin:$PATH .venv/bin/python benchmarks/bench_rocm_kernels.py \
  --profile campaign6-profiler --repeat 3 --warmup 1 --json \
  > docs/benchmarks/data/rocm_mi300x_campaign6_2026-04-30/raw/rocm_campaign6_profiler_mi300x.json
```

- [ ] **Step 3: Capture rocprof evidence**

Run on MI300X:

```bash
REPO_ROOT=$(pwd)
cd "$REPO_ROOT/docs/benchmarks/data/rocm_mi300x_campaign6_2026-04-30/profiler"
rocprof --hip-trace --stats -- \
  "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/benchmarks/bench_rocm_kernels.py" \
  --profile campaign6-profiler --repeat 1 --warmup 0 --json \
  > ../logs/rocm_campaign6_rocprof.log 2>&1
```

If the installed ROCm profiler uses a different command spelling, record the
exact command and failure or success in the report and keep the profiler
artifacts under the Campaign 6 profiler directory.

- [ ] **Step 4: Render assets and write report**

Run:

```bash
python scripts/render_rocm_campaign6_assets.py \
  --data-dir docs/benchmarks/data/rocm_mi300x_campaign6_2026-04-30 \
  --plot-dir docs/benchmarks/plots
```

Write `docs/benchmarks/reports/rocm_mi300x_campaign6_2026-04-30.md` with:

```text
scope and retained public behavior
host, ROCm, HIP compiler, driver/runtime, gfx target, and git commit
correctness summary for expectation and matmul
timing tables for expectation and matmul
profiler evidence summary
comparison against CPU scalar, optimized CPU selector when available, CUDA where relevant from checked evidence, and external baselines where comparable
Campaign 6 terminal-status table for all retained, rejected, unavailable, and next-wave surfaces
remaining headroom and recommended next ROCm campaign
```

- [ ] **Step 5: Refresh user-facing docs**

Update README and roadmap:

```text
README latest ROCm source-build evidence points to Campaign 6 report
README planning sources include Campaign 6 plan and Campaign 6 report
README broad performance landscape stays broad across CPU, CUDA, ROCm, and external rows
docs/roadmap.md latest ROCm/HIP campaign plan points to Campaign 6
docs/roadmap.md latest ROCm/HIP report points to Campaign 6
docs/plans/rocm_next_waves_plan.md marks Campaign 6 complete only after evidence exists
docs/architecture/rocm_backend.md records Campaign 6 outcome only after evidence exists
```

- [ ] **Step 6: Validate report assets**

Run:

```bash
python -m pytest tests/test_rocm_campaign6_assets.py -q
uv run python scripts/validate.py
```

## Task 6: Review, Merge, Push, And CI Closeout

**Files:**
- No new code files unless review fixes require them.
- Test: `uv run python scripts/validate.py`

- [ ] **Step 1: Commit implementation in sensible chunks**

Use concise commits:

```bash
git add docs/architecture/rocm_backend.md docs/benchmarks/protocol.md tests/test_phase12_rocm_foundation.py
git commit -m "test: define ROCm campaign 6 parity gates"

git add CMakeLists.txt src/hip/expectation_hip.hip.cpp src/hip/device_pauli_sum.hip.cpp src/hip/device_pauli_sum.hip.hpp bindings/python/pauli_sum_py.cpp scripts/validate.py tests/test_validate_entrypoint.py
git commit -m "feat: add HIP statevector expectation"

git add CMakeLists.txt src/hip/matmul_hip.hip.cpp src/hip/device_pauli_sum.hip.cpp bindings/python/pauli_sum_py.cpp scripts/validate.py tests/test_validate_entrypoint.py
git commit -m "feat: add HIP Pauli matmul"

git add benchmarks/bench_rocm_kernels.py scripts/render_rocm_campaign6_assets.py tests/test_rocm_campaign6_assets.py
git commit -m "bench: add ROCm campaign 6 parity profiles"

git add README.md docs/roadmap.md docs/plans/rocm_next_waves_plan.md docs/architecture/rocm_backend.md docs/benchmarks/reports/rocm_mi300x_campaign6_2026-04-30.md docs/benchmarks/data/rocm_mi300x_campaign6_2026-04-30 docs/benchmarks/plots/rocm_mi300x_campaign6_parity.svg docs/benchmarks/plots/accelerator_landscape_with_rocm.svg
git commit -m "bench: report ROCm campaign 6 parity"
```

- [ ] **Step 2: Request independent review**

Give the reviewer:

```text
branch: codex/rocm-campaign6
goal: retain HIP expectation_statevector and HIP matmul parity on MI300X
source docs: docs/plans/mi300x_rocm_optimization_campaign6_plan.md, docs/architecture/rocm_backend.md, docs/benchmarks/protocol.md, docs/quality/code_review.md
validation: local uv run python scripts/validate.py and MI300X pytest/benchmark/profiler commands
known boundaries: no HIP DLPack, no external HIP pointer statevectors, no public streams, no public workspaces, no ROCm wheels, no multi-GPU, no simultaneous CUDA+HIP
```

Resolve P0/P1 findings before merge. Record P2 deferrals in the report or
roadmap with a named follow-up.

- [ ] **Step 3: Merge locally, validate, push, and confirm CI**

Run:

```bash
git switch main
git pull --ff-only origin main
git merge --ff-only codex/rocm-campaign6
uv run python scripts/validate.py
git push origin main
gh run list --branch main --limit 1
gh run watch <run-id> --exit-status
git branch -d codex/rocm-campaign6
```

Do not close the campaign until merged-main validation passes, the push
succeeds, CI is green, and the local feature branch is deleted.

## Remaining Headroom After Campaign 6

Campaign 6 retained HIP expectation and matmul parity on the MI300X
source-build lane. The measured results point to one of these next ROCm tracks:

```text
Campaign 7 ROCm portability and release-support evidence now that expectation and matmul parity are stable on MI300X
Campaign 7 HIP performance hardening only if future profiler evidence shows expectation or matmul is dominated by a concrete retained-operation bottleneck
Campaign 7 backend-neutral accelerator architecture only if simultaneous CUDA+HIP, multi-GPU ROCm, or cross-backend object ownership becomes the next blocker
```

Public HIP DLPack remains blocked until a real ROCm consumer consumes a
versioned read-only `kDLROCM` capsule and rejects mutation of the imported view.

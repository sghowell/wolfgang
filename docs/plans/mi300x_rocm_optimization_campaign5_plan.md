# MI300X ROCm Optimization Campaign 5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide and, if real ROCm consumer evidence permits, implement the first HIP `DeviceCommutationMatrix` DLPack interop surface while closing public stream, graph, workspace, expectation, matmul, portability, packaging, multi-GPU, and simultaneous CUDA+HIP follow-ups with explicit statuses.

**Architecture:** Campaign 5 is the Wave 4 public-boundary campaign after MI300X resident commutation outputs and HIP simplify are already retained. ROCm/HIP remains source-build-only, MI300X `gfx942` evidenced, and mutually exclusive with CUDA. HIP DLPack may become public only for a read-only dense `uint8` `DeviceCommutationMatrix` export using DLPack `kDLROCM`; public streams, graph handles, public workspaces, HIP expectation, HIP matmul, ROCm wheels, multi-GPU execution, broader AMD portability claims, and simultaneous CUDA+HIP builds remain rejected or out of scope unless this campaign records a complete contract and measured acceptance evidence.

**Tech Stack:** C++20, nanobind, CMake HIP language support, ROCm/HIP runtime, DLPack protocol structs already vendored in `bindings/python/pauli_sum_py.cpp`, PyTorch ROCm or another named ROCm DLPack consumer when available, AMD Instinct MI300X `gfx942`, rocprof, pytest, NumPy, existing FastPauli CPU/CUDA/ROCm benchmark-report infrastructure.

---

## Status

```text
complete
```

## Baseline

Campaign 4 is complete:

```text
plan: docs/plans/mi300x_rocm_optimization_campaign4_plan.md
report: docs/benchmarks/reports/rocm_mi300x_campaign4_2026-04-30.md
evidence: docs/benchmarks/data/rocm_mi300x_campaign4_2026-04-30/
retained public HIP surfaces: DevicePauliSum.to_device(), DevicePauliSum.to_host(), DevicePauliSum.commutes_with_device(), DeviceCommutationMatrix.to_host(), DeviceCommutationMatrix.count_commuting(), DeviceCommutationMatrix.conflict_degrees(), DevicePauliSum.simplify()
retained private Campaign 4 change: generic multi-word HIP simplify parallel sorted-index reduce_by_key
```

Campaign 4 closed private simplify performance headroom. The remaining ROCm
headroom is no longer another simplify hillclimb by default. It is a set of
public API, interop, release-support, and additional-operation decisions:

```text
HIP DLPack or Python consumer interop needs a named PyTorch ROCm, CuPy ROCm, or equivalent consumer
public stream or graph execution needs lifetime, synchronization, and error-propagation contracts
public HIP workspace handles need a measured benefit and accepted ownership model
HIP expectation and HIP matmul need CPU/CUDA parity fixtures promoted to HIP
ROCm portability, CI, and packaging need evidence beyond a single MI300X gfx942 source-build host
backend-neutral multi-accelerator work is required before simultaneous CUDA+HIP or multi-GPU ROCm claims
```

## Campaign 5 Scope

In scope:

```text
HIP read-only DLPack contract for dense DeviceCommutationMatrix uint8 buffers
real ROCm consumer probe using PyTorch ROCm as the primary candidate and CuPy ROCm only if available
HIP __cuda_array_interface__ guard proving HIP objects are not mislabeled as CUDA memory
HIP DLPack capsule lifetime, max_version, copy, stream-token, moved-from, and read-only behavior
benchmark rows for HIP DLPack producer cost, consumer construction, consumer sum, dense to_host comparison, and compact count comparison
rocprof trace/stats capture for DLPack producer plus consumer timing where tooling permits
explicit Campaign 5 decisions for public streams, graph execution, public workspaces, HIP expectation, HIP matmul, portability, ROCm wheels, multi-GPU, and simultaneous CUDA+HIP
README broad performance landscape refresh only if retained Campaign 5 rows add new comparable evidence
Campaign 5 report with terminal statuses for every Campaign 4 remaining-headroom item
```

Hard out of implementation scope unless a task below explicitly accepts the
surface with evidence:

```text
CUDA Array Interface exposure from HIP objects
mutable HIP DLPack exports
copy=True DLPack export
legacy unversioned DLPack capsules without read-only flags
public HIP stream arguments on Pauli operations
public HIP graph replay handles
public HIP workspace handles
HIP expectation kernels
HIP matmul kernels
multi-GPU MI300X execution
ROCm binary wheels
additional AMD GPU support claims beyond MI300X gfx942 evidence
simultaneous CUDA+HIP source builds
```

## Retention Gates

HIP DLPack may be retained only when every gate passes:

```text
__dlpack_device__ returns kDLROCM device type 10 with the HIP device ordinal
__dlpack__(max_version=(1, 0), copy=None|False, stream=None|positive integer) returns a versioned read-only capsule
copy=True raises BufferError
stream=0 raises ValueError because the Python DLPack protocol treats stream 0 as ambiguous
legacy __dlpack__() without max_version raises BufferError because read-only flags require the versioned capsule path
HIP __cuda_array_interface__ remains unavailable and names HIP or ROCm in the error
at least one real ROCm consumer consumes the capsule on MI300X and matches DeviceCommutationMatrix.to_host()
the same consumer rejects mutation of the imported view or reports an immutable/read-only view that prevents writes
consumer timing is labeled separately from FastPauli kernel timing
capsule owner lifetime keeps the producing matrix alive until the consumer releases the view
single-consumer capsule reuse is rejected by the consumer or by the capsule state
CPU-only and CUDA-only builds behave unchanged
public headers include no HIP or ROCm runtime headers
FASTPAULI_ENABLE_CUDA=ON with FASTPAULI_ENABLE_HIP=ON remains a configure-time error
```

If PyTorch ROCm, CuPy ROCm, or another named ROCm DLPack consumer cannot be
installed, imported, and validated on the MI300X host, Campaign 5 must not
retain HIP DLPack or leave HIP `__dlpack__` / `__dlpack_device__` exposed. In
that case, the implementation path must keep HIP DLPack unavailable, tests must
assert the unavailable behavior, and the report records `blocked_external` with
the exact package, version, install command, import result, and consumer error.
If a consumer imports the capsule but permits mutation of the read-only view,
Campaign 5 must reject HIP DLPack retention and record that consumer result.

Public stream or graph execution may be retained only if a complete contract
and benchmark evidence meet these gates:

```text
Python API shape is specified before implementation
ownership of stream or graph handles is specified
host synchronization boundary is specified
error propagation from asynchronous HIP work is specified
shape-change and moved-from behavior is specified
device mismatch behavior is specified
end-to-end median speed improves by at least 5 percent on a retained public operation, or the surface is rejected_with_evidence
```

Public HIP workspace handles may be retained only if:

```text
the API prevents use-after-free and cross-device reuse
ownership is explicit in Python and C++
retained benchmarks show allocation pressure dominates a public operation
pre-reserved workspace reuse improves a retained public operation by at least 10 percent median without correctness regressions
```

## Campaign 5 Execution-Control Decision Record

Campaign 5 records terminal public-boundary decisions rather than adding new
stream, graph, or workspace APIs:

```text
public streams: rejected_with_evidence
public graphs: rejected_with_evidence
public workspaces: rejected_with_evidence
```

Public streams are rejected because Campaign 5 does not accept a Python handle
type, ownership rule, synchronization boundary, asynchronous error-propagation
contract, shape-change behavior, or device-mismatch behavior. Public graphs are
rejected for the same reason plus the absence of a graph replay lifetime and
shape-stability contract. Public workspaces are rejected because Campaign 5 does
not establish an ownership-safe Python API, cross-device/use-after-free
prevention, or a retained-operation benchmark showing at least a 10 percent
median speedup from pre-reserved reuse.

The checked decision evidence is emitted by:

```text
benchmarks/bench_rocm_kernels.py --profile interop-campaign5-stream-workspace-decisions
docs/benchmarks/data/rocm_mi300x_campaign5_2026-04-30/raw/rocm_campaign5_stream_workspace_decisions_mi300x.json
```

## Evidence Layout

Use this evidence root:

```text
docs/benchmarks/data/rocm_mi300x_campaign5_2026-04-30/
docs/benchmarks/data/rocm_mi300x_campaign5_2026-04-30/logs/
docs/benchmarks/data/rocm_mi300x_campaign5_2026-04-30/raw/
docs/benchmarks/data/rocm_mi300x_campaign5_2026-04-30/profiler/
docs/benchmarks/data/rocm_mi300x_campaign5_2026-04-30/summary.json
docs/benchmarks/reports/rocm_mi300x_campaign5_2026-04-30.md
docs/benchmarks/plots/rocm_mi300x_campaign5_interop.svg
docs/benchmarks/plots/accelerator_landscape_with_rocm.svg
```

If execution crosses UTC midnight, keep this evidence root only if the report
states the exact execution dates and why the root remains named
`rocm_mi300x_campaign5_2026-04-30`.

## Acceptance Criteria

Campaign 5 is complete only when every applicable item has a terminal status
in the Campaign 5 report:

```text
local CPU-only validation passes with FASTPAULI_ENABLE_HIP=OFF
HIP source build succeeds on MI300X with FASTPAULI_ENABLE_HIP=ON and FASTPAULI_HIP_ARCHITECTURES=gfx942
CUDA+HIP configure-time rejection still passes
public headers still contain no HIP or ROCm runtime headers
HIP __cuda_array_interface__ remains unavailable
HIP DLPack is either retained with real ROCm consumer and read-only mutation evidence or not exposed with a blocked_external report and unavailable-behavior tests
HIP DLPack retained rows report kDLROCM device type 10, shape, dtype, strides, read-only status, consumer library, consumer version, consumer backend, correctness digest, and timing boundaries
public streams and graph execution are accepted or rejected with explicit evidence
public HIP workspaces are accepted or rejected with explicit evidence
HIP expectation and HIP matmul receive next-campaign or out-of-scope statuses
ROCm portability, CI, packaging, multi-GPU, and simultaneous CUDA+HIP receive next-campaign or out-of-scope statuses
README broad performance landscape remains CPU/CUDA/ROCm/external rather than a narrow ROCm-only plot if refreshed
independent review is recorded before merge
```

## Task 1: Contracts, Benchmark Schema, And Source-Of-Truth Wiring

**Files:**
- Modify: `docs/architecture/rocm_backend.md`
- Modify: `docs/benchmarks/protocol.md`
- Modify: `docs/plans/rocm_next_waves_plan.md`
- Modify: `docs/roadmap.md`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `scripts/validate.py`
- Test: `uv run python scripts/validate.py`

- [ ] **Step 1: Add the Campaign 5 ROCm boundary contract**

Add a `Campaign 5 Interop And Execution-Control Boundary` section to
`docs/architecture/rocm_backend.md` with these decisions:

```text
Campaign 5 is the first ROCm wave allowed to change HIP Python interop behavior.
HIP DLPack may be retained only for read-only dense DeviceCommutationMatrix uint8 export with DLPack kDLROCM device typing.
HIP __cuda_array_interface__ remains unavailable because HIP pointers must not be presented as CUDA pointers.
HIP DLPack requires a real ROCm consumer validation on MI300X before public retention.
Public stream, graph, and workspace APIs remain unavailable unless Campaign 5 records a complete contract and measured acceptance evidence.
HIP expectation, HIP matmul, ROCm wheels, multi-GPU ROCm, broader portability, and simultaneous CUDA+HIP source builds remain separate campaigns.
```

- [ ] **Step 2: Add Campaign 5 benchmark fields**

Add a `ROCm Campaign 5 interop and execution-control fields` subsection to
`docs/benchmarks/protocol.md` requiring these JSON fields:

```text
campaign: rocm_mi300x_campaign5
operation: commutation_interop, stream_graph_decision, workspace_decision, expectation_decision, matmul_decision, portability_decision, packaging_decision, multi_gpu_decision, or multi_backend_decision
backend: hip
mode: dlpack_pytorch, dlpack_cupy, dlpack_other_consumer, cuda_array_interface_guard, stream_graph_probe, workspace_probe, expectation_decision, matmul_decision, portability_decision, packaging_decision, multi_gpu_decision, or simultaneous_cuda_hip_decision
status: ok, retained, rejected_with_evidence, blocked_external, unavailable, or out_of_scope_with_next_trigger
final_status: retained, rejected_with_evidence, blocked_external, unavailable, or out_of_scope_with_next_trigger
hip_dlpack_device_type: 10 when HIP DLPack is retained, otherwise null
hip_dlpack_device_type_name: kDLROCM when HIP DLPack is retained, otherwise unavailable
consumer_library
consumer_version
consumer_backend
consumer_available
consumer_import_error
consumer_correctness_passed
consumer_read_only_enforced
consumer_mutation_error
hip_dlpack_export_seconds and p10/p90/min/max variants when retained
consumer_from_dlpack_seconds and p10/p90/min/max variants when a consumer runs
consumer_sum_seconds and p10/p90/min/max variants when a consumer runs
hip_device_output_to_host_seconds and p10/p90/min/max variants
hip_count_commuting_axis_none_seconds and p10/p90/min/max variants
timing_boundary: dlpack_producer, framework_consumer, compact_consumer, device_output_to_host, decision_only, or benchmark_only
correctness_digest with matrix_shape, host_sum, consumer_sum, and canonical_matrix_hash
campaign5_terminal_statuses for DLPack, CUDA Array Interface guard, streams, graphs, workspaces, expectation, matmul, portability, ROCm wheels, multi-GPU, and simultaneous CUDA+HIP
```

- [ ] **Step 3: Wire the new plan into source-of-truth docs**

Update these files:

```text
README.md: link docs/plans/mi300x_rocm_optimization_campaign5_plan.md in Planning Sources and name it as the next ROCm campaign
docs/roadmap.md: identify Campaign 5 as planned next ROCm work after Campaign 4
docs/plans/rocm_next_waves_plan.md: mark Wave 4 as the active next wave and point to this executable Campaign 5 plan
AGENTS.md: add this plan to the Read First list near the other ROCm campaign plans
scripts/validate.py: add docs/plans/mi300x_rocm_optimization_campaign5_plan.md to SOURCE_OF_TRUTH_PATHS
```

- [ ] **Step 4: Validate docs wiring**

Run:

```bash
uv run python scripts/validate.py
```

Expected local result: source-of-truth files resolve, stale-marker scan passes,
CPU-only validation passes, HIP-specific tests skip on macOS without ROCm, and
benchmark smokes complete.

- [ ] **Step 5: Commit**

```bash
git add README.md AGENTS.md docs/architecture/rocm_backend.md docs/benchmarks/protocol.md docs/plans/rocm_next_waves_plan.md docs/plans/mi300x_rocm_optimization_campaign5_plan.md docs/roadmap.md scripts/validate.py
git commit -m "docs: plan ROCm campaign 5 interop"
```

## Task 2: Red Tests For HIP DLPack And Guardrails

**Files:**
- Modify: `tests/test_phase12_rocm_foundation.py`
- Test: `python -m pytest tests/test_phase12_rocm_foundation.py -q`

- [ ] **Step 1: Split the current unavailable interop test**

Replace `test_hip_interop_surfaces_remain_explicitly_unavailable_when_available`
with a CUDA Array Interface guard that always remains unavailable for HIP:

```python
def test_hip_cuda_array_interface_remains_unavailable_when_available() -> None:
    _require_hip_runtime()

    matrix = fastpauli.PauliSum.from_labels(["XI"], [1.0]).to_device().commutes_with_device(
        fastpauli.PauliSum.from_labels(["IX"], [1.0]).to_device(),
    )

    with pytest.raises((BufferError, RuntimeError, ValueError), match="HIP|ROCm|CUDA"):
        matrix.__cuda_array_interface__
```

- [ ] **Step 2: Add the HIP DLPack producer contract test**

Add this test:

```python
def test_hip_device_commutation_matrix_dlpack_contract_when_available() -> None:
    _require_hip_runtime()

    lhs = fastpauli.PauliSum.from_labels(["XI", "ZI"], [1.0, 1.0]).to_device()
    rhs = fastpauli.PauliSum.from_labels(["IX", "ZZ", "XX"], [1.0, 1.0, 1.0]).to_device()
    matrix = lhs.commutes_with_device(rhs)

    assert matrix.__dlpack_device__() == (10, matrix.device)

    with pytest.raises(BufferError, match="copy=True"):
        matrix.__dlpack__(copy=True)
    with pytest.raises(ValueError, match="stream=0"):
        matrix.__dlpack__(stream=0)
    with pytest.raises(ValueError, match="max_version"):
        matrix.__dlpack__(max_version=(0, 0))
    with pytest.raises(BufferError, match="max_version"):
        matrix.__dlpack__()

    capsule = matrix.__dlpack__(max_version=(1, 0))
    assert "dltensor_versioned" in repr(capsule)
```

This test should fail on MI300X before the DLPack implementation because HIP
currently rejects DLPack.

- [ ] **Step 3: Add a PyTorch ROCm consumer test**

Add this optional consumer test:

```python
def test_hip_device_commutation_matrix_dlpack_pytorch_rocm_consumer_when_available() -> None:
    _require_hip_runtime()
    torch = pytest.importorskip("torch", reason="torch not importable")
    if not getattr(torch.version, "hip", None):
        pytest.skip("torch importable but not a ROCm build")
    if not torch.cuda.is_available():
        pytest.skip("torch ROCm importable but no HIP device is visible through torch")

    lhs = fastpauli.PauliSum.from_labels(["XI", "ZI"], [1.0, 1.0]).to_device()
    rhs = fastpauli.PauliSum.from_labels(["IX", "ZZ", "XX"], [1.0, 1.0, 1.0]).to_device()
    matrix = lhs.commutes_with_device(rhs)
    capsule = matrix.__dlpack__(max_version=(1, 0))

    try:
        torch_view = torch.utils.dlpack.from_dlpack(capsule)
    except Exception as exc:  # noqa: BLE001 - PyTorch owns exact consumer exceptions.
        pytest.skip(f"PyTorch ROCm cannot consume FastPauli HIP DLPack: {type(exc).__name__}: {exc}")

    assert tuple(torch_view.shape) == matrix.shape
    assert torch_view.dtype == torch.uint8
    assert torch_view.is_cuda
    assert torch_view.device.index in (None, matrix.device)
    np.testing.assert_array_equal(torch_view.cpu().numpy(), matrix.to_host().astype(np.uint8))
    assert int(torch_view.sum().item()) == matrix.count_commuting()

    original = matrix.to_host().astype(np.uint8)
    try:
        torch_view[0, 0] = 1 - int(torch_view[0, 0].item())
    except (RuntimeError, TypeError, ValueError):
        pass
    else:
        mutated = matrix.to_host().astype(np.uint8)
        pytest.fail(
            "PyTorch ROCm accepted mutation of FastPauli's read-only HIP DLPack view; "
            f"original={int(original[0, 0])} mutated={int(mutated[0, 0])}"
        )
```

- [ ] **Step 4: Add single-consumer and lifetime coverage when a consumer runs**

Extend the PyTorch test or add a second test with this behavior:

```python
def test_hip_dlpack_capsule_lifetime_when_pytorch_rocm_available() -> None:
    _require_hip_runtime()
    torch = pytest.importorskip("torch", reason="torch not importable")
    if not getattr(torch.version, "hip", None) or not torch.cuda.is_available():
        pytest.skip("PyTorch ROCm DLPack consumer is unavailable")

    def make_view() -> object:
        lhs = fastpauli.PauliSum.from_labels(["XX", "ZI"], [1.0, 2.0]).to_device()
        rhs = fastpauli.PauliSum.from_labels(["YY", "XI"], [1.0, 1.0j]).to_device()
        return torch.utils.dlpack.from_dlpack(lhs.commutes_with_device(rhs))

    retained_view = make_view()
    assert int(retained_view.sum().item()) >= 0

    matrix = fastpauli.PauliSum.from_labels(["XI"], [1.0]).to_device().commutes_with_device(
        fastpauli.PauliSum.from_labels(["IX"], [1.0]).to_device(),
    )
    capsule = matrix.__dlpack__(max_version=(1, 0))
    torch.utils.dlpack.from_dlpack(capsule)
    with pytest.raises(Exception):  # noqa: BLE001 - consumer controls reused-capsule exception type.
        torch.utils.dlpack.from_dlpack(capsule)
```

- [ ] **Step 5: Run local and MI300X tests**

Run locally:

```bash
python -m pytest tests/test_phase12_rocm_foundation.py -q
```

Expected local result: HIP tests skip and non-HIP tests pass.

Run on MI300X before implementation:

```bash
PATH=/opt/rocm/bin:$PATH .venv/bin/python -m pytest tests/test_phase12_rocm_foundation.py -q
```

Expected MI300X pre-implementation result: the new HIP DLPack producer test
fails because HIP DLPack is not implemented. Keep the failure output in the
campaign notes before implementing.

- [ ] **Step 6: Commit**

```bash
git add tests/test_phase12_rocm_foundation.py
git commit -m "test: define HIP DLPack interop guards"
```

## Task 3: HIP DLPack Producer Implementation

**Files:**
- Modify: `include/fastpauli/device_commutation_matrix.hpp`
- Modify: `src/device_commutation_matrix_stub.cpp`
- Modify: `src/cuda/device_commutation_matrix.cu`
- Modify: `src/hip/device_commutation_matrix.hip.cpp`
- Modify: `bindings/python/pauli_sum_py.cpp`
- Test: `python -m pytest tests/test_phase11_cuda_kernels.py tests/test_phase12_rocm_foundation.py -q`

- [ ] **Step 1: Add backend-neutral DLPack accessors without HIP headers**

Add these declarations to `include/fastpauli/device_commutation_matrix.hpp`:

```cpp
  [[nodiscard]] std::uintptr_t data_pointer_for_dlpack() const;
  [[nodiscard]] int dlpack_device_type() const;
```

These declarations use only standard integer types and must not include DLPack,
HIP, or CUDA headers.

- [ ] **Step 2: Implement CPU-only stub behavior**

Add to `src/device_commutation_matrix_stub.cpp`:

```cpp
std::uintptr_t DeviceCommutationMatrix::data_pointer_for_dlpack() const {
  throw_cuda_rebuild_guidance();
}

int DeviceCommutationMatrix::dlpack_device_type() const {
  throw_cuda_rebuild_guidance();
}
```

If the stub uses a differently named local rebuild helper, use the existing
helper and keep the public error intent: rebuild with CUDA or HIP.

- [ ] **Step 3: Preserve CUDA DLPack behavior through the new accessors**

Add to `src/cuda/device_commutation_matrix.cu`:

```cpp
std::uintptr_t DeviceCommutationMatrix::data_pointer_for_dlpack() const {
  return data_pointer_for_cuda_array_interface();
}

int DeviceCommutationMatrix::dlpack_device_type() const {
  if (!impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }
  return 2;
}
```

The existing CUDA DLPack and CUDA Array Interface tests must continue to pass.

- [ ] **Step 4: Implement HIP DLPack pointer and device type**

Add to `src/hip/device_commutation_matrix.hip.cpp`:

```cpp
std::uintptr_t DeviceCommutationMatrix::data_pointer_for_dlpack() const {
  if (!impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }
  return reinterpret_cast<std::uintptr_t>(impl_->data);
}

int DeviceCommutationMatrix::dlpack_device_type() const {
  if (!impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }
  return 10;
}
```

Keep `data_pointer_for_cuda_array_interface()` throwing on HIP.

- [ ] **Step 5: Refactor Python DLPack binding to use backend accessors**

In `bindings/python/pauli_sum_py.cpp`, add:

```cpp
constexpr int kDlROCMDeviceType = 10;
```

Change `device_commutation_matrix_dlpack()` to call
`matrix.data_pointer_for_dlpack()` and set:

```cpp
managed->dl_tensor.device = DLDevice{matrix.dlpack_device_type(), matrix.device()};
```

Change `device_commutation_matrix_dlpack_device()` to:

```cpp
nb::tuple device_commutation_matrix_dlpack_device(const DeviceCommutationMatrix& matrix) {
  return nb::make_tuple(matrix.dlpack_device_type(), matrix.device());
}
```

Update stream error wording from CUDA-specific to accelerator-neutral wording:

```text
stream must be None or an integer accelerator stream token
```

- [ ] **Step 6: Run focused tests**

Run locally:

```bash
python -m pytest tests/test_phase11_cuda_kernels.py tests/test_phase12_rocm_foundation.py -q
```

Expected local result: CUDA and HIP runtime tests skip without local devices,
non-accelerator tests pass.

Run on MI300X:

```bash
PATH=/opt/rocm/bin:$PATH .venv/bin/python -m pytest \
  tests/test_phase12_rocm_foundation.py::test_hip_cuda_array_interface_remains_unavailable_when_available \
  tests/test_phase12_rocm_foundation.py::test_hip_device_commutation_matrix_dlpack_contract_when_available \
  tests/test_phase12_rocm_foundation.py::test_hip_device_commutation_matrix_dlpack_pytorch_rocm_consumer_when_available \
  tests/test_phase12_rocm_foundation.py::test_hip_dlpack_capsule_lifetime_when_pytorch_rocm_available \
  -q
```

Expected MI300X result: producer and guard tests pass. PyTorch tests either
pass with a ROCm PyTorch consumer or skip with an exact consumer-unavailable
reason. If all real consumer tests skip, do not retain HIP DLPack as a public
claim until Task 4 records a named consumer that passes.

If no real consumer validates, stop the HIP DLPack retention path before the
Step 7 commit. Keep or restore HIP `__dlpack__` and `__dlpack_device__`
unavailable behavior, keep the CUDA Array Interface guard, add or preserve a
HIP DLPack unavailable test for the `blocked_external` path, and continue with
Task 4 decision rows. Do not commit a candidate public HIP DLPack producer in a
state where no named ROCm consumer has passed correctness and read-only
mutation checks.

- [ ] **Step 7: Commit**

```bash
git add include/fastpauli/device_commutation_matrix.hpp src/device_commutation_matrix_stub.cpp src/cuda/device_commutation_matrix.cu src/hip/device_commutation_matrix.hip.cpp bindings/python/pauli_sum_py.cpp
git commit -m "feat: add HIP DLPack producer path"
```

## Task 4: ROCm Consumer Probe And Benchmark Profiles

**Files:**
- Modify: `benchmarks/bench_rocm_kernels.py`
- Test: `python -m pytest tests/test_phase12_rocm_foundation.py::test_rocm_simplify_benchmark_smoke_reports_campaign4_fields -q`

- [ ] **Step 1: Add Campaign 5 profile names**

Add these profiles to `benchmarks/bench_rocm_kernels.py`:

```text
interop-campaign5-dlpack-consumers
interop-campaign5-stream-workspace-decisions
interop-campaign5-profiler
```

- [ ] **Step 2: Add consumer availability helpers**

Add helper functions that return structured dictionaries for PyTorch ROCm and
CuPy ROCm:

```python
def probe_pytorch_rocm_consumer() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "consumer_library": "torch",
            "consumer_available": False,
            "consumer_import_error": f"{type(exc).__name__}: {exc}",
            "consumer_version": None,
            "consumer_backend": "unavailable",
        }
    return {
        "consumer_library": "torch",
        "consumer_available": bool(getattr(torch.version, "hip", None) and torch.cuda.is_available()),
        "consumer_import_error": "",
        "consumer_version": getattr(torch, "__version__", "unknown"),
        "consumer_backend": "rocm" if getattr(torch.version, "hip", None) else "not_rocm",
    }
```

Use the same structure for CuPy if it imports. Do not import either library at
FastPauli package import time.

- [ ] **Step 3: Add HIP DLPack timing rows**

For a retained consumer, record separate timings:

```text
hip_dlpack_export_seconds: matrix.__dlpack__(max_version=(1, 0))
consumer_from_dlpack_seconds: torch.utils.dlpack.from_dlpack(capsule) or cupy.from_dlpack(capsule)
consumer_sum_seconds: consumer sum over the imported dense uint8 view
consumer_read_only_enforced: mutation attempt raises or the consumer reports read-only immutability
consumer_mutation_error: exact mutation exception type and message when mutation is rejected
hip_device_output_to_host_seconds: matrix.to_host()
hip_count_commuting_axis_none_seconds: matrix.count_commuting()
```

The row must set:

```text
campaign: rocm_mi300x_campaign5
operation: commutation_interop
mode: dlpack_pytorch or dlpack_cupy
status: ok
final_status: retained
hip_dlpack_device_type: 10
hip_dlpack_device_type_name: kDLROCM
timing_boundary: framework_consumer
consumer_correctness_passed: true
consumer_read_only_enforced: true
```

- [ ] **Step 4: Add unavailable consumer rows**

When a consumer cannot run, emit a row rather than dropping the case:

```text
status: unavailable
final_status: blocked_external
consumer_available: false
consumer_import_error: exact import or runtime error
consumer_correctness_passed: false
timing_boundary: decision_only
```

- [ ] **Step 5: Add decision rows for streams, graphs, workspaces, expectation, matmul, portability, wheels, multi-GPU, and simultaneous CUDA+HIP**

Each decision row must name the item and final status. Expected Campaign 5
defaults are:

```text
CUDA Array Interface guard: rejected_with_evidence because HIP pointers are not CUDA pointers
public streams: rejected_with_evidence unless Task 5 accepts a complete API and measured benefit
public graphs: rejected_with_evidence unless Task 5 accepts a complete API and measured benefit
public workspaces: rejected_with_evidence unless Task 5 accepts ownership plus measured benefit
HIP expectation: out_of_scope_with_next_trigger
HIP matmul: out_of_scope_with_next_trigger
portability beyond MI300X gfx942: out_of_scope_with_next_trigger
ROCm wheels: out_of_scope_with_next_trigger
multi-GPU ROCm: out_of_scope_with_next_trigger
simultaneous CUDA+HIP: unavailable because configure-time rejection remains active
```

- [ ] **Step 6: Run benchmark smoke locally**

Run:

```bash
python benchmarks/bench_rocm_kernels.py --profile interop-campaign5-dlpack-consumers --repeat 1 --warmup 0 --json
```

Expected local result: HIP unavailable rows with `final_status` values and no
runtime import of optional consumers unless the benchmark profile is run.

- [ ] **Step 7: Run MI300X benchmark captures**

Run on MI300X:

```bash
PATH=/opt/rocm/bin:$PATH FASTPAULI_BENCHMARK_GIT_COMMIT=$(git rev-parse HEAD) \
  .venv/bin/python benchmarks/bench_rocm_kernels.py \
  --profile interop-campaign5-dlpack-consumers --repeat 5 --warmup 2 --json \
  --output docs/benchmarks/data/rocm_mi300x_campaign5_2026-04-30/raw/rocm_campaign5_dlpack_consumers_mi300x.json

PATH=/opt/rocm/bin:$PATH FASTPAULI_BENCHMARK_GIT_COMMIT=$(git rev-parse HEAD) \
  .venv/bin/python benchmarks/bench_rocm_kernels.py \
  --profile interop-campaign5-stream-workspace-decisions --repeat 5 --warmup 2 --json \
  --output docs/benchmarks/data/rocm_mi300x_campaign5_2026-04-30/raw/rocm_campaign5_stream_workspace_decisions_mi300x.json
```

- [ ] **Step 8: Commit**

```bash
git add benchmarks/bench_rocm_kernels.py docs/benchmarks/data/rocm_mi300x_campaign5_2026-04-30/raw
git commit -m "bench: add ROCm campaign 5 interop profiles"
```

## Task 5: Stream, Graph, And Workspace Decision Evidence

**Files:**
- Modify: `docs/plans/mi300x_rocm_optimization_campaign5_plan.md`
- Test: `python benchmarks/bench_rocm_kernels.py --profile interop-campaign5-stream-workspace-decisions --repeat 1 --warmup 0 --json`

- [ ] **Step 1: Evaluate public stream and graph acceptance gates**

Record whether Campaign 5 has a complete public API contract for:

```text
stream handle type
stream ownership
event ownership
host synchronization
error propagation
device mismatch handling
shape-change behavior
Python lifetime
benchmark improvement
```

If any item is missing, set stream and graph final status to
`rejected_with_evidence` and do not add a public API.

- [ ] **Step 2: Evaluate public workspace acceptance gates**

Record whether Campaign 5 has evidence for:

```text
allocation pressure dominating a retained public HIP operation
ownership-safe Python API
cross-device rejection
use-after-free prevention
at least 10 percent median speedup from pre-reserved reuse
```

If any item is missing, set public workspace final status to
`rejected_with_evidence` and keep `src/hip/workspace_hip.hip.*` private.

- [ ] **Step 3: Run decision benchmark profile**

Run locally:

```bash
python benchmarks/bench_rocm_kernels.py --profile interop-campaign5-stream-workspace-decisions --repeat 1 --warmup 0 --json
```

Expected result: structured decision rows with no public stream, graph, or
workspace API retained unless every acceptance gate is satisfied.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/bench_rocm_kernels.py docs/plans/mi300x_rocm_optimization_campaign5_plan.md
git commit -m "docs: record ROCm stream workspace decisions"
```

## Task 6: Profiler Capture, Report, Plots, And Asset Tests

**Files:**
- Create: `scripts/render_rocm_campaign5_assets.py`
- Create: `tests/test_rocm_campaign5_assets.py`
- Create: `docs/benchmarks/reports/rocm_mi300x_campaign5_2026-04-30.md`
- Create: `docs/benchmarks/plots/rocm_mi300x_campaign5_interop.svg`
- Modify: `docs/benchmarks/plots/accelerator_landscape_with_rocm.svg`
- Modify: `README.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/plans/rocm_next_waves_plan.md`
- Test: `python -m pytest tests/test_rocm_campaign5_assets.py -q`

- [ ] **Step 1: Capture rocprof evidence on MI300X**

Run:

```bash
PATH=/opt/rocm/bin:$PATH FASTPAULI_BENCHMARK_GIT_COMMIT=$(git rev-parse HEAD) \
  rocprof -d docs/benchmarks/data/rocm_mi300x_campaign5_2026-04-30/profiler \
  --hip-trace --stats \
  .venv/bin/python benchmarks/bench_rocm_kernels.py \
  --profile interop-campaign5-profiler --repeat 1 --warmup 0 --json \
  --output docs/benchmarks/data/rocm_mi300x_campaign5_2026-04-30/raw/rocm_campaign5_profiler_mi300x.json
```

If rocprof cannot capture consumer framework kernels, keep the FastPauli HIP
trace and record the exact provider limitation in the report.

- [ ] **Step 2: Add the Campaign 5 renderer**

Create `scripts/render_rocm_campaign5_assets.py` that:

```text
loads all raw Campaign 5 JSON files
emits docs/benchmarks/data/rocm_mi300x_campaign5_2026-04-30/summary.json
emits docs/benchmarks/plots/rocm_mi300x_campaign5_interop.svg
refreshes docs/benchmarks/plots/accelerator_landscape_with_rocm.svg only when retained comparable rows exist
preserves CPU/CUDA/ROCm/external rows in the landscape
records unavailable consumer reasons in summary.json
```

- [ ] **Step 3: Add asset tests**

Create `tests/test_rocm_campaign5_assets.py` with tests that verify:

```text
summary.json campaign is rocm_mi300x_campaign5
every row has final_status
DLPack rows have consumer fields
retained DLPack rows have hip_dlpack_device_type 10 and hip_dlpack_device_type_name kDLROCM
CUDA Array Interface guard row is not retained
stream, graph, workspace, expectation, matmul, portability, ROCm wheels, multi-GPU, and simultaneous CUDA+HIP have terminal statuses
renderer reproduces summary and plots from checked raw inputs
```

- [ ] **Step 4: Write the Campaign 5 report**

Create `docs/benchmarks/reports/rocm_mi300x_campaign5_2026-04-30.md` with:

```text
scope
host and build inventory
implementation outcome
consumer availability table
DLPack contract table
timing table for producer, consumer, dense to_host, and compact count rows
stream, graph, and workspace decision table
additional-operation and release-support status table
profiler evidence and limitations
README landscape status
remaining headroom and recommended next campaign
```

- [ ] **Step 5: Update README, roadmap, and next-waves status**

Update:

```text
README.md: latest ROCm source-build evidence points to the Campaign 5 report after completion
docs/roadmap.md: Campaign 5 is marked complete after report evidence exists
docs/plans/rocm_next_waves_plan.md: Wave 4 Campaign 5 status moves from planned to complete, with next recommended wave named
```

- [ ] **Step 6: Validate assets**

Run:

```bash
python scripts/render_rocm_campaign5_assets.py \
  --data-dir docs/benchmarks/data/rocm_mi300x_campaign5_2026-04-30 \
  --plot-dir docs/benchmarks/plots

python -m pytest tests/test_rocm_campaign5_assets.py -q
```

- [ ] **Step 7: Commit**

```bash
git add README.md docs/roadmap.md docs/plans/rocm_next_waves_plan.md scripts/render_rocm_campaign5_assets.py tests/test_rocm_campaign5_assets.py docs/benchmarks/data/rocm_mi300x_campaign5_2026-04-30 docs/benchmarks/plots/rocm_mi300x_campaign5_interop.svg docs/benchmarks/plots/accelerator_landscape_with_rocm.svg docs/benchmarks/reports/rocm_mi300x_campaign5_2026-04-30.md
git commit -m "bench: report ROCm campaign 5 interop"
```

## Task 7: Full Validation, Review, Merge, Push, And CI

**Files:**
- Modify as needed only for review fixes.
- Test: `uv run python scripts/validate.py`

- [ ] **Step 1: Run full local validation**

Run:

```bash
git diff --check
uv run python scripts/validate.py
```

Expected result: both commands pass locally. HIP runtime tests may skip on
macOS CPU-only validation.

- [ ] **Step 2: Run full MI300X validation**

Run on MI300X:

```bash
PATH=/opt/rocm/bin:$PATH .venv/bin/python -m pytest tests/test_phase12_rocm_foundation.py -q
PATH=/opt/rocm/bin:$PATH .venv/bin/python -m pytest tests/test_rocm_campaign5_assets.py -q
PATH=/opt/rocm/bin:$PATH .venv/bin/python benchmarks/bench_rocm_kernels.py --profile interop-campaign5-dlpack-consumers --repeat 1 --warmup 0 --json
```

Expected result: tests pass or real consumer tests skip with exact external
consumer reasons. The benchmark emits terminal-status rows.

- [ ] **Step 3: Complete independent agent-driven review**

Review scope:

```text
HIP DLPack lifetime and device typing
CUDA behavior preservation
HIP __cuda_array_interface__ guard
consumer benchmark integrity
stream, graph, and workspace decision evidence
report and plot claims
```

Resolve blocking findings and rerun the affected local and MI300X checks.

- [ ] **Step 4: Merge and push**

Run:

```bash
git switch main
git merge --ff-only codex/rocm-campaign5
uv run python scripts/validate.py
git push origin main
gh run watch --exit-status
git branch -d codex/rocm-campaign5
```

Use the actual feature branch name if it differs. Do not delete the branch
until `main` has been pushed and CI is green.

## Campaign 5 Closeout Decision

At the end of Campaign 5, the report must name the next ROCm campaign. The
expected next campaign is one of:

```text
Campaign 6 HIP expectation and matmul parity, if interop is retained or blocked by external packages and no execution-control blocker remains
Campaign 6 ROCm portability and release-support evidence, if public interop is accepted and additional operation work should wait
Campaign 6 backend-neutral accelerator design, if simultaneous CUDA+HIP or multi-GPU ROCm becomes the dominant blocker
```

The decision must be evidence-based and recorded in the Campaign 5 report,
`docs/roadmap.md`, and `docs/plans/rocm_next_waves_plan.md`.

Closeout result:

```text
report: docs/benchmarks/reports/rocm_mi300x_campaign5_2026-04-30.md
outcome: HIP DLPack rejected_with_evidence because PyTorch ROCm consumed the candidate versioned kDLROCM capsule in a temporary candidate probe but accepted mutation of the read-only view
next campaign: Campaign 6 HIP expectation and HIP matmul parity
```

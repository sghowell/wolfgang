"""CUDA kernel behavior and CPU/GPU equivalence."""

from __future__ import annotations

import gc
import json
import subprocess
import sys
from types import SimpleNamespace

import fastpauli
import fastpauli._fastpauli_core as core
import numpy as np
import pytest

from benchmarks import bench_cuda_kernels


def _cuda_status() -> dict[str, object]:
    return dict(core._cuda_status())


def _require_cuda_runtime() -> None:
    status = _cuda_status()
    if not status["built"]:
        pytest.skip(str(status["skip_reason"]))
    if not status["runtime_available"]:
        pytest.skip(str(status["skip_reason"]))


def _require_supported_cupy_runtime() -> None:
    cupy = pytest.importorskip("cupy", reason="CuPy is required for CUDA interop tests")
    bench_cuda_kernels._require_supported_cupy_runtime_for_current_cuda_architecture(cupy)


def _labels_and_coeffs(op: fastpauli.PauliSum) -> tuple[list[str], list[complex]]:
    labels, coeffs = op.to_labels()
    return list(labels), [complex(value) for value in coeffs]


def _assert_same_operator(lhs: fastpauli.PauliSum, rhs: fastpauli.PauliSum) -> None:
    lhs_labels, lhs_coeffs = _labels_and_coeffs(lhs)
    rhs_labels, rhs_coeffs = _labels_and_coeffs(rhs)
    assert lhs_labels == rhs_labels
    np.testing.assert_allclose(lhs_coeffs, rhs_coeffs, rtol=1.0e-12, atol=1.0e-12)


@pytest.fixture(autouse=True)
def _collect_accelerator_python_wrappers() -> object:
    yield
    cupy = sys.modules.get("cupy")
    if cupy is not None:
        try:
            cupy.cuda.runtime.deviceSynchronize()
        except Exception:
            pass
        try:
            cupy.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
    gc.collect()


def test_phase_eleven_public_cuda_kernel_surface_is_exposed() -> None:
    assert hasattr(fastpauli.DevicePauliSum, "simplify")
    assert hasattr(fastpauli.DevicePauliSum, "expectation_statevector")
    assert hasattr(fastpauli.DevicePauliSum, "commutes_with")
    assert hasattr(fastpauli.DevicePauliSum, "commutes_with_into")
    assert hasattr(fastpauli.DevicePauliSum, "commutes_with_device")
    assert hasattr(fastpauli.DevicePauliSum, "matmul")
    assert hasattr(fastpauli.DeviceCommutationMatrix, "count_commuting")
    assert hasattr(fastpauli.DeviceCommutationMatrix, "conflict_degrees")
    assert hasattr(fastpauli.DeviceCommutationMatrix, "__dlpack__")
    assert hasattr(fastpauli.DeviceCommutationMatrix, "__dlpack_device__")


def test_cuda_binding_lifecycle_subprocess_exits_without_nanobind_leaks() -> None:
    _require_cuda_runtime()

    script = """
import gc
import sys

import fastpauli

op = fastpauli.PauliSum.from_labels(["X", "Z"], [1.0, 2.0])
device = op.to_device()
matrix = device.commutes_with_device(device)
assert matrix.count_commuting() >= 0
assert sys.getrefcount(op) == 2
assert sys.getrefcount(device) == 2
assert sys.getrefcount(matrix) == 2
del matrix, device, op
gc.collect()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    combined_output = completed.stdout + completed.stderr
    assert "nanobind: leaked" not in combined_output


def test_cuda_cupy_commutation_consumers_subprocess_exit_without_nanobind_leaks() -> None:
    _require_cuda_runtime()
    _require_supported_cupy_runtime()

    script = """
import gc
import fastpauli
import cupy

lhs = fastpauli.PauliSum.from_labels(["XI", "ZI"], [1.0, 1.0]).to_device()
rhs = fastpauli.PauliSum.from_labels(["IX", "ZZ", "XX"], [1.0, 1.0, 1.0]).to_device()
matrix = lhs.commutes_with_device(rhs)

dlpack_view = cupy.from_dlpack(matrix)
array_view = cupy.asarray(matrix)

assert int(cupy.sum(dlpack_view).get()) == matrix.count_commuting()
assert int(cupy.sum(array_view).get()) == matrix.count_commuting()

del dlpack_view, array_view, matrix, lhs, rhs
cupy.get_default_memory_pool().free_all_blocks()
gc.collect()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    combined_output = completed.stdout + completed.stderr
    assert "nanobind: leaked" not in combined_output


def test_cuda_simplify_matches_cpu_canonical_output() -> None:
    _require_cuda_runtime()

    op = fastpauli.PauliSum.from_labels(
        ["XX", "ZI", "XX", "II", "YY", "YY"],
        [1.0 + 2.0j, -0.5, -0.25 + 0.5j, 1.0e-14, 2.0j, -2.0j],
    )

    expected = op.simplify()
    actual = op.to_device().simplify().to_host()

    _assert_same_operator(actual, expected)


def test_cuda_simplify_tolerance_errors_match_cpu() -> None:
    _require_cuda_runtime()

    device_op = fastpauli.PauliSum.from_labels(["X"], [1.0]).to_device()

    with pytest.raises(ValueError, match="simplify tolerances"):
        device_op.simplify(atol=-1.0)
    with pytest.raises(ValueError, match="simplify tolerances"):
        device_op.simplify(rtol=float("nan"))


def test_private_cuda_workspace_probe_reports_lifetime_without_pointers() -> None:
    _require_cuda_runtime()
    if not hasattr(core, "_cuda_workspace_probe_for_testing"):
        pytest.skip("private CUDA workspace probe is available only in CUDA builds")

    report = core._cuda_workspace_probe_for_testing(
        device=0,
        reserve_bytes=(4096, 8192, 4096),
        reset=True,
        release=True,
    )

    assert report["cuda_enabled"] is True
    assert report["runtime_available"] is True
    assert report["status"] == "ok"
    assert report["device_ordinal"] == 0
    assert report["allocation_count"] == 2
    assert report["growth_count"] == 2
    assert report["high_watermark_bytes"] == 8192

    snapshots = list(report["snapshots"])
    assert snapshots[0]["label"] == "before_reserve"
    assert snapshots[1]["reserved_bytes"] == 4096
    assert snapshots[2]["reserved_bytes"] == 8192
    assert snapshots[3]["reserved_bytes"] == 8192
    assert snapshots[-2]["label"] == "after_reset"
    assert snapshots[-2]["reserved_bytes"] == 8192
    assert snapshots[-1]["label"] == "after_release"
    assert snapshots[-1]["reserved_bytes"] == 0
    assert all("pointer" not in snapshot for snapshot in snapshots)


def test_private_cuda_workspace_probe_rejects_invalid_device() -> None:
    _require_cuda_runtime()
    if not hasattr(core, "_cuda_workspace_probe_for_testing"):
        pytest.skip("private CUDA workspace probe is available only in CUDA builds")

    device_count = int(_cuda_status()["device_count"])
    with pytest.raises(ValueError, match="device ordinal is out of range"):
        core._cuda_workspace_probe_for_testing(device=device_count, reserve_bytes=(1024,))


def test_cuda_statevector_expectation_matches_cpu_for_complex_dtypes() -> None:
    _require_cuda_runtime()

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


def test_cuda_statevector_expectation_rejects_invalid_host_arrays() -> None:
    _require_cuda_runtime()

    device_op = fastpauli.PauliSum.from_labels(["ZI", "IZ"], [1.0, -0.5]).to_device()

    with pytest.raises(TypeError, match="complex64 or complex128"):
        device_op.expectation_statevector(np.ones(4, dtype=np.float64))
    with pytest.raises(TypeError, match="C-contiguous"):
        device_op.expectation_statevector(np.ones(8, dtype=np.complex128)[::2])
    with pytest.raises(ValueError, match="len\\(psi\\) == 2 \\*\\* num_qubits"):
        device_op.expectation_statevector(np.ones(2, dtype=np.complex128))


def test_cuda_statevector_expectation_rejects_invalid_cuda_array_interface_metadata() -> None:
    _require_cuda_runtime()

    class FakeCudaArray:
        def __init__(self, interface: dict[str, object]) -> None:
            self.__cuda_array_interface__ = interface

    device_op = fastpauli.PauliSum.from_labels(["ZI", "IZ"], [1.0, -0.5]).to_device()

    with pytest.raises(TypeError, match="complex64 or complex128"):
        device_op.expectation_statevector(
            FakeCudaArray({"shape": (4,), "typestr": "<f8", "data": (1, False), "version": 3})
        )
    with pytest.raises(TypeError, match="C-contiguous"):
        device_op.expectation_statevector(
            FakeCudaArray({"shape": (4,), "typestr": "<c16", "strides": (32,), "data": (1, False), "version": 3})
        )
    with pytest.raises(ValueError, match="1-dimensional"):
        device_op.expectation_statevector(
            FakeCudaArray({"shape": (2, 2), "typestr": "<c16", "data": (1, False), "version": 3})
        )


def test_cuda_empty_expectation_still_validates_cuda_array_interface_pointer() -> None:
    _require_cuda_runtime()

    class FakeCudaArray:
        def __init__(self, interface: dict[str, object]) -> None:
            self.__cuda_array_interface__ = interface

    device_op = fastpauli.PauliSum.empty(2).to_device()

    with pytest.raises(TypeError, match="non-null"):
        device_op.expectation_statevector(
            FakeCudaArray({"shape": (4,), "typestr": "<c16", "data": (0, False), "version": 3})
        )


def test_cuda_statevector_expectation_accepts_cuda_array_interface_when_available() -> None:
    _require_cuda_runtime()
    cupy = pytest.importorskip("cupy", reason="CuPy is required for CUDA array interface tests")

    op = fastpauli.PauliSum.from_labels(["ZI", "IZ", "XX"], [1.0, -0.5, 0.25])
    psi = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    device_psi = cupy.asarray(psi)

    np.testing.assert_allclose(
        op.to_device().expectation_statevector(device_psi),
        op.expectation_statevector(psi),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_cuda_commutation_matches_cpu_and_keeps_guardrails() -> None:
    _require_cuda_runtime()

    lhs = fastpauli.PauliSum.from_labels(["XX", "ZI", "IZ"], [1.0, 2.0, -1.0])
    rhs = fastpauli.PauliSum.from_labels(["YY", "XI"], [1.0, 1.0j])
    lhs_device = lhs.to_device()
    rhs_device = rhs.to_device()

    np.testing.assert_array_equal(lhs_device.commutes_with(rhs_device), lhs.commutes_with(rhs))

    with pytest.raises(ValueError, match="commutation matrix entry count exceeds"):
        lhs_device.commutes_with(rhs_device, max_commutation_matrix_entries=3)


def test_cuda_device_commutation_matrix_matches_cpu_and_exposes_cuda_array_interface() -> None:
    _require_cuda_runtime()

    lhs = fastpauli.PauliSum.from_labels(["XX", "ZI", "IZ"], [1.0, 2.0, -1.0])
    rhs = fastpauli.PauliSum.from_labels(["YY", "XI"], [1.0, 1.0j])
    lhs_device = lhs.to_device()
    rhs_device = rhs.to_device()

    matrix = lhs_device.commutes_with_device(rhs_device)

    assert isinstance(matrix, fastpauli.DeviceCommutationMatrix)
    assert matrix.shape == (lhs.num_terms, rhs.num_terms)
    assert matrix.device == lhs_device.device
    assert matrix.dtype == "uint8"
    assert matrix.num_entries == lhs.num_terms * rhs.num_terms

    cuda_interface = matrix.__cuda_array_interface__
    assert cuda_interface["shape"] == matrix.shape
    assert cuda_interface["typestr"] == "|u1"
    assert cuda_interface["version"] >= 3
    assert cuda_interface["strides"] in (None, (matrix.shape[1], 1))
    assert isinstance(cuda_interface["data"][0], int)
    assert cuda_interface["data"][0] != 0 or matrix.num_entries == 0
    assert cuda_interface.get("stream") in (None, 1)

    np.testing.assert_array_equal(matrix.to_host(), lhs.commutes_with(rhs))


def test_cuda_device_commutation_matrix_count_commuting_matches_numpy() -> None:
    _require_cuda_runtime()

    lhs = fastpauli.PauliSum.from_labels(["XI", "ZI"], [1.0, 1.0]).to_device()
    rhs = fastpauli.PauliSum.from_labels(["IX", "ZZ", "XX"], [1.0, 1.0, 1.0]).to_device()
    matrix = lhs.commutes_with_device(rhs)
    host = matrix.to_host().astype(np.uint64)

    assert matrix.count_commuting() == int(host.sum())
    np.testing.assert_array_equal(matrix.count_commuting(axis=1), host.sum(axis=1, dtype=np.uint64))
    np.testing.assert_array_equal(matrix.count_commuting(axis=0), host.sum(axis=0, dtype=np.uint64))


def test_cuda_device_commutation_matrix_count_commuting_rejects_bad_axis() -> None:
    _require_cuda_runtime()

    matrix = fastpauli.DeviceCommutationMatrix.empty((2, 3), device=0)

    with pytest.raises(ValueError, match="axis"):
        matrix.count_commuting(axis=2)


def test_cuda_device_commutation_matrix_conflict_degrees_matches_numpy() -> None:
    _require_cuda_runtime()

    lhs = fastpauli.PauliSum.from_labels(["XI", "ZI"], [1.0, 1.0]).to_device()
    rhs = fastpauli.PauliSum.from_labels(["IX", "ZZ", "XX"], [1.0, 1.0, 1.0]).to_device()
    matrix = lhs.commutes_with_device(rhs)
    host_conflicts = np.logical_not(matrix.to_host()).astype(np.uint64)

    assert matrix.conflict_degrees() == int(host_conflicts.sum())
    np.testing.assert_array_equal(
        matrix.conflict_degrees(axis=1),
        host_conflicts.sum(axis=1, dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        matrix.conflict_degrees(axis=0),
        host_conflicts.sum(axis=0, dtype=np.uint64),
    )


def test_cuda_device_commutation_matrix_conflict_degrees_rejects_bad_axis() -> None:
    _require_cuda_runtime()

    matrix = fastpauli.DeviceCommutationMatrix.empty((2, 3), device=0)

    with pytest.raises(ValueError, match="axis"):
        matrix.conflict_degrees(axis=2)


def test_cuda_device_commutation_matrix_dlpack_cupy_consumer_matches_numpy() -> None:
    _require_cuda_runtime()
    _require_supported_cupy_runtime()

    script = """
import gc
import cupy
import fastpauli
import numpy as np

lhs = fastpauli.PauliSum.from_labels(["XI", "ZI"], [1.0, 1.0]).to_device()
rhs = fastpauli.PauliSum.from_labels(["IX", "ZZ", "XX"], [1.0, 1.0, 1.0]).to_device()
matrix = lhs.commutes_with_device(rhs)
cupy_view = cupy.from_dlpack(matrix)

assert matrix.__dlpack_device__() == (2, matrix.device)
assert cupy_view.shape == matrix.shape
assert cupy_view.dtype == cupy.uint8
np.testing.assert_array_equal(cupy.asnumpy(cupy_view), matrix.to_host().astype(np.uint8))
assert int(cupy.sum(cupy_view).get()) == matrix.count_commuting()

del cupy_view, matrix, lhs, rhs
cupy.get_default_memory_pool().free_all_blocks()
gc.collect()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "nanobind: leaked" not in completed.stdout + completed.stderr


def test_cuda_device_commutation_matrix_dlpack_lifetime_and_capsule_guardrails() -> None:
    _require_cuda_runtime()
    _require_supported_cupy_runtime()
    cupy = pytest.importorskip("cupy", reason="CuPy is required for DLPack tests")

    lhs = fastpauli.PauliSum.from_labels(["XI", "ZI"], [1.0, 1.0]).to_device()
    rhs = fastpauli.PauliSum.from_labels(["IX", "ZZ", "XX"], [1.0, 1.0, 1.0]).to_device()
    matrix = lhs.commutes_with_device(rhs)
    capsule = None
    cupy_view = None
    versioned_capsule = None
    versioned_view = None
    retained_view = None

    try:
        with pytest.raises(BufferError, match="copy=True"):
            matrix.__dlpack__(copy=True)
        with pytest.raises(ValueError, match="stream=0"):
            matrix.__dlpack__(stream=0)
        with pytest.raises(BufferError, match="max_version"):
            matrix.__dlpack__(max_version=(0, 0))
        with pytest.raises(BufferError, match="max_version"):
            matrix.__dlpack__()

        capsule = matrix.__dlpack__(max_version=(1, 0))
        cupy_view = cupy.from_dlpack(capsule)
        with pytest.raises(Exception):
            cupy.from_dlpack(capsule)

        versioned_capsule = matrix.__dlpack__(max_version=(1, 0))
        versioned_view = cupy.from_dlpack(versioned_capsule)
        np.testing.assert_array_equal(
            cupy.asnumpy(versioned_view),
            matrix.to_host().astype(np.uint8),
        )

        def make_view() -> object:
            local_lhs = fastpauli.PauliSum.from_labels(["XX", "ZI"], [1.0, 2.0]).to_device()
            local_rhs = fastpauli.PauliSum.from_labels(["YY", "XI"], [1.0, 1.0j]).to_device()
            local_matrix = local_lhs.commutes_with_device(local_rhs)
            local_view = cupy.from_dlpack(local_matrix)
            del local_matrix, local_lhs, local_rhs
            gc.collect()
            return local_view

        retained_view = make_view()
        assert int(cupy.sum(retained_view).get()) >= 0
        assert int(cupy.sum(cupy_view).get()) == matrix.count_commuting()
    finally:
        del matrix, lhs, rhs
        del capsule, cupy_view, versioned_capsule, versioned_view, retained_view
        cupy.get_default_memory_pool().free_all_blocks()
        gc.collect()


def test_cuda_cupy_compile_failure_is_skipped_when_runtime_lacks_arch_support(monkeypatch) -> None:
    _require_cuda_runtime()

    import cupy

    compile_exception = cupy.cuda.compiler.CompileException(
        "nvrtc: error: invalid value for --gpu-architecture (-arch)",
        "kernel.cu",
        "demo_kernel",
        (),
        "nvrtc",
    )

    monkeypatch.setattr(cupy, "asarray", lambda *_args, **_kwargs: (_ for _ in ()).throw(compile_exception))

    with pytest.raises(pytest.skip.Exception, match="CuPy runtime does not support"):
        bench_cuda_kernels._require_supported_cupy_runtime_for_current_cuda_architecture(cupy)


def test_cuda_cupy_reduction_compile_failure_reports_unavailable(monkeypatch) -> None:
    compile_exception = RuntimeError("nvrtc: error: invalid value for --gpu-architecture (-arch)")
    fake_view = SimpleNamespace(shape=(1, 1), dtype="uint8")
    fake_cupy = SimpleNamespace(
        uint8="uint8",
        asarray=lambda _output: fake_view,
        asnumpy=lambda value: value,
        sum=lambda _view, axis=None: (_ for _ in ()).throw(compile_exception),
    )

    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setattr(bench_cuda_kernels, "_require_supported_cupy_runtime_for_current_cuda_architecture", lambda cupy: None)
    monkeypatch.setattr(bench_cuda_kernels, "_cuda_compute_capability", lambda: (10, 3))
    monkeypatch.setattr(
        bench_cuda_kernels,
        "timed_call",
        lambda fn, warmup, repeat: (fn(), {"median": 0.0, "min": 0.0, "max": 0.0}),
    )

    lhs = SimpleNamespace(
        num_terms=1,
        commutes_with_device=lambda rhs_device, max_commutation_matrix_entries: object(),
    )
    rhs = SimpleNamespace(num_terms=1)

    report = bench_cuda_kernels.timed_cupy_device_output_consumer_cuda_commutation(
        lhs_device=lhs,
        rhs_device=rhs,
        expected=[[True]],
        max_entries=1,
        warmup=0,
        repeat=1,
    )

    assert report["available"] is False
    assert "gpu-architecture" in report["unavailable_reason"]


def test_cuda_device_commutation_matrix_dlpack_pytorch_consumer_matches_numpy() -> None:
    _require_cuda_runtime()
    torch = pytest.importorskip("torch", reason="torch not importable")
    if not torch.cuda.is_available():
        pytest.skip("torch importable but torch.cuda.is_available() is false")

    lhs = fastpauli.PauliSum.from_labels(["XI", "ZI"], [1.0, 1.0]).to_device()
    rhs = fastpauli.PauliSum.from_labels(["IX", "ZZ", "XX"], [1.0, 1.0, 1.0]).to_device()
    matrix = lhs.commutes_with_device(rhs)
    capsule = matrix.__dlpack__(max_version=(1, 0))
    try:
        torch_view = torch.utils.dlpack.from_dlpack(capsule)
    except Exception as exc:
        pytest.skip(
            "torch CUDA available but torch.utils.dlpack cannot consume the "
            f"versioned read-only capsule: {type(exc).__name__}: {exc}"
        )

    assert tuple(torch_view.shape) == matrix.shape
    assert torch_view.dtype == torch.uint8
    assert torch_view.is_cuda
    assert torch_view.device.index in (None, matrix.device)
    np.testing.assert_array_equal(torch_view.cpu().numpy(), matrix.to_host().astype(np.uint8))


def test_private_fused_consumer_hook_reports_cpu_only_unavailable() -> None:
    status = _cuda_status()
    if status["built"]:
        pytest.skip("CPU-only unavailable contract is tested only without CUDA build support")
    if not hasattr(core, "_benchmark_cuda_fused_commutation_consumer"):
        pytest.skip("private fused consumer hook is not exposed in this CPU-only build")

    report = core._benchmark_cuda_fused_commutation_consumer("csr_anticommutation_graph")
    assert report["status"] == "unavailable"
    assert report["mode"] == "csr_anticommutation_graph"
    assert "WOLFGANG_ENABLE_CUDA=ON" in report["unavailable_reason"]
    assert not hasattr(fastpauli, "_benchmark_cuda_fused_commutation_consumer")

    with pytest.raises(ValueError, match="mode must be"):
        core._benchmark_cuda_fused_commutation_consumer("typoed_mode")

    with pytest.raises(RuntimeError, match="WOLFGANG_ENABLE_CUDA=ON"):
        core._benchmark_cuda_fused_commutation_consumer(
            "csr_anticommutation_graph",
            require_cuda=True,
        )


def test_private_campaign8_device_resident_consumer_hook_reports_cpu_only_unavailable() -> None:
    status = _cuda_status()
    if status["built"]:
        pytest.skip("CPU-only unavailable contract is tested only without CUDA build support")
    if not hasattr(core, "_benchmark_cuda_device_resident_consumer"):
        pytest.skip("private campaign8 hook is not exposed in this CPU-only build")

    report = core._benchmark_cuda_device_resident_consumer("device_resident_graph")
    assert report["status"] == "unavailable"
    assert report["mode"] == "device_resident_graph"
    assert report["campaign"] == "h100_campaign8"
    assert report["device_resident_graph_status"] == "unavailable"
    assert "WOLFGANG_ENABLE_CUDA=ON" in report["unavailable_reason"]
    assert not hasattr(fastpauli, "_benchmark_cuda_device_resident_consumer")

    with pytest.raises(ValueError, match="mode must be"):
        core._benchmark_cuda_device_resident_consumer("typoed_mode")

    with pytest.raises(RuntimeError, match="WOLFGANG_ENABLE_CUDA=ON"):
        core._benchmark_cuda_device_resident_consumer(
            "device_resident_graph",
            require_cuda=True,
        )


def _expected_csr_from_commutation_matrix(host: np.ndarray) -> tuple[list[int], list[int]]:
    anti_commuting = np.logical_not(np.asarray(host, dtype=np.bool_))
    row_offsets = [0]
    col_indices: list[int] = []
    for row in anti_commuting:
        cols = np.flatnonzero(row)
        col_indices.extend(int(col) for col in cols)
        row_offsets.append(len(col_indices))
    return row_offsets, col_indices


def test_private_cuda_fused_csr_consumer_matches_dense_matrix() -> None:
    _require_cuda_runtime()

    lhs = fastpauli.PauliSum.from_labels(["XI", "ZI", "YY"], [1.0, 1.0, 1.0]).to_device()
    rhs = fastpauli.PauliSum.from_labels(["IX", "ZZ", "XX"], [1.0, 1.0, 1.0]).to_device()
    matrix = lhs.commutes_with_device(rhs)
    host = matrix.to_host()
    expected_offsets, expected_cols = _expected_csr_from_commutation_matrix(host)

    report = core._benchmark_cuda_fused_commutation_consumer(
        "csr_anticommutation_graph",
        matrix,
        include_outputs=True,
        require_cuda=True,
    )

    assert report["status"] == "ok"
    assert report["mode"] == "csr_anticommutation_graph"
    assert report["rows"] == host.shape[0]
    assert report["cols"] == host.shape[1]
    assert list(report["row_offsets"]) == expected_offsets
    assert list(report["col_indices"]) == expected_cols
    assert report["correctness_digest"]["edge_count"] == len(expected_cols)
    assert report["output_sizes"]["row_offsets_uint64"] == host.shape[0] + 1
    assert report["output_sizes"]["col_indices_uint64"] == len(expected_cols)


def test_private_cuda_fused_conflict_degrees_match_counts() -> None:
    _require_cuda_runtime()

    lhs = fastpauli.PauliSum.from_labels(["XI", "ZI", "YY"], [1.0, 1.0, 1.0]).to_device()
    rhs = fastpauli.PauliSum.from_labels(["IX", "ZZ", "XX"], [1.0, 1.0, 1.0]).to_device()
    matrix = lhs.commutes_with_device(rhs)
    host = matrix.to_host().astype(np.bool_)
    conflicts = np.logical_not(host).astype(np.uint64)

    report = core._benchmark_cuda_fused_commutation_consumer(
        "conflict_degrees",
        matrix,
        include_outputs=True,
        require_cuda=True,
    )

    np.testing.assert_array_equal(report["row_conflicts"], conflicts.sum(axis=1, dtype=np.uint64))
    np.testing.assert_array_equal(report["col_conflicts"], conflicts.sum(axis=0, dtype=np.uint64))
    assert report["correctness_digest"]["row_conflict_sum"] == int(conflicts.sum())
    assert report["correctness_digest"]["col_conflict_sum"] == int(conflicts.sum())


def test_private_cuda_fused_grouping_summary_is_deterministic() -> None:
    _require_cuda_runtime()

    lhs = fastpauli.PauliSum.from_labels(["XI", "ZI", "YY"], [1.0, 1.0, 1.0]).to_device()
    rhs = fastpauli.PauliSum.from_labels(["IX", "ZZ", "XX"], [1.0, 1.0, 1.0]).to_device()
    matrix = lhs.commutes_with_device(rhs)
    host = matrix.to_host().astype(np.bool_)
    row_conflicts = np.logical_not(host).sum(axis=1, dtype=np.uint64)
    expected_top_rows = sorted(
        range(row_conflicts.size),
        key=lambda index: (-int(row_conflicts[index]), index),
    )[:2]

    report = core._benchmark_cuda_fused_commutation_consumer(
        "grouping_summary",
        matrix,
        top_k=2,
        include_outputs=True,
        require_cuda=True,
    )

    assert list(report["top_row_indices"]) == expected_top_rows
    assert list(report["top_row_conflicts"]) == [
        int(row_conflicts[index]) for index in expected_top_rows
    ]
    np.testing.assert_array_equal(report["row_conflicts"], row_conflicts)


def test_private_campaign8_device_resident_graph_returns_compact_digest() -> None:
    _require_cuda_runtime()

    lhs = fastpauli.PauliSum.from_labels(["XI", "ZI", "YY"], [1.0, 1.0, 1.0]).to_device()
    rhs = fastpauli.PauliSum.from_labels(["IX", "ZZ", "XX"], [1.0, 1.0, 1.0]).to_device()
    matrix = lhs.commutes_with_device(rhs)
    host = matrix.to_host().astype(np.bool_)
    conflicts = np.logical_not(host).astype(np.uint64)
    row_offsets, col_indices = _expected_csr_from_commutation_matrix(host)

    report = core._benchmark_cuda_device_resident_consumer(
        "device_resident_graph",
        matrix,
        include_outputs=True,
        require_cuda=True,
    )

    assert report["status"] == "ok"
    assert report["campaign"] == "h100_campaign8"
    assert report["mode"] == "device_resident_graph"
    assert report["boundary"] == "compact_host_copy"
    assert report["timing_boundary"] == "device_resident_consumer"
    assert report["device_resident_graph_status"] == "retained"
    assert report["public_grouping_api_status"] == "not_applicable"
    assert report["dlpack_interop_status"] == "not_applicable"
    assert report["stream_graph_status"] == "not_applicable"
    assert report["scatter_tuning_status"] == "rejected_no_consumer"
    assert report["validation_csr_status"] == "available"
    assert report["output_sizes"]["full_csr_host_bytes"] > 0
    assert "col_indices" not in report
    assert list(report["validation_row_offsets"]) == row_offsets
    assert list(report["validation_col_indices"]) == col_indices
    np.testing.assert_array_equal(report["row_conflicts"], conflicts.sum(axis=1, dtype=np.uint64))
    np.testing.assert_array_equal(report["col_conflicts"], conflicts.sum(axis=0, dtype=np.uint64))
    assert report["correctness_digest"]["edge_count"] == int(conflicts.sum())
    assert report["correctness_digest"]["validation_csr_edge_count"] == len(col_indices)


def test_private_campaign8_device_grouping_consumer_is_deterministic() -> None:
    _require_cuda_runtime()

    lhs = fastpauli.PauliSum.from_labels(["XI", "ZI", "YY"], [1.0, 1.0, 1.0]).to_device()
    rhs = fastpauli.PauliSum.from_labels(["IX", "ZZ", "XX"], [1.0, 1.0, 1.0]).to_device()
    matrix = lhs.commutes_with_device(rhs)

    first = core._benchmark_cuda_device_resident_consumer(
        "device_grouping_consumer",
        matrix,
        top_k=2,
        include_outputs=True,
        require_cuda=True,
    )
    second = core._benchmark_cuda_device_resident_consumer(
        "device_grouping_consumer",
        matrix,
        top_k=2,
        include_outputs=True,
        require_cuda=True,
    )

    assert first["status"] == "ok"
    assert first["public_grouping_api_status"] == "deferred"
    assert first["boundary"] == "private_benchmark_only"
    assert first["top_row_indices"] == second["top_row_indices"]
    assert first["top_row_conflicts"] == second["top_row_conflicts"]
    assert first["correctness_digest"] == second["correctness_digest"]
    assert first["output_sizes"]["full_csr_host_bytes"] == 0


def test_private_campaign8_deferred_and_implemented_modes_report_explicit_reasons() -> None:
    _require_cuda_runtime()

    matrix = fastpauli.PauliSum.from_labels(["XI"], [1.0]).to_device().commutes_with_device(
        fastpauli.PauliSum.from_labels(["IX"], [1.0]).to_device()
    )

    dlpack = core._benchmark_cuda_device_resident_consumer(
        "dlpack_consumer",
        matrix,
        require_cuda=True,
    )
    stream_graph = core._benchmark_cuda_device_resident_consumer(
        "stream_graph_probe",
        matrix,
        require_cuda=True,
    )
    scatter = core._benchmark_cuda_device_resident_consumer(
        "csr_scatter_ab",
        matrix,
        require_cuda=True,
    )

    assert dlpack["status"] == "ok"
    assert dlpack["dlpack_interop_status"] == "implemented"
    assert dlpack["boundary"] == "framework_consumer"
    assert dlpack["output_sizes"]["dense_uint8_device_bytes"] == matrix.num_entries
    assert dlpack["correctness_digest"]["commuting_count"] == matrix.count_commuting()
    assert dlpack["unavailable_reason"] == ""
    assert stream_graph["status"] == "unavailable"
    assert stream_graph["stream_graph_status"] == "deferred"
    assert "CUDA Graph" in stream_graph["unavailable_reason"]
    assert scatter["status"] == "unavailable"
    assert scatter["scatter_tuning_status"] == "rejected_no_consumer"
    assert "full CSR" in scatter["unavailable_reason"]


def test_cuda_device_commutation_matrix_cupy_consumer_matches_numpy() -> None:
    _require_cuda_runtime()
    _require_supported_cupy_runtime()

    script = """
import gc
import cupy
import fastpauli
import numpy as np

lhs = fastpauli.PauliSum.from_labels(["XI", "ZI"], [1.0, 1.0]).to_device()
rhs = fastpauli.PauliSum.from_labels(["IX", "ZZ", "XX"], [1.0, 1.0, 1.0]).to_device()
matrix = lhs.commutes_with_device(rhs)
cupy_view = cupy.asarray(matrix)

assert cupy_view.shape == matrix.shape
assert cupy_view.dtype == cupy.uint8
np.testing.assert_array_equal(cupy.asnumpy(cupy_view), matrix.to_host().astype(np.uint8))
assert int(cupy.sum(cupy_view).get()) == matrix.count_commuting()

del cupy_view, matrix, lhs, rhs
cupy.get_default_memory_pool().free_all_blocks()
gc.collect()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "nanobind: leaked" not in completed.stdout + completed.stderr


def test_cuda_device_commutation_matrix_reuse_and_guardrails() -> None:
    _require_cuda_runtime()

    status = _cuda_status()
    device_count = int(status["device_count"])
    if device_count < 2:
        pytest.skip("wrong-output-device check requires at least two visible CUDA devices")

    def exercise() -> None:
        lhs = fastpauli.PauliSum.from_labels(["XX", "ZI", "IZ"], [1.0, 2.0, -1.0])
        rhs = fastpauli.PauliSum.from_labels(["YY", "XI"], [1.0, 1.0j])
        lhs_device = lhs.to_device()
        rhs_device = rhs.to_device()

        output = fastpauli.DeviceCommutationMatrix.empty(
            (lhs.num_terms, rhs.num_terms),
            device=lhs_device.device,
        )
        same = lhs_device.commutes_with_device(rhs_device, output=output)

        assert same is output
        np.testing.assert_array_equal(output.to_host(), lhs.commutes_with(rhs))

        wrong_shape = fastpauli.DeviceCommutationMatrix.empty(
            (lhs.num_terms + 1, rhs.num_terms),
            device=lhs_device.device,
        )
        with pytest.raises(ValueError, match="output shape"):
            lhs_device.commutes_with_device(rhs_device, output=wrong_shape)

        with pytest.raises(ValueError, match="commutation matrix entry count exceeds"):
            lhs_device.commutes_with_device(rhs_device, max_commutation_matrix_entries=3)

        wrong_device = fastpauli.DeviceCommutationMatrix.empty(
            (lhs.num_terms, rhs.num_terms),
            device=1,
        )
        with pytest.raises(ValueError, match="same device"):
            lhs_device.commutes_with_device(rhs_device, output=wrong_device)

    exercise()
    gc.collect()


def test_cuda_commutation_can_fill_reused_bool_output_buffer() -> None:
    _require_cuda_runtime()

    lhs = fastpauli.PauliSum.from_labels(["XX", "ZI", "IZ"], [1.0, 2.0, -1.0])
    rhs = fastpauli.PauliSum.from_labels(["YY", "XI"], [1.0, 1.0j])
    lhs_device = lhs.to_device()
    rhs_device = rhs.to_device()
    output = np.empty(lhs.num_terms * rhs.num_terms, dtype=np.bool_)

    result = lhs_device.commutes_with_into(rhs_device, output)

    assert result is None
    np.testing.assert_array_equal(output.reshape(lhs.num_terms, rhs.num_terms), lhs.commutes_with(rhs))

    with pytest.raises(ValueError, match="output buffer size"):
        lhs_device.commutes_with_into(rhs_device, output[:-1])
    with pytest.raises(TypeError, match="output dtype must be bool"):
        lhs_device.commutes_with_into(rhs_device, np.empty(output.size, dtype=np.uint8))


def test_cuda_commutation_matches_cpu_for_two_word_inputs() -> None:
    _require_cuda_runtime()

    lhs = fastpauli.PauliSum.from_sparse_list(
        [
            ("X", [0], 1.0),
            ("Z", [64], -0.5),
            ("YZ", [1, 64], 0.25j),
        ],
        num_qubits=65,
    )
    rhs = fastpauli.PauliSum.from_sparse_list(
        [
            ("Y", [0], -1.0),
            ("X", [64], 2.0),
            ("XZ", [0, 64], 0.75),
        ],
        num_qubits=65,
    )

    np.testing.assert_array_equal(
        lhs.to_device().commutes_with(rhs.to_device()),
        lhs.commutes_with(rhs),
    )


def test_cuda_matmul_matches_cpu_and_keeps_guardrails() -> None:
    _require_cuda_runtime()

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


def test_cuda_benchmark_smoke_reports_availability_and_timing_fields() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/bench_cuda_kernels.py",
            "--smoke",
            "--repeat",
            "1",
            "--json",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["benchmark"] == "cuda_kernels"
    assert "cuda_status" in report
    assert "cases" in report
    if report["cuda_status"]["built"] and report["cuda_status"]["runtime_available"]:
        assert report["cases"]
        first = report["cases"][0]["results"]
        assert "cpu_scalar_seconds" in first
        assert "cpu_default_seconds" in first
        assert "cpu_default_p10_seconds" in first
        assert "cpu_default_p90_seconds" in first
        assert "cpu_scalar_p10_seconds" in first
        assert "cpu_scalar_p90_seconds" in first
        assert "repeat_count" in first
        assert "warmup_count" in first
        assert "cuda_transfer_inclusive_seconds" in first
        assert "cuda_transfer_inclusive_p10_seconds" in first
        assert "cuda_transfer_inclusive_p90_seconds" in first
        assert "cuda_device_resident_seconds" in first
        assert "cuda_device_resident_p10_seconds" in first
        assert "cuda_device_resident_p90_seconds" in first
        assert "instrumentation" in report["cases"][0]
        instrumentation = report["cases"][0]["instrumentation"]
        assert instrumentation["workspace"]["enabled"] is False
        assert instrumentation["cuda_stream_mode"] == "default_stream_synchronize_before_return"
        pairwise = next(case for case in report["cases"] if case["name"] == "pairwise_commutation")
        assert "cuda_device_resident_preallocated_seconds" in pairwise["results"]
        assert "cuda_device_resident_preallocated_p10_seconds" in pairwise["results"]
        assert "cuda_device_resident_preallocated_p90_seconds" in pairwise["results"]
        assert "cuda_device_resident_reused_device_output_seconds" in pairwise["results"]


def _stub_cuda_benchmark_report_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench_cuda_kernels, "benchmark_environment", lambda *args, **kwargs: {"stub": True})
    monkeypatch.setattr(bench_cuda_kernels, "git_commit", lambda: "test-commit")
    monkeypatch.setattr(bench_cuda_kernels, "command_string", lambda: "bench_cuda_kernels.py")
    monkeypatch.setattr(bench_cuda_kernels.core, "_build_info", lambda: {"available_cpu_backends": []})
    monkeypatch.setattr(
        bench_cuda_kernels.core,
        "_cuda_status",
        lambda: {"built": True, "runtime_available": True, "skip_reason": None},
    )

    def fake_case(*, warmup: int, repeat: int, **_: object) -> dict[str, object]:
        return {
            "name": "stub_case",
            "results": {
                "warmup_count": warmup,
                "repeat_count": repeat,
            },
        }

    monkeypatch.setattr(bench_cuda_kernels, "run_simplify_case", fake_case)
    monkeypatch.setattr(bench_cuda_kernels, "run_expectation_case", fake_case)
    monkeypatch.setattr(bench_cuda_kernels, "run_commutation_case", fake_case)
    monkeypatch.setattr(bench_cuda_kernels, "run_matmul_case", fake_case)



def test_cuda_benchmark_smoke_preserves_explicit_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_cuda_benchmark_report_dependencies(monkeypatch)

    report = bench_cuda_kernels.build_report(
        bench_cuda_kernels.argparse.Namespace(
            json=True,
            smoke=True,
            profile="default",
            repeat=3,
            warmup=10,
            seed=9051,
            output=None,
        )
    )

    assert report["timing_policy"]["profile"] == "smoke"
    assert report["timing_policy"]["warmup"] == 10
    assert {case["results"]["warmup_count"] for case in report["cases"]} == {10}



def test_cuda_benchmark_smoke_defaults_warmup_when_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_cuda_benchmark_report_dependencies(monkeypatch)

    report = bench_cuda_kernels.build_report(
        bench_cuda_kernels.argparse.Namespace(
            json=True,
            smoke=True,
            profile="default",
            repeat=3,
            warmup=None,
            seed=9051,
            output=None,
        )
    )

    assert report["timing_policy"]["profile"] == "smoke"
    assert report["timing_policy"]["warmup"] == 0
    assert {case["results"]["warmup_count"] for case in report["cases"]} == {0}


def test_cuda_kernel_benchmark_output_option_writes_report(tmp_path) -> None:
    output = tmp_path / "cuda_kernels_smoke.json"
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/bench_cuda_kernels.py",
            "--smoke",
            "--repeat",
            "1",
            "--json",
            "--output",
            str(output),
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(completed.stdout)


def test_cuda_benchmark_selects_all_available_cpu_variants_for_pairwise_case() -> None:
    build_info = {
        "available_cpu_backends": ["scalar", "avx2", "avx512", "neon"],
        "unavailable_cpu_backends": {"tbb": "not_compiled", "sve": "not_compiled"},
        "optimized_cpu_kernels": {
            "tbb": [],
            "avx512": ["commutes_with_words_1_2"],
            "avx2": ["commutes_with_words_1_2"],
            "neon": ["commutes_with_words_1_2", "full_group_commutation_graph_words_1_2"],
            "sve": [],
        },
    }

    assert bench_cuda_kernels.cpu_optimized_selectors_for_case(
        "pairwise_commutation",
        build_info,
    ) == ["avx512", "avx2", "neon"]
    assert bench_cuda_kernels.cpu_optimized_unavailable_for_case(
        "pairwise_commutation",
        build_info,
    ) == {"tbb": "not_compiled", "sve": "not_compiled"}
    assert bench_cuda_kernels.cpu_optimized_selectors_for_case(
        "statevector_expectation",
        build_info,
    ) == []


def test_cuda_benchmark_campaign2_timing_and_instrumentation_helpers() -> None:
    summary = bench_cuda_kernels.timing_summary([3.0, 1.0, 2.0, 4.0, 5.0])
    assert summary == {
        "median": 3.0,
        "p10": 1.0,
        "p90": 5.0,
        "min": 1.0,
        "max": 5.0,
    }

    instrumentation = bench_cuda_kernels.campaign2_instrumentation_for_case(
        "statevector_expectation",
    )
    assert instrumentation["workspace"]["mode"] == "absent"
    assert instrumentation["temporary_storage_bytes"]["available"] is False
    assert instrumentation["allocation_count"]["available"] is False
    assert instrumentation["result_materialization"] == (
        "host scalar complex result copied from device accumulator"
    )

    campaign3 = bench_cuda_kernels.campaign2_instrumentation_for_case(
        "simplify_duplicate_pressure",
        {"num_qubits": 16, "num_terms": 1000, "survivor_count": 100},
    )
    assert campaign3["temporary_storage_bytes"]["available"] is True
    assert campaign3["temporary_storage_bytes"]["implementation_path"] == (
        "packed_key32_sort_reduce"
    )
    assert campaign3["allocation_count"]["available"] is True
    assert campaign3["duplicate_survivor_count"] == 100


def test_cuda_benchmark_preallocated_timing_schema_helper() -> None:
    result = {"results": {}}

    bench_cuda_kernels.add_preallocated_commutation_timing_fields(
        result,
        {"median": 3.0, "p10": 2.0, "p90": 4.0, "min": 1.0, "max": 5.0},
    )

    assert result["results"] == {
        "cuda_device_resident_preallocated_seconds": 3.0,
        "cuda_device_resident_preallocated_p10_seconds": 2.0,
        "cuda_device_resident_preallocated_p90_seconds": 4.0,
        "cuda_device_resident_preallocated_min_seconds": 1.0,
        "cuda_device_resident_preallocated_max_seconds": 5.0,
    }

    bench_cuda_kernels.add_reused_device_output_commutation_timing_fields(
        result,
        {"median": 6.0, "p10": 5.0, "p90": 7.0, "min": 4.0, "max": 8.0},
    )
    assert result["results"]["cuda_device_resident_reused_device_output_seconds"] == 6.0
    assert result["results"]["cuda_device_resident_reused_device_output_p10_seconds"] == 5.0
    assert result["results"]["cuda_device_resident_reused_device_output_p90_seconds"] == 7.0

    result["instrumentation"] = {}
    bench_cuda_kernels.add_public_device_output_commutation_timing_fields(
        result,
        allocate_timing={"median": 9.0, "p10": 8.0, "p90": 10.0, "min": 7.0, "max": 11.0},
        reuse_timing={"median": 4.0, "p10": 3.0, "p90": 5.0, "min": 2.0, "max": 6.0},
        to_host_timing={"median": 1.0, "p10": 0.8, "p90": 1.2, "min": 0.7, "max": 1.3},
        cuda_array_interface_timing={
            "median": 0.1,
            "p10": 0.08,
            "p90": 0.12,
            "min": 0.07,
            "max": 0.13,
        },
    )
    assert result["results"]["cuda_device_output_allocate_seconds"] == 9.0
    assert result["results"]["cuda_device_output_reuse_seconds"] == 4.0
    assert result["results"]["cuda_device_output_to_host_seconds"] == 1.0
    assert result["results"]["cuda_device_output_dense_to_host_seconds"] == 1.0
    assert result["results"]["cuda_device_output_cuda_array_interface_export_seconds"] == 0.1
    assert result["instrumentation"]["result_materialization_target"] == "device_uint8_matrix"
    assert result["instrumentation"]["timing_boundary"] == (
        "device_output_allocating,device_output_reused,device_output_to_host"
    )
    assert result["instrumentation"]["public_device_output"]["timing_boundaries"] == [
        "device_output_allocating",
        "device_output_reused",
        "device_output_to_host",
        "device_output_cuda_array_interface_export",
    ]

    bench_cuda_kernels.add_device_output_consumer_timing_fields(
        result,
        {
            "total": {"median": 0.3, "p10": 0.2, "p90": 0.4, "min": 0.1, "max": 0.5},
            "axis0": {"median": 0.6, "p10": 0.5, "p90": 0.7, "min": 0.4, "max": 0.8},
            "axis1": {"median": 0.9, "p10": 0.8, "p90": 1.0, "min": 0.7, "max": 1.1},
            "to_host_bytes": 88,
        },
    )
    assert result["results"]["cuda_device_output_consumer_total_seconds"] == 0.3
    assert result["results"]["cuda_device_output_consumer_axis0_seconds"] == 0.6
    assert result["results"]["cuda_device_output_consumer_axis1_seconds"] == 0.9
    assert result["results"]["cuda_device_output_consumer_to_host_bytes"] == 88
    assert result["instrumentation"]["device_output_consumer"]["status"] == "public_compact_summary"

    bench_cuda_kernels.add_cupy_consumer_timing_fields(
        result,
        {
            "available": True,
            "unavailable_reason": None,
            "asarray": {"median": 1.1, "p10": 1.0, "p90": 1.2, "min": 0.9, "max": 1.3},
            "total": {"median": 1.4, "p10": 1.3, "p90": 1.5, "min": 1.2, "max": 1.6},
            "axis0": {"median": 1.7, "p10": 1.6, "p90": 1.8, "min": 1.5, "max": 1.9},
            "axis1": {"median": 2.0, "p10": 1.9, "p90": 2.1, "min": 1.8, "max": 2.2},
            "dense_to_host": {"median": 2.3, "p10": 2.2, "p90": 2.4, "min": 2.1, "max": 2.5},
        },
    )
    assert result["results"]["cupy_consumer_available"] is True
    assert result["results"]["cupy_asarray_export_seconds"] == 1.1
    assert result["results"]["cupy_sum_total_seconds"] == 1.4
    assert result["results"]["cupy_sum_axis0_seconds"] == 1.7
    assert result["results"]["cupy_sum_axis1_seconds"] == 2.0
    assert result["results"]["cupy_dense_to_host_seconds"] == 2.3

"""CUDA backend foundation behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
import wolfgang_quantum
import wolfgang_quantum._wolfgang_core as core

ROOT = Path(__file__).resolve().parents[1]


def _labels_and_coeffs(op: wolfgang_quantum.PauliSum) -> tuple[list[str], list[complex]]:
    labels, coeffs = op.to_labels()
    return list(labels), [complex(value) for value in coeffs]


def test_phase_ten_public_cuda_transfer_surface_is_exposed() -> None:
    assert hasattr(wolfgang_quantum, "DevicePauliSum")
    assert hasattr(wolfgang_quantum, "DeviceCommutationMatrix")
    assert hasattr(wolfgang_quantum.PauliSum, "to_device")
    assert hasattr(wolfgang_quantum, "cuda_available")
    assert hasattr(wolfgang_quantum, "cuda_devices")
    assert hasattr(core, "_cuda_status")


def test_cuda_status_distinguishes_build_time_absence() -> None:
    status = core._cuda_status()
    build_info = core._build_info()

    assert status["built"] == build_info["cuda_enabled"]
    assert "cuda_toolkit_version" in build_info
    assert "CMAKE_CUDA_HOST_COMPILER" in build_info["compiler_build_config"]
    assert "WOLFGANG_CUDA_HOST_COMPILER" in build_info["compiler_build_config"]
    assert "WOLFGANG_CUDA_HOST_COMPILER_SOURCE" in build_info["compiler_build_config"]
    assert isinstance(status["devices"], list)
    if not build_info["cuda_enabled"]:
        assert status["runtime_available"] is False
        assert status["device_count"] == 0
        assert "built without CUDA" in status["skip_reason"]
        assert "WOLFGANG_ENABLE_CUDA=ON" in status["skip_reason"]


def test_cmake_cuda_build_records_host_compiler_without_forcing_selection() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "set(CMAKE_CUDA_HOST_COMPILER" not in cmake
    assert "CUDAHOSTCXX" in cmake
    assert "WOLFGANG_CUDA_HOST_COMPILER_METADATA" in cmake


def test_cmake_cuda_build_pins_cudatoolkit_root_from_selected_compiler() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "get_filename_component(WOLFGANG_CUDAToolkit_ROOT_FROM_CMAKE_CUDA_COMPILER" in cmake
    assert "set(CUDAToolkit_ROOT \"${WOLFGANG_CUDAToolkit_ROOT_FROM_CMAKE_CUDA_COMPILER}\" CACHE PATH \"Resolved CUDA toolkit root\" FORCE)" in cmake
    assert "set(CUDAToolkit_ROOT \"${WOLFGANG_CUDAToolkit_ROOT_FROM_CUDACXX}\" CACHE PATH \"Resolved CUDA toolkit root\" FORCE)" in cmake


def test_cmake_cuda_build_links_driver_api_when_driver_symbols_are_used() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "target_link_libraries(_wolfgang_core PRIVATE CUDA::cuda_driver CUDA::cudart)" in cmake


def test_to_device_reports_clear_cpu_only_error_or_round_trips() -> None:
    op = wolfgang_quantum.PauliSum.from_labels(
        ["IXYZ", "ZZII", "IIII"],
        [1.25 + 0.5j, -2.0j, 3.0],
    )

    cuda_status = core._cuda_status()
    hip_status = core._hip_status()
    metal_status = core._metal_status()
    if not cuda_status["built"] and not hip_status["built"] and not metal_status["built"]:
        with pytest.raises(RuntimeError, match="built without CUDA.*WOLFGANG_ENABLE_CUDA=ON"):
            op.to_device()
        return

    if cuda_status["built"]:
        if not cuda_status["runtime_available"]:
            pytest.skip(cuda_status["skip_reason"])
        expected_backend = "cuda"
    else:
        if hip_status["built"]:
            if not hip_status["runtime_available"]:
                pytest.skip(hip_status["skip_reason"])
            expected_backend = "hip"
        elif not metal_status["runtime_available"]:
            pytest.skip(metal_status["skip_reason"])
        else:
            expected_backend = "metal"

    device_op = op.to_device(device=0)

    assert isinstance(device_op, wolfgang_quantum.DevicePauliSum)
    assert device_op.backend == expected_backend
    assert device_op.num_qubits == op.num_qubits
    assert device_op.num_terms == op.num_terms
    assert device_op.device == 0

    host_op = device_op.to_host()
    assert _labels_and_coeffs(host_op) == _labels_and_coeffs(op)


def test_device_commutation_matrix_reports_clear_cpu_only_error_or_allocates() -> None:
    cuda_status = core._cuda_status()
    hip_status = core._hip_status()
    metal_status = core._metal_status()

    if not cuda_status["built"] and not hip_status["built"] and not metal_status["built"]:
        with pytest.raises(RuntimeError, match="built without CUDA.*WOLFGANG_ENABLE_CUDA=ON"):
            wolfgang_quantum.DeviceCommutationMatrix.empty((2, 3), device=0)
        return

    if cuda_status["built"]:
        if not cuda_status["runtime_available"]:
            pytest.skip(cuda_status["skip_reason"])
    elif hip_status["built"]:
        if not hip_status["runtime_available"]:
            pytest.skip(hip_status["skip_reason"])
    elif metal_status["built"]:
        if not metal_status["runtime_available"]:
            pytest.skip(metal_status["skip_reason"])

    matrix = wolfgang_quantum.DeviceCommutationMatrix.empty((2, 3), device=0)
    assert matrix.shape == (2, 3)
    assert matrix.device == 0
    assert matrix.dtype == "uint8"
    assert matrix.num_entries == 6


def test_public_headers_do_not_include_cuda_or_thrust_headers() -> None:
    forbidden_tokens = ("cuda_runtime", "cuda.h", "thrust/")
    offenders: list[str] = []

    for header in sorted((ROOT / "include" / "fastpauli").glob("*.hpp")):
        source = header.read_text(encoding="utf-8")
        if any(token in source for token in forbidden_tokens):
            offenders.append(str(header.relative_to(ROOT)))

    assert not offenders, "public CPU headers must not include CUDA headers: " + ", ".join(offenders)

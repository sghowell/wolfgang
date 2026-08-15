from __future__ import annotations

import fastpauli
import fastpauli._fastpauli_core as core
import pytest


def test_accelerator_status_reports_structured_backend_sets() -> None:
    status = core._accelerator_status()
    build_info = core._build_info()

    assert status["active_backend"] in {"none", "cuda", "hip", "metal"}
    assert "compiled_backends" in status
    assert "available_backends" in status
    assert "compiled_accelerator_backends" in status
    assert "available_accelerator_backends" in status
    assert status["compiled_backends"][0] == "cpu"

    compiled_accelerators = set(status["compiled_accelerator_backends"])
    available_accelerators = set(status["available_accelerator_backends"])
    assert compiled_accelerators <= {"cuda", "hip", "metal"}
    assert available_accelerators <= compiled_accelerators
    assert set(status["compiled_backends"]) == {"cpu", *compiled_accelerators}
    assert set(status["available_backends"]) == {"cpu", *available_accelerators}
    assert build_info["accelerator_build_mode"] in {
        "cpu_only",
        "cuda_only",
        "hip_only",
        "metal_only",
        "unsupported_mixed_cuda_hip_request",
        "unsupported_mixed_accelerator_request",
    }
    assert set(build_info["compiled_accelerator_backends"]) == compiled_accelerators
    assert set(build_info["runtime_visible_accelerator_backends"]) == available_accelerators


def test_backend_selector_policy_is_testable_without_mixed_hardware() -> None:
    select = core._accelerator_backend_selection_for_testing

    assert select(None, True, False, False, False, False, False) == "cuda"
    assert select("auto", False, False, True, False, False, False) == "hip"
    assert select("auto", False, False, False, False, True, False) == "metal"
    assert select(None, True, True, True, False, False, False) == "cuda"
    assert select(None, True, False, True, True, False, False) == "hip"
    assert select(None, True, False, False, False, True, True) == "metal"
    assert select("cuda", True, False, True, True, True, True) == "cuda"
    assert select("hip", True, True, True, False, True, True) == "hip"
    assert select("metal", True, True, True, False, True, True) == "metal"

    with pytest.raises(RuntimeError, match='require backend="cuda", backend="hip", or backend="metal"'):
        select(None, True, True, True, True, True, True)
    with pytest.raises(RuntimeError, match='require backend="cuda", backend="hip", or backend="metal"'):
        select("auto", True, False, True, False, True, False)
    with pytest.raises(RuntimeError, match="built without CUDA support"):
        select("cuda", False, False, True, True, True, True)
    with pytest.raises(RuntimeError, match="built without HIP support"):
        select("hip", True, True, False, False, True, True)
    with pytest.raises(RuntimeError, match="built without Metal support"):
        select("metal", True, True, True, True, False, False)
    with pytest.raises(ValueError, match="backend must be None, 'auto', 'cuda', 'hip', or 'metal'"):
        select("bogus", True, True, True, True, True, True)
    with pytest.raises(ValueError, match="backend must be None, 'auto', 'cuda', 'hip', or 'metal'"):
        select("", True, True, True, True, True, True)


def test_backend_device_validation_policy_is_testable_without_mixed_hardware() -> None:
    validate = core._accelerator_context_validation_for_testing

    assert validate("commutes_with", "cuda", 0, "cuda", 0) == "ok"
    assert validate("matmul", "hip", 2, "hip", 2) == "ok"
    assert validate("commutes_with", "metal", 0, "metal", 0) == "ok"

    with pytest.raises(ValueError, match="operation=commutes_with"):
        validate("commutes_with", "cuda", 0, "hip", 0)
    with pytest.raises(ValueError, match="left_backend=cuda"):
        validate("commutes_with", "cuda", 0, "hip", 0)
    with pytest.raises(ValueError, match="right_backend=hip"):
        validate("commutes_with", "cuda", 0, "hip", 0)
    with pytest.raises(ValueError, match="same device"):
        validate("matmul", "hip", 0, "hip", 1)
    with pytest.raises(ValueError, match="left_device=0"):
        validate("matmul", "hip", 0, "hip", 1)
    with pytest.raises(ValueError, match="right_device=1"):
        validate("matmul", "hip", 0, "hip", 1)


def test_to_device_accepts_backend_selector_and_rejects_invalid_values() -> None:
    op = fastpauli.PauliSum.from_labels(["X"], [1.0])
    status = core._accelerator_status()

    with pytest.raises(ValueError, match="backend must be None, 'auto', 'cuda', 'hip', or 'metal'"):
        op.to_device(backend="bogus")

    if "cuda" not in status["compiled_accelerator_backends"]:
        with pytest.raises(RuntimeError, match="built without CUDA support"):
            op.to_device(backend="cuda")
    if "hip" not in status["compiled_accelerator_backends"]:
        with pytest.raises(RuntimeError, match="built without HIP support"):
            op.to_device(backend="hip")
    if "metal" not in status["compiled_accelerator_backends"]:
        with pytest.raises(RuntimeError, match="built without Metal support"):
            op.to_device(backend="metal")
    if not status["compiled_accelerator_backends"]:
        with pytest.raises(RuntimeError, match="built without CUDA, HIP, or Metal accelerator support"):
            op.to_device(backend=None)
        with pytest.raises(RuntimeError, match="built without CUDA, HIP, or Metal accelerator support"):
            op.to_device(backend="auto")


def test_device_commutation_matrix_accepts_backend_selector_and_exposes_property() -> None:
    assert "backend" in dir(fastpauli.DeviceCommutationMatrix)

    status = core._accelerator_status()

    with pytest.raises(ValueError, match="backend must be None, 'auto', 'cuda', 'hip', or 'metal'"):
        fastpauli.DeviceCommutationMatrix.empty((1, 1), backend="bogus")

    if "cuda" not in status["compiled_accelerator_backends"]:
        with pytest.raises(RuntimeError, match="built without CUDA support"):
            fastpauli.DeviceCommutationMatrix.empty((1, 1), backend="cuda")
    if "hip" not in status["compiled_accelerator_backends"]:
        with pytest.raises(RuntimeError, match="built without HIP support"):
            fastpauli.DeviceCommutationMatrix.empty((1, 1), backend="hip")
    if "metal" not in status["compiled_accelerator_backends"]:
        with pytest.raises(RuntimeError, match="built without Metal support"):
            fastpauli.DeviceCommutationMatrix.empty((1, 1), backend="metal")
    if not status["compiled_accelerator_backends"]:
        with pytest.raises(RuntimeError, match="built without CUDA, HIP, or Metal accelerator support"):
            fastpauli.DeviceCommutationMatrix.empty((1, 1), backend=None)
        with pytest.raises(RuntimeError, match="built without CUDA, HIP, or Metal accelerator support"):
            fastpauli.DeviceCommutationMatrix.empty((1, 1), backend="auto")

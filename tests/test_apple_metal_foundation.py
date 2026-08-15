"""Apple Metal source-build accelerator foundation behavior."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import fastpauli
import fastpauli._fastpauli_core as core
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _cmake_executable() -> str:
    try:
        import cmake as cmake_package  # type: ignore[import-not-found]
    except Exception:
        cmake_package = None
    if cmake_package is not None:
        packaged_cmake = Path(cmake_package.CMAKE_BIN_DIR) / "cmake"
        if packaged_cmake.exists():
            return str(packaged_cmake)
    venv_cmake = Path(sys.executable).with_name("cmake")
    if venv_cmake.exists():
        return str(venv_cmake)
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake is not available")
    return cmake


def test_cmake_helper_prefers_packaged_cmake(monkeypatch) -> None:
    fake_cmake = types.SimpleNamespace(CMAKE_BIN_DIR="/opt/cmake/bin")
    original_exists = Path.exists

    monkeypatch.setitem(sys.modules, "cmake", fake_cmake)
    monkeypatch.setattr(sys, "executable", "/usr/bin/python")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/cmake")
    monkeypatch.setattr(
        Path,
        "exists",
        lambda self: True if str(self) in {"/usr/bin/cmake", "/opt/cmake/bin/cmake"} else original_exists(self),
    )

    assert _cmake_executable() == "/opt/cmake/bin/cmake"


def _run_cmake_configure_with_options(*options: str) -> subprocess.CompletedProcess[str]:
    build_dir_name = (
        "pytest-metal-"
        + "-".join(option.replace("=", "-").replace("_", "-") for option in options)
    )
    build_dir = ROOT / "_skbuild" / build_dir_name
    return subprocess.run(
        [
            _cmake_executable(),
            "-S",
            str(ROOT),
            "-B",
            str(build_dir),
            "-DWOLFGANG_ENABLE_NATIVE=OFF",
            f"-DPython_EXECUTABLE={sys.executable}",
            *options,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _labels_and_coeffs(op: fastpauli.PauliSum) -> tuple[list[str], list[complex]]:
    labels, coeffs = op.to_labels()
    return list(labels), [complex(value) for value in coeffs]


def _assert_same_operator(lhs: fastpauli.PauliSum, rhs: fastpauli.PauliSum) -> None:
    lhs_labels, lhs_coeffs = _labels_and_coeffs(lhs)
    rhs_labels, rhs_coeffs = _labels_and_coeffs(rhs)
    assert lhs_labels == rhs_labels
    np.testing.assert_allclose(lhs_coeffs, rhs_coeffs, rtol=1.0e-12, atol=1.0e-12)


def _require_metal_runtime() -> dict:
    status = core._metal_status()
    if not status["runtime_available"]:
        pytest.skip(status["skip_reason"])
    return status


def _multiword_label(num_qubits: int, positions: dict[int, str]) -> str:
    chars = ["I"] * num_qubits
    for qubit, pauli in positions.items():
        chars[num_qubits - 1 - qubit] = pauli
    return "".join(chars)


def test_cmake_declares_metal_flag_and_rejects_mixed_accelerators() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert (
        '_wolfgang_bool_option(WOLFGANG_ENABLE_METAL FASTPAULI_ENABLE_METAL '
        '"Build Apple Metal backend support" OFF)'
    ) in cmake
    assert "WOLFGANG_ENABLE_CUDA and WOLFGANG_ENABLE_METAL cannot both be ON" in cmake
    assert "WOLFGANG_ENABLE_HIP and WOLFGANG_ENABLE_METAL cannot both be ON" in cmake

    cuda_mix = _run_cmake_configure_with_options(
        "-DWOLFGANG_ENABLE_CUDA=ON",
        "-DWOLFGANG_ENABLE_METAL=ON",
        "-DWOLFGANG_ENABLE_TBB=OFF",
    )
    assert cuda_mix.returncode != 0
    assert (
        "WOLFGANG_ENABLE_CUDA and WOLFGANG_ENABLE_METAL cannot both be ON"
        in cuda_mix.stdout + cuda_mix.stderr
    )

    hip_mix = _run_cmake_configure_with_options(
        "-DWOLFGANG_ENABLE_HIP=ON",
        "-DWOLFGANG_ENABLE_METAL=ON",
        "-DWOLFGANG_ENABLE_TBB=OFF",
    )
    assert hip_mix.returncode != 0
    assert (
        "WOLFGANG_ENABLE_HIP and WOLFGANG_ENABLE_METAL cannot both be ON"
        in hip_mix.stdout + hip_mix.stderr
    )


def test_private_metal_source_layout_exists_and_public_headers_stay_framework_free() -> None:
    required_paths = (
        "src/metal/accelerator_metal.mm",
        "src/metal/device_pauli_sum_metal.mm",
        "src/metal/device_commutation_matrix_metal.mm",
        "src/metal/commutation_metal.mm",
        "src/metal/workspace_metal.mm",
        "src/metal/kernels/commutation.metal",
        "src/metal/device_pauli_sum_metal.hpp",
        "src/metal/device_commutation_matrix_metal.hpp",
        "src/metal/workspace_metal.hpp",
    )
    missing = [path for path in required_paths if not (ROOT / path).exists()]
    assert not missing, "missing private Metal source files: " + ", ".join(missing)

    forbidden_tokens = (
        "<Metal/",
        "<Foundation/",
        "<MetalPerformanceShaders",
        "MTLDevice",
        "MTLBuffer",
        "MPSGraph",
    )
    offenders: list[str] = []
    for header in sorted((ROOT / "include" / "fastpauli").glob("*.hpp")):
        source = header.read_text(encoding="utf-8")
        if any(token in source for token in forbidden_tokens):
            offenders.append(str(header.relative_to(ROOT)))

    assert not offenders, "public headers must not expose Apple framework types: " + ", ".join(offenders)


def test_cpu_only_build_reports_metal_absence_or_metal_metadata_when_enabled() -> None:
    info = core._build_info()
    status = core._metal_status()
    accelerator_status_fn = getattr(core, "_accelerator_status", None)
    accelerator_status = (
        accelerator_status_fn() if accelerator_status_fn is not None else {"metal": {"built": info["metal_enabled"]}}
    )

    assert "metal_enabled" in info
    assert "metal_kernels" in info
    assert "metal_capability_summary" in info
    assert "capability_summary" in status
    assert "metal" in accelerator_status

    if info["metal_enabled"]:
        assert info["accelerator_build_mode"] == "metal_only"
        assert status["built"] is True
        assert "metal" in info["compiled_accelerator_backends"]
        assert "metal" in info["compiled_backends"]
        assert "commutes_with" in info["metal_kernels"]
        assert accelerator_status["metal"]["built"] is True
        assert accelerator_status["active_backend"] in {"metal", "none"}
        if status["runtime_available"]:
            assert status["capability_summary"]
            assert info["metal_capability_summary"] == status["capability_summary"]
        return

    assert info["accelerator_build_mode"] in {"cpu_only", "cuda_only", "hip_only"}
    assert status["built"] is False
    assert status["runtime_available"] is False
    assert status["device_count"] == 0
    assert "built without Metal" in status["skip_reason"]
    assert info["metal_kernels"] == []
    assert accelerator_status["metal"]["built"] is False


def test_non_metal_build_reports_metal_absence_for_any_other_backend() -> None:
    info = core._build_info()
    status = core._metal_status()

    if info["metal_enabled"]:
        pytest.skip("Metal source build is active")

    assert info["accelerator_build_mode"] in {"cpu_only", "cuda_only", "hip_only"}
    assert status["built"] is False
    assert status["runtime_available"] is False
    assert status["device_count"] == 0
    assert "built without Metal" in status["skip_reason"]
    assert info["metal_kernels"] == []


def test_backend_selector_policy_includes_metal_without_mixed_hardware() -> None:
    select = getattr(core, "_accelerator_backend_selection_for_testing", None)
    if select is None:
        pytest.skip("internal accelerator backend selector test hook is unavailable in this build")

    assert select(None, False, False, False, False, True, False) == "metal"
    assert select("auto", False, False, False, False, True, True) == "metal"
    assert select("metal", False, False, False, False, True, True) == "metal"

    with pytest.raises(RuntimeError, match="built without Metal support"):
        select("metal", True, True, False, False, False, False)
    with pytest.raises(RuntimeError, match='require backend="cuda", backend="hip", or backend="metal"'):
        select("auto", True, True, False, False, True, True)
    with pytest.raises(ValueError, match="backend must be None, 'auto', 'cuda', 'hip', or 'metal'"):
        select("bogus", True, True, False, False, True, True)


def test_to_device_rejects_metal_when_not_compiled() -> None:
    status = core._metal_status()
    if status["built"]:
        pytest.skip("Metal source build is active")

    op = fastpauli.PauliSum.from_labels(["X"], [1.0])
    with pytest.raises(RuntimeError, match="built without Metal support"):
        op.to_device(backend="metal")
    with pytest.raises(RuntimeError, match="built without Metal support"):
        fastpauli.DeviceCommutationMatrix.empty((1, 1), backend="metal")


def test_metal_benchmark_output_option_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "metal_benchmark_smoke.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks" / "bench_metal_kernels.py"),
            "--smoke",
            "--repeat",
            "1",
            "--json",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(completed.stdout)


@pytest.mark.parametrize(
    ("labels", "coeffs"),
    [
        ([], []),
        (["XIZ", "YYI"], [1.0, -2.0j]),
        (
            [
                _multiword_label(130, {0: "X", 65: "Z", 129: "Y"}),
                _multiword_label(130, {2: "Z", 66: "X", 100: "Y"}),
            ],
            [3.0 - 1.0j, -0.25],
        ),
        (["XYZ", "XYZ", "III", "XII"], [1.0, 2.0, 0.0, -4.0j]),
    ],
)
def test_metal_transfer_round_trip_when_available(labels: list[str], coeffs: list[complex]) -> None:
    _require_metal_runtime()

    if labels:
        host = fastpauli.PauliSum.from_labels(labels, coeffs)
    else:
        host = fastpauli.PauliSum.empty(num_qubits=7)

    device_op = host.to_device(backend="metal")

    assert device_op.backend == "metal"
    assert device_op.device == 0
    assert device_op.num_qubits == host.num_qubits
    assert device_op.num_terms == host.num_terms
    _assert_same_operator(device_op.to_host(), host)


def test_metal_commutation_and_compact_consumers_match_cpu_when_available() -> None:
    _require_metal_runtime()

    lhs = fastpauli.PauliSum.from_labels(
        [
            _multiword_label(130, {0: "Y", 2: "X"}),
            _multiword_label(130, {1: "Z", 2: "Z"}),
            _multiword_label(130, {}),
            _multiword_label(130, {0: "X", 64: "Y", 129: "Z"}),
        ],
        [1.0, -2.0j, 0.5, 1.25],
    )
    rhs = fastpauli.PauliSum.from_labels(
        [
            _multiword_label(130, {0: "X", 2: "Y"}),
            _multiword_label(130, {0: "Z", 1: "Z", 2: "Z"}),
            _multiword_label(130, {1: "Z", 64: "X", 100: "Y"}),
        ],
        [2.0, 3.0j, -1.0],
    )
    lhs_device = lhs.to_device(backend="metal")
    rhs_device = rhs.to_device(backend="metal")
    expected = np.asarray(lhs.commutes_with(rhs), dtype=np.bool_)

    np.testing.assert_array_equal(lhs_device.commutes_with(rhs_device), expected)

    matrix = lhs_device.commutes_with_device(rhs_device)
    assert matrix.backend == "metal"
    assert matrix.device == 0
    np.testing.assert_array_equal(matrix.to_host(), expected)
    assert matrix.count_commuting() == int(expected.sum())
    np.testing.assert_array_equal(matrix.count_commuting(axis=0), expected.sum(axis=0))
    np.testing.assert_array_equal(matrix.count_commuting(axis=1), expected.sum(axis=1))
    np.testing.assert_array_equal(matrix.conflict_degrees(axis=0), (~expected).sum(axis=0))
    np.testing.assert_array_equal(matrix.conflict_degrees(axis=1), (~expected).sum(axis=1))

    output = fastpauli.DeviceCommutationMatrix.empty(expected.shape, backend="metal")
    lhs_device.commutes_with_device(rhs_device, output=output)
    np.testing.assert_array_equal(output.to_host(), expected)

    with pytest.raises(ValueError, match="max_commutation_matrix_entries"):
        lhs_device.commutes_with_device(rhs_device, max_commutation_matrix_entries=1)


@pytest.mark.parametrize(
    ("selector", "num_qubits"),
    [
        ("words1", 64),
        ("words2", 96),
        ("generic_2d", 130),
        ("flat_generic", 130),
    ],
)
def test_metal_forced_commutation_selectors_match_cpu_when_available(
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
    num_qubits: int,
) -> None:
    _require_metal_runtime()

    high_qubit = num_qubits - 1
    lhs = fastpauli.PauliSum.from_labels(
        [
            _multiword_label(num_qubits, {0: "X"}),
            _multiword_label(num_qubits, {high_qubit: "Z"}),
        ],
        [1.0, -0.5j],
    )
    rhs = fastpauli.PauliSum.from_labels(
        [
            _multiword_label(num_qubits, {0: "Z"}),
            _multiword_label(num_qubits, {high_qubit: "X"}),
            _multiword_label(num_qubits, {}),
        ],
        [2.0, 1.0j, -1.0],
    )
    expected = np.asarray(lhs.commutes_with(rhs), dtype=np.bool_)
    assert np.any(expected)
    assert np.any(~expected)
    lhs_device = lhs.to_device(backend="metal")
    rhs_device = rhs.to_device(backend="metal")

    monkeypatch.setenv("FASTPAULI_EXPERIMENTAL_METAL_COMMUTATION_KERNEL", selector)
    np.testing.assert_array_equal(lhs_device.commutes_with(rhs_device), expected)

    output = fastpauli.DeviceCommutationMatrix.empty(expected.shape, backend="metal")
    core._copy_device_commutation_matrix_from_host_for_testing(
        output,
        np.ascontiguousarray(~expected, dtype=np.bool_),
    )
    result = lhs_device.commutes_with_device(rhs_device, output=output)
    assert result is output
    np.testing.assert_array_equal(output.to_host(), expected)


def test_metal_device_matrix_interop_exports_remain_unavailable_when_available() -> None:
    _require_metal_runtime()

    lhs = fastpauli.PauliSum.from_labels(["X"], [1.0]).to_device(backend="metal")
    rhs = fastpauli.PauliSum.from_labels(["Z"], [1.0]).to_device(backend="metal")
    matrix = lhs.commutes_with_device(rhs)

    with pytest.raises(RuntimeError, match="Metal"):
        matrix.__cuda_array_interface__
    with pytest.raises(RuntimeError, match="Metal"):
        matrix.__dlpack__(max_version=(1, 0))
    with pytest.raises(RuntimeError, match="Metal"):
        matrix.__dlpack_device__()

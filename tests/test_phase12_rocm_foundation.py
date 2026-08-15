"""ROCm/HIP backend foundation behavior."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import fastpauli
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _labels_and_coeffs(op: fastpauli.PauliSum) -> tuple[list[str], list[complex]]:
    labels, coeffs = op.to_labels()
    return list(labels), [complex(value) for value in coeffs]


def _assert_same_operator(lhs: fastpauli.PauliSum, rhs: fastpauli.PauliSum) -> None:
    lhs_labels, lhs_coeffs = _labels_and_coeffs(lhs)
    rhs_labels, rhs_coeffs = _labels_and_coeffs(rhs)
    assert lhs_labels == rhs_labels
    np.testing.assert_allclose(lhs_coeffs, rhs_coeffs, rtol=1.0e-12, atol=1.0e-12)


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


def run_cmake_configure_with_options(*options: str) -> subprocess.CompletedProcess[str]:
    build_dir_name = (
        "pytest-rocm-"
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
            *options,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_cpu_only_build_reports_hip_absence() -> None:
    import fastpauli._fastpauli_core as core

    info = core._build_info()
    status = core._hip_status()
    accelerator_status = core._accelerator_status()

    if info["hip_enabled"]:
        assert info["hip_architectures"] != "not_available"
        assert info["rocm_toolkit_version"] != "not_available"
        assert {
            "simplify",
            "expectation_statevector",
            "commutes_with",
            "commutes_with_device",
            "commutation_count_consumers",
            "matmul",
        } <= set(info["hip_kernels"])
        assert status["built"] is True
        assert status["runtime_available"] == info["hip_runtime_available"]
        assert accelerator_status["hip"]["built"] is True
        assert accelerator_status["cuda"]["built"] == info["cuda_enabled"]
        assert accelerator_status["active_backend"] in {"hip", "none"}
        return

    assert info["hip_enabled"] is False
    assert info["hip_architectures"] == "not_available"
    assert info["rocm_toolkit_version"] == "not_available"
    assert info["hip_runtime_available"] is False
    assert info["hip_kernels"] == []

    assert status["built"] is False
    assert status["runtime_available"] is False
    assert status["device_count"] == 0
    assert "built without HIP" in status["skip_reason"]

    if accelerator_status["cuda"]["runtime_available"]:
        assert accelerator_status["active_backend"] == "cuda"
    elif accelerator_status["metal"]["runtime_available"]:
        assert accelerator_status["active_backend"] == "metal"
    else:
        assert accelerator_status["active_backend"] == "none"
    assert accelerator_status["cuda"]["built"] == info["cuda_enabled"]
    assert accelerator_status["hip"]["built"] is False


def test_cmake_rejects_cuda_and_hip_together() -> None:
    completed = run_cmake_configure_with_options(
        "-DWOLFGANG_ENABLE_CUDA=ON",
        "-DWOLFGANG_ENABLE_HIP=ON",
        "-DWOLFGANG_ENABLE_TBB=OFF",
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert "WOLFGANG_ENABLE_CUDA and WOLFGANG_ENABLE_HIP cannot both be ON" in output
    assert "target-specific accelerator build policy" in output


def test_public_headers_do_not_include_rocm_or_hip_headers() -> None:
    forbidden_tokens = ("hip/", "hip_runtime", "hsa/", "rocm/")
    offenders: list[str] = []

    for header in sorted((ROOT / "include" / "fastpauli").glob("*.hpp")):
        source = header.read_text(encoding="utf-8").lower()
        if any(token in source for token in forbidden_tokens):
            offenders.append(str(header.relative_to(ROOT)))

    assert not offenders, "public headers must not include ROCm/HIP headers: " + ", ".join(offenders)


def test_cmake_hip_compiler_discovery_uses_rocm_clang_not_hipcc_wrapper() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert 'set(CMAKE_HIP_COMPILER "/opt/rocm/bin/hipcc"' not in cmake
    assert "/opt/rocm/bin/amdclang++" in cmake or "/opt/rocm/llvm/bin/clang++" in cmake


def test_hip_round_trip_when_available() -> None:
    import fastpauli._fastpauli_core as core

    status = core._hip_status()
    if not status["runtime_available"]:
        pytest.skip(status["skip_reason"])

    op = fastpauli.PauliSum.from_labels(["XIZ", "YYI"], [1.0, -2.0j])
    device_op = op.to_device(device=0)

    assert device_op.backend == "hip"
    assert device_op.device == 0
    assert _labels_and_coeffs(device_op.to_host()) == _labels_and_coeffs(op)


def test_hip_empty_round_trip_when_available() -> None:
    import fastpauli._fastpauli_core as core

    status = core._hip_status()
    if not status["runtime_available"]:
        pytest.skip(status["skip_reason"])

    host = fastpauli.PauliSum.empty(num_qubits=5)
    actual = host.to_device(device=0).to_host()

    assert actual.num_terms == 0
    assert actual.num_qubits == 5


def test_hip_invalid_device_error_when_available() -> None:
    import fastpauli._fastpauli_core as core

    status = core._hip_status()
    if not status["runtime_available"]:
        pytest.skip(status["skip_reason"])

    op = fastpauli.PauliSum.from_labels(["XI"], [1.0])
    with pytest.raises(ValueError, match="HIP device ordinal is out of range"):
        op.to_device(device=status["device_count"])


def _require_hip_runtime() -> dict:
    import fastpauli._fastpauli_core as core

    status = core._hip_status()
    if not status["runtime_available"]:
        pytest.skip(status["skip_reason"])
    return status


def _assert_hip_simplify_matches_cpu(
    op: fastpauli.PauliSum,
    *,
    atol: float = 1.0e-12,
    rtol: float = 0.0,
) -> None:
    expected = op.simplify(atol=atol, rtol=rtol)
    actual = op.to_device().simplify(atol=atol, rtol=rtol).to_host()

    actual_labels, actual_coeffs = actual.to_labels()
    expected_labels, expected_coeffs = expected.to_labels()
    assert list(actual_labels) == list(expected_labels)
    np.testing.assert_allclose(
        np.asarray(actual_coeffs, dtype=np.complex128),
        np.asarray(expected_coeffs, dtype=np.complex128),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def _assert_hip_commutation_matches_cpu(
    lhs: fastpauli.PauliSum,
    rhs: fastpauli.PauliSum,
) -> None:
    lhs_device = lhs.to_device()
    rhs_device = rhs.to_device()
    expected = np.asarray(lhs.commutes_with(rhs), dtype=np.bool_).reshape(
        lhs.num_terms,
        rhs.num_terms,
    )
    actual = np.asarray(lhs_device.commutes_with(rhs_device), dtype=np.bool_).reshape(
        lhs.num_terms,
        rhs.num_terms,
    )
    np.testing.assert_array_equal(actual, expected)


def _assert_hip_device_commutation_matrix_matches_cpu(
    lhs: fastpauli.PauliSum,
    rhs: fastpauli.PauliSum,
) -> None:
    lhs_device = lhs.to_device()
    rhs_device = rhs.to_device()
    expected = np.asarray(lhs.commutes_with(rhs), dtype=np.bool_).reshape(
        lhs.num_terms,
        rhs.num_terms,
    )

    matrix = lhs_device.commutes_with_device(rhs_device)

    assert isinstance(matrix, fastpauli.DeviceCommutationMatrix)
    assert matrix.shape == (lhs.num_terms, rhs.num_terms)
    assert matrix.device == lhs_device.device
    assert matrix.dtype == "uint8"
    assert matrix.num_entries == lhs.num_terms * rhs.num_terms
    np.testing.assert_array_equal(matrix.to_host(), expected)


def test_hip_commutation_matches_cpu_for_edge_cases_when_available() -> None:
    _require_hip_runtime()

    cases = [
        (
            fastpauli.PauliSum.empty(num_qubits=3),
            fastpauli.PauliSum.from_labels(["XII", "IYZ"], [1.0, -1.0]),
        ),
        (
            fastpauli.PauliSum.from_labels(["XII", "ZZZ"], [1.0, 2.0]),
            fastpauli.PauliSum.empty(num_qubits=3),
        ),
        (
            fastpauli.PauliSum.from_labels(["X", "Z"], [1.0, 1.0]),
            fastpauli.PauliSum.from_labels(["Y", "I"], [1.0, 1.0]),
        ),
        (
            fastpauli.PauliSum.from_sparse_list(
                [
                    ("X", [0], 1.0),
                    ("Z", [64], -0.5),
                    ("YZ", [1, 64], 0.25j),
                ],
                num_qubits=65,
            ),
            fastpauli.PauliSum.from_sparse_list(
                [
                    ("Y", [0], -1.0),
                    ("X", [64], 2.0),
                    ("XZ", [0, 64], 0.75),
                ],
                num_qubits=65,
            ),
        ),
    ]

    for lhs, rhs in cases:
        _assert_hip_commutation_matches_cpu(lhs, rhs)


def test_hip_commutation_matches_cpu_for_random_inputs_when_available() -> None:
    _require_hip_runtime()

    rng = np.random.default_rng(942)
    alphabet = np.asarray(["I", "X", "Y", "Z"])
    for num_qubits, lhs_terms, rhs_terms in [(5, 7, 6), (70, 4, 5)]:
        lhs_labels = [
            "".join(rng.choice(alphabet, size=num_qubits).tolist())
            for _ in range(lhs_terms)
        ]
        rhs_labels = [
            "".join(rng.choice(alphabet, size=num_qubits).tolist())
            for _ in range(rhs_terms)
        ]
        lhs = fastpauli.PauliSum.from_labels(lhs_labels, np.ones(lhs_terms).tolist())
        rhs = fastpauli.PauliSum.from_labels(rhs_labels, np.ones(rhs_terms).tolist())
        _assert_hip_commutation_matches_cpu(lhs, rhs)


def test_hip_commutation_guardrails_and_reused_output_when_available() -> None:
    _require_hip_runtime()

    lhs = fastpauli.PauliSum.from_labels(["XX", "ZI", "IZ"], [1.0, 2.0, -1.0])
    rhs = fastpauli.PauliSum.from_labels(["YY", "XI"], [1.0, 1.0j])
    lhs_device = lhs.to_device()
    rhs_device = rhs.to_device()

    output = np.empty(lhs.num_terms * rhs.num_terms, dtype=np.bool_)
    assert lhs_device.commutes_with_into(rhs_device, output) is None
    np.testing.assert_array_equal(
        output.reshape(lhs.num_terms, rhs.num_terms),
        lhs.commutes_with(rhs),
    )

    with pytest.raises(ValueError, match="output buffer size"):
        lhs_device.commutes_with_into(rhs_device, output[:-1])
    with pytest.raises(ValueError, match="same num_qubits"):
        lhs_device.commutes_with(fastpauli.PauliSum.from_labels(["XXX"], [1.0]).to_device())
    with pytest.raises(ValueError, match="commutation matrix entry count exceeds"):
        lhs_device.commutes_with(rhs_device, max_commutation_matrix_entries=3)


def test_hip_commutation_rejects_different_devices_when_available() -> None:
    status = _require_hip_runtime()
    if int(status["device_count"]) < 2:
        pytest.skip("different-device HIP check requires at least two visible HIP devices")

    lhs = fastpauli.PauliSum.from_labels(["XX"], [1.0]).to_device(device=0)
    rhs = fastpauli.PauliSum.from_labels(["YY"], [1.0]).to_device(device=1)
    with pytest.raises(ValueError, match="same device"):
        lhs.commutes_with(rhs)


def test_hip_device_commutation_matrix_matches_cpu_when_available() -> None:
    _require_hip_runtime()

    rng = np.random.default_rng(943)
    alphabet = np.asarray(["I", "X", "Y", "Z"])
    randomized_lhs = [
        "".join(rng.choice(alphabet, size=70).tolist())
        for _ in range(4)
    ]
    randomized_rhs = [
        "".join(rng.choice(alphabet, size=70).tolist())
        for _ in range(5)
    ]

    cases = [
        (
            fastpauli.PauliSum.empty(num_qubits=3),
            fastpauli.PauliSum.from_labels(["XII", "IYZ"], [1.0, -1.0]),
        ),
        (
            fastpauli.PauliSum.from_labels(["XII", "ZZZ"], [1.0, 2.0]),
            fastpauli.PauliSum.empty(num_qubits=3),
        ),
        (
            fastpauli.PauliSum.from_labels(["X"], [1.0]),
            fastpauli.PauliSum.from_labels(["Z"], [1.0j]),
        ),
        (
            fastpauli.PauliSum.from_labels(["XX"], [1.0]),
            fastpauli.PauliSum.from_labels(["YY", "XI", "IZ"], [1.0, 1.0j, -0.5]),
        ),
        (
            fastpauli.PauliSum.from_labels(["XI", "ZI"], [1.0, 2.0]),
            fastpauli.PauliSum.from_labels(["ZZ"], [1.0]),
        ),
        (
            fastpauli.PauliSum.from_labels(["XX", "ZI", "IZ"], [1.0, 2.0, -1.0]),
            fastpauli.PauliSum.from_labels(["YY", "XI"], [1.0, 1.0j]),
        ),
        (
            fastpauli.PauliSum.from_sparse_list(
                [
                    ("X", [0], 1.0),
                    ("Z", [64], -0.5),
                    ("YZ", [1, 64], 0.25j),
                ],
                num_qubits=65,
            ),
            fastpauli.PauliSum.from_sparse_list(
                [
                    ("Y", [0], -1.0),
                    ("X", [64], 2.0),
                    ("XZ", [0, 64], 0.75),
                ],
                num_qubits=65,
            ),
        ),
        (
            fastpauli.PauliSum.from_labels(randomized_lhs, np.ones(4).tolist()),
            fastpauli.PauliSum.from_labels(randomized_rhs, np.ones(5).tolist()),
        ),
    ]

    for lhs, rhs in cases:
        _assert_hip_device_commutation_matrix_matches_cpu(lhs, rhs)


def test_hip_device_commutation_matrix_counts_match_numpy_when_available() -> None:
    _require_hip_runtime()

    lhs = fastpauli.PauliSum.from_labels(["XI", "ZI"], [1.0, 1.0]).to_device()
    rhs = fastpauli.PauliSum.from_labels(["IX", "ZZ", "XX"], [1.0, 1.0, 1.0]).to_device()
    matrix = lhs.commutes_with_device(rhs)
    host = matrix.to_host().astype(np.uint64)

    assert matrix.count_commuting() == int(host.sum())
    np.testing.assert_array_equal(
        matrix.count_commuting(axis=1),
        host.sum(axis=1, dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        matrix.count_commuting(axis=0),
        host.sum(axis=0, dtype=np.uint64),
    )

    host_conflicts = np.logical_not(host.astype(np.bool_)).astype(np.uint64)
    assert matrix.conflict_degrees() == int(host_conflicts.sum())
    np.testing.assert_array_equal(
        matrix.conflict_degrees(axis=1),
        host_conflicts.sum(axis=1, dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        matrix.conflict_degrees(axis=0),
        host_conflicts.sum(axis=0, dtype=np.uint64),
    )


def test_hip_device_commutation_matrix_reuse_and_guardrails_when_available() -> None:
    status = _require_hip_runtime()

    lhs = fastpauli.PauliSum.from_labels(["XI", "ZI"], [1.0, 1.0]).to_device()
    rhs = fastpauli.PauliSum.from_labels(["IX", "ZZ", "XX"], [1.0, 1.0, 1.0]).to_device()
    output = fastpauli.DeviceCommutationMatrix.empty(
        (lhs.num_terms, rhs.num_terms),
        device=lhs.device,
    )

    same = lhs.commutes_with_device(rhs, output=output)

    assert same is output
    np.testing.assert_array_equal(output.to_host(), lhs.to_host().commutes_with(rhs.to_host()))

    wrong_shape = fastpauli.DeviceCommutationMatrix.empty((1, rhs.num_terms), device=lhs.device)
    with pytest.raises(ValueError, match="output shape"):
        lhs.commutes_with_device(rhs, output=wrong_shape)

    with pytest.raises(ValueError, match="commutation matrix entry count exceeds"):
        lhs.commutes_with_device(rhs, max_commutation_matrix_entries=3)

    if status["device_count"] >= 2:
        wrong_device = fastpauli.DeviceCommutationMatrix.empty(
            (lhs.num_terms, rhs.num_terms),
            device=1 if lhs.device == 0 else 0,
        )
        with pytest.raises(ValueError, match="same device"):
            lhs.commutes_with_device(rhs, output=wrong_device)


def test_hip_cuda_array_interface_remains_unavailable_when_available() -> None:
    _require_hip_runtime()

    matrix = fastpauli.PauliSum.from_labels(["XI"], [1.0]).to_device().commutes_with_device(
        fastpauli.PauliSum.from_labels(["IX"], [1.0]).to_device(),
    )

    with pytest.raises((BufferError, RuntimeError, ValueError), match="HIP|ROCm|CUDA"):
        matrix.__cuda_array_interface__


def test_hip_dlpack_surfaces_remain_unavailable_when_available() -> None:
    _require_hip_runtime()

    lhs = fastpauli.PauliSum.from_labels(["XI", "ZI"], [1.0, 1.0]).to_device()
    rhs = fastpauli.PauliSum.from_labels(["IX", "ZZ", "XX"], [1.0, 1.0, 1.0]).to_device()
    matrix = lhs.commutes_with_device(rhs)

    with pytest.raises((BufferError, RuntimeError, ValueError), match="HIP|ROCm|DLPack"):
        matrix.__dlpack__(max_version=(1, 0))
    with pytest.raises((BufferError, RuntimeError, ValueError), match="HIP|ROCm|DLPack"):
        matrix.__dlpack_device__()


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
        labels = [
            "".join(rng.choice(alphabet, size=num_qubits).tolist())
            for _ in range(terms)
        ]
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
    with pytest.raises(ValueError, match=r"len\(psi\) == 2 \*\* num_qubits"):
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


def test_hip_matmul_multiword_and_empty_cases_when_available() -> None:
    _require_hip_runtime()

    cases = [
        (
            fastpauli.PauliSum.empty(70),
            fastpauli.PauliSum.from_sparse_list([("X", [64], 1.0)], num_qubits=70),
        ),
        (
            fastpauli.PauliSum.from_sparse_list(
                [("XY", [0, 70], 1.0j), ("Z", [69], -2.0)],
                num_qubits=72,
            ),
            fastpauli.PauliSum.from_sparse_list(
                [("YZ", [1, 70], -0.5), ("X", [69], 3.0)],
                num_qubits=72,
            ),
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



def test_hip_simplify_campaign4_generic_multiword_pressure_when_available() -> None:
    _require_hip_runtime()

    rng = np.random.default_rng(40404)
    labels = []
    coeffs = []
    alphabet = np.asarray(["I", "X", "Y", "Z"])
    for _ in range(96):
        labels.append("".join(rng.choice(alphabet, size=193).tolist()))
        coeffs.append(complex(float(rng.normal()), float(rng.normal())))
    labels.extend(labels[:32])
    coeffs.extend([-value for value in coeffs[:16]])
    coeffs.extend(coeffs[16:32])

    op = fastpauli.PauliSum.from_labels(labels, coeffs)
    _assert_hip_simplify_matches_cpu(op, atol=1.0e-11, rtol=1.0e-12)


def test_hip_simplify_campaign4_one_and_two_word_regression_when_available() -> None:
    _require_hip_runtime()

    cases = [
        fastpauli.PauliSum.from_labels(["X" * 24, "X" * 24, "Z" * 24], [1.0, 2.0, -3.0]),
        fastpauli.PauliSum.from_sparse_list(
            [("XY", [0, 70], 1.0), ("XY", [0, 70], -0.25), ("Z", [69], 2.0)],
            num_qubits=72,
        ),
    ]
    for op in cases:
        _assert_hip_simplify_matches_cpu(op, atol=0.0, rtol=0.0)


def test_rocm_kernel_benchmark_smoke_reports_protocol_fields() -> None:
    script = ROOT / "benchmarks" / "bench_rocm_kernels.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--smoke", "--repeat", "1", "--warmup", "0", "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["benchmark"] == "rocm_kernels"
    assert report["profile"] == "smoke"
    assert "HIP" in report["environment"]
    assert report["cases"]
    row = report["cases"][0]
    assert row["operation"] == "commutes_with"
    assert row["dataset"]["lhs_terms"] == 128
    assert row["dataset"]["rhs_terms"] == 128
    assert "cpu_scalar_seconds" in row
    assert "available_cpu_selector_seconds" in row
    if report["hip_status"]["runtime_available"]:
        assert row["status"] == "ok"
        assert row["correctness_passed"] is True
        assert row["transfer_inclusive_seconds"] is not None
        assert row["device_resident_seconds"] is not None
    else:
        assert row["status"] == "hip_unavailable"
        assert row["unavailable_reason"]


def test_rocm_simplify_benchmark_smoke_reports_campaign3_fields() -> None:
    script = ROOT / "benchmarks" / "bench_rocm_kernels.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--profile",
            "simplify-smoke",
            "--repeat",
            "1",
            "--warmup",
            "0",
            "--json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["benchmark"] == "rocm_kernels"
    assert report["profile"] == "simplify-smoke"
    row = report["cases"][0]
    assert row["operation"] == "simplify"
    assert row["backend"] == "hip"
    assert row["dataset"]["num_terms"] == 128
    assert row["hip_simplify_strategy"] in {
        "rocthrust_default",
        "hipcub_radix_sort_reduce",
        "custom_packed_key",
        "unavailable",
    }
    assert "hip_simplify_transfer_seconds" in row
    assert "hip_simplify_device_resident_seconds" in row
    assert "hip_simplify_to_host_seconds" in row
    assert row["result_materialization_target"] == "device_pauli_sum"
    assert row["timing_boundary"] == "device_resident"
    assert {
        "DLPack",
        "streams",
        "workspaces",
        "packed summaries",
        "expectation",
        "matmul",
        "portability",
        "ROCm wheels",
        "multi-GPU",
        "simultaneous CUDA+HIP",
    } <= set(row["campaign3_headroom_statuses"])


def test_rocm_simplify_benchmark_smoke_reports_campaign4_fields() -> None:
    script = ROOT / "benchmarks" / "bench_rocm_kernels.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--profile",
            "simplify-campaign4-baseline",
            "--repeat",
            "1",
            "--warmup",
            "0",
            "--json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["benchmark"] == "rocm_kernels"
    assert report["profile"] == "simplify-campaign4-baseline"
    rows = report["cases"]
    assert rows
    row = rows[0]
    assert row["operation"] == "simplify"
    assert row["campaign"] == "rocm_mi300x_campaign4"
    assert row["hip_simplify_strategy"] in {
        "rocthrust_default",
        "rocthrust_generic_parallel_reduce_by_key",
        "custom_packed_key",
        "rocprim_scratch_probe",
        "hipcub_scratch_probe",
        "unavailable",
    }
    assert row["hip_simplify_strategy_status"] in {
        "retained",
        "rejected_with_evidence",
        "benchmark_only",
        "unavailable",
        "blocked_external",
    }
    assert row["hip_simplify_strategy_reason"]
    assert row["hip_simplify_key_shape"] in {
        "empty",
        "identity",
        "packed32",
        "key1",
        "key2",
        "generic_multiword",
    }
    assert row["hip_workspace_mode"] in {
        "absent",
        "grow_inside_timing",
        "pre_reserved_outside_timing",
        "benchmark_only",
        "unavailable",
    }
    assert isinstance(row["hip_workspace_reserved_bytes"], int)
    assert isinstance(row["hip_workspace_high_watermark_bytes"], int)
    assert isinstance(row["hip_workspace_allocation_count"], int)
    assert isinstance(row["hip_workspace_growth_count"], int)
    assert row["generic_multiword_parallelism"] in {
        "serial_kernel",
        "reduce_by_key",
        "segmented_reduce",
        "not_applicable",
    }
    assert {
        "workspace",
        "custom packed key",
        "generic multi-word",
        "DLPack",
        "streams",
        "expectation",
        "matmul",
        "portability",
        "ROCm wheels",
        "multi-GPU",
        "simultaneous CUDA+HIP",
    } <= set(row["campaign4_terminal_statuses"])


def test_rocm_campaign5_interop_benchmark_reports_terminal_fields() -> None:
    script = ROOT / "benchmarks" / "bench_rocm_kernels.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--profile",
            "interop-campaign5-dlpack-consumers",
            "--repeat",
            "1",
            "--warmup",
            "0",
            "--json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["benchmark"] == "rocm_kernels"
    assert report["profile"] == "interop-campaign5-dlpack-consumers"
    rows = report["cases"]
    assert rows
    modes = {row["mode"] for row in rows}
    assert {"dlpack_pytorch", "dlpack_cupy", "cuda_array_interface_guard"} <= modes
    for row in rows:
        assert row["campaign"] == "rocm_mi300x_campaign5"
        assert row["final_status"] in {
            "rejected_with_evidence",
            "blocked_external",
            "unavailable",
            "out_of_scope_with_next_trigger",
        }
        assert row["timing_boundary"] in {
            "decision_only",
            "framework_consumer",
            "compact_consumer",
            "device_output_to_host",
            "benchmark_only",
        }
        assert {
            "DLPack",
            "CUDA Array Interface guard",
            "streams",
            "graphs",
            "workspaces",
            "expectation",
            "matmul",
            "portability",
            "ROCm wheels",
            "multi-GPU",
            "simultaneous CUDA+HIP",
        } <= set(row["campaign5_terminal_statuses"])
        if str(row["mode"]).startswith("dlpack_"):
            assert row["candidate_probe_evidence_kind"] == "not_run_in_retained_build"
            assert row["candidate_probe_source_file"] is None
            assert row["candidate_probe_consumer_correctness_passed"] is None
            assert row["candidate_probe_consumer_read_only_enforced"] is None
            assert row["candidate_probe_mutation_result"] == "not_run"


def test_rocm_campaign6_benchmark_reports_parity_fields() -> None:
    script = ROOT / "benchmarks" / "bench_rocm_kernels.py"
    for profile, operation in [
        ("campaign6-expectation-parity", "expectation_statevector"),
        ("campaign6-matmul-parity", "matmul"),
    ]:
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--profile",
                profile,
                "--repeat",
                "1",
                "--warmup",
                "0",
                "--json",
            ],
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

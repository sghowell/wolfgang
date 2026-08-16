"""Phase 9 CPU backend dispatch and benchmark metadata tests."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import wolfgang_quantum._wolfgang_core as core
from wolfgang_quantum import PauliSum

ROOT = Path(__file__).resolve().parents[1]


def run_backend_probe(selector: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["WOLFGANG_CPU_BACKEND"] = selector
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "import wolfgang_quantum._wolfgang_core as core; "
                "print(json.dumps(core._build_info(), sort_keys=True))"
            ),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def run_backend_expectation_probe(selector: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["WOLFGANG_CPU_BACKEND"] = selector
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "import numpy as np; "
                "from wolfgang_quantum import PauliSum; "
                "op = PauliSum.from_labels(['ZI', 'XX'], [1.25, -0.5]); "
                "psi = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128); "
                "value = op.expectation_statevector(psi); "
                "print(json.dumps({'real': float(np.real(value)), 'imag': float(np.imag(value))}))"
            ),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def run_backend_commutation_probe(selector: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["WOLFGANG_CPU_BACKEND"] = selector
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "import numpy as np; "
                "from wolfgang_quantum import PauliSum; "
                "lhs = PauliSum.from_labels(['XI', 'YZ', 'ZZ', 'IX'], [1, 2, 3, 4]); "
                "rhs = PauliSum.from_labels(['ZX', 'YY', 'IZ'], [1, 2, 3]); "
                "value = np.asarray(lhs.commutes_with(rhs), dtype=np.bool_).reshape(4, 3); "
                "print(json.dumps(value.astype(int).tolist()))"
            ),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def run_cmake_configure_with_options(*options: str) -> subprocess.CompletedProcess[str]:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake is not available")
    build_dir = ROOT / "_skbuild" / ("pytest-" + "-".join(options).replace("=", "-").replace("_", "-"))
    return subprocess.run(
        [
            cmake,
            "-S",
            str(ROOT),
            "-B",
            str(build_dir),
            "-DWOLFGANG_ENABLE_CUDA=OFF",
            "-DWOLFGANG_ENABLE_NATIVE=OFF",
            f"-DPython_EXECUTABLE={sys.executable}",
            *options,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_build_info_reports_cpu_backend_dispatch_surface() -> None:
    info = core._build_info()

    assert info["cpu_backend"] == "scalar"
    assert info["active_cpu_backend"] == "scalar"
    assert info["requested_cpu_backend"] == "auto"
    assert info["cpu_backend_env_var"] == "WOLFGANG_CPU_BACKEND"
    assert info["cpu_cmake_options"]["WOLFGANG_ENABLE_ARM_NEON"] == "auto"
    assert info["cpu_cmake_options"]["WOLFGANG_ENABLE_ARM_SVE"] == "auto"
    assert info["cpu_backend_build_flags"]["scalar"] is True
    assert "optimized_cpu_kernels" in info
    assert set(info["optimized_cpu_kernels"]).issuperset({"tbb", "avx2", "avx512", "neon", "sve"})
    assert info["compiler_build_config"]["CMAKE_CXX_COMPILER_ID"]
    assert info["cpu_auto_dispatch_thresholds"]["tbb_pairwise_entries"] == 331776
    assert "scalar" in info["compiled_cpu_backends"]
    assert "scalar" in info["available_cpu_backends"]
    candidate_statuses = {
        candidate["name"]: candidate["status"] for candidate in info["cpu_backend_candidates"]
    }
    assert candidate_statuses["avx2"] in {"available", "not_compiled", "hardware_unavailable"}
    assert candidate_statuses["avx512"] in {"available", "not_compiled", "hardware_unavailable"}
    assert any(
        candidate["name"] == "scalar" and candidate["status"] == "available"
        for candidate in info["cpu_backend_candidates"]
    )


def test_forced_architecture_simd_configure_is_implemented_on_capable_toolchains() -> None:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        completed = run_cmake_configure_with_options(
            "-DWOLFGANG_ENABLE_TBB=OFF",
            "-DWOLFGANG_ENABLE_ARM_NEON=ON",
        )
        assert completed.returncode == 0, completed.stderr
        assert "not implemented" not in completed.stderr
        return

    if machine in {"x86_64", "amd64"}:
        completed = run_cmake_configure_with_options(
            "-DWOLFGANG_ENABLE_TBB=OFF",
            "-DWOLFGANG_ENABLE_AVX2=ON",
        )
        assert completed.returncode == 0, completed.stderr
        assert "not implemented" not in completed.stderr
        return

    pytest.skip(f"no forced SIMD configure expectation for {machine}")


@pytest.mark.parametrize("selector", ["auto", "scalar"])
def test_fastpauli_cpu_backend_accepts_auto_and_scalar(selector: str) -> None:
    completed = run_backend_probe(selector)

    assert completed.returncode == 0, completed.stderr
    info = json.loads(completed.stdout)
    assert info["requested_cpu_backend"] == selector
    assert info["active_cpu_backend"] == "scalar"


def test_fastpauli_cpu_backend_rejects_invalid_selector() -> None:
    completed = run_backend_probe("bogus")

    assert completed.returncode != 0
    assert "WOLFGANG_CPU_BACKEND" in completed.stderr


def test_forced_optimized_backend_availability_matches_reported_status() -> None:
    candidates = {
        candidate["name"]: candidate["status"]
        for candidate in core._build_info()["cpu_backend_candidates"]
        if candidate["name"] != "scalar"
    }
    assert {"tbb", "avx2", "avx512", "neon", "sve"}.issubset(candidates)

    for selector, status in candidates.items():
        completed = run_backend_probe(selector)
        if status == "available":
            assert completed.returncode == 0, completed.stderr
            assert json.loads(completed.stdout)["active_cpu_backend"] == selector
        else:
            assert completed.returncode != 0
            assert f"WOLFGANG_CPU_BACKEND={selector}" in completed.stderr
            assert status in completed.stderr


def test_available_optimized_backend_commutation_matches_forced_scalar() -> None:
    scalar = run_backend_commutation_probe("scalar")
    assert scalar.returncode == 0, scalar.stderr
    scalar_value = json.loads(scalar.stdout)

    candidates = {
        candidate["name"]: candidate["status"]
        for candidate in core._build_info()["cpu_backend_candidates"]
        if candidate["name"] != "scalar"
    }
    available = [selector for selector, status in candidates.items() if status == "available"]
    if not available:
        pytest.skip("no optimized CPU backend candidates are compiled and available")

    for selector in available:
        completed = run_backend_commutation_probe(selector)
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout) == scalar_value


def test_forced_optimized_backend_rejects_scalar_only_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = next(
        (
            candidate["name"]
            for candidate in core._build_info()["cpu_backend_candidates"]
            if candidate["name"] != "scalar" and candidate["status"] == "available"
        ),
        None,
    )
    if backend is None:
        pytest.skip("no optimized CPU backend candidates are compiled and available")

    op = PauliSum.from_labels(["ZI", "XX"], [1.0, 2.0])
    psi = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    monkeypatch.setenv("WOLFGANG_CPU_BACKEND", backend)
    expected = f"WOLFGANG_CPU_BACKEND={backend}"

    with pytest.raises(RuntimeError, match=expected):
        op.simplify()
    with pytest.raises(RuntimeError, match=expected):
        op.matmul(op)
    with pytest.raises(RuntimeError, match=expected):
        op.expectation_statevector(psi)
    with pytest.raises(RuntimeError, match=expected):
        op.group_commuting(mode="qwc")


def test_forced_simd_backend_rejects_unsupported_commutation_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    available_simd = [
        candidate["name"]
        for candidate in core._build_info()["cpu_backend_candidates"]
        if candidate["name"] in {"avx2", "avx512", "neon"} and candidate["status"] == "available"
    ]
    if not available_simd:
        pytest.skip("no SIMD CPU backend candidates are compiled and available")

    op = PauliSum.from_labels(["X" + ("I" * 128)], [1.0])
    expected = "supports commutation kernels only for packed widths of 1 or 2"
    for selector in available_simd:
        monkeypatch.setenv("WOLFGANG_CPU_BACKEND", selector)
        with pytest.raises(RuntimeError, match=expected):
            op.commutes_with(op)
        with pytest.raises(RuntimeError, match=expected):
            op.group_commuting(mode="full")


def test_compute_methods_fail_clearly_when_forced_backend_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = next(
        (
            candidate["name"]
            for candidate in core._build_info()["cpu_backend_candidates"]
            if candidate["name"] != "scalar" and candidate["status"] != "available"
        ),
        None,
    )
    if backend is None:
        pytest.skip("all optimized CPU backend candidates are available")

    op = PauliSum.from_labels(["ZI"], [1.0])
    psi = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    monkeypatch.setenv("WOLFGANG_CPU_BACKEND", backend)

    with pytest.raises(RuntimeError, match=f"WOLFGANG_CPU_BACKEND={backend}"):
        op.expectation_statevector(psi)


def test_cpu_dispatch_benchmark_smoke_outputs_backend_metadata() -> None:
    script = ROOT / "benchmarks" / "bench_cpu_dispatch.py"
    assert script.exists()

    completed = subprocess.run(
        [sys.executable, str(script), "--smoke", "--repeat", "1", "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["benchmark"] == "cpu_dispatch"
    assert report["fastpauli_build_info"]["active_cpu_backend"] == "scalar"
    assert "scalar" in report["fastpauli_build_info"]["available_cpu_backends"]
    assert "WOLFGANG_ENABLE_AVX2" in report["environment"]["cpu_cmake_options"]
    assert "WOLFGANG_ENABLE_AVX512" in report["environment"]["cpu_cmake_options"]
    assert "WOLFGANG_ENABLE_ARM_NEON" in report["environment"]["cpu_cmake_options"]
    assert "WOLFGANG_ENABLE_ARM_SVE" in report["environment"]["cpu_cmake_options"]
    assert "compiler_cpu_flags" in report["environment"]
    assert "instruction_set_probe" in report["environment"]
    case_names = {case["name"] for case in report["cases"]}
    assert {
        "auto_statevector_expectation",
        "forced_scalar_statevector_expectation",
        "forced_scalar_pairwise_commutation",
        "auto_pairwise_commutation",
        "forced_scalar_full_grouping",
        "auto_full_grouping",
        "optimized_backend_availability",
    }.issubset(case_names)

    available_optimized = [
        candidate["name"]
        for candidate in report["fastpauli_build_info"]["cpu_backend_candidates"]
        if candidate["name"] != "scalar" and candidate["status"] == "available"
    ]
    for backend in available_optimized:
        assert f"forced_{backend}_pairwise_commutation" in case_names
        assert f"forced_{backend}_full_grouping" in case_names

    availability = next(case for case in report["cases"] if case["name"] == "optimized_backend_availability")
    statuses = {item["backend"]: item["status"] for item in availability["results"]["optimized_backends"]}
    assert statuses["avx2"] in {"available", "not_compiled", "hardware_unavailable"}
    assert statuses["avx512"] in {"available", "not_compiled", "hardware_unavailable"}

    for case in report["cases"]:
        assert "active_cpu_backend" in case["dataset"]
        if case["name"].endswith("_statevector_expectation"):
            assert "fastpauli_max_seconds" in case["results"]
        if case["name"].endswith("_pairwise_commutation"):
            assert case["results"]["matches_forced_scalar"] is True
        if case["name"].endswith("_full_grouping"):
            assert case["results"]["matches_forced_scalar"] is True

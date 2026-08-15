"""Competitive baseline benchmark orchestration tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def test_competitive_baseline_benchmark_smoke_reports_optional_libraries() -> None:
    script = ROOT / "benchmarks" / "bench_competitive_baselines.py"
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
    assert report["benchmark"] == "competitive_baselines"
    assert "competitor_correctness_checked" in report["correctness_checks"]
    assert report["correctness_checks"]["fastpauli_cases_executed"] is True
    assert {"qiskit", "openfermion", "cupy", "cuquantum", "cudaq", "qiskit_aer"}.issubset(
        report["competitors"]
    )
    assert {case["name"] for case in report["cases"]} == {
        "simplify",
        "multiply",
        "qiskit_grouping",
        "cuquantum_statevector_expectation",
        "cupy_commutation_consumer",
    }
    for case in report["cases"]:
        assert "fastpauli_scalar_seconds" in case["results"]
        assert "competitor_seconds" in case["results"]
        assert "competitor_available" in case["results"]
        assert "competitor_correctness_checked" in case["results"]
        if not case["results"]["competitor_available"]:
            assert case["results"]["competitor_correctness_checked"] is False

    multiply = next(case for case in report["cases"] if case["name"] == "multiply")
    assert "competitor_operand_semantics" in multiply["dataset"]
    assert "competitor_intermediate_terms" in multiply["dataset"]

    cuquantum = next(
        case for case in report["cases"] if case["name"] == "cuquantum_statevector_expectation"
    )
    assert "competitor_semantic_mapping" in cuquantum["dataset"]
    assert "competitor_timing_boundary" in cuquantum["dataset"]
    assert "fastpauli_cuda_timing_boundary" in cuquantum["dataset"]
    assert "fastpauli_cuda_available" in cuquantum["results"]
    assert "fastpauli_cuda_device_statevector_available" in cuquantum["results"]
    assert "competitor_transfer_inclusive_seconds" in cuquantum["results"]
    assert "fastpauli_cuda_operator_resident_host_statevector_seconds" in cuquantum["results"]

    cupy_consumer = next(
        case for case in report["cases"] if case["name"] == "cupy_commutation_consumer"
    )
    assert "competitor_semantic_mapping" in cupy_consumer["dataset"]
    assert "competitor_timing_boundary" in cupy_consumer["dataset"]
    assert "fastpauli_cuda_compact_count_seconds" in cupy_consumer["results"]
    assert "competitor_dense_to_host_seconds" in cupy_consumer["results"]


def test_competitive_baseline_benchmark_can_write_report(tmp_path: Path) -> None:
    script = ROOT / "benchmarks" / "bench_competitive_baselines.py"
    output = tmp_path / "competitive_baselines.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
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
    stdout_report = json.loads(completed.stdout)
    written_report = json.loads(output.read_text(encoding="utf-8"))
    assert written_report == stdout_report
    assert written_report["benchmark"] == "competitive_baselines"


def test_cupy_import_status_is_independent_from_cuquantum(monkeypatch) -> None:
    from benchmarks import bench_competitive_baselines as baselines

    fake_cupy = SimpleNamespace(__name__="cupy")

    def fake_import_module(name: str) -> object:
        if name == "cupy":
            return fake_cupy
        if name == "cuquantum":
            raise ModuleNotFoundError("No module named 'cuquantum'")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(baselines.importlib, "import_module", fake_import_module)

    cupy, cupy_error = baselines.import_cupy_module()
    custatevec, cuda_data_type, cuquantum_error = baselines.import_cuquantum_statevector_stack()

    assert cupy is fake_cupy
    assert cupy_error is None
    assert custatevec is None
    assert cuda_data_type is None
    assert "cuquantum" in str(cuquantum_error)


def test_cupy_invalid_architecture_compile_failure_is_reported_unavailable(monkeypatch) -> None:
    from benchmarks import bench_competitive_baselines as baselines

    class FakeCompileException(Exception):
        pass

    fake_cupy = SimpleNamespace(
        __name__="cupy",
        cuda=SimpleNamespace(
            compiler=SimpleNamespace(CompileException=FakeCompileException),
        ),
    )

    compile_error = FakeCompileException("nvrtc: error: invalid value for --gpu-architecture (-arch)")

    def fake_asarray(_matrix: object) -> object:
        raise compile_error

    fake_cupy.asarray = fake_asarray

    monkeypatch.setattr(
        baselines.fastpauli._wolfgang_core,
        "_cuda_status",
        lambda: {
            "built": True,
            "runtime_available": True,
            "skip_reason": "",
            "devices": [{"compute_capability": (10, 3)}],
        },
    )

    lhs = SimpleNamespace(
        commutes_with=lambda _rhs: [[True]],
        num_terms=1,
        to_device=lambda: SimpleNamespace(
            commutes_with_device=lambda _rhs: SimpleNamespace(count_commuting=lambda: 1),
        ),
    )
    monkeypatch.setattr(baselines, "generate_labels", lambda **kwargs: (["X"], baselines.np.asarray([1.0 + 0.0j])))
    monkeypatch.setattr(baselines.PauliSum, "from_labels", lambda labels, coeffs: lhs)
    monkeypatch.setattr(baselines, "timed_call", lambda fn, warmup, repeat: (fn(), {"median": 0.0, "min": 0.0, "max": 0.0}))

    report = baselines.run_cupy_commutation_consumer_case(
        SimpleNamespace(smoke=True, num_qubits=8, lhs_terms=1, rhs_terms=1, term_weight=2, seed=1, warmup=0, repeat=1),
        fake_cupy,
        None,
    )

    assert report["results"]["competitor_available"] is False
    assert report["results"]["competitor_correctness_checked"] is False
    assert "gpu-architecture" in report["results"]["competitor_unavailable_reason"]


def test_cupy_reduction_compile_failure_is_reported_unavailable(monkeypatch) -> None:
    from benchmarks import bench_competitive_baselines as baselines

    class FakeCompileException(Exception):
        pass

    compile_error = FakeCompileException("nvrtc: error: invalid value for --gpu-architecture (-arch)")
    fake_view = SimpleNamespace(shape=(1, 1), dtype="uint8")
    fake_cupy = SimpleNamespace(
        __name__="cupy",
        uint8="uint8",
        asarray=lambda _matrix: fake_view,
        asnumpy=lambda value: value,
        cuda=SimpleNamespace(
            compiler=SimpleNamespace(CompileException=FakeCompileException),
        ),
    )

    def fake_sum(_view: object) -> object:
        raise compile_error

    fake_cupy.sum = fake_sum

    monkeypatch.setattr(
        baselines.fastpauli._wolfgang_core,
        "_cuda_status",
        lambda: {
            "built": True,
            "runtime_available": True,
            "skip_reason": "",
            "devices": [{"compute_capability": (10, 3)}],
        },
    )

    lhs = SimpleNamespace(
        commutes_with=lambda _rhs: [[True]],
        num_terms=1,
        to_device=lambda: SimpleNamespace(
            commutes_with_device=lambda _rhs: SimpleNamespace(count_commuting=lambda: 1),
        ),
    )
    monkeypatch.setattr(baselines, "generate_labels", lambda **kwargs: (["X"], baselines.np.asarray([1.0 + 0.0j])))
    monkeypatch.setattr(baselines.PauliSum, "from_labels", lambda labels, coeffs: lhs)
    monkeypatch.setattr(baselines, "timed_call", lambda fn, warmup, repeat: (fn(), {"median": 0.0, "min": 0.0, "max": 0.0}))

    report = baselines.run_cupy_commutation_consumer_case(
        SimpleNamespace(smoke=True, num_qubits=8, lhs_terms=1, rhs_terms=1, term_weight=2, seed=1, warmup=0, repeat=1),
        fake_cupy,
        None,
    )

    assert report["results"]["competitor_available"] is False
    assert report["results"]["competitor_correctness_checked"] is False
    assert "gpu-architecture" in report["results"]["competitor_unavailable_reason"]

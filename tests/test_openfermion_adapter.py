"""Phase 7 optional OpenFermion adapter tests."""

from __future__ import annotations

import builtins
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from wolfgang_quantum import PauliSum

OPENFERMION_INSTALL_HINT = "Install Wolfgang with the openfermion extra to use this adapter."
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def qubit_operator_type() -> Any:
    openfermion_ops = pytest.importorskip("openfermion.ops")
    return openfermion_ops.QubitOperator


def assert_qubit_operator_close(lhs: Any, rhs: Any) -> None:
    assert set(lhs.terms) == set(rhs.terms)
    for term in lhs.terms:
        np.testing.assert_allclose(lhs.terms[term], rhs.terms[term], rtol=1.0e-12, atol=1.0e-12)


def test_importing_fastpauli_does_not_import_openfermion() -> None:
    code = "import sys; import fastpauli; raise SystemExit('openfermion' in sys.modules)"

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_single_term_round_trip(qubit_operator_type: Any) -> None:
    source = qubit_operator_type("X0 Y2", 1.25 - 0.5j)

    op = PauliSum.from_openfermion(source)
    exported = op.to_openfermion()

    assert op.num_qubits == 3
    assert op.to_sparse_list() == [("XY", [0, 2], 1.25 - 0.5j)]
    assert_qubit_operator_close(exported, source)


def test_multi_term_sparse_round_trip(qubit_operator_type: Any) -> None:
    source = (
        qubit_operator_type("Z0", 2.0)
        + qubit_operator_type("X1 Y4", -0.25j)
        + qubit_operator_type("Y3 Z5", 3.0 + 1.0j)
    )

    op = PauliSum.from_openfermion(source)
    exported = op.to_openfermion()

    assert op.num_qubits == 6
    assert op.to_sparse_list() == [
        ("Z", [0], 2.0 + 0.0j),
        ("XY", [1, 4], -0.25j),
        ("YZ", [3, 5], 3.0 + 1.0j),
    ]
    assert_qubit_operator_close(exported, source)


def test_identity_term_round_trip_with_inferred_zero_qubits(qubit_operator_type: Any) -> None:
    source = qubit_operator_type((), 3.5 + 0.25j)

    op = PauliSum.from_openfermion(source)
    exported = op.to_openfermion()

    assert op.num_qubits == 0
    assert op.num_terms == 1
    assert op.to_sparse_list() == [("", [], 3.5 + 0.25j)]
    assert_qubit_operator_close(exported, source)


def test_identity_term_round_trip_with_explicit_num_qubits(qubit_operator_type: Any) -> None:
    source = qubit_operator_type((), -2.0j)

    op = PauliSum.from_openfermion(source, num_qubits=4)

    assert op.num_qubits == 4
    assert op.to_labels()[0] == ["IIII"]
    assert_qubit_operator_close(op.to_openfermion(), source)


def test_empty_operator_behavior(qubit_operator_type: Any) -> None:
    source = qubit_operator_type()

    inferred = PauliSum.from_openfermion(source)
    explicit = PauliSum.from_openfermion(source, num_qubits=5)

    assert inferred.num_qubits == 0
    assert inferred.num_terms == 0
    assert explicit.num_qubits == 5
    assert explicit.num_terms == 0
    assert inferred.to_openfermion().terms == {}
    assert explicit.to_openfermion().terms == {}


def test_provided_num_qubits_validation(qubit_operator_type: Any) -> None:
    source = qubit_operator_type("X3", 1.0)

    with pytest.raises(ValueError, match="out of range"):
        PauliSum.from_openfermion(source, num_qubits=3)


def test_duplicate_terms_match_after_simplify(qubit_operator_type: Any) -> None:
    source = (
        qubit_operator_type("X0", 1.0)
        + qubit_operator_type("X0", -0.25)
        + qubit_operator_type("Z2", 2.0j)
    )

    op = PauliSum.from_openfermion(source)
    exported = op.simplify().to_openfermion()

    assert_qubit_operator_close(exported, source)


def test_to_openfermion_rejects_non_paulisum_receiver() -> None:
    with pytest.raises(TypeError):
        PauliSum.to_openfermion(object())  # type: ignore[misc]


def test_from_openfermion_rejects_wrong_input_type(qubit_operator_type: Any) -> None:
    with pytest.raises(TypeError, match="QubitOperator"):
        PauliSum.from_openfermion(object())


def test_missing_openfermion_dependency_raises_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def fail_openfermion_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "openfermion" or name.startswith("openfermion."):
            raise ModuleNotFoundError("No module named 'openfermion'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_openfermion_import)

    with pytest.raises(ImportError, match=OPENFERMION_INSTALL_HINT):
        PauliSum.empty(1).to_openfermion()

    with pytest.raises(ImportError, match=OPENFERMION_INSTALL_HINT):
        PauliSum.from_openfermion(object())


def test_openfermion_conversion_benchmark_smoke_outputs_protocol_metadata(
    qubit_operator_type: Any,
) -> None:
    script = ROOT / "benchmarks" / "bench_openfermion_conversion.py"
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
    assert report["benchmark"] == "openfermion_conversion"
    assert report["fastpauli_build_info"]["cpu_backend"] == "scalar"
    assert {case["name"] for case in report["cases"]} == {
        "round_trip_conversion",
        "large_sparse_conversion",
    }
    for case in report["cases"]:
        assert "num_qubits" in case["dataset"]
        assert case["results"]["fastpauli_scalar_seconds"] >= 0.0
        assert case["results"]["openfermion_baseline_seconds"] >= 0.0

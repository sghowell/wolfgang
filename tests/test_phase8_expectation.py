"""Phase 8 CPU expectation-value tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from wolfgang_quantum import PauliSum

ROOT = Path(__file__).resolve().parents[1]


PAULI_MATRICES = {
    "I": np.asarray([[1, 0], [0, 1]], dtype=np.complex128),
    "X": np.asarray([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.asarray([[1, 0], [0, -1]], dtype=np.complex128),
}


def dense_matrix(labels: list[str], coeffs: list[complex]) -> np.ndarray:
    width = len(labels[0])
    matrix = np.zeros((1 << width, 1 << width), dtype=np.complex128)
    for label, coeff in zip(labels, coeffs, strict=True):
        term = PAULI_MATRICES[label[0]]
        for pauli in label[1:]:
            term = np.kron(term, PAULI_MATRICES[pauli])
        matrix += coeff * term
    return matrix


def normalized_state(width: int, dtype: np.dtype[np.complexfloating]) -> np.ndarray:
    rng = np.random.default_rng(9100 + width)
    raw = (
        rng.normal(size=1 << width) + 1j * rng.normal(size=1 << width)
    ).astype(dtype)
    return raw / np.linalg.norm(raw)


def direct_z_count_expectation(
    labels: list[str],
    coeffs: list[complex],
    counts: dict[str, float],
) -> complex:
    total = float(sum(counts.values()))
    result = 0.0 + 0.0j
    for label, coeff in zip(labels, coeffs, strict=True):
        weighted = 0.0
        for bitstring, count in counts.items():
            sign = 1.0
            for pauli, bit in zip(label, bitstring, strict=True):
                if pauli == "Z" and bit == "1":
                    sign = -sign
            weighted += float(count) * sign
        result += complex(coeff) * (weighted / total)
    return result


@pytest.mark.parametrize("width", range(1, 9))
def test_expectation_statevector_matches_dense_matrix_for_n_le_8(width: int) -> None:
    rng = np.random.default_rng(8100 + width)
    alphabet = np.asarray(["I", "X", "Y", "Z"])
    term_count = min(10, 2 + width)
    labels = ["".join(rng.choice(alphabet, size=width).tolist()) for _ in range(term_count)]
    coeffs = (
        rng.normal(size=term_count) + 1j * rng.normal(size=term_count)
    ).astype(np.complex128)
    psi = normalized_state(width, np.dtype(np.complex128))
    op = PauliSum.from_labels(labels, coeffs.tolist())

    expected = np.vdot(psi, dense_matrix(labels, coeffs.tolist()) @ psi)

    assert op.expectation_statevector(psi) == pytest.approx(expected, abs=1.0e-11)


def test_expectation_statevector_accepts_complex64_and_complex128() -> None:
    labels = ["XI", "YZ", "II"]
    coeffs = [0.5 - 0.25j, -1.25 + 0.75j, 0.125 + 0.0j]
    op = PauliSum.from_labels(labels, coeffs)

    psi128 = normalized_state(2, np.dtype(np.complex128))
    psi64 = psi128.astype(np.complex64)

    expected128 = np.vdot(psi128, dense_matrix(labels, coeffs) @ psi128)
    expected64 = np.vdot(
        psi64.astype(np.complex128),
        dense_matrix(labels, coeffs) @ psi64.astype(np.complex128),
    )

    assert op.expectation_statevector(psi128) == pytest.approx(expected128, abs=1.0e-12)
    assert op.expectation_statevector(psi64) == pytest.approx(expected64, abs=1.0e-6)


def test_expectation_statevector_diagonal_duplicate_terms_match_dense_matrix() -> None:
    base_labels = ["ZZII", "IZZI", "ZIIZ", "IIZZ", "ZIZI"]
    labels = [base_labels[index % len(base_labels)] for index in range(80)]
    coeffs = [
        complex(np.sin(index + 1) * 0.125, np.cos(index + 3) * 0.0625)
        for index in range(len(labels))
    ]
    psi = normalized_state(4, np.dtype(np.complex128))
    op = PauliSum.from_labels(labels, coeffs)

    expected = np.vdot(psi, dense_matrix(labels, coeffs) @ psi)

    assert op.expectation_statevector(psi) == pytest.approx(expected, abs=1.0e-11)


def test_expectation_statevector_handles_empty_and_zero_qubit_identity_terms() -> None:
    empty = PauliSum.empty(2)
    assert empty.expectation_statevector(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)) == 0j

    identity = PauliSum.from_sparse_list([("", [], 2.5 - 0.75j)], num_qubits=0)
    assert identity.expectation_statevector(np.asarray([1.0 + 0.0j], dtype=np.complex128)) == pytest.approx(2.5 - 0.75j)


def test_expectation_statevector_rejects_invalid_inputs() -> None:
    op = PauliSum.from_labels(["ZI"], [1.0])

    with pytest.raises(ValueError, match="1-dimensional"):
        op.expectation_statevector(np.ones((2, 2), dtype=np.complex128))

    with pytest.raises(TypeError, match="complex64 or complex128"):
        op.expectation_statevector(np.ones(4, dtype=np.float64))

    with pytest.raises(ValueError, match="2 \\*\\* num_qubits"):
        op.expectation_statevector(np.ones(2, dtype=np.complex128))

    with pytest.raises(TypeError, match="contiguous"):
        op.expectation_statevector(np.ones(8, dtype=np.complex128)[::2])

    with pytest.raises(ValueError, match="num_qubits <= 63"):
        PauliSum.empty(64).expectation_statevector(np.ones(1, dtype=np.complex128))


def test_expectation_z_counts_matches_direct_python_computation() -> None:
    labels = ["ZI", "IZ", "ZZ", "II"]
    coeffs = [1.25, -0.5 + 0.25j, 2.0, 0.75]
    counts = {"00": 7, "01": 5, "10": 3, "11": 11}
    op = PauliSum.from_labels(labels, coeffs)

    assert op.expectation_z_counts(counts) == pytest.approx(
        direct_z_count_expectation(labels, coeffs, counts),
        abs=1.0e-12,
    )


def test_expectation_z_counts_uses_dense_label_bitstring_endianness() -> None:
    counts = {"01": 10}

    assert PauliSum.from_labels(["IZ"], [1.0]).expectation_z_counts(counts) == pytest.approx(-1.0)
    assert PauliSum.from_labels(["ZI"], [1.0]).expectation_z_counts(counts) == pytest.approx(1.0)


def test_expectation_z_counts_rejects_non_diagonal_terms() -> None:
    with pytest.raises(ValueError, match="diagonal"):
        PauliSum.from_labels(["X"], [1.0]).expectation_z_counts({"0": 1})

    with pytest.raises(ValueError, match="diagonal"):
        PauliSum.from_labels(["Y"], [1.0]).expectation_z_counts({"0": 1})


def test_expectation_z_counts_rejects_invalid_counts() -> None:
    op = PauliSum.from_labels(["ZZ"], [1.0])

    with pytest.raises(ValueError, match="total count"):
        op.expectation_z_counts({})

    with pytest.raises(ValueError, match="length"):
        op.expectation_z_counts({"0": 1})

    with pytest.raises(ValueError, match="0 or 1"):
        op.expectation_z_counts({"02": 1})

    with pytest.raises(ValueError, match="non-negative"):
        op.expectation_z_counts({"00": -1})

    with pytest.raises(ValueError, match="finite"):
        op.expectation_z_counts({"00": float("inf")})

    with pytest.raises(ValueError, match="numeric"):
        op.expectation_z_counts({"00": object()})


def test_expectation_benchmark_smoke_outputs_protocol_metadata() -> None:
    script = ROOT / "benchmarks" / "bench_expectation.py"
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
    assert report["benchmark"] == "expectation"
    assert report["fastpauli_build_info"]["cpu_backend"] == "scalar"
    assert {case["name"] for case in report["cases"]} == {
        "statevector_few_terms_large_state",
        "statevector_many_terms_small_state",
        "statevector_diagonal_many_terms",
        "z_counts",
    }
    for case in report["cases"]:
        assert "num_qubits" in case["dataset"]
        assert "duplicate_rate" in case["dataset"]
        assert case["results"]["fastpauli_scalar_seconds"] >= 0.0

    statevector_cases = [case for case in report["cases"] if case["name"].startswith("statevector_")]
    for case in statevector_cases:
        assert "operator_random_seed" in case["dataset"]
        assert "statevector_random_seed" in case["dataset"]

    z_count_case = next(case for case in report["cases"] if case["name"] == "z_counts")
    assert "operator_random_seed" in z_count_case["dataset"]
    assert "counts_random_seed" in z_count_case["dataset"]

    diagonal_case = next(
        case for case in report["cases"] if case["name"] == "statevector_diagonal_many_terms"
    )
    assert diagonal_case["dataset"]["operator_family"] == "diagonal_z_only"

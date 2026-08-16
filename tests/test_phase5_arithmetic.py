"""Phase 5 arithmetic and multiplication tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from wolfgang_quantum import PauliSum

ROOT = Path(__file__).resolve().parents[1]

SINGLE_QUBIT_MATRICES: dict[str, np.ndarray] = {
    "I": np.asarray([[1, 0], [0, 1]], dtype=np.complex128),
    "X": np.asarray([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.asarray([[1, 0], [0, -1]], dtype=np.complex128),
}


def assert_labels_and_coeffs(op: PauliSum, labels: list[str], coeffs: list[complex]) -> None:
    actual_labels, actual_coeffs = op.to_labels()

    assert actual_labels == labels
    np.testing.assert_allclose(actual_coeffs, np.asarray(coeffs, dtype=np.complex128), rtol=0.0, atol=0.0)


def assert_same_pauli_sum(actual: PauliSum, expected: PauliSum) -> None:
    actual_labels, actual_coeffs = actual.to_labels()
    expected_labels, expected_coeffs = expected.to_labels()

    assert actual_labels == expected_labels
    np.testing.assert_allclose(actual_coeffs, expected_coeffs, rtol=1.0e-12, atol=1.0e-12)


def label_for(num_qubits: int, entries: dict[int, str]) -> str:
    chars = ["I"] * num_qubits
    for qubit, pauli in entries.items():
        chars[num_qubits - 1 - qubit] = pauli
    return "".join(chars)


def dense_matrix(labels: list[str], coeffs: np.ndarray) -> np.ndarray:
    if not labels:
        raise ValueError("empty labels cannot infer matrix width")

    dimension = 1 << len(labels[0])
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for label, coeff in zip(labels, coeffs, strict=True):
        term = np.asarray([[1]], dtype=np.complex128)
        for pauli in label:
            term = np.kron(term, SINGLE_QUBIT_MATRICES[pauli])
        matrix += complex(coeff) * term
    return matrix


def op_matrix(op: PauliSum) -> np.ndarray:
    labels, coeffs = op.to_labels()
    if not labels:
        dimension = 1 << op.num_qubits
        return np.zeros((dimension, dimension), dtype=np.complex128)
    return dense_matrix(labels, coeffs)


def random_labels(rng: np.random.Generator, num_terms: int, num_qubits: int) -> list[str]:
    alphabet = np.asarray(["I", "X", "Y", "Z"])
    return ["".join(rng.choice(alphabet, size=num_qubits).tolist()) for _ in range(num_terms)]


def random_op(seed: int, *, num_terms: int, num_qubits: int) -> PauliSum:
    rng = np.random.default_rng(seed)
    labels = random_labels(rng, num_terms=num_terms, num_qubits=num_qubits)
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    return PauliSum.from_labels(labels, coeffs.tolist())


def test_addition_concatenates_without_implicit_simplify() -> None:
    lhs = PauliSum.from_labels(["X", "Z"], [1.0, 2.0])
    rhs = PauliSum.from_labels(["X", "I"], [3.0, -4.0])

    combined = lhs + rhs

    assert_labels_and_coeffs(combined, ["X", "Z", "X", "I"], [1.0, 2.0, 3.0, -4.0])


def test_addition_rejects_mismatched_num_qubits() -> None:
    with pytest.raises(ValueError, match="same num_qubits"):
        _ = PauliSum.from_labels(["X"], [1.0]) + PauliSum.from_labels(["XX"], [1.0])


def test_scalar_multiplication_preserves_terms_including_zero() -> None:
    op = PauliSum.from_labels(["X", "Z"], [1.0 + 2.0j, -3.0])

    assert_labels_and_coeffs(
        op * (2.0 - 0.5j),
        ["X", "Z"],
        [(1.0 + 2.0j) * (2.0 - 0.5j), -3.0 * (2.0 - 0.5j)],
    )
    assert_labels_and_coeffs((1.5j) * op, ["X", "Z"], [(1.0 + 2.0j) * 1.5j, -3.0 * 1.5j])
    zero = 0.0 * op
    assert zero.num_terms == op.num_terms
    assert_labels_and_coeffs(zero, ["X", "Z"], [0.0, 0.0])


@pytest.mark.parametrize(
    ("lhs", "rhs", "label", "phase"),
    [
        ("X", "Y", "Z", 1.0j),
        ("Y", "X", "Z", -1.0j),
        ("Y", "Z", "X", 1.0j),
        ("Z", "Y", "X", -1.0j),
        ("Z", "X", "Y", 1.0j),
        ("X", "Z", "Y", -1.0j),
    ],
)
def test_single_qubit_multiplication_phase_fixtures(
    lhs: str,
    rhs: str,
    label: str,
    phase: complex,
) -> None:
    product = PauliSum.from_labels([lhs], [2.0]) @ PauliSum.from_labels([rhs], [3.0])

    assert_labels_and_coeffs(product, [label], [6.0 * phase])


def test_matmul_simplify_false_preserves_nested_loop_order() -> None:
    lhs = PauliSum.from_labels(["X", "Z"], [1.0, 2.0])
    rhs = PauliSum.from_labels(["Y", "I"], [3.0, 4.0])

    product = lhs.matmul(rhs, simplify=False)

    assert_labels_and_coeffs(product, ["Z", "X", "X", "Z"], [3.0j, 4.0, -6.0j, 8.0])


def test_matmul_default_simplifies_duplicate_products() -> None:
    lhs = PauliSum.from_labels(["X", "X"], [1.0, 2.0])
    rhs = PauliSum.from_labels(["Y"], [3.0])

    product = lhs @ rhs

    assert_labels_and_coeffs(product, ["Z"], [9.0j])


@pytest.mark.parametrize("num_qubits", [1, 65])
def test_duplicate_heavy_matmul_threshold_path_matches_materialized_reference(num_qubits: int) -> None:
    if num_qubits == 1:
        lhs_pool = ["X", "Z"]
        rhs_pool = ["Y", "I"]
    else:
        lhs_pool = [
            label_for(num_qubits, {0: "X", 64: "Z"}),
            label_for(num_qubits, {1: "Y", 64: "X"}),
        ]
        rhs_pool = [
            label_for(num_qubits, {0: "Y", 64: "I"}),
            label_for(num_qubits, {1: "Z", 64: "Y"}),
        ]

    lhs_labels = [lhs_pool[index % len(lhs_pool)] for index in range(128)]
    rhs_labels = [rhs_pool[index % len(rhs_pool)] for index in range(128)]
    lhs_coeffs = [complex((index % 17) - 8, (index % 5) - 2) for index in range(128)]
    rhs_coeffs = [complex((index % 11) - 5, (index % 7) - 3) for index in range(128)]
    lhs = PauliSum.from_labels(lhs_labels, lhs_coeffs)
    rhs = PauliSum.from_labels(rhs_labels, rhs_coeffs)

    optimized = lhs.matmul(rhs, simplify=True)
    materialized_reference = lhs.matmul(rhs, simplify=False).simplify()

    assert_same_pauli_sum(optimized, materialized_reference)


def test_matmul_guardrail_rejects_before_large_allocation() -> None:
    lhs = PauliSum(num_qubits=1, num_terms=3)
    rhs = PauliSum(num_qubits=1, num_terms=4)

    with pytest.raises(ValueError, match="max_intermediate_terms"):
        lhs.matmul(rhs, max_intermediate_terms=11)


def test_matmul_guardrail_rejects_overflow_dimensions() -> None:
    if not hasattr(PauliSum, "_checked_matmul_size_for_testing"):
        pytest.skip("matmul size guardrail hook is unavailable in this build")
    with pytest.raises(ValueError, match="intermediate term count overflows"):
        PauliSum._checked_matmul_size_for_testing(
            lhs_terms=(1 << 63),
            rhs_terms=2,
            max_intermediate_terms=(1 << 63),
        )


def test_small_random_associativity_after_simplify() -> None:
    a = random_op(101, num_terms=5, num_qubits=3)
    b = random_op(102, num_terms=4, num_qubits=3)
    c = random_op(103, num_terms=3, num_qubits=3)

    lhs = ((a @ b) @ c).simplify(atol=0.0, rtol=0.0)
    rhs = (a @ (b @ c)).simplify(atol=0.0, rtol=0.0)

    lhs_labels, lhs_coeffs = lhs.to_labels()
    rhs_labels, rhs_coeffs = rhs.to_labels()
    assert lhs_labels == rhs_labels
    np.testing.assert_allclose(lhs_coeffs, rhs_coeffs, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
def test_small_random_dense_matrix_comparison(num_qubits: int) -> None:
    lhs = random_op(200 + num_qubits, num_terms=5, num_qubits=num_qubits)
    rhs = random_op(300 + num_qubits, num_terms=4, num_qubits=num_qubits)

    product = lhs @ rhs

    np.testing.assert_allclose(op_matrix(product), op_matrix(lhs) @ op_matrix(rhs), rtol=1.0e-12, atol=1.0e-12)


def test_multiplication_benchmark_smoke_outputs_protocol_metadata() -> None:
    script = ROOT / "benchmarks" / "bench_multiply.py"
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
    assert report["benchmark"] == "multiply"
    assert report["fastpauli_build_info"]["cpu_backend"] == "scalar"
    assert {case["name"] for case in report["cases"]} == {
        "single_term",
        "small_cross_product",
        "simplified_duplicate_cross_product",
    }
    for case in report["cases"]:
        assert "max_intermediate_terms" in case["dataset"]
        assert case["results"]["fastpauli_scalar_seconds"] >= 0.0
        assert case["results"]["python_baseline_seconds"] >= 0.0

    simplified = next(
        case for case in report["cases"] if case["name"] == "simplified_duplicate_cross_product"
    )
    assert simplified["dataset"]["simplify_output"] is True
    assert simplified["dataset"]["duplicate_pressure"] == "intentional repeated product keys"

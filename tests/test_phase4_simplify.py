"""Phase 4 simplify and canonical-ordering tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from wolfgang_quantum import PauliSum

ROOT = Path(__file__).resolve().parents[1]


def label_for(num_qubits: int, entries: dict[int, str]) -> str:
    chars = ["I"] * num_qubits
    for qubit, pauli in entries.items():
        chars[num_qubits - 1 - qubit] = pauli
    return "".join(chars)


def assert_labels_and_coeffs(op: PauliSum, labels: list[str], coeffs: list[complex]) -> None:
    actual_labels, actual_coeffs = op.to_labels()

    assert actual_labels == labels
    np.testing.assert_array_equal(actual_coeffs, np.asarray(coeffs, dtype=np.complex128))


def assert_same_labels_and_coeffs(actual: PauliSum, expected: PauliSum) -> None:
    actual_labels, actual_coeffs = actual.to_labels()
    expected_labels, expected_coeffs = expected.to_labels()

    assert actual_labels == expected_labels
    np.testing.assert_array_equal(actual_coeffs, expected_coeffs)


def test_simplify_combines_duplicates_and_returns_one_word_canonical_order() -> None:
    op = PauliSum.from_labels(["Z", "X", "X", "I", "Z"], [1.0, 2.0, 3.0, 4.0, -0.25])

    simplified = op.simplify(atol=0.0, rtol=0.0)

    assert_labels_and_coeffs(simplified, ["I", "Z", "X"], [4.0, 0.75, 5.0])


def test_simplify_applies_tolerance_formula_with_inclusive_threshold() -> None:
    op = PauliSum.from_labels(
        ["X", "X", "Z", "Y"],
        [0.4, 0.6, 10.0, 1.0 + 1.0e-9],
    )

    simplified = op.simplify(atol=0.0, rtol=0.1)

    assert_labels_and_coeffs(simplified, ["Z", "Y"], [10.0, 1.0 + 1.0e-9])


@pytest.mark.parametrize(("atol", "rtol"), [(-1.0e-12, 0.0), (0.0, -1.0e-12)])
def test_simplify_rejects_negative_tolerances(atol: float, rtol: float) -> None:
    with pytest.raises(ValueError, match="tolerances must be non-negative"):
        PauliSum.from_labels(["X"], [1.0]).simplify(atol=atol, rtol=rtol)


def test_simplify_all_zero_output_preserves_qubit_count() -> None:
    simplified = PauliSum.from_labels(["XX", "ZZ"], [1.0e-13, 0.0]).simplify()

    assert simplified.num_qubits == 2
    assert simplified.num_terms == 0
    assert_labels_and_coeffs(simplified, [], [])
    assert simplified.to_sparse_list() == []


def test_simplify_returns_multiword_canonical_order() -> None:
    num_qubits = 130
    q64_z = label_for(num_qubits, {64: "Z"})
    q64_x = label_for(num_qubits, {64: "X"})
    q0_z = label_for(num_qubits, {0: "Z"})
    q0_x = label_for(num_qubits, {0: "X"})
    op = PauliSum.from_labels([q0_x, q64_x, q0_z, q64_z], [4.0, 2.0, 3.0, 1.0])

    simplified = op.simplify(atol=0.0, rtol=0.0)

    assert_labels_and_coeffs(simplified, [q64_z, q64_x, q0_z, q0_x], [1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize("num_qubits", [1, 65])
def test_duplicate_heavy_simplify_threshold_path_matches_sort_path(num_qubits: int) -> None:
    if num_qubits == 1:
        pool = ["I", "Z", "X", "Y"]
    else:
        pool = [
            label_for(num_qubits, {0: "X"}),
            label_for(num_qubits, {64: "Z"}),
            label_for(num_qubits, {0: "Y", 64: "X"}),
            label_for(num_qubits, {1: "Z", 64: "Y"}),
        ]

    labels = [pool[index % len(pool)] for index in range(4096)]
    coeffs = [
        complex((index % 13) - 6, (index % 7) - 3)
        for index in range(len(labels))
    ]
    sums = [sum(coeffs[index] for index, label in enumerate(labels) if label == pool_label)
            for pool_label in pool]

    optimized = PauliSum.from_labels(labels, coeffs).simplify(atol=0.0, rtol=0.0)
    sort_path_reference = PauliSum.from_labels(pool, sums).simplify(atol=0.0, rtol=0.0)

    assert_same_labels_and_coeffs(optimized, sort_path_reference)


@st.composite
def pauli_sum_inputs(draw: st.DrawFn) -> tuple[int, list[str], list[complex]]:
    num_qubits = draw(st.integers(min_value=1, max_value=70))
    num_terms = draw(st.integers(min_value=0, max_value=25))
    labels = draw(
        st.lists(
            st.text(alphabet="IXYZ", min_size=num_qubits, max_size=num_qubits),
            min_size=num_terms,
            max_size=num_terms,
        )
    )
    finite_float = st.floats(
        min_value=-10.0,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    )
    coeffs = draw(
        st.lists(
            st.builds(complex, finite_float, finite_float),
            min_size=num_terms,
            max_size=num_terms,
        )
    )
    return num_qubits, labels, coeffs


@given(pauli_sum_inputs())
@settings(max_examples=50, deadline=None)
def test_simplify_is_idempotent(data: tuple[int, list[str], list[complex]]) -> None:
    num_qubits, labels, coeffs = data
    op = PauliSum.from_labels(labels, coeffs) if labels else PauliSum.empty(num_qubits)

    once = op.simplify(atol=1.0e-9, rtol=1.0e-12)
    twice = once.simplify(atol=1.0e-9, rtol=1.0e-12)

    once_labels, once_coeffs = once.to_labels()
    twice_labels, twice_coeffs = twice.to_labels()
    assert once_labels == twice_labels
    np.testing.assert_array_equal(once_coeffs, twice_coeffs)


def random_labels(rng: np.random.Generator, num_terms: int, num_qubits: int) -> list[str]:
    alphabet = np.asarray(["I", "X", "Y", "Z"])
    return ["".join(rng.choice(alphabet, size=num_qubits).tolist()) for _ in range(num_terms)]


@pytest.mark.parametrize("num_qubits", [1, 2, 4])
def test_simplify_preserves_qiskit_matrix_semantics_for_small_random_inputs(
    num_qubits: int,
) -> None:
    qiskit_quantum_info: Any = pytest.importorskip("qiskit.quantum_info")
    rng = np.random.default_rng(8675309 + num_qubits)
    labels = random_labels(rng, num_terms=16, num_qubits=num_qubits)
    labels.extend(labels[:4])
    coeffs = rng.normal(size=len(labels)) + 1j * rng.normal(size=len(labels))
    source = qiskit_quantum_info.SparsePauliOp(labels, coeffs=coeffs)

    simplified = PauliSum.from_qiskit(source).simplify(atol=0.0, rtol=0.0).to_qiskit()

    np.testing.assert_allclose(
        simplified.to_matrix(),
        source.to_matrix(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_simplify_benchmark_smoke_outputs_protocol_metadata() -> None:
    script = ROOT / "benchmarks" / "bench_simplify.py"
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
    assert report["benchmark"] == "simplify"
    assert report["fastpauli_build_info"]["cpu_backend"] == "scalar"
    for key in [
        "compiler",
        "cmake",
        "cpu_vendor_or_soc",
        "physical_core_count",
        "logical_cpu_count",
        "instruction_sets",
        "active_fastpauli_cpu_backend",
        "cpu_cmake_options",
        "oneTBB",
        "CUDA",
        "thread_settings",
    ]:
        assert key in report["environment"]
    assert {case["name"] for case in report["cases"]} == {"low_duplicate", "high_duplicate"}
    for case in report["cases"]:
        assert case["dataset"]["num_qubits"] == 8
        assert case["dataset"]["coefficient_dtype"] == "complex128"
        assert "duplicate_rate" in case["dataset"]
        assert case["results"]["fastpauli_scalar_seconds"] >= 0.0
        assert case["results"]["python_baseline_seconds"] >= 0.0

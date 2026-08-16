"""Phase 6 commutation and grouping tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from wolfgang_quantum import PauliSum

ROOT = Path(__file__).resolve().parents[1]


def labels(op: PauliSum) -> list[str]:
    exported_labels, _ = op.to_labels()
    return list(exported_labels)


def labels_and_coeffs(op: PauliSum) -> tuple[list[str], np.ndarray]:
    exported_labels, coeffs = op.to_labels()
    return list(exported_labels), coeffs


def full_commutes(lhs: str, rhs: str) -> bool:
    parity = 0
    for lhs_pauli, rhs_pauli in zip(lhs, rhs, strict=True):
        parity ^= int((lhs_pauli in {"X", "Y"} and rhs_pauli in {"Z", "Y"})
                      != (lhs_pauli in {"Z", "Y"} and rhs_pauli in {"X", "Y"}))
    return parity == 0


def qwc_compatible(lhs: str, rhs: str) -> bool:
    for lhs_pauli, rhs_pauli in zip(lhs, rhs, strict=True):
        if lhs_pauli != "I" and rhs_pauli != "I" and lhs_pauli != rhs_pauli:
            return False
    return True


def assert_group_internal_compatibility(groups: list[PauliSum], *, mode: str) -> None:
    predicate = qwc_compatible if mode == "qwc" else full_commutes
    for group in groups:
        group_labels = labels(group)
        for index, lhs in enumerate(group_labels):
            for rhs in group_labels[index + 1:]:
                assert predicate(lhs, rhs), (mode, group_labels, lhs, rhs)


def grouped_labels(groups: list[PauliSum]) -> list[list[str]]:
    return [labels(group) for group in groups]


def test_commutes_with_scalar_vector_and_matrix_shapes() -> None:
    single_x = PauliSum.from_labels(["X"], [1.0])
    single_z = PauliSum.from_labels(["Z"], [1.0])
    rhs = PauliSum.from_labels(["X", "Y", "I"], [1.0, 2.0, 3.0])
    lhs = PauliSum.from_labels(["X", "Z"], [1.0, 2.0])

    assert single_x.commutes_with(single_x) is True
    assert single_x.commutes_with(single_z) is False

    lhs_single_vector = single_x.commutes_with(rhs)
    assert lhs_single_vector.dtype == np.bool_
    assert lhs_single_vector.shape == (3,)
    assert lhs_single_vector.flags.writeable
    np.testing.assert_array_equal(lhs_single_vector, np.asarray([True, False, True]))

    rhs_single_vector = lhs.commutes_with(single_x)
    assert rhs_single_vector.dtype == np.bool_
    assert rhs_single_vector.shape == (2,)
    np.testing.assert_array_equal(rhs_single_vector, np.asarray([True, False]))

    matrix = lhs.commutes_with(rhs)
    assert matrix.dtype == np.bool_
    assert matrix.shape == (2, 3)
    assert matrix.flags.writeable
    np.testing.assert_array_equal(
        matrix,
        np.asarray(
            [
                [True, False, True],
                [False, False, True],
            ],
            dtype=np.bool_,
        ),
    )


def test_commutes_with_rejects_mismatched_num_qubits() -> None:
    with pytest.raises(ValueError, match="same num_qubits"):
        PauliSum.from_labels(["X"], [1.0]).commutes_with(PauliSum.from_labels(["XX"], [1.0]))


def test_commutes_with_guardrail_rejects_before_large_allocation() -> None:
    lhs = PauliSum(num_qubits=1, num_terms=3)
    rhs = PauliSum(num_qubits=1, num_terms=4)

    with pytest.raises(ValueError, match="max_commutation_matrix_entries"):
        lhs.commutes_with(rhs, max_commutation_matrix_entries=11)


def test_commutes_with_guardrail_rejects_overflow_dimensions() -> None:
    if not hasattr(PauliSum, "_checked_commutation_size_for_testing"):
        pytest.skip("commutation size guardrail hook is unavailable in this build")
    with pytest.raises(ValueError, match="commutation matrix entry count overflows"):
        PauliSum._checked_commutation_size_for_testing(
            lhs_terms=(1 << 63),
            rhs_terms=2,
            max_commutation_matrix_entries=(1 << 63),
        )


def test_commutation_symmetry_property() -> None:
    rng = np.random.default_rng(1979)
    alphabet = np.asarray(["I", "X", "Y", "Z"])

    for _ in range(30):
        lhs_terms = int(rng.integers(1, 8))
        rhs_terms = int(rng.integers(1, 8))
        lhs_labels = ["".join(rng.choice(alphabet, size=4).tolist()) for _ in range(lhs_terms)]
        rhs_labels = ["".join(rng.choice(alphabet, size=4).tolist()) for _ in range(rhs_terms)]
        lhs = PauliSum.from_labels(lhs_labels, np.ones(lhs_terms).tolist())
        rhs = PauliSum.from_labels(rhs_labels, np.ones(rhs_terms).tolist())

        lhs_rhs = np.asarray(lhs.commutes_with(rhs), dtype=np.bool_).reshape(lhs_terms, rhs_terms)
        rhs_lhs = np.asarray(rhs.commutes_with(lhs), dtype=np.bool_).reshape(rhs_terms, lhs_terms)
        np.testing.assert_array_equal(lhs_rhs, rhs_lhs.T)


def test_group_commuting_qwc_groups_are_valid_and_deterministic() -> None:
    op = PauliSum.from_labels(
        ["XI", "IX", "XX", "YY", "ZI", "IZ", "II"],
        [1.0, -2.0, 3.0, 4.0, 5.0, -6.0, 7.0],
    )

    groups = op.group_commuting(mode="qwc")
    repeated = op.group_commuting(mode="qwc")

    assert_group_internal_compatibility(groups, mode="qwc")
    assert grouped_labels(groups) == grouped_labels(repeated)
    assert sorted(label for group in grouped_labels(groups) for label in group) == sorted(labels(op))


def test_group_commuting_full_groups_are_valid_and_deterministic() -> None:
    op = PauliSum.from_labels(
        ["XI", "IX", "XX", "YY", "ZI", "IZ", "II"],
        [1.0, -2.0, 3.0, 4.0, 5.0, -6.0, 7.0],
    )

    groups = op.group_commuting(mode="full")
    repeated = op.group_commuting(mode="full")

    assert_group_internal_compatibility(groups, mode="full")
    assert grouped_labels(groups) == grouped_labels(repeated)
    assert sorted(label for group in grouped_labels(groups) for label in group) == sorted(labels(op))


def test_group_commuting_full_streaming_path_matches_graph_path() -> None:
    op = PauliSum.from_labels(
        ["XI", "IX", "XX", "YY", "ZI", "IZ", "II"],
        [1.0, -2.0, 3.0, 4.0, 5.0, -6.0, 7.0],
    )

    graph_groups = op.group_commuting(mode="full")
    streaming_groups = op.group_commuting(mode="full", max_terms_for_graph=0)

    assert grouped_labels(streaming_groups) == grouped_labels(graph_groups)
    assert_group_internal_compatibility(streaming_groups, mode="full")


def test_group_commuting_preserves_coefficients_in_greedy_order() -> None:
    op = PauliSum.from_labels(["II", "XI", "XX", "IZ"], [1.0, 2.0, 3.0, 4.0])

    groups = op.group_commuting(mode="qwc")

    assert grouped_labels(groups) == [["XX", "XI", "II"], ["IZ"]]
    first_labels, first_coeffs = labels_and_coeffs(groups[0])
    second_labels, second_coeffs = labels_and_coeffs(groups[1])
    assert first_labels == ["XX", "XI", "II"]
    assert second_labels == ["IZ"]
    np.testing.assert_array_equal(first_coeffs, np.asarray([3.0, 2.0, 1.0], dtype=np.complex128))
    np.testing.assert_array_equal(second_coeffs, np.asarray([4.0], dtype=np.complex128))


def test_group_commuting_rejects_invalid_mode_and_strategy() -> None:
    op = PauliSum.from_labels(["X"], [1.0])

    with pytest.raises(ValueError, match="mode"):
        op.group_commuting(mode="diagonal")

    with pytest.raises(ValueError, match="strategy"):
        op.group_commuting(strategy="smallest_first")


def test_grouping_benchmark_smoke_outputs_protocol_metadata() -> None:
    script = ROOT / "benchmarks" / "bench_grouping.py"
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
    assert report["benchmark"] == "grouping"
    assert report["fastpauli_build_info"]["cpu_backend"] == "scalar"
    assert {case["name"] for case in report["cases"]} == {
        "pairwise_commutation",
        "qwc_grouping",
        "full_grouping",
        "guardrail_rejection",
    }
    for case in report["cases"]:
        assert "num_qubits" in case["dataset"]
        assert case["results"]["fastpauli_scalar_seconds"] >= 0.0

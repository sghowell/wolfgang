"""Phase 2 packed representation and parsing/export tests."""

from __future__ import annotations

import numpy as np
import pytest
from wolfgang_quantum import PauliSum


def label_for(num_qubits: int, entries: dict[int, str]) -> str:
    chars = ["I"] * num_qubits
    for qubit, pauli in entries.items():
        chars[num_qubits - 1 - qubit] = pauli
    return "".join(chars)


def assert_complex128_array(values: object, expected: list[complex]) -> None:
    assert isinstance(values, np.ndarray)
    assert values.dtype == np.complex128
    np.testing.assert_array_equal(values, np.asarray(expected, dtype=np.complex128))


def test_from_labels_to_labels_round_trip_preserves_order_and_coefficients() -> None:
    op = PauliSum.from_labels(["I", "X", "Y", "Z"], [1, 2.5, np.complex64(3 + 4j), -1j])

    labels, coeffs = op.to_labels()

    assert labels == ["I", "X", "Y", "Z"]
    assert op.num_qubits == 1
    assert op.num_terms == 4
    assert_complex128_array(coeffs, [1, 2.5, 3 + 4j, -1j])


def test_from_labels_defaults_coefficients_to_one() -> None:
    op = PauliSum.from_labels(["XXI", "IZZ", "YYI"])

    labels, coeffs = op.to_labels()

    assert labels == ["XXI", "IZZ", "YYI"]
    assert_complex128_array(coeffs, [1 + 0j, 1 + 0j, 1 + 0j])


def test_dense_label_endianness_fixture() -> None:
    op = PauliSum.from_labels(["XYZ"])

    assert op.to_sparse_list() == [("ZYX", [0, 1, 2], 1.0 + 0.0j)]


def test_from_labels_rejects_bare_string_input() -> None:
    with pytest.raises(ValueError, match="not a single string"):
        PauliSum.from_labels("XYZ")


def test_sparse_list_to_dense_label_endianness_fixture() -> None:
    op = PauliSum.from_sparse_list([("ZX", [1, 4], 1.0)], num_qubits=5)

    labels, coeffs = op.to_labels()

    assert labels == ["XIIZI"]
    assert_complex128_array(coeffs, [1.0 + 0.0j])


def test_from_sparse_list_preserves_term_order_and_exports_sorted_qubits() -> None:
    op = PauliSum.from_sparse_list(
        [
            ("XZ", [4, 1], 2.0),
            ("Y", [0], -0.5j),
            ("", [], 3.0),
        ],
        num_qubits=5,
    )

    assert op.to_sparse_list() == [
        ("ZX", [1, 4], 2.0 + 0.0j),
        ("Y", [0], -0.5j),
        ("", [], 3.0 + 0.0j),
    ]
    labels, coeffs = op.to_labels()
    assert labels == ["XIIZI", "IIIIY", "IIIII"]
    assert_complex128_array(coeffs, [2.0, -0.5j, 3.0])


def test_empty_operator_is_explicit_and_round_trips() -> None:
    op = PauliSum.empty(5)

    labels, coeffs = op.to_labels()

    assert op.num_qubits == 5
    assert op.num_terms == 0
    assert labels == []
    assert_complex128_array(coeffs, [])
    assert op.to_sparse_list() == []


def test_from_sparse_list_accepts_empty_input_when_num_qubits_is_explicit() -> None:
    op = PauliSum.from_sparse_list([], num_qubits=3)

    assert op.num_qubits == 3
    assert op.num_terms == 0
    assert op.to_sparse_list() == []


@pytest.mark.parametrize(
    ("labels", "coeffs"),
    [
        ([], None),
        (["X", "YY"], None),
        (["A"], None),
        (["X", "Y"], [1.0]),
        (["X", "Y"], 2.0),
    ],
)
def test_from_labels_rejects_invalid_inputs(labels: list[str], coeffs: object | None) -> None:
    with pytest.raises(ValueError):
        PauliSum.from_labels(labels, coeffs)


@pytest.mark.parametrize(
    ("triples", "num_qubits"),
    [
        ([("XZ", [0], 1.0)], 2),
        ([("A", [0], 1.0)], 1),
        ([("XZ", [0, 0], 1.0)], 2),
        ([("X", [2], 1.0)], 2),
        ([("X", [-1], 1.0)], 2),
        ([("X", [0], object())], 2),
    ],
)
def test_from_sparse_list_rejects_invalid_inputs(triples: list[tuple[object, object, object]], num_qubits: int) -> None:
    with pytest.raises(ValueError):
        PauliSum.from_sparse_list(triples, num_qubits=num_qubits)


def test_negative_explicit_num_qubits_is_invalid() -> None:
    with pytest.raises(ValueError):
        PauliSum.empty(-1)

    with pytest.raises(ValueError):
        PauliSum.from_sparse_list([], num_qubits=-1)


@pytest.mark.parametrize("num_qubits", [64, 65, 128, 129])
def test_multiword_dense_and_sparse_round_trip(num_qubits: int) -> None:
    entries = {0: "Y", num_qubits - 1: "X"}
    if num_qubits > 64:
        entries[64] = "Z"
    label = label_for(num_qubits, entries)

    op = PauliSum.from_labels([label], [1.25 - 0.5j])

    labels, coeffs = op.to_labels()
    assert labels == [label]
    assert_complex128_array(coeffs, [1.25 - 0.5j])

    sparse = op.to_sparse_list()
    expected_local = "".join(entries[index] for index in sorted(entries))
    assert sparse == [(expected_local, sorted(entries), 1.25 - 0.5j)]


def test_final_word_high_bits_are_zeroed_for_non_multiple_of_64_qubits() -> None:
    op = PauliSum.from_sparse_list([("XY", [0, 64], 1.0)], num_qubits=65)

    if not hasattr(op, "_packed_words_for_testing"):
        pytest.skip("_packed_words_for_testing is unavailable in this build")

    x_words, z_words = op._packed_words_for_testing()

    assert x_words == [1, 1]
    assert z_words == [0, 1]


def test_single_label_accepts_scalar_coefficient() -> None:
    labels, coeffs = PauliSum.from_labels(["X"], 2.0).to_labels()

    assert labels == ["X"]
    assert_complex128_array(coeffs, [2.0])

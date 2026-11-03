"""Packed-array I/O and operator algebra on host PauliSum.

The oracles are the existing label I/O and matmul: from_arrays must agree with
from_labels on packed words, and commutator/anticommutator must agree with the
composed products A @ B -/+ B @ A after simplify.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from wolfgang_quantum import PauliSum

WIDTHS = [0, 1, 5, 63, 64, 65, 128, 130]


def random_operator(rng: np.random.Generator, num_qubits: int, num_terms: int) -> PauliSum:
    if num_qubits == 0:
        coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
        return PauliSum.from_arrays(
            np.zeros((num_terms, 0), dtype=np.uint64),
            np.zeros((num_terms, 0), dtype=np.uint64),
            coeffs,
            0,
        )
    labels = ["".join(rng.choice(list("IXYZ"), size=num_qubits)) for _ in range(num_terms)]
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    return PauliSum.from_labels(labels, coeffs)


def labels_coeffs(op: PauliSum) -> tuple[list[str], np.ndarray]:
    labels, coeffs = op.to_labels()
    return list(labels), np.asarray(coeffs)


# --- packed array I/O -------------------------------------------------------------


@pytest.mark.parametrize("num_qubits", WIDTHS)
def test_to_arrays_round_trips_and_matches_label_packing(num_qubits: int) -> None:
    rng = np.random.default_rng(100 + num_qubits)
    op = random_operator(rng, num_qubits, 17)
    x, z, coeffs = op.to_arrays()
    words = math.ceil(num_qubits / 64)
    assert x.dtype == np.uint64 and z.dtype == np.uint64 and coeffs.dtype == np.complex128
    assert x.shape == z.shape == (17, words)
    assert coeffs.shape == (17,)
    assert PauliSum.from_arrays(x, z, coeffs, num_qubits) == op


def test_packed_bit_layout_matches_documented_convention() -> None:
    # Label characters run from qubit n-1 (left) to qubit 0 (right).
    op = PauliSum.from_labels(["XIZ", "YII"], [1.0, 2.0])
    x, z, _ = op.to_arrays()
    assert x[:, 0].tolist() == [0b100, 0b100]
    assert z[:, 0].tolist() == [0b001, 0b100]
    wide = PauliSum.from_sparse_list([("X", [64], 1.0), ("Z", [0], 1.0)], num_qubits=70)
    wx, wz, _ = wide.to_arrays()
    assert wx.tolist() == [[0, 1], [0, 0]]
    assert wz.tolist() == [[0, 0], [1, 0]]


def test_to_arrays_views_are_read_only_and_outlive_the_operator() -> None:
    op = PauliSum.from_labels(["XY", "ZZ"], [1.5, -2.0j])
    x, z, coeffs = op.to_arrays()
    for array in (x, z, coeffs):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[...] = 0
    expected = (x.copy(), z.copy(), coeffs.copy())
    del op
    import gc

    gc.collect()
    np.testing.assert_array_equal(x, expected[0])
    np.testing.assert_array_equal(z, expected[1])
    np.testing.assert_array_equal(coeffs, expected[2])


def test_empty_operator_arrays() -> None:
    x, z, coeffs = PauliSum.empty(70).to_arrays()
    assert x.shape == z.shape == (0, 2) and coeffs.shape == (0,)
    rebuilt = PauliSum.from_arrays(x, z, coeffs, 70)
    assert rebuilt.num_terms == 0 and rebuilt.num_qubits == 70


def test_from_arrays_copies_and_accepts_non_contiguous_and_real_coefficients() -> None:
    base_x = np.array([[1, 99], [2, 99]], dtype=np.uint64)
    base_z = np.array([[0, 99], [3, 99]], dtype=np.uint64)
    coeffs = np.array([1.0, -0.5], dtype=np.float32)
    op = PauliSum.from_arrays(base_x[:, :1], base_z[:, :1], coeffs, 2)
    base_x[:] = 0
    labels, values = labels_coeffs(op)
    # Word 2 = 0b10 sets x on qubit 1 and word 3 = 0b11 sets z on qubits 0 and 1.
    assert labels == ["IX", "YZ"]
    np.testing.assert_array_equal(values, [1.0 + 0j, -0.5 + 0j])


@pytest.mark.parametrize(
    ("x", "z", "coeffs", "num_qubits", "error", "match"),
    [
        (np.zeros((1, 1), np.int64), np.zeros((1, 1), np.uint64), [1.0], 3, TypeError, "uint64"),
        (np.zeros((1, 1), np.uint64), np.zeros((1, 1), np.float64), [1.0], 3, TypeError, "uint64"),
        (np.zeros(1, np.uint64), np.zeros(1, np.uint64), [1.0], 3, ValueError, "shape"),
        (np.zeros((1, 2), np.uint64), np.zeros((1, 2), np.uint64), [1.0], 3, ValueError, "shape"),
        (np.zeros((2, 1), np.uint64), np.zeros((2, 1), np.uint64), [1.0], 3, ValueError, "shape"),
        (np.zeros((1, 1), np.uint64), np.zeros((1, 1), np.uint64), ["a"], 3, TypeError, "numeric"),
        (np.zeros((1, 1), np.uint64), np.zeros((1, 1), np.uint64), [[1.0]], 3, ValueError, "1-dim"),
        (np.full((1, 1), 8, np.uint64), np.zeros((1, 1), np.uint64), [1.0], 3, ValueError, "num_qubits"),
        (np.zeros((1, 2), np.uint64), np.array([[0, 1 << 6]], np.uint64), [1.0], 70, ValueError, "num_qubits"),
    ],
)
def test_from_arrays_rejects_malformed_input(
    x: np.ndarray, z: np.ndarray, coeffs: object, num_qubits: int, error: type, match: str
) -> None:
    with pytest.raises(error, match=match):
        PauliSum.from_arrays(x, z, coeffs, num_qubits)


def test_from_arrays_full_final_word_is_valid() -> None:
    full = np.array([[np.iinfo(np.uint64).max]], dtype=np.uint64)
    op = PauliSum.from_arrays(full, np.zeros_like(full), [1.0], 64)
    assert op.to_labels()[0] == ["X" * 64]


# --- container protocol and equality -----------------------------------------------


def test_len_indexing_slicing_and_iteration() -> None:
    op = PauliSum.from_labels(["XI", "IZ", "YY", "ZX"], [1, 2, 3, 4])
    assert len(op) == op.num_terms == 4
    assert labels_coeffs(op[0])[0] == ["XI"]
    assert labels_coeffs(op[-1])[0] == ["ZX"]
    assert labels_coeffs(op[1:3])[0] == ["IZ", "YY"]
    assert labels_coeffs(op[::-2])[0] == ["ZX", "IZ"]
    assert op[4:].num_terms == 0 and op[4:].num_qubits == 2
    assert [labels_coeffs(term)[0][0] for term in op] == ["XI", "IZ", "YY", "ZX"]
    assert labels_coeffs(op[np.int64(2)])[0] == ["YY"]
    with pytest.raises(IndexError):
        op[4]
    with pytest.raises(IndexError):
        op[-5]
    with pytest.raises(TypeError):
        op["0"]  # type: ignore[index]


def test_structural_equality_semantics() -> None:
    a = PauliSum.from_labels(["XX", "ZI"], [1.0, 2.0])
    assert a == PauliSum.from_labels(["XX", "ZI"], [1.0, 2.0])
    assert a != PauliSum.from_labels(["ZI", "XX"], [2.0, 1.0])  # order matters
    assert a != PauliSum.from_labels(["XX", "XX", "ZI"], [0.5, 0.5, 2.0])  # unmerged
    assert PauliSum.from_labels(["X"], [0.0]) == PauliSum.from_labels(["X"], [-0.0])
    assert PauliSum.from_labels(["X"], [math.nan]) != PauliSum.from_labels(["X"], [math.nan])
    assert PauliSum.empty(2) != PauliSum.empty(3)
    assert (a == "XX") is False
    with pytest.raises(TypeError):
        hash(a)


def test_equiv_ignores_order_and_duplicates_within_tolerance() -> None:
    a = PauliSum.from_labels(["XX", "ZI"], [1.0, 2.0])
    b = PauliSum.from_labels(["ZI", "XX", "XX"], [2.0, 0.25, 0.75 + 1e-14])
    assert a.equiv(b)
    assert not a.equiv(b, atol=1e-16)
    assert not a.equiv(PauliSum.from_labels(["XXI", "ZII"], [1.0, 2.0]))
    with pytest.raises(ValueError):
        a.equiv(b, atol=-1.0)


# --- arithmetic ------------------------------------------------------------------


def test_subtraction_and_negation_are_exact() -> None:
    a = PauliSum.from_labels(["XY", "ZZ"], [complex(math.inf, 0.0), 1.0 - 2.0j])
    negated = -a
    _, coeffs = labels_coeffs(negated)
    assert coeffs[0].real == -math.inf and coeffs[0].imag == 0.0  # no 0 * inf NaN
    assert coeffs[1] == -1.0 + 2.0j
    b = PauliSum.from_labels(["XY"], [0.5])
    diff = PauliSum.from_labels(["ZZ"], [3.0]) - b
    assert labels_coeffs(diff)[0] == ["ZZ", "XY"]
    np.testing.assert_array_equal(labels_coeffs(diff)[1], [3.0, -0.5])
    with pytest.raises(ValueError):
        a - PauliSum.from_labels(["XYZ"], [1.0])


def test_adjoint_conjugates_coefficients() -> None:
    op = PauliSum.from_labels(["XY", "ZZ"], [1.0 + 2.0j, -3.0j])
    labels, coeffs = labels_coeffs(op.adjoint())
    assert labels == ["XY", "ZZ"]
    np.testing.assert_array_equal(coeffs, [1.0 - 2.0j, 3.0j])


@pytest.mark.parametrize("num_qubits", [0, 3, 64, 70, 130])
def test_commutator_and_anticommutator_match_composed_products(num_qubits: int) -> None:
    rng = np.random.default_rng(500 + num_qubits)
    a = random_operator(rng, num_qubits, 23)
    b = random_operator(rng, num_qubits, 19)
    composed_commutator = (a @ b - b @ a).simplify()
    composed_anticommutator = (a @ b + b @ a).simplify()
    assert a.commutator(b).equiv(composed_commutator, atol=1e-10)
    assert a.anticommutator(b).equiv(composed_anticommutator, atol=1e-10)
    # The fused results are already simplified into canonical order.
    assert a.commutator(b) == a.commutator(b).simplify()


def test_commutator_known_values_and_unsimplified_order() -> None:
    x = PauliSum.from_labels(["X"], [1.0])
    y = PauliSum.from_labels(["Y"], [1.0])
    z = PauliSum.from_labels(["Z"], [1.0])
    assert x.commutator(y) == PauliSum.from_labels(["Z"], [2.0j])  # [X, Y] = 2iZ
    assert x.anticommutator(y).num_terms == 0
    assert x.anticommutator(x) == PauliSum.from_labels(["I"], [2.0])
    assert x.commutator(x).num_terms == 0

    lhs = PauliSum.from_labels(["X", "Z"], [1.0, 1.0])
    rhs = PauliSum.from_labels(["Y", "X", "Z"], [1.0, 1.0, 1.0])
    raw = lhs.commutator(rhs, simplify=False)
    # Only anticommuting pairs survive, in lhs-outer, rhs-inner order:
    # X*Y, X*Z, Z*Y, Z*X.
    assert labels_coeffs(raw)[0] == ["Z", "Y", "X", "Y"]
    np.testing.assert_array_equal(labels_coeffs(raw)[1], [2j, -2j, -2j, 2j])
    assert raw.num_terms == 4
    assert z.commutator(x) == PauliSum.from_labels(["Y"], [2.0j])


def test_commutator_guards_and_validation() -> None:
    a = PauliSum.from_labels(["X", "Z", "Y"], [1.0, 1.0, 1.0])
    b = PauliSum.from_labels(["X", "Z", "Y"], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        a.commutator(b, max_intermediate_terms=8)
    assert a.commutator(b, max_intermediate_terms=9).num_terms > 0
    assert a.commutator(a).num_terms == 0
    with pytest.raises(ValueError):
        a.commutator(PauliSum.from_labels(["XX"], [1.0]))
    assert a.commutator(PauliSum.empty(1)).num_terms == 0


# --- review regressions ----------------------------------------------------------


def test_commutator_scales_after_multiplying_and_rejects_non_finite() -> None:
    big = PauliSum.from_labels(["X"], [1e308])
    tiny = PauliSum.from_labels(["Z"], [1e-10])
    assert big.commutator(tiny).equiv((big @ tiny - tiny @ big).simplify())
    # A NaN or infinity raises even when its pairs would all cancel.
    poisoned = PauliSum.from_labels(["X", "Z"], [math.inf, 1.0])
    x = PauliSum.from_labels(["X"], [1.0])
    with pytest.raises(ValueError, match="finite"):
        poisoned.commutator(x)
    with pytest.raises(ValueError, match="finite"):
        x.anticommutator(PauliSum.from_labels(["Z"], [math.nan]))
    with pytest.raises(ValueError, match="commutator intermediate term count"):
        x.commutator(x, max_intermediate_terms=0)


def test_from_labels_coefficient_paths() -> None:
    class ArrayLike:
        """Duck-typed array: has dtype/ndim but is not a NumPy array."""

        dtype = object()
        ndim = 1

        def __iter__(self):
            return iter([1.0, 2.0j])

    op = PauliSum.from_labels(["X", "Z"], ArrayLike())
    np.testing.assert_array_equal(op.to_labels()[1], [1.0, 2.0j])
    bools = PauliSum.from_labels(["X", "Z"], np.array([True, False]))
    np.testing.assert_array_equal(bools.to_labels()[1], [1.0, 0.0])
    arrays = PauliSum.from_arrays(
        np.zeros((2, 1), np.uint64), np.zeros((2, 1), np.uint64), np.array([True, False]), 1
    )
    np.testing.assert_array_equal(arrays.to_arrays()[2], [1.0, 0.0])
    with pytest.raises(TypeError, match="native byte order"):
        PauliSum.from_arrays(np.zeros((1, 1), ">u8"), np.zeros((1, 1), ">u8"), [1.0], 3)


def test_getitem_none_and_to_arrays_self_type() -> None:
    op = PauliSum.from_labels(["X"], [1.0])
    with pytest.raises(TypeError, match="integers or slices"):
        op[None]  # type: ignore[index]
    with pytest.raises(TypeError):
        PauliSum.to_arrays(1)  # type: ignore[arg-type]
    assert not PauliSum.empty(2)  # empty operators are falsy via __len__

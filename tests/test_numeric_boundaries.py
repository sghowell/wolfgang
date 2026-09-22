"""Numerical boundary regressions for the public CPU oracle."""

import numpy as np
import pytest
from wolfgang_quantum import PauliSum


@pytest.mark.parametrize("width", [0, 1, 65, 130])
@pytest.mark.parametrize("rtol", [0.0, 0.5, 1.0])
def test_simplify_handles_finite_components_with_overflowing_magnitude(width, rtol):
    coefficient = complex(1.7e308, 1.7e308)
    result = PauliSum.from_labels(["X" * width], [coefficient]).simplify(atol=0, rtol=rtol)
    assert result.num_terms == (0 if rtol == 1.0 else 1)
    if result.num_terms:
        np.testing.assert_array_equal(result.to_labels()[1], [coefficient])


@pytest.mark.parametrize("width", [1, 65, 130])
def test_simplify_large_magnitude_does_not_erase_unrelated_terms(width):
    op = PauliSum.from_labels(["X" * width, "Z" * width], [complex(1.7e308, 1.7e308), 1])
    assert op.simplify(atol=0, rtol=0).num_terms == 2


@pytest.mark.parametrize("width", [1, 65])
def test_simplify_hash_path_preserves_large_finite_coefficient(width):
    op = PauliSum.from_labels(["X" * width] * 4096, [complex(1.7e308, 1.7e308)] + [0] * 4095)
    np.testing.assert_array_equal(op.simplify(atol=0).to_labels()[1], [complex(1.7e308, 1.7e308)])


@pytest.mark.parametrize("coefficient", [complex(float("nan"), 0), complex(float("inf"), 1)])
def test_simplify_rejects_nonfinite_coefficients(coefficient):
    with pytest.raises(ValueError, match="finite coefficients"):
        PauliSum.from_labels(["X"], [coefficient]).simplify()


def test_simplify_rejects_overflow_during_duplicate_reduction():
    with pytest.raises(OverflowError, match="coefficient"):
        PauliSum.from_labels(["X", "X"], [1.7e308, 1.7e308]).simplify()


@pytest.mark.parametrize("scale", [1.0, 1e308, 1e-308])
def test_counts_expectation_is_invariant_to_finite_rescaling(scale):
    counts = {"0": 1.7 * scale, "1": 0.85 * scale}
    assert PauliSum.from_labels(["Z"]).expectation_z_counts(counts) == pytest.approx(1 / 3)
    assert PauliSum.from_labels(["I"]).expectation_z_counts(counts) == pytest.approx(1)


def test_relative_tolerance_uses_input_maximum_on_each_call():
    op = PauliSum.from_labels(["X", "X", "Y"], [1, 1, 0.15])
    once = op.simplify(atol=0, rtol=0.1)
    assert once.to_labels()[0] == ["X", "Y"]
    assert once.simplify(atol=0, rtol=0.1).to_labels()[0] == ["X"]


@pytest.mark.parametrize("width", [1, 65])
@pytest.mark.parametrize("terms", [127, 128, 256])
def test_fused_matmul_rejects_duplicate_accumulation_overflow(width, terms):
    lhs = PauliSum.from_labels(["X" * width] * terms, [1.7e308] * terms)
    rhs = PauliSum.from_labels(["I" * width])
    with pytest.raises(OverflowError, match="coefficient"):
        lhs.matmul(rhs)


@pytest.mark.parametrize("width", [1, 65])
def test_fused_matmul_rejects_nonfinite_product_before_simplifying(width):
    lhs = PauliSum.from_labels(["X" * width] * 128, [1.7e308] * 128)
    rhs = PauliSum.from_labels(["I" * width], [2])
    with pytest.raises(ValueError, match="finite coefficients"):
        lhs.matmul(rhs)

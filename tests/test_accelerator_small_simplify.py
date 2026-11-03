"""Fused small-operator simplify on CUDA and HIP equals CPU simplify bit for bit.

Operators of one to four packed words (up to 256 qubits) with at most
small_simplify_capacity(words) terms (1,024 for one or two words, 512 for three
or four) run as one fused launch that sums each duplicate run in input order,
so coefficients match CPU simplify exactly, signed zeros included. Survivor
sets match too, except where a coefficient lies within the device hypot's
few-ulp error of the drop threshold; random inputs here never do. The
benchmark switch WOLFGANG_<BACKEND>_BENCH_SMALL_SIMPLIFY=off keeps the
multi-launch path, whose tree-ordered sums match only to rounding.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import wolfgang_quantum
import wolfgang_quantum._wolfgang_core as core

BACKENDS = ("cuda", "hip")


def _require(backend: str) -> None:
    report = dict(getattr(core, f"_{backend}_status")())
    if not report.get("built") or not report.get("runtime_available"):
        pytest.skip(str(report.get("skip_reason", f"{backend} is unavailable")))


def _capacity(num_qubits: int) -> int:
    return 1024 if (num_qubits + 63) // 64 <= 2 else 512


def _operator(num_qubits: int, num_terms: int, duplicates: float, seed: int) -> wolfgang_quantum.PauliSum:
    rng = np.random.default_rng(seed)
    unique = max(1, int(num_terms * (1.0 - duplicates)))
    letters = np.asarray(list("IXYZ"), dtype="<U1")
    chars = letters[rng.integers(0, 4, size=(unique, num_qubits))]
    base = np.ascontiguousarray(chars).view(f"<U{num_qubits}").ravel()
    labels = np.concatenate([base, base[rng.integers(0, unique, num_terms - unique)]]).tolist()
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    return wolfgang_quantum.PauliSum.from_labels(labels, coeffs)


def _assert_bit_identical(actual: wolfgang_quantum.PauliSum, expected: wolfgang_quantum.PauliSum) -> None:
    assert actual == expected
    assert np.array_equal(actual.to_arrays()[2].view(np.uint64), expected.to_arrays()[2].view(np.uint64))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("num_qubits", [20, 64, 100, 128, 130, 200, 256])
@pytest.mark.parametrize("which", ["two", "small", "capacity"])
@pytest.mark.parametrize("duplicates", [0.0, 0.6, 0.97])
def test_small_simplify_is_bit_identical_to_cpu(backend: str, num_qubits: int, which: str, duplicates: float) -> None:
    _require(backend)
    num_terms = {"two": 2, "small": 129, "capacity": _capacity(num_qubits)}[which]
    op = _operator(num_qubits, num_terms, duplicates, seed=num_qubits * 31 + num_terms)
    device = op.to_device(backend=backend)
    for kwargs in ({}, {"atol": 0.5}, {"atol": 0.0, "rtol": 0.2}, {"atol": 0.05, "rtol": 0.2}):
        _assert_bit_identical(device.simplify(**kwargs).to_host(), op.simplify(**kwargs))


@pytest.mark.parametrize("backend", BACKENDS)
def test_small_simplify_threshold_is_not_fma_contracted(backend: str) -> None:
    # atol + rtol * max computed as one FMA keeps the 0.278... term's threshold
    # an ulp higher than CPU's separately rounded threshold and drops it.
    _require(backend)
    op = wolfgang_quantum.PauliSum.from_labels(["X", "Z"], [1.3927415900879758, 0.2785483180185952])
    expected = op.simplify(rtol=0.2)
    assert expected.num_terms == 2
    _assert_bit_identical(op.to_device(backend=backend).simplify(rtol=0.2).to_host(), expected)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("num_qubits", [64, 128, 256])
def test_small_simplify_all_y_key_equal_to_sentinel(backend: str, num_qubits: int) -> None:
    # An all-Y term at a multiple of 64 qubits packs to all-ones words, the
    # padding sentinel's key; the input-index tie-break must keep it.
    _require(backend)
    labels = ["Y" * num_qubits, "X" * num_qubits, "Y" * num_qubits] * 20 + ["Y" * num_qubits]
    coeffs = np.arange(1, len(labels) + 1, dtype=np.float64) * (1.0 - 0.5j)
    op = wolfgang_quantum.PauliSum.from_labels(labels, coeffs)
    _assert_bit_identical(op.to_device(backend=backend).simplify().to_host(), op.simplify())


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("num_qubits", [24, 200])
def test_small_simplify_capacity_boundary(backend: str, num_qubits: int) -> None:
    # One term past the capacity takes the multi-launch path.
    _require(backend)
    for num_terms in (_capacity(num_qubits), _capacity(num_qubits) + 1):
        op = _operator(num_qubits, num_terms, 0.5, seed=num_terms)
        actual = op.to_device(backend=backend).simplify().to_host()
        expected = op.simplify()
        assert list(actual.to_labels()[0]) == list(expected.to_labels()[0])
        np.testing.assert_allclose(actual.to_arrays()[2], expected.to_arrays()[2], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("num_qubits", [24, 100, 200])
def test_small_simplify_long_runs_and_signed_zeros(backend: str, num_qubits: int) -> None:
    # One run fills most of the block; input-order sums make the result exact,
    # where a tree-ordered reduction would differ in the last bits.
    _require(backend)
    rng = np.random.default_rng(num_qubits)
    count = _capacity(num_qubits)
    labels = ["Y" * num_qubits] * (count - 2) + ["X" * num_qubits, "Z" * num_qubits]
    coeffs = rng.normal(size=count) + 1j * rng.normal(size=count)
    op = wolfgang_quantum.PauliSum.from_labels(labels, coeffs)
    _assert_bit_identical(op.to_device(backend=backend).simplify().to_host(), op.simplify())
    zeros = wolfgang_quantum.PauliSum.from_labels(
        ["X" * num_qubits, "X" * num_qubits, "Z" * num_qubits], [complex(-0.0, 1.0), complex(-0.0, 2.0), complex(-0.0, 1.0)]
    )
    expected = zeros.simplify(atol=0.0)
    assert np.signbit(expected.to_arrays()[2].real).all()
    _assert_bit_identical(zeros.to_device(backend=backend).simplify(atol=0.0).to_host(), expected)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("num_qubits", [24, 130])
def test_small_simplify_errors_and_cancellation(backend: str, num_qubits: int) -> None:
    _require(backend)
    op = _operator(num_qubits, 300, 0.5, seed=5)
    assert (op + (-op)).to_device(backend=backend).simplify().num_terms == 0
    x, z, coeffs = (array.copy() for array in op.to_arrays())
    coeffs[17] = complex(0.0, math.inf)
    bad = wolfgang_quantum.PauliSum.from_arrays(x, z, coeffs, num_qubits).to_device(backend=backend)
    with pytest.raises(ValueError, match="finite"):
        bad.simplify()
    label = "X" * num_qubits
    overflow = wolfgang_quantum.PauliSum.from_labels(
        ["Z" * num_qubits] * 10 + [label, label], np.concatenate([np.ones(10), [1.7e308, 1.7e308]])
    ).to_device(backend=backend)
    with pytest.raises(OverflowError):
        overflow.simplify()
    huge = wolfgang_quantum.PauliSum.from_labels([label, "Z" * num_qubits], [complex(1.5e308, 1.5e308), 1.0])
    for kwargs in ({}, {"atol": 1e-12, "rtol": 1e-300}):
        _assert_bit_identical(huge.to_device(backend=backend).simplify(**kwargs).to_host(), huge.simplify(**kwargs))


@pytest.mark.parametrize("backend", BACKENDS)
def test_small_simplify_switch_matches_multi_launch_path(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    _require(backend)
    variable = f"WOLFGANG_{backend.upper()}_BENCH_SMALL_SIMPLIFY"
    op = _operator(60, 700, 0.5, seed=11)
    device = op.to_device(backend=backend)
    fused = device.simplify().to_host()
    monkeypatch.setenv(variable, "off")
    multi = device.simplify().to_host()
    assert list(fused.to_labels()[0]) == list(multi.to_labels()[0])
    np.testing.assert_allclose(fused.to_arrays()[2], multi.to_arrays()[2], rtol=1e-12, atol=1e-12)
    monkeypatch.setenv(variable, "bogus")
    with pytest.raises(ValueError, match="SMALL_SIMPLIFY"):
        device.simplify()

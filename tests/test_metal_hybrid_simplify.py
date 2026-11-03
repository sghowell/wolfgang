"""Metal hybrid simplify (GPU key sort, CPU float64 reduction) equals CPU simplify.

From the hybrid threshold, one-word Metal simplify sorts on the GPU and sums
duplicate runs in input order on the CPU, so results are bit-identical to CPU
simplify. The benchmark selector WOLFGANG_METAL_BENCH_SIMPLIFY=host_bridge keeps
the previous host bridge for comparison.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import wolfgang_quantum
import wolfgang_quantum._wolfgang_core as core

HYBRID_MIN_TERMS = 196_608


def _require_metal() -> None:
    status = dict(core._metal_status())
    if not status.get("built") or not status.get("runtime_available"):
        pytest.skip(str(status.get("skip_reason", "Metal is unavailable")))


def _operator(num_qubits: int, num_terms: int, duplicates: float, seed: int) -> wolfgang_quantum.PauliSum:
    rng = np.random.default_rng(seed)
    unique = max(1, int(num_terms * (1.0 - duplicates)))
    letters = np.asarray(list("IXYZ"), dtype="<U1")
    chars = letters[rng.integers(0, 4, size=(unique, num_qubits))]
    base = np.ascontiguousarray(chars).view(f"<U{num_qubits}").ravel()
    labels = np.concatenate([base, base[rng.integers(0, unique, num_terms - unique)]]).tolist()
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    return wolfgang_quantum.PauliSum.from_labels(labels, coeffs)


def _assert_bit_identical(actual, expected) -> None:
    # == treats 0.0 and -0.0 as equal; compare raw coefficient bits too.
    assert actual == expected
    assert np.array_equal(actual.to_arrays()[2].view(np.uint64), expected.to_arrays()[2].view(np.uint64))


@pytest.mark.parametrize("num_qubits", [5, 33, 64])
@pytest.mark.parametrize("duplicates", [0.0, 0.9])
def test_hybrid_simplify_is_bit_identical_to_cpu(num_qubits: int, duplicates: float) -> None:
    _require_metal()
    op = _operator(num_qubits, HYBRID_MIN_TERMS + 17, duplicates, seed=num_qubits)
    device = op.to_device(backend="metal")
    for kwargs in ({}, {"atol": 0.5}, {"atol": 0.0, "rtol": 0.2}):
        _assert_bit_identical(device.simplify(**kwargs).to_host(), op.simplify(**kwargs))
    negated = -op  # negation produces -0.0 components
    _assert_bit_identical(negated.to_device(backend="metal").simplify().to_host(), negated.simplify())


def test_hybrid_simplify_matches_host_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_metal()
    op = _operator(40, HYBRID_MIN_TERMS, 0.5, seed=7)
    device = op.to_device(backend="metal")
    hybrid = device.simplify().to_host()
    monkeypatch.setenv("WOLFGANG_METAL_BENCH_SIMPLIFY", "host_bridge")
    assert device.simplify().to_host() == hybrid


def test_hybrid_simplify_errors_match_cpu() -> None:
    _require_metal()
    base = _operator(10, HYBRID_MIN_TERMS, 0.0, seed=3)
    x, z, coeffs = (array.copy() for array in base.to_arrays())
    coeffs[5] = complex(math.inf, 0.0)
    non_finite = wolfgang_quantum.PauliSum.from_arrays(x, z, coeffs, 10)
    with pytest.raises(ValueError, match="finite"):
        non_finite.simplify()
    with pytest.raises(ValueError, match="finite"):
        non_finite.to_device(backend="metal").simplify()

    coeffs[:] = 1.7e308
    z[:] = 0
    x[:] = 1  # every term is the same Pauli, so the sum overflows
    overflow = wolfgang_quantum.PauliSum.from_arrays(x, z, coeffs, 10)
    with pytest.raises(OverflowError):
        overflow.simplify()
    with pytest.raises(OverflowError):
        overflow.to_device(backend="metal").simplify()


def test_hybrid_simplify_full_cancellation() -> None:
    _require_metal()
    op = _operator(20, HYBRID_MIN_TERMS // 2, 0.0, seed=11)
    cancelled = (op + (-op)).to_device(backend="metal").simplify()
    assert cancelled.num_terms == 0
    assert cancelled.num_qubits == 20

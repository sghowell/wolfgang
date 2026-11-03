"""Device simplify follows the CPU numeric contract on every accelerator.

`docs/architecture/semantic_contracts.md` defines simplify on the CPU: the
drop rule, `ValueError` for non-finite input components, `OverflowError` when a
duplicate sum overflows, and safe handling of finite components whose complex
magnitude exceeds the largest double. Each available backend (CUDA, HIP, Metal)
must match it. CUDA and HIP sum long duplicate runs in a different association
order, so coefficients are compared to rounding, not bit for bit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import wolfgang_quantum
import wolfgang_quantum._wolfgang_core as core

BACKENDS = ("cuda", "hip", "metal")


def _require(backend: str) -> None:
    status = getattr(core, f"_{backend}_status", None)
    if status is None:
        pytest.skip(f"{backend} status is not exposed")
    report = dict(status())
    if not report.get("built") or not report.get("runtime_available"):
        pytest.skip(str(report.get("skip_reason", f"{backend} is unavailable")))


def _operator(num_qubits: int, num_terms: int, duplicates: float, seed: int) -> wolfgang_quantum.PauliSum:
    rng = np.random.default_rng(seed)
    unique = max(1, int(num_terms * (1.0 - duplicates)))
    letters = np.asarray(list("IXYZ"), dtype="<U1")
    chars = letters[rng.integers(0, 4, size=(unique, num_qubits))]
    base = np.ascontiguousarray(chars).view(f"<U{num_qubits}").ravel()
    labels = np.concatenate([base, base[rng.integers(0, unique, num_terms - unique)]]).tolist()
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    return wolfgang_quantum.PauliSum.from_labels(labels, coeffs)


def _assert_matches_cpu(backend: str, op: wolfgang_quantum.PauliSum, **kwargs: float) -> None:
    expected = op.simplify(**kwargs)
    actual = op.to_device(backend=backend).simplify(**kwargs).to_host()
    assert list(actual.to_labels()[0]) == list(expected.to_labels()[0])
    np.testing.assert_allclose(actual.to_arrays()[2], expected.to_arrays()[2], rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("num_qubits", [5, 40, 64, 100, 130, 300])
@pytest.mark.parametrize("num_terms", [1, 2, 1000, 20_000])
@pytest.mark.parametrize("duplicates", [0.0, 0.9])
def test_device_simplify_matches_cpu(backend: str, num_qubits: int, num_terms: int, duplicates: float) -> None:
    _require(backend)
    op = _operator(num_qubits, num_terms, duplicates, seed=num_qubits * 7 + num_terms)
    for kwargs in ({}, {"atol": 0.5}, {"atol": 0.0, "rtol": 0.2}):
        _assert_matches_cpu(backend, op, **kwargs)


@pytest.mark.parametrize("backend", BACKENDS)
def test_device_simplify_identity_only_operator(backend: str) -> None:
    # Zero qubits pack to zero words: every term is the identity.
    _require(backend)
    op = wolfgang_quantum.PauliSum.from_labels(["", "", ""], [1.0 + 2.0j, -0.5, 0.25j])
    _assert_matches_cpu(backend, op)
    cancelling = wolfgang_quantum.PauliSum.from_labels(["", ""], [1.0, -1.0])
    assert cancelling.to_device(backend=backend).simplify().to_host().num_terms == 0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("num_qubits", [0, 24, 100, 130])
@pytest.mark.parametrize("bad", [complex(math.inf, 0.0), complex(0.0, -math.inf), complex(math.nan, 1.0)])
def test_device_simplify_rejects_non_finite_coefficients(backend: str, num_qubits: int, bad: complex) -> None:
    _require(backend)
    base = _operator(num_qubits, 5000, 0.5, seed=num_qubits + 1) if num_qubits else (
        wolfgang_quantum.PauliSum.from_labels([""] * 50, np.ones(50))
    )
    x, z, coeffs = (array.copy() for array in base.to_arrays())
    coeffs[len(coeffs) // 2] = bad
    op = wolfgang_quantum.PauliSum.from_arrays(x, z, coeffs, num_qubits)
    with pytest.raises(ValueError, match="finite"):
        op.simplify()
    with pytest.raises(ValueError, match="finite"):
        op.to_device(backend=backend).simplify()
    single = wolfgang_quantum.PauliSum.from_arrays(x[:1], z[:1], coeffs[len(coeffs) // 2 : len(coeffs) // 2 + 1], num_qubits)
    with pytest.raises(ValueError, match="finite"):
        single.to_device(backend=backend).simplify()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("num_qubits", [0, 24, 100, 130])
def test_device_simplify_raises_on_duplicate_sum_overflow(backend: str, num_qubits: int) -> None:
    _require(backend)
    label = "X" * num_qubits
    others = ["Z" * num_qubits] * 100 if num_qubits else []
    op = wolfgang_quantum.PauliSum.from_labels(
        others + [label, label], np.concatenate([np.ones(len(others)), [1.7e308, 1.7e308]])
    )
    with pytest.raises(OverflowError):
        op.simplify()
    with pytest.raises(OverflowError):
        op.to_device(backend=backend).simplify()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("num_qubits", [24, 60, 100, 130])
def test_device_simplify_keeps_finite_coefficients_with_overflowing_magnitude(backend: str, num_qubits: int) -> None:
    # |c| exceeds DBL_MAX although both components are finite. At rtol=0 the CPU
    # threshold is atol itself; with rtol > 0 it is compared in half units.
    _require(backend)
    huge = complex(1.5e308, 1.5e308)
    a = "X" * num_qubits
    b = "Z" * num_qubits
    _assert_matches_cpu(backend, wolfgang_quantum.PauliSum.from_labels([a], [huge]))
    mixed = wolfgang_quantum.PauliSum.from_labels([a, b, b], [huge, 1.0, 2.0])
    _assert_matches_cpu(backend, mixed)
    _assert_matches_cpu(backend, mixed, atol=1e-12, rtol=1e-300)
    assert mixed.to_device(backend=backend).simplify(rtol=1e-300).to_host().num_terms == 1

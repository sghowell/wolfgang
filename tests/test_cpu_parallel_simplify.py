"""oneTBB simplify must equal scalar simplify bit for bit.

Both paths sort terms by (packed key, input index) and sum each duplicate run in
input order, so structural equality (identical terms, order and coefficient
bits) is the contract, not a tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
import wolfgang_quantum
import wolfgang_quantum._wolfgang_core as core

SELECTOR = "WOLFGANG_CPU_BACKEND"


def _tbb_simplify_compiled() -> bool:
    return "simplify" in core._build_info()["optimized_cpu_kernels"].get("tbb", [])


def _operator(rng: np.random.Generator, num_qubits: int, num_terms: int, duplicates: float):
    unique = max(1, int(num_terms * (1.0 - duplicates)))
    letters = np.asarray(list("IXYZ"), dtype="<U1")
    chars = letters[rng.integers(0, 4, size=(unique, num_qubits))]
    base = np.ascontiguousarray(chars).view(f"<U{num_qubits}").ravel()
    labels = np.concatenate([base, base[rng.integers(0, unique, num_terms - unique)]]).tolist()
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    return wolfgang_quantum.PauliSum.from_labels(labels, coeffs)


def _simplify_with(monkeypatch: pytest.MonkeyPatch, selector: str, op, **kwargs):
    monkeypatch.setenv(SELECTOR, selector)
    try:
        return op.simplify(**kwargs)
    finally:
        monkeypatch.delenv(SELECTOR, raising=False)


@pytest.mark.parametrize("num_qubits", [3, 24, 64, 100, 130])
@pytest.mark.parametrize("num_terms", [1, 1000, 70_000])
@pytest.mark.parametrize("duplicates", [0.0, 0.5, 0.95])
def test_tbb_simplify_matches_scalar_bit_for_bit(
    monkeypatch: pytest.MonkeyPatch, num_qubits: int, num_terms: int, duplicates: float
) -> None:
    if not _tbb_simplify_compiled():
        pytest.skip("oneTBB simplify is not compiled in this build")
    rng = np.random.default_rng(num_qubits * 1_000_003 + num_terms + int(duplicates * 100))
    op = _operator(rng, num_qubits, num_terms, duplicates)
    scalar = _simplify_with(monkeypatch, "scalar", op)
    parallel = _simplify_with(monkeypatch, "tbb", op)
    assert parallel == scalar
    assert _simplify_with(monkeypatch, "auto", op) == scalar


def test_tbb_simplify_drop_rules_and_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _tbb_simplify_compiled():
        pytest.skip("oneTBB simplify is not compiled in this build")
    rng = np.random.default_rng(7)
    op = _operator(rng, 40, 50_000, 0.9)
    cancelling = op + (-op)
    for kwargs in ({}, {"atol": 1e-3}, {"atol": 0.0, "rtol": 0.25}):
        assert _simplify_with(monkeypatch, "tbb", op, **kwargs) == _simplify_with(
            monkeypatch, "scalar", op, **kwargs
        )
    assert _simplify_with(monkeypatch, "tbb", cancelling).num_terms == 0
    huge = wolfgang_quantum.PauliSum.from_labels(["XZ", "ZX", "ZX"], [1.5e308 + 1.5e308j, 1.0, 2.0])
    for kwargs in ({}, {"rtol": 1e-300}):
        assert _simplify_with(monkeypatch, "tbb", huge, **kwargs) == _simplify_with(
            monkeypatch, "scalar", huge, **kwargs
        )
    overflow = wolfgang_quantum.PauliSum.from_labels(["X", "X"], [1.7e308, 1.7e308])
    with pytest.raises(OverflowError):
        _simplify_with(monkeypatch, "tbb", overflow)


def test_forced_simd_backends_still_reject_simplify(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled = core._build_info()["compiled_cpu_backends"]
    simd = next((name for name in ("neon", "avx2", "avx512") if name in compiled), None)
    if simd is None:
        pytest.skip("no SIMD CPU backend is compiled")
    op = wolfgang_quantum.PauliSum.from_labels(["X"], [1.0])
    monkeypatch.setenv(SELECTOR, simd)
    with pytest.raises(RuntimeError, match="simplify"):
        op.simplify()

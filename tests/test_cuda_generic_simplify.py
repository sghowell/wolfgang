"""CUDA generic multi-word simplify qualification against the CPU oracle.

Operators wider than 128 qubits bypass the specialized one- and two-word CUDA
key paths and use the generic index-sort reduction. These tests pin that path
to CPU canonical output across widths, sizes, duplicate pressure, exact
cancellation and relative drop thresholds, for both the production parallel
reduction and the retained benchmark-only serial kernel.
"""

from __future__ import annotations

import numpy as np
import pytest
import wolfgang_quantum
import wolfgang_quantum._wolfgang_core as core

GENERIC_SELECTOR = "WOLFGANG_CUDA_BENCH_GENERIC_MULTIWORD_REDUCTION"


def _require_cuda_runtime() -> None:
    status = dict(core._cuda_status())
    if not status["built"] or not status["runtime_available"]:
        pytest.skip(str(status["skip_reason"]))


def _operator(
    rng: np.random.Generator,
    *,
    num_qubits: int,
    num_terms: int,
    duplicate_fraction: float,
    cancel_pairs: int = 0,
) -> wolfgang_quantum.PauliSum:
    if num_terms == 0:
        return wolfgang_quantum.PauliSum.empty(num_qubits)
    alphabet = np.asarray(list("IXYZ"))
    unique_count = max(1, int(round(num_terms * (1.0 - duplicate_fraction))))
    base = ["".join(rng.choice(alphabet, size=num_qubits).tolist()) for _ in range(unique_count)]
    labels = base + [base[int(index)] for index in rng.integers(0, unique_count, num_terms - unique_count)]
    coeffs = (rng.normal(size=len(labels)) + 1j * rng.normal(size=len(labels))).tolist()
    # Exact negated copies must cancel to zero and be dropped on every backend.
    for pair in range(min(cancel_pairs, unique_count)):
        labels.append(base[pair])
        coeffs.append(-sum(c for label, c in zip(labels[:-1], coeffs, strict=False) if label == base[pair]))
    return wolfgang_quantum.PauliSum.from_labels(labels, np.asarray(coeffs, dtype=np.complex128))


def _assert_matches_cpu(op: wolfgang_quantum.PauliSum, *, atol: float, rtol: float) -> None:
    expected = op.simplify(atol=atol, rtol=rtol)
    actual = op.to_device().simplify(atol=atol, rtol=rtol).to_host()
    actual_labels, actual_coeffs = actual.to_labels()
    expected_labels, expected_coeffs = expected.to_labels()
    assert list(actual_labels) == list(expected_labels)
    np.testing.assert_allclose(actual_coeffs, expected_coeffs, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("strategy", ["", "serial_kernel"])
@pytest.mark.parametrize("num_qubits", [130, 256, 1024])
@pytest.mark.parametrize("num_terms", [0, 1, 128, 4096])
@pytest.mark.parametrize("duplicate_fraction", [0.0, 0.9])
def test_cuda_generic_simplify_matches_cpu(
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
    num_qubits: int,
    num_terms: int,
    duplicate_fraction: float,
) -> None:
    _require_cuda_runtime()
    if strategy:
        monkeypatch.setenv(GENERIC_SELECTOR, strategy)
    else:
        monkeypatch.delenv(GENERIC_SELECTOR, raising=False)
    rng = np.random.default_rng(13_000 + num_qubits + num_terms)
    op = _operator(
        rng,
        num_qubits=num_qubits,
        num_terms=num_terms,
        duplicate_fraction=duplicate_fraction,
        cancel_pairs=4,
    )
    _assert_matches_cpu(op, atol=1.0e-11, rtol=0.0)


@pytest.mark.parametrize("strategy", ["", "serial_kernel"])
def test_cuda_generic_simplify_relative_threshold_and_full_cancellation(
    monkeypatch: pytest.MonkeyPatch, strategy: str
) -> None:
    _require_cuda_runtime()
    if strategy:
        monkeypatch.setenv(GENERIC_SELECTOR, strategy)
    else:
        monkeypatch.delenv(GENERIC_SELECTOR, raising=False)

    wide = 193
    big = "X" * wide
    small = "Z" * wide
    gone = "Y" * wide
    op = wolfgang_quantum.PauliSum.from_labels(
        [big, small, gone, big, gone], [10.0, 1.0e-3, 2.5, 5.0, -2.5]
    )
    # rtol drops `small` relative to the largest input magnitude; `gone` cancels exactly.
    _assert_matches_cpu(op, atol=0.0, rtol=1.0e-3)

    cancelled = wolfgang_quantum.PauliSum.from_labels([big, big], [1.0 + 1.0j, -1.0 - 1.0j])
    assert cancelled.to_device().simplify().to_host().num_terms == 0


def test_cuda_generic_simplify_rejects_unknown_benchmark_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_cuda_runtime()
    monkeypatch.setenv(GENERIC_SELECTOR, "not_a_strategy")
    op = wolfgang_quantum.PauliSum.from_labels(["X" * 130], [1.0]).to_device()
    with pytest.raises(ValueError, match=GENERIC_SELECTOR):
        op.simplify()

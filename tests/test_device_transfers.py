"""Device-to-host transfers return exactly what the device holds.

CUDA `DevicePauliSum.to_host()` copies through a per-thread pinned staging
buffer in batches (src/detail/staged_transfer.hpp), double-buffered once an
operator exceeds half the buffer (8 MiB). HIP stages operators up to 256 KiB
and copies larger ones directly into vectors advised for huge pages. These
round trips cross both sizes, run on several threads (each has its own
buffer, freed at thread exit), and include the empty operator.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import wolfgang_quantum
import wolfgang_quantum._wolfgang_core as core

BACKENDS = ("cuda", "hip")


def _require(backend: str) -> None:
    status = getattr(core, f"_{backend}_status", None)
    if status is None:
        pytest.skip(f"{backend} status is not exposed")
    report = dict(status())
    if not report.get("built") or not report.get("runtime_available"):
        pytest.skip(str(report.get("skip_reason", f"{backend} is unavailable")))


def _operator(num_qubits: int, num_terms: int, seed: int) -> wolfgang_quantum.PauliSum:
    rng = np.random.default_rng(seed)
    words = (num_qubits + 63) // 64
    x = rng.integers(0, 2**63, size=(num_terms, words), dtype=np.uint64)
    z = rng.integers(0, 2**63, size=(num_terms, words), dtype=np.uint64)
    if num_qubits % 64:
        tail = np.uint64((1 << (num_qubits % 64)) - 1)
        x[:, -1] &= tail
        z[:, -1] &= tail
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    return wolfgang_quantum.PauliSum.from_arrays(x, z, coeffs, num_qubits)


def _assert_round_trip(op: wolfgang_quantum.PauliSum, backend: str) -> None:
    back = op.to_device(backend=backend).to_host()
    assert back.num_qubits == op.num_qubits
    assert back.num_terms == op.num_terms
    for actual, expected in zip(back.to_arrays(), op.to_arrays(), strict=True):
        assert np.array_equal(actual, expected)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("num_qubits", "num_terms"),
    [(5, 0), (5, 1), (24, 3), (100, 4096), (130, 65_537), (100, 400_000), (24, 1_200_000)],
)
def test_to_host_round_trips(backend: str, num_qubits: int, num_terms: int) -> None:
    # 400,000 two-word terms (19.2 MB) and 1.2M one-word terms (38.4 MB) take
    # several double-buffered batches; the smaller cases fit in one.
    _require(backend)
    _assert_round_trip(_operator(num_qubits, num_terms, seed=num_terms + num_qubits), backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_to_host_round_trips_on_several_threads(backend: str) -> None:
    _require(backend)
    ops = [_operator(width, terms, seed=width) for width, terms in ((24, 50_000), (100, 300_000), (300, 2_000))]

    def run(index: int) -> None:
        _assert_round_trip(ops[index % len(ops)], backend)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(run, range(12)))


@pytest.mark.parametrize("backend", BACKENDS)
def test_to_host_after_simplify_matches_cpu(backend: str) -> None:
    _require(backend)
    op = _operator(24, 300_000, seed=5)
    expected = op.simplify()
    actual = op.to_device(backend=backend).simplify().to_host()
    assert np.array_equal(actual.to_arrays()[0], expected.to_arrays()[0])
    assert np.array_equal(actual.to_arrays()[1], expected.to_arrays()[1])
    np.testing.assert_allclose(actual.to_arrays()[2], expected.to_arrays()[2], rtol=1e-11, atol=1e-11)

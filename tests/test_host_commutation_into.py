"""Host pairwise commutation written straight into a caller's NumPy buffer.

`PauliSum.commutes_with_into(other, output)` fills a one-dimensional NumPy bool
array of length `self.num_terms * other.num_terms`, like
`DevicePauliSum.commutes_with_into`. `PauliSum.commutes_with` fills its result
the same way, without a zero-filled intermediate or a copy. Every entry is
written, so whatever the buffer held before must not matter.
"""

from __future__ import annotations

import numpy as np
import pytest
import wolfgang_quantum
import wolfgang_quantum._wolfgang_core as core

SELECTORS = ["auto", "scalar"] + [
    name for name in ("tbb", "avx512", "avx2", "neon") if name in core._build_info()["available_cpu_backends"]
]


def _operator(num_qubits: int, num_terms: int, seed: int) -> wolfgang_quantum.PauliSum:
    rng = np.random.default_rng(seed)
    words = (num_qubits + 63) // 64
    x = rng.integers(0, 2**63, size=(num_terms, words), dtype=np.uint64)
    z = rng.integers(0, 2**63, size=(num_terms, words), dtype=np.uint64)
    if num_qubits % 64:
        tail = np.uint64((1 << (num_qubits % 64)) - 1)
        x[:, -1] &= tail
        z[:, -1] &= tail
    return wolfgang_quantum.PauliSum.from_arrays(x, z, np.ones(num_terms, dtype=np.complex128), num_qubits)


@pytest.mark.parametrize("selector", SELECTORS)
@pytest.mark.parametrize("num_qubits", [24, 100, 300])
@pytest.mark.parametrize(("lhs_terms", "rhs_terms"), [(577, 613), (1, 33), (33, 1), (1, 1)])
def test_commutes_with_into_matches_commutes_with(
    monkeypatch: pytest.MonkeyPatch, selector: str, num_qubits: int, lhs_terms: int, rhs_terms: int
) -> None:
    if selector in ("avx512", "avx2", "neon") and num_qubits > 128:
        pytest.skip("SIMD commutation covers one- and two-word operators")
    lhs = _operator(num_qubits, lhs_terms, seed=lhs_terms + num_qubits)
    rhs = _operator(num_qubits, rhs_terms, seed=rhs_terms + 2 * num_qubits)
    monkeypatch.setenv("WOLFGANG_CPU_BACKEND", "scalar")
    expected = np.asarray(lhs.commutes_with(rhs)).reshape(-1)
    monkeypatch.setenv("WOLFGANG_CPU_BACKEND", selector)
    assert np.array_equal(np.asarray(lhs.commutes_with(rhs)).reshape(-1), expected)
    for fill in (False, True):
        output = np.full(lhs_terms * rhs_terms, fill, dtype=np.bool_)
        assert lhs.commutes_with_into(rhs, output) is None
        assert np.array_equal(output, expected), fill


def test_commutes_with_into_validates_the_output() -> None:
    op = _operator(24, 8, seed=1)
    with pytest.raises(TypeError, match="dtype must be bool"):
        op.commutes_with_into(op, np.zeros(64, dtype=np.uint8))
    with pytest.raises(ValueError, match="1-dimensional"):
        op.commutes_with_into(op, np.zeros((8, 8), dtype=np.bool_))
    with pytest.raises(TypeError, match="C-contiguous"):
        op.commutes_with_into(op, np.zeros(128, dtype=np.bool_)[::2])
    with pytest.raises(ValueError, match="does not match"):
        op.commutes_with_into(op, np.zeros(63, dtype=np.bool_))
    read_only = np.zeros(64, dtype=np.bool_)
    read_only.setflags(write=False)
    with pytest.raises((TypeError, ValueError, BufferError)):
        op.commutes_with_into(op, read_only)
    other = _operator(25, 8, seed=2)
    with pytest.raises(ValueError):
        op.commutes_with_into(other, np.zeros(64, dtype=np.bool_))


def test_commutes_with_into_enforces_the_dense_output_guardrail() -> None:
    op = _operator(24, 8, seed=3)
    output = np.zeros(64, dtype=np.bool_)
    with pytest.raises(ValueError, match="max_commutation_matrix_entries"):
        op.commutes_with_into(op, output, max_commutation_matrix_entries=63)
    assert not output.any()  # rejected before writing

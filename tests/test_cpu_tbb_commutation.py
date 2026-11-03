"""oneTBB pairwise commutation matches the scalar kernel.

Each oneTBB chunk runs the widest SIMD row kernel the build and CPU support for
one- and two-word operators, and the scalar row kernel for wider ones. These
tests compare forced oneTBB results with forced scalar at one, two and five
words, with term counts that leave partial SIMD lanes and uneven chunks, under
several thread caps.
"""

from __future__ import annotations

import numpy as np
import pytest
import wolfgang_quantum
import wolfgang_quantum._wolfgang_core as core


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


@pytest.mark.parametrize("num_qubits", [24, 64, 100, 300])
@pytest.mark.parametrize(("lhs_terms", "rhs_terms"), [(577, 1023), (1, 9), (1500, 7)])
def test_forced_onetbb_commutation_matches_scalar(
    monkeypatch: pytest.MonkeyPatch, num_qubits: int, lhs_terms: int, rhs_terms: int
) -> None:
    if "tbb" not in core._build_info()["available_cpu_backends"]:
        pytest.skip("oneTBB is not available")
    lhs = _operator(num_qubits, lhs_terms, seed=lhs_terms + num_qubits)
    rhs = _operator(num_qubits, rhs_terms, seed=rhs_terms + 3 * num_qubits)
    monkeypatch.setenv("WOLFGANG_CPU_BACKEND", "scalar")
    expected = np.asarray(lhs.commutes_with(rhs))
    monkeypatch.setenv("WOLFGANG_CPU_BACKEND", "tbb")
    try:
        for cap in (1, 2, 3):
            wolfgang_quantum.set_num_threads(cap)
            assert np.array_equal(np.asarray(lhs.commutes_with(rhs)), expected), cap
    finally:
        wolfgang_quantum.set_num_threads(1_000_000)  # a cap above the hardware means no cap

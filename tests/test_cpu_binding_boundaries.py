"""Python ownership and scheduling checks for native-only computation scopes."""
from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from wolfgang_quantum import PauliSum


def test_coefficient_export_owns_independent_complex128_storage() -> None:
    coefficients = np.arange(4096, dtype=np.float64) * (1 + 2j)
    op = PauliSum.from_labels(["X"] * len(coefficients), coefficients)
    exported = op.to_labels()[1]
    assert exported.dtype == np.complex128
    assert exported.flags.owndata
    np.testing.assert_array_equal(exported, coefficients)
    exported[:] = 0
    np.testing.assert_array_equal(op.to_labels()[1], coefficients)
    del op
    exported[:] = 3j
    assert (exported == 3j).all()


def test_simplify_allows_another_python_thread_to_run() -> None:
    op = PauliSum.from_labels(["X"] * 250000, [1] * 250000)
    ready, start, progressed = threading.Event(), threading.Event(), threading.Event()

    def worker():
        ready.set()
        start.wait()
        progressed.set()

    thread = threading.Thread(target=worker)
    thread.start()
    ready.wait()
    previous = sys.getswitchinterval()
    try:
        # Suppress interpreter timeslicing: the native call must release GIL.
        sys.setswitchinterval(60)
        start.set()
        for _ in range(32):
            result = op.simplify()
            if progressed.is_set():
                break
        assert progressed.is_set()
        assert result.to_labels()[1][0] == 250000
    finally:
        sys.setswitchinterval(previous)
        start.set()
        thread.join(timeout=5)


def test_concurrent_readonly_operations_match_serial_results() -> None:
    op = PauliSum.from_labels(["XX", "YZ", "II", "XX"], [1, 2j, 0.3, -0.5])
    psi = np.array([0.5, 0.5j, -0.5, -0.5j])

    def compute(_):
        return op.simplify().to_labels()[1], op.expectation_statevector(psi)

    expected_coefficients, expected_expectation = compute(0)
    with ThreadPoolExecutor(max_workers=4) as executor:
        for coefficients, expectation in executor.map(compute, range(40)):
            np.testing.assert_array_equal(coefficients, expected_coefficients)
            assert expectation == expected_expectation


def test_forced_optimized_backend_does_not_silently_use_streaming_scalar(monkeypatch) -> None:
    import pytest
    import wolfgang_quantum as wg

    optimized = [name for name in wg.capabilities().cpu.available if name != "scalar"]
    if not optimized:
        pytest.skip("no optimized backend available")
    op = PauliSum.from_labels(["XX", "ZZ"])
    for backend in optimized:
        monkeypatch.setenv("WOLFGANG_CPU_BACKEND", backend)
        with pytest.raises(RuntimeError, match="scalar CPU coverage"):
            op.group_commuting(mode="full", max_terms_for_graph=0)

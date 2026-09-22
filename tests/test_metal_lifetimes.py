"""Autorelease diagnostics cover synchronous calls from ordinary Python threads."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest
import wolfgang_quantum as wg


def test_metal_calls_supply_autorelease_pools() -> None:
    if not wg.metal_available():
        pytest.skip("Metal runtime unavailable")
    probe = r'''
import gc
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from wolfgang_quantum import PauliSum

def exercise():
    host = PauliSum.from_labels(["XI", "IZ", "YZ", "II"])
    expected = host.commutes_with(host)
    device = host.to_device(backend="metal")
    matrix = device.commutes_with_device(device)
    for mode in ("cpu", "gpu", "gpu_parallel_total"):
        os.environ["WOLFGANG_EXPERIMENTAL_METAL_COMPACT_CONSUMER"] = mode
        for _ in range(4):
            np.testing.assert_array_equal(device.commutes_with(device), expected)
            device.commutes_with_device(device, output=matrix)
            np.testing.assert_array_equal(matrix.to_host(), expected)
            assert matrix.count_commuting() == int(expected.sum())
            np.testing.assert_array_equal(matrix.count_commuting(axis=1), expected.sum(axis=1))
            np.testing.assert_array_equal(matrix.count_commuting(axis=0), expected.sum(axis=0))
            assert device.simplify().to_host().to_labels()[0] == host.simplify().to_labels()[0]
    del matrix, device
    gc.collect()

exercise()
with ThreadPoolExecutor(max_workers=1) as executor:
    executor.submit(exercise).result()
'''
    env = os.environ.copy()
    env["OBJC_DEBUG_MISSING_POOLS"] = "YES"
    result = subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True,
        check=False, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MISSING POOLS" not in result.stderr, result.stderr

"""HIP keeps operation scratch in a per-thread block cache.

On an MI300X a hipMalloc/hipFree pair for the simplify pipeline's
multi-megabyte scratch costs up to about 100 us, while ROCm caches small blocks
itself (docs/benchmarks/reports/hip_device_counts_mi300x_2026-11-03.md). Each
thread therefore keeps its freed scratch and rocThrust temporaries for reuse
(src/hip/pool_allocator.hip.hpp). These tests pin the cache's contract: blocks
are reused, requests above the limit are not retained, a thread's blocks are
freed when it exits, and the optional pool (which caches by itself) bypasses
the cache.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import wolfgang_quantum
import wolfgang_quantum._wolfgang_core as core

ROOT = Path(__file__).resolve().parents[1]


def _require() -> None:
    status = dict(core._hip_status())
    if not status["built"] or not status["runtime_available"]:
        pytest.skip(str(status["skip_reason"]))
    if not hasattr(core, "_hip_thread_cache_status_for_testing"):
        pytest.skip("private HIP thread-cache hook requires internal bindings")


def _counters() -> dict[str, int]:
    gc.collect()
    return dict(core._hip_thread_cache_status_for_testing())


def _operator(num_qubits: int, num_terms: int, seed: int) -> wolfgang_quantum.PauliSum:
    # Random packed words with a 50% duplicate share, built from arrays so
    # multi-million-term operators stay quick to create.
    rng = np.random.default_rng(seed)
    unique = num_terms // 2
    mask = (1 << num_qubits) - 1
    x = rng.integers(0, mask, size=unique, dtype=np.uint64, endpoint=True)
    z = rng.integers(0, mask, size=unique, dtype=np.uint64, endpoint=True)
    picks = rng.integers(0, unique, size=num_terms - unique)
    x = np.concatenate([x, x[picks]]).reshape(-1, 1)
    z = np.concatenate([z, z[picks]]).reshape(-1, 1)
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    return wolfgang_quantum.PauliSum.from_arrays(x, z, coeffs, num_qubits)


def _same(actual: wolfgang_quantum.PauliSum, expected: wolfgang_quantum.PauliSum) -> None:
    ax, az, ac = actual.to_arrays()
    ex, ez, ec = expected.to_arrays()
    assert np.array_equal(ax, ex) and np.array_equal(az, ez)
    np.testing.assert_allclose(ac, ec, rtol=1e-11, atol=1e-11)


def test_simplify_scratch_is_reused_across_calls() -> None:
    _require()
    op = _operator(24, 50_000, seed=1)
    device = op.to_device(backend="hip")
    expected = op.simplify()
    _same(device.simplify().to_host(), expected)
    first = _counters()
    assert first["cached_bytes"] > 0
    for _ in range(3):
        _same(device.simplify().to_host(), expected)
    again = _counters()
    assert again["block_allocations"] == first["block_allocations"]
    assert again["cached_bytes"] == first["cached_bytes"]


def test_scratch_above_the_limit_is_not_retained() -> None:
    _require()
    limit = _counters()["limit_bytes"]
    # The packed-key pipeline holds about 56 bytes per term (keys, sorted keys,
    # sorted values, run keys, run sums), so this request exceeds the limit.
    num_terms = limit // 56 + 200_000
    op = _operator(24, num_terms, seed=2)
    before = _counters()
    result = op.to_device(backend="hip").simplify().to_host()
    after = _counters()
    assert after["cached_bytes"] == before["cached_bytes"]
    assert after["block_allocations"] == before["block_allocations"]
    _same(result, op.simplify())


def test_thread_exit_frees_its_blocks() -> None:
    _require()
    op = _operator(24, 60_000, seed=3)
    device = op.to_device(backend="hip")
    expected = op.simplify()
    before = _counters()
    seen: dict[str, object] = {}

    def run() -> None:
        seen["result"] = device.simplify().to_host()
        seen["during"] = dict(core._hip_thread_cache_status_for_testing())

    worker = threading.Thread(target=run)
    worker.start()
    worker.join()
    _same(seen["result"], expected)
    assert seen["during"]["cached_bytes"] > before["cached_bytes"]
    # join() returns once Python releases the thread state; the C++
    # thread-exit destructors that free the blocks run just after.
    deadline = time.monotonic() + 5.0
    while _counters()["cached_bytes"] != before["cached_bytes"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert _counters()["cached_bytes"] == before["cached_bytes"]


def test_optional_pool_bypasses_the_cache() -> None:
    _require()
    script = (
        "import json, numpy as np, wolfgang_quantum, wolfgang_quantum._wolfgang_core as core\n"
        "rng = np.random.default_rng(4)\n"
        "x = rng.integers(0, 1 << 24, size=40000, dtype=np.uint64).reshape(-1, 1)\n"
        "z = rng.integers(0, 1 << 24, size=40000, dtype=np.uint64).reshape(-1, 1)\n"
        "op = wolfgang_quantum.PauliSum.from_arrays(x, z, rng.normal(size=40000) + 0j, 24)\n"
        "op.to_device(backend='hip').simplify().to_host()\n"
        "print(json.dumps(dict(core._hip_thread_cache_status_for_testing())))\n"
    )
    env = {**os.environ, "WOLFGANG_HIP_BENCH_ENABLE_POOL": "1"}
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True, text=True, check=True
    )
    status = json.loads(completed.stdout.strip().splitlines()[-1])
    assert status["cached_bytes"] == 0
    assert status["block_allocations"] == 0

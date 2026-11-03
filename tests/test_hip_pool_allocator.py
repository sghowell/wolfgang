"""Wolfgang's HIP allocator: hipMalloc by default, optional pool without leaks.

On ROCm, plain hipMalloc/hipFree measured faster than the stream-ordered pool
for Wolfgang's small operations, so the pool is opt-in
(WOLFGANG_HIP_BENCH_ENABLE_POOL=1, benchmark-only). The default must not use
it; with the pool enabled, every public DevicePauliSum operation must return
its used bytes to the starting value once results are released; and both
allocators must produce the same results.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import wolfgang_quantum
import wolfgang_quantum._wolfgang_core as core

ROOT = Path(__file__).resolve().parents[1]
ENABLE = "WOLFGANG_HIP_BENCH_ENABLE_POOL"


def _require_hip_pool_hook() -> None:
    status = dict(core._hip_status())
    if not status["built"] or not status["runtime_available"]:
        pytest.skip(str(status["skip_reason"]))
    if not hasattr(core, "_hip_pool_status_for_testing"):
        pytest.skip("private HIP pool hook requires internal bindings")


def _used_bytes() -> int:
    gc.collect()
    return int(core._hip_pool_status_for_testing(0)["used_bytes"])


def _workload() -> list[tuple[str, object]]:
    rng = np.random.default_rng(31)
    results: list[tuple[str, object]] = []
    for width in (12, 60, 100, 130):
        labels = ["".join(rng.choice(list("IXYZ"), size=width)) for _ in range(300)]
        labels += labels[:150]
        op = wolfgang_quantum.PauliSum.from_labels(labels, rng.normal(size=len(labels)) + 0j)
        device = op.to_device(backend="hip")
        results.append((f"simplify_{width}", device.simplify().to_host().to_labels()))
        results.append((f"single_{width}", op[0:1].to_device(backend="hip").simplify().to_host().to_labels()))
        results.append((f"matmul_{width}", device.matmul(device).to_host().num_terms))
        results.append((f"commutes_{width}", int(device.commutes_with(device).sum())))
        matrix = device.commutes_with_device(device)
        results.append((f"commutes_device_{width}", int(matrix.count_commuting())))
        del matrix
    small = wolfgang_quantum.PauliSum.from_labels(["XZIY", "ZZII", "IIXX"], [0.5, -1.0, 2.0])
    psi = np.zeros(16, dtype=np.complex128)
    psi[3] = 1.0
    results.append(("expectation", complex(small.to_device(backend="hip").expectation_statevector(psi))))
    return results


def _pool_leak_check() -> dict[str, object]:
    """Run in a subprocess with the pool enabled."""
    if not core._hip_pool_status_for_testing(0)["enabled"]:
        return {"enabled": False}
    _workload()  # warm the pool so cached blocks exist
    baseline = _used_bytes()
    _workload()
    after = _used_bytes()
    held = wolfgang_quantum.PauliSum.from_labels(["XY"], [1.0]).to_device(backend="hip")
    holding = _used_bytes()
    del held
    return {"enabled": True, "baseline": baseline, "after": after, "holding": holding, "released": _used_bytes()}


def _run(script: str, enable_pool: bool) -> dict[str, object]:
    env = dict(os.environ)
    env.pop(ENABLE, None)
    if enable_pool:
        env[ENABLE] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True, text=True, check=True
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_default_allocator_is_hipmalloc() -> None:
    _require_hip_pool_hook()
    script = (
        "import json, wolfgang_quantum._wolfgang_core as core;"
        "print(json.dumps(dict(core._hip_pool_status_for_testing(0))))"
    )
    assert _run(script, enable_pool=False)["enabled"] is False


def test_enabled_pool_returns_usage_to_baseline() -> None:
    _require_hip_pool_hook()
    script = (
        "import json, sys; sys.path.insert(0, 'tests');"
        "import test_hip_pool_allocator as t;"
        "print(json.dumps(t._pool_leak_check()))"
    )
    report = _run(script, enable_pool=True)
    if not report["enabled"]:
        pytest.skip("HIP device does not support memory pools")
    assert report["after"] == report["baseline"]
    assert report["holding"] > report["baseline"]
    assert report["released"] == report["baseline"]


def test_pool_and_hipmalloc_results_match() -> None:
    _require_hip_pool_hook()
    script = (
        "import json, sys; sys.path.insert(0, 'tests');"
        "import test_hip_pool_allocator as t, wolfgang_quantum._wolfgang_core as core;"
        "print(json.dumps({'enabled': core._hip_pool_status_for_testing(0)['enabled'],"
        " 'results': [[n, repr(v)] for n, v in t._workload()]}))"
    )
    default = _run(script, enable_pool=False)
    pooled = _run(script, enable_pool=True)
    assert default["enabled"] is False
    assert pooled["results"] == default["results"]

"""Wolfgang's private HIP memory pool: no pooled leaks, and a working fallback.

Every public DevicePauliSum operation must return the pool's used bytes to
their starting value once its results are released, and the hipMalloc fallback
(forced with WOLFGANG_HIP_BENCH_DISABLE_POOL=1) must produce the same results.
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


def test_public_operations_return_pool_usage_to_baseline() -> None:
    _require_hip_pool_hook()
    if not core._hip_pool_status_for_testing(0)["enabled"]:
        pytest.skip("HIP device does not use the memory pool")
    _workload()  # warm the pool so cached blocks exist
    baseline = _used_bytes()
    _workload()
    assert _used_bytes() == baseline
    held = wolfgang_quantum.PauliSum.from_labels(["XY"], [1.0]).to_device(backend="hip")
    assert _used_bytes() > baseline
    del held
    assert _used_bytes() == baseline


def test_hipmalloc_fallback_matches_pooled_results() -> None:
    _require_hip_pool_hook()
    script = (
        "import json, sys; sys.path.insert(0, 'tests');"
        "import test_hip_pool_allocator as t, wolfgang_quantum._wolfgang_core as core;"
        "print(json.dumps({'enabled': core._hip_pool_status_for_testing(0)['enabled'],"
        " 'results': [[n, repr(v)] for n, v in t._workload()]}))"
    )
    outputs = {}
    for disabled in ("0", "1"):
        env = dict(os.environ, WOLFGANG_HIP_BENCH_DISABLE_POOL=disabled)
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True, text=True, check=True
        )
        outputs[disabled] = json.loads(completed.stdout.strip().splitlines()[-1])
    assert outputs["1"]["enabled"] is False
    assert outputs["1"]["results"] == outputs["0"]["results"]

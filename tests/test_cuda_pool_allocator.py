"""Wolfgang's private CUDA memory pool: no pooled leaks, and a working fallback.

Every public DevicePauliSum operation must return the pool's used bytes to
their starting value once its results are released, and the cudaMalloc fallback
(forced with WOLFGANG_CUDA_BENCH_DISABLE_POOL=1) must produce the same results.
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


def _require_cuda_pool_hook() -> None:
    status = dict(core._cuda_status())
    if not status["built"] or not status["runtime_available"]:
        pytest.skip(str(status["skip_reason"]))
    if not hasattr(core, "_cuda_pool_status_for_testing"):
        pytest.skip("private CUDA pool hook requires internal bindings")


def _used_bytes() -> int:
    gc.collect()
    return int(core._cuda_pool_status_for_testing(0)["used_bytes"])


def _workload() -> list[tuple[str, object]]:
    rng = np.random.default_rng(31)
    results: list[tuple[str, object]] = []
    for width in (12, 60, 100, 130):
        labels = ["".join(rng.choice(list("IXYZ"), size=width)) for _ in range(300)]
        labels += labels[:150]
        op = wolfgang_quantum.PauliSum.from_labels(labels, rng.normal(size=len(labels)) + 0j)
        device = op.to_device()
        results.append((f"simplify_{width}", device.simplify().to_host().to_labels()))
        results.append((f"single_{width}", op[0:1].to_device().simplify().to_host().to_labels()))
        results.append((f"matmul_{width}", device.matmul(device).to_host().num_terms))
        results.append((f"commutes_{width}", int(device.commutes_with(device).sum())))
        matrix = device.commutes_with_device(device)
        results.append((f"commutes_device_{width}", int(matrix.count_commuting())))
        results.append((f"commuting_rows_{width}", matrix.count_commuting(axis=1).tolist()))
        results.append((f"conflicts_{width}", int(matrix.conflict_degrees())))
        del matrix
    small = wolfgang_quantum.PauliSum.from_labels(["XZIY", "ZZII", "IIXX"], [0.5, -1.0, 2.0])
    psi = np.zeros(16, dtype=np.complex128)
    psi[3] = 1.0
    results.append(("expectation", complex(small.to_device().expectation_statevector(psi))))
    return results


def test_public_operations_return_pool_usage_to_baseline() -> None:
    _require_cuda_pool_hook()
    if not core._cuda_pool_status_for_testing(0)["enabled"]:
        pytest.skip("CUDA device does not use the memory pool")
    _workload()  # warm the pool so cached blocks exist
    baseline = _used_bytes()
    _workload()
    assert _used_bytes() == baseline
    held = wolfgang_quantum.PauliSum.from_labels(["XY"], [1.0]).to_device()
    assert _used_bytes() > baseline
    del held
    assert _used_bytes() == baseline


def test_cudamalloc_fallback_matches_pooled_results() -> None:
    _require_cuda_pool_hook()
    script = (
        "import json, sys; sys.path.insert(0, 'tests');"
        "import test_cuda_pool_allocator as t, wolfgang_quantum._wolfgang_core as core;"
        "print(json.dumps({'enabled': core._cuda_pool_status_for_testing(0)['enabled'],"
        " 'results': [[n, repr(v)] for n, v in t._workload()]}))"
    )
    outputs = {}
    for disabled in ("0", "1"):
        env = dict(os.environ, WOLFGANG_CUDA_BENCH_DISABLE_POOL=disabled)
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True, text=True, check=True
        )
        outputs[disabled] = json.loads(completed.stdout.strip().splitlines()[-1])
    assert outputs["1"]["enabled"] is False
    assert outputs["1"]["results"] == outputs["0"]["results"]


def test_commutation_output_and_scratch_come_from_the_pool() -> None:
    _require_cuda_pool_hook()
    if not core._cuda_pool_status_for_testing(0)["enabled"]:
        pytest.skip("CUDA device does not use the memory pool")
    rng = np.random.default_rng(5)
    labels = ["".join(rng.choice(list("IXYZ"), size=20)) for _ in range(512)]
    device = wolfgang_quantum.PauliSum.from_labels(labels, np.ones(len(labels))).to_device()
    device.commutes_with_device(device).count_commuting()  # warm the pool
    baseline = _used_bytes()
    matrix = device.commutes_with_device(device)
    assert _used_bytes() >= baseline + 512 * 512
    matrix.count_commuting(axis=0)
    matrix.conflict_degrees(axis=1)
    del matrix
    assert _used_bytes() == baseline


# Spins for `cycles` clocks, then copies n bytes: a consumer read still queued on a stream.
_DELAYED_BYTE_COPY = r"""
extern "C" __global__ void delayed_byte_copy(long long cycles, unsigned char* out, const unsigned char* src, int n) {
  const long long start = clock64();
  while (clock64() - start < cycles) {
  }
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    out[i] = src[i];
  }
}
"""


@pytest.mark.parametrize("protocol", ["cuda_array_interface", "dlpack"])
def test_exported_commutation_output_outlives_queued_consumer_work(protocol: str) -> None:
    _require_cuda_pool_hook()
    cupy = pytest.importorskip("cupy", reason="CuPy is required for CUDA interop tests")
    rng = np.random.default_rng(9)
    labels = ["".join(rng.choice(list("IXYZ"), size=16)) for _ in range(256)]
    device = wolfgang_quantum.PauliSum.from_labels(labels, np.ones(len(labels))).to_device()
    identity = wolfgang_quantum.PauliSum.from_labels(["I" * 16] * 256, np.ones(256)).to_device()
    matrix = device.commutes_with_device(device)
    view = cupy.asarray(matrix) if protocol == "cuda_array_interface" else cupy.from_dlpack(matrix)
    expected = cupy.asnumpy(view)
    assert not expected.all()  # some pairs anticommute, so a reused block would differ
    out = cupy.zeros(view.size, dtype=cupy.uint8)
    kernel = cupy.RawKernel(_DELAYED_BYTE_COPY, "delayed_byte_copy")
    stream = cupy.cuda.Stream(non_blocking=True)
    with stream:
        kernel((1,), (256,), (cupy.int64(100_000_000), out, view.ravel(), cupy.int32(view.size)))
    del view, matrix
    gc.collect()  # the exported matrix synchronizes before its block returns to the pool
    reused = device.commutes_with_device(identity)  # same size, all ones
    stream.synchronize()
    np.testing.assert_array_equal(cupy.asnumpy(out).reshape(expected.shape), expected)
    assert int(reused.count_commuting()) == 256 * 256

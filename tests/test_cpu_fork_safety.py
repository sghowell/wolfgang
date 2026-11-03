"""oneTBB does not support fork(): children forked after a oneTBB kernel ran must not use it.

The parent runs simplify through oneTBB and forks. In the child, auto dispatch
must run the bit-identical scalar kernel, and a forced
WOLFGANG_CPU_BACKEND=tbb must raise instead of crashing or hanging inside
oneTBB. The probe runs in a subprocess with a timeout, so a hang fails the
test rather than the test run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import wolfgang_quantum._wolfgang_core as core

_PROBE = r"""
import json, os, signal, sys
import numpy as np
from wolfgang_quantum import PauliSum

rng = np.random.default_rng(7)
x = rng.integers(0, 1 << 20, size=(20000, 1), dtype=np.uint64)
z = rng.integers(0, 1 << 20, size=(20000, 1), dtype=np.uint64)
x[1::2] = x[::2]
z[1::2] = z[::2]
op = PauliSum.from_arrays(x, z, rng.normal(size=20000) + 0j, 20)
os.environ["WOLFGANG_CPU_BACKEND"] = "tbb"
expected = op.simplify().to_arrays()
read, write = os.pipe()
pid = os.fork()
if pid == 0:
    signal.alarm(30)  # a hang inside oneTBB ends the child instead of the test
    result = {}
    os.environ["WOLFGANG_CPU_BACKEND"] = "auto"
    got = op.simplify().to_arrays()
    result["auto_matches"] = all(np.array_equal(a, b) for a, b in zip(got, expected))
    os.environ["WOLFGANG_CPU_BACKEND"] = "tbb"
    try:
        op.simplify()
        result["forced"] = "ran"
    except RuntimeError as error:
        result["forced"] = str(error)
    os.write(write, json.dumps(result).encode())
    os._exit(0)
os.close(write)
_, status = os.waitpid(pid, 0)
payload = os.read(read, 65536).decode()
print(json.dumps({"status": status, "child": json.loads(payload) if payload else None}))
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork() is POSIX-only")
def test_forked_child_avoids_onetbb_after_parent_used_it() -> None:
    info = dict(core._build_info())
    if "tbb" not in info.get("available_cpu_backends", []):
        pytest.skip("oneTBB backend is not available in this build")
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=120, check=False
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["status"] == 0, report
    assert report["child"]["auto_matches"] is True
    assert "unavailable_after_fork" in report["child"]["forced"]

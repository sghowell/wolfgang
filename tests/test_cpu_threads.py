"""Capping the threads Wolfgang's oneTBB CPU kernels use.

Release wheels bundle oneTBB, so parallel simplify runs by default. A process
in a multiprocessing pool, an MPI rank or a CPU-limited container needs a way
to stop each process from starting one oneTBB thread per core:
`wolfgang_quantum.set_num_threads(n)` and the `WOLFGANG_NUM_THREADS`
environment variable set a process-wide cap, `get_num_threads()` reports the
threads the kernels may use, and auto dispatch uses that count for its oneTBB
concurrency floor.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import wolfgang_quantum
import wolfgang_quantum._wolfgang_core as core

ROOT = Path(__file__).resolve().parents[1]
FLOOR = int(core._build_info()["cpu_auto_dispatch_thresholds"]["tbb_simplify_min_concurrency"])


def _tbb_available() -> bool:
    return "tbb" in core._build_info()["available_cpu_backends"]


def _run(code: str, **env: str) -> dict:
    environment = {k: v for k, v in os.environ.items() if k not in ("WOLFGANG_NUM_THREADS", "WOLFGANG_CPU_BACKEND")}
    environment.update(env)
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=environment, capture_output=True, text=True
    )
    if completed.returncode != 0:
        return {"error": completed.stderr.strip().splitlines()[-1]}
    return json.loads(completed.stdout.strip().splitlines()[-1])


REPORT = (
    "import json, wolfgang_quantum as w, wolfgang_quantum._wolfgang_core as core\n"
    "info = core._build_info()\n"
    "print(json.dumps({'threads': w.get_num_threads(), 'eligible': info['cpu_auto_tbb_eligible'],"
    " 'limit': info['cpu_thread_limit']}))\n"
)


def test_public_thread_functions_are_exported() -> None:
    assert callable(wolfgang_quantum.set_num_threads)
    assert callable(wolfgang_quantum.get_num_threads)
    assert {"set_num_threads", "get_num_threads"} <= set(wolfgang_quantum.__all__)
    stub = (ROOT / "python" / "wolfgang_quantum" / "__init__.pyi").read_text(encoding="utf-8")
    assert "set_num_threads" in stub and "get_num_threads" in stub


def test_get_num_threads_reports_a_positive_count() -> None:
    threads = wolfgang_quantum.get_num_threads()
    assert isinstance(threads, int) and threads >= 1
    if not _tbb_available():
        assert threads == 1


@pytest.mark.parametrize("bad", [0, -1, -(2**70)])
def test_set_num_threads_rejects_non_positive_values(bad: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        wolfgang_quantum.set_num_threads(bad)


def test_set_num_threads_rejects_oversized_values() -> None:
    with pytest.raises(ValueError, match="too large"):
        wolfgang_quantum.set_num_threads(2**70)


def test_numpy_integers_and_huge_caps_are_accepted() -> None:
    # A cap at or above oneTBB's default concurrency means no cap: it must not
    # raise oneTBB's own limit, and values near 2^31 and 2^32 must not reach
    # oneTBB, which narrows them internally.
    code = (
        "import json, numpy as np, wolfgang_quantum as w\n"
        "before = w.get_num_threads()\n"
        "w.set_num_threads(np.int64(2))\n"
        "two = w.get_num_threads()\n"
        "out = {'before': before, 'two': two}\n"
        "for cap in (2**31 + 1, 2**32 + 1, 2**40):\n"
        "    w.set_num_threads(cap)\n"
        "    x = np.arange(20000, dtype=np.uint64).reshape(-1, 1) % 4096\n"
        "    op = w.PauliSum.from_arrays(x, x.copy(), np.ones(20000) + 0j, 12)\n"
        "    out[str(cap)] = [w.get_num_threads(), op.simplify().num_terms]\n"
        "print(json.dumps(out))\n"
    )
    result = _run(code, WOLFGANG_CPU_BACKEND="auto")
    assert "error" not in result, result
    assert result["two"] == min(2, result["before"])
    for cap in (2**31 + 1, 2**32 + 1, 2**40):
        assert result[str(cap)] == [result["before"], 4096]


@pytest.mark.parametrize("bad", [1.0, "2", None, True])
def test_set_num_threads_rejects_non_integers(bad: object) -> None:
    with pytest.raises(TypeError):
        wolfgang_quantum.set_num_threads(bad)  # type: ignore[arg-type]


def test_set_num_threads_caps_the_count_and_auto_dispatch() -> None:
    code = (
        "import json, wolfgang_quantum as w, wolfgang_quantum._wolfgang_core as core\n"
        "before = w.get_num_threads()\n"
        "w.set_num_threads(2)\n"
        "two = w.get_num_threads(); eligible_two = core._build_info()['cpu_auto_tbb_eligible']\n"
        "w.set_num_threads(10_000)\n"
        "print(json.dumps({'before': before, 'two': two, 'eligible_two': eligible_two,"
        " 'large': w.get_num_threads()}))\n"
    )
    result = _run(code)
    if not _tbb_available():
        assert result == {"before": 1, "two": 1, "eligible_two": False, "large": 1}
        return
    assert result["two"] == min(2, result["before"])
    assert result["eligible_two"] is (result["two"] >= FLOOR)  # two threads meet the simplify floor
    assert result["large"] == result["before"]  # never above oneTBB's own concurrency


def test_environment_variable_sets_the_cap() -> None:
    result = _run(REPORT, WOLFGANG_NUM_THREADS="3")
    if not _tbb_available():
        assert result["threads"] == 1
        return
    default = _run(REPORT)
    assert result["threads"] == min(3, default["threads"])
    assert result["limit"] == result["threads"]
    assert result["eligible"] is (result["threads"] >= FLOOR)
    assert default["eligible"] is (default["threads"] >= FLOOR)


@pytest.mark.parametrize("bad", ["0", "-2", "four", "3.5", " 4"])
def test_invalid_environment_variable_raises(bad: str) -> None:
    result = _run(REPORT, WOLFGANG_NUM_THREADS=bad)
    assert "WOLFGANG_NUM_THREADS must be a positive integer" in result.get("error", "")


def test_invalid_environment_variable_raises_on_forced_onetbb_commutation() -> None:
    if not _tbb_available():
        pytest.skip("oneTBB is not available")
    code = (
        "import wolfgang_quantum as w\n"
        "op = w.PauliSum.from_labels(['XZ', 'ZX', 'YY'], [1.0, 1.0, 1.0])\n"
        "op.commutes_with(op)\n"
        "print('{}')\n"
    )
    result = _run(code, WOLFGANG_CPU_BACKEND="tbb", WOLFGANG_NUM_THREADS="abc")
    assert "WOLFGANG_NUM_THREADS must be a positive integer" in result.get("error", "")


def test_child_forked_after_onetbb_reports_one_thread() -> None:
    if not _tbb_available() or not hasattr(os, "fork"):
        pytest.skip("needs oneTBB and fork()")
    code = (
        "import json, os, numpy as np, wolfgang_quantum as w\n"
        "x = np.arange(20000, dtype=np.uint64).reshape(-1, 1)\n"
        "w.PauliSum.from_arrays(x, x.copy(), np.ones(20000) + 0j, 16).simplify()\n"
        "read, write = os.pipe()\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    os.write(write, str(w.get_num_threads()).encode()); os._exit(0)\n"
        "os.waitpid(pid, 0)\n"
        "print(json.dumps({'parent': w.get_num_threads(), 'child': int(os.read(read, 32))}))\n"
    )
    result = _run(code, WOLFGANG_CPU_BACKEND="tbb")
    assert result["child"] == 1
    assert result["parent"] >= 1


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="counts threads through /proc")
def test_cap_bounds_the_onetbb_worker_threads() -> None:
    usable = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    if not _tbb_available() or usable < 8:
        pytest.skip("needs oneTBB and at least 8 usable CPUs")
    code = (
        "import json, os, sys, numpy as np, wolfgang_quantum as w\n"
        "cap = int(sys.argv[1])\n"
        "if cap: w.set_num_threads(cap)\n"
        "before = len(os.listdir('/proc/self/task'))\n"
        "x = np.arange(400000, dtype=np.uint64).reshape(-1, 1)\n"
        "op = w.PauliSum.from_arrays(x, x.copy(), np.ones(400000) + 0j, 20)\n"
        "for _ in range(5): op.simplify()\n"
        "print(json.dumps({'added': len(os.listdir('/proc/self/task')) - before}))\n"
    )
    environment = {k: v for k, v in os.environ.items() if k != "WOLFGANG_NUM_THREADS"}
    environment["WOLFGANG_CPU_BACKEND"] = "tbb"

    def added(cap: int) -> int:
        completed = subprocess.run(
            [sys.executable, "-c", code, str(cap)], cwd=ROOT, env=environment, capture_output=True, text=True, check=True
        )
        return int(json.loads(completed.stdout.strip().splitlines()[-1])["added"])

    assert added(4) <= 3  # the calling thread plus at most three oneTBB workers
    assert added(0) > added(4)


def test_explicit_cap_replaces_the_environment_cap() -> None:
    code = (
        "import json, wolfgang_quantum as w\n"
        "w.set_num_threads(1)\n"
        "print(json.dumps({'threads': w.get_num_threads()}))\n"
    )
    assert _run(code, WOLFGANG_NUM_THREADS="3")["threads"] == 1


@pytest.mark.parametrize("backend", ["auto", "tbb"])
def test_results_do_not_depend_on_the_thread_count(backend: str) -> None:
    # Under auto, a cap of one runs scalar; forcing tbb runs oneTBB at one
    # thread too. Every result must be identical, bit for bit.
    if backend == "tbb" and not _tbb_available():
        pytest.skip("oneTBB is not available")
    code = (
        "import hashlib, json, numpy as np, wolfgang_quantum as w\n"
        "rng = np.random.default_rng(3)\n"
        "x = rng.integers(0, 1 << 20, size=(60000, 1), dtype=np.uint64)\n"
        "z = rng.integers(0, 1 << 20, size=(60000, 1), dtype=np.uint64)\n"
        "x = np.concatenate([x, x[:30000]]); z = np.concatenate([z, z[:30000]])\n"
        "c = rng.normal(size=90000) + 1j * rng.normal(size=90000)\n"
        "op = w.PauliSum.from_arrays(x, z, c, 20)\n"
        "out = {}\n"
        "for n in (1, 2, 3, 4, 64):\n"
        "    w.set_num_threads(n)\n"
        "    s = op.simplify()\n"
        "    out[n] = hashlib.sha256(b''.join(a.tobytes() for a in s.to_arrays())).hexdigest()\n"
        "print(json.dumps(out))\n"
    )
    result = _run(code, WOLFGANG_CPU_BACKEND=backend)
    assert "error" not in result, result
    assert len(set(result.values())) == 1

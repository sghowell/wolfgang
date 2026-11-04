"""Auto dispatch's oneTBB thread floors.

Auto runs simplify (from 8,192 terms) on oneTBB with at least two threads, and
pairwise commutation (from 331,776 entries) with at least two threads, or three
when an AVX2 or AVX-512 kernel covers the operator: each oneTBB thread runs that
kernel, and one core's two hyperthreads share its vector units
(docs/benchmarks/reports/cpu_thread_floor_2026-11-03.md). One-word operators
switch at the same matrix size as wider ones
(docs/benchmarks/reports/host_commutation_into_2026-11-03.md).

oneTBB use is observed through the fork guard: a child forked after a oneTBB
kernel ran reports one thread.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import wolfgang_quantum._wolfgang_core as core

ROOT = Path(__file__).resolve().parents[1]
INFO = core._build_info()
THRESHOLDS = INFO["cpu_auto_dispatch_thresholds"]
SIMD = any(name in INFO["available_cpu_backends"] for name in ("avx512", "avx2", "neon"))
X86_SIMD = any(name in INFO["available_cpu_backends"] for name in ("avx512", "avx2"))

PROBE = """
import json, os, numpy as np, wolfgang_quantum as w
default = w.get_num_threads()
w.set_num_threads({cap})
threads = w.get_num_threads()
rng = np.random.default_rng(1)
{workload}
read, write = os.pipe()
pid = os.fork()
if pid == 0:
    # Lift the cap: the child then reports one thread only if oneTBB ran in
    # the parent and is unusable here.
    w.set_num_threads(1_000_000)
    os.write(write, str(w.get_num_threads()).encode())
    os._exit(0)
os.waitpid(pid, 0)
print(json.dumps({{"default": default, "threads": threads, "child": int(os.read(read, 32))}}))
"""

COMMUTATION = (
    "x = rng.integers(0, 2**63, size=({terms}, {words}), dtype=np.uint64)\n"
    "op = w.PauliSum.from_arrays(x, x.copy(), np.ones({terms}) + 0j, 64 * {words})\n"
    "op.commutes_with(op)"
)
SIMPLIFY = (
    "x = rng.integers(0, 2**20, size=(20000, 1), dtype=np.uint64)\n"
    "w.PauliSum.from_arrays(x, x.copy(), np.ones(20000) + 0j, 20).simplify()"
)


def _ran_onetbb(cap: int, workload: str) -> tuple[int, bool]:
    if "tbb" not in INFO["available_cpu_backends"] or not hasattr(os, "fork"):
        pytest.skip("needs oneTBB and fork()")
    environment = {k: v for k, v in os.environ.items() if k not in ("WOLFGANG_NUM_THREADS", "WOLFGANG_CPU_BACKEND")}
    completed = subprocess.run(
        [sys.executable, "-c", PROBE.format(cap=cap, workload=workload)],
        cwd=ROOT, env=environment, capture_output=True, text=True, check=True,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    if result["default"] < max(cap, 2):
        pytest.skip(f"this host gives oneTBB fewer than {max(cap, 2)} threads")
    return result["threads"], result["child"] == 1


def test_floors_are_reported() -> None:
    assert THRESHOLDS["tbb_simplify_min_concurrency"] == 2
    assert THRESHOLDS["tbb_pairwise_min_concurrency"] == 2
    assert THRESHOLDS["tbb_pairwise_min_concurrency_over_x86_simd"] == 3
    assert "tbb_pairwise_one_word_x86_simd_entries" not in THRESHOLDS


@pytest.mark.parametrize("cap", [1, 2])
def test_simplify_uses_onetbb_from_two_threads(cap: int) -> None:
    threads, used = _ran_onetbb(cap, SIMPLIFY)
    assert used is (threads >= 2)


@pytest.mark.parametrize("cap", [2, 3])
def test_commutation_over_simd_uses_onetbb_from_three_threads_on_x86(cap: int) -> None:
    # Over AVX2 or AVX-512 two threads can be one core's hyperthreads, which
    # share its vector units; Arm cores have no SMT, so NEON needs two.
    if not SIMD:
        pytest.skip("no SIMD commutation kernel is available")
    threads, used = _ran_onetbb(cap, COMMUTATION.format(terms=600, words=2))
    assert used is (threads >= (3 if X86_SIMD else 2))


@pytest.mark.parametrize("cap", [1, 2])
def test_commutation_without_simd_uses_onetbb_from_two_threads(cap: int) -> None:
    # Five words: no SIMD kernel covers the operator, so the alternative is scalar.
    threads, used = _ran_onetbb(cap, COMMUTATION.format(terms=600, words=5))
    assert used is (threads >= 2)


@pytest.mark.parametrize("terms", [500, 600])
def test_one_word_commutation_switches_at_the_base_threshold(terms: int) -> None:
    # 500 x 500 is below the 331,776-entry switch point and 600 x 600 above it.
    # Once the kernels wrote straight into the result, one-word operators on
    # AVX2 or AVX-512 no longer needed a larger matrix.
    if not SIMD:
        pytest.skip("no SIMD commutation kernel is available")
    threads, used = _ran_onetbb(3, COMMUTATION.format(terms=terms, words=1))
    if threads < 3:
        pytest.skip("needs three oneTBB threads")
    assert used is (terms * terms >= THRESHOLDS["tbb_pairwise_entries"])

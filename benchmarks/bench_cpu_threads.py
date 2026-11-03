"""CPU simplify under a oneTBB thread cap, alone and with processes sharing cores.

Two measurements on one operator (``--terms`` terms, 24 qubits, half of them
duplicates):

``scaling``
    One process; median simplify time after ``set_num_threads(n)`` for each n
    in ``--threads`` (``0`` means no cap).

``shared``
    ``--processes`` worker processes start together, each running ``--calls``
    simplify calls. The row reports the wall time until the last worker
    finishes, with no cap and with each worker capped at
    ``cores // processes`` threads.

Every result is checked against the first worker's output.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from _benchmark_metadata import benchmark_environment, command_string, git_provenance
except ModuleNotFoundError:
    from benchmarks._benchmark_metadata import (
        benchmark_environment,
        command_string,
        git_provenance,
    )

import wolfgang_quantum
import wolfgang_quantum._wolfgang_core as core


def make_operator(num_terms: int) -> wolfgang_quantum.PauliSum:
    rng = np.random.default_rng(20261103)
    unique = num_terms // 2
    x = rng.integers(0, 1 << 24, size=(unique, 1), dtype=np.uint64)
    z = rng.integers(0, 1 << 24, size=(unique, 1), dtype=np.uint64)
    picks = rng.integers(0, unique, size=num_terms - unique)
    x = np.concatenate([x, x[picks]])
    z = np.concatenate([z, z[picks]])
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    return wolfgang_quantum.PauliSum.from_arrays(x, z, coeffs, 24)


def worker(num_terms: int, calls: int, threads: int) -> None:
    if threads:
        wolfgang_quantum.set_num_threads(threads)
    op = make_operator(num_terms)
    op.simplify()
    start = time.perf_counter()
    for _ in range(calls):
        result = op.simplify()
    elapsed = time.perf_counter() - start
    digest = int(result.to_arrays()[2].view(np.uint64).sum(dtype=np.uint64))
    print(json.dumps({"seconds": elapsed, "threads": wolfgang_quantum.get_num_threads(), "digest": digest}))


def shared_run(args: argparse.Namespace, threads: int) -> dict[str, Any]:
    command = [sys.executable, __file__, "--worker", str(args.terms), str(args.calls), str(threads)]
    start = time.perf_counter()
    procs = [subprocess.Popen(command, stdout=subprocess.PIPE, text=True) for _ in range(args.processes)]
    outputs = [json.loads(p.communicate()[0].strip().splitlines()[-1]) for p in procs]
    wall = time.perf_counter() - start
    if len({o["digest"] for o in outputs}) != 1:
        raise RuntimeError("workers disagree")
    return {
        "threads_per_worker": outputs[0]["threads"],
        "wall_seconds": wall,
        "worker_seconds": [o["seconds"] for o in outputs],
        "digest": outputs[0]["digest"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms", type=int, default=500_000)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 16, 0])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--label", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", nargs=3, type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(*args.worker)
        return 0
    if "tbb" not in core._build_info()["available_cpu_backends"]:
        sys.stdout.write(json.dumps({"status": "skipped", "reason": "oneTBB is not available"}) + "\n")
        return 0
    op = make_operator(args.terms)
    reference = op.simplify()
    default_threads = wolfgang_quantum.get_num_threads()
    scaling = []
    for threads in args.threads:
        wolfgang_quantum.set_num_threads(threads if threads else 1_000_000)
        result = op.simplify()
        if result != reference:
            raise RuntimeError(f"simplify differs with {threads} threads")
        times = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            op.simplify()
            times.append(time.perf_counter() - start)
        scaling.append({"cap": threads, "threads": wolfgang_quantum.get_num_threads(),
                        "median_seconds": float(np.median(times)), "seconds": times})
    cores = os.cpu_count() or 1
    cap = max(1, cores // args.processes)
    shared = {"uncapped": shared_run(args, 0), "capped": shared_run(args, cap)}
    if shared["uncapped"]["digest"] != shared["capped"]["digest"]:
        raise RuntimeError("capped and uncapped workers disagree")
    report = {
        "status": "ok",
        "benchmark": "cpu_threads",
        "label": args.label,
        "command": command_string(),
        "git": git_provenance(),
        "environment": benchmark_environment(dict(core._build_info()), numpy_version=np.__version__),
        "cpu_count": cores,
        "default_threads": default_threads,
        "terms": args.terms,
        "scaling": scaling,
        "processes": args.processes,
        "calls_per_worker": args.calls,
        "shared": shared,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

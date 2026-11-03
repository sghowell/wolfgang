"""Where oneTBB starts to beat the alternative at low thread counts.

Auto dispatch runs simplify and large pairwise commutation on oneTBB only when
it may use at least ``kAutoTbbSimplifyMinConcurrency`` threads. This benchmark
times, on the same deterministic operators, the kernel auto would use instead
(scalar simplify; the best compiled SIMD commutation kernel, or scalar) against
oneTBB capped at 1, 2, 3 and 4 threads, from the dispatch thresholds upward.

Variants are forced with ``WOLFGANG_CPU_BACKEND`` and capped with
``wolfgang_quantum.set_num_threads``; each repeat runs them in a shuffled order
and the per-call median over repeats is reported. Under ``taskset`` the
affinity mask bounds oneTBB's default concurrency, so a cap above it changes
nothing: every row records the threads the kernels could actually use.

Run it pinned to emulate small machines, for example a two-vCPU VM whose
vCPUs are one core's hyperthreads:

    taskset -c 0,1 python benchmarks/bench_cpu_thread_floor.py --output pair.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections.abc import Callable
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

SELECTOR = "WOLFGANG_CPU_BACKEND"


def operator(num_qubits: int, num_terms: int, seed: int, duplicates: float) -> wolfgang_quantum.PauliSum:
    rng = np.random.default_rng(seed)
    words = (num_qubits + 63) // 64
    distinct = max(1, int(num_terms * (1 - duplicates)))
    x = rng.integers(0, 2**63, size=(distinct, words), dtype=np.uint64)
    z = rng.integers(0, 2**63, size=(distinct, words), dtype=np.uint64)
    if num_qubits % 64:
        tail = np.uint64((1 << (num_qubits % 64)) - 1)
        x[:, -1] &= tail
        z[:, -1] &= tail
    pick = rng.integers(0, distinct, size=num_terms)
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    return wolfgang_quantum.PauliSum.from_arrays(x[pick], z[pick], coeffs, num_qubits)


def best_simd(words: int) -> str | None:
    available = set(core._build_info()["available_cpu_backends"])
    if words > 2:
        return None  # SIMD commutation covers one- and two-word operators
    for name in ("avx512", "avx2", "neon"):
        if name in available:
            return name
    return None


def timed(call: Callable[[], Any], target_seconds: float) -> float:
    """Seconds per call, with enough calls to fill about target_seconds."""
    start = time.perf_counter()
    call()
    first = time.perf_counter() - start
    calls = max(1, int(target_seconds / max(first, 1e-6)))
    start = time.perf_counter()
    for _ in range(calls):
        call()
    return (time.perf_counter() - start) / calls


def run_variant(selector: str, cap: int | None, call: Callable[[], Any], target_seconds: float) -> tuple[float, int]:
    os.environ[SELECTOR] = selector
    if cap is not None:
        wolfgang_quantum.set_num_threads(cap)
    threads = wolfgang_quantum.get_num_threads()
    return timed(call, target_seconds), threads


def measure(
    call: Callable[[], Any], alternative: str, caps: list[int], repeat: int, target_seconds: float
) -> dict[str, Any]:
    variants = [(alternative, None)] + [("tbb", cap) for cap in caps]
    samples: dict[str, list[float]] = {}
    threads: dict[str, int] = {}
    run_variant(alternative, None, call, target_seconds / 4)  # warm caches and pages
    for _ in range(repeat):
        order = variants[:]
        random.shuffle(order)
        for selector, cap in order:
            key = selector if cap is None else f"tbb_{cap}"
            seconds, used = run_variant(selector, cap, call, target_seconds)
            samples.setdefault(key, []).append(seconds)
            if cap is not None:
                threads[key] = used  # the alternative is single-threaded
    medians = {key: float(np.median(values)) for key, values in samples.items()}
    return {
        "alternative": alternative,
        "median_seconds": medians,
        "samples_seconds": samples,
        "threads_used": threads,
        "tbb_speedup_vs_alternative": {
            key: medians[alternative] / value for key, value in medians.items() if key.startswith("tbb_")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--simplify-sizes", default="8192,16384,32768,65536,262144,1048576")
    parser.add_argument("--simplify-qubits", default="24,130")
    parser.add_argument("--duplicates", default="0.5,0.0")
    parser.add_argument("--commutation-sizes", default="576,1024,2048,4096")
    parser.add_argument("--commutation-qubits", default="24,100,300")
    parser.add_argument("--caps", default="1,2,3,4")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--target-seconds", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if "tbb" not in core._build_info()["available_cpu_backends"]:
        sys.stdout.write(json.dumps({"status": "skipped", "reason": "oneTBB is not available"}) + "\n")
        return 0
    random.seed(args.seed)
    caps = [int(value) for value in args.caps.split(",")]
    rows: list[dict[str, Any]] = []
    for duplicates in (float(value) for value in args.duplicates.split(",")):
        for num_qubits in (int(value) for value in args.simplify_qubits.split(",")):
            for num_terms in (int(value) for value in args.simplify_sizes.split(",")):
                op = operator(num_qubits, num_terms, args.seed + num_terms + num_qubits, duplicates)
                row = measure(op.simplify, "scalar", caps, args.repeat, args.target_seconds)
                row.update(operation="simplify", num_qubits=num_qubits, num_terms=num_terms, duplicates=duplicates)
                rows.append(row)
                sys.stderr.write(f"simplify {num_terms} x {num_qubits}q dup {duplicates}: "
                                 f"{row['tbb_speedup_vs_alternative']}\n")
    for num_qubits in (int(value) for value in args.commutation_qubits.split(",")):
        for size in (int(value) for value in args.commutation_sizes.split(",")):
            op = operator(num_qubits, size, args.seed + size + 7 * num_qubits, 0.0)
            alternative = best_simd((num_qubits + 63) // 64) or "scalar"
            row = measure(lambda op=op: op.commutes_with(op), alternative, caps, args.repeat, args.target_seconds)
            row.update(operation="commutes_with", num_qubits=num_qubits, num_terms=size, duplicates=0.0)
            rows.append(row)
            sys.stderr.write(f"commutes_with {size}x{size} x {num_qubits}q vs {alternative}: "
                             f"{row['tbb_speedup_vs_alternative']}\n")
    os.environ.pop(SELECTOR, None)
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    report = {
        "benchmark": "cpu_thread_floor",
        "command": command_string(),
        "git": git_provenance(),
        "affinity": affinity,
        "logical_cpus": os.cpu_count(),
        "repeat": args.repeat,
        "target_seconds": args.target_seconds,
        "environment": benchmark_environment(dict(core._build_info()), numpy_version=np.__version__),
        "rows": rows,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

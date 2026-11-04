"""Single-threaded pairwise commutation per kernel, for comparing two builds.

Each forced selector (``WOLFGANG_CPU_BACKEND``) times ``PauliSum.commutes_with``
on deterministic square operators and reports the median of ``--calls`` calls
after ``--warmup`` calls. Run it once per build, alternating builds, pinned to
one core (for example ``taskset -c 4``), then ``--summarize`` the directory of
reports into per-label medians of the per-run medians.
"""

from __future__ import annotations

import argparse
import json
import os
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

SELECTOR = "WOLFGANG_CPU_BACKEND"


def operator(num_qubits: int, num_terms: int, seed: int) -> wolfgang_quantum.PauliSum:
    rng = np.random.default_rng(seed)
    words = (num_qubits + 63) // 64
    x = rng.integers(0, 2**63, size=(num_terms, words), dtype=np.uint64)
    z = rng.integers(0, 2**63, size=(num_terms, words), dtype=np.uint64)
    if num_qubits % 64:
        tail = np.uint64((1 << (num_qubits % 64)) - 1)
        x[:, -1] &= tail
        z[:, -1] &= tail
    return wolfgang_quantum.PauliSum.from_arrays(x, z, np.ones(num_terms, dtype=np.complex128), num_qubits)


def summarize_reports(directory: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    medians: dict[str, dict[str, dict[str, list[float]]]] = {}
    for path in sorted(directory.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("benchmark") != "cpu_commutation_kernels":
            continue
        label = report.get("label", "")
        runs.append({
            "file": path.name,
            "label": label,
            "command": report.get("command"),
            "git_commit": report["git"].get("commit"),
            "git_dirty": report["git"].get("dirty"),
            "affinity": report.get("affinity"),
            "loaded_native_build": report["environment"].get("loaded_native_build", {}),
        })
        for selector, cases in report["median_seconds"].items():
            for case, seconds in cases.items():
                medians.setdefault(selector, {}).setdefault(case, {}).setdefault(label, []).append(seconds)
    rows = {
        selector: {
            case: {
                label: {"per_run_median_seconds": values, "median_of_medians_seconds": float(np.median(values))}
                for label, values in by_label.items()
            }
            for case, by_label in cases.items()
        }
        for selector, cases in medians.items()
    }
    return {"runs": runs, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selectors", default="avx512,avx2,neon,scalar",
                        help="forced selectors to time; unavailable ones are skipped")
    parser.add_argument("--cases", default="24:576,24:4096,100:576,100:4096", help="qubits:terms pairs")
    parser.add_argument("--calls", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--label", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summarize", type=Path, help="aggregate saved reports in this directory")
    args = parser.parse_args()
    if args.summarize:
        text = json.dumps(summarize_reports(args.summarize), indent=2, sort_keys=True)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        else:
            sys.stdout.write(text + "\n")
        return 0
    available = set(core._build_info()["available_cpu_backends"])
    selectors = [name for name in args.selectors.split(",") if name in available]
    cases = [tuple(int(part) for part in case.split(":")) for case in args.cases.split(",")]
    operators = {(qubits, terms): operator(qubits, terms, args.seed + qubits + terms) for qubits, terms in cases}
    results: dict[str, dict[str, float]] = {}
    for selector in selectors:
        os.environ[SELECTOR] = selector
        for (qubits, terms), op in operators.items():
            if selector != "scalar" and (qubits + 63) // 64 > 2:
                continue  # SIMD commutation covers one- and two-word operators
            for _ in range(args.warmup):
                op.commutes_with(op)
            times = []
            for _ in range(args.calls):
                start = time.perf_counter()
                op.commutes_with(op)
                times.append(time.perf_counter() - start)
            results.setdefault(selector, {})[f"{qubits}q_{terms}x{terms}"] = float(np.median(times))
    os.environ.pop(SELECTOR, None)
    report = {
        "benchmark": "cpu_commutation_kernels",
        "label": args.label,
        "command": command_string(),
        "git": git_provenance(),
        "seed": args.seed,
        "calls": args.calls,
        "warmup": args.warmup,
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "environment": benchmark_environment(dict(core._build_info()), numpy_version=np.__version__),
        "median_seconds": results,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

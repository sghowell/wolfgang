"""Per-call overhead of small CUDA or HIP DevicePauliSum operations.

Small operands make kernel time negligible, so these timings expose the fixed
cost of each public call: device discovery, allocation, transfers, launches and
the closing synchronization. Each variant runs ``--calls`` times back to back
and the per-call median over ``--repeat`` batches is reported. Run it once per
build to compare checkouts; CPU simplify on the same operand is the unchanged
control.
"""

from __future__ import annotations

import argparse
import json
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


def operator(num_qubits: int, num_terms: int, seed: int) -> wolfgang_quantum.PauliSum:
    rng = np.random.default_rng(seed)
    letters = np.asarray(list("IXYZ"), dtype="<U1")
    chars = letters[rng.integers(0, 4, size=(num_terms, num_qubits))]
    labels = np.ascontiguousarray(chars).view(f"<U{num_qubits}").ravel().tolist()
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    return wolfgang_quantum.PauliSum.from_labels(labels, coeffs)


def per_call_seconds(fn: Callable[[], Any], *, calls: int, repeat: int) -> dict[str, Any]:
    for _ in range(max(3, calls // 10)):
        fn()
    batches = []
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(calls):
            fn()
        batches.append((time.perf_counter() - start) / calls)
    return {"median_seconds": float(np.median(batches)), "batch_seconds": batches}


def summarize_reports(directory: Path) -> dict[str, Any]:
    """Aggregate saved per-build reports into medians of per-run medians plus provenance."""
    runs: list[dict[str, Any]] = []
    medians: dict[str, dict[str, list[float]]] = {}
    for path in sorted(directory.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("benchmark") not in ("cuda_call_overhead", "hip_call_overhead"):
            continue
        label = report.get("label", "")
        runs.append(
            {
                "file": path.name,
                "benchmark": report["benchmark"],
                "label": label,
                "command": report.get("command"),
                "git_commit": report["git"].get("commit"),
                "git_dirty": report["git"].get("dirty"),
                "loaded_native_build": report["environment"].get("loaded_native_build", {}),
            }
        )
        for name, row in report["rows"].items():
            medians.setdefault(name, {}).setdefault(label, []).append(row["median_seconds"])
    if len({run["benchmark"] for run in runs}) > 1:
        raise SystemExit("refusing to summarize CUDA and HIP reports together; use one directory per backend")
    rows = {
        name: {
            label: {"per_run_median_seconds": values, "median_of_medians_seconds": float(np.median(values))}
            for label, values in by_label.items()
        }
        for name, by_label in medians.items()
    }
    return {"runs": runs, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "hip"), default="cuda")
    parser.add_argument("--calls", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=7)
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
    status = getattr(core, f"_{args.backend}_status")
    if not status().get("runtime_available"):
        sys.stdout.write(json.dumps({"status": "skipped"}) + "\n")
        return 0

    tiny = operator(24, 1, 1)
    small = operator(24, 128, 2)
    medium = operator(24, 4096, 3)
    edge = operator(24, 1024, 6)
    wide_edge = operator(200, 512, 7)
    wide = operator(130, 128, 4)
    psi = np.zeros(2**12, dtype=np.complex128)
    psi[0] = 1.0
    expectation_op = operator(12, 64, 5)
    device = {name: op.to_device(backend=args.backend) for name, op in
              {"tiny": tiny, "small": small, "medium": medium, "wide": wide,
               "edge": edge, "wide_edge": wide_edge, "expectation": expectation_op}.items()}

    variants: dict[str, Callable[[], Any]] = {
        "to_device_1_term": lambda: tiny.to_device(backend=args.backend),
        "to_host_1_term": device["tiny"].to_host,
        "simplify_1_term": device["tiny"].simplify,
        "simplify_128_terms_24q": device["small"].simplify,
        "simplify_4096_terms_24q": device["medium"].simplify,
        "simplify_128_terms_130q": device["wide"].simplify,
        "simplify_1024_terms_24q": device["edge"].simplify,
        "simplify_512_terms_200q": device["wide_edge"].simplify,
        "matmul_128x128_24q": lambda: device["small"].matmul(device["small"]),
        "commutes_with_device_128x128": lambda: device["small"].commutes_with_device(device["small"]),
        "expectation_64_terms_12q_host_psi": lambda: device["expectation"].expectation_statevector(psi),
        "round_trip_simplify_128_terms": lambda: small.to_device(backend=args.backend).simplify().to_host(),
        "cpu_simplify_128_terms_control": small.simplify,
    }
    rows = {name: per_call_seconds(fn, calls=args.calls, repeat=args.repeat) for name, fn in variants.items()}
    # A live matrix changes how the driver serves later cudaMalloc calls, so the
    # rows that need one run after every other row has been measured.
    matrix = device["small"].commutes_with_device(device["small"])
    rows["count_commuting_128x128"] = per_call_seconds(matrix.count_commuting, calls=args.calls, repeat=args.repeat)
    rows["conflict_degrees_rows_128x128"] = per_call_seconds(
        lambda: matrix.conflict_degrees(axis=1), calls=args.calls, repeat=args.repeat
    )
    report = {
        "benchmark": f"{args.backend}_call_overhead",
        "label": args.label,
        "command": command_string(),
        "git": git_provenance(),
        "calls": args.calls,
        "repeat": args.repeat,
        "environment": benchmark_environment(dict(core._build_info()), numpy_version=np.__version__),
        f"{args.backend}_status": dict(status()),
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

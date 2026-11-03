"""Host-device transfer times of CUDA or HIP DevicePauliSum operators.

Times ``PauliSum.to_device()`` and ``DevicePauliSum.to_host()`` for operators
of 1 to 4,000,000 terms at 24 and 100 qubits (one and two packed words). Each
row reports the median of individual call times after two warmup calls; small
operators take 200 samples and large ones 15. Run it once per build to compare
checkouts. Every round trip is checked against the source arrays before timing.
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

QUBITS = (24, 100)
TERMS = (1, 128, 4096, 65_536, 1_000_000, 4_000_000)


def random_operator(num_qubits: int, num_terms: int, seed: int) -> wolfgang_quantum.PauliSum:
    rng = np.random.default_rng(seed)
    words = (num_qubits + 63) // 64
    x = rng.integers(0, 2**63, size=(num_terms, words), dtype=np.uint64)
    z = rng.integers(0, 2**63, size=(num_terms, words), dtype=np.uint64)
    if num_qubits % 64:
        tail = np.uint64((1 << (num_qubits % 64)) - 1)
        x[:, -1] &= tail
        z[:, -1] &= tail
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    return wolfgang_quantum.PauliSum.from_arrays(x, z, coeffs, num_qubits)


def median_call_seconds(fn: Callable[[], Any], samples: int) -> dict[str, Any]:
    fn()
    fn()
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return {"median_seconds": float(np.median(times)), "samples": samples}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    status = getattr(core, f"_{args.backend}_status")
    if not status().get("runtime_available"):
        return {"status": "skipped", "reason": str(status().get("skip_reason", ""))}
    rows: dict[str, Any] = {}
    for num_qubits in QUBITS:
        for num_terms in TERMS:
            op = random_operator(num_qubits, num_terms, seed=num_qubits * 1_000_003 + num_terms)
            device = op.to_device(backend=args.backend)
            back = device.to_host()
            if not all(np.array_equal(a, b) for a, b in zip(back.to_arrays(), op.to_arrays(), strict=True)):
                raise RuntimeError(f"round trip changed {num_qubits}q/{num_terms}")
            samples = 200 if num_terms <= 65_536 else 15
            payload = num_terms * (2 * ((num_qubits + 63) // 64) * 8 + 16)
            for name, fn in (
                ("to_device", lambda: op.to_device(backend=args.backend)),
                ("to_host", device.to_host),
            ):
                rows[f"{name}_{num_qubits}q_{num_terms}"] = {
                    **median_call_seconds(fn, samples),
                    "num_qubits": num_qubits,
                    "num_terms": num_terms,
                    "payload_bytes": payload,
                }
    return {
        "status": "ok",
        "benchmark": f"{args.backend}_host_transfers",
        "label": args.label,
        "command": command_string(),
        "git": git_provenance(),
        "environment": benchmark_environment(dict(core._build_info()), numpy_version=np.__version__),
        f"{args.backend}_status": dict(status()),
        "rows": rows,
    }


def summarize_reports(directory: Path) -> dict[str, Any]:
    """Aggregate saved per-build reports into medians of per-run medians plus provenance."""
    runs: list[dict[str, Any]] = []
    medians: dict[str, dict[str, list[float]]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("benchmark") not in ("cuda_host_transfers", "hip_host_transfers"):
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
            meta[name] = {k: row[k] for k in ("num_qubits", "num_terms", "payload_bytes")}
    if len({run["benchmark"] for run in runs}) > 1:
        raise SystemExit("refusing to summarize CUDA and HIP reports together; use one directory per backend")
    rows = {
        name: {
            **meta[name],
            **{
                label: {"per_run_median_seconds": values, "median_of_medians_seconds": float(np.median(values))}
                for label, values in by_label.items()
            },
        }
        for name, by_label in medians.items()
    }
    return {"runs": runs, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "hip"), default="cuda")
    parser.add_argument("--label", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summarize", type=Path, help="aggregate saved reports in this directory")
    args = parser.parse_args()
    report = summarize_reports(args.summarize) if args.summarize else build_report(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

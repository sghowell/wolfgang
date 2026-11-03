"""Metal simplify: hybrid (GPU sort, CPU float64 reduce) vs host bridge vs CPU.

For one-word operators the public Metal simplify uses the hybrid from its term
threshold. Each repetition times the hybrid (forced with
WOLFGANG_METAL_BENCH_SIMPLIFY=hybrid, so sizes below the threshold measure it
too), the previous host bridge (forced with =host_bridge) and CPU simplify in a
randomized order. Hybrid and host-bridge outputs are checked against CPU simplify bit for
bit before any timing is kept. The Metal variants time DevicePauliSum.simplify
from an operator already in shared buffers (created once, outside the timed
region) to a new one; on unified memory the bridge's transfer is its two
copies. CPU simplify times PauliSum.simplify on the host copy.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
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

SELECTOR = "WOLFGANG_METAL_BENCH_SIMPLIFY"
# Powers of two and sizes just past them: the sort network spans
# next_power_of_two(terms) positions, so 2**k + 1 is its worst case.
SIZES = (
    4_096, 8_192, 12_288, 16_384, 16_385, 65_536, 65_537,
    262_144, 262_145, 1_048_576, 1_048_577,
)
SMOKE_SIZES = (2_000,)


def random_operator(rng: np.random.Generator, num_qubits: int, num_terms: int, duplicates: float):
    unique = max(1, int(num_terms * (1.0 - duplicates)))
    letters = np.asarray(list("IXYZ"), dtype="<U1")
    chars = letters[rng.integers(0, 4, size=(unique, num_qubits))]
    base = np.ascontiguousarray(chars).view(f"<U{num_qubits}").ravel()
    labels = np.concatenate([base, base[rng.integers(0, unique, num_terms - unique)]]).tolist()
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    return wolfgang_quantum.PauliSum.from_labels(labels, coeffs)


def bit_identical(actual: Any, expected: Any) -> bool:
    return actual == expected and bool(
        np.array_equal(actual.to_arrays()[2].view(np.uint64), expected.to_arrays()[2].view(np.uint64))
    )


def first_call_child(variant: str, num_terms: int) -> None:
    """Times the first and second simplify in this fresh process (pipeline build included)."""
    os.environ[SELECTOR] = variant
    rng = np.random.default_rng(num_terms)
    device = random_operator(rng, 40, num_terms, 0.0).to_device(backend="metal")
    timings = []
    for _ in range(2):
        start = time.perf_counter()
        device.simplify()
        timings.append(time.perf_counter() - start)
    sys.stdout.write(json.dumps({"variant": variant, "first_seconds": timings[0], "second_seconds": timings[1]}) + "\n")


def first_call_rows(processes: int, num_terms: int) -> list[dict[str, Any]]:
    rows = []
    for _ in range(processes):
        for variant in ("hybrid", "host_bridge"):
            output = subprocess.run(
                [sys.executable, __file__, "--first-call-child", variant, "--sizes", str(num_terms)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            rows.append(json.loads(output.strip().splitlines()[-1]))
    return rows


def summarize_reports(directory: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    medians: dict[tuple[int, int, float], dict[str, list[float]]] = {}
    raw: dict[tuple[int, int, float], dict[str, list[float]]] = {}
    output_terms: dict[tuple[int, int, float], set[int]] = {}
    environment: dict[str, Any] = {}
    for path in sorted(directory.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("benchmark") != "metal_hybrid_simplify":
            continue
        environment = environment or report["environment"]
        runs.append(
            {
                "file": path.name,
                "seed": report.get("seed"),
                "command": report.get("command"),
                "git_commit": report["git"].get("commit"),
                "git_dirty": report["git"].get("dirty"),
                "loaded_native_build": report["environment"].get("loaded_native_build", {}),
            }
        )
        for row in report["rows"]:
            key = (row["num_qubits"], row["num_terms"], row["duplicate_fraction"])
            output_terms.setdefault(key, set()).add(row["output_terms"])
            for name, data in row["variants"].items():
                medians.setdefault(key, {}).setdefault(name, []).append(data["median_seconds"])
                raw.setdefault(key, {}).setdefault(name, []).extend(data["raw_seconds"])
    rows = [
        {
            "num_qubits": qubits,
            "num_terms": terms,
            "duplicate_fraction": duplicates,
            "output_terms": sorted(output_terms[(qubits, terms, duplicates)]),
            "median_of_medians_seconds": {k: float(np.median(v)) for k, v in by_variant.items()},
            "per_run_median_seconds": by_variant,
            "raw_seconds_iqr": {
                k: [float(np.percentile(v, 25)), float(np.percentile(v, 75))]
                for k, v in raw[(qubits, terms, duplicates)].items()
            },
            "raw_seconds_min_max": {
                k: [float(min(v)), float(max(v))] for k, v in raw[(qubits, terms, duplicates)].items()
            },
        }
        for (qubits, terms, duplicates), by_variant in sorted(medians.items())
    ]
    keys = ("operating_system", "cpu_architecture", "logical_cpu_count", "python_version",
            "compiler", "compiler_build_config", "compiler_cpu_flags", "cpu_cmake_options",
            "thread_settings", "oneTBB", "apple_hardware", "Metal", "active_fastpauli_cpu_backend")
    return {
        "simplify_arguments": "defaults: atol=1e-12, rtol=0.0",
        "correctness": "hybrid and host-bridge outputs checked bit-identical to CPU simplify before timing",
        "environment": {key: environment.get(key) for key in keys if key in environment},
        "runs": runs,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--sizes", type=int, nargs="+", help="override the term counts")
    parser.add_argument("--qubits", type=int, nargs="+", default=[24, 60])
    parser.add_argument("--duplicates", type=float, nargs="+", default=[0.0, 0.5, 0.9])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summarize", type=Path)
    parser.add_argument(
        "--first-call-processes",
        type=int,
        default=0,
        help="instead of steady-state timing, time the first call in this many fresh processes",
    )
    parser.add_argument("--first-call-child", choices=("hybrid", "host_bridge"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.first_call_child:
        first_call_child(args.first_call_child, (args.sizes or [12_288])[0])
        return 0
    if args.summarize:
        text = json.dumps(summarize_reports(args.summarize), indent=2, sort_keys=True)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        else:
            sys.stdout.write(text + "\n")
        return 0
    status = dict(core._metal_status())
    if not status.get("runtime_available"):
        sys.stdout.write(json.dumps({"status": "skipped", "reason": status.get("skip_reason", "")}) + "\n")
        return 0

    if args.first_call_processes:
        num_terms = (args.sizes or [12_288])[0]
        report = {
            "benchmark": "metal_hybrid_simplify_first_call",
            "command": command_string(),
            "git": git_provenance(),
            "num_terms": num_terms,
            "num_qubits": 40,
            "rows": first_call_rows(args.first_call_processes, num_terms),
            "environment": benchmark_environment(dict(core._build_info()), numpy_version=np.__version__),
        }
        text = json.dumps(report, indent=2, sort_keys=True)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        else:
            sys.stdout.write(text + "\n")
        return 0

    rng = np.random.default_rng(args.seed)
    order_rng = random.Random(args.seed)
    rows = []
    sizes = SMOKE_SIZES if args.smoke else tuple(args.sizes or SIZES)
    for num_qubits in args.qubits:
        for num_terms in sizes:
            for duplicates in args.duplicates:
                op = random_operator(rng, num_qubits, num_terms, duplicates)
                device = op.to_device(backend="metal")

                def with_selector(value: str | None, fn: Callable[[], Any]) -> Callable[[], Any]:
                    def run() -> Any:
                        if value is None:
                            os.environ.pop(SELECTOR, None)
                        else:
                            os.environ[SELECTOR] = value
                        return fn()

                    return run

                variants = {
                    "metal_hybrid": with_selector("hybrid", device.simplify),
                    "metal_host_bridge": with_selector("host_bridge", device.simplify),
                    "cpu_simplify": with_selector(None, op.simplify),
                }
                expected = op.simplify()
                for name in ("metal_hybrid", "metal_host_bridge"):
                    if not bit_identical(variants[name]().to_host(), expected):
                        raise RuntimeError(f"{name} differs from CPU simplify")
                samples: dict[str, list[float]] = {name: [] for name in variants}
                for iteration in range(args.warmup + args.repeat):
                    names = list(variants)
                    order_rng.shuffle(names)
                    for name in names:
                        start = time.perf_counter()
                        variants[name]()
                        elapsed = time.perf_counter() - start
                        if iteration >= args.warmup:
                            samples[name].append(elapsed)
                os.environ.pop(SELECTOR, None)
                rows.append(
                    {
                        "num_qubits": num_qubits,
                        "num_terms": num_terms,
                        "duplicate_fraction": duplicates,
                        "output_terms": expected.num_terms,
                        "variants": {
                            name: {"median_seconds": float(np.median(v)), "raw_seconds": v}
                            for name, v in samples.items()
                        },
                    }
                )
    report = {
        "benchmark": "metal_hybrid_simplify",
        "command": command_string(),
        "git": git_provenance(),
        "seed": args.seed,
        "warmup": args.warmup,
        "repeat": args.repeat,
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

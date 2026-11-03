"""Qualify CUDA simplify duplicate-reduction paths against CPU simplify.

Two suites:

``generic``
    Operators wider than 128 qubits (the generic index-sort path). Each
    repetition runs the production parallel reduce-by-key, the retained
    benchmark-only serial kernel, and the CPU oracle in a randomized order, so
    all three share warm state and drift. CPU simplify is the unchanged control.

``specialized``
    One-word (packed 32-bit key and two-mask key) and two-word operators.
    Run this suite once per build to compare checkouts; the CPU row is the
    unchanged control for paired runs.

Device-resident rows time ``DevicePauliSum.simplify`` on an operator already on
the GPU; the call synchronizes before returning and includes output
allocation. Transfer-inclusive rows time host-to-device copy, simplify and
device-to-host materialization. Every output is checked against CPU canonical
labels and coefficients before its timings are kept.
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

GENERIC_SELECTOR = "WOLFGANG_CUDA_BENCH_GENERIC_MULTIWORD_REDUCTION"
GENERIC_QUBITS = (130, 256, 1024)
GENERIC_TERMS = (1, 128, 4096, 65536)
SPECIALIZED_CASES = (
    ("one_word_packed32", 24),
    ("one_word_two_mask", 60),
    ("two_word", 100),
)
SPECIALIZED_TERMS = (50_000, 1_000_000)
DUPLICATE_FRACTIONS = (0.0, 0.9)
SMOKE_GENERIC = ((130, 128),)
SMOKE_SPECIALIZED = (("two_word", 100, 4096),)


def random_operator(
    rng: np.random.Generator, *, num_qubits: int, num_terms: int, duplicate_fraction: float
) -> wolfgang_quantum.PauliSum:
    unique_count = max(1, int(round(num_terms * (1.0 - duplicate_fraction))))
    letters = np.asarray(list("IXYZ"), dtype="<U1")
    chars = letters[rng.integers(0, 4, size=(unique_count, num_qubits))]
    base = np.ascontiguousarray(chars).view(f"<U{num_qubits}").ravel()
    picks = rng.integers(0, unique_count, size=num_terms - unique_count)
    labels = np.concatenate([base, base[picks]]).tolist()
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    return wolfgang_quantum.PauliSum.from_labels(labels, coeffs.astype(np.complex128))


def same_operator(actual: wolfgang_quantum.PauliSum, expected: wolfgang_quantum.PauliSum) -> bool:
    actual_labels, actual_coeffs = actual.to_labels()
    expected_labels, expected_coeffs = expected.to_labels()
    return list(actual_labels) == list(expected_labels) and bool(
        np.allclose(actual_coeffs, expected_coeffs, rtol=1.0e-12, atol=1.0e-12)
    )


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_seconds": float(np.median(ordered)),
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
        "samples": len(ordered),
    }


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def set_selector(value: str | None) -> None:
    if value is None:
        os.environ.pop(GENERIC_SELECTOR, None)
    else:
        os.environ[GENERIC_SELECTOR] = value


def run_generic_case(
    rng: np.random.Generator,
    order_rng: random.Random,
    *,
    num_qubits: int,
    num_terms: int,
    duplicate_fraction: float,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    op = random_operator(
        rng, num_qubits=num_qubits, num_terms=num_terms, duplicate_fraction=duplicate_fraction
    )
    device_op = op.to_device()
    expected = op.simplify()
    # name -> (benchmark selector, callable). The selector is set before the
    # timer starts so environment updates stay outside the measured region.
    variants: dict[str, tuple[str | None, Callable[[], Any]]] = {
        "cuda_generic_parallel_reduce_by_key": ("reduce_by_key", device_op.simplify),
        "cuda_generic_parallel_transfer_inclusive": (
            "reduce_by_key",
            lambda: op.to_device().simplify().to_host(),
        ),
        "cuda_generic_serial_kernel": ("serial_kernel", device_op.simplify),
        "cpu_simplify_control": (None, op.simplify),
    }
    previous = os.environ.get(GENERIC_SELECTOR)
    try:
        for name, (selector, fn) in variants.items():
            set_selector(selector)
            result = fn()
            host = result if isinstance(result, wolfgang_quantum.PauliSum) else result.to_host()
            if not same_operator(host, expected):
                raise RuntimeError(f"{name} disagrees with CPU simplify for {num_qubits}q/{num_terms}")
        samples: dict[str, list[float]] = {name: [] for name in variants}
        for iteration in range(warmup + repeat):
            names = list(variants)
            order_rng.shuffle(names)
            for name in names:
                selector, fn = variants[name]
                set_selector(selector)
                _, seconds = timed(fn)
                if iteration >= warmup:
                    samples[name].append(seconds)
    finally:
        set_selector(previous)
    rows = {name: {**summarize(values), "raw_seconds": values} for name, values in samples.items()}
    parallel = rows["cuda_generic_parallel_reduce_by_key"]["median_seconds"]
    transfer = rows["cuda_generic_parallel_transfer_inclusive"]["median_seconds"]
    cpu = rows["cpu_simplify_control"]["median_seconds"]
    return {
        "suite": "generic",
        "num_qubits": num_qubits,
        "num_terms": num_terms,
        "duplicate_fraction": duplicate_fraction,
        "output_terms": expected.num_terms,
        "correctness_checked": "first output of each variant against CPU simplify",
        "variants": rows,
        "parallel_speedup_vs_serial": rows["cuda_generic_serial_kernel"]["median_seconds"] / parallel,
        "device_resident_speedup_vs_cpu": cpu / parallel,
        "transfer_inclusive_speedup_vs_cpu": cpu / transfer,
    }


def run_specialized_case(
    rng: np.random.Generator,
    *,
    path: str,
    num_qubits: int,
    num_terms: int,
    duplicate_fraction: float,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    op = random_operator(
        rng, num_qubits=num_qubits, num_terms=num_terms, duplicate_fraction=duplicate_fraction
    )
    device_op = op.to_device()
    expected = op.simplify()
    variants: dict[str, Callable[[], Any]] = {
        "cuda_device_resident": device_op.simplify,
        "cuda_transfer_inclusive": lambda: op.to_device().simplify().to_host(),
        "cpu_simplify_control": op.simplify,
    }
    for name, fn in variants.items():
        result = fn()
        host = result.to_host() if name == "cuda_device_resident" else result
        if not same_operator(host, expected):
            raise RuntimeError(f"{name} disagrees with CPU simplify for {path}/{num_terms}")
    samples: dict[str, list[float]] = {name: [] for name in variants}
    for iteration in range(warmup + repeat):
        for name, fn in variants.items():
            _, seconds = timed(fn)
            if iteration >= warmup:
                samples[name].append(seconds)
    return {
        "suite": "specialized",
        "path": path,
        "num_qubits": num_qubits,
        "num_terms": num_terms,
        "duplicate_fraction": duplicate_fraction,
        "output_terms": expected.num_terms,
        "correctness_checked": "first output of each variant against CPU simplify",
        "variants": {
            name: {**summarize(values), "raw_seconds": values} for name, values in samples.items()
        },
    }


def cuda_environment() -> dict[str, Any]:
    status = dict(core._cuda_status())
    build = dict(core._build_info())
    return {
        "cuda_status": status,
        "compiled_cuda_architectures": build.get("cuda_architectures", "not_available"),
        "cuda_toolkit_version": build.get("cuda_toolkit_version", "not_available"),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    status = dict(core._cuda_status())
    if not status.get("runtime_available"):
        return {"status": "skipped", "reason": str(status.get("skip_reason", "CUDA unavailable"))}
    rng = np.random.default_rng(args.seed)
    order_rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    if args.suite in ("generic", "all"):
        cases = SMOKE_GENERIC if args.smoke else [(q, t) for q in GENERIC_QUBITS for t in GENERIC_TERMS]
        for num_qubits, num_terms in cases:
            for duplicate_fraction in DUPLICATE_FRACTIONS:
                rows.append(
                    run_generic_case(
                        rng,
                        order_rng,
                        num_qubits=num_qubits,
                        num_terms=num_terms,
                        duplicate_fraction=duplicate_fraction,
                        warmup=args.warmup,
                        repeat=args.repeat,
                    )
                )
    if args.suite in ("specialized", "all"):
        cases = (
            SMOKE_SPECIALIZED
            if args.smoke
            else [(path, q, t) for path, q in SPECIALIZED_CASES for t in SPECIALIZED_TERMS]
        )
        for path, num_qubits, num_terms in cases:
            rows.append(
                run_specialized_case(
                    rng,
                    path=path,
                    num_qubits=num_qubits,
                    num_terms=num_terms,
                    duplicate_fraction=0.5,
                    warmup=args.warmup,
                    repeat=args.repeat,
                )
            )
    build_info = dict(core._build_info())
    return {
        "status": "ok",
        "benchmark": "cuda_simplify_paths",
        "label": args.label,
        "command": command_string(),
        "git": git_provenance(),
        "seed": args.seed,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "environment": benchmark_environment(build_info, numpy_version=np.__version__),
        "cuda": cuda_environment(),
        "rows": rows,
    }


def report_provenance(report: dict[str, Any]) -> dict[str, Any]:
    loaded = report["environment"].get("loaded_native_build", {})
    return {
        "label": report.get("label", ""),
        "command": report.get("command", ""),
        "seed": report.get("seed"),
        "git_commit": report["git"].get("commit"),
        "git_dirty": report["git"].get("dirty"),
        "loaded_native_build": loaded,
    }


def summarize_reports(directory: Path) -> dict[str, Any]:
    """Aggregate saved reports into the tracked summary (medians plus provenance)."""
    generic_rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    paired: dict[tuple[str, int, int, str], dict[str, list[float]]] = {}
    for path in sorted(directory.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("benchmark") != "cuda_simplify_paths":
            continue
        runs.append({"file": path.name, **report_provenance(report)})
        for row in report["rows"]:
            medians = {name: v["median_seconds"] for name, v in row["variants"].items()}
            if row["suite"] == "generic":
                generic_rows.append(
                    {
                        "file": path.name,
                        **{k: row[k] for k in ("num_qubits", "num_terms", "duplicate_fraction")},
                        "output_terms": row["output_terms"],
                        "median_seconds": medians,
                    }
                )
            else:
                for name, median in medians.items():
                    key = (row["path"], row["num_qubits"], row["num_terms"], name)
                    paired.setdefault(key, {}).setdefault(report.get("label", ""), []).append(median)
    specialized = [
        {
            "path": path,
            "num_qubits": qubits,
            "num_terms": terms,
            "variant": variant,
            "per_rerun_median_seconds": by_label,
            "median_of_medians_seconds": {
                label: float(np.median(values)) for label, values in by_label.items()
            },
        }
        for (path, qubits, terms, variant), by_label in sorted(paired.items())
    ]
    return {"runs": runs, "generic_rows": generic_rows, "specialized_rows": specialized}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("generic", "specialized", "all"), default="all")
    parser.add_argument("--smoke", action="store_true", help="run one tiny case per suite")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--label", default="", help="free-form build label for paired runs")
    parser.add_argument("--output", type=Path, help="write the JSON report to this path")
    parser.add_argument(
        "--summarize", type=Path, help="aggregate saved reports in this directory instead of running"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = summarize_reports(args.summarize) if args.summarize else build_report(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

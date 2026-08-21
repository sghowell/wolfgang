#!/usr/bin/env python3
"""Characterize CPU auto-dispatch thresholds for pairwise commutation."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import wolfgang_quantum as wolfgang
import wolfgang_quantum._wolfgang_core as core
from wolfgang_quantum import PauliSum

try:
    from _benchmark_metadata import benchmark_environment, command_string, git_commit
except ModuleNotFoundError:
    from benchmarks._benchmark_metadata import (
        benchmark_environment,
        command_string,
        git_commit,
    )


@contextmanager
def forced_backend(selector: str) -> Iterator[None]:
    previous = os.environ.get("WOLFGANG_CPU_BACKEND")
    os.environ["WOLFGANG_CPU_BACKEND"] = selector
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("WOLFGANG_CPU_BACKEND", None)
        else:
            os.environ["WOLFGANG_CPU_BACKEND"] = previous


def timed_call(fn: Any, *, warmup: int, repeat: int) -> tuple[Any, dict[str, float]]:
    result: Any = None
    for _ in range(warmup):
        result = fn()

    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        timings.append(time.perf_counter() - start)

    return result, {
        "median": statistics.median(timings),
        "min": min(timings),
        "max": max(timings),
    }


def make_operands(
    *,
    num_qubits: int,
    lhs_terms: int,
    rhs_terms: int,
    seed: int,
) -> tuple[PauliSum, PauliSum]:
    rng = np.random.default_rng(seed)
    alphabet = np.asarray(["I", "X", "Y", "Z"])
    lhs_labels = ["".join(rng.choice(alphabet, size=num_qubits).tolist()) for _ in range(lhs_terms)]
    rhs_labels = ["".join(rng.choice(alphabet, size=num_qubits).tolist()) for _ in range(rhs_terms)]
    lhs_coeffs = rng.normal(size=lhs_terms) + 1j * rng.normal(size=lhs_terms)
    rhs_coeffs = rng.normal(size=rhs_terms) + 1j * rng.normal(size=rhs_terms)
    return PauliSum.from_labels(lhs_labels, lhs_coeffs.tolist()), PauliSum.from_labels(
        rhs_labels,
        rhs_coeffs.tolist(),
    )


def backend_is_available(build_info: dict[str, Any], backend: str) -> bool:
    return any(
        candidate["name"] == backend and candidate["status"] == "available"
        for candidate in build_info["cpu_backend_candidates"]
    )


def simd_commutation_supports_num_qubits(num_qubits: int) -> bool:
    return ((num_qubits + 63) // 64) <= 2


def backend_supports_commutation_case(backend: str, num_qubits: int) -> bool:
    if backend == "tbb":
        return True
    if backend in {"avx512", "avx2", "neon"}:
        return simd_commutation_supports_num_qubits(num_qubits)
    return backend == "scalar"


def auto_pairwise_backend_hint(
    build_info: dict[str, Any],
    matrix_entries: int,
    num_qubits: int,
) -> str:
    threshold = build_info["cpu_auto_dispatch_thresholds"]["tbb_pairwise_entries"]
    if matrix_entries >= threshold and backend_is_available(build_info, "tbb"):
        return "tbb"
    if not simd_commutation_supports_num_qubits(num_qubits):
        return "scalar"
    for backend in ("avx512", "avx2", "neon"):
        if backend_is_available(build_info, backend):
            return backend
    return "scalar"


def time_pairwise(
    selector: str,
    lhs: PauliSum,
    rhs: PauliSum,
    *,
    warmup: int,
    repeat: int,
) -> tuple[np.ndarray, dict[str, float], str]:
    with forced_backend(selector):
        build_info = core._build_info()
        result, timings = timed_call(lambda: lhs.commutes_with(rhs), warmup=warmup, repeat=repeat)

    result_array = np.asarray(result, dtype=np.bool_).reshape(lhs.num_terms, rhs.num_terms)
    return result_array, timings, build_info["active_cpu_backend"]


def case_dimensions(args: argparse.Namespace, threshold: int) -> list[tuple[str, int, int]]:
    if args.smoke:
        side_at = max(1, math.ceil(math.sqrt(threshold)))
        side_below = max(1, side_at // 2)
        return [
            ("below_threshold", side_below, side_below),
            ("at_threshold", side_at, side_at),
        ]

    below = max(1, threshold // 4)
    at_or_above = threshold
    above = threshold * 4
    side_below = max(1, math.floor(math.sqrt(below)))
    side_at = max(1, math.ceil(math.sqrt(at_or_above)))
    side_above = max(1, math.ceil(math.sqrt(above)))
    return [
        ("below_threshold", side_below, side_below),
        ("at_threshold", side_at, side_at),
        ("above_threshold", side_above, side_above),
    ]


def build_case(
    *,
    name: str,
    lhs_terms: int,
    rhs_terms: int,
    args: argparse.Namespace,
    threshold: int,
    build_info: dict[str, Any],
) -> dict[str, Any]:
    matrix_entries = lhs_terms * rhs_terms
    lhs, rhs = make_operands(
        num_qubits=args.num_qubits,
        lhs_terms=lhs_terms,
        rhs_terms=rhs_terms,
        seed=args.seed + matrix_entries,
    )
    scalar_result, scalar_timings, _ = time_pairwise(
        "scalar",
        lhs,
        rhs,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    auto_result, auto_timings, auto_active_backend = time_pairwise(
        "auto",
        lhs,
        rhs,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    if not np.array_equal(auto_result, scalar_result):
        raise RuntimeError(f"auto pairwise commutation differs from scalar for {name}")

    optimized_results: dict[str, dict[str, float | bool | str]] = {}
    for candidate in build_info["cpu_backend_candidates"]:
        selector = candidate["name"]
        if selector == "scalar" or candidate["status"] != "available":
            continue
        if not backend_supports_commutation_case(selector, args.num_qubits):
            continue
        result, timings, active_backend = time_pairwise(
            selector,
            lhs,
            rhs,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        matches_scalar = bool(np.array_equal(result, scalar_result))
        if not matches_scalar:
            raise RuntimeError(f"{selector} pairwise commutation differs from scalar for {name}")
        optimized_results[selector] = {
            "active_backend": active_backend,
            "seconds": timings["median"],
            "min_seconds": timings["min"],
            "max_seconds": timings["max"],
            "matches_forced_scalar": matches_scalar,
        }

    return {
        "name": name,
        "dataset": {
            "num_qubits": args.num_qubits,
            "lhs_terms": lhs_terms,
            "rhs_terms": rhs_terms,
            "matrix_entries": matrix_entries,
            "threshold_region": "below" if matrix_entries < threshold else "at_or_above",
            "auto_effective_backend_hint": auto_pairwise_backend_hint(
                build_info,
                matrix_entries,
                args.num_qubits,
            ),
            "auto_active_backend_report": auto_active_backend,
            "coefficient_dtype": "complex128",
            "operator_construction_method": "deterministic dense labels",
            "random_seed": args.seed + matrix_entries,
        },
        "results": {
            "scalar_seconds": scalar_timings["median"],
            "scalar_min_seconds": scalar_timings["min"],
            "scalar_max_seconds": scalar_timings["max"],
            "auto_seconds": auto_timings["median"],
            "auto_min_seconds": auto_timings["min"],
            "auto_max_seconds": auto_timings["max"],
            "matches_forced_scalar": True,
            "optimized_backends": optimized_results,
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")

    build_info = core._build_info()
    threshold = build_info["cpu_auto_dispatch_thresholds"]["tbb_pairwise_entries"]
    effective_args = argparse.Namespace(**vars(args))
    if args.smoke:
        effective_args.warmup = 0
        effective_args.num_qubits = min(args.num_qubits, 7)

    cases = [
        build_case(
            name=name,
            lhs_terms=lhs_terms,
            rhs_terms=rhs_terms,
            args=effective_args,
            threshold=threshold,
            build_info=build_info,
        )
        for name, lhs_terms, rhs_terms in case_dimensions(effective_args, threshold)
    ]

    return {
        "benchmark": "cpu_thresholds",
        "git_commit": git_commit(),
        "command": command_string(),
        "environment": benchmark_environment(build_info, numpy_version=np.__version__),
        "fastpauli_version": wolfgang.__version__,
        "fastpauli_build_info": build_info,
        "thresholds": {
            "tbb_pairwise_entries": threshold,
            "neon_full_grouping_scalar_min_entries": build_info["cpu_auto_dispatch_thresholds"][
                "neon_full_grouping_scalar_min_entries"
            ],
        },
        "correctness_checks": {
            "enabled": True,
            "reference": "forced scalar pairwise commutation",
            "failure_mode": "raises RuntimeError if auto or an available optimized selector differs from scalar",
        },
        "timing_policy": {
            "warmup": effective_args.warmup,
            "repeat": effective_args.repeat,
            "summary": "median seconds",
        },
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--smoke", action="store_true", help="run validation-sized cases")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7241)
    parser.add_argument("--num-qubits", type=int, default=65)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

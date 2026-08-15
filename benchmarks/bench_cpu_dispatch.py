#!/usr/bin/env python3
"""Phase 9 CPU backend dispatch benchmark and availability report."""

from __future__ import annotations

import argparse
import json
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


def make_operator_and_state(
    num_qubits: int, num_terms: int, seed: int
) -> tuple[PauliSum, np.ndarray]:
    rng = np.random.default_rng(seed)
    alphabet = np.asarray(["I", "X", "Y", "Z"])
    labels = ["".join(rng.choice(alphabet, size=num_qubits).tolist()) for _ in range(num_terms)]
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    raw = rng.normal(size=1 << num_qubits) + 1j * rng.normal(size=1 << num_qubits)
    psi = np.asarray(raw, dtype=np.complex128)
    psi = psi / np.linalg.norm(psi)
    return PauliSum.from_labels(labels, coeffs.tolist()), psi


def make_commutation_operands(
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


def exported_group_labels(groups: list[PauliSum]) -> list[list[str]]:
    exported: list[list[str]] = []
    for group in groups:
        labels, _ = group.to_labels()
        exported.append(list(labels))
    return exported


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


def infer_auto_pairwise_backend(
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


def infer_auto_full_grouping_backend(build_info: dict[str, Any], num_qubits: int) -> str:
    if not simd_commutation_supports_num_qubits(num_qubits):
        return "scalar"
    for backend in ("avx512", "avx2", "neon"):
        if backend_is_available(build_info, backend):
            return backend
    return "scalar"


def run_statevector_case(
    *,
    name: str,
    selector: str,
    op: PauliSum,
    psi: np.ndarray,
    dataset: dict[str, Any],
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    with forced_backend(selector):
        build_info = core._build_info()
        result, timings = timed_call(
            lambda: op.expectation_statevector(psi),
            warmup=warmup,
            repeat=repeat,
        )

    dataset = {
        **dataset,
        "requested_cpu_backend": selector,
        "active_cpu_backend": build_info["active_cpu_backend"],
    }
    return {
        "name": name,
        "dataset": dataset,
        "results": {
            "fastpauli_seconds": timings["median"],
            "fastpauli_min_seconds": timings["min"],
            "fastpauli_max_seconds": timings["max"],
            "expectation_real": float(np.real(result)),
            "expectation_imag": float(np.imag(result)),
        },
    }


def run_pairwise_case(
    *,
    name: str,
    selector: str,
    lhs: PauliSum,
    rhs: PauliSum,
    scalar_reference: np.ndarray | None,
    dataset: dict[str, Any],
    warmup: int,
    repeat: int,
) -> tuple[dict[str, Any], np.ndarray]:
    with forced_backend(selector):
        build_info = core._build_info()
        result, timings = timed_call(
            lambda: lhs.commutes_with(rhs),
            warmup=warmup,
            repeat=repeat,
        )

    result_array = np.asarray(result, dtype=np.bool_).reshape(lhs.num_terms, rhs.num_terms)
    if scalar_reference is None:
        matches_scalar = True
    else:
        matches_scalar = bool(np.array_equal(result_array, scalar_reference))
        if not matches_scalar:
            raise RuntimeError(f"{selector} pairwise commutation differs from forced scalar")

    return (
        {
            "name": name,
            "dataset": {
                **dataset,
                "requested_cpu_backend": selector,
                "active_cpu_backend": build_info["active_cpu_backend"],
                "effective_backend_hint": (
                    infer_auto_pairwise_backend(
                        build_info,
                        lhs.num_terms * rhs.num_terms,
                        lhs.num_qubits,
                    )
                    if selector == "auto"
                    else build_info["active_cpu_backend"]
                ),
            },
            "results": {
                "fastpauli_seconds": timings["median"],
                "fastpauli_min_seconds": timings["min"],
                "fastpauli_max_seconds": timings["max"],
                "matrix_entries": lhs.num_terms * rhs.num_terms,
                "matches_forced_scalar": matches_scalar,
            },
        },
        result_array,
    )


def run_full_grouping_case(
    *,
    name: str,
    selector: str,
    op: PauliSum,
    scalar_reference: list[list[str]] | None,
    dataset: dict[str, Any],
    warmup: int,
    repeat: int,
) -> tuple[dict[str, Any], list[list[str]]]:
    with forced_backend(selector):
        build_info = core._build_info()
        result, timings = timed_call(
            lambda: op.group_commuting(mode="full"),
            warmup=warmup,
            repeat=repeat,
        )

    labels = exported_group_labels(result)
    if scalar_reference is None:
        matches_scalar = True
    else:
        matches_scalar = labels == scalar_reference
        if not matches_scalar:
            raise RuntimeError(f"{selector} full grouping differs from forced scalar")

    return (
        {
            "name": name,
            "dataset": {
                **dataset,
                "requested_cpu_backend": selector,
                "active_cpu_backend": build_info["active_cpu_backend"],
                "effective_backend_hint": (
                    infer_auto_full_grouping_backend(build_info, op.num_qubits)
                    if selector == "auto"
                    else build_info["active_cpu_backend"]
                ),
            },
            "results": {
                "fastpauli_seconds": timings["median"],
                "fastpauli_min_seconds": timings["min"],
                "fastpauli_max_seconds": timings["max"],
                "groups": len(result),
                "matches_forced_scalar": matches_scalar,
            },
        },
        labels,
    )


def optimized_availability_case(build_info: dict[str, Any]) -> dict[str, Any]:
    optimized = []
    for candidate in build_info["cpu_backend_candidates"]:
        if candidate["name"] == "scalar":
            continue
        optimized.append(
            {
                "backend": candidate["name"],
                "status": candidate["status"],
            }
        )

    return {
        "name": "optimized_backend_availability",
        "dataset": {
            "requested_cpu_backend": build_info["requested_cpu_backend"],
            "active_cpu_backend": build_info["active_cpu_backend"],
            "availability_scope": "compiled and runtime CPU backend candidates",
        },
        "results": {
            "optimized_backends": optimized,
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")

    warmup = 0 if args.smoke else args.warmup
    num_qubits = 7 if args.smoke else args.num_qubits
    commutation_qubits = 7 if args.smoke else args.commutation_qubits
    num_terms = 24 if args.smoke else args.num_terms
    op, psi = make_operator_and_state(num_qubits, num_terms, args.seed)
    lhs_terms = 8 if args.smoke else args.lhs_terms
    rhs_terms = 6 if args.smoke else args.rhs_terms
    lhs, rhs = make_commutation_operands(
        commutation_qubits,
        lhs_terms,
        rhs_terms,
        args.seed + 1000,
    )
    group_terms = 10 if args.smoke else args.group_terms
    grouping_op, _ = make_commutation_operands(
        commutation_qubits,
        group_terms,
        1,
        args.seed + 2000,
    )
    dataset = {
        "num_qubits": num_qubits,
        "num_terms": num_terms,
        "statevector_length": 1 << num_qubits,
        "statevector_dtype": "complex128",
        "coefficient_dtype": "complex128",
        "random_seed": args.seed,
        "operator_construction_method": "deterministic dense labels",
    }
    commutation_dataset = {
        "num_qubits": commutation_qubits,
        "lhs_terms": lhs_terms,
        "rhs_terms": rhs_terms,
        "matrix_entries": lhs_terms * rhs_terms,
        "coefficient_dtype": "complex128",
        "random_seed": args.seed + 1000,
        "operator_construction_method": "deterministic dense labels",
        "optimized_kernel": "pairwise_commutation",
    }
    grouping_dataset = {
        "num_qubits": commutation_qubits,
        "num_terms": group_terms,
        "coefficient_dtype": "complex128",
        "random_seed": args.seed + 2000,
        "operator_construction_method": "deterministic dense labels",
        "grouping_mode": "full",
        "strategy": "largest_first",
        "optimized_kernel": "full_group_commutation_graph",
    }

    default_build_info = core._build_info()
    auto_case = run_statevector_case(
        name="auto_statevector_expectation",
        selector="auto",
        op=op,
        psi=psi,
        dataset=dataset,
        warmup=warmup,
        repeat=args.repeat,
    )
    scalar_case = run_statevector_case(
        name="forced_scalar_statevector_expectation",
        selector="scalar",
        op=op,
        psi=psi,
        dataset=dataset,
        warmup=warmup,
        repeat=args.repeat,
    )

    if not np.allclose(
        auto_case["results"]["expectation_real"] + 1j * auto_case["results"]["expectation_imag"],
        scalar_case["results"]["expectation_real"]
        + 1j * scalar_case["results"]["expectation_imag"],
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError("auto and forced scalar expectation results differ")

    scalar_pairwise_case, scalar_pairwise_result = run_pairwise_case(
        name="forced_scalar_pairwise_commutation",
        selector="scalar",
        lhs=lhs,
        rhs=rhs,
        scalar_reference=None,
        dataset=commutation_dataset,
        warmup=warmup,
        repeat=args.repeat,
    )
    auto_pairwise_case, _ = run_pairwise_case(
        name="auto_pairwise_commutation",
        selector="auto",
        lhs=lhs,
        rhs=rhs,
        scalar_reference=scalar_pairwise_result,
        dataset=commutation_dataset,
        warmup=warmup,
        repeat=args.repeat,
    )
    optimized_pairwise_cases: list[dict[str, Any]] = []
    optimized_grouping_cases: list[dict[str, Any]] = []
    scalar_grouping_case, scalar_grouping_result = run_full_grouping_case(
        name="forced_scalar_full_grouping",
        selector="scalar",
        op=grouping_op,
        scalar_reference=None,
        dataset=grouping_dataset,
        warmup=warmup,
        repeat=args.repeat,
    )
    auto_grouping_case, _ = run_full_grouping_case(
        name="auto_full_grouping",
        selector="auto",
        op=grouping_op,
        scalar_reference=scalar_grouping_result,
        dataset=grouping_dataset,
        warmup=warmup,
        repeat=args.repeat,
    )
    for candidate in default_build_info["cpu_backend_candidates"]:
        selector = candidate["name"]
        if selector == "scalar" or candidate["status"] != "available":
            continue
        if backend_supports_commutation_case(selector, commutation_qubits):
            case, _ = run_pairwise_case(
                name=f"forced_{selector}_pairwise_commutation",
                selector=selector,
                lhs=lhs,
                rhs=rhs,
                scalar_reference=scalar_pairwise_result,
                dataset=commutation_dataset,
                warmup=warmup,
                repeat=args.repeat,
            )
            optimized_pairwise_cases.append(case)

        if backend_supports_commutation_case(selector, commutation_qubits):
            grouping_case, _ = run_full_grouping_case(
                name=f"forced_{selector}_full_grouping",
                selector=selector,
                op=grouping_op,
                scalar_reference=scalar_grouping_result,
                dataset=grouping_dataset,
                warmup=warmup,
                repeat=args.repeat,
            )
            optimized_grouping_cases.append(grouping_case)

    return {
        "benchmark": "cpu_dispatch",
        "git_commit": git_commit(),
        "command": command_string(),
        "environment": benchmark_environment(default_build_info, numpy_version=np.__version__),
        "fastpauli_version": wolfgang.__version__,
        "fastpauli_build_info": default_build_info,
        "timing_policy": {
            "warmup": warmup,
            "repeat": args.repeat,
            "summary": "median seconds",
        },
        "baselines": [
            "Wolfgang auto CPU dispatch",
            "Wolfgang forced scalar CPU",
            "Wolfgang optimized CPU candidates where compiled and available",
        ],
        "cases": [
            auto_case,
            scalar_case,
            scalar_pairwise_case,
            auto_pairwise_case,
            *optimized_pairwise_cases,
            scalar_grouping_case,
            auto_grouping_case,
            *optimized_grouping_cases,
            optimized_availability_case(default_build_info),
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--smoke", action="store_true", help="run validation-sized cases")
    parser.add_argument("--repeat", type=int, default=5, help="timed repetitions")
    parser.add_argument("--warmup", type=int, default=1, help="untimed warmup repetitions")
    parser.add_argument("--seed", type=int, default=6211, help="deterministic RNG seed")
    parser.add_argument("--num-qubits", type=int, default=10)
    parser.add_argument("--num-terms", type=int, default=128)
    parser.add_argument("--commutation-qubits", type=int, default=65)
    parser.add_argument("--lhs-terms", type=int, default=128)
    parser.add_argument("--rhs-terms", type=int, default=128)
    parser.add_argument("--group-terms", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Deterministic Phase 7 OpenFermion conversion benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import wolfgang_quantum as wolfgang
from wolfgang_quantum import PauliSum
from wolfgang_quantum.openfermion import OPENFERMION_INSTALL_HINT

try:
    from _benchmark_metadata import benchmark_environment, command_string, git_commit
except ModuleNotFoundError:
    from benchmarks._benchmark_metadata import (
        benchmark_environment,
        command_string,
        git_commit,
    )

try:
    from openfermion.ops import QubitOperator
except ImportError as exc:  # pragma: no cover - exercised by validation skip policy.
    raise SystemExit(OPENFERMION_INSTALL_HINT) from exc


PAULIS = np.asarray(["X", "Y", "Z"])


def make_term(
    rng: np.random.Generator, *, num_qubits: int, term_weight: int
) -> tuple[tuple[int, str], ...]:
    active_qubits = rng.choice(num_qubits, size=min(term_weight, num_qubits), replace=False)
    return tuple(sorted((int(qubit), str(rng.choice(PAULIS))) for qubit in active_qubits))


def generate_qubit_operator(
    *,
    num_qubits: int,
    num_terms: int,
    term_weight: int,
    seed: int,
) -> QubitOperator:
    rng = np.random.default_rng(seed)
    op = QubitOperator()
    for _ in range(num_terms):
        coeff = complex(rng.normal(), rng.normal())
        op += QubitOperator(make_term(rng, num_qubits=num_qubits, term_weight=term_weight), coeff)
    return op


def rebuild_openfermion(source: QubitOperator) -> QubitOperator:
    output = QubitOperator()
    for term, coeff in source.terms.items():
        output += QubitOperator(term, coeff)
    return output


def qubit_operator_close(lhs: QubitOperator, rhs: QubitOperator) -> bool:
    if set(lhs.terms) != set(rhs.terms):
        return False
    return all(
        np.allclose(lhs.terms[term], rhs.terms[term], rtol=1.0e-12, atol=1.0e-12)
        for term in lhs.terms
    )


def timed_call(fn: Callable[[], Any], *, warmup: int, repeat: int) -> tuple[Any, dict[str, float]]:
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


def run_case(
    *,
    name: str,
    num_qubits: int,
    requested_terms: int,
    term_weight: int,
    seed: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    source = generate_qubit_operator(
        num_qubits=num_qubits,
        num_terms=requested_terms,
        term_weight=term_weight,
        seed=seed,
    )

    fast_result, fast_timings = timed_call(
        lambda: PauliSum.from_openfermion(source).to_openfermion(),
        warmup=warmup,
        repeat=repeat,
    )
    baseline_result, baseline_timings = timed_call(
        lambda: rebuild_openfermion(source),
        warmup=warmup,
        repeat=repeat,
    )

    if not qubit_operator_close(fast_result, source):
        raise RuntimeError("Wolfgang OpenFermion round-trip changed operator semantics")
    if not qubit_operator_close(baseline_result, source):
        raise RuntimeError("OpenFermion baseline rebuild changed operator semantics")

    actual_terms = len(source.terms)
    return {
        "name": name,
        "dataset": {
            "num_qubits": num_qubits,
            "num_terms": actual_terms,
            "requested_terms": requested_terms,
            "term_weight_distribution": f"fixed term_weight={term_weight}",
            "duplicate_rate": 1.0 - (actual_terms / requested_terms if requested_terms else 1.0),
            "coefficient_dtype": "complex128",
            "random_seed": seed,
            "operator_construction_method": "deterministic OpenFermion QubitOperator terms",
        },
        "results": {
            "fastpauli_scalar_seconds": fast_timings["median"],
            "openfermion_baseline_seconds": baseline_timings["median"],
            "fastpauli_scalar_min_seconds": fast_timings["min"],
            "openfermion_baseline_min_seconds": baseline_timings["min"],
            "fastpauli_terms": len(fast_result.terms),
            "openfermion_baseline_terms": len(baseline_result.terms),
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    num_qubits = 8 if args.smoke else args.num_qubits
    small_terms = 16 if args.smoke else args.round_trip_terms
    large_terms = 64 if args.smoke else args.large_terms
    term_weight = 2 if args.smoke else args.term_weight
    warmup = 0 if args.smoke else args.warmup

    cases = [
        run_case(
            name="round_trip_conversion",
            num_qubits=num_qubits,
            requested_terms=small_terms,
            term_weight=term_weight,
            seed=args.seed,
            warmup=warmup,
            repeat=args.repeat,
        ),
        run_case(
            name="large_sparse_conversion",
            num_qubits=num_qubits,
            requested_terms=large_terms,
            term_weight=term_weight,
            seed=args.seed + 1,
            warmup=warmup,
            repeat=args.repeat,
        ),
    ]

    build_info = wolfgang._wolfgang_core._build_info()
    return {
        "benchmark": "openfermion_conversion",
        "git_commit": git_commit(),
        "command": command_string(),
        "environment": benchmark_environment(build_info, numpy_version=np.__version__),
        "fastpauli_version": wolfgang.__version__,
        "fastpauli_build_info": build_info,
        "openfermion_version": __import__("openfermion").__version__,
        "timing_policy": {
            "warmup": warmup,
            "repeat": args.repeat,
            "summary": "median seconds",
        },
        "baselines": ["Wolfgang scalar CPU", "OpenFermion QubitOperator rebuild"],
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-qubits", type=int, default=64)
    parser.add_argument("--round-trip-terms", type=int, default=256)
    parser.add_argument("--large-terms", type=int, default=10_000)
    parser.add_argument("--term-weight", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4219)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use tiny Phase 7 benchmark-smoke dimensions.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit the full benchmark report as JSON."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_qubits < 1:
        raise SystemExit("--num-qubits must be positive")
    if args.round_trip_terms < 1 or args.large_terms < 1:
        raise SystemExit("--round-trip-terms and --large-terms must be positive")
    if args.term_weight < 0:
        raise SystemExit("--term-weight must be non-negative")
    if args.warmup < 0 or args.repeat < 1:
        raise SystemExit("--warmup must be non-negative and --repeat must be positive")

    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print(f"Benchmark revision: {report['git_commit']}")
    print(f"Command: {report['command']}")
    print(f"Environment: {report['environment']}")
    print(f"Baselines: {', '.join(report['baselines'])}")
    for case in report["cases"]:
        dataset = case["dataset"]
        results = case["results"]
        print(
            f"{case['name']}: num_qubits={dataset['num_qubits']} "
            f"num_terms={dataset['num_terms']} "
            f"Wolfgang={results['fastpauli_scalar_seconds']:.6g}s "
            f"OpenFermion={results['openfermion_baseline_seconds']:.6g}s"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Deterministic Phase 4 simplify benchmark.

The benchmark compares the Wolfgang scalar simplify path with a pure-Python
packed-key baseline on low-duplicate and high-duplicate datasets. It is a
measurement harness, not a speedup claim; reports must be interpreted with the
environment metadata emitted by this script.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

import numpy as np
import wolfgang_quantum as wolfgang
from wolfgang_quantum import PauliSum

try:
    from _benchmark_metadata import (
        benchmark_environment,
        command_string,
        git_commit,
    )
except ModuleNotFoundError:
    from benchmarks._benchmark_metadata import (
        benchmark_environment,
        command_string,
        git_commit,
    )

PAULIS = np.asarray(["X", "Y", "Z"])


def make_label(rng: np.random.Generator, num_qubits: int, term_weight: int) -> str:
    chars = ["I"] * num_qubits
    active_qubits = rng.choice(num_qubits, size=min(term_weight, num_qubits), replace=False)
    for qubit in active_qubits:
        chars[num_qubits - 1 - int(qubit)] = str(rng.choice(PAULIS))
    return "".join(chars)


def generate_dataset(
    *,
    num_qubits: int,
    num_terms: int,
    term_weight: int,
    duplicate_rate: float,
    seed: int,
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    unique_terms = max(1, min(num_terms, round(num_terms * (1.0 - duplicate_rate))))
    pool: list[str] = []
    seen: set[str] = set()
    # Weighted labels can collide, especially in the tiny smoke benchmark. Keep
    # drawing until the requested unique pool is reached so duplicate_rate means
    # what the report says whenever the label space is large enough.
    attempts = 0
    max_attempts = max(1_000, unique_terms * 100)
    while len(pool) < unique_terms and attempts < max_attempts:
        attempts += 1
        label = make_label(rng, num_qubits, term_weight)
        if label not in seen:
            seen.add(label)
            pool.append(label)
    if not pool:
        raise RuntimeError("benchmark dataset generation produced no labels")
    unique_terms = len(pool)

    labels = list(pool)
    labels.extend(
        pool[int(index)] for index in rng.integers(0, unique_terms, size=num_terms - unique_terms)
    )
    rng.shuffle(labels)
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    actual_duplicate_rate = 1.0 - (len(set(labels)) / num_terms if num_terms else 1.0)
    metadata = {
        "num_qubits": num_qubits,
        "num_terms": num_terms,
        "term_weight": term_weight,
        "duplicate_rate": duplicate_rate,
        "actual_duplicate_rate": actual_duplicate_rate,
        "coefficient_dtype": "complex128",
        "random_seed": seed,
        "operator_construction_method": "deterministic weighted labels with duplicate pool",
    }
    return labels, np.asarray(coeffs, dtype=np.complex128), metadata


def packed_key(label: str) -> tuple[int, ...]:
    num_qubits = len(label)
    words = (num_qubits + 63) // 64
    x_words = [0] * words
    z_words = [0] * words
    for label_offset, pauli in enumerate(label):
        qubit = num_qubits - 1 - label_offset
        bit = 1 << (qubit % 64)
        word = qubit // 64
        if pauli in {"X", "Y"}:
            x_words[word] |= bit
        if pauli in {"Z", "Y"}:
            z_words[word] |= bit

    key: list[int] = []
    for word in range(words):
        key.extend((x_words[word], z_words[word]))
    return tuple(key)


def python_simplify(
    labels: list[str],
    coeffs: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> list[tuple[tuple[int, ...], complex]]:
    max_abs_input = max((abs(complex(coeff)) for coeff in coeffs), default=0.0)
    threshold = atol + rtol * max_abs_input
    accumulators: dict[tuple[int, ...], complex] = {}
    for label, coeff in zip(labels, coeffs, strict=True):
        key = packed_key(label)
        accumulators[key] = accumulators.get(key, 0.0 + 0.0j) + complex(coeff)

    return sorted(
        ((key, coeff) for key, coeff in accumulators.items() if abs(coeff) > threshold),
        key=lambda item: item[0],
    )


def fastpauli_pairs(op: PauliSum) -> list[tuple[tuple[int, ...], complex]]:
    labels, coeffs = op.to_labels()
    return [
        (packed_key(label), complex(coeff)) for label, coeff in zip(labels, coeffs, strict=True)
    ]


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


def run_case(
    *,
    name: str,
    duplicate_rate: float,
    num_qubits: int,
    num_terms: int,
    term_weight: int,
    seed: int,
    warmup: int,
    repeat: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    labels, coeffs, dataset = generate_dataset(
        num_qubits=num_qubits,
        num_terms=num_terms,
        term_weight=term_weight,
        duplicate_rate=duplicate_rate,
        seed=seed,
    )
    op = PauliSum.from_labels(labels, coeffs.tolist())

    fast_result, fast_timings = timed_call(
        lambda: op.simplify(atol=atol, rtol=rtol),
        warmup=warmup,
        repeat=repeat,
    )
    python_result, python_timings = timed_call(
        lambda: python_simplify(labels, coeffs, atol=atol, rtol=rtol),
        warmup=warmup,
        repeat=repeat,
    )

    fast_pairs = fastpauli_pairs(fast_result)
    if len(fast_pairs) != len(python_result):
        raise RuntimeError("Wolfgang and Python simplify produced different term counts")
    for (fast_key, fast_coeff), (python_key, python_coeff) in zip(
        fast_pairs,
        python_result,
        strict=True,
    ):
        if fast_key != python_key or not np.allclose(fast_coeff, python_coeff, rtol=0.0, atol=0.0):
            raise RuntimeError("Wolfgang and Python simplify produced different canonical results")

    return {
        "name": name,
        "dataset": dataset,
        "results": {
            "fastpauli_scalar_seconds": fast_timings["median"],
            "python_baseline_seconds": python_timings["median"],
            "fastpauli_scalar_min_seconds": fast_timings["min"],
            "python_baseline_min_seconds": python_timings["min"],
            "fastpauli_terms": fast_result.num_terms,
            "python_baseline_terms": len(python_result),
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    num_qubits = 8 if args.smoke else args.num_qubits
    num_terms = 32 if args.smoke else args.num_terms
    term_weight = 2 if args.smoke else args.term_weight
    warmup = 0 if args.smoke else args.warmup

    cases = [
        run_case(
            name="low_duplicate",
            duplicate_rate=0.05,
            num_qubits=num_qubits,
            num_terms=num_terms,
            term_weight=term_weight,
            seed=args.seed,
            warmup=warmup,
            repeat=args.repeat,
            atol=args.atol,
            rtol=args.rtol,
        ),
        run_case(
            name="high_duplicate",
            duplicate_rate=0.90,
            num_qubits=num_qubits,
            num_terms=num_terms,
            term_weight=term_weight,
            seed=args.seed + 1,
            warmup=warmup,
            repeat=args.repeat,
            atol=args.atol,
            rtol=args.rtol,
        ),
    ]
    build_info = wolfgang._wolfgang_core._build_info()
    return {
        "benchmark": "simplify",
        "git_commit": git_commit(),
        "command": command_string(),
        "environment": benchmark_environment(build_info, numpy_version=np.__version__),
        "fastpauli_version": wolfgang.__version__,
        "fastpauli_build_info": build_info,
        "timing_policy": {
            "warmup": warmup,
            "repeat": args.repeat,
            "summary": "median seconds",
        },
        "correctness_checks": {
            "enabled": True,
            "reference": "pure Python packed-key simplify",
            "failure_mode": "raises RuntimeError if term counts, canonical keys, or coefficients differ",
        },
        "baselines": ["Wolfgang scalar CPU", "pure Python packed-key reference"],
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-qubits", type=int, default=64)
    parser.add_argument("--num-terms", type=int, default=10_000)
    parser.add_argument("--term-weight", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use tiny Phase 4 benchmark-smoke dimensions.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit the full benchmark report as JSON."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_qubits < 1:
        raise SystemExit("--num-qubits must be positive")
    if args.num_terms < 1:
        raise SystemExit("--num-terms must be positive")
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
        results = case["results"]
        dataset = case["dataset"]
        print(
            f"{case['name']}: num_qubits={dataset['num_qubits']} "
            f"num_terms={dataset['num_terms']} duplicate_rate={dataset['duplicate_rate']} "
            f"Wolfgang={results['fastpauli_scalar_seconds']:.6g}s "
            f"Python={results['python_baseline_seconds']:.6g}s "
            f"terms={results['fastpauli_terms']}"
        )


if __name__ == "__main__":
    main()

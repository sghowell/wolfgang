#!/usr/bin/env python3
"""Deterministic Phase 5 multiplication benchmark.

The benchmark compares FastPauli scalar Pauli-sum multiplication with a
pure-Python dense-label reference. It exercises both a single-term phase case
and a small cross-product case, and reports the guardrail that caps
intermediate terms before allocation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

import fastpauli
import numpy as np
from fastpauli import PauliSum

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
LOCAL_PRODUCTS: dict[tuple[str, str], tuple[str, complex]] = {
    ("I", "I"): ("I", 1.0),
    ("I", "X"): ("X", 1.0),
    ("I", "Y"): ("Y", 1.0),
    ("I", "Z"): ("Z", 1.0),
    ("X", "I"): ("X", 1.0),
    ("Y", "I"): ("Y", 1.0),
    ("Z", "I"): ("Z", 1.0),
    ("X", "X"): ("I", 1.0),
    ("Y", "Y"): ("I", 1.0),
    ("Z", "Z"): ("I", 1.0),
    ("X", "Y"): ("Z", 1.0j),
    ("Y", "X"): ("Z", -1.0j),
    ("Y", "Z"): ("X", 1.0j),
    ("Z", "Y"): ("X", -1.0j),
    ("Z", "X"): ("Y", 1.0j),
    ("X", "Z"): ("Y", -1.0j),
}

def make_label(rng: np.random.Generator, num_qubits: int, term_weight: int) -> str:
    chars = ["I"] * num_qubits
    active_qubits = rng.choice(num_qubits, size=min(term_weight, num_qubits), replace=False)
    for qubit in active_qubits:
        chars[num_qubits - 1 - int(qubit)] = str(rng.choice(PAULIS))
    return "".join(chars)


def generate_operator(
    *,
    num_qubits: int,
    num_terms: int,
    term_weight: int,
    seed: int,
) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = [make_label(rng, num_qubits, term_weight) for _ in range(num_terms)]
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    return labels, np.asarray(coeffs, dtype=np.complex128)


def duplicate_rate(labels: list[str]) -> float:
    if not labels:
        return 0.0
    return 1.0 - (len(set(labels)) / len(labels))


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


def multiply_labels(lhs: str, rhs: str) -> tuple[str, complex]:
    if len(lhs) != len(rhs):
        raise ValueError("labels must have matching widths")

    phase = 1.0 + 0.0j
    output_chars: list[str] = []
    for lhs_pauli, rhs_pauli in zip(lhs, rhs, strict=True):
        out_pauli, local_phase = LOCAL_PRODUCTS[(lhs_pauli, rhs_pauli)]
        output_chars.append(out_pauli)
        phase *= local_phase
    return "".join(output_chars), phase


def python_multiply(
    lhs_labels: list[str],
    lhs_coeffs: np.ndarray,
    rhs_labels: list[str],
    rhs_coeffs: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    labels: list[str] = []
    coeffs: list[complex] = []
    for lhs_label, lhs_coeff in zip(lhs_labels, lhs_coeffs, strict=True):
        for rhs_label, rhs_coeff in zip(rhs_labels, rhs_coeffs, strict=True):
            out_label, phase = multiply_labels(lhs_label, rhs_label)
            labels.append(out_label)
            coeffs.append(complex(lhs_coeff) * complex(rhs_coeff) * phase)
    return labels, np.asarray(coeffs, dtype=np.complex128)


def python_multiply_simplified(
    lhs_labels: list[str],
    lhs_coeffs: np.ndarray,
    rhs_labels: list[str],
    rhs_coeffs: np.ndarray,
    *,
    atol: float = 1.0e-12,
    rtol: float = 0.0,
) -> list[tuple[tuple[int, ...], complex]]:
    labels, coeffs = python_multiply(lhs_labels, lhs_coeffs, rhs_labels, rhs_coeffs)
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
    lhs_labels: list[str],
    lhs_coeffs: np.ndarray,
    rhs_labels: list[str],
    rhs_coeffs: np.ndarray,
    dataset: dict[str, Any],
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    max_intermediate_terms = int(dataset["max_intermediate_terms"])
    simplify_output = bool(dataset["simplify_output"])
    lhs = PauliSum.from_labels(lhs_labels, lhs_coeffs.tolist())
    rhs = PauliSum.from_labels(rhs_labels, rhs_coeffs.tolist())

    fast_result, fast_timings = timed_call(
        lambda: lhs.matmul(
            rhs,
            simplify=simplify_output,
            max_intermediate_terms=max_intermediate_terms,
        ),
        warmup=warmup,
        repeat=repeat,
    )

    if simplify_output:
        python_result, python_timings = timed_call(
            lambda: python_multiply_simplified(lhs_labels, lhs_coeffs, rhs_labels, rhs_coeffs),
            warmup=warmup,
            repeat=repeat,
        )
        fast_labels, fast_coeffs = fast_result.to_labels()
        fast_pairs = [
            (packed_key(label), complex(coeff))
            for label, coeff in zip(fast_labels, fast_coeffs, strict=True)
        ]
        if len(fast_pairs) != len(python_result):
            raise RuntimeError("FastPauli and Python simplified multiply produced different term counts")
        for (fast_key, fast_coeff), (python_key, python_coeff) in zip(
            fast_pairs,
            python_result,
            strict=True,
        ):
            if fast_key != python_key or not np.allclose(fast_coeff, python_coeff, rtol=1.0e-12, atol=1.0e-12):
                raise RuntimeError("FastPauli and Python simplified multiply produced different outputs")
        python_terms = len(python_result)
    else:
        python_result, python_timings = timed_call(
            lambda: python_multiply(lhs_labels, lhs_coeffs, rhs_labels, rhs_coeffs),
            warmup=warmup,
            repeat=repeat,
        )
        fast_labels, fast_coeffs = fast_result.to_labels()
        python_labels, python_coeffs = python_result
        if fast_labels != python_labels:
            raise RuntimeError("FastPauli and Python multiply produced different labels")
        if not np.allclose(fast_coeffs, python_coeffs, rtol=1.0e-12, atol=1.0e-12):
            raise RuntimeError("FastPauli and Python multiply produced different coefficients")
        python_terms = len(python_labels)

    return {
        "name": name,
        "dataset": dataset,
        "results": {
            "fastpauli_scalar_seconds": fast_timings["median"],
            "python_baseline_seconds": python_timings["median"],
            "fastpauli_scalar_min_seconds": fast_timings["min"],
            "python_baseline_min_seconds": python_timings["min"],
            "fastpauli_terms": fast_result.num_terms,
            "python_baseline_terms": python_terms,
        },
    }


def build_single_term_case(args: argparse.Namespace) -> dict[str, Any]:
    num_qubits = 8 if args.smoke else args.num_qubits
    lhs_label = "X" + ("I" * (num_qubits - 1))
    rhs_label = "Y" + ("I" * (num_qubits - 1))
    dataset = {
        "num_qubits": num_qubits,
        "lhs_terms": 1,
        "rhs_terms": 1,
        "intermediate_terms": 1,
        "max_intermediate_terms": args.max_intermediate_terms,
        "term_weight_distribution": "fixed single active qubit",
        "duplicate_rate": 0.0,
        "lhs_duplicate_rate": 0.0,
        "rhs_duplicate_rate": 0.0,
        "coefficient_dtype": "complex128",
        "random_seed": "deterministic_single_term",
        "operator_construction_method": "explicit X @ Y phase fixture",
        "simplify_output": False,
    }
    return {
        "name": "single_term",
        "lhs_labels": [lhs_label],
        "lhs_coeffs": np.asarray([2.0 + 0.5j], dtype=np.complex128),
        "rhs_labels": [rhs_label],
        "rhs_coeffs": np.asarray([-0.75 + 1.25j], dtype=np.complex128),
        "dataset": dataset,
    }


def build_cross_product_case(args: argparse.Namespace) -> dict[str, Any]:
    num_qubits = 8 if args.smoke else args.num_qubits
    lhs_terms = 8 if args.smoke else args.lhs_terms
    rhs_terms = 6 if args.smoke else args.rhs_terms
    term_weight = 2 if args.smoke else args.term_weight
    lhs_labels, lhs_coeffs = generate_operator(
        num_qubits=num_qubits,
        num_terms=lhs_terms,
        term_weight=term_weight,
        seed=args.seed,
    )
    rhs_labels, rhs_coeffs = generate_operator(
        num_qubits=num_qubits,
        num_terms=rhs_terms,
        term_weight=term_weight,
        seed=args.seed + 1,
    )
    intermediate_terms = lhs_terms * rhs_terms
    lhs_duplicate_rate = duplicate_rate(lhs_labels)
    rhs_duplicate_rate = duplicate_rate(rhs_labels)
    dataset = {
        "num_qubits": num_qubits,
        "lhs_terms": lhs_terms,
        "rhs_terms": rhs_terms,
        "intermediate_terms": intermediate_terms,
        "max_intermediate_terms": args.max_intermediate_terms,
        "term_weight_distribution": f"fixed term_weight={term_weight}",
        "duplicate_rate": max(lhs_duplicate_rate, rhs_duplicate_rate),
        "duplicate_rate_scope": "max_operand_duplicate_rate",
        "lhs_duplicate_rate": lhs_duplicate_rate,
        "rhs_duplicate_rate": rhs_duplicate_rate,
        "coefficient_dtype": "complex128",
        "random_seed": args.seed,
        "operator_construction_method": "deterministic weighted labels",
        "simplify_output": False,
    }
    return {
        "name": "small_cross_product",
        "lhs_labels": lhs_labels,
        "lhs_coeffs": lhs_coeffs,
        "rhs_labels": rhs_labels,
        "rhs_coeffs": rhs_coeffs,
        "dataset": dataset,
    }


def build_simplified_duplicate_cross_product_case(args: argparse.Namespace) -> dict[str, Any]:
    num_qubits = 8 if args.smoke else args.num_qubits
    lhs_terms = 8 if args.smoke else args.lhs_terms
    rhs_terms = 6 if args.smoke else args.rhs_terms
    lhs_pool = [
        "X" + ("I" * (num_qubits - 1)),
        "Z" + ("I" * (num_qubits - 1)),
    ]
    rhs_pool = [
        "Y" + ("I" * (num_qubits - 1)),
        "I" * num_qubits,
    ]
    lhs_labels = [lhs_pool[index % len(lhs_pool)] for index in range(lhs_terms)]
    rhs_labels = [rhs_pool[index % len(rhs_pool)] for index in range(rhs_terms)]
    rng = np.random.default_rng(args.seed + 20)
    lhs_coeffs = rng.normal(size=lhs_terms) + 1j * rng.normal(size=lhs_terms)
    rhs_coeffs = rng.normal(size=rhs_terms) + 1j * rng.normal(size=rhs_terms)
    intermediate_terms = lhs_terms * rhs_terms
    dataset = {
        "num_qubits": num_qubits,
        "lhs_terms": lhs_terms,
        "rhs_terms": rhs_terms,
        "intermediate_terms": intermediate_terms,
        "max_intermediate_terms": args.max_intermediate_terms,
        "term_weight_distribution": "two repeated one-local Pauli pools",
        "duplicate_rate": max(duplicate_rate(lhs_labels), duplicate_rate(rhs_labels)),
        "duplicate_pressure": "intentional repeated product keys",
        "lhs_duplicate_rate": duplicate_rate(lhs_labels),
        "rhs_duplicate_rate": duplicate_rate(rhs_labels),
        "coefficient_dtype": "complex128",
        "random_seed": args.seed + 20,
        "operator_construction_method": "deterministic repeated labels with random coefficients",
        "simplify_output": True,
    }
    return {
        "name": "simplified_duplicate_cross_product",
        "lhs_labels": lhs_labels,
        "lhs_coeffs": np.asarray(lhs_coeffs, dtype=np.complex128),
        "rhs_labels": rhs_labels,
        "rhs_coeffs": np.asarray(rhs_coeffs, dtype=np.complex128),
        "dataset": dataset,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_intermediate_terms < 1:
        raise SystemExit("--max-intermediate-terms must be positive")

    warmup = 0 if args.smoke else args.warmup
    cases = []
    for case in (
        build_single_term_case(args),
        build_cross_product_case(args),
        build_simplified_duplicate_cross_product_case(args),
    ):
        cases.append(
            run_case(
                name=case["name"],
                lhs_labels=case["lhs_labels"],
                lhs_coeffs=case["lhs_coeffs"],
                rhs_labels=case["rhs_labels"],
                rhs_coeffs=case["rhs_coeffs"],
                dataset=case["dataset"],
                warmup=warmup,
                repeat=args.repeat,
            )
        )

    build_info = fastpauli._fastpauli_core._build_info()
    return {
        "benchmark": "multiply",
        "git_commit": git_commit(),
        "command": command_string(),
        "environment": benchmark_environment(build_info, numpy_version=np.__version__),
        "fastpauli_version": fastpauli.__version__,
        "fastpauli_build_info": build_info,
        "timing_policy": {
            "warmup": warmup,
            "repeat": args.repeat,
            "summary": "median seconds",
        },
        "correctness_checks": {
            "enabled": True,
            "reference": "pure Python dense-label multiplication",
            "failure_mode": "raises RuntimeError if output labels or coefficients differ",
        },
        "baselines": ["FastPauli scalar CPU", "pure Python dense-label reference"],
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-qubits", type=int, default=64)
    parser.add_argument("--lhs-terms", type=int, default=256)
    parser.add_argument("--rhs-terms", type=int, default=256)
    parser.add_argument("--term-weight", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2753)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--max-intermediate-terms", type=int, default=50_000_000)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use tiny Phase 5 benchmark-smoke dimensions.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the full benchmark report as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_qubits < 1:
        raise SystemExit("--num-qubits must be positive")
    if args.lhs_terms < 1 or args.rhs_terms < 1:
        raise SystemExit("--lhs-terms and --rhs-terms must be positive")
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
            f"lhs_terms={dataset['lhs_terms']} rhs_terms={dataset['rhs_terms']} "
            f"intermediate_terms={dataset['intermediate_terms']} "
            f"max_intermediate_terms={dataset['max_intermediate_terms']} "
            f"FastPauli={results['fastpauli_scalar_seconds']:.6g}s "
            f"Python={results['python_baseline_seconds']:.6g}s"
        )


if __name__ == "__main__":
    main()

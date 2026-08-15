#!/usr/bin/env python3
"""Deterministic Phase 6 commutation and grouping benchmark.

The benchmark measures Wolfgang's scalar pairwise commutation path, QWC
grouping, full-commutation grouping, and the dense commutation guardrail. It
also runs compact pure-Python references to verify semantic parity for the
measured datasets.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from typing import Any

import wolfgang_quantum as fastpauli
import numpy as np
from wolfgang_quantum import PauliSum

try:
    from _benchmark_metadata import benchmark_environment, command_string, git_commit
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


def pauli_weight(label: str) -> int:
    return sum(pauli != "I" for pauli in label)


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


def full_commutes(lhs: str, rhs: str) -> bool:
    parity = 0
    for lhs_pauli, rhs_pauli in zip(lhs, rhs, strict=True):
        parity ^= int((lhs_pauli in {"X", "Y"} and rhs_pauli in {"Z", "Y"})
                      != (lhs_pauli in {"Z", "Y"} and rhs_pauli in {"X", "Y"}))
    return parity == 0


def qwc_compatible(lhs: str, rhs: str) -> bool:
    for lhs_pauli, rhs_pauli in zip(lhs, rhs, strict=True):
        if lhs_pauli != "I" and rhs_pauli != "I" and lhs_pauli != rhs_pauli:
            return False
    return True


def python_commutation_matrix(lhs_labels: list[str], rhs_labels: list[str]) -> np.ndarray:
    return np.asarray(
        [[full_commutes(lhs, rhs) for rhs in rhs_labels] for lhs in lhs_labels],
        dtype=np.bool_,
    )


def python_group(labels: list[str], *, mode: str) -> list[list[str]]:
    predicate = qwc_compatible if mode == "qwc" else full_commutes
    ordered = sorted(
        range(len(labels)),
        key=lambda index: (-pauli_weight(labels[index]), packed_key(labels[index]), index),
    )
    groups: list[list[int]] = []
    for term in ordered:
        for group in groups:
            if all(predicate(labels[term], labels[existing]) for existing in group):
                group.append(term)
                break
        else:
            groups.append([term])
    return [[labels[index] for index in group] for group in groups]


def exported_group_labels(groups: list[PauliSum]) -> list[list[str]]:
    exported: list[list[str]] = []
    for group in groups:
        labels, _ = group.to_labels()
        exported.append(list(labels))
    return exported


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


def duplicate_rate(labels: list[str]) -> float:
    if not labels:
        return 0.0
    return 1.0 - (len(set(labels)) / len(labels))


def run_pairwise_case(args: argparse.Namespace) -> dict[str, Any]:
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
    lhs = PauliSum.from_labels(lhs_labels, lhs_coeffs.tolist())
    rhs = PauliSum.from_labels(rhs_labels, rhs_coeffs.tolist())

    fast_result, fast_timings = timed_call(
        lambda: lhs.commutes_with(rhs, max_commutation_matrix_entries=args.max_commutation_matrix_entries),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    python_result, python_timings = timed_call(
        lambda: python_commutation_matrix(lhs_labels, rhs_labels),
        warmup=args.warmup,
        repeat=args.repeat,
    )

    if not np.array_equal(fast_result, python_result):
        raise RuntimeError("Wolfgang and Python pairwise commutation produced different results")

    return {
        "name": "pairwise_commutation",
        "dataset": {
            "num_qubits": num_qubits,
            "lhs_terms": lhs_terms,
            "rhs_terms": rhs_terms,
            "matrix_entries": lhs_terms * rhs_terms,
            "max_commutation_matrix_entries": args.max_commutation_matrix_entries,
            "term_weight_distribution": f"fixed term_weight={term_weight}",
            "lhs_duplicate_rate": duplicate_rate(lhs_labels),
            "rhs_duplicate_rate": duplicate_rate(rhs_labels),
            "coefficient_dtype": "complex128",
            "random_seed": args.seed,
            "operator_construction_method": "deterministic weighted labels",
        },
        "results": {
            "fastpauli_scalar_seconds": fast_timings["median"],
            "python_baseline_seconds": python_timings["median"],
            "fastpauli_scalar_min_seconds": fast_timings["min"],
            "python_baseline_min_seconds": python_timings["min"],
            "commutation_entries": lhs_terms * rhs_terms,
        },
    }


def run_grouping_case(args: argparse.Namespace, *, mode: str) -> dict[str, Any]:
    num_qubits = 8 if args.smoke else args.num_qubits
    num_terms = 10 if args.smoke else args.group_terms
    term_weight = 2 if args.smoke else args.term_weight
    labels, coeffs = generate_operator(
        num_qubits=num_qubits,
        num_terms=num_terms,
        term_weight=term_weight,
        seed=args.seed + (10 if mode == "qwc" else 20),
    )
    op = PauliSum.from_labels(labels, coeffs.tolist())

    fast_result, fast_timings = timed_call(
        lambda: op.group_commuting(mode=mode, max_terms_for_graph=args.max_terms_for_graph),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    python_result, python_timings = timed_call(
        lambda: python_group(labels, mode=mode),
        warmup=args.warmup,
        repeat=args.repeat,
    )

    fast_labels = exported_group_labels(fast_result)
    if fast_labels != python_result:
        raise RuntimeError(f"Wolfgang and Python {mode} grouping produced different groups")

    return {
        "name": f"{mode}_grouping",
        "dataset": {
            "num_qubits": num_qubits,
            "num_terms": num_terms,
            "term_weight_distribution": f"fixed term_weight={term_weight}",
            "duplicate_rate": duplicate_rate(labels),
            "coefficient_dtype": "complex128",
            "random_seed": args.seed + (10 if mode == "qwc" else 20),
            "operator_construction_method": "deterministic weighted labels",
            "grouping_mode": mode,
            "strategy": "largest_first",
            "max_terms_for_graph": args.max_terms_for_graph,
        },
        "results": {
            "fastpauli_scalar_seconds": fast_timings["median"],
            "python_baseline_seconds": python_timings["median"],
            "fastpauli_scalar_min_seconds": fast_timings["min"],
            "python_baseline_min_seconds": python_timings["min"],
            "fastpauli_groups": len(fast_result),
            "python_baseline_groups": len(python_result),
        },
    }


def run_guardrail_case(args: argparse.Namespace) -> dict[str, Any]:
    lhs_terms = 3
    rhs_terms = 4
    max_entries = 11
    lhs = PauliSum(num_qubits=1, num_terms=lhs_terms)
    rhs = PauliSum(num_qubits=1, num_terms=rhs_terms)

    def fast_guardrail() -> str:
        try:
            lhs.commutes_with(rhs, max_commutation_matrix_entries=max_entries)
        except ValueError as error:
            return str(error)
        raise RuntimeError("commutation guardrail did not reject oversized dense output")

    def python_guardrail() -> bool:
        return lhs_terms * rhs_terms > max_entries

    fast_result, fast_timings = timed_call(fast_guardrail, warmup=args.warmup, repeat=args.repeat)
    python_result, python_timings = timed_call(
        python_guardrail,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    if not python_result or "max_commutation_matrix_entries" not in fast_result:
        raise RuntimeError("guardrail benchmark did not observe the expected rejection")

    return {
        "name": "guardrail_rejection",
        "dataset": {
            "num_qubits": 1,
            "lhs_terms": lhs_terms,
            "rhs_terms": rhs_terms,
            "matrix_entries": lhs_terms * rhs_terms,
            "max_commutation_matrix_entries": max_entries,
            "guardrail_scope": "dense pairwise commutation output",
        },
        "results": {
            "fastpauli_scalar_seconds": fast_timings["median"],
            "python_baseline_seconds": python_timings["median"],
            "fastpauli_scalar_min_seconds": fast_timings["min"],
            "python_baseline_min_seconds": python_timings["min"],
            "fastpauli_guardrail_rejected": True,
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.smoke:
        args.warmup = 0

    cases = [
        run_pairwise_case(args),
        run_grouping_case(args, mode="qwc"),
        run_grouping_case(args, mode="full"),
        run_guardrail_case(args),
    ]

    build_info = fastpauli._wolfgang_core._build_info()
    return {
        "benchmark": "grouping",
        "git_commit": git_commit(),
        "command": command_string(),
        "environment": benchmark_environment(build_info, numpy_version=np.__version__),
        "fastpauli_version": fastpauli.__version__,
        "fastpauli_build_info": build_info,
        "timing_policy": {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "summary": "median seconds",
        },
        "correctness_checks": {
            "enabled": True,
            "reference": "pure Python dense-label commutation, grouping, and guardrail reference",
            "failure_mode": "raises RuntimeError if commutation matrices, groups, or guardrail behavior differ",
        },
        "baselines": [
            "Wolfgang scalar CPU",
            "pure Python dense-label reference",
        ],
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-qubits", type=int, default=64)
    parser.add_argument("--lhs-terms", type=int, default=256)
    parser.add_argument("--rhs-terms", type=int, default=256)
    parser.add_argument("--group-terms", type=int, default=512)
    parser.add_argument("--term-weight", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3181)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--max-commutation-matrix-entries", type=int, default=100_000_000)
    parser.add_argument("--max-terms-for-graph", type=int, default=50_000)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use tiny Phase 6 benchmark-smoke dimensions.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the full benchmark report as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_qubits < 1:
        raise SystemExit("--num-qubits must be positive")
    if args.lhs_terms < 1 or args.rhs_terms < 1 or args.group_terms < 1:
        raise SystemExit("--lhs-terms, --rhs-terms, and --group-terms must be positive")
    if args.term_weight < 0:
        raise SystemExit("--term-weight must be non-negative")
    if args.warmup < 0 or args.repeat < 1:
        raise SystemExit("--warmup must be non-negative and --repeat must be positive")
    if args.max_commutation_matrix_entries < 1:
        raise SystemExit("--max-commutation-matrix-entries must be positive")
    if args.max_terms_for_graph < 0:
        raise SystemExit("--max-terms-for-graph must be non-negative")

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
        if case["name"] == "guardrail_rejection":
            print(
                f"{case['name']}: entries={dataset['matrix_entries']} "
                f"max_entries={dataset['max_commutation_matrix_entries']} "
                f"Wolfgang={results['fastpauli_scalar_seconds']:.6g}s"
            )
            continue
        print(
            f"{case['name']}: num_qubits={dataset['num_qubits']} "
            f"terms={dataset.get('num_terms', dataset.get('lhs_terms'))} "
            f"Wolfgang={results['fastpauli_scalar_seconds']:.6g}s "
            f"Python={results['python_baseline_seconds']:.6g}s"
        )


if __name__ == "__main__":
    main()

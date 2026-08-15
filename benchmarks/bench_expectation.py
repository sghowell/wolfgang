#!/usr/bin/env python3
"""Deterministic Phase 8 expectation-value benchmark.

The benchmark compares Wolfgang scalar CPU expectation kernels with direct
Python reference implementations for statevector and Z-count workloads. Cases
separate few-terms/large-statevector pressure from many-terms/small-statevector
pressure so later CPU-dispatch and CUDA work has stable baseline evidence.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
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
    diagonal_only: bool = False,
) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    labels: list[str] = []
    alphabet = np.asarray(["Z"]) if diagonal_only else PAULIS
    for _ in range(num_terms):
        chars = ["I"] * num_qubits
        active_qubits = rng.choice(num_qubits, size=min(term_weight, num_qubits), replace=False)
        for qubit in active_qubits:
            chars[num_qubits - 1 - int(qubit)] = str(rng.choice(alphabet))
        labels.append("".join(chars))
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    return labels, np.asarray(coeffs, dtype=np.complex128)


def duplicate_rate(labels: list[str]) -> float:
    if not labels:
        return 0.0
    return 1.0 - (len(set(labels)) / len(labels))


def normalized_state(num_qubits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=1 << num_qubits) + 1j * rng.normal(size=1 << num_qubits)
    psi = np.asarray(raw, dtype=np.complex128)
    return psi / np.linalg.norm(psi)


def masks_from_labels(labels: list[str]) -> list[tuple[int, int]]:
    masks: list[tuple[int, int]] = []
    for label in labels:
        x_mask = 0
        z_mask = 0
        num_qubits = len(label)
        for dense_index, pauli in enumerate(label):
            qubit = num_qubits - 1 - dense_index
            if pauli in {"X", "Y"}:
                x_mask |= 1 << qubit
            if pauli in {"Z", "Y"}:
                z_mask |= 1 << qubit
        masks.append((x_mask, z_mask))
    return masks


def python_statevector_expectation(
    labels: list[str],
    coeffs: np.ndarray,
    psi: np.ndarray,
) -> complex:
    result = 0.0 + 0.0j
    for (x_mask, z_mask), coeff in zip(masks_from_labels(labels), coeffs, strict=True):
        yz_phase = (1j) ** ((x_mask & z_mask).bit_count() % 4)
        term = 0.0 + 0.0j
        for basis in range(psi.size):
            phase = -yz_phase if (z_mask & basis).bit_count() & 1 else yz_phase
            term += np.conj(psi[basis ^ x_mask]) * phase * psi[basis]
        result += complex(coeff) * term
    return result


def generate_counts(num_qubits: int, *, rows: int, seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    counts: dict[str, int] = {}
    max_states = 1 << num_qubits
    for state in rng.choice(max_states, size=min(rows, max_states), replace=False):
        bitstring = format(int(state), f"0{num_qubits}b")
        counts[bitstring] = int(rng.integers(1, 1000))
    return counts


def python_z_count_expectation(
    labels: list[str],
    coeffs: np.ndarray,
    counts: dict[str, int],
) -> complex:
    total = float(sum(counts.values()))
    result = 0.0 + 0.0j
    for label, coeff in zip(labels, coeffs, strict=True):
        weighted = 0.0
        for bitstring, count in counts.items():
            sign = 1.0
            for pauli, bit in zip(label, bitstring, strict=True):
                if pauli == "Z" and bit == "1":
                    sign = -sign
            weighted += float(count) * sign
        result += complex(coeff) * (weighted / total)
    return result


def run_statevector_case(
    *,
    name: str,
    labels: list[str],
    coeffs: np.ndarray,
    psi: np.ndarray,
    dataset: dict[str, Any],
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    op = PauliSum.from_labels(labels, coeffs.tolist())
    fast_result, fast_timings = timed_call(
        lambda: op.expectation_statevector(psi),
        warmup=warmup,
        repeat=repeat,
    )
    python_result, python_timings = timed_call(
        lambda: python_statevector_expectation(labels, coeffs, psi),
        warmup=warmup,
        repeat=repeat,
    )
    if not np.allclose(fast_result, python_result, rtol=1.0e-11, atol=1.0e-11):
        raise RuntimeError("Wolfgang and Python statevector expectations differ")

    return {
        "name": name,
        "dataset": dataset,
        "results": {
            "fastpauli_scalar_seconds": fast_timings["median"],
            "python_baseline_seconds": python_timings["median"],
            "fastpauli_scalar_min_seconds": fast_timings["min"],
            "python_baseline_min_seconds": python_timings["min"],
            "expectation_real": float(np.real(fast_result)),
            "expectation_imag": float(np.imag(fast_result)),
        },
    }


def run_z_counts_case(
    *,
    labels: list[str],
    coeffs: np.ndarray,
    counts: dict[str, int],
    dataset: dict[str, Any],
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    op = PauliSum.from_labels(labels, coeffs.tolist())
    fast_result, fast_timings = timed_call(
        lambda: op.expectation_z_counts(counts),
        warmup=warmup,
        repeat=repeat,
    )
    python_result, python_timings = timed_call(
        lambda: python_z_count_expectation(labels, coeffs, counts),
        warmup=warmup,
        repeat=repeat,
    )
    if not np.allclose(fast_result, python_result, rtol=1.0e-12, atol=1.0e-12):
        raise RuntimeError("Wolfgang and Python Z-count expectations differ")

    return {
        "name": "z_counts",
        "dataset": dataset,
        "results": {
            "fastpauli_scalar_seconds": fast_timings["median"],
            "python_baseline_seconds": python_timings["median"],
            "fastpauli_scalar_min_seconds": fast_timings["min"],
            "python_baseline_min_seconds": python_timings["min"],
            "expectation_real": float(np.real(fast_result)),
            "expectation_imag": float(np.imag(fast_result)),
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")

    warmup = 0 if args.smoke else args.warmup
    few_terms_qubits = 8 if args.smoke else args.large_state_qubits
    few_terms = 4 if args.smoke else args.few_terms
    many_terms_qubits = 5 if args.smoke else args.small_state_qubits
    many_terms = 24 if args.smoke else args.many_terms
    diagonal_terms_qubits = 6 if args.smoke else args.small_state_qubits
    diagonal_terms = 24 if args.smoke else args.many_terms
    z_count_qubits = 8 if args.smoke else args.z_count_qubits
    z_count_terms = 8 if args.smoke else args.z_count_terms
    z_count_rows = 16 if args.smoke else args.z_count_rows
    few_state_seed = args.seed + 10
    many_state_seed = args.seed + 11
    z_counts_seed = args.seed + 12

    few_labels, few_coeffs = generate_operator(
        num_qubits=few_terms_qubits,
        num_terms=few_terms,
        term_weight=3,
        seed=args.seed,
    )
    many_labels, many_coeffs = generate_operator(
        num_qubits=many_terms_qubits,
        num_terms=many_terms,
        term_weight=2,
        seed=args.seed + 1,
    )
    diagonal_labels, diagonal_coeffs = generate_operator(
        num_qubits=diagonal_terms_qubits,
        num_terms=diagonal_terms,
        term_weight=3,
        seed=args.seed + 3,
        diagonal_only=True,
    )
    z_labels, z_coeffs = generate_operator(
        num_qubits=z_count_qubits,
        num_terms=z_count_terms,
        term_weight=3,
        seed=args.seed + 2,
        diagonal_only=True,
    )

    cases = [
        run_statevector_case(
            name="statevector_few_terms_large_state",
            labels=few_labels,
            coeffs=few_coeffs,
            psi=normalized_state(few_terms_qubits, few_state_seed),
            dataset={
                "num_qubits": few_terms_qubits,
                "num_terms": few_terms,
                "statevector_length": 1 << few_terms_qubits,
                "coefficient_dtype": "complex128",
                "statevector_dtype": "complex128",
                "term_weight_distribution": "fixed term_weight=3",
                "duplicate_rate": duplicate_rate(few_labels),
                "operator_random_seed": args.seed,
                "statevector_random_seed": few_state_seed,
                "operator_construction_method": "deterministic weighted labels",
            },
            warmup=warmup,
            repeat=args.repeat,
        ),
        run_statevector_case(
            name="statevector_many_terms_small_state",
            labels=many_labels,
            coeffs=many_coeffs,
            psi=normalized_state(many_terms_qubits, many_state_seed),
            dataset={
                "num_qubits": many_terms_qubits,
                "num_terms": many_terms,
                "statevector_length": 1 << many_terms_qubits,
                "coefficient_dtype": "complex128",
                "statevector_dtype": "complex128",
                "term_weight_distribution": "fixed term_weight=2",
                "duplicate_rate": duplicate_rate(many_labels),
                "operator_random_seed": args.seed + 1,
                "statevector_random_seed": many_state_seed,
                "operator_construction_method": "deterministic weighted labels",
            },
            warmup=warmup,
            repeat=args.repeat,
        ),
        run_statevector_case(
            name="statevector_diagonal_many_terms",
            labels=diagonal_labels,
            coeffs=diagonal_coeffs,
            psi=normalized_state(diagonal_terms_qubits, args.seed + 13),
            dataset={
                "num_qubits": diagonal_terms_qubits,
                "num_terms": diagonal_terms,
                "statevector_length": 1 << diagonal_terms_qubits,
                "coefficient_dtype": "complex128",
                "statevector_dtype": "complex128",
                "term_weight_distribution": "fixed diagonal term_weight=3",
                "duplicate_rate": duplicate_rate(diagonal_labels),
                "operator_family": "diagonal_z_only",
                "operator_random_seed": args.seed + 3,
                "statevector_random_seed": args.seed + 13,
                "operator_construction_method": "deterministic diagonal weighted labels",
            },
            warmup=warmup,
            repeat=args.repeat,
        ),
        run_z_counts_case(
            labels=z_labels,
            coeffs=z_coeffs,
            counts=generate_counts(z_count_qubits, rows=z_count_rows, seed=z_counts_seed),
            dataset={
                "num_qubits": z_count_qubits,
                "num_terms": z_count_terms,
                "count_rows": z_count_rows,
                "coefficient_dtype": "complex128",
                "term_weight_distribution": "fixed diagonal term_weight=3",
                "duplicate_rate": duplicate_rate(z_labels),
                "operator_random_seed": args.seed + 2,
                "counts_random_seed": z_counts_seed,
                "operator_construction_method": "deterministic diagonal weighted labels",
            },
            warmup=warmup,
            repeat=args.repeat,
        ),
    ]

    build_info = fastpauli._wolfgang_core._build_info()
    return {
        "benchmark": "expectation",
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
        "baselines": [
            "Wolfgang scalar CPU",
            "direct Python expectation reference",
        ],
        "correctness_checks": {
            "enabled": True,
            "reference": "direct Python statevector and Z-count expectation reference",
            "failure_mode": "raises RuntimeError if expectation values differ beyond tolerance",
        },
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--smoke", action="store_true", help="run tiny validation-sized cases")
    parser.add_argument("--repeat", type=int, default=5, help="timed repetitions per case")
    parser.add_argument("--warmup", type=int, default=1, help="untimed warmup repetitions")
    parser.add_argument("--seed", type=int, default=4211, help="deterministic RNG seed")
    parser.add_argument("--large-state-qubits", type=int, default=12)
    parser.add_argument("--few-terms", type=int, default=8)
    parser.add_argument("--small-state-qubits", type=int, default=7)
    parser.add_argument("--many-terms", type=int, default=512)
    parser.add_argument("--z-count-qubits", type=int, default=12)
    parser.add_argument("--z-count-terms", type=int, default=128)
    parser.add_argument("--z-count-rows", type=int, default=512)
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

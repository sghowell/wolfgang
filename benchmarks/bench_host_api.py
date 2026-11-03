"""Host PauliSum boundary and algebra timings.

Suites:

``boundary``
    Build and export the same operator through dense labels
    (``from_labels`` / ``to_labels``) and through packed arrays
    (``from_arrays`` / ``to_arrays``). ``from_labels`` with a NumPy
    coefficient array also measures the bulk coefficient conversion when
    compared across builds.

``commutator``
    Fused ``A.commutator(B)`` / ``A.anticommutator(B)`` against the composed
    ``(A @ B -/+ B @ A).simplify()`` on the same operands.

Variants within a case run in a randomized order each repetition. Fused and
composed results are checked for equivalence before timings are kept.
"""

from __future__ import annotations

import argparse
import json
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

BOUNDARY_CASES = ((24, 100_000), (24, 1_000_000), (100, 100_000), (100, 1_000_000))
COMMUTATOR_CASES = ((24, 500), (24, 2000), (100, 500), (100, 2000))
SMOKE_BOUNDARY = ((24, 1000),)
SMOKE_COMMUTATOR = ((24, 50),)


def random_labels(rng: np.random.Generator, num_qubits: int, num_terms: int) -> list[str]:
    letters = np.asarray(list("IXYZ"), dtype="<U1")
    chars = letters[rng.integers(0, 4, size=(num_terms, num_qubits))]
    return np.ascontiguousarray(chars).view(f"<U{num_qubits}").ravel().tolist()


def summarize(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)
    return {
        "median_seconds": float(np.median(ordered)),
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
        "raw_seconds": samples,
    }


def run_variants(
    variants: dict[str, Callable[[], Any]],
    order_rng: random.Random,
    *,
    warmup: int,
    repeat: int,
) -> dict[str, dict[str, Any]]:
    samples: dict[str, list[float]] = {name: [] for name in variants}
    for iteration in range(warmup + repeat):
        names = list(variants)
        order_rng.shuffle(names)
        for name in names:
            start = time.perf_counter()
            variants[name]()
            elapsed = time.perf_counter() - start
            if iteration >= warmup:
                samples[name].append(elapsed)
    return {name: summarize(values) for name, values in samples.items()}


def boundary_case(
    rng: np.random.Generator, order_rng: random.Random, num_qubits: int, num_terms: int, args: Any
) -> dict[str, Any]:
    labels = random_labels(rng, num_qubits, num_terms)
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    op = wolfgang_quantum.PauliSum.from_labels(labels, coeffs)
    variants: dict[str, Callable[[], Any]] = {
        "from_labels_numpy_coeffs": lambda: wolfgang_quantum.PauliSum.from_labels(labels, coeffs),
        "to_labels": op.to_labels,
    }
    if hasattr(op, "to_arrays"):
        x, z, packed_coeffs = (array.copy() for array in op.to_arrays())
        rebuilt = wolfgang_quantum.PauliSum.from_arrays(x, z, packed_coeffs, num_qubits)
        if not rebuilt.equiv(op, atol=0.0):
            raise RuntimeError("from_arrays round trip disagrees with from_labels")
        variants["from_arrays"] = lambda: wolfgang_quantum.PauliSum.from_arrays(
            x, z, packed_coeffs, num_qubits
        )
        variants["to_arrays"] = op.to_arrays
    return {
        "suite": "boundary",
        "num_qubits": num_qubits,
        "num_terms": num_terms,
        "variants": run_variants(variants, order_rng, warmup=args.warmup, repeat=args.repeat),
    }


def commutator_case(
    rng: np.random.Generator, order_rng: random.Random, num_qubits: int, num_terms: int, args: Any
) -> dict[str, Any]:
    def operand() -> wolfgang_quantum.PauliSum:
        coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
        return wolfgang_quantum.PauliSum.from_labels(
            random_labels(rng, num_qubits, num_terms), coeffs
        ).simplify()

    a = operand()
    b = operand()
    fused_c = a.commutator(b)
    fused_a = a.anticommutator(b)
    if not fused_c.equiv((a @ b - b @ a).simplify(), atol=1e-9):
        raise RuntimeError("fused commutator disagrees with composed products")
    if not fused_a.equiv((a @ b + b @ a).simplify(), atol=1e-9):
        raise RuntimeError("fused anticommutator disagrees with composed products")
    variants: dict[str, Callable[[], Any]] = {
        "fused_commutator": lambda: a.commutator(b),
        "composed_commutator": lambda: (a @ b - b @ a).simplify(),
        "fused_anticommutator": lambda: a.anticommutator(b),
        "composed_anticommutator": lambda: (a @ b + b @ a).simplify(),
    }
    rows = run_variants(variants, order_rng, warmup=args.warmup, repeat=args.repeat)
    return {
        "suite": "commutator",
        "num_qubits": num_qubits,
        "num_terms_per_operand": num_terms,
        "commutator_output_terms": fused_c.num_terms,
        "anticommutator_output_terms": fused_a.num_terms,
        "variants": rows,
        "commutator_speedup": rows["composed_commutator"]["median_seconds"]
        / rows["fused_commutator"]["median_seconds"],
        "anticommutator_speedup": rows["composed_anticommutator"]["median_seconds"]
        / rows["fused_anticommutator"]["median_seconds"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("boundary", "commutator", "all"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--label", default="")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    order_rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    if args.suite in ("boundary", "all"):
        for num_qubits, num_terms in SMOKE_BOUNDARY if args.smoke else BOUNDARY_CASES:
            rows.append(boundary_case(rng, order_rng, num_qubits, num_terms, args))
    if args.suite in ("commutator", "all") and hasattr(wolfgang_quantum.PauliSum, "commutator"):
        for num_qubits, num_terms in SMOKE_COMMUTATOR if args.smoke else COMMUTATOR_CASES:
            rows.append(commutator_case(rng, order_rng, num_qubits, num_terms, args))
    report = {
        "benchmark": "host_api",
        "label": args.label,
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

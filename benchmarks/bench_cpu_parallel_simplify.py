"""CPU simplify: scalar vs oneTBB vs auto dispatch, with a Qiskit baseline.

Within each repetition the variants run in a randomized order. The oneTBB result
must equal the scalar result structurally (bit for bit) before any timing is
kept; Qiskit's ``SparsePauliOp.simplify`` is timed on the same terms when Qiskit
is installed and is checked for the same canonical term set.
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

SELECTOR = "WOLFGANG_CPU_BACKEND"
CASES = [(q, n, d) for q in (24, 100, 130) for n in (2_000, 10_000, 30_000, 100_000, 1_000_000) for d in (0.0, 0.5, 0.9)]
SMOKE_CASES = [(24, 2_000, 0.5)]


def random_operator(rng: np.random.Generator, num_qubits: int, num_terms: int, duplicates: float):
    unique = max(1, int(num_terms * (1.0 - duplicates)))
    letters = np.asarray(list("IXYZ"), dtype="<U1")
    chars = letters[rng.integers(0, 4, size=(unique, num_qubits))]
    base = np.ascontiguousarray(chars).view(f"<U{num_qubits}").ravel()
    labels = np.concatenate([base, base[rng.integers(0, unique, num_terms - unique)]])
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    return labels, coeffs


def with_selector(selector: str, fn: Callable[[], Any]) -> Callable[[], Any]:
    def run() -> Any:
        os.environ[SELECTOR] = selector
        return fn()

    return run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    build = dict(core._build_info())
    tbb_compiled = "simplify" in build["optimized_cpu_kernels"].get("tbb", [])
    try:
        from qiskit.quantum_info import SparsePauliOp
    except ImportError:
        SparsePauliOp = None

    rng = np.random.default_rng(args.seed)
    order_rng = random.Random(args.seed)
    rows = []
    for num_qubits, num_terms, duplicates in SMOKE_CASES if args.smoke else CASES:
        labels, coeffs = random_operator(rng, num_qubits, num_terms, duplicates)
        op = wolfgang_quantum.PauliSum.from_labels(labels.tolist(), coeffs)
        variants: dict[str, Callable[[], Any]] = {"scalar": with_selector("scalar", op.simplify)}
        if tbb_compiled:
            variants["tbb"] = with_selector("tbb", op.simplify)
            variants["auto"] = with_selector("auto", op.simplify)
        scalar_result = variants["scalar"]()
        for name in ("tbb", "auto"):
            if name in variants and variants[name]() != scalar_result:
                raise RuntimeError(f"{name} simplify differs from scalar for {num_qubits}q/{num_terms}")
        if SparsePauliOp is not None:
            qiskit_op = SparsePauliOp(labels.tolist(), coeffs)
            qiskit_labels = sorted(qiskit_op.simplify(atol=1e-12).paulis.to_labels())
            if qiskit_labels != sorted(scalar_result.to_labels()[0]):
                raise RuntimeError("Qiskit simplify keeps a different term set")
            variants["qiskit"] = lambda q=qiskit_op: q.simplify(atol=1e-12)
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
                "output_terms": scalar_result.num_terms,
                "variants": {
                    name: {"median_seconds": float(np.median(values)), "raw_seconds": values}
                    for name, values in samples.items()
                },
            }
        )
    report = {
        "benchmark": "cpu_parallel_simplify",
        "command": command_string(),
        "git": git_provenance(),
        "seed": args.seed,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "tbb_simplify_compiled": tbb_compiled,
        "environment": benchmark_environment(build, numpy_version=np.__version__),
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

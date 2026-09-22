#!/usr/bin/env python3
"""Paired-run inputs for coefficient export and unchanged construction controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import wolfgang_quantum as wg
import wolfgang_quantum._wolfgang_core as core


def measure(*, terms: int, repeat: int, seed: int) -> dict:
    op = wg.PauliSum.from_labels(["X"] * terms, np.arange(terms, dtype=np.complex128))
    labels, coefficients = ["IX", "YZ"] * 64, [1, 2j] * 64
    small = wg.PauliSum.from_labels(["X", "X", "Y"], [1, 2, 0.5])
    functions = {
        "coefficient_export": (op.to_labels, 1),
        "construction_control": (lambda: wg.PauliSum.from_labels(labels, coefficients), 100),
        "small_simplify_guard": (small.simplify, 100),
    }
    # Check the exported buffer against an independent exact oracle before timing.
    np.testing.assert_array_equal(op.to_labels()[1], np.arange(terms, dtype=np.complex128))
    samples: dict[str, list[float]] = {name: [] for name in functions}
    for function, _ in functions.values():
        for _ in range(5):
            function()
    rng = random.Random(seed)
    for _ in range(repeat):
        names = list(functions)
        rng.shuffle(names)
        for name in names:
            function, batch = functions[name]
            start = time.perf_counter()
            for _ in range(batch):
                function()
            samples[name].append((time.perf_counter() - start) / batch)
    artifact = Path(core.__file__)
    return {
        "benchmark": "python_boundary", "terms": terms, "repeat": repeat, "seed": seed,
        "version": wg.__version__, "correct": True,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "native_source_sha256": core._build_info().get("native_source_sha256"),
        "environment": {
            "os": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
            "active_backend": core._build_info()["active_cpu_backend"],
            "cpu_cmake_options": core._build_info()["cpu_cmake_options"],
            "threads": "one calling thread; report oneTBB setting in cpu_cmake_options",
        },
        "cases": {name: {"seconds": values, "median": statistics.median(values)} for name, values in samples.items()},
        "boundary": "Python method including allocation and result destruction; prebuilt input",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms", type=int, default=100000)
    parser.add_argument("--repeat", type=int, default=21)
    parser.add_argument("--seed", type=int, default=29021)
    parser.add_argument("--baseline-python", type=Path, help="Run paired comparisons against this interpreter.")
    parser.add_argument("--reruns", type=int, default=9)
    args = parser.parse_args()
    if args.terms < 1 or args.repeat < 1:
        parser.error("terms and repeat must be positive")
    if args.baseline_python is None:
        print(json.dumps(measure(terms=args.terms, repeat=args.repeat, seed=args.seed), indent=2))
        return
    if args.reruns < 3:
        parser.error("paired comparisons require at least three reruns")
    rng = random.Random(args.seed)
    interpreters = {"baseline": str(args.baseline_python), "candidate": sys.executable}
    runs = []
    for index in range(args.reruns):
        order = list(interpreters)
        rng.shuffle(order)
        variants = {}
        for variant in order:
            result = subprocess.run([
                interpreters[variant], str(Path(__file__).resolve()), "--terms", str(args.terms),
                "--repeat", str(args.repeat), "--seed", str(args.seed + index),
            ], check=True, capture_output=True, text=True, env={**os.environ, "WOLFGANG_CPU_BACKEND": "scalar"})
            variants[variant] = json.loads(result.stdout)
        runs.append({"rerun": index + 1, "order": order, "variants": variants})
    summary = {}
    for case in runs[0]["variants"]["baseline"]["cases"]:
        means = {variant: statistics.fmean(run["variants"][variant]["cases"][case]["median"] for run in runs) for variant in interpreters}
        summary[case] = {**means, "candidate_baseline_ratio": means["candidate"] / means["baseline"]}
    print(json.dumps({"seed": args.seed, "runs": runs, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

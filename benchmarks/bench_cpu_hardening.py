#!/usr/bin/env python3
"""Run the CPU hardening benchmark suite with correctness checks enabled."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

try:
    from _benchmark_metadata import ROOT, command_string, git_commit
except ModuleNotFoundError:
    from benchmarks._benchmark_metadata import ROOT, command_string, git_commit


PROFILE_ARGS: dict[str, dict[str, list[str]]] = {
    "smoke": {
        "simplify": ["--smoke"],
        "multiply": ["--smoke"],
        "grouping": ["--smoke"],
        "expectation": ["--smoke"],
    },
    "default": {
        "simplify": [],
        "multiply": [],
        "grouping": [],
        "expectation": [],
    },
    "stress": {
        "simplify": [
            "--num-qubits",
            "65",
            "--num-terms",
            "20000",
            "--term-weight",
            "4",
        ],
        "multiply": [
            "--num-qubits",
            "65",
            "--lhs-terms",
            "256",
            "--rhs-terms",
            "256",
            "--term-weight",
            "4",
        ],
        "grouping": [
            "--num-qubits",
            "65",
            "--lhs-terms",
            "512",
            "--rhs-terms",
            "512",
            "--group-terms",
            "1024",
            "--term-weight",
            "4",
        ],
        "expectation": [
            "--large-state-qubits",
            "14",
            "--few-terms",
            "16",
            "--small-state-qubits",
            "8",
            "--many-terms",
            "1024",
            "--z-count-qubits",
            "16",
            "--z-count-terms",
            "512",
            "--z-count-rows",
            "2048",
        ],
    },
}


def benchmark_script(name: str) -> str:
    if name == "grouping":
        return "bench_grouping.py"
    return f"bench_{name}.py"


def case_summary(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for case in cases:
        results = case.get("results", {})
        summary.append(
            {
                "name": case.get("name", "unknown"),
                "fastpauli_scalar_seconds": results.get("fastpauli_scalar_seconds"),
                "fastpauli_scalar_min_seconds": results.get("fastpauli_scalar_min_seconds"),
                "python_baseline_seconds": results.get("python_baseline_seconds"),
                "dataset": case.get("dataset", {}),
            }
        )
    return summary


def run_operation(name: str, *, profile: str, repeat: int, warmup: int) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "benchmarks" / benchmark_script(name)),
        *PROFILE_ARGS[profile][name],
        "--repeat",
        str(repeat),
        "--json",
    ]
    if "--smoke" not in PROFILE_ARGS[profile][name]:
        command.extend(["--warmup", str(warmup)])

    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    operation: dict[str, Any] = {
        "benchmark": name,
        "command": " ".join(command),
        "returncode": completed.returncode,
        "correctness_checked": False,
        "correctness_checks": {},
        "stderr": completed.stderr,
        "cases": [],
    }
    if completed.returncode != 0:
        operation["stdout"] = completed.stdout
        return operation

    report = json.loads(completed.stdout)
    correctness_checks = report.get("correctness_checks", {})
    operation["correctness_checks"] = correctness_checks
    operation["correctness_checked"] = bool(correctness_checks.get("enabled"))
    operation["cases"] = case_summary(report["cases"])
    operation["environment"] = report.get("environment", {})
    operation["fastpauli_build_info"] = report.get("fastpauli_build_info", {})
    return operation


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    operations = [
        run_operation(name, profile=args.profile, repeat=args.repeat, warmup=args.warmup)
        for name in ("simplify", "multiply", "grouping", "expectation")
    ]
    failed = [
        operation
        for operation in operations
        if operation["returncode"] != 0 or not operation["correctness_checked"]
    ]
    if failed:
        first = failed[0]
        raise SystemExit(
            f"{first['benchmark']} benchmark failed or omitted correctness metadata "
            f"with exit code {first['returncode']}\n"
            f"{first['stderr']}"
        )

    return {
        "benchmark": "cpu_hardening",
        "git_commit": git_commit(),
        "command": command_string(),
        "profile": args.profile,
        "timing_policy": {
            "repeat": args.repeat,
            "warmup": 0 if args.profile == "smoke" else args.warmup,
            "summary": "median seconds from operation benchmark reports",
        },
        "operations": operations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_ARGS),
        default="default",
        help="benchmark size profile",
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")

    report = build_report(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

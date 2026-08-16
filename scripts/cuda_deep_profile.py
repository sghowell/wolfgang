#!/usr/bin/env python3
"""Plan or run the deep CUDA profiling ladder for FastPauli.

The script is intentionally orchestration-only: it does not interpret profiler
metrics or make performance claims. It emits the exact commands needed for an
H100-style hillclimb run so reports can preserve command provenance, then can
optionally execute the same ladder and collect stdout/stderr under one output
directory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "cuda_deep_profile"
DEFAULT_CUDA_PATH = "/usr/local/cuda-12.9/bin"
DEFAULT_ARCHITECTURES = "90"

OPERATIONS = (
    "simplify_duplicate_pressure",
    "statevector_expectation",
    "pairwise_commutation",
    "matmul_product_generation_simplify",
)

KERNEL_REGEX_BY_OPERATION: dict[str, str | None] = {
    # The benchmark subprocess already isolates one operation per NCU pass.
    # Profiling several operation-local launches is more reliable for
    # CCCL/Thrust-heavy paths because their generated kernel names are
    # toolkit-version dependent.
    "simplify_duplicate_pressure": None,
    "statevector_expectation": "expectation_statevector_.*kernel",
    "pairwise_commutation": "commutation_kernel",
    "matmul_product_generation_simplify": None,
}

NCU_LAUNCH_COUNT_BY_OPERATION = {
    "simplify_duplicate_pressure": 16,
    "statevector_expectation": 4,
    "pairwise_commutation": 1,
    "matmul_product_generation_simplify": 24,
}

COMPETITOR_PACKAGES = {
    "cpu": ("qiskit>=1.0", "openfermion>=1.7.1"),
    "gpu": ("cupy-cuda12x", "cuquantum-python-cu12", "cudaq", "qiskit-aer-gpu"),
}


@dataclass(frozen=True)
class Step:
    name: str
    category: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    output_file: Path | None = None
    required: bool = True

    def command_string(self) -> str:
        env_parts = [f"{key}={shlex.quote(value)}" for key, value in sorted(self.env.items())]
        command_parts = [shlex.quote(part) for part in self.command]
        return " ".join((*env_parts, *command_parts))

    def as_dict(self) -> dict[str, Any]:
        item: dict[str, Any] = {
            "name": self.name,
            "category": self.category,
            "command": self.command_string(),
            "required": self.required,
        }
        if self.output_file is not None:
            item["output_file"] = str(self.output_file)
        return item


def selected_operations(values: list[str]) -> list[str]:
    if not values:
        return list(OPERATIONS)
    selected: list[str] = []
    for value in values:
        for raw_item in value.split(","):
            item = raw_item.strip()
            if item:
                selected.append(item)
    unknown = [operation for operation in selected if operation not in OPERATIONS]
    if unknown:
        raise SystemExit(f"unknown operation(s): {', '.join(sorted(unknown))}")
    return selected


def base_env(args: argparse.Namespace) -> dict[str, str]:
    path_value = os.environ.get("PATH", "")
    return {
        "PATH": f"{args.cuda_bin}:{path_value}",
        "WOLFGANG_CUDA_ARCHITECTURES": args.cuda_architectures,
    }


def python_command(args: argparse.Namespace) -> str:
    return args.python or sys.executable


def report_path(output_root: Path, name: str) -> Path:
    return output_root / name


def build_steps(args: argparse.Namespace) -> list[Step]:
    operations = selected_operations(args.operation)
    output_root = Path(args.output_root)
    py = python_command(args)
    env = base_env(args)
    cuda_build_env = {
        **env,
        "WOLFGANG_VALIDATE_CUDA": "1",
        "CUDACXX": str(Path(args.cuda_bin) / "nvcc"),
        "CUDAHOSTCXX": args.cuda_host_compiler,
    }

    steps: list[Step] = [
        Step(
            name="cuda validation",
            category="validation",
            command=[py, "scripts/validate.py"],
            env=cuda_build_env,
            output_file=report_path(output_root, "validation.log"),
            required=True,
        ),
        Step(
            name="cuda scaling benchmark",
            category="benchmark",
            command=[
                py,
                "benchmarks/bench_cuda_scaling.py",
                "--profile",
                args.profile,
                "--operation",
                ",".join(operations),
                "--repeat",
                str(args.repeat),
                "--warmup",
                str(args.warmup),
                "--json",
            ],
            env=env,
            output_file=report_path(output_root, f"cuda_scaling_{args.profile}.json"),
            required=True,
        ),
        Step(
            name="nsys cuda api timeline",
            category="profile",
            command=[
                "nsys",
                "profile",
                "--force-overwrite=true",
                "--stats=true",
                "--trace=cuda,nvtx,osrt",
                f"--output={report_path(output_root, 'nsys_cuda_api_timeline')}",
                py,
                "benchmarks/bench_cuda_scaling.py",
                "--profile",
                args.profile,
                "--operation",
                ",".join(operations),
                "--repeat",
                "1",
                "--warmup",
                "0",
                "--json",
            ],
            env=env,
            output_file=report_path(output_root, "nsys_cuda_api_timeline.stdout"),
            required=args.require_profiler_artifacts,
        ),
    ]

    for operation in operations:
        kernel_filter = KERNEL_REGEX_BY_OPERATION[operation]
        ncu_command = [
            "ncu",
            "--target-processes",
            "all",
            "--set",
            "detailed",
            "--section",
            "SpeedOfLight",
            "--section",
            "MemoryWorkloadAnalysis",
            "--section",
            "LaunchStats",
            "--section",
            "Occupancy",
            "--section",
            "SchedulerStats",
            "--section",
            "WarpStateStats",
            "--kernel-name-base",
            "function",
        ]
        if kernel_filter is not None:
            ncu_command.extend(["--kernel-name", f"regex:{kernel_filter}"])
        ncu_command.extend(
            [
                "--launch-count",
                str(NCU_LAUNCH_COUNT_BY_OPERATION[operation]),
                "--force-overwrite",
                "--export",
                str(report_path(output_root, f"ncu_{operation}_detailed")),
                py,
                "benchmarks/bench_cuda_scaling.py",
                "--profile",
                args.profile,
                "--operation",
                operation,
                "--repeat",
                "1",
                "--warmup",
                "0",
                "--json",
            ]
        )
        steps.append(
            Step(
                name=f"ncu {operation} detailed",
                category="profile",
                command=ncu_command,
                env=env,
                output_file=report_path(output_root, f"ncu_{operation}_detailed.stdout"),
                required=args.require_profiler_artifacts,
            )
        )

    for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
        steps.append(
            Step(
                name=f"compute sanitizer {tool}",
                category="sanitizer",
                command=[
                    "compute-sanitizer",
                    "--tool",
                    tool,
                    "--error-exitcode",
                    "99",
                    py,
                    "-m",
                    "pytest",
                    "tests/test_phase11_cuda_kernels.py",
                    "-q",
                ],
                env=env,
                output_file=report_path(output_root, f"compute_sanitizer_{tool}.log"),
                required=(tool == "memcheck"),
            )
        )

    extension_path = resolve_extension_path()
    steps.extend(
        [
            Step(
                name="cuobjdump sass inventory",
                category="binary_inspection",
                command=["cuobjdump", "--dump-sass", extension_path],
                env=env,
                output_file=report_path(output_root, "cuobjdump_sass.txt"),
                required=False,
            ),
            Step(
                name="cuobjdump ptx inventory",
                category="binary_inspection",
                command=["cuobjdump", "--dump-ptx", extension_path],
                env=env,
                output_file=report_path(output_root, "cuobjdump_ptx.txt"),
                required=False,
            ),
            Step(
                name="nvdisasm sass listing",
                category="binary_inspection",
                command=["nvdisasm", "--print-line-info", extension_path],
                env=env,
                output_file=report_path(output_root, "nvdisasm_sass.txt"),
                required=False,
            ),
        ]
    )

    steps.extend(competitor_steps(args, output_root, py, env))
    return steps


def competitor_steps(
    args: argparse.Namespace,
    output_root: Path,
    py: str,
    env: dict[str, str],
) -> list[Step]:
    if args.competitor_set == "none":
        return []

    packages: list[str] = []
    if args.competitor_set in {"cpu", "all"}:
        packages.extend(COMPETITOR_PACKAGES["cpu"])
    if args.competitor_set in {"gpu", "all"}:
        packages.extend(COMPETITOR_PACKAGES["gpu"])

    return [
        Step(
            name=f"install {args.competitor_set} competitor packages",
            category="competitor_setup",
            command=[py, "-m", "pip", "install", *packages],
            env=env,
            output_file=report_path(output_root, f"install_{args.competitor_set}_competitors.log"),
            required=False,
        ),
        Step(
            name="competitive baseline benchmark",
            category="benchmark",
            command=[
                py,
                "benchmarks/bench_competitive_baselines.py",
                "--repeat",
                str(args.repeat),
                "--warmup",
                str(args.warmup),
                "--json",
            ],
            env=env,
            output_file=report_path(output_root, "competitive_baselines.json"),
            required=False,
        ),
    ]


def resolve_extension_path() -> str:
    try:
        spec = importlib.util.find_spec("wolfgang_quantum._wolfgang_core")
    except ModuleNotFoundError:
        return "<wolfgang_extension_path>"
    if spec is None or spec.origin is None:
        return "<wolfgang_extension_path>"
    return spec.origin


def run_step(step: Step) -> dict[str, Any]:
    executable = step.command[0]
    if shutil.which(executable, path=step.env.get("PATH")) is None and not Path(executable).exists():
        status = "missing_executable"
        return {**step.as_dict(), "status": status, "returncode": None}

    env = {**os.environ, **step.env}
    if step.output_file is not None:
        step.output_file.parent.mkdir(parents=True, exist_ok=True)
        with step.output_file.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                step.command,
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
    else:
        completed = subprocess.run(step.command, cwd=ROOT, env=env, check=False, text=True)

    return {
        **step.as_dict(),
        "status": "success" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    operations = selected_operations(args.operation)
    steps = build_steps(args)
    execute = bool(args.execute and not args.dry_run)
    report: dict[str, Any] = {
        "script": "scripts/cuda_deep_profile.py",
        "execute": execute,
        "profile": args.profile,
        "operations": operations,
        "output_root": str(Path(args.output_root)),
        "cuda_architectures": args.cuda_architectures,
        "competitor_set": args.competitor_set,
        "require_profiler_artifacts": args.require_profiler_artifacts,
        "steps": [step.as_dict() for step in steps],
    }

    if not execute:
        return report

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for step in steps:
        result = run_step(step)
        results.append(result)
        if step.required and result["status"] != "success" and not args.continue_on_error:
            break
    report["results"] = results
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="emit commands without running them")
    mode.add_argument("--execute", action="store_true", help="run the profiling ladder")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--profile",
        choices=("smoke", "default", "stress", "extreme"),
        default="default",
        help="bench_cuda_scaling.py profile for profiling commands",
    )
    parser.add_argument(
        "--operation",
        action="append",
        default=[],
        help="operation name or comma-separated operation list; defaults to all operations",
    )
    parser.add_argument("--repeat", type=int, default=3, help="timed repetitions for benchmarks")
    parser.add_argument("--warmup", type=int, default=1, help="untimed warmups for benchmarks")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python", default=sys.executable, help="Python executable to profile")
    parser.add_argument("--cuda-bin", default=DEFAULT_CUDA_PATH, help="CUDA toolkit bin directory")
    parser.add_argument("--cuda-architectures", default=DEFAULT_ARCHITECTURES)
    parser.add_argument("--cuda-host-compiler", default="/usr/bin/g++")
    parser.add_argument(
        "--competitor-set",
        choices=("none", "cpu", "gpu", "all"),
        default="none",
        help="add optional open-source competitor install and benchmark commands",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue after a required step fails when --execute is used",
    )
    parser.add_argument(
        "--require-profiler-artifacts",
        action="store_true",
        help=(
            "mark Nsight Systems and Nsight Compute steps as required; use for completion "
            "or exhaustion runs after profiler permissions are configured"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")

    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for step in report["steps"]:
            print(f"[{step['category']}] {step['name']}")
            print(f"  {step['command']}")
            if "output_file" in step:
                print(f"  output: {step['output_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Print the reproducible ROCm Campaign 7 release-support command lane."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = "docs/benchmarks/data/rocm_mi300x_campaign7_2026-04-30"
RAW_ROOT = f"{DATA_ROOT}/raw"
LOG_ROOT = f"{DATA_ROOT}/logs"
PROFILER_ROOT = f"{DATA_ROOT}/profiler"
ROCM_PATH = "PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH"
HIP_BUILD_FLAGS = (
    "--config-settings=cmake.define.WOLFGANG_ENABLE_HIP=ON "
    "--config-settings=cmake.define.WOLFGANG_HIP_ARCHITECTURES=gfx942"
)


@dataclass(frozen=True)
class LaneCommand:
    name: str
    purpose: str
    command: str
    artifact: str


def lane_commands() -> list[LaneCommand]:
    python = ".venv/bin/python"
    return [
        LaneCommand(
            name="host-inventory",
            purpose="Capture git revision, OS, CPU, GPU, ROCm runtime, and HIP compiler metadata.",
            command=(
                f"mkdir -p {LOG_ROOT} {RAW_ROOT} {PROFILER_ROOT} && "
                f"(git rev-parse HEAD; hostname; uname -a; cat /etc/os-release; "
                "python --version; lscpu; "
                f"{ROCM_PATH} rocminfo; {ROCM_PATH} amd-smi static; "
                f"{ROCM_PATH} hipcc --version) > {LOG_ROOT}/host_inventory_mi300x.log 2>&1"
            ),
            artifact=f"{LOG_ROOT}/host_inventory_mi300x.log",
        ),
        LaneCommand(
            name="package-metadata",
            purpose="Record Python package state for the build runner.",
            command=f"{python} -m pip freeze > {LOG_ROOT}/python_packages_mi300x.txt",
            artifact=f"{LOG_ROOT}/python_packages_mi300x.txt",
        ),
        LaneCommand(
            name="cpu-only-control",
            purpose="Validate that the CPU-only lane does not require ROCm.",
            command="python scripts/validate.py > "
            f"{LOG_ROOT}/cpu_only_control_validate.log 2>&1",
            artifact=f"{LOG_ROOT}/cpu_only_control_validate.log",
        ),
        LaneCommand(
            name="hip-source-build",
            purpose="Build the HIP source lane for MI300X gfx942.",
            command=(
                f"{ROCM_PATH} {python} -m pip install -e .[test] {HIP_BUILD_FLAGS} "
                f"> {LOG_ROOT}/hip_source_build_mi300x.log 2>&1"
            ),
            artifact=f"{LOG_ROOT}/hip_source_build_mi300x.log",
        ),
        LaneCommand(
            name="hip-pytest",
            purpose="Run retained HIP correctness tests.",
            command=(
                f"{ROCM_PATH} {python} -m pytest tests/test_phase12_rocm_foundation.py -q "
                f"> {LOG_ROOT}/hip_pytest_mi300x.log 2>&1"
            ),
            artifact=f"{LOG_ROOT}/hip_pytest_mi300x.log",
        ),
        LaneCommand(
            name="cuda-hip-rejection",
            purpose="Validate the CUDA+HIP configure-time rejection.",
            command=(
                f"{python} -m cmake -S . -B /tmp/wolfgang-campaign7-cuda-hip-reject "
                "-DWOLFGANG_ENABLE_CUDA=ON -DWOLFGANG_ENABLE_HIP=ON "
                f"> {LOG_ROOT}/cuda_hip_rejection.log 2>&1; "
                f"status=$?; echo exit_code=$status >> {LOG_ROOT}/cuda_hip_rejection.log; "
                f"test $status -ne 0; "
                f"grep -q 'cannot both be ON' {LOG_ROOT}/cuda_hip_rejection.log"
            ),
            artifact=f"{LOG_ROOT}/cuda_hip_rejection.log",
        ),
        LaneCommand(
            name="campaign7-release-smoke",
            purpose="Benchmark retained transfer, commutation, compact-consumer, simplify, expectation, and matmul paths.",
            command=(
                f"{ROCM_PATH} {python} benchmarks/bench_rocm_kernels.py "
                "--profile campaign7-release-smoke --repeat 5 --warmup 2 --json "
                f"--output {RAW_ROOT}/rocm_campaign7_release_smoke_mi300x.json"
            ),
            artifact=f"{RAW_ROOT}/rocm_campaign7_release_smoke_mi300x.json",
        ),
        LaneCommand(
            name="campaign7-duplicate-pressure",
            purpose="Stress retained simplify and matmul paths under larger duplicate-pressure cases.",
            command=(
                f"{ROCM_PATH} {python} benchmarks/bench_rocm_kernels.py "
                "--profile campaign7-duplicate-pressure --repeat 5 --warmup 2 --json "
                f"--output {RAW_ROOT}/rocm_campaign7_duplicate_pressure_mi300x.json"
            ),
            artifact=f"{RAW_ROOT}/rocm_campaign7_duplicate_pressure_mi300x.json",
        ),
        LaneCommand(
            name="campaign7-profiler",
            purpose="Emit correctness-checked profiler-smoke rows before rocprof wrapping.",
            command=(
                f"{ROCM_PATH} {python} benchmarks/bench_rocm_kernels.py "
                "--profile campaign7-profiler --repeat 3 --warmup 1 --json "
                f"--output {RAW_ROOT}/rocm_campaign7_profiler_mi300x.json"
            ),
            artifact=f"{RAW_ROOT}/rocm_campaign7_profiler_mi300x.json",
        ),
        LaneCommand(
            name="rocprof",
            purpose="Capture rocprof HIP trace and stats for representative retained operations.",
            command=(
                f"{ROCM_PATH} rocprof -d {PROFILER_ROOT} --hip-trace --stats "
                f"{python} benchmarks/bench_rocm_kernels.py --profile campaign7-profiler "
                "--repeat 1 --warmup 0 --json "
                f"--output {RAW_ROOT}/rocm_campaign7_profiler_rocprof_mi300x.json "
                f"> {PROFILER_ROOT}/rocprof_campaign7.log 2>&1"
            ),
            artifact=PROFILER_ROOT,
        ),
        LaneCommand(
            name="render-assets",
            purpose="Render Campaign 7 summary JSON and SVG plots.",
            command=(
                "python scripts/render_rocm_campaign7_assets.py "
                f"--data-dir {DATA_ROOT} --plot-dir docs/benchmarks/plots"
            ),
            artifact=f"{DATA_ROOT}/summary.json",
        ),
        LaneCommand(
            name="report-validation",
            purpose="Run local validation after report and docs updates.",
            command="python scripts/validate.py",
            artifact="validation output",
        ),
    ]


def render_text(commands: list[LaneCommand]) -> str:
    lines = [
        "# Wolfgang ROCm Campaign 7 Release-Support Lane",
        "",
        f"evidence_root: {DATA_ROOT}",
        "hip_environment: WOLFGANG_ENABLE_HIP=ON WOLFGANG_HIP_ARCHITECTURES=gfx942",
        "",
    ]
    for index, item in enumerate(commands, start=1):
        lines.extend(
            [
                f"## {index}. {item.name}",
                f"purpose: {item.purpose}",
                f"artifact: {item.artifact}",
                "command:",
                item.command,
                "",
            ]
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help="print the command lane in human-readable form",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the command lane as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = lane_commands()
    if args.json:
        print(json.dumps([asdict(item) for item in commands], indent=2))
        return
    print(render_text(commands))


if __name__ == "__main__":
    main()

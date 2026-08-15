#!/usr/bin/env python3
"""Emit and record ROCm Campaign 8 architecture-readiness commands.

Campaign 8 is a planning-and-harness slice. This script intentionally avoids
claiming ROCm runtime support from a local CPU-only machine. It records the
commands and terminal statuses needed to decide when later ROCm implementation
or packaging work may start.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from archive_portability import resolve_source_snapshot

ROOT = Path(__file__).resolve().parents[1]

CAMPAIGN = "rocm_campaign8_architecture_readiness"
DEFAULT_EVIDENCE_DIR = (
    ROOT / "docs/benchmarks/data/rocm_campaign8_architecture_readiness_2026-05-01"
)

TERMINAL_STATUSES = {
    "backend_neutral_object_model": "accepted_for_future_implementation",
    "simultaneous_cuda_hip_source_builds": "unavailable",
    "multi_gpu_rocm_execution": "out_of_scope_with_next_trigger",
    "non_mi300x_amd_portability": "blocked_external",
    "rocm_wheel_packaging_design": "accepted_for_future_implementation",
    "rocm_ci_hardware_policy": "accepted_for_future_implementation",
    "rocm_clean_machine_install_tests": "accepted_for_future_implementation",
    "rocprofv3_migration": "accepted_for_future_implementation",
    "legacy_rocprof_retention": "retained",
    "external_hip_statevector_contract": "accepted_for_future_implementation",
    "hip_dlpack_reconsideration_contract": "accepted_for_future_implementation",
    "hip_cuda_array_interface_policy": "rejected_with_evidence",
    "public_streams_policy": "rejected_with_evidence",
    "public_graphs_policy": "rejected_with_evidence",
    "public_workspaces_policy": "rejected_with_evidence",
    "targeted_rocm_performance_reopen": "accepted_for_future_implementation",
    "source_build_release_lane_retention": "retained",
}


@dataclass(frozen=True)
class ReadinessCommand:
    label: str
    command: str
    status: str
    notes: str


def git_revision(*, source_commit: str | None = None) -> str:
    return resolve_source_snapshot(ROOT, source_commit=source_commit)["commit"]


def command_inventory(evidence_dir: Path) -> list[ReadinessCommand]:
    raw = evidence_dir / "raw"
    profiler = evidence_dir / "profiler"
    return [
        ReadinessCommand(
            "host-inventory",
            "python -m platform && python -c 'import platform; print(platform.platform())'",
            "retained",
            "Local and accelerator hosts must record OS, Python, CPU, GPU, runtime, and compiler inventory.",
        ),
        ReadinessCommand(
            "cpu-only-control",
            "uv run python scripts/validate.py",
            "retained",
            "CPU-only validation remains the first control lane for ROCm planning work.",
        ),
        ReadinessCommand(
            "cuda-hip-rejection",
            (
                "uv run python -m cmake -S . -B /tmp/fastpauli-campaign8-cuda-hip-reject "
                "-DWOLFGANG_ENABLE_CUDA=ON -DWOLFGANG_ENABLE_HIP=ON"
            ),
            "retained",
            "This command must fail until backend-neutral CUDA/HIP implementation is accepted.",
        ),
        ReadinessCommand(
            "hip-source-build-mi300x",
            (
                "PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH .venv/bin/python -m pip install -e .[test] "
                "--config-settings=cmake.define.WOLFGANG_ENABLE_HIP=ON "
                "--config-settings=cmake.define.WOLFGANG_HIP_ARCHITECTURES=gfx942"
            ),
            "retained",
            "Retains the MI300X source-build lane from Campaign 7.",
        ),
        ReadinessCommand(
            "hip-source-build-alternate-amd",
            (
                "PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH .venv/bin/python -m pip install -e .[test] "
                "--config-settings=cmake.define.WOLFGANG_ENABLE_HIP=ON "
                "--config-settings=cmake.define.WOLFGANG_HIP_ARCHITECTURES=<gfx-target>"
            ),
            "blocked_external",
            "This requires a real non-MI300X AMD GPU host before any support claim is made.",
        ),
        ReadinessCommand(
            "hip-retained-operation-tests",
            "PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH .venv/bin/python -m pytest tests/test_phase12_rocm_foundation.py -q",
            "retained",
            "Retained HIP operation tests must pass on each claimed AMD GPU architecture.",
        ),
        ReadinessCommand(
            "rocm-release-smoke",
            (
                "PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH .venv/bin/python "
                "benchmarks/bench_rocm_kernels.py --profile campaign7-release-smoke --repeat 5 --warmup 2 --json "
                f"--output {raw}/rocm_campaign8_release_smoke.json"
            ),
            "retained",
            "Campaign 8 keeps the Campaign 7 source-build release-smoke boundary.",
        ),
        ReadinessCommand(
            "rocprof-legacy",
            (
                "PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH rocprof "
                f"-d {profiler}/legacy --hip-trace --stats "
                ".venv/bin/python benchmarks/bench_rocm_kernels.py "
                "--profile campaign7-profiler --repeat 1 --warmup 0 --json "
                f"--output {raw}/rocm_campaign8_profiler_legacy.json"
            ),
            "retained",
            "Legacy rocprof remains accepted while it produces HIP trace and stats evidence.",
        ),
        ReadinessCommand(
            "rocprofv3",
            (
                "PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH rocprofv3 --hip-trace --stats "
                f"-d {profiler}/rocprofv3 -- "
                ".venv/bin/python benchmarks/bench_rocm_kernels.py "
                "--profile campaign7-profiler --repeat 1 --warmup 0 --json "
                f"--output {raw}/rocm_campaign8_profiler_rocprofv3.json"
            ),
            "accepted_for_future_implementation",
            "Preferred ROCm 7.x migration lane when rocprofv3 is installed and option-compatible.",
        ),
        ReadinessCommand(
            "clean-machine-sdist-install",
            "python -m build --sdist --outdir /tmp/fastpauli-campaign8-dist && python -m pip install /tmp/fastpauli-campaign8-dist/fastpauli-*.tar.gz",
            "accepted_for_future_implementation",
            "ROCm wheels remain unavailable; clean-machine tests are required before any wheel claim.",
        ),
        ReadinessCommand(
            "packaging-policy-check",
            "uv run python -m pytest tests/test_rocm_campaign8_plan.py -q",
            "accepted_for_future_implementation",
            "Checks Campaign 8 packaging, portability, and terminal-status policy routing.",
        ),
        ReadinessCommand(
            "render-assets",
            (
                "uv run python scripts/render_rocm_campaign8_assets.py "
                f"--data-dir {evidence_dir} --plot-dir docs/benchmarks/plots"
            ),
            "accepted_for_future_implementation",
            "Validates Campaign 8 summary status keys and preserves the broad README landscape plot.",
        ),
        ReadinessCommand(
            "report-validation",
            "uv run python -m pytest tests/test_rocm_campaign8_plan.py tests/test_rocm_campaign8_assets.py -q",
            "accepted_for_future_implementation",
            "Validates the Campaign 8 report, renderer, evidence schema, and source-of-truth routing.",
        ),
    ]


def print_commands(commands: list[ReadinessCommand]) -> None:
    for item in commands:
        print(f"[{item.label}]")
        print(item.command)
        print(f"status: {item.status}")
        print(f"notes: {item.notes}")
        print()


def write_evidence(evidence_dir: Path, *, source_commit: str | None = None) -> Path:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "logs").mkdir(exist_ok=True)
    (evidence_dir / "raw").mkdir(exist_ok=True)
    (evidence_dir / "profiler").mkdir(exist_ok=True)
    commands = command_inventory(evidence_dir)

    command_payload = {
        "campaign": CAMPAIGN,
        "git_revision": git_revision(source_commit=source_commit),
        "commands": [asdict(item) for item in commands],
    }
    (evidence_dir / "raw/readiness_commands.json").write_text(
        json.dumps(command_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    log_lines = [
        f"campaign: {CAMPAIGN}",
        f"git_revision: {command_payload['git_revision']}",
        f"python: {sys.version.split()[0]}",
        f"platform: {platform.platform()}",
        "",
    ]
    for item in commands:
        log_lines.extend(
            [
                f"[{item.label}]",
                item.command,
                f"status: {item.status}",
                f"notes: {item.notes}",
                "",
            ]
        )
    (evidence_dir / "logs/readiness_lane.txt").write_text(
        "\n".join(log_lines),
        encoding="utf-8",
    )

    summary = {
        "campaign": CAMPAIGN,
        "command": f"{Path(__file__).as_posix()} --write-evidence {evidence_dir.as_posix()}",
        "git_revision": command_payload["git_revision"],
        "runtime_changes": "none",
        "local_cpu_only_validation": {
            "status": "external_closeout_required",
            "command": "uv run python scripts/validate.py",
            "evidence": "Run separately during Campaign 8 closeout; this readiness lane only records the command contract.",
        },
        "cuda_hip_configure_rejection": {
            "status": "external_closeout_required",
            "command": next(item.command for item in commands if item.label == "cuda-hip-rejection"),
            "expected_result": "configure fails with cannot-both-be-ON diagnostic",
            "evidence": "Run separately during Campaign 8 closeout; this readiness lane only records the expected configure failure.",
        },
        "alternate_amd_gpu_availability": {
            "status": "blocked_external",
            "reason": "No non-MI300X AMD GPU host was provided for Campaign 8 execution.",
        },
        "packaging_gate": {
            "status": "accepted_for_future_implementation",
            "decision": "ROCm wheels remain unavailable until package channel, runtime policy, CI hardware, and clean-machine install tests exist.",
        },
        "rocprofv3_status": {
            "status": "accepted_for_future_implementation",
            "decision_doc": "docs/plans/rocm_profiler_migration_campaign8_decision.md",
        },
        "interop_reconsideration_status": {
            "status": "accepted_for_future_implementation",
            "decision_doc": "docs/plans/rocm_hip_interop_reconsideration_campaign8_decision.md",
        },
        "targeted_performance_reopen_status": {
            "status": "accepted_for_future_implementation",
            "decision": "A future ROCm performance campaign requires a retained operation, profiler artifact, measured bottleneck, proposed implementation, correctness oracle, A/B timing boundary, and rejection criteria.",
        },
        "campaign8_terminal_statuses": TERMINAL_STATUSES,
        "evidence_files": [
            "raw/readiness_commands.json",
            "logs/readiness_lane.txt",
        ],
    }
    summary_path = evidence_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-commands", action="store_true", help="print the Campaign 8 command inventory")
    parser.add_argument(
        "--write-evidence",
        nargs="?",
        const=str(DEFAULT_EVIDENCE_DIR),
        help="write command inventory, logs, and summary.json under the evidence directory",
    )
    parser.add_argument("--source-commit", help="explicit 40-hex source commit for archive/no-.git runs")
    args = parser.parse_args()

    evidence_dir = Path(args.write_evidence) if args.write_evidence else DEFAULT_EVIDENCE_DIR
    commands = command_inventory(evidence_dir)
    if args.print_commands or not args.write_evidence:
        print_commands(commands)
    if args.write_evidence:
        summary_path = write_evidence(evidence_dir, source_commit=args.source_commit)
        print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate ROCm Campaign 8 evidence and preserve the broad README plot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED_TERMINAL_STATUSES = {
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
REQUIRED_TERMINAL_KEYS = set(EXPECTED_TERMINAL_STATUSES)

ALLOWED_STATUSES = {
    "accepted_for_future_implementation",
    "retained",
    "rejected_with_evidence",
    "blocked_external",
    "unavailable",
    "out_of_scope_with_next_trigger",
}


def load_summary(data_dir: Path) -> dict[str, Any]:
    summary_path = data_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing Campaign 8 summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    statuses = summary.get("campaign8_terminal_statuses")
    if not isinstance(statuses, dict):
        raise ValueError("summary.json must contain campaign8_terminal_statuses object")
    keys = set(statuses)
    if keys != REQUIRED_TERMINAL_KEYS:
        missing = sorted(REQUIRED_TERMINAL_KEYS - keys)
        extra = sorted(keys - REQUIRED_TERMINAL_KEYS)
        raise ValueError(f"campaign8_terminal_statuses key mismatch: missing={missing}, extra={extra}")
    bad_statuses = {
        key: value
        for key, value in statuses.items()
        if value not in ALLOWED_STATUSES
    }
    if bad_statuses:
        raise ValueError(f"invalid Campaign 8 terminal statuses: {bad_statuses}")
    status_mismatches = {
        key: {"expected": EXPECTED_TERMINAL_STATUSES[key], "actual": value}
        for key, value in statuses.items()
        if value != EXPECTED_TERMINAL_STATUSES[key]
    }
    if status_mismatches:
        raise ValueError(f"Campaign 8 terminal status mismatch: {status_mismatches}")
    if summary.get("runtime_changes") != "none":
        raise ValueError("Campaign 8 summary must record runtime_changes='none'")
    return summary


def validate_landscape_plot(plot_dir: Path) -> Path:
    plot = plot_dir / "accelerator_landscape_with_rocm.svg"
    if not plot.exists():
        raise FileNotFoundError(f"missing broad accelerator landscape plot: {plot}")
    text = plot.read_text(encoding="utf-8")
    required_tokens = ("CPU", "CUDA", "ROCm")
    missing = [token for token in required_tokens if token not in text]
    if missing:
        raise ValueError(f"broad accelerator landscape plot is missing tokens: {missing}")
    return plot


def write_manifest(data_dir: Path, summary: dict[str, Any], plot: Path) -> Path:
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "campaign": summary["campaign"],
        "git_revision": summary["git_revision"],
        "validated_terminal_statuses": EXPECTED_TERMINAL_STATUSES,
        "preserved_broad_landscape_plot": plot.as_posix(),
        "runtime_changes": summary["runtime_changes"],
    }
    manifest_path = raw_dir / "render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("docs/benchmarks/data/rocm_campaign8_architecture_readiness_2026-05-01"),
    )
    parser.add_argument("--plot-dir", type=Path, default=Path("docs/benchmarks/plots"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    summary = load_summary(args.data_dir)
    plot = validate_landscape_plot(args.plot_dir)
    if args.check_only:
        print("Campaign 8 assets validated")
    else:
        print(write_manifest(args.data_dir, summary, plot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

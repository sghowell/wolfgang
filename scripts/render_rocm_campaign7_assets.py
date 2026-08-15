#!/usr/bin/env python3
"""Render summary evidence and plots for ROCm MI300X Campaign 7."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from render_rocm_campaign3_assets import (
    display_path,
    read_json,
    render_bar_svg,
    timing_value,
    write_text,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "docs/benchmarks/data/rocm_mi300x_campaign7_2026-04-30"
DEFAULT_PLOT_DIR = ROOT / "docs/benchmarks/plots"
ROCM_CAMPAIGN6_SUMMARY = (
    ROOT / "docs/benchmarks/data/rocm_mi300x_campaign6_2026-04-30/summary.json"
)

RELEASE_SUPPORT_PLOT = "rocm_mi300x_campaign7_release_support.svg"
LANDSCAPE_PLOT = "accelerator_landscape_with_rocm.svg"

TERMINAL_ITEMS = {
    "mi300x_repeatability",
    "cpu_only_control",
    "rocm_source_build_runbook",
    "rocm_ci_or_release_lane",
    "rocm_packaging_policy",
    "rocm_wheel_support",
    "alternate_amd_gpu_portability",
    "profiler_availability",
    "duplicate_pressure_simplify",
    "duplicate_pressure_matmul",
    "external_statevector_interop",
    "hip_dlpack",
    "hip_cuda_array_interface",
    "public_streams",
    "public_graphs",
    "public_workspaces",
    "multi_gpu_rocm",
    "simultaneous_cuda_hip",
    "backend_neutral_accelerator_design",
}

REQUIRED_RETAINED_MODES = {
    "retained_transfer",
    "retained_commutation",
    "retained_device_consumers",
    "retained_simplify",
    "retained_expectation",
    "retained_matmul",
}

REQUIRED_PRESSURE_MODES = {
    "simplify_duplicate_pressure",
    "matmul_duplicate_pressure",
}


def load_raw_reports(data_dir: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in sorted((data_dir / "raw").glob("*.json")):
        payload = read_json(path)
        payload["_source_file"] = display_path(path)
        reports.append(payload)
    return reports


def flatten_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in reports:
        for row in report.get("cases", []):
            if row.get("campaign") != "rocm_mi300x_campaign7":
                continue
            item = dict(row)
            item["report_profile"] = report.get("profile", row.get("profile", "unknown"))
            item["source_file"] = report.get("_source_file", "unknown")
            item["git_commit"] = report.get("git_commit")
            rows.append(item)
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Campaign 7 summary has no rows")

    terminal_coverage: set[str] = set()
    retained_modes: set[str] = set()
    pressure_modes: set[str] = set()
    profiler_seen = False
    all_unavailable = True
    for row in rows:
        statuses = row.get("campaign7_terminal_statuses", {})
        if not isinstance(statuses, dict):
            raise ValueError(f"Campaign 7 row lacks terminal statuses: {row.get('source_file')}")
        if set(statuses) != TERMINAL_ITEMS:
            missing = TERMINAL_ITEMS - set(statuses)
            extra = set(statuses) - TERMINAL_ITEMS
            raise ValueError(
                f"Campaign 7 rows omit terminal statuses: missing={sorted(missing)} extra={sorted(extra)}"
            )
        terminal_coverage.update(statuses)

        final_status = row.get("final_status")
        mode = str(row.get("mode"))
        if final_status != "unavailable":
            all_unavailable = False
        if final_status == "retained" and mode in REQUIRED_RETAINED_MODES:
            retained_modes.add(mode)
            if row.get("correctness_passed") is not True:
                raise ValueError(f"retained Campaign 7 row lacks correctness: {row.get('source_file')}")
            for field in ("correctness_digest", "timing_boundary"):
                if row.get(field) is None:
                    raise ValueError(
                        f"retained Campaign 7 row omits {field}: {row.get('source_file')}"
                    )
        if row.get("operation") == "profiler_smoke" and final_status == "passed":
            profiler_seen = True
        if mode in REQUIRED_PRESSURE_MODES and final_status == "rejected_with_evidence":
            pressure_modes.add(mode)

    missing_terminal = TERMINAL_ITEMS - terminal_coverage
    if missing_terminal:
        raise ValueError(f"Campaign 7 rows omit terminal statuses: {sorted(missing_terminal)}")

    if not all_unavailable:
        missing_retained = REQUIRED_RETAINED_MODES - retained_modes
        if missing_retained:
            raise ValueError(f"Campaign 7 retained evidence is missing: {sorted(missing_retained)}")
        if not profiler_seen:
            raise ValueError("Campaign 7 profiler evidence is missing")
        missing_pressure = REQUIRED_PRESSURE_MODES - pressure_modes
        if missing_pressure:
            raise ValueError(f"Campaign 7 duplicate-pressure decisions are missing: {sorted(missing_pressure)}")


def summarize_row(row: dict[str, Any]) -> dict[str, Any]:
    dataset = row.get("dataset", {})
    return {
        "profile": row.get("report_profile", row.get("profile")),
        "case": row.get("case"),
        "operation": row.get("operation"),
        "mode": row.get("mode"),
        "backend": row.get("backend"),
        "host_role": row.get("host_role"),
        "status": row.get("status"),
        "final_status": row.get("final_status"),
        "timing_boundary": row.get("timing_boundary"),
        "git_commit": row.get("git_commit"),
        "num_qubits": dataset.get("num_qubits"),
        "num_terms": dataset.get("num_terms"),
        "lhs_terms": dataset.get("lhs_terms"),
        "rhs_terms": dataset.get("rhs_terms"),
        "words": dataset.get("words"),
        "gpu_name": row.get("gpu_name", row.get("device_name")),
        "gfx_target": row.get("gfx_target"),
        "rocm_runtime_version": row.get("rocm_runtime_version"),
        "rocm_toolkit_version": row.get("rocm_toolkit_version"),
        "hip_compiler_version": row.get("hip_compiler_version"),
        "cpu_scalar_seconds": row.get("cpu_scalar_seconds"),
        "hip_transfer_seconds": row.get("hip_transfer_seconds"),
        "hip_to_host_seconds": row.get("hip_to_host_seconds"),
        "hip_device_output_reuse_seconds": row.get("hip_device_output_reuse_seconds"),
        "hip_count_commuting_axis_none_seconds": row.get(
            "hip_count_commuting_axis_none_seconds"
        ),
        "hip_simplify_device_resident_seconds": row.get(
            "hip_simplify_device_resident_seconds"
        ),
        "hip_expectation_device_resident_seconds": row.get(
            "hip_expectation_device_resident_seconds"
        ),
        "hip_matmul_device_resident_seconds": row.get("hip_matmul_device_resident_seconds"),
        "correctness_passed": row.get("correctness_passed"),
        "correctness_digest": row.get("correctness_digest"),
        "decision_reason": row.get("decision_reason"),
        "build_command": row.get("build_command"),
        "validation_command": row.get("validation_command"),
        "profiler_command": row.get("profiler_command"),
        "unavailable_reason": row.get("unavailable_reason"),
        "source_file": row.get("source_file"),
    }


def profiler_artifacts(data_dir: Path) -> list[str]:
    profiler = data_dir / "profiler"
    if not profiler.exists():
        return []
    return [display_path(path) for path in sorted(profiler.rglob("*")) if path.is_file()]


def terminal_statuses(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        statuses = row.get("campaign7_terminal_statuses")
        if isinstance(statuses, dict):
            return statuses
    return {}


def status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("final_status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def release_plot_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    colors = {
        "CPU scalar": "#334155",
        "HIP transfer": "#0e7490",
        "HIP to_host": "#0891b2",
        "HIP commutation": "#2563eb",
        "HIP compact consumer": "#0284c7",
        "HIP simplify": "#7c3aed",
        "HIP expectation": "#be123c",
        "HIP matmul": "#ca8a04",
    }
    specs_by_mode = {
        "retained_transfer": [
            ("CPU scalar", "cpu_scalar_seconds"),
            ("HIP transfer", "hip_transfer_seconds"),
            ("HIP to_host", "hip_to_host_seconds"),
        ],
        "retained_commutation": [
            ("CPU scalar", "cpu_scalar_seconds"),
            ("HIP commutation", "hip_device_output_reuse_seconds"),
        ],
        "retained_device_consumers": [
            ("CPU scalar", "cpu_scalar_seconds"),
            ("HIP compact consumer", "hip_count_commuting_axis_none_seconds"),
            ("HIP compact consumer", "hip_compact_consumer_seconds"),
        ],
        "retained_simplify": [
            ("CPU scalar", "cpu_scalar_seconds"),
            ("HIP simplify", "hip_simplify_device_resident_seconds"),
        ],
        "retained_expectation": [
            ("CPU scalar", "cpu_scalar_seconds"),
            ("HIP expectation", "hip_expectation_device_resident_seconds"),
        ],
        "retained_matmul": [
            ("CPU scalar", "cpu_scalar_seconds"),
            ("HIP matmul", "hip_matmul_device_resident_seconds"),
        ],
        "simplify_duplicate_pressure": [
            ("CPU scalar", "cpu_scalar_seconds"),
            ("HIP simplify", "hip_simplify_device_resident_seconds"),
        ],
        "matmul_duplicate_pressure": [
            ("CPU scalar", "cpu_scalar_seconds"),
            ("HIP matmul", "hip_matmul_device_resident_seconds"),
        ],
        "rocprof_availability": [
            ("CPU scalar", "cpu_scalar_seconds"),
            ("HIP simplify", "hip_simplify_device_resident_seconds"),
            ("HIP expectation", "hip_expectation_device_resident_seconds"),
            ("HIP matmul", "hip_matmul_device_resident_seconds"),
        ],
    }
    points: list[dict[str, Any]] = []
    for row in rows:
        if row.get("final_status") not in {"passed", "retained", "rejected_with_evidence"}:
            continue
        if row.get("timing_boundary") == "decision_only":
            continue
        case = str(row.get("case"))
        specs = specs_by_mode.get(str(row.get("mode")), [])
        for series, key in specs:
            seconds = timing_value(row, key)
            if seconds is None:
                continue
            points.append(
                {
                    "label": f"{case} | {series}",
                    "series": series,
                    "seconds": seconds,
                    "color": colors.get(series, "#64748b"),
                }
            )
    return points


def previous_landscape_points() -> list[dict[str, Any]]:
    if not ROCM_CAMPAIGN6_SUMMARY.exists():
        return []
    summary = read_json(ROCM_CAMPAIGN6_SUMMARY)
    points: list[dict[str, Any]] = []
    for point in summary.get("readme_performance_landscape", []):
        seconds = timing_value(point, "seconds")
        if seconds is None:
            continue
        points.append(
            {
                "label": str(point.get("label", point.get("series", "unknown"))),
                "series": str(point.get("series", "unknown")),
                "seconds": seconds,
                "color": str(point.get("color", "#64748b")),
            }
        )
    return points


def landscape_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points = previous_landscape_points()
    for point in release_plot_points(rows):
        item = dict(point)
        if str(item["series"]).startswith("HIP"):
            item["series"] = "ROCm " + str(item["series"])
            item["label"] = "ROCm " + str(item["label"])
        points.append(item)
    return points


def build_summary(data_dir: Path, plot_dir: Path) -> dict[str, Any]:
    reports = load_raw_reports(data_dir)
    rows = flatten_rows(reports)
    validate_rows(rows)
    plot_dir.mkdir(parents=True, exist_ok=True)

    release_points = release_plot_points(rows)
    if release_points:
        render_bar_svg(
            title="MI300X ROCm Campaign 7 Release Support",
            subtitle="Retained HIP operations, duplicate-pressure probes, and profiler-smoke timings",
            points=release_points,
            output=plot_dir / RELEASE_SUPPORT_PLOT,
        )

    landscape = landscape_points(rows)
    if landscape:
        render_bar_svg(
            title="FastPauli Accelerator Performance Landscape",
            subtitle="Checked CPU, CUDA, external, and ROCm evidence; log-scale seconds",
            points=landscape,
            output=plot_dir / LANDSCAPE_PLOT,
        )

    return {
        "campaign": "rocm_mi300x_campaign7",
        "date": "2026-04-30",
        "data_dir": display_path(data_dir),
        "reports_loaded": [report.get("_source_file") for report in reports],
        "rows": [summarize_row(row) for row in rows],
        "status_counts": status_counts(rows),
        "terminal_statuses": terminal_statuses(rows),
        "profiler_artifacts": profiler_artifacts(data_dir),
        "plots": {
            "release_support": display_path(plot_dir / RELEASE_SUPPORT_PLOT),
            "landscape": display_path(plot_dir / LANDSCAPE_PLOT),
        },
        "readme_performance_landscape": landscape,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(args.data_dir, args.plot_dir)
    write_text(args.data_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render summary evidence and plots for ROCm MI300X Campaign 6."""

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
DEFAULT_DATA_DIR = ROOT / "docs/benchmarks/data/rocm_mi300x_campaign6_2026-04-30"
DEFAULT_PLOT_DIR = ROOT / "docs/benchmarks/plots"
ROCM_CAMPAIGN5_SUMMARY = (
    ROOT / "docs/benchmarks/data/rocm_mi300x_campaign5_2026-04-30/summary.json"
)

PARITY_PLOT = "rocm_mi300x_campaign6_parity.svg"
LANDSCAPE_PLOT = "accelerator_landscape_with_rocm.svg"

TERMINAL_ITEMS = {
    "expectation",
    "matmul",
    "external device pointers",
    "DLPack",
    "CUDA Array Interface guard",
    "streams",
    "graphs",
    "workspaces",
    "portability",
    "ROCm wheels",
    "multi-GPU",
    "simultaneous CUDA+HIP",
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
            if row.get("campaign") != "rocm_mi300x_campaign6":
                continue
            item = dict(row)
            item["report_profile"] = report.get("profile", row.get("profile", "unknown"))
            item["source_file"] = report.get("_source_file", "unknown")
            item["git_commit"] = report.get("git_commit")
            rows.append(item)
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Campaign 6 summary has no rows")

    terminal_coverage: set[str] = set()
    retained_operations: set[str] = set()
    all_unavailable = True
    for row in rows:
        terminal_coverage.update(row.get("campaign6_terminal_statuses", {}))
        final_status = row.get("final_status")
        if final_status != "unavailable":
            all_unavailable = False
        if final_status == "retained":
            retained_operations.add(str(row.get("operation")))
            if row.get("correctness_passed") is not True:
                raise ValueError(f"retained Campaign 6 row lacks correctness: {row.get('source_file')}")
            required = (
                "correctness_digest",
                "timing_boundary",
            )
            missing = [field for field in required if row.get(field) is None]
            if missing:
                raise ValueError(f"retained Campaign 6 row omits {missing}: {row.get('source_file')}")

    missing_terminal = TERMINAL_ITEMS - terminal_coverage
    if missing_terminal:
        raise ValueError(f"Campaign 6 rows omit terminal statuses: {sorted(missing_terminal)}")

    required_operations = {"expectation_statevector", "matmul"}
    if not all_unavailable and not required_operations <= retained_operations:
        missing = sorted(required_operations - retained_operations)
        raise ValueError(f"Campaign 6 retained evidence is missing: {missing}")


def summarize_row(row: dict[str, Any]) -> dict[str, Any]:
    dataset = row.get("dataset", {})
    return {
        "profile": row.get("report_profile", row.get("profile")),
        "case": row.get("case"),
        "operation": row.get("operation"),
        "mode": row.get("mode"),
        "status": row.get("status"),
        "final_status": row.get("final_status"),
        "timing_boundary": row.get("timing_boundary"),
        "git_commit": row.get("git_commit"),
        "num_qubits": dataset.get("num_qubits"),
        "num_terms": dataset.get("num_terms"),
        "lhs_terms": dataset.get("lhs_terms"),
        "rhs_terms": dataset.get("rhs_terms"),
        "state_size": dataset.get("state_size"),
        "words": dataset.get("words"),
        "device_name": row.get("device_name"),
        "gfx_target": row.get("gfx_target"),
        "cpu_scalar_seconds": row.get("cpu_scalar_seconds"),
        "hip_expectation_transfer_seconds": row.get("hip_expectation_transfer_seconds"),
        "hip_expectation_device_resident_seconds": row.get(
            "hip_expectation_device_resident_seconds"
        ),
        "hip_expectation_result_copy_seconds": row.get("hip_expectation_result_copy_seconds"),
        "hip_matmul_transfer_seconds": row.get("hip_matmul_transfer_seconds"),
        "hip_matmul_device_resident_seconds": row.get("hip_matmul_device_resident_seconds"),
        "hip_matmul_to_host_seconds": row.get("hip_matmul_to_host_seconds"),
        "hip_matmul_simplify_output": row.get("hip_matmul_simplify_output"),
        "correctness_passed": row.get("correctness_passed"),
        "correctness_digest": row.get("correctness_digest"),
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
        statuses = row.get("campaign6_terminal_statuses")
        if isinstance(statuses, dict):
            return statuses
    return {}


def status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("final_status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def parity_plot_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    colors = {
        "CPU scalar": "#334155",
        "HIP expectation transfer": "#0e7490",
        "HIP expectation operator-resident": "#2563eb",
        "HIP matmul transfer": "#7c3aed",
        "HIP matmul resident": "#be123c",
        "HIP matmul to_host": "#ca8a04",
    }
    points: list[dict[str, Any]] = []
    for row in rows:
        if row.get("final_status") != "retained":
            continue
        case = str(row.get("case"))
        specs = [
            ("CPU scalar", "cpu_scalar_seconds"),
            ("HIP expectation transfer", "hip_expectation_transfer_seconds"),
            (
                "HIP expectation operator-resident",
                "hip_expectation_device_resident_seconds",
            ),
            ("HIP matmul transfer", "hip_matmul_transfer_seconds"),
            ("HIP matmul resident", "hip_matmul_device_resident_seconds"),
            ("HIP matmul toHost", "hip_matmul_to_host_seconds"),
        ]
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
    if not ROCM_CAMPAIGN5_SUMMARY.exists():
        return []
    summary = read_json(ROCM_CAMPAIGN5_SUMMARY)
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
    for point in parity_plot_points(rows):
        if not str(point["series"]).startswith("HIP"):
            if point["series"] == "CPU scalar":
                points.append(point)
            continue
        item = dict(point)
        item["series"] = "ROCm " + str(item["series"])
        item["label"] = "ROCm " + str(item["label"])
        points.append(item)
    return points


def build_summary(data_dir: Path, plot_dir: Path) -> dict[str, Any]:
    reports = load_raw_reports(data_dir)
    rows = flatten_rows(reports)
    validate_rows(rows)
    plot_dir.mkdir(parents=True, exist_ok=True)

    parity_points = parity_plot_points(rows)
    if parity_points:
        render_bar_svg(
            title="MI300X ROCm Campaign 6 Parity",
            subtitle="CPU scalar vs HIP transfer-inclusive, retained-kernel, and explicit output materialization timings",
            points=parity_points,
            output=plot_dir / PARITY_PLOT,
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
        "campaign": "rocm_mi300x_campaign6",
        "date": "2026-04-30",
        "data_dir": display_path(data_dir),
        "reports_loaded": [report.get("_source_file") for report in reports],
        "rows": [summarize_row(row) for row in rows],
        "status_counts": status_counts(rows),
        "terminal_statuses": terminal_statuses(rows),
        "profiler_artifacts": profiler_artifacts(data_dir),
        "plots": {
            "parity": display_path(plot_dir / PARITY_PLOT),
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

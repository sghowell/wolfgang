#!/usr/bin/env python3
"""Render summary evidence for the ROCm MI300X Campaign 2 report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

RAW_FILES = (
    "rocm_commutation_device_output_scaling_mi300x.json",
    "rocm_commutation_compact_consumers_mi300x.json",
    "rocm_commutation_campaign2_profiler_mi300x.json",
    "rocm_commutation_campaign2_counters_mi300x.json",
)

PLOT_FILE = "rocm_mi300x_campaign2_boundaries.svg"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_reports(data_dir: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for name in RAW_FILES:
        path = data_dir / "raw" / name
        if path.exists():
            report = load_json(path)
            report["_source_file"] = str(path)
            reports.append(report)
    return reports


def flatten_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in reports:
        for row in report.get("cases", []):
            item = dict(row)
            item["profile"] = report.get("profile", "unknown")
            item["source_file"] = report.get("_source_file", "unknown")
            rows.append(item)
    return rows


def best_cpu_selector(row: dict[str, Any]) -> tuple[str, float] | None:
    values = {
        name: float(seconds)
        for name, seconds in row.get("available_cpu_selector_seconds", {}).items()
        if seconds is not None
    }
    if not values:
        return None
    name = min(values, key=values.get)
    return name, values[name]


def summarize_row(row: dict[str, Any]) -> dict[str, Any]:
    best_selector = best_cpu_selector(row)
    return {
        "profile": row.get("profile"),
        "case": row.get("case"),
        "status": row.get("status"),
        "correctness_passed": row.get("correctness_passed"),
        "entries": row.get("entries"),
        "num_qubits": row.get("num_qubits"),
        "lhs_terms": row.get("lhs_terms"),
        "rhs_terms": row.get("rhs_terms"),
        "device_name": row.get("device_name"),
        "gfx_target": row.get("gfx_target"),
        "cpu_scalar_seconds": row.get("cpu_scalar_seconds"),
        "best_cpu_selector": best_selector[0] if best_selector else None,
        "best_cpu_selector_seconds": best_selector[1] if best_selector else None,
        "hip_device_operand_host_output_seconds": row.get("hip_device_operand_host_output_seconds"),
        "hip_device_output_allocate_seconds": row.get("hip_device_output_allocate_seconds"),
        "hip_device_output_reuse_seconds": row.get("hip_device_output_reuse_seconds"),
        "hip_device_output_to_host_seconds": row.get("hip_device_output_to_host_seconds"),
        "hip_count_commuting_axis_none_seconds": row.get("hip_count_commuting_axis_none_seconds"),
        "hip_count_commuting_axis_0_seconds": row.get("hip_count_commuting_axis_0_seconds"),
        "hip_count_commuting_axis_1_seconds": row.get("hip_count_commuting_axis_1_seconds"),
        "hip_conflict_degrees_axis_none_seconds": row.get("hip_conflict_degrees_axis_none_seconds"),
        "hip_conflict_degrees_axis_0_seconds": row.get("hip_conflict_degrees_axis_0_seconds"),
        "hip_conflict_degrees_axis_1_seconds": row.get("hip_conflict_degrees_axis_1_seconds"),
        "correctness_digest": row.get("correctness_digest"),
        "source_file": row.get("source_file"),
    }


def profiler_artifacts(data_dir: Path) -> list[str]:
    profiler = data_dir / "profiler"
    if not profiler.exists():
        return []
    return [
        str(path)
        for path in sorted(profiler.rglob("*"))
        if path.is_file()
    ]


def validation_logs(data_dir: Path) -> list[str]:
    logs = data_dir / "logs"
    if not logs.exists():
        return []
    return [
        str(path)
        for path in sorted(logs.rglob("*"))
        if path.is_file()
    ]


def row_by_case(rows: list[dict[str, Any]], case: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("case") == case and row.get("status") == "ok":
            return row
    return None


def log_x(value: float, *, min_value: float, max_value: float, left: float, width: float) -> float:
    clipped = max(min_value, min(max_value, value))
    numerator = math.log10(clipped) - math.log10(min_value)
    denominator = math.log10(max_value) - math.log10(min_value)
    return left + width * numerator / denominator


def render_boundary_svg(row: dict[str, Any], output: Path) -> None:
    best_selector = best_cpu_selector(row)
    series: list[tuple[str, float, str]] = [
        ("CPU scalar", float(row["cpu_scalar_seconds"]), "#334155"),
    ]
    if best_selector is not None:
        series.append((f"CPU {best_selector[0]}", best_selector[1], "#0f766e"))
    series.extend(
        [
            ("HIP host output", float(row["hip_device_operand_host_output_seconds"]), "#0e7490"),
            ("HIP allocate", float(row["hip_device_output_allocate_seconds"]), "#2563eb"),
            ("HIP reuse", float(row["hip_device_output_reuse_seconds"]), "#7c3aed"),
            ("HIP to_host", float(row["hip_device_output_to_host_seconds"]), "#be123c"),
            ("HIP count total", float(row["hip_count_commuting_axis_none_seconds"]), "#ca8a04"),
            ("HIP conflict total", float(row["hip_conflict_degrees_axis_none_seconds"]), "#ea580c"),
        ]
    )
    min_value = min(value for _, value, _ in series) * 0.65
    max_value = max(value for _, value, _ in series) * 1.35

    width = 980
    height = 520
    chart_left = 260
    chart_top = 96
    chart_width = 620
    row_height = 42

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="48" y="52" font-family="Inter, Arial, sans-serif" font-size="26" font-weight="700" fill="#0f172a">MI300X ROCm Campaign 2 Boundary Timings</text>',
        '<text x="48" y="78" font-family="Inter, Arial, sans-serif" font-size="14" fill="#475569">4096 x 4096 pairwise commutation, log-scale seconds, lower is better</text>',
        f'<line x1="{chart_left}" y1="{chart_top - 18}" x2="{chart_left + chart_width}" y2="{chart_top - 18}" stroke="#cbd5e1" stroke-width="1"/>',
    ]

    tick_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    for tick in tick_values:
        if tick < min_value or tick > max_value:
            continue
        x = log_x(tick, min_value=min_value, max_value=max_value, left=chart_left, width=chart_width)
        lines.append(
            f'<line x1="{x:.1f}" y1="{chart_top - 26}" x2="{x:.1f}" y2="{chart_top + row_height * len(series)}" stroke="#e2e8f0" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{chart_top - 34}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="12" fill="#64748b">{tick:g}s</text>'
        )

    for index, (label, value, color) in enumerate(series):
        y = chart_top + index * row_height
        x_end = log_x(value, min_value=min_value, max_value=max_value, left=chart_left, width=chart_width)
        lines.extend(
            [
                f'<text x="48" y="{y + 20}" font-family="Inter, Arial, sans-serif" font-size="14" fill="#0f172a">{label}</text>',
                f'<rect x="{chart_left}" y="{y + 4}" width="{max(2.0, x_end - chart_left):.1f}" height="22" rx="4" fill="{color}"/>',
                f'<text x="{x_end + 10:.1f}" y="{y + 20}" font-family="Inter, Arial, sans-serif" font-size="13" fill="#334155">{value:.3g}s</text>',
            ]
        )

    digest = row.get("correctness_digest", {})
    footer = (
        f"entries={row.get('entries'):,} | commuting={digest.get('commuting_count'):,} | "
        f"device={row.get('device_name')} | gfx={row.get('gfx_target')}"
    )
    lines.append(
        f'<text x="48" y="{height - 36}" font-family="Inter, Arial, sans-serif" font-size="13" fill="#64748b">{footer}</text>'
    )
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(data_dir: Path, plot_dir: Path) -> dict[str, Any]:
    reports = load_reports(data_dir)
    rows = flatten_rows(reports)
    summarized_rows = [summarize_row(row) for row in rows]
    scaling_large = row_by_case(rows, "campaign2_large_dense_output")
    compact_large = row_by_case(rows, "campaign2_compact_large")
    profiler_row = row_by_case(rows, "campaign2_profiler_dense_pairs")

    plot_path = None
    if scaling_large is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        output = plot_dir / PLOT_FILE
        render_boundary_svg(scaling_large, output)
        plot_path = str(output)

    return {
        "campaign": "rocm_mi300x_campaign2",
        "data_dir": str(data_dir),
        "reports_loaded": [report.get("_source_file") for report in reports],
        "rows": summarized_rows,
        "large_device_output_row": summarize_row(scaling_large) if scaling_large else None,
        "large_compact_consumer_row": summarize_row(compact_large) if compact_large else None,
        "profiler_row": summarize_row(profiler_row) if profiler_row else None,
        "plot": plot_path,
        "profiler_artifacts": profiler_artifacts(data_dir),
        "validation_logs": validation_logs(data_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path, default=Path("docs/benchmarks/plots"))
    parser.add_argument("--summary-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(args.data_dir, args.plot_dir)
    output = args.summary_output or args.data_dir / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

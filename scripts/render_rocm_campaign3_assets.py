#!/usr/bin/env python3
"""Render summary evidence and plots for ROCm MI300X Campaign 3."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30"
DEFAULT_PLOT_DIR = ROOT / "docs/benchmarks/plots"
CUDA_CAMPAIGN10_SUMMARY = (
    ROOT / "docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29/summary.json"
)
ROCM_CAMPAIGN2_SUMMARY = (
    ROOT / "docs/benchmarks/data/rocm_mi300x_campaign2_2026-04-30/summary.json"
)

SIMPLIFY_PLOT = "rocm_mi300x_campaign3_simplify.svg"
LANDSCAPE_PLOT = "accelerator_landscape_with_rocm.svg"

HEADROOM_ITEMS = {
    "DLPack",
    "streams",
    "workspaces",
    "packed summaries",
    "expectation",
    "matmul",
    "portability",
    "ROCm wheels",
    "multi-GPU",
    "simultaneous CUDA+HIP",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def load_raw_reports(data_dir: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in sorted((data_dir / "raw").glob("*.json")):
        payload = read_json(path)
        payload["_source_file"] = display_path(path)
        reports.append(payload)
    return reports


def flatten_simplify_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in reports:
        for row in report.get("cases", []):
            if row.get("operation") != "simplify":
                continue
            item = dict(row)
            item["report_profile"] = report.get("profile", row.get("profile", "unknown"))
            item["source_file"] = report.get("_source_file", "unknown")
            rows.append(item)
    return rows


def timing_value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= 0.0:
        return None
    return numeric


def summarize_row(row: dict[str, Any]) -> dict[str, Any]:
    dataset = row.get("dataset", {})
    return {
        "profile": row.get("report_profile", row.get("profile")),
        "case": row.get("case"),
        "status": row.get("status"),
        "correctness_passed": row.get("correctness_passed"),
        "num_qubits": dataset.get("num_qubits"),
        "num_terms": dataset.get("num_terms"),
        "words": dataset.get("words"),
        "duplicate_rate": dataset.get("duplicate_rate"),
        "actual_duplicate_rate": dataset.get("actual_duplicate_rate"),
        "device_name": row.get("device_name"),
        "gfx_target": row.get("gfx_target"),
        "cpu_scalar_seconds": row.get("cpu_scalar_seconds"),
        "hip_simplify_transfer_seconds": row.get("hip_simplify_transfer_seconds"),
        "hip_simplify_device_resident_seconds": row.get(
            "hip_simplify_device_resident_seconds"
        ),
        "hip_simplify_to_host_seconds": row.get("hip_simplify_to_host_seconds"),
        "hip_simplify_strategy": row.get("hip_simplify_strategy"),
        "hip_simplify_strategy_status": row.get("hip_simplify_strategy_status"),
        "hip_simplify_strategy_unavailable_reason": row.get(
            "hip_simplify_strategy_unavailable_reason"
        ),
        "hip_simplify_output_terms": row.get("hip_simplify_output_terms"),
        "hip_simplify_output_words": row.get("hip_simplify_output_words"),
        "correctness_digest": row.get("correctness_digest"),
        "campaign3_headroom_statuses": row.get("campaign3_headroom_statuses", {}),
        "source_file": row.get("source_file"),
    }


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Campaign 3 summary has no simplify rows")
    covered_items: set[str] = set()
    ok_rows = 0
    for row in rows:
        statuses = row.get("campaign3_headroom_statuses", {})
        covered_items.update(statuses)
        if row.get("status") == "ok":
            ok_rows += 1
            required = (
                "hip_simplify_transfer_seconds",
                "hip_simplify_device_resident_seconds",
                "hip_simplify_to_host_seconds",
                "hip_simplify_output_terms",
                "hip_simplify_output_words",
                "correctness_digest",
            )
            missing = [field for field in required if row.get(field) is None]
            if missing:
                raise ValueError(f"ok simplify row omits {missing}: {row.get('source_file')}")
            if row.get("hip_simplify_strategy_status") != "retained":
                raise ValueError(f"ok simplify row is not retained: {row.get('source_file')}")
    if ok_rows == 0:
        raise ValueError("Campaign 3 summary has no successful simplify rows")
    missing_items = HEADROOM_ITEMS - covered_items
    if missing_items:
        raise ValueError(f"Campaign 3 rows omit headroom statuses: {sorted(missing_items)}")


def profiler_artifacts(data_dir: Path) -> list[str]:
    profiler = data_dir / "profiler"
    if not profiler.exists():
        return []
    return [
        display_path(path)
        for path in sorted(profiler.rglob("*"))
        if path.is_file()
    ]


def log_x(value: float, *, min_value: float, max_value: float, left: float, width: float) -> float:
    clipped = max(min_value, min(max_value, value))
    numerator = math.log10(clipped) - math.log10(min_value)
    denominator = math.log10(max_value) - math.log10(min_value)
    return left + width * numerator / denominator


def render_bar_svg(
    *,
    title: str,
    subtitle: str,
    points: list[dict[str, Any]],
    output: Path,
) -> None:
    if not points:
        raise ValueError(f"no points to render for {output}")
    values = [float(point["seconds"]) for point in points]
    min_value = max(min(values) * 0.55, 1.0e-8)
    max_value = max(values) * 1.45
    width = 1120
    row_height = 34
    chart_left = 390
    chart_top = 104
    chart_width = 600
    height = chart_top + row_height * len(points) + 80

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="48" y="52" font-family="Inter, Arial, sans-serif" font-size="26" font-weight="700" fill="#0f172a">{escape(title)}</text>',
        f'<text x="48" y="78" font-family="Inter, Arial, sans-serif" font-size="14" fill="#475569">{escape(subtitle)}</text>',
    ]
    for tick in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        if tick < min_value or tick > max_value:
            continue
        x = log_x(tick, min_value=min_value, max_value=max_value, left=chart_left, width=chart_width)
        lines.append(
            f'<line x1="{x:.1f}" y1="{chart_top - 24}" x2="{x:.1f}" y2="{height - 48}" stroke="#e2e8f0" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{chart_top - 34}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="12" fill="#64748b">{tick:g}s</text>'
        )

    for index, point in enumerate(points):
        y = chart_top + index * row_height
        seconds = float(point["seconds"])
        x_end = log_x(seconds, min_value=min_value, max_value=max_value, left=chart_left, width=chart_width)
        color = point.get("color", "#2563eb")
        label = point["label"]
        series = point.get("series", "")
        lines.extend(
            [
                f'<text x="48" y="{y + 19}" font-family="Inter, Arial, sans-serif" font-size="13" fill="#0f172a">{escape(label)}</text>',
                f'<rect x="{chart_left}" y="{y + 5}" width="{max(2.0, x_end - chart_left):.1f}" height="18" rx="3" fill="{color}"/>',
                f'<text x="{x_end + 10:.1f}" y="{y + 19}" font-family="Inter, Arial, sans-serif" font-size="12" fill="#334155">{seconds:.3g}s {escape(series)}</text>',
            ]
        )
    lines.append(
        f'<text x="48" y="{height - 30}" font-family="Inter, Arial, sans-serif" font-size="12" fill="#64748b">Log-scale seconds, lower is better. Generated from checked raw benchmark JSON.</text>'
    )
    lines.append("</svg>")
    write_text(output, "\n".join(lines))


def simplify_plot_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    colors = {
        "CPU scalar": "#334155",
        "HIP transfer": "#0e7490",
        "HIP resident": "#2563eb",
        "HIP to_host": "#be123c",
    }
    points: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("report_profile") == "simplify-campaign3-profiler":
            continue
        case = str(row.get("case"))
        for series, key in (
            ("CPU scalar", "cpu_scalar_seconds"),
            ("HIP transfer", "hip_simplify_transfer_seconds"),
            ("HIP resident", "hip_simplify_device_resident_seconds"),
            ("HIP to_host", "hip_simplify_to_host_seconds"),
        ):
            seconds = timing_value(row, key)
            if seconds is None:
                continue
            points.append(
                {
                    "label": f"{case} | {series}",
                    "series": series,
                    "seconds": seconds,
                    "color": colors[series],
                }
            )
    return points


def cuda_landscape_points() -> list[dict[str, Any]]:
    if not CUDA_CAMPAIGN10_SUMMARY.exists():
        return []
    summary = read_json(CUDA_CAMPAIGN10_SUMMARY)
    best_by_series: dict[str, tuple[str, float]] = {}
    for row in summary.get("readme_performance_landscape", []):
        category = str(row.get("category", "CUDA Campaign 10"))
        for point in row.get("points", []):
            seconds = timing_value(point, "seconds")
            if seconds is None:
                continue
            series = str(point.get("series", "unknown"))
            previous = best_by_series.get(series)
            if previous is None or seconds < previous[1]:
                best_by_series[series] = (category, seconds)
    colors = {
        "CPU scalar": "#334155",
        "CPU optimized": "#0f766e",
        "CPU AVX512": "#0f766e",
        "CUDA transfer-inclusive": "#7c3aed",
        "CUDA device-resident": "#2563eb",
        "CUDA compact graph consumer": "#9333ea",
        "CUDA compact grouping consumer": "#a855f7",
        "CUDA CSR export baseline": "#64748b",
        "CuPy DLPack": "#be123c",
        "PyTorch DLPack": "#dc2626",
    }
    return [
        {
            "label": f"{series} | {category[:44]}",
            "series": series,
            "seconds": seconds,
            "color": colors.get(series, "#64748b"),
        }
        for series, (category, seconds) in sorted(best_by_series.items())
    ]


def rocm_campaign2_landscape_points() -> list[dict[str, Any]]:
    if not ROCM_CAMPAIGN2_SUMMARY.exists():
        return []
    summary = read_json(ROCM_CAMPAIGN2_SUMMARY)
    points: list[dict[str, Any]] = []
    for row in summary.get("rows", []):
        if row.get("status") != "ok":
            continue
        case = str(row.get("case"))
        for series, key, color in (
            ("ROCm HIP commutation allocate", "hip_device_output_allocate_seconds", "#0891b2"),
            ("ROCm HIP commutation compact count", "hip_count_commuting_axis_none_seconds", "#ca8a04"),
        ):
            seconds = timing_value(row, key)
            if seconds is None:
                continue
            points.append(
                {
                    "label": f"{series} | {case}",
                    "series": series,
                    "seconds": seconds,
                    "color": color,
                }
            )
    return points[:4]


def landscape_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points = cuda_landscape_points()
    points.extend(rocm_campaign2_landscape_points())
    for point in simplify_plot_points(rows):
        if point["series"].startswith("HIP"):
            point = dict(point)
            point["series"] = "ROCm " + str(point["series"])
            point["label"] = "ROCm " + str(point["label"])
            point["color"] = {
                "ROCm HIP transfer": "#0e7490",
                "ROCm HIP resident": "#2563eb",
                "ROCm HIP to_host": "#be123c",
            }.get(point["series"], "#0e7490")
            points.append(point)
        elif point["series"] == "CPU scalar" and "campaign3_duplicate_heavy" in point["label"]:
            points.append(point)
    return points


def strategy_decisions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for row in rows:
        strategy = str(row.get("hip_simplify_strategy", "unavailable"))
        decisions[strategy] = {
            "strategy": strategy,
            "status": row.get("hip_simplify_strategy_status"),
            "reason": row.get("hip_simplify_strategy_unavailable_reason"),
            "source_file": row.get("source_file"),
        }
    for strategy in ("rocthrust_default", "hipcub_radix_sort_reduce", "custom_packed_key"):
        decisions.setdefault(
            strategy,
            {
                "strategy": strategy,
                "status": "unavailable" if strategy != "rocthrust_default" else "retained",
                "reason": (
                    None
                    if strategy == "rocthrust_default"
                    else "not present in raw Campaign 3 strategy evidence"
                ),
                "source_file": None,
            },
        )
    return [decisions[strategy] for strategy in sorted(decisions)]


def build_summary(data_dir: Path, plot_dir: Path) -> dict[str, Any]:
    reports = load_raw_reports(data_dir)
    rows = flatten_simplify_rows(reports)
    validate_rows(rows)
    summarized_rows = [summarize_row(row) for row in rows]
    plot_dir.mkdir(parents=True, exist_ok=True)

    render_bar_svg(
        title="MI300X ROCm Campaign 3 Simplify",
        subtitle="CPU scalar vs HIP transfer, device-resident simplify, and explicit to_host",
        points=simplify_plot_points(rows),
        output=plot_dir / SIMPLIFY_PLOT,
    )
    render_bar_svg(
        title="FastPauli Accelerator Performance Landscape",
        subtitle="Broad checked CPU, CUDA, external, and ROCm evidence; log-scale seconds",
        points=landscape_points(rows),
        output=plot_dir / LANDSCAPE_PLOT,
    )

    return {
        "campaign": "rocm_mi300x_campaign3",
        "date": "2026-04-30",
        "data_dir": display_path(data_dir),
        "reports_loaded": [report.get("_source_file") for report in reports],
        "rows": summarized_rows,
        "strategy_decisions": strategy_decisions(rows),
        "headroom_statuses": rows[0].get("campaign3_headroom_statuses", {}) if rows else {},
        "profiler_artifacts": profiler_artifacts(data_dir),
        "plots": {
            "simplify": display_path(plot_dir / SIMPLIFY_PLOT),
            "landscape": display_path(plot_dir / LANDSCAPE_PLOT),
        },
        "readme_performance_landscape": landscape_points(rows),
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

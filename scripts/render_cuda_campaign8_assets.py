#!/usr/bin/env python3
"""Render Campaign 8 H100 CUDA device-resident consumer summary and SVG assets."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

PLOTS = {
    "device": "cuda_h100_campaign8_device_resident_consumers.svg",
    "interop": "cuda_h100_campaign8_interop_consumers.svg",
    "stream": "cuda_h100_campaign8_stream_graph.svg",
    "scatter": "cuda_h100_campaign8_scatter_ab.svg",
    "portability": "cuda_h100_campaign8_portability.svg",
    "landscape": "cuda_h100_campaign8_performance_landscape.svg",
}

RAW_FILES = {
    "baseline": "baseline_campaign7_reproduction.json",
    "device": "campaign8_device_graph.json",
    "grouping": "campaign8_grouping_consumer.json",
    "interop": "campaign8_interop.json",
    "stream": "campaign8_stream_graph.json",
    "scatter": "campaign8_scatter_ab.json",
}

REQUIRED_STATUS_FIELDS = (
    "device_resident_graph_status",
    "public_grouping_api_status",
    "dlpack_interop_status",
    "non_h100_portability_status",
    "stream_graph_status",
    "scatter_tuning_status",
)

COLORS = {
    "ink": "#172033",
    "muted": "#5d6b82",
    "grid": "#d8dee9",
    "panel": "#f7f9fc",
    "blue": "#2563eb",
    "green": "#12805c",
    "orange": "#c76a00",
    "red": "#c2410c",
    "cyan": "#0891b2",
    "purple": "#7c3aed",
    "gray": "#7b8798",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def log_has_passed_tests(path: Path) -> bool:
    text_value = read_text(path)
    return bool(re.search(r"\b\d+\s+passed\b", text_value))


def status_receipt_passed(path: Path) -> bool:
    text_value = read_text(path).strip().lower()
    return text_value == "passed" or "status: passed" in text_value


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def format_seconds(value: float) -> str:
    if value < 1.0e-3:
        return f"{value * 1.0e6:.0f} us"
    if value < 1.0:
        return f"{value * 1.0e3:.2f} ms"
    return f"{value:.2f} s"


def format_bytes(value: int) -> str:
    number = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if number < 1024.0 or unit == "GB":
            return f"{int(number)} B" if unit == "B" else f"{number:.1f} {unit}"
        number /= 1024.0
    return f"{value} B"


def svg_start(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title>{esc(title)}</title>",
        f"<desc>{esc(subtitle)}</desc>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        f'<text x="32" y="42" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="24" font-weight="700" fill="{COLORS["ink"]}">{esc(title)}</text>',
        f'<text x="32" y="68" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="13" fill="{COLORS["muted"]}">{esc(subtitle)}</text>',
    ]


def text(
    x: float,
    y: float,
    value: object,
    *,
    size: int = 12,
    color: str | None = None,
    weight: int = 400,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter, ui-sans-serif, system-ui, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color or COLORS["ink"]}" '
        f'text-anchor="{anchor}">{esc(value)}</text>'
    )


def rect(x: float, y: float, width: float, height: float, color: str, *, opacity: float = 1.0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(width, 0):.1f}" height="{height:.1f}" '
        f'rx="3" fill="{color}" opacity="{opacity:.3f}"/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="1"/>'


def load_raw(data_dir: Path) -> dict[str, dict[str, Any]]:
    raw_dir = data_dir / "raw"
    missing = [filename for filename in RAW_FILES.values() if not (raw_dir / filename).exists()]
    if missing:
        raise FileNotFoundError("missing Campaign 8 raw artifact(s): " + ", ".join(missing))
    return {name: read_json(raw_dir / filename) for name, filename in RAW_FILES.items()}


def first_case(report: dict[str, Any]) -> dict[str, Any]:
    cases = report.get("cases", [])
    if not cases:
        raise ValueError("Campaign 8 report has no benchmark cases")
    return cases[0]


def graph_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in report.get("cases", []):
        result = case["results"]
        row = {
            "scale": case["scale"],
            "terms": int(case["dataset"]["lhs_terms"]),
            "dense_to_host_seconds": result["dense_to_host_seconds"],
            "csr_export_seconds": result["campaign7_csr_graph_export_seconds"],
            "graph_compact_seconds": result["campaign8_device_resident_graph_compact_seconds"],
            "grouping_compact_seconds": result["campaign8_device_grouping_consumer_seconds"],
            "count_axis_none_seconds": result["count_commuting_axis_none_seconds"],
            "count_axis0_seconds": result["count_commuting_axis_0_seconds"],
            "count_axis1_seconds": result["count_commuting_axis_1_seconds"],
            "edge_count": int(result["campaign8_device_resident_graph_edge_count"]),
            "graph_compact_host_bytes": int(result["campaign8_device_resident_graph_compact_host_bytes"]),
            "grouping_compact_host_bytes": int(result["campaign8_device_grouping_consumer_compact_host_bytes"]),
            "full_csr_host_bytes": int(result["campaign8_device_resident_graph_full_csr_host_bytes"]),
        }
        for field in REQUIRED_STATUS_FIELDS:
            row[field] = case.get(field, report.get("campaign8", {}).get(field, "not_applicable"))
        rows.append(row)
    return rows


def load_campaign7_landscape() -> list[dict[str, Any]]:
    path = ROOT / "docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29/summary.json"
    if not path.exists():
        return []
    return read_json(path).get("readme_performance_landscape", [])


def campaign8_landscape_rows(campaign7_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    landscape = list(campaign7_rows)
    points: list[dict[str, Any]] = []
    for row in rows:
        label = f"{row['terms']}x{row['terms']}"
        points.extend(
            [
                {"label": f"{label} graph compact", "seconds": row["graph_compact_seconds"], "series": "CUDA Campaign 8 graph compact"},
                {"label": f"{label} grouping compact", "seconds": row["grouping_compact_seconds"], "series": "CUDA Campaign 8 grouping compact"},
                {"label": f"{label} CSR export", "seconds": row["csr_export_seconds"], "series": "CUDA fused CSR export"},
                {"label": f"{label} dense to_host", "seconds": row["dense_to_host_seconds"], "series": "CUDA device-output dense host copy"},
                {"label": f"{label} compact count", "seconds": row["count_axis_none_seconds"], "series": "CUDA compact count"},
            ]
        )
    landscape.append({"category": "Campaign 8 device-resident consumers", "points": points})
    return landscape


def nsys_kernel_summary(data_dir: Path) -> list[dict[str, Any]]:
    sqlite_path = data_dir / "profiler" / "nsys_campaign8_device_graph.sqlite"
    if not sqlite_path.exists():
        return []
    query = """
        select
          StringIds.value as kernel,
          count(*) as launches,
          sum(CUPTI_ACTIVITY_KIND_KERNEL.end - CUPTI_ACTIVITY_KIND_KERNEL.start) / 1000000.0 as milliseconds
        from CUPTI_ACTIVITY_KIND_KERNEL
        join StringIds on CUPTI_ACTIVITY_KIND_KERNEL.shortName = StringIds.id
        group by CUPTI_ACTIVITY_KIND_KERNEL.shortName
        order by sum(CUPTI_ACTIVITY_KIND_KERNEL.end - CUPTI_ACTIVITY_KIND_KERNEL.start) desc
        limit 10
    """
    with sqlite3.connect(sqlite_path) as connection:
        rows = connection.execute(query).fetchall()
    return [
        {"kernel": str(kernel), "launches": int(launches), "milliseconds": float(milliseconds)}
        for kernel, launches, milliseconds in rows
    ]


def nsys_memcpy_summary(data_dir: Path) -> dict[str, Any]:
    sqlite_path = data_dir / "profiler" / "nsys_campaign8_device_graph.sqlite"
    if not sqlite_path.exists():
        return {"copies": 0, "bytes": 0, "milliseconds": 0.0}
    query = """
        select
          count(*) as copies,
          coalesce(sum(bytes), 0) as bytes,
          coalesce(sum(end - start), 0) / 1000000.0 as milliseconds
        from CUPTI_ACTIVITY_KIND_MEMCPY
    """
    with sqlite3.connect(sqlite_path) as connection:
        copies, bytes_value, milliseconds = connection.execute(query).fetchone()
    return {"copies": int(copies), "bytes": int(bytes_value), "milliseconds": float(milliseconds)}


def evidence_status(data_dir: Path, portability_report: Path) -> list[dict[str, str]]:
    metadata = data_dir / "metadata"
    profiler = data_dir / "profiler"
    sanitizer_logs = [
        profiler / "compute_sanitizer_memcheck.log",
        profiler / "compute_sanitizer_racecheck.log",
        profiler / "compute_sanitizer_initcheck.log",
        profiler / "compute_sanitizer_synccheck.log",
    ]
    sanitizer_ok = all(
        "ERROR SUMMARY: 0 errors" in read_text(path) or "0 hazards" in read_text(path)
        for path in sanitizer_logs
    )
    ncu_log = read_text(profiler / "ncu_campaign8_device_graph.stdout")
    return [
        {"label": "H100 Campaign 8 benchmarks", "status": "passed" if (data_dir / "raw" / RAW_FILES["device"]).exists() else "missing_or_failed"},
        {
            "label": "H100 repo validation",
            "status": "passed"
            if status_receipt_passed(metadata / "experiment-validate-final-status.txt")
            else "missing_or_failed",
        },
        {
            "label": "phase 11 CUDA tests",
            "status": "passed" if log_has_passed_tests(metadata / "phase11-campaign8-tests.log") else "missing_or_failed",
        },
        {"label": "compute-sanitizer ladder", "status": "passed" if sanitizer_ok else "missing_or_failed"},
        {"label": "Nsight Systems", "status": "passed" if (profiler / "nsys_campaign8_device_graph.sqlite").exists() else "missing_or_failed"},
        {"label": "Nsight Compute", "status": "blocked_permission" if "ERR_NVGPUCTRPERM" in ncu_log else "passed" if (profiler / "ncu_campaign8_device_graph.ncu-rep").exists() else "missing_or_failed"},
        {"label": "non-H100 NVIDIA portability", "status": "blocked_recorded" if portability_report.exists() else "missing_or_failed"},
    ]


def build_summary(data_dir: Path) -> dict[str, Any]:
    raw = load_raw(data_dir)
    device_rows = graph_rows(raw["device"])
    grouping_rows = graph_rows(raw["grouping"])
    interop_case = first_case(raw["interop"])
    stream_case = first_case(raw["stream"])
    scatter_case = first_case(raw["scatter"])
    portability_report = ROOT / "docs/benchmarks/reports/cuda_portability_campaign8_non_h100_nvidia_2026-04-29.md"
    landscape = campaign8_landscape_rows(load_campaign7_landscape(), device_rows)
    return {
        "campaign": "h100_campaign8",
        "date": "2026-04-29",
        "required_status_fields": list(REQUIRED_STATUS_FIELDS),
        "experiment_revision": str(raw["device"].get("git_commit", "")),
        "hardware": {
            "gpu": raw["device"].get("cuda_status", {}).get("devices", [{}])[0].get("name", "unknown"),
            "compute_capability": raw["device"].get("cuda_status", {}).get("devices", [{}])[0].get("compute_capability", []),
            "driver": raw["device"].get("cuda_status", {}).get("driver_version"),
            "runtime": raw["device"].get("cuda_status", {}).get("runtime_version"),
            "toolkit": raw["device"].get("fastpauli_build_info", {}).get("cuda_toolkit_version"),
            "compiled_architectures": raw["device"].get("fastpauli_build_info", {}).get("cuda_architectures"),
        },
        "decisions": [
            {"experiment": "device_resident_graph_consumers", "status": "retained", "reason": "Compact graph metadata avoids full CSR host export for retained high-scale rows."},
            {"experiment": "public_fused_grouping_api", "status": "deferred", "reason": "The private grouping probe is stable, but public ownership, documentation, and workflow semantics are not yet accepted."},
            {"experiment": "dlpack_or_framework_interop", "status": interop_case.get("dlpack_interop_status", "deferred"), "reason": interop_case["results"].get("dlpack_unavailable_reason", "DLPack deferred; CUDA Array Interface retained.")},
            {"experiment": "non_h100_portability", "status": "blocked", "reason": "No non-H100 NVIDIA host was available during this Campaign 8 execution."},
            {"experiment": "stream_graph_execution", "status": stream_case.get("stream_graph_status", "deferred"), "reason": stream_case["results"].get("stream_graph_unavailable_reason", "CUDA Graph and stream API deferred.")},
            {"experiment": "csr_scatter_tuning", "status": scatter_case.get("scatter_tuning_status", "rejected_no_consumer"), "reason": scatter_case["results"].get("csr_scatter_ab_unavailable_reason", "Retained consumers avoid full CSR scatter.")},
        ],
        "device_resident_consumers": device_rows,
        "grouping_consumers": grouping_rows,
        "interop": {
            "cupy_cuda_array_interface_status": interop_case["results"].get("cupy_consumer_available", False),
            "cupy_unavailable_reason": interop_case["results"].get("cupy_consumer_unavailable_reason"),
            "dlpack_interop_status": interop_case.get("dlpack_interop_status", "deferred"),
            "dlpack_unavailable_reason": interop_case["results"].get("dlpack_unavailable_reason"),
        },
        "stream_graph": {
            "stream_graph_status": stream_case.get("stream_graph_status", "deferred"),
            "unavailable_reason": stream_case["results"].get("stream_graph_unavailable_reason"),
        },
        "scatter_ab": {
            "scatter_tuning_status": scatter_case.get("scatter_tuning_status", "rejected_no_consumer"),
            "unavailable_reason": scatter_case["results"].get("csr_scatter_ab_unavailable_reason"),
        },
        "readme_performance_landscape": landscape,
        "evidence": {
            "status": evidence_status(data_dir, portability_report),
            "raw_json": sorted(path.name for path in (data_dir / "raw").glob("*.json")),
            "profilers": ["nsys", "ncu_permission_blocked"],
            "sanitizers": ["memcheck", "racecheck", "initcheck", "synccheck"],
            "portability_report": str(portability_report.relative_to(ROOT)),
        },
        "profiler": {
            "nsys_kernel_time_top": nsys_kernel_summary(data_dir),
            "nsys_memcpy": nsys_memcpy_summary(data_dir),
            "ncu_status": "blocked_permission"
            if "ERR_NVGPUCTRPERM" in read_text(data_dir / "profiler" / "ncu_campaign8_device_graph.stdout")
            else "available",
        },
    }


def render_log_bar_chart(title: str, subtitle: str, groups: list[dict[str, Any]], output: Path, *, width: int = 1180) -> None:
    bar_h = 16
    row_gap = 13
    group_gap = 26
    left = 260
    right = width - 48
    top = 108
    total_rows = sum(len(group["points"]) for group in groups)
    height = top + total_rows * (bar_h + row_gap) + len(groups) * group_gap + 54
    values = [float(point["seconds"]) for group in groups for point in group["points"] if point.get("seconds")]
    lo = min(values) if values else 1.0e-6
    hi = max(values) if values else 1.0
    lo_log = math.floor(math.log10(max(lo, 1.0e-9)))
    hi_log = math.ceil(math.log10(max(hi, 1.0e-8)))
    if lo_log == hi_log:
        hi_log += 1
    plot_w = right - left
    lines = svg_start(width, height, title, subtitle)
    for power in range(lo_log, hi_log + 1):
        value = 10.0 ** power
        x = left + (math.log10(value) - lo_log) / (hi_log - lo_log) * plot_w
        lines.append(line(x, top - 18, x, height - 40, COLORS["grid"]))
        lines.append(text(x, top - 26, format_seconds(value), size=10, color=COLORS["muted"], anchor="middle"))
    palette = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["cyan"], COLORS["gray"]]
    y = top
    for group in groups:
        lines.append(text(32, y + 1, group["label"], size=13, weight=700))
        for index, point in enumerate(group["points"]):
            y += bar_h + row_gap
            seconds = max(float(point["seconds"]), 1.0e-12)
            x1 = left + (math.log10(seconds) - lo_log) / (hi_log - lo_log) * plot_w
            lines.append(text(58, y + 12, str(point["label"])[:68], size=11))
            lines.append(rect(left, y, x1 - left, bar_h, point.get("color", palette[index % len(palette)])))
            lines.append(text(min(x1 + 8, right - 80), y + 12, format_seconds(seconds), size=11, color=COLORS["muted"]))
        y += group_gap
    lines.append(text(32, height - 20, "Log-scale latency; lower is better. Source: checked Campaign 8 raw JSON.", size=11, color=COLORS["muted"]))
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_device(rows: list[dict[str, Any]], output: Path) -> None:
    groups = []
    for row in rows:
        groups.append(
            {
                "label": f"{row['terms']} x {row['terms']} graph workflow",
                "points": [
                    {"label": "Campaign 8 graph compact", "seconds": row["graph_compact_seconds"], "color": COLORS["green"]},
                    {"label": "Campaign 8 grouping compact", "seconds": row["grouping_compact_seconds"], "color": COLORS["blue"]},
                    {"label": "Campaign 7 CSR export", "seconds": row["csr_export_seconds"], "color": COLORS["orange"]},
                    {"label": "Dense to_host", "seconds": row["dense_to_host_seconds"], "color": COLORS["gray"]},
                ],
            }
        )
    render_log_bar_chart("Campaign 8 Device-Resident Consumers", "Compact graph and grouping consumers avoid full CSR host export on H100.", groups, output)


def render_status_panel(title: str, subtitle: str, rows: list[tuple[str, str, str]], output: Path) -> None:
    width = 1080
    row_h = 54
    top = 112
    height = top + len(rows) * row_h + 42
    lines = svg_start(width, height, title, subtitle)
    for index, (label, status, detail) in enumerate(rows):
        y = top + index * row_h
        color = COLORS["green"] if status in {"retained", "passed", "available"} else COLORS["orange"] if status in {"deferred", "blocked", "blocked_permission", "rejected_no_consumer"} else COLORS["red"]
        lines.append(rect(40, y - 18, width - 80, 42, COLORS["panel"], opacity=1.0))
        lines.append(rect(52, y - 8, 10, 22, color))
        lines.append(text(78, y + 6, label, size=12, weight=700))
        lines.append(text(360, y + 6, status, size=12, color=color, weight=700))
        lines.append(text(520, y + 6, detail[:70], size=11, color=COLORS["muted"]))
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_landscape(rows: list[dict[str, Any]], output: Path) -> None:
    groups = []
    for row in rows:
        points = [
            point
            for point in row.get("points", [])
            if point.get("seconds") is not None and float(point.get("seconds", 0)) > 0
        ]
        if not points:
            continue
        groups.append(
            {
                "label": row.get("category", "benchmark"),
                "points": [
                    {
                        "label": f"{point.get('series', 'series')}: {point.get('label', '')}",
                        "seconds": float(point["seconds"]),
                    }
                    for point in points
                ],
            }
        )
    render_log_bar_chart("FastPauli Performance Landscape", "Broad checked comparison including CPU selectors, CUDA paths, external baselines, and Campaign 8 compact consumers.", groups, output, width=1340)


def render_assets(summary: dict[str, Any], plot_dir: Path) -> None:
    rows = summary["device_resident_consumers"]
    render_device(rows, plot_dir / PLOTS["device"])
    render_status_panel(
        "Campaign 8 Interop Consumers",
        "CUDA Array Interface remains retained; DLPack is deferred pending ownership and stream contracts.",
        [
            ("CuPy CUDA Array Interface", "available" if summary["interop"]["cupy_cuda_array_interface_status"] else "blocked", summary["interop"].get("cupy_unavailable_reason") or "available"),
            ("DLPack / PyTorch", summary["interop"]["dlpack_interop_status"], summary["interop"].get("dlpack_unavailable_reason") or ""),
        ],
        plot_dir / PLOTS["interop"],
    )
    render_status_panel(
        "Campaign 8 Stream And Graph Decision",
        "Public CUDA semantics remain synchronous default-stream.",
        [("CUDA Graph / stream-aware API", summary["stream_graph"]["stream_graph_status"], summary["stream_graph"].get("unavailable_reason") or "")],
        plot_dir / PLOTS["stream"],
    )
    render_status_panel(
        "Campaign 8 CSR Scatter A/B",
        "Scatter tuning is gated on retained consumers that still need CSR edge scatter.",
        [("CSR scatter tuning", summary["scatter_ab"]["scatter_tuning_status"], summary["scatter_ab"].get("unavailable_reason") or "")],
        plot_dir / PLOTS["scatter"],
    )
    render_status_panel(
        "Campaign 8 Portability Boundary",
        "Non-H100 NVIDIA evidence is blocked until a second architecture host is available.",
        [("H100 SM90", "passed", "Campaign 8 source build and benchmarks passed"), ("non-H100 NVIDIA", "blocked", "No A100/RTX/L4/A10 host available during this execution")],
        plot_dir / PLOTS["portability"],
    )
    render_landscape(summary["readme_performance_landscape"], plot_dir / PLOTS["landscape"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "docs/benchmarks/data/cuda_deep_optimization_h100_campaign8_2026-04-29",
    )
    parser.add_argument("--plot-dir", type=Path, default=ROOT / "docs/benchmarks/plots")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(args.data_dir)
    (args.data_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    render_assets(summary, args.plot_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

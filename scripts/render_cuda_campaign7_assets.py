#!/usr/bin/env python3
"""Render Campaign 7 H100 CUDA fused-consumer summary and SVG assets."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

PLOTS = {
    "fused": "cuda_h100_campaign7_fused_consumers.svg",
    "grouping": "cuda_h100_campaign7_grouping_summaries.svg",
    "profiler": "cuda_h100_campaign7_profiler_breakdown.svg",
    "portability": "cuda_h100_campaign7_portability.svg",
    "landscape": "cuda_h100_campaign7_performance_landscape.svg",
    "status": "cuda_h100_campaign7_evidence_status.svg",
}

RAW_FILES = {
    "fused": "fused_graph_stress.json",
    "grouping": "fused_grouping_stress.json",
    "count_gate": "count_specialization_ab.json",
    "nsys": "nsys_fused_consumers.json",
}

COLORS = {
    "ink": "#18212f",
    "muted": "#627084",
    "grid": "#d8dee9",
    "panel": "#f7f9fc",
    "blue": "#276ef1",
    "cyan": "#0891b2",
    "green": "#12805c",
    "orange": "#c76a00",
    "red": "#c2410c",
    "purple": "#7c3aed",
    "gray": "#8792a2",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_start(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title>{esc(title)}</title>",
        f"<desc>{esc(subtitle)}</desc>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        f'<text x="32" y="42" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="24" font-weight="700" fill="{COLORS["ink"]}">{esc(title)}</text>',
        f'<text x="32" y="68" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="13" fill="{COLORS["muted"]}">{esc(subtitle)}</text>',
    ]


def text(x: float, y: float, value: object, *, size: int = 12, color: str | None = None,
         weight: int = 400, anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter, ui-sans-serif, system-ui, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color or COLORS["ink"]}" '
        f'text-anchor="{anchor}">{esc(value)}</text>'
    )


def rect(x: float, y: float, width: float, height: float, color: str, *, rx: int = 3,
         opacity: float = 1.0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(width, 0):.1f}" height="{height:.1f}" '
        f'rx="{rx}" fill="{color}" opacity="{opacity:.3f}"/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str, *, width: float = 1.0) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width:.1f}"/>'
    )


def format_seconds(value: float) -> str:
    if value < 1.0e-3:
        return f"{value * 1.0e6:.0f} us"
    if value < 1.0:
        return f"{value * 1.0e3:.2f} ms"
    return f"{value:.2f} s"


def format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    number = float(value)
    for unit in units:
        if number < 1024.0 or unit == units[-1]:
            return f"{number:.1f} {unit}" if unit != "B" else f"{int(number)} B"
        number /= 1024.0
    return f"{value} B"


def load_raw(data_dir: Path) -> dict[str, dict[str, Any]]:
    raw_dir = data_dir / "raw"
    missing = [name for name in RAW_FILES.values() if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError("missing Campaign 7 raw artifact(s): " + ", ".join(missing))
    return {key: read_json(raw_dir / filename) for key, filename in RAW_FILES.items()}


def fused_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in report.get("cases", []):
        result = case["results"]
        rows.append(
            {
                "scale": case["scale"],
                "terms": int(case["dataset"]["lhs_terms"]),
                "dense_to_host_seconds": result["cuda_device_output_dense_to_host_seconds"],
                "csr_seconds": result["cuda_fused_graph_csr_seconds"],
                "conflict_degrees_seconds": result["cuda_fused_conflict_degrees_seconds"],
                "grouping_summary_seconds": result["cuda_fused_grouping_summary_seconds"],
                "device_output_reuse_seconds": result["cuda_device_output_reuse_seconds"],
                "edge_count": int(result["cuda_fused_graph_csr_edge_count"]),
                "csr_host_bytes": int(result["cuda_fused_graph_csr_host_bytes"]),
                "conflict_host_bytes": int(result["cuda_fused_conflict_degrees_host_bytes"]),
                "count_specialization_status": result.get("count_specialization_status"),
                "bitpacked_decision_status": result.get("bitpacked_decision_status"),
            }
        )
    return rows


def load_campaign6_landscape(data_dir: Path) -> list[dict[str, Any]]:
    summary = data_dir.parent / "cuda_deep_optimization_h100_campaign6_2026-04-29" / "summary.json"
    if not summary.exists():
        return []
    return read_json(summary).get("readme_performance_landscape", [])


def campaign7_landscape_rows(campaign6_rows: list[dict[str, Any]], fused: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(campaign6_rows)
    points: list[dict[str, Any]] = []
    for row in fused:
        points.extend(
            [
                {
                    "label": f"{row['terms']}x{row['terms']} grouping summary",
                    "seconds": row["grouping_summary_seconds"],
                    "series": "CUDA fused grouping",
                },
                {
                    "label": f"{row['terms']}x{row['terms']} conflict degrees",
                    "seconds": row["conflict_degrees_seconds"],
                    "series": "CUDA fused degree summary",
                },
                {
                    "label": f"{row['terms']}x{row['terms']} CSR graph export",
                    "seconds": row["csr_seconds"],
                    "series": "CUDA fused CSR export",
                },
            ]
        )
    rows.append(
        {
            "category": "Campaign 7 fused consumers",
            "points": points,
            "baseline_seconds": fused[-1]["dense_to_host_seconds"] if fused else None,
        }
    )
    return rows


def collect_status(data_dir: Path, portability_report: Path) -> list[dict[str, str]]:
    metadata = data_dir / "metadata"
    profiler = data_dir / "profiler"
    raw = data_dir / "raw"
    sanitizer_logs = [
        profiler / "compute_sanitizer_memcheck.log",
        profiler / "compute_sanitizer_racecheck.log",
        profiler / "compute_sanitizer_initcheck.log",
        profiler / "compute_sanitizer_synccheck.log",
    ]
    sanitizer_ok = all(
        path.exists()
        and ("ERROR SUMMARY: 0 errors" in read_text(path) or "0 hazards" in read_text(path))
        for path in sanitizer_logs
    )
    return [
        {
            "label": "H100 repo validation",
            "status": "passed"
            if "Successfully built fastpauli-0.1.0.tar.gz" in read_text(metadata / "experiment-validate-final.log")
            else "missing_or_failed",
        },
        {
            "label": "phase 11 CUDA tests",
            "status": "passed" if "26 passed" in read_text(profiler / "compute_sanitizer_memcheck.log") else "missing_or_failed",
        },
        {
            "label": "compute-sanitizer ladder",
            "status": "passed" if sanitizer_ok else "missing_or_failed",
        },
        {
            "label": "Campaign 7 fused benchmark",
            "status": "passed" if (raw / RAW_FILES["fused"]).exists() else "missing_or_failed",
        },
        {
            "label": "Nsight Systems",
            "status": "passed"
            if (profiler / "nsys_campaign7_fused_consumers.sqlite").exists()
            else "missing_or_failed",
        },
        {
            "label": "Nsight Compute",
            "status": "passed"
            if (profiler / "ncu_campaign7_fused_consumers.ncu-rep").exists()
            and (profiler / "ncu_campaign7_fused_consumers_details.csv").exists()
            else "missing_or_failed",
        },
        {
            "label": "non-H100 NVIDIA portability",
            "status": "blocked_recorded" if portability_report.exists() else "missing_or_failed",
        },
    ]


def gpu_summary(data_dir: Path) -> str:
    path = data_dir / "metadata" / "gpu.csv"
    if not path.exists():
        return "unknown"
    rows = list(csv.reader(path.open("r", encoding="utf-8", newline="")))
    if len(rows) < 2:
        return path.read_text(encoding="utf-8").strip()
    header = [item.strip() for item in rows[0]]
    values = [item.strip() for item in rows[1]]
    fields = dict(zip(header, values, strict=False))
    bits = [fields.get("name", values[0] if values else "unknown")]
    if fields.get("compute_cap"):
        bits.append(f"SM {fields['compute_cap']}")
    if fields.get("driver_version"):
        bits.append(f"driver {fields['driver_version']}")
    return ", ".join(bits)


def ncu_key_metrics(data_dir: Path) -> list[dict[str, Any]]:
    path = data_dir / "profiler" / "ncu_campaign7_fused_consumers_details.csv"
    if not path.exists():
        return []
    wanted = {"Duration", "Memory Throughput", "DRAM Throughput", "Achieved Occupancy"}
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            metric = row.get("Metric Name", "")
            if metric not in wanted:
                continue
            rows.append(
                {
                    "kernel": row.get("Kernel Name", ""),
                    "metric": metric,
                    "unit": row.get("Metric Unit", ""),
                    "value": row.get("Metric Value", ""),
                    "block_size": row.get("Block Size", ""),
                    "grid_size": row.get("Grid Size", ""),
                }
            )
    return rows


def render_log_bar_chart(
    title: str,
    subtitle: str,
    groups: list[dict[str, Any]],
    output: Path,
    *,
    width: int = 1180,
) -> None:
    bar_h = 16
    row_gap = 14
    group_gap = 28
    left = 245
    right = width - 48
    top = 108
    total_rows = sum(len(group["points"]) for group in groups)
    height = top + total_rows * (bar_h + row_gap) + len(groups) * group_gap + 56
    values = [point["seconds"] for group in groups for point in group["points"] if point["seconds"] > 0]
    lo = min(values) if values else 1.0e-6
    hi = max(values) if values else 1.0
    lo_log = math.floor(math.log10(lo))
    hi_log = math.ceil(math.log10(hi))
    if lo_log == hi_log:
        hi_log += 1
    plot_w = right - left
    lines = svg_start(width, height, title, subtitle)
    for tick_power in range(lo_log, hi_log + 1):
        value = 10.0 ** tick_power
        x = left + (math.log10(value) - lo_log) / (hi_log - lo_log) * plot_w
        lines.append(line(x, top - 18, x, height - 42, COLORS["grid"]))
        lines.append(text(x, top - 26, format_seconds(value), size=10, color=COLORS["muted"], anchor="middle"))
    colors = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["cyan"]]
    y = top
    for group in groups:
        lines.append(text(32, y + 1, group["label"], size=13, weight=700))
        for index, point in enumerate(group["points"]):
            y += bar_h + row_gap
            seconds = max(float(point["seconds"]), 1.0e-12)
            x0 = left
            x1 = left + (math.log10(seconds) - lo_log) / (hi_log - lo_log) * plot_w
            color = point.get("color", colors[index % len(colors)])
            lines.append(text(58, y + 12, point["label"], size=11, color=COLORS["ink"]))
            lines.append(rect(x0, y, x1 - x0, bar_h, color))
            lines.append(text(min(x1 + 8, right - 76), y + 12, format_seconds(seconds), size=11, color=COLORS["muted"]))
        y += group_gap
    lines.append(text(32, height - 20, "Log-scale latency; lower is better. Source: checked Campaign 7 raw JSON.", size=11, color=COLORS["muted"]))
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_fused(rows: list[dict[str, Any]], output: Path) -> None:
    groups = []
    for row in rows:
        groups.append(
            {
                "label": f"{row['terms']} x {row['terms']} pairwise commutation",
                "points": [
                    {"label": "Dense to_host", "seconds": row["dense_to_host_seconds"], "color": COLORS["gray"]},
                    {"label": "CSR graph export", "seconds": row["csr_seconds"], "color": COLORS["orange"]},
                    {"label": "Conflict degrees", "seconds": row["conflict_degrees_seconds"], "color": COLORS["blue"]},
                    {"label": "Grouping summary", "seconds": row["grouping_summary_seconds"], "color": COLORS["green"]},
                ],
            }
        )
    render_log_bar_chart(
        "Campaign 7 Fused Consumers",
        "Dense host materialization versus benchmark-only fused graph and grouping consumers on H100.",
        groups,
        output,
    )


def render_grouping(rows: list[dict[str, Any]], output: Path) -> None:
    width = 1120
    height = 120 + len(rows) * 78
    lines = svg_start(
        width,
        height,
        "Campaign 7 Grouping Summaries",
        "Compact summaries avoid full dense host materialization while preserving deterministic conflict ordering.",
    )
    y = 108
    for row in rows:
        dense = row["dense_to_host_seconds"]
        grouping = row["grouping_summary_seconds"]
        speedup = dense / grouping if grouping > 0 else 0.0
        lines.append(rect(32, y - 18, width - 64, 58, COLORS["panel"], rx=6))
        lines.append(text(54, y + 4, f"{row['terms']} x {row['terms']}", size=13, weight=700))
        lines.append(text(220, y + 4, f"grouping summary {format_seconds(grouping)}", size=12, color=COLORS["green"]))
        lines.append(text(470, y + 4, f"dense to_host {format_seconds(dense)}", size=12, color=COLORS["muted"]))
        lines.append(text(710, y + 4, f"{speedup:.1f}x less latency than dense host copy", size=12, color=COLORS["blue"], weight=700))
        lines.append(text(54, y + 26, f"compact host bytes {format_bytes(row['conflict_host_bytes'])}; CSR export bytes {format_bytes(row['csr_host_bytes'])}", size=11, color=COLORS["muted"]))
        y += 78
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_profiler(rows: list[dict[str, Any]], ncu_metrics: list[dict[str, Any]], output: Path) -> None:
    groups = [
        {
            "label": "Largest Campaign 7 H100 row",
            "points": [
                {"label": "CSR graph export", "seconds": rows[-1]["csr_seconds"], "color": COLORS["orange"]},
                {"label": "Dense to_host", "seconds": rows[-1]["dense_to_host_seconds"], "color": COLORS["gray"]},
                {"label": "Conflict degrees", "seconds": rows[-1]["conflict_degrees_seconds"], "color": COLORS["blue"]},
                {"label": "Grouping summary", "seconds": rows[-1]["grouping_summary_seconds"], "color": COLORS["green"]},
            ],
        }
    ]
    render_log_bar_chart(
        "Campaign 7 Profiler-Gated Decision",
        "End-to-end timing shows count kernels are not the dominant retained fused-consumer bottleneck.",
        groups,
        output,
        width=1160,
    )


def render_portability(output: Path) -> None:
    width = 1040
    height = 260
    lines = svg_start(
        width,
        height,
        "Campaign 7 Portability Boundary",
        "No non-H100 NVIDIA host was available; claims remain H100 source-build evidence only.",
    )
    steps = [
        ("H100 SM90", "passed", COLORS["green"]),
        ("A100 SM80", "blocked", COLORS["orange"]),
        ("RTX 6000 Ada SM89", "blocked", COLORS["orange"]),
        ("L4/A10 SM89/SM86", "blocked", COLORS["orange"]),
    ]
    x = 60
    y = 130
    for label, status, color in steps:
        lines.append(rect(x, y, 190, 42, color, rx=6, opacity=0.14))
        lines.append(rect(x, y, 8, 42, color, rx=4))
        lines.append(text(x + 20, y + 18, label, size=12, weight=700))
        lines.append(text(x + 20, y + 35, status, size=11, color=color))
        x += 235
    lines.append(text(60, 220, "Broad NVIDIA GPU claims require replacing the blocker with a retained-consumer run on a second architecture.", size=12, color=COLORS["muted"]))
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
                        "label": f"{point.get('series', 'series')}: {point.get('label', '')}"[:62],
                        "seconds": float(point["seconds"]),
                    }
                    for point in points
                ],
            }
        )
    render_log_bar_chart(
        "FastPauli Performance Landscape",
        "Broad checked comparison including CPU selectors, CUDA paths, external baselines, and Campaign 7 fused consumers.",
        groups,
        output,
        width=1320,
    )


def render_status(statuses: list[dict[str, str]], output: Path) -> None:
    width = 1020
    row_h = 34
    top = 100
    height = top + len(statuses) * row_h + 46
    lines = svg_start(
        width,
        height,
        "Campaign 7 Evidence Status",
        "Validation, sanitizer, profiler, benchmark, and portability gates from checked artifacts.",
    )
    for index, item in enumerate(statuses):
        y = top + index * row_h
        ok = item["status"] in {"passed", "blocked_recorded"}
        color = COLORS["green"] if item["status"] == "passed" else COLORS["orange"] if ok else COLORS["red"]
        lines.append(rect(40, y, 18, 18, color))
        lines.append(text(72, y + 14, item["label"], size=12))
        lines.append(text(540, y + 14, item["status"], size=12, color=color, weight=700))
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(data_dir: Path) -> dict[str, Any]:
    raw = load_raw(data_dir)
    rows = fused_rows(raw["fused"])
    count_rows = fused_rows(raw["count_gate"])
    portability_report = Path("docs/benchmarks/reports/cuda_portability_campaign7_non_h100_nvidia_2026-04-29.md")
    campaign6_landscape = load_campaign6_landscape(data_dir)
    count_status = raw["count_gate"].get("campaign7", {}).get("count_specialization_status")
    bitpacked_status = raw["count_gate"].get("campaign7", {}).get("bitpacked_decision_status")
    if count_status or bitpacked_status:
        rows = [
            {
                **row,
                "count_specialization_status": count_status or row.get("count_specialization_status"),
                "bitpacked_decision_status": bitpacked_status or row.get("bitpacked_decision_status"),
            }
            for row in rows
        ]
    landscape = campaign7_landscape_rows(campaign6_landscape, rows)
    default_report = raw["fused"]
    return {
        "campaign": "h100_campaign7",
        "date": "2026-04-29",
        "experiment_revision": read_text(data_dir / "metadata" / "experiment-revision.txt").strip()
        or str(default_report.get("git_commit", "")),
        "hardware": {
            "gpu": gpu_summary(data_dir),
            "cuda_toolkit": default_report.get("fastpauli_build_info", {}).get("cuda_toolkit_version"),
            "compiled_cuda_architectures": default_report.get("fastpauli_build_info", {}).get("cuda_architectures"),
            "available_cpu_backends": default_report.get("fastpauli_build_info", {}).get("available_cpu_backends", []),
        },
        "decisions": [
            {
                "experiment": "fused_commutation_consumers",
                "status": "benchmark_only_retained",
                "reason": "Private CSR, conflict-degree, and grouping-summary helpers provide evidence without public API expansion.",
            },
            {
                "experiment": "count_reduction_specialization",
                "status": count_status or "rejected_not_dominant",
                "reason": "Profiler and timing evidence show retained grouping summaries are compact and CSR export is dominated by edge-list transfer.",
            },
            {
                "experiment": "async_stream_api",
                "status": "deferred",
                "reason": "No complete public lifetime, event, stream capture, error propagation, and Python ownership contract is accepted.",
            },
            {
                "experiment": "bitpacked_output",
                "status": bitpacked_status or "deferred_no_dense_capacity_or_bandwidth_trigger",
                "reason": "Dense fused consumers did not prove a packed-layout capacity or bandwidth trigger.",
            },
            {
                "experiment": "non_h100_portability",
                "status": "blocked_recorded",
                "reason": "No non-H100 NVIDIA host was available; claims remain H100 source-build evidence only.",
            },
        ],
        "fused_consumers": rows,
        "count_gate": count_rows,
        "ncu_key_metrics": ncu_key_metrics(data_dir),
        "readme_performance_landscape": landscape,
        "evidence": {
            "status": collect_status(data_dir, portability_report),
            "raw_json": sorted(path.name for path in (data_dir / "raw").glob("*.json")),
            "profilers": ["nsys", "ncu"],
            "sanitizers": ["memcheck", "racecheck", "initcheck", "synccheck"],
            "portability_report": str(portability_report),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29"),
    )
    parser.add_argument("--plot-dir", type=Path, default=Path("docs/benchmarks/plots"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(args.data_dir)
    (args.data_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = summary["fused_consumers"]
    render_fused(rows, args.plot_dir / PLOTS["fused"])
    render_grouping(rows, args.plot_dir / PLOTS["grouping"])
    render_profiler(rows, summary["ncu_key_metrics"], args.plot_dir / PLOTS["profiler"])
    render_portability(args.plot_dir / PLOTS["portability"])
    render_landscape(summary["readme_performance_landscape"], args.plot_dir / PLOTS["landscape"])
    render_status(summary["evidence"]["status"], args.plot_dir / PLOTS["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

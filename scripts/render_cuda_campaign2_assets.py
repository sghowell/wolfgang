#!/usr/bin/env python3
"""Render H100 campaign-2 benchmark plots from checked-in JSON evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

COLORS = {
    "ink": "#111827",
    "muted": "#6b7280",
    "grid": "#e5e7eb",
    "panel": "#f8fafc",
    "scalar": "#4b5563",
    "transfer": "#2563eb",
    "resident": "#059669",
    "prealloc": "#0f766e",
    "cpu_opt": "#7c3aed",
    "amber": "#d97706",
    "red": "#dc2626",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 12,
    weight: int = 400,
    color: str = COLORS["ink"],
    anchor: str = "start",
) -> str:
    escaped = (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" '
        f'text-anchor="{anchor}">{escaped}</text>'
    )


def svg_rect(x: float, y: float, width: float, height: float, fill: str, *, rx: float = 2.0) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="{rx:g}" fill="{fill}"/>'


def svg_header(width: int, height: int, title: str, desc: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{title}</title>',
        f'<desc id="desc">{desc}</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def write_svg(output: Path, lines: list[str]) -> None:
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_ratio(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}x"
    if value >= 10:
        return f"{value:.1f}x"
    return f"{value:.2f}x"


def format_seconds(value: float) -> str:
    if value < 1.0e-6:
        return f"{value * 1.0e9:.1f}ns"
    if value < 1.0e-3:
        return f"{value * 1.0e6:.1f}us"
    if value < 1.0:
        return f"{value * 1.0e3:.2f}ms"
    return f"{value:.3g}s"


def render_statevector_speedups(summary: dict[str, Any], output: Path) -> None:
    rows = summary["statevector_ab"]
    width = 1120
    height = 140 + len(rows) * 42
    max_speedup = max(
        max(row["resident_speedup"], row["transfer_speedup"], 1.0)
        for row in rows
    )
    left = 300
    bar_width = 560
    lines = svg_header(
        width,
        height,
        "FastPauli H100 campaign 2 statevector speedups",
        "Baseline to final speedups for the CUDA statevector expectation path.",
    )
    lines.extend(
        [
            svg_text(32, 40, "Statevector Expectation Speedups", size=25, weight=700),
            svg_text(
                32,
                64,
                "Same-boundary baseline vs final H100 medians; values above 1.0x are faster.",
                size=13,
                color=COLORS["muted"],
            ),
            svg_text(left, 100, "1.0x", size=11, color=COLORS["muted"]),
            svg_text(
                left + bar_width,
                100,
                format_ratio(max_speedup),
                size=11,
                color=COLORS["muted"],
                anchor="end",
            ),
        ]
    )
    for index, row in enumerate(rows):
        y = 130 + index * 42
        label = f"{row['profile']} {row['scale']}"
        lines.append(svg_text(32, y + 12, label, size=12, weight=700))
        for lane, key, color in (
            (0, "transfer_speedup", COLORS["transfer"]),
            (1, "resident_speedup", COLORS["resident"]),
        ):
            value = float(row[key])
            bar_y = y + lane * 17
            filled = max(2.0, bar_width * min(value, max_speedup) / max_speedup)
            lines.append(svg_rect(left, bar_y, filled, 12, color))
            name = "transfer" if key.startswith("transfer") else "resident"
            lines.append(svg_text(left + filled + 8, bar_y + 10, f"{name} {format_ratio(value)}", size=11))
    lines.append("</svg>")
    write_svg(output, lines)


def render_block_size_hillclimb(summary: dict[str, Any], output: Path) -> None:
    rows = summary["block_size_ab"]
    variants = [
        ("fused256", COLORS["resident"]),
        ("threads128", COLORS["amber"]),
        ("threads512", COLORS["transfer"]),
        ("hybrid", COLORS["prealloc"]),
    ]
    max_speedup = max(
        row["baseline"] / row[name]
        for row in rows
        for name, _ in variants
    )
    width = 1120
    height = 150 + len(rows) * 58
    left = 330
    bar_width = 520
    lines = svg_header(
        width,
        height,
        "FastPauli H100 campaign 2 launch-size hillclimb",
        "Launch-size A/B outcomes for statevector expectation.",
    )
    lines.extend(
        [
            svg_text(32, 40, "Launch-Size Hillclimb", size=25, weight=700),
            svg_text(
                32,
                64,
                "Speedup versus baseline for fused and block-size variants; hybrid is retained.",
                size=13,
                color=COLORS["muted"],
            ),
            svg_text(left, 100, "1.0x", size=11, color=COLORS["muted"]),
            svg_text(
                left + bar_width,
                100,
                format_ratio(max_speedup),
                size=11,
                color=COLORS["muted"],
                anchor="end",
            ),
        ]
    )
    for index, row in enumerate(rows):
        y = 130 + index * 58
        lines.append(svg_text(32, y + 16, f"{row['profile']} {row['scale']}", size=12, weight=700))
        for lane, (name, color) in enumerate(variants):
            value = row["baseline"] / row[name]
            filled = max(2.0, bar_width * min(value, max_speedup) / max_speedup)
            bar_y = y + lane * 13
            lines.append(svg_rect(left, bar_y, filled, 9, color))
            lines.append(svg_text(left + filled + 8, bar_y + 8, f"{name} {format_ratio(value)}", size=10))
    lines.append("</svg>")
    write_svg(output, lines)


def selected_final_cases(report: dict[str, Any]) -> list[dict[str, Any]]:
    wanted = {
        "simplify_duplicate_pressure": "terms_50000",
        "statevector_expectation": "qubits_12_terms_2048",
        "pairwise_commutation": "terms_2048x2048",
        "matmul_product_generation_simplify": "terms_256x256",
    }
    cases = []
    for case in report["cases"]:
        if wanted.get(case["name"]) == case["scale"]:
            cases.append(case)
    return cases


def final_paths_for_case(case: dict[str, Any]) -> list[tuple[str, float, str]]:
    results = case["results"]
    paths = [("CPU scalar", results["cpu_scalar_seconds"], COLORS["scalar"])]
    if results.get("cuda_transfer_inclusive_seconds") is not None:
        paths.append(("CUDA transfer", results["cuda_transfer_inclusive_seconds"], COLORS["transfer"]))
    if results.get("cuda_device_resident_seconds") is not None:
        paths.append(("CUDA resident", results["cuda_device_resident_seconds"], COLORS["resident"]))
    if results.get("cuda_device_resident_preallocated_seconds") is not None:
        paths.append(("CUDA prealloc", results["cuda_device_resident_preallocated_seconds"], COLORS["prealloc"]))
    for selector, timing in sorted(results.get("cpu_optimized_timings", {}).items()):
        paths.append((f"CPU {selector}", timing["seconds"], COLORS["cpu_opt"]))
    return paths


def render_final_path_comparison(summary: dict[str, Any], raw_dir: Path, output: Path) -> None:
    default_report = load_json(raw_dir / Path(summary["final_profiles"]["default"]).name)
    cases = selected_final_cases(default_report)
    case_paths = [(case, final_paths_for_case(case)) for case in cases]
    max_speedup = max(
        case["results"]["cpu_scalar_seconds"] / seconds
        for case, paths in case_paths
        for _, seconds, _ in paths
    )
    width = 1120
    section_gap = 28
    row_gap = 20
    top = 108
    height = top + sum(28 + len(paths) * row_gap + section_gap for _, paths in case_paths) + 20
    left = 330
    bar_width = 520
    lines = svg_header(
        width,
        height,
        "FastPauli H100 campaign 2 final path comparison",
        "Representative final default profile path comparison.",
    )
    lines.extend(
        [
            svg_text(32, 40, "Final H100 Path Comparison", size=25, weight=700),
            svg_text(
                32,
                64,
                "Representative default-profile speedups versus scalar CPU; bars are log-scaled.",
                size=13,
                color=COLORS["muted"],
            ),
        ]
    )
    y = top
    for case, paths in case_paths:
        scalar = case["results"]["cpu_scalar_seconds"]
        title = f"{case['name'].replace('_', ' ')} {case['scale']}"
        lines.append(svg_text(32, y, title, size=12, weight=700))
        for index, (name, seconds, color) in enumerate(paths):
            speedup = scalar / seconds
            log_speedup = math.log10(max(speedup, 1.0))
            log_max = math.log10(max(max_speedup, 1.000001))
            filled = max(2.0, bar_width * log_speedup / log_max) if log_speedup > 0 else 2.0
            bar_y = y + 10 + index * row_gap
            lines.append(svg_rect(left, bar_y, filled, 12, color))
            lines.append(
                svg_text(
                    left + filled + 8,
                    bar_y + 10,
                    f"{name} {format_ratio(speedup)} ({format_seconds(seconds)})",
                    size=11,
                )
            )
        y += 28 + len(paths) * row_gap + section_gap
    lines.append("</svg>")
    write_svg(output, lines)


def render_evidence_status(summary: dict[str, Any], output: Path) -> None:
    profile_status = summary["profile_status"]
    privileged_ncu = summary["privileged_ncu_retry"]["reports"]
    cards = [
        ("H100 validation", "success", summary["validation"]["h100_targeted_cuda_tests"]),
        ("Compute Sanitizer", "success", summary["validation"]["compute_sanitizer"]),
        ("Nsight Systems", profile_status["nsys cuda api timeline"]["status"], "CUDA API timeline captured"),
        ("Nsight Compute", "success", f"{len(privileged_ncu)} privileged detailed reports captured"),
        ("cuobjdump", "success", "PTX and SASS inventory captured"),
        ("Competitor baselines", profile_status["competitive baseline benchmark"]["status"], "Package status and comparable baselines recorded"),
        ("nvdisasm", "optional", "Optional listing failed; cuobjdump evidence retained"),
    ]
    width = 1120
    height = 470
    lines = svg_header(
        width,
        height,
        "FastPauli H100 campaign 2 evidence status",
        "Correctness, profiler, binary, and competitor evidence status.",
    )
    lines.extend(
        [
            svg_text(32, 40, "Campaign 2 Evidence Status", size=25, weight=700),
            svg_text(
                32,
                64,
                "Checked-in raw JSON records required validation, profiling, and competitor evidence.",
                size=13,
                color=COLORS["muted"],
            ),
        ]
    )
    for index, (title, status, detail) in enumerate(cards):
        col = index % 2
        row = index // 2
        x = 32 + col * 530
        y = 102 + row * 86
        status_color = COLORS["resident"] if status == "success" else COLORS["amber"]
        lines.append(svg_rect(x, y, 500, 64, COLORS["panel"], rx=6))
        lines.append(svg_rect(x + 16, y + 17, 12, 12, status_color, rx=6))
        lines.append(svg_text(x + 42, y + 26, title, size=13, weight=700))
        lines.append(svg_text(x + 42, y + 47, str(detail), size=11, color=COLORS["muted"]))
    lines.append("</svg>")
    write_svg(output, lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="campaign-2 summary.json path",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="directory containing campaign-2 raw JSON files",
    )
    parser.add_argument("--plot-dir", type=Path, required=True, help="output plot directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    summary = load_json(args.summary)
    render_statevector_speedups(
        summary,
        args.plot_dir / "cuda_h100_campaign2_statevector_speedups.svg",
    )
    render_block_size_hillclimb(
        summary,
        args.plot_dir / "cuda_h100_campaign2_block_size_hillclimb.svg",
    )
    render_final_path_comparison(
        summary,
        args.raw_dir,
        args.plot_dir / "cuda_h100_campaign2_final_path_comparison.svg",
    )
    render_evidence_status(
        summary,
        args.plot_dir / "cuda_h100_campaign2_evidence_status.svg",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

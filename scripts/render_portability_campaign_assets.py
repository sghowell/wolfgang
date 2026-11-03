#!/usr/bin/env python3
"""Render the portability and interop campaign speedup chart as a static SVG.

Each row compares a new path with the path it replaced, at one named timing
boundary, using medians from the cited report. The chart is hand-written SVG
(no plotting dependency), like the other renderers in this directory.
"""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "benchmarks" / "plots" / "portability_campaign_speedups.svg"

# Categorical slots 1-4 of the validated reference palette (light surface).
BACKEND_COLORS = {"CPU": "#2a78d6", "CUDA": "#eb6834", "HIP": "#1baf7a", "Metal": "#eda100"}
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

# (backend, label, speedup as reported, before, after, source report)
ROWS: list[tuple[str, str, float, str, str, str]] = [
    ("CPU", "pip wheel with oneTBB: simplify, 1M terms (M4 Pro)", 5.57, "70.7 ms", "12.7 ms", "onetbb_wheels_2026-11-03.md"),
    ("CPU", "pip wheel with oneTBB: simplify, 1M terms (EPYC, 16 vCPU)", 2.15, "141 ms", "65.4 ms", "onetbb_wheels_2026-11-03.md"),
    ("CUDA", "Hand 1M-term operator to CuPy (zero copy vs via host)", 244.0, "8.83 ms", "36.2 us", "device_arrays_h100_2026-11-03.md"),
    ("CUDA", "Build operator from CuPy arrays, 1M terms (vs via host)", 247.0, "20.7 ms", "83.7 us", "device_arrays_h100_2026-11-03.md"),
    ("CUDA", "Fused small simplify, 128 terms", 7.80, "143.5 us", "18.4 us", "small_simplify_2026-11-03.md"),
    ("CUDA", "Pooled commutation output, 128 x 128 (sole cudaMalloc buffer)", 6.06, "64.1 us", "10.6 us", "cuda_commutation_pool_h100_2026-11-03.md"),
    ("HIP", "Single-term simplify", 8.22, "254.5 us", "31.0 us", "hip_parity_mi300x_2026-11-03.md"),
    ("HIP", "Fused small simplify, 128 terms", 5.68, "152.7 us", "26.9 us", "small_simplify_2026-11-03.md"),
    ("Metal", "Precompiled kernels: first simplify call", 1.26, "3.89 ms", "3.08 ms", "metal_followups_2026-11-03.md"),
    ("Metal", "Parallel reduction: hybrid simplify, 4M unique terms", 1.12, "85.6 ms", "76.7 ms", "metal_followups_2026-11-03.md"),
]


WIDTH = 980
LABEL_X = 24
PLOT_LEFT = 470
PLOT_RIGHT = WIDTH - 70
TOP = 118
ROW_HEIGHT = 34
BAR_HEIGHT = 14
AXIS_MAX = 400.0


def x_for(ratio: float) -> float:
    return PLOT_LEFT + (PLOT_RIGHT - PLOT_LEFT) * math.log10(ratio) / math.log10(AXIS_MAX)


def fmt_ratio(ratio: float) -> str:
    if ratio >= 100:
        return f"{ratio:.0f}x"
    if ratio >= 10:
        return f"{ratio:.1f}x"
    return f"{ratio:.2f}x"


def render(rows: list[tuple[str, str, float, str, str, str]]) -> str:
    height = TOP + ROW_HEIGHT * len(rows) + 92
    esc = html.escape
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" '
        'role="img" aria-labelledby="title desc" font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif">',
        '<title id="title">Measured speedups from the portability and interop campaign</title>',
        '<desc id="desc">Horizontal bars on a log scale. '
        + esc("; ".join(f"{b} {label}: {fmt_ratio(ratio)}" for b, label, ratio, _, _, _ in rows))
        + "</desc>",
        f'<rect width="{WIDTH}" height="{height}" fill="{SURFACE}"/>',
        f'<text x="{LABEL_X}" y="38" font-size="20" font-weight="600" fill="{TEXT_PRIMARY}">'
        "What the portability and interop campaign changed</text>",
        f'<text x="{LABEL_X}" y="62" font-size="13" fill="{TEXT_SECONDARY}">'
        "Speedup of each new path over the one it replaced: medians at one named boundary per row. Log scale.</text>",
    ]
    legend_x = LABEL_X
    for backend, color in BACKEND_COLORS.items():
        out.append(f'<rect x="{legend_x}" y="80" width="12" height="12" rx="3" fill="{color}"/>')
        out.append(f'<text x="{legend_x + 18}" y="91" font-size="13" fill="{TEXT_PRIMARY}">{backend}</text>')
        legend_x += 30 + 9 * len(backend) + 24
    plot_bottom = TOP + ROW_HEIGHT * len(rows)
    for tick in (1, 2, 5, 10, 20, 50, 100, 200):
        x = x_for(tick)
        out.append(f'<line x1="{x:.1f}" y1="{TOP - 8}" x2="{x:.1f}" y2="{plot_bottom}" stroke="{GRID}" stroke-width="1"/>')
        out.append(
            f'<text x="{x:.1f}" y="{plot_bottom + 18}" font-size="12" text-anchor="middle" fill="{TEXT_SECONDARY}">{tick}x</text>'
        )
    for index, (backend, label, ratio, before, after, source) in enumerate(rows):
        y = TOP + index * ROW_HEIGHT
        mid = y + ROW_HEIGHT / 2
        end = x_for(ratio)
        color = BACKEND_COLORS[backend]
        tooltip = esc(f"{backend}: {label}. {before} to {after} ({fmt_ratio(ratio)}). Source: {source}")
        out.append(f'<g><title>{tooltip}</title>')
        out.append(
            f'<text x="{LABEL_X}" y="{mid + 4:.1f}" font-size="13" fill="{TEXT_PRIMARY}">'
            f'<tspan font-weight="600">{backend}</tspan>  {esc(label)}</text>'
        )
        # Bar anchored at 1x with a 4px rounded data end and a square baseline end.
        width = max(end - PLOT_LEFT, 2.0)
        top = mid - BAR_HEIGHT / 2
        radius = min(4.0, width / 2)
        out.append(
            f'<path d="M{PLOT_LEFT:.1f},{top:.1f} h{width - radius:.1f} a{radius},{radius} 0 0 1 {radius},{radius} '
            f'v{BAR_HEIGHT - 2 * radius:.1f} a{radius},{radius} 0 0 1 -{radius},{radius} h-{width - radius:.1f} z" fill="{color}"/>'
        )
        out.append(
            f'<text x="{end + 8:.1f}" y="{mid + 4:.1f}" font-size="13" font-weight="600" fill="{TEXT_PRIMARY}">'
            f"{fmt_ratio(ratio)}</text>"
        )
        out.append("</g>")
    out.append(
        f'<line x1="{PLOT_LEFT}" y1="{TOP - 8}" x2="{PLOT_LEFT}" y2="{plot_bottom}" stroke="{TEXT_SECONDARY}" stroke-width="1"/>'
    )
    out.append(
        f'<text x="{LABEL_X}" y="{plot_bottom + 48}" font-size="12" fill="{TEXT_SECONDARY}">'
        "Hosts: NVIDIA H100, AMD MI300X, Apple M4 Pro, AMD EPYC 9654 VM. Each row's report in docs/benchmarks/reports/ "
        "has the boundary, commands and spread.</text>"
    )
    out.append(
        f'<text x="{LABEL_X}" y="{plot_bottom + 66}" font-size="12" fill="{TEXT_SECONDARY}">'
        "Rows are not comparable with each other: they measure different operations, sizes and hardware.</text>"
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if any(ratio <= 0 or not after for _, _, ratio, _, after, _ in ROWS):
        raise SystemExit("every row needs a measured speedup and 'after' value")
    args.output.write_text(render(ROWS), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

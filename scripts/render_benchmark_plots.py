#!/usr/bin/env python3
"""Render lightweight benchmark plots from checked-in evidence reports."""

from __future__ import annotations

import argparse
import html
import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BackendTiming:
    label: str
    seconds: float
    color: str


@dataclass(frozen=True)
class CudaBenchmarkCase:
    name: str
    dataset: str
    cpu_scalar_seconds: float
    cpu_optimized_timings: tuple[BackendTiming, ...]
    cuda_transfer_seconds: float
    cuda_device_seconds: float

    def backend_timings(self) -> tuple[BackendTiming, ...]:
        return (
            BackendTiming("CPU scalar", self.cpu_scalar_seconds, "#6b7280"),
            *self.cpu_optimized_timings,
            BackendTiming(
                "CUDA transfer",
                self.cuda_transfer_seconds,
                "#2563eb",
            ),
            BackendTiming(
                "CUDA resident",
                self.cuda_device_seconds,
                "#059669",
            ),
        )


def split_markdown_table_row(row: str) -> list[str]:
    stripped = row.strip()
    stripped = stripped.removeprefix("|")
    stripped = stripped.removesuffix("|")
    return [cell.strip() for cell in stripped.split("|")]


def parse_seconds(cell: str) -> float:
    value = float(cell.strip())
    if value <= 0.0:
        raise ValueError(f"benchmark timing must be positive: {cell!r}")
    return value


def parse_optimized_cpu(cell: str) -> tuple[BackendTiming, ...]:
    if cell.strip().lower() in {"n/a", "na", ""}:
        return ()

    timings: list[BackendTiming] = []
    for entry in re.split(r"[,;]", cell):
        entry = entry.strip()
        if not entry:
            continue
        match = re.fullmatch(r"([^:]+):\s*([0-9.eE+-]+)", entry)
        if match is None:
            raise ValueError(f"malformed CPU optimized timing: {cell!r}")
        timings.append(
            BackendTiming(
                f"CPU {match.group(1).strip()}",
                parse_seconds(match.group(2)),
                "#7c3aed",
            )
        )
    return tuple(timings)


def parse_cuda_default_cases(report: Path) -> list[CudaBenchmarkCase]:
    lines = report.read_text(encoding="utf-8").splitlines()
    in_default_section = False
    rows: list[str] = []
    for line in lines:
        if line.startswith("## Benchmark Default"):
            in_default_section = True
            continue
        if in_default_section and line.startswith("## "):
            break
        if in_default_section and line.startswith("| "):
            rows.append(line)

    data_rows = [
        row
        for row in rows
        if not row.startswith("| ---") and not row.startswith("| Case |")
    ]
    cases: list[CudaBenchmarkCase] = []
    for row in data_rows:
        cells = split_markdown_table_row(row)
        if len(cells) < 7:
            continue
        cases.append(
            CudaBenchmarkCase(
                name=cells[0],
                dataset=cells[1],
                cpu_scalar_seconds=parse_seconds(cells[2]),
                cpu_optimized_timings=parse_optimized_cpu(cells[3]),
                cuda_transfer_seconds=parse_seconds(cells[4]),
                cuda_device_seconds=parse_seconds(cells[5]),
            )
        )
    if not cases:
        raise ValueError(f"no CUDA default benchmark rows found in {report}")
    return cases


def format_speedup(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}x"
    if value >= 10:
        return f"{value:.1f}x"
    return f"{value:.2f}x"


def format_seconds(value: float) -> str:
    if value < 1.0e-4:
        return f"{value:.2e} s"
    if value < 1.0:
        return f"{value:.4f} s"
    return f"{value:.3f} s"


def bar_width(speedup: float, max_speedup: float, max_width: float) -> float:
    if speedup <= 1.0:
        return 2.0
    denominator = math.log10(max(max_speedup, 1.01))
    return max_width * (math.log10(speedup) / denominator)


def render_cuda_speedup_svg(cases: list[CudaBenchmarkCase], source_report: Path) -> str:
    left = 270
    max_width = 530
    bar_height = 16
    bar_gap = 8
    case_gap = 34
    width = 960
    case_heights = [
        36 + len(case.backend_timings()) * (bar_height + bar_gap) + case_gap
        for case in cases
    ]
    height = 136 + sum(case_heights) + 38
    max_speedup = max(
        case.cpu_scalar_seconds / timing.seconds
        for case in cases
        for timing in case.backend_timings()
    )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        "<title id=\"title\">Wolfgang H100 CUDA backend speedups</title>",
        "<desc id=\"desc\">CPU scalar, optimized CPU, CUDA transfer-inclusive, and "
        "CUDA device-resident speedups versus CPU scalar from the checked-in H100 "
        "CUDA benchmark default report.</desc>",
        "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>",
        "<text x=\"32\" y=\"34\" font-family=\"Arial, sans-serif\" font-size=\"24\" "
        "font-weight=\"700\" fill=\"#111827\">H100 CUDA default backend speedups</text>",
        "<text x=\"32\" y=\"58\" font-family=\"Arial, sans-serif\" font-size=\"13\" "
        "fill=\"#4b5563\">Compared with CPU scalar on the same "
        "deterministic dataset; bar lengths use log scale.</text>",
        "<text x=\"32\" y=\"78\" font-family=\"Arial, sans-serif\" font-size=\"12\" "
        f"fill=\"#6b7280\">Source: {html.escape(str(source_report))}</text>",
        f"<line x1=\"{left}\" y1=\"94\" x2=\"{left + max_width}\" y2=\"94\" stroke=\"#d1d5db\"/>",
        f"<text x=\"{left}\" y=\"88\" font-family=\"Arial, sans-serif\" font-size=\"11\" "
        "fill=\"#6b7280\">1x</text>",
        f"<text x=\"{left + max_width}\" y=\"88\" font-family=\"Arial, sans-serif\" font-size=\"11\" "
        f"text-anchor=\"end\" fill=\"#6b7280\">{format_speedup(max_speedup)}</text>",
    ]

    y_cursor = 120
    for index, case in enumerate(cases):
        y = y_cursor
        case_label = case.name.replace("_", " ")
        lines.extend(
            [
                f"<text x=\"32\" y=\"{y}\" font-family=\"Arial, sans-serif\" "
                f"font-size=\"15\" font-weight=\"700\" fill=\"#111827\">{html.escape(case_label)}</text>",
                f"<text x=\"32\" y=\"{y + 18}\" font-family=\"Arial, sans-serif\" "
                f"font-size=\"11\" fill=\"#6b7280\">{html.escape(case.dataset)}</text>",
            ]
        )
        bar_y = y + 30
        for timing in case.backend_timings():
            speedup = case.cpu_scalar_seconds / timing.seconds
            width_px = bar_width(speedup, max_speedup, max_width)
            lines.extend(
                [
                    f"<rect x=\"{left}\" y=\"{bar_y}\" width=\"{width_px:.2f}\" "
                    f"height=\"{bar_height}\" rx=\"2\" fill=\"{timing.color}\"/>",
                    f"<text x=\"{left + max(width_px, 4) + 8:.2f}\" y=\"{bar_y + 12}\" font-family=\"Arial, sans-serif\" "
                    f"font-size=\"12\" fill=\"#111827\">{html.escape(timing.label)} {format_speedup(speedup)} "
                    f"({format_seconds(timing.seconds)})</text>",
                ]
            )
            bar_y += bar_height + bar_gap
        y_cursor += case_heights[index]

    lines.extend(
        [
            f"<rect x=\"32\" y=\"{height - 28}\" width=\"14\" height=\"14\" fill=\"#6b7280\"/>",
            f"<text x=\"52\" y=\"{height - 17}\" font-family=\"Arial, sans-serif\" font-size=\"12\" "
            "fill=\"#374151\">CPU scalar</text>",
            f"<rect x=\"148\" y=\"{height - 28}\" width=\"14\" height=\"14\" fill=\"#7c3aed\"/>",
            f"<text x=\"168\" y=\"{height - 17}\" font-family=\"Arial, sans-serif\" font-size=\"12\" "
            "fill=\"#374151\">CPU optimized</text>",
            f"<rect x=\"302\" y=\"{height - 28}\" width=\"14\" height=\"14\" fill=\"#2563eb\"/>",
            f"<text x=\"322\" y=\"{height - 17}\" font-family=\"Arial, sans-serif\" font-size=\"12\" "
            "fill=\"#374151\">CUDA transfer-inclusive</text>",
            f"<rect x=\"516\" y=\"{height - 28}\" width=\"14\" height=\"14\" fill=\"#059669\"/>",
            f"<text x=\"536\" y=\"{height - 17}\" font-family=\"Arial, sans-serif\" font-size=\"12\" "
            "fill=\"#374151\">CUDA device-resident</text>",
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = parse_cuda_default_cases(args.cuda_report)
    svg = render_cuda_speedup_svg(cases, args.cuda_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()

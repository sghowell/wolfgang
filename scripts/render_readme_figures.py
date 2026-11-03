#!/usr/bin/env python3
"""Render the README performance figures from committed benchmark data.

Three hand-written SVGs (no plotting dependency), each with light and dark
styles that follow the viewer's color scheme:

``simplify_by_backend.svg``
    Simplify time against operator size on each backend.
``wolfgang_vs_other_libraries.svg``
    Wolfgang against Qiskit, OpenFermion and cuStateVec.
``call_time_by_slice.svg``
    Per-call times on the H100 and MI300X across the slices that changed them.

Every value is read from a JSON file under docs/benchmarks/data/, named in the
``SOURCES`` notes, so rerunning this script after a new report updates the
figures. Run with ``--check`` to fail when the committed SVGs are stale.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "benchmarks" / "data"
PLOTS = ROOT / "docs" / "benchmarks" / "plots"

# Categorical slots 1-5 of the validated reference palette (light / dark). The
# backend colors match docs/benchmarks/plots/portability_campaign_speedups.svg.
COLORS = {
    "s1": ("#2a78d6", "#3987e5"),
    "s2": ("#eb6834", "#d95926"),
    "s3": ("#1baf7a", "#199e70"),
    "s4": ("#eda100", "#c98500"),
    "s5": ("#e87ba4", "#d55181"),
    "neutral": ("#6f6e69", "#a3a29d"),
}
STYLE = """
.bg { fill: #fcfcfb; } .t1 { fill: #0b0b0b; } .t2 { fill: #52514e; } .t3 { fill: #75746f; }
.grid { stroke: #e4e3df; } .axis { stroke: #c9c8c3; }
%(light)s
@media (prefers-color-scheme: dark) {
  .bg { fill: #1a1a19; } .t1 { fill: #f2f1ed; } .t2 { fill: #b5b4ae; } .t3 { fill: #94938d; }
  .grid { stroke: #33322f; } .axis { stroke: #4a4945; }
  %(dark)s
}
"""
FONT = "-apple-system, Segoe UI, Helvetica, Arial, sans-serif"


def series_style() -> str:
    # Stroke and fill classes are separate: a CSS fill would override a path's
    # fill="none" presentation attribute.
    def rules(mode: int) -> str:
        return " ".join(f".s-{k} {{ stroke: {v[mode]}; }} .f-{k} {{ fill: {v[mode]}; }}" for k, v in COLORS.items())

    return STYLE % {"light": rules(0), "dark": rules(1)}


def load(relative: str) -> Any:
    return json.loads((DATA / relative).read_text(encoding="utf-8"))


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def fmt_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.3g} \u00b5s"
    if seconds < 1:
        return f"{seconds * 1e3:.3g} ms"
    return f"{seconds:.3g} s"


def fmt_terms(terms: float) -> str:
    if terms >= 1e6:
        return f"{terms / 1e6:g}M"
    if terms >= 1e3:
        return f"{terms / 1e3:g}K"
    return f"{terms:g}"


# --- data -----------------------------------------------------------------


@dataclass
class Series:
    key: str  # color slot
    label: str
    points: list[tuple[float, float]]
    dashed: bool = False
    marker: str = "circle"


def overhead(relative: str, row: str, label: str) -> float:
    return load(relative)["rows"][row][label]["median_of_medians_seconds"]


def paths_row(relative: str, terms: int, variant: str, label: str) -> float:
    for row in load(relative)["specialized_rows"]:
        if row["path"] == "one_word_packed32" and row["num_terms"] == terms and row["variant"] == variant:
            return row["median_of_medians_seconds"][label]
    raise KeyError(f"{relative}: no one_word_packed32 row at {terms} terms for {variant}")


def cpu_rows(relative: str) -> list[dict[str, Any]]:
    rows = [r for r in load(relative)["rows"] if r["num_qubits"] == 24 and r["duplicate_fraction"] == 0.5]
    return sorted(rows, key=lambda r: r["num_terms"])


def metal_rows(relative: str, sizes: Sequence[int]) -> list[tuple[float, float]]:
    # The hybrid summaries store the duplicate fraction either as 0.5 or as 500 (per mille).
    out = []
    for row in load(relative)["rows"]:
        if row["num_qubits"] == 24 and row["duplicate_fraction"] in (0.5, 500.0) and row["num_terms"] in sizes:
            out.append((row["num_terms"], row["median_of_medians_seconds"]["metal_hybrid"]))
    return sorted(out)


CPU_X86 = "cpu_parallel_simplify_2026-11-03/summary_x86_64_26vcpu_onetbb2021_11.json"
CUDA_OVERHEAD = "cuda_device_counts_h100_2026-11-03/overhead_summary.json"
CUDA_PATHS = "cuda_device_counts_h100_2026-11-03/paths_summary.json"
HIP_OVERHEAD = "hip_device_counts_mi300x_2026-11-03/overhead_summary.json"
HIP_PATHS = "hip_device_counts_mi300x_2026-11-03/paths_summary.json"
METAL_HYBRID = "metal_hybrid_simplify_2026-11-03/summary_apple_m4_pro.json"
METAL_LARGE = "metal_followups_2026-11-03/summary_large_candidate.json"
COMPETITIVE = "cuda_deep_optimization_h100_2026-04-28/summary.json"


def simplify_series() -> list[Series]:
    cpu = cpu_rows(CPU_X86)
    small = [(128, "simplify_128_terms_24q"), (1024, "simplify_1024_terms_24q"), (4096, "simplify_4096_terms_24q")]

    def accelerator(overhead_file: str, paths_file: str, label: str, variant: str) -> list[tuple[float, float]]:
        points = [(terms, overhead(overhead_file, row, label)) for terms, row in small]
        points += [(terms, paths_row(paths_file, terms, variant, label)) for terms in (50_000, 1_000_000)]
        return points

    metal = metal_rows(METAL_HYBRID, (12_288, 16_384, 65_536, 262_144))
    metal += metal_rows(METAL_LARGE, (1_048_576, 4_194_304))
    cpu_small = [(128, overhead(CUDA_OVERHEAD, "cpu_simplify_128_terms_control", "cand"))]
    return [
        Series("s1", "CPU scalar (Xeon 8480+)",
               cpu_small + [(r["num_terms"], r["median_of_medians_seconds"]["scalar"]) for r in cpu]),
        Series("s5", "CPU oneTBB, 26 threads (Xeon)", [(r["num_terms"], r["median_of_medians_seconds"]["tbb"]) for r in cpu]),
        Series("s2", "CUDA, H100", accelerator(CUDA_OVERHEAD, CUDA_PATHS, "cand", "cuda_device_resident")),
        Series("s3", "HIP, MI300X", accelerator(HIP_OVERHEAD, HIP_PATHS, "cache", "hip_device_resident")),
        Series("s4", "Metal, M4 Pro", metal),
    ]


def qiskit_series() -> list[Series]:
    cpu = cpu_rows(CPU_X86)

    def pick(column: str) -> list[tuple[float, float]]:
        return [(r["num_terms"], r["median_of_medians_seconds"][column]) for r in cpu]

    return [
        Series("neutral", "Qiskit SparsePauliOp.simplify", pick("qiskit"), dashed=True, marker="square"),
        Series("s1", "Wolfgang, scalar", pick("scalar")),
        Series("s5", "Wolfgang, oneTBB 26 threads", pick("tbb")),
    ]


@dataclass
class Comparison:
    key: str
    label: str
    detail: str
    wolfgang: float
    other: float


def competitive_cases() -> list[Comparison]:
    cases = {c["name"]: c for c in load(COMPETITIVE)["competitive_cases"]}

    def scalar(name: str) -> tuple[float, float]:
        result = cases[name]["results"]
        return result["fastpauli_scalar_seconds"], result["competitor_seconds"]

    simplify = scalar("simplify")
    multiply = scalar("multiply")
    grouping = scalar("qiskit_grouping")
    expectation = cases["cuquantum_statevector_expectation"]["results"]
    return [
        Comparison("s1", "vs Qiskit simplify", "64 qubits, 10,000 terms", *simplify),
        Comparison("s1", "vs OpenFermion multiply", "256 x 256 terms", *multiply),
        Comparison("s1", "vs Qiskit group_commuting", "512 terms", *grouping),
        Comparison("s2", "vs cuStateVec expectation", "14 qubits, 4,096 terms, H100",
                   expectation["fastpauli_cuda_device_resident_seconds"], expectation["competitor_seconds"]),
    ]


@dataclass
class Step:
    label: str
    commit: str


def slice_series() -> tuple[list[Step], dict[str, list[Series]]]:
    rows = [
        ("simplify_128_terms_24q", "s1", "simplify, 128 terms"),
        ("simplify_4096_terms_24q", "s2", "simplify, 4,096 terms"),
        ("matmul_128x128_24q", "s3", "matmul, 128 x 128"),
        ("round_trip_simplify_128_terms", "s4", "round trip, 128 terms (to_device, simplify, to_host)"),
    ]
    # (report summary, label) per step; the first step is the earliest baseline.
    cuda = [
        ("cuda_allocation_reuse_h100_2026-11-03/summary_call_overhead.json", "base"),
        ("cuda_allocation_reuse_h100_2026-11-03/summary_call_overhead.json", "cand"),
        ("small_simplify_2026-11-03/summary_cuda_h100.json", "cand"),
        ("cuda_device_counts_h100_2026-11-03/overhead_summary.json", "cand"),
    ]
    hip = [
        ("hip_parity_mi300x_2026-11-03/summary_call_overhead.json", "base"),
        ("hip_parity_mi300x_2026-11-03/summary_call_overhead.json", "s2-cand"),
        ("small_simplify_2026-11-03/summary_hip_mi300x.json", "cand"),
        ("hip_device_counts_mi300x_2026-11-03/overhead_summary.json", "cache"),
    ]
    steps = [
        Step("before", "before"),
        Step("pooled alloc", "HIP parity"),
        Step("fused small", "fused small"),
        Step("device counts", "counts + cache"),
    ]

    def build(sources: list[tuple[str, str]]) -> list[Series]:
        return [
            Series(key, label, [(i, overhead(file, row, run_label)) for i, (file, run_label) in enumerate(sources)])
            for row, key, label in rows
        ]

    return steps, {"CUDA, H100": build(cuda), "HIP, MI300X": build(hip)}


# --- drawing --------------------------------------------------------------


@dataclass
class LogAxis:
    lo: float
    hi: float
    start: float
    end: float

    def map(self, value: float) -> float:
        span = math.log10(self.hi) - math.log10(self.lo)
        return self.start + (self.end - self.start) * (math.log10(value) - math.log10(self.lo)) / span

    def decades(self) -> list[float]:
        return [10.0**e for e in range(math.ceil(math.log10(self.lo) - 1e-9), math.floor(math.log10(self.hi) + 1e-9) + 1)]


@dataclass
class Canvas:
    width: int
    height: int
    title: str
    description: str
    parts: list[str] = field(default_factory=list)

    def add(self, element: str) -> None:
        self.parts.append(element)

    def text(self, x: float, y: float, value: str, cls: str = "t2", size: int = 12, anchor: str = "start",
             weight: str = "400") -> None:
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" font-size="{size}" text-anchor="{anchor}" '
                 f'font-weight="{weight}">{esc(value)}</text>')

    def svg(self) -> str:
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" role="img" aria-labelledby="title desc" '
            f'font-family="{FONT}">\n'
            f'<title id="title">{esc(self.title)}</title>\n<desc id="desc">{esc(self.description)}</desc>\n'
            f'<style>{series_style()}</style>\n'
            f'<rect class="bg" width="{self.width}" height="{self.height}"/>\n'
        )
        return head + "\n".join(self.parts) + "\n</svg>\n"


def marker(x: float, y: float, key: str, shape: str) -> str:
    if shape == "square":
        return f'<rect class="f-{key}" x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" rx="1.5" stroke="none"/>'
    return f'<circle class="f-{key}" cx="{x:.1f}" cy="{y:.1f}" r="4.5" stroke="none"/>'


def draw_line(canvas: Canvas, series: Series, xmap: Callable[[float], float], ymap: Callable[[float], float]) -> None:
    coordinates = [(xmap(x), ymap(y)) for x, y in series.points]
    path = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(coordinates))
    dash = ' stroke-dasharray="6 4"' if series.dashed else ""
    canvas.add(f'<path class="s-{series.key}" d="{path}" fill="none" stroke-width="2" stroke-linejoin="round"{dash}/>')
    for x, y in coordinates:
        canvas.add(marker(x, y, series.key, series.marker))


def place_labels(wanted: list[float], gap: float, top: float, bottom: float) -> list[float]:
    """Spread label baselines at least `gap` apart, keeping their order."""
    order = sorted(range(len(wanted)), key=lambda i: wanted[i])
    placed = [0.0] * len(wanted)
    previous = top - gap
    for i in order:
        placed[i] = max(wanted[i], previous + gap)
        previous = placed[i]
    overflow = previous - bottom
    if overflow > 0:
        for i in order:
            placed[i] -= overflow
    return placed


def log_grid(canvas: Canvas, axis: LogAxis, other_start: float, other_end: float, vertical: bool,
             label: Callable[[float], str]) -> None:
    for value in axis.decades():
        p = axis.map(value)
        if vertical:
            canvas.add(f'<line class="grid" x1="{p:.1f}" y1="{other_start}" x2="{p:.1f}" y2="{other_end}"/>')
            canvas.text(p, other_end + 18, label(value), "t3", 11, "middle")
        else:
            canvas.add(f'<line class="grid" x1="{other_start}" y1="{p:.1f}" x2="{other_end}" y2="{p:.1f}"/>')
            canvas.text(other_start - 8, p + 4, label(value), "t3", 11, "end")


def time_label(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:g} \u00b5s"
    if seconds < 1:
        return f"{seconds * 1e3:g} ms"
    return f"{seconds:g} s"


def end_labels(canvas: Canvas, items: list[tuple[Series, float, float]], x: float, top: float, bottom: float) -> None:
    placed = place_labels([y for _, _, y in items], 15, top, bottom)
    for (series, last_x, last_y), y in zip(items, placed, strict=True):
        canvas.add(f'<line class="s-{series.key}" x1="{last_x + 6:.1f}" y1="{last_y:.1f}" x2="{x - 4:.1f}" '
                   f'y2="{y - 4:.1f}" stroke-width="1" opacity="0.6"/>')
        canvas.text(x, y, series.label, "t1", 12)


def figure_simplify() -> str:
    series = simplify_series()
    canvas = Canvas(
        980, 600, "Simplify time by backend and operator size",
        "Log-log lines of simplify time against term count. "
        + "; ".join(f"{s.label}: " + ", ".join(f"{fmt_terms(x)} terms {fmt_time(y)}" for x, y in s.points) for s in series),
    )
    left, right, top, bottom = 90, 720, 112, 500
    xaxis = LogAxis(100, 5e6, left, right)
    yaxis = LogAxis(3e-6, 0.5, bottom, top)
    canvas.text(24, 38, "Simplify time by backend and operator size", "t1", 20, weight="600")
    canvas.text(24, 62, "Median time per call, 24 qubits, log-log. Accelerator times are device-resident "
                "(operator already on the device).", "t2", 13)
    canvas.text(24, 82, "Where a line sits below CPU scalar, that backend is faster at that size.", "t2", 13)
    log_grid(canvas, xaxis, top, bottom, True, lambda v: fmt_terms(v) + " terms" if v >= 1e3 else f"{v:g}")
    log_grid(canvas, yaxis, left, right, False, time_label)
    canvas.add(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>')
    items = []
    for s in series:
        draw_line(canvas, s, xaxis.map, yaxis.map)
        x, y = s.points[-1]
        items.append((s, xaxis.map(x), yaxis.map(y)))
    end_labels(canvas, items, right + 18, top + 8, bottom)
    canvas.text(24, 540, "128-4,096 terms on CUDA and HIP, and 128 terms on the CPU: distinct terms, one call "
                "(call-overhead suite). Other points: half of the terms are duplicates.", "t3", 11)
    canvas.text(24, 558, "Metal runs its GPU-sort hybrid from 12,288 terms; below that the public path is the "
                "CPU. CPU on a Xeon 8480+ host (the H100's); Metal on an Apple M4 Pro.", "t3", 11)
    canvas.text(24, 576, "Sources: cpu_parallel_simplify, cuda_device_counts, hip_device_counts, "
                "metal_hybrid_simplify and metal_followups reports (docs/benchmarks/).", "t3", 11)
    return canvas.svg()


def figure_libraries() -> str:
    series = qiskit_series()
    cases = competitive_cases()
    canvas = Canvas(
        980, 570, "Wolfgang against other libraries",
        "Left: simplify time against term count for Qiskit and Wolfgang. "
        + "; ".join(f"{s.label}: " + ", ".join(f"{fmt_terms(x)} {fmt_time(y)}" for x, y in s.points) for s in series)
        + ". Right: speedup of Wolfgang over each library. "
        + "; ".join(f"{c.label} ({c.detail}): {c.other / c.wolfgang:.1f}x" for c in cases),
    )
    canvas.text(24, 38, "Wolfgang against other libraries", "t1", 20, weight="600")
    canvas.text(24, 62, "Left: simplify on the same Xeon 8480+ host, 24 qubits, half duplicates (November 2026). "
                "Right: earlier comparisons (April 2026).", "t2", 13)
    # Left panel: lines.
    left, right, top, bottom = 84, 400, 110, 440
    xaxis = LogAxis(1e3, 2e6, left, right)
    yaxis = LogAxis(1e-4, 0.5, bottom, top)
    log_grid(canvas, xaxis, top, bottom, True, lambda v: fmt_terms(v))
    log_grid(canvas, yaxis, left, right, False, time_label)
    canvas.text((left + right) / 2, bottom + 38, "terms", "t3", 11, "middle")
    items = []
    for s in series:
        draw_line(canvas, s, xaxis.map, yaxis.map)
        x, y = s.points[-1]
        items.append((s, xaxis.map(x), yaxis.map(y)))
    legend_y = top - 22
    for row, s in enumerate(series):
        y = legend_y + row * 18 - 18 * (len(series) - 1)
        dash = ' stroke-dasharray="6 4"' if s.dashed else ""
        canvas.add(f'<line class="s-{s.key}" x1="{left + 12}" y1="{y + 18 * (len(series) - 1) + 40 - 4}" '
                   f'x2="{left + 34}" y2="{y + 18 * (len(series) - 1) + 40 - 4}" stroke-width="2"{dash}/>')
        canvas.text(left + 40, y + 18 * (len(series) - 1) + 40, s.label, "t1", 12)
    last = {s.label: s.points[-1][1] for s in series}
    qiskit = last["Qiskit SparsePauliOp.simplify"]
    canvas.text(left, bottom + 58, f"At 1M terms Wolfgang is {qiskit / last['Wolfgang, oneTBB 26 threads']:.1f}x "
                f"faster than Qiskit with oneTBB and {qiskit / last['Wolfgang, scalar']:.2f}x with the scalar kernel.",
                "t2", 11)
    # Right panel: horizontal bars of speedup, log scale.
    bar_left, bar_right = 470, 880
    axis = LogAxis(1, 500, bar_left, bar_right)
    row_top = 146
    for value in (1, 10, 100):
        p = axis.map(value)
        canvas.add(f'<line class="grid" x1="{p:.1f}" y1="{row_top - 14}" x2="{p:.1f}" y2="{row_top + 4 * 74 - 20}"/>')
        canvas.text(p, row_top + 4 * 74 - 4, f"{value:g}x", "t3", 11, "middle")
    canvas.text(bar_left, row_top - 46, "Speedup of Wolfgang over each library (log scale)", "t2", 12)
    for i, case in enumerate(cases):
        y = row_top + i * 74
        ratio = case.other / case.wolfgang
        canvas.text(bar_left, y - 16, case.label, "t1", 12, weight="600")
        canvas.text(bar_left + 7.2 * len(case.label) + 8, y - 16, case.detail, "t3", 11)
        width = axis.map(ratio) - bar_left
        canvas.add(f'<rect class="f-{case.key}" x="{bar_left}" y="{y - 6}" width="{width:.1f}" height="16" rx="4" stroke="none"/>')
        ratio_text = f"{ratio:.0f}x" if ratio >= 100 else f"{ratio:.1f}x" if ratio >= 10 else f"{ratio:.2f}x"
        canvas.text(bar_left + width + 6, y + 7, ratio_text, "t1", 12, weight="600")
        canvas.text(bar_left + width + 6 + 8 * len(ratio_text), y + 7,
                    f"({fmt_time(case.wolfgang)} vs {fmt_time(case.other)})", "t3", 11)
    canvas.text(24, 520, "April comparisons: Wolfgang scalar CPU, except cuStateVec (Wolfgang CUDA, device-resident), "
                "on an H100 PCIe host with Qiskit 2.4.1, OpenFermion 1.7.1, cuQuantum 24.8.", "t3", 11)
    canvas.text(24, 538, "November simplify: Qiskit 1.4.5. Sources: cpu_parallel_simplify_2026-11-03 and "
                "cuda_deep_optimization_h100_2026-04-28 (docs/benchmarks/).", "t3", 11)
    return canvas.svg()


def figure_slices() -> str:
    steps, panels = slice_series()
    canvas = Canvas(
        980, 600, "Per-call times across the campaign slices",
        "Two panels, CUDA on an H100 and HIP on an MI300X; lines of per-call time across four steps. "
        + "; ".join(f"{panel}: " + "; ".join(f"{s.label} " + " -> ".join(fmt_time(y) for _, y in s.points)
                                              for s in series) for panel, series in panels.items()),
    )
    canvas.text(24, 38, "Per-call times across the campaign slices", "t1", 20, weight="600")
    canvas.text(24, 62, "Median time of one call on small device-resident operators (24 qubits), after each slice "
                "that changed it. Log scale.", "t2", 13)
    panel_width, gap = 410, 70
    top, bottom = 110, 430
    yaxis = LogAxis(1e-5, 2e-3, bottom, top)
    for p, (panel, series) in enumerate(panels.items()):
        left = 90 + p * (panel_width + gap)
        right = left + panel_width - 170
        canvas.text(left, top - 18, panel, "t1", 14, weight="600")
        log_grid(canvas, yaxis, left, right, False, time_label)
        xs = [left + (right - left) * i / (len(steps) - 1) for i in range(len(steps))]
        for i, x in enumerate(xs):
            canvas.add(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}"/>')
            name = steps[i].label if p == 0 else steps[i].commit
            canvas.text(x, bottom + 18, f"{i + 1}", "t3", 11, "middle")
            canvas.text(x, bottom + 32, name, "t3", 10, "middle")
        items = []
        for s in series:
            draw_line(canvas, s, lambda i, xs=xs: xs[int(i)], yaxis.map)
            items.append((s, xs[-1], yaxis.map(s.points[-1][1])))
        placed = place_labels([y for _, _, y in items], 15, top + 8, bottom)
        short = {"s1": "simplify 128", "s2": "simplify 4,096", "s3": "matmul", "s4": "round trip"}
        for (s, _, last_y), y in zip(items, placed, strict=True):
            canvas.text(right + 12, y, f"{short[s.key]}: {fmt_time(s.points[0][1])} \u2192 "
                        f"{fmt_time(s.points[-1][1])}", "t1", 11)
    legend_y = 490
    legend_x = 90
    for _, key, label in [(None, s.key, s.label) for s in next(iter(panels.values()))]:
        canvas.add(f'<rect class="f-{key}" x="{legend_x}" y="{legend_y - 10}" width="12" height="12" rx="3" stroke="none"/>')
        canvas.text(legend_x + 18, legend_y, label, "t1", 12)
        legend_x += 18 + 6.4 * len(label) + 22
    canvas.text(90, 520, "Steps: 1 before the usability campaign's allocation work; 2 pooled CUDA allocation / HIP "
                "parity (first HIP campaign slice); 3 fused small-operator simplify;", "t3", 11)
    canvas.text(90, 538, "4 device-side simplify counts (HIP: with the per-thread scratch cache). Each step's value is "
                "the candidate of a paired run against the step before.", "t3", 11)
    canvas.text(90, 556, "Sources: cuda_allocation_reuse, hip_parity, small_simplify, cuda_device_counts and "
                "hip_device_counts reports (docs/benchmarks/).", "t3", 11)
    return canvas.svg()


FIGURES: dict[str, Callable[[], str]] = {
    "simplify_by_backend.svg": figure_simplify,
    "wolfgang_vs_other_libraries.svg": figure_libraries,
    "call_time_by_slice.svg": figure_slices,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if a committed figure differs from the data")
    args = parser.parse_args()
    stale = []
    for name, render in FIGURES.items():
        text = render()
        path = PLOTS / name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(name)
        else:
            path.write_text(text, encoding="utf-8")
    if stale:
        raise SystemExit("stale figures (rerun scripts/render_readme_figures.py): " + ", ".join(stale))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

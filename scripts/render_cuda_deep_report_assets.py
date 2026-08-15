#!/usr/bin/env python3
"""Render checked-in assets for the CUDA deep optimization report."""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RAW_FILENAMES = {
    "final_smoke": "cuda_scaling_smoke_final.json",
    "final_default": "cuda_scaling_default_final.json",
    "final_stress": "cuda_scaling_stress_final.json",
    "final_extreme": "cuda_scaling_extreme_final.json",
    "competitive": "competitive_baselines_final.json",
    "expectation_baseline_default": "expectation_baseline_default.json",
    "expectation_baseline_stress": "expectation_baseline_stress.json",
    "expectation_baseline_extreme": "expectation_baseline_extreme.json",
    "expectation_bytecopy_default": "expectation_bytecopy_default.json",
    "expectation_bytecopy_stress": "expectation_bytecopy_stress.json",
    "expectation_bytecopy_extreme": "expectation_bytecopy_extreme.json",
    "comm_baseline_default": "comm_baseline_default.json",
    "comm_baseline_stress": "comm_baseline_stress.json",
    "comm_baseline_extreme": "comm_baseline_extreme.json",
    "comm_specialized_default": "comm_specialized_default.json",
    "comm_specialized_stress": "comm_specialized_stress.json",
    "comm_specialized_extreme": "comm_specialized_extreme.json",
    "ncu": "ncu_selected_metrics.json",
}

EXPERIMENT_PROVENANCE = {
    "baseline_commit": "aeeebbaa2d3d33b7d414974075911af56e16451a",
    "retained_bytecopy_patch_id": "d8da88a96579f20118cdb2bbc955a9c6099e942c",
    "rejected_commutation_specialization_patch_id": "ebaa400b2fd184cae3ffbeb4cae7111e28061071",
    "status": (
        "raw benchmark JSON was captured from H100 experiment clones with uncommitted "
        "patches against the baseline commit; patch-id values identify the exact A/B diffs"
    ),
}

COLORS = {
    "gray": "#4b5563",
    "blue": "#2563eb",
    "green": "#059669",
    "teal": "#0f766e",
    "purple": "#7c3aed",
    "amber": "#b45309",
    "red": "#dc2626",
    "ink": "#111827",
    "muted": "#6b7280",
    "grid": "#d1d5db",
    "panel": "#f9fafb",
}


@dataclass(frozen=True)
class Timing:
    label: str
    seconds: float
    color: str


def load_raw(raw_dir: Path) -> dict[str, Any]:
    data = {}
    missing = []
    for key, filename in RAW_FILENAMES.items():
        path = raw_dir / filename
        if not path.exists():
            missing.append(str(path))
            continue
        data[key] = json.loads(path.read_text(encoding="utf-8"))
    if missing:
        raise FileNotFoundError("missing CUDA deep raw artifact(s): " + ", ".join(missing))
    return data


def metric(results: dict[str, Any], key: str) -> float | None:
    value = results.get(key)
    return None if value is None else float(value)


def format_seconds(value: float) -> str:
    if value < 1.0e-4:
        return f"{value:.2e}s"
    if value < 0.01:
        return f"{value * 1000.0:.2f}ms"
    if value < 1.0:
        return f"{value:.3f}s"
    return f"{value:.2f}s"


def format_ratio(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}x"
    if value >= 10:
        return f"{value:.1f}x"
    return f"{value:.2f}x"


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
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" '
        f'text-anchor="{anchor}">{html.escape(text)}</text>'
    )


def svg_rect(x: float, y: float, width: float, height: float, fill: str, *, rx: float = 2.0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(width, 0.0):.1f}" '
        f'height="{height:.1f}" rx="{rx:.1f}" fill="{fill}"/>'
    )


def svg_header(width: int, height: int, title: str, desc: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{html.escape(title)}</title>',
        f'<desc id="desc">{html.escape(desc)}</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def find_case(report: dict[str, Any], name: str, scale: str) -> dict[str, Any]:
    for case in report["cases"]:
        if case["name"] == name and case["scale"] == scale:
            return case
    raise KeyError(f"case not found: {name} {scale}")


def case_key(case: dict[str, Any]) -> tuple[str, str]:
    return (str(case["name"]), str(case["scale"]))


def paired_cases(left_report: dict[str, Any], right_report: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    left_cases = {case_key(case): case for case in left_report["cases"]}
    right_cases = {case_key(case): case for case in right_report["cases"]}
    if set(left_cases) != set(right_cases):
        missing_left = sorted(set(right_cases) - set(left_cases))
        missing_right = sorted(set(left_cases) - set(right_cases))
        raise ValueError(
            "A/B case mismatch: "
            f"missing from left={missing_left}; missing from right={missing_right}"
        )
    return [(left_cases[key], right_cases[key]) for key in sorted(left_cases)]


def representative_default_cases(default_report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        find_case(default_report, "simplify_duplicate_pressure", "terms_50000"),
        find_case(default_report, "statevector_expectation", "qubits_12_terms_2048"),
        find_case(default_report, "pairwise_commutation", "terms_2048x2048"),
        find_case(default_report, "matmul_product_generation_simplify", "terms_256x256"),
    ]


def timings_for_case(case: dict[str, Any]) -> list[Timing]:
    results = case["results"]
    timings = [Timing("CPU scalar", float(results["cpu_scalar_seconds"]), COLORS["gray"])]
    optimized = results.get("cpu_optimized_timings", {})
    for backend in ("tbb", "avx512", "avx2", "neon", "sve"):
        if backend in optimized:
            timings.append(
                Timing(f"CPU {backend}", float(optimized[backend]["seconds"]), COLORS["purple"])
            )
    timings.append(
        Timing("CUDA transfer", float(results["cuda_transfer_inclusive_seconds"]), COLORS["blue"])
    )
    timings.append(
        Timing("CUDA resident", float(results["cuda_device_resident_seconds"]), COLORS["green"])
    )
    preallocated = metric(results, "cuda_device_resident_preallocated_seconds")
    if preallocated is not None:
        timings.append(Timing("CUDA preallocated", preallocated, COLORS["teal"]))
    return timings


def render_path_speedups(default_report: dict[str, Any], output: Path) -> None:
    cases = representative_default_cases(default_report)
    width = 1120
    row_height = 24
    case_gap = 40
    left = 315
    max_bar = 540
    top = 116
    height = top + sum(54 + row_height * len(timings_for_case(case)) + case_gap for case in cases)
    max_speedup = max(
        float(case["results"]["cpu_scalar_seconds"]) / timing.seconds
        for case in cases
        for timing in timings_for_case(case)
    )
    lines = svg_header(
        width,
        height,
        "Wolfgang H100 backend speedups",
        "Representative default-profile speedups versus scalar CPU for all captured Wolfgang paths.",
    )
    lines += [
        svg_text(32, 38, "H100 Backend Speedups", size=25, weight=700),
        svg_text(
            32,
            62,
            "Representative default-profile cases; bars are log-scaled speedup versus CPU scalar.",
            size=13,
            color=COLORS["muted"],
        ),
        f'<line x1="{left}" y1="88" x2="{left + max_bar}" y2="88" stroke="{COLORS["grid"]}"/>',
        svg_text(left, 82, "1x", size=11, color=COLORS["muted"]),
        svg_text(left + max_bar, 82, format_ratio(max_speedup), size=11, color=COLORS["muted"], anchor="end"),
    ]

    y = top
    denominator = math.log10(max(max_speedup, 1.01))
    for case in cases:
        case_title = case["name"].replace("_", " ")
        scale = case["scale"]
        cpu = float(case["results"]["cpu_scalar_seconds"])
        lines.append(svg_text(32, y, case_title, size=15, weight=700))
        lines.append(svg_text(32, y + 18, scale, size=12, color=COLORS["muted"]))
        bar_y = y + 32
        for timing in timings_for_case(case):
            speedup = cpu / timing.seconds
            bar_width = max(2.0, max_bar * (math.log10(max(speedup, 1.0)) / denominator))
            lines.append(svg_rect(left, bar_y - 13, bar_width, 16, timing.color))
            label = f"{timing.label} {format_ratio(speedup)} ({format_seconds(timing.seconds)})"
            lines.append(svg_text(left + bar_width + 8, bar_y, label, size=12))
            bar_y += row_height
        y = bar_y + case_gap

    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def scale_measure(case: dict[str, Any]) -> float:
    dataset = case["dataset"]
    if "entries" in dataset:
        return float(dataset["entries"])
    if "statevector_length" in dataset:
        return float(dataset["statevector_length"])
    if "intermediate_terms" in dataset:
        return float(dataset["intermediate_terms"])
    return float(dataset.get("num_terms", dataset.get("lhs_terms", 1)))


def render_scaling(final_reports: list[dict[str, Any]], output: Path) -> None:
    operations = [
        "simplify_duplicate_pressure",
        "statevector_expectation",
        "pairwise_commutation",
        "matmul_product_generation_simplify",
    ]
    panels: dict[str, list[tuple[float, float, str]]] = {operation: [] for operation in operations}
    for report in final_reports:
        for case in report["cases"]:
            operation = case["name"]
            if operation not in panels:
                continue
            cpu = float(case["results"]["cpu_scalar_seconds"])
            cuda = float(case["results"]["cuda_device_resident_seconds"])
            panels[operation].append((scale_measure(case), cpu / cuda, case["scale"]))

    width = 1120
    height = 780
    lines = svg_header(
        width,
        height,
        "Wolfgang H100 scaling",
        "CUDA device-resident speedup versus scalar CPU across default, stress, and extreme scale points.",
    )
    lines += [
        svg_text(32, 38, "H100 Scaling Across CUDA Hot Paths", size=25, weight=700),
        svg_text(
            32,
            62,
            "Device-resident CUDA speedup versus scalar CPU; each panel uses log x/y scales.",
            size=13,
            color=COLORS["muted"],
        ),
    ]
    panel_w = 500
    panel_h = 265
    panel_positions = [(32, 100), (588, 100), (32, 430), (588, 430)]
    for operation, (x0, y0) in zip(operations, panel_positions, strict=True):
        points = sorted(panels[operation])
        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = max(1.0, min(point[1] for point in points) * 0.75)
        max_y = max(point[1] for point in points) * 1.25
        log_x0 = math.log10(min_x)
        log_x1 = math.log10(max_x)
        log_y0 = math.log10(min_y)
        log_y1 = math.log10(max_y)

        lines.append(svg_rect(x0, y0, panel_w, panel_h, COLORS["panel"], rx=6))
        lines.append(svg_text(x0 + 18, y0 + 28, operation.replace("_", " "), size=14, weight=700))
        plot_x = x0 + 58
        plot_y = y0 + 50
        plot_w = panel_w - 92
        plot_h = panel_h - 92
        lines.append(svg_rect(plot_x, plot_y, plot_w, plot_h, "#ffffff", rx=2))
        lines.append(f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="{COLORS["grid"]}"/>')
        lines.append(f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="{COLORS["grid"]}"/>')
        prev: tuple[float, float] | None = None
        for x_value, speedup, label in points:
            x = plot_x + plot_w * (math.log10(x_value) - log_x0) / max(log_x1 - log_x0, 1.0e-12)
            y = plot_y + plot_h * (1.0 - (math.log10(speedup) - log_y0) / max(log_y1 - log_y0, 1.0e-12))
            if prev is not None:
                lines.append(
                    f'<line x1="{prev[0]:.1f}" y1="{prev[1]:.1f}" x2="{x:.1f}" y2="{y:.1f}" '
                    f'stroke="{COLORS["green"]}" stroke-width="2"/>'
                )
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{COLORS["green"]}"/>')
            lines.append(svg_text(x + 6, y - 6, format_ratio(speedup), size=10, color=COLORS["ink"]))
            prev = (x, y)
        lines.append(svg_text(plot_x, plot_y + plot_h + 28, "problem size (log)", size=11, color=COLORS["muted"]))
        lines.append(svg_text(plot_x - 10, plot_y - 8, "speedup", size=11, color=COLORS["muted"], anchor="end"))
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_optimization_deltas(raw: dict[str, Any], output: Path) -> None:
    expectation_rows = []
    for profile_key, profile_name in [
        ("expectation_bytecopy_default", "default"),
        ("expectation_bytecopy_stress", "stress"),
        ("expectation_bytecopy_extreme", "extreme"),
    ]:
        for base_case, exp_case in paired_cases(
            raw[f"expectation_baseline_{profile_name}"],
            raw[profile_key],
        ):
            ratio = float(base_case["results"]["cuda_device_resident_seconds"]) / float(
                exp_case["results"]["cuda_device_resident_seconds"]
            )
            expectation_rows.append((f"expectation {profile_name} {exp_case['scale']}", ratio, True))

    comm_rows = []
    for profile_name in ("default", "stress", "extreme"):
        base_report = raw[f"comm_baseline_{profile_name}"]
        spec_report = raw[f"comm_specialized_{profile_name}"]
        for base_case, spec_case in paired_cases(base_report, spec_report):
            ratio = float(base_case["results"]["cuda_device_resident_preallocated_seconds"]) / float(
                spec_case["results"]["cuda_device_resident_preallocated_seconds"]
            )
            comm_rows.append((f"comm specialization {profile_name} {spec_case['scale']}", ratio, False))

    rows = expectation_rows[-8:] + comm_rows[-8:]
    render_optimization_delta_rows(rows, output)


def render_optimization_delta_rows(rows: list[tuple[str, float, bool]], output: Path) -> None:
    width = 1120
    height = 550
    left = 420
    axis = left + 210
    max_right = 360
    row_h = 25
    lines = svg_header(
        width,
        height,
        "Wolfgang CUDA optimization deltas",
        "A/B ratios for the retained statevector byte-copy optimization and rejected commutation specialization.",
    )
    lines += [
        svg_text(32, 38, "A/B Optimization Deltas", size=25, weight=700),
        svg_text(
            32,
            62,
            "Ratio is baseline seconds divided by experiment seconds; above 1.0x is faster.",
            size=13,
            color=COLORS["muted"],
        ),
        f'<line x1="{axis}" y1="90" x2="{axis}" y2="{height - 44}" stroke="{COLORS["grid"]}"/>',
        svg_text(axis, 84, "1.0x", size=11, color=COLORS["muted"], anchor="middle"),
    ]
    y = 110
    for label, ratio, retained in rows:
        color = COLORS["green"] if retained and ratio >= 1.0 else COLORS["red"] if ratio < 1.0 else COLORS["amber"]
        lines.append(svg_text(32, y + 12, label, size=11, color=COLORS["ink"]))
        if ratio >= 1.0:
            width_px = min(max_right, max_right * math.log10(ratio + 0.05) / math.log10(1.35))
            lines.append(svg_rect(axis, y, width_px, 16, color))
            text_x = axis + width_px + 8
            anchor = "start"
        else:
            width_px = min(190, 190 * math.log10((1.0 / ratio) + 0.05) / math.log10(4.5))
            lines.append(svg_rect(axis - width_px, y, width_px, 16, color))
            text_x = axis - width_px - 8
            anchor = "end"
        lines.append(svg_text(text_x, y + 12, format_ratio(ratio), size=11, color=COLORS["ink"], anchor=anchor))
        y += row_h
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_optimization_deltas_from_summary(summary: dict[str, Any], output: Path) -> None:
    rows: list[tuple[str, float, bool]] = []
    for item in summary.get("retained_experiments", [])[-8:]:
        rows.append(
            (
                f"expectation {item['profile']} {item['scale']}",
                float(item["device_resident_ratio"]),
                True,
            )
        )
    for item in summary.get("rejected_experiments", [])[-8:]:
        rows.append(
            (
                f"comm specialization {item['profile']} {item['scale']}",
                float(item["preallocated_ratio"]),
                False,
            )
        )
    render_optimization_delta_rows(rows, output)


def first_metric_row(ncu: dict[str, Any], key: str, contains: str | None = None) -> dict[str, str]:
    for row in ncu[key]:
        if contains is None or contains in row["Kernel Name"]:
            return row
    raise KeyError(f"no NCU metric row for {key}")


def render_profiler_bottlenecks(ncu: dict[str, Any], output: Path) -> None:
    rows = [
        ("commutation default", first_metric_row(ncu, "pairwise_default")),
        ("commutation stress", first_metric_row(ncu, "pairwise_stress")),
        ("expectation stress", first_metric_row(ncu, "statevector_stress")),
        ("matmul product stress", first_metric_row(ncu, "matmul_stress", "matmul_product_kernel")),
        ("simplify sort stress", first_metric_row(ncu, "simplify_stress", "DeviceMergeSortMergeKernel")),
    ]
    width = 1120
    height = 430
    lines = svg_header(
        width,
        height,
        "Wolfgang H100 Nsight Compute bottlenecks",
        "Selected Nsight Compute throughput and occupancy metrics for custom and CCCL kernels.",
    )
    lines += [
        svg_text(32, 38, "Nsight Compute Bottleneck Signals", size=25, weight=700),
        svg_text(
            32,
            62,
            "Selected detailed-counter samples. Low DRAM percent with high SM percent points away from raw HBM bandwidth.",
            size=13,
            color=COLORS["muted"],
        ),
    ]
    y = 105
    left = 300
    bar_w = 520
    for label, row in rows:
        sm = float(row["sm__throughput.avg.pct_of_peak_sustained_elapsed"])
        mem = float(row["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"])
        occ = float(row["sm__warps_active.avg.pct_of_peak_sustained_active"])
        duration = float(row["gpu__time_duration.sum"])
        lines.append(svg_text(32, y + 16, label, size=13, weight=700))
        lines.append(svg_rect(left, y, bar_w * sm / 100.0, 12, COLORS["green"]))
        lines.append(svg_text(left + bar_w + 12, y + 10, f"SM {sm:.1f}%", size=11))
        lines.append(svg_rect(left, y + 18, bar_w * mem / 100.0, 12, COLORS["blue"]))
        lines.append(svg_text(left + bar_w + 12, y + 28, f"Memory {mem:.1f}%", size=11))
        lines.append(svg_rect(left, y + 36, bar_w * occ / 100.0, 12, COLORS["purple"]))
        lines.append(svg_text(left + bar_w + 12, y + 46, f"Active warps {occ:.1f}%; {duration:.1f}us", size=11))
        y += 62
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_architecture(output: Path) -> None:
    width = 1180
    height = 760
    lines = svg_header(
        width,
        height,
        "Wolfgang CPU CUDA H100 architecture",
        "Software and hardware boundary diagram for CPU selectors, PCIe movement, H100 memory, and SM90 kernels.",
    )
    lines += [
        svg_text(32, 38, "Wolfgang Execution And Hardware Architecture", size=25, weight=700),
        svg_text(
            32,
            62,
            "Host CPU selectors, PCIe transfers, H100 HBM3-resident buffers, and SM90 kernel execution boundaries.",
            size=13,
            color=COLORS["muted"],
        ),
    ]

    def labeled_box(x: int, y: int, w: int, h: int, title: str, lines_text: list[str], fill: str) -> None:
        lines.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
            f'fill="{fill}" stroke="{COLORS["grid"]}"/>'
        )
        lines.append(svg_text(x + 14, y + 26, title, size=14, weight=700))
        for index, body in enumerate(lines_text):
            lines.append(svg_text(x + 14, y + 50 + index * 17, body, size=11, color=COLORS["muted"]))

    labeled_box(44, 116, 205, 104, "Python API", ["PauliSum", "DevicePauliSum", "NumPy / CuPy inputs"], COLORS["panel"])
    labeled_box(300, 116, 225, 104, "C++ Packed Core", ["canonical X/Z uint64 words", "complex128 coefficients", "guardrails and semantics"], "#eef2ff")
    labeled_box(590, 96, 220, 124, "CPU Execution", ["scalar correctness path", "oneTBB thread parallelism", "AVX2 / AVX-512 / NEON selectors"], "#f5f3ff")
    labeled_box(590, 292, 220, 124, "CUDA Mirror", ["device X/Z word buffers", "device coefficient buffer", "optional device statevector"], "#ecfdf5")
    labeled_box(46, 290, 205, 124, "Host Memory", ["NumPy bool matrices", "host statevector copy path", "preallocated output buffers"], "#fff7ed")
    labeled_box(870, 118, 244, 92, "PCIe Boundary", ["operator transfers", "statevector transfers", "host result materialization"], "#f8fafc")
    labeled_box(868, 276, 250, 156, "H100 Device", ["HBM3 global memory", "SM90 resident blocks", "L2 and shared-memory reductions", "CCCL/CUB sort and reduce"], "#eff6ff")
    labeled_box(870, 492, 244, 112, "External Baselines", ["Qiskit / OpenFermion CPU", "cuStateVec Pauli expectations", "CUDA-Q / Aer as workflow probes"], "#f8fafc")

    h100_x = 896
    h100_y = 346
    for row in range(2):
        for col in range(4):
            x = h100_x + col * 48
            y = h100_y + row * 35
            lines.append(svg_rect(x, y, 34, 22, "#dbeafe", rx=3))
            lines.append(svg_text(x + 17, y + 15, "SM", size=9, anchor="middle", color=COLORS["blue"]))
    lines.append(svg_text(896, 424, "SM90 grid denotes many resident SMs; diagram is schematic.", size=10, color=COLORS["muted"]))

    arrows = [
        (249, 168, 300, 168),
        (525, 146, 590, 144),
        (525, 188, 590, 348),
        (810, 354, 868, 354),
        (994, 276, 994, 210),
        (868, 382, 810, 382),
        (590, 382, 251, 352),
        (994, 432, 994, 492),
    ]
    for x1, y1, x2, y2 in arrows:
        lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{COLORS["ink"]}" stroke-width="1.8" marker-end="url(#arrow)"/>')
    lines.insert(
        4,
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#111827"/></marker></defs>',
    )
    lines.append(
        svg_text(
            44,
            690,
            "Interpretation: Wolfgang keeps portable CPU wheels by default; CUDA source builds move packed operands to HBM3, execute custom/CCCL kernels on SM90, and report transfer, resident, and preallocated boundaries separately.",
            size=13,
            color=COLORS["muted"],
        )
    )
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_kernel_flows(output: Path) -> None:
    width = 1180
    height = 760
    lines = svg_header(
        width,
        height,
        "Wolfgang CUDA kernel flows",
        "Dataflow diagrams for simplify, expectation, commutation, and matmul plus simplify kernels.",
    )
    lines += [
        svg_text(32, 38, "CUDA Kernel And Algorithm Flows", size=25, weight=700),
        svg_text(
            32,
            62,
            "Dominant device stages, host/device boundaries, and profiler-driven bottleneck interpretation.",
            size=13,
            color=COLORS["muted"],
        ),
    ]
    lanes = [
        (
            "Simplify",
            ["packed terms", "canonical key transform", "CCCL merge sort", "reduce_by_key coefficients", "compact survivors"],
            "temporary storage and CCCL launch overhead dominate after custom key generation",
            COLORS["green"],
        ),
        (
            "Expectation",
            ["host/CuPy psi boundary", "device complex view", "per-term bit phase", "block shared reduction", "device final reduce"],
            "byte-copy removes host conversion; true resident path uses CUDA array interface",
            COLORS["blue"],
        ),
        (
            "Commutation",
            ["lhs/rhs XZ buffers", "2D entry mapping", "word parity + popcount", "device bool matrix", "host/prealloc materialize"],
            "large cases are output-materialization sensitive, not only instruction-stream sensitive",
            COLORS["purple"],
        ),
        (
            "Matmul+simplify",
            ["lhs x rhs product grid", "XOR X/Z words", "phase exponent", "intermediate terms", "simplify pipeline"],
            "product generation feeds the same CCCL sort/reduce pressure as simplify",
            COLORS["amber"],
        ),
    ]
    y = 118
    for title, stages, note, color in lanes:
        lines.append(svg_text(42, y + 35, title, size=15, weight=700))
        x = 205
        for index, stage in enumerate(stages):
            lines.append(svg_rect(x, y, 150, 52, "#ffffff", rx=6))
            lines.append(f'<rect x="{x}" y="{y}" width="150" height="52" rx="6" fill="none" stroke="{color}" stroke-width="1.7"/>')
            words = stage.split(" ")
            if len(stage) > 18 and len(words) > 2:
                midpoint = len(words) // 2
                lines.append(svg_text(x + 75, y + 23, " ".join(words[:midpoint]), size=10, anchor="middle"))
                lines.append(svg_text(x + 75, y + 38, " ".join(words[midpoint:]), size=10, anchor="middle"))
            else:
                lines.append(svg_text(x + 75, y + 31, stage, size=10, anchor="middle"))
            if index < len(stages) - 1:
                lines.append(f'<line x1="{x + 150}" y1="{y + 26}" x2="{x + 177}" y2="{y + 26}" stroke="{COLORS["ink"]}" stroke-width="1.6" marker-end="url(#arrow)"/>')
            x += 177
        lines.append(svg_text(205, y + 78, note, size=11, color=COLORS["muted"]))
        y += 142
    lines.insert(
        4,
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#111827"/></marker></defs>',
    )
    lines.append(
        svg_text(
            42,
            708,
            "Profiler interpretation: small kernel edits are no longer the first lever; remaining headroom is reusable workspaces, device-resident lifetimes, and explicit stream boundaries.",
            size=13,
            color=COLORS["muted"],
        )
    )
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(raw: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "source": "scripts/render_cuda_deep_report_assets.py",
        "raw_artifacts": RAW_FILENAMES,
        "experiment_provenance": EXPERIMENT_PROVENANCE,
        "final_scaling": {},
        "retained_experiments": [],
        "rejected_experiments": [],
        "competitors": raw["competitive"]["competitors"],
    }
    for profile in ("smoke", "default", "stress", "extreme"):
        report = raw[f"final_{profile}"]
        summary["final_scaling"][profile] = {
            "git_commit": report.get("git_commit"),
            "timing_policy": report.get("timing_policy"),
            "cases": [
                {
                    "name": case["name"],
                    "scale": case["scale"],
                    "dataset": case["dataset"],
                    "results": case["results"],
                }
                for case in report["cases"]
            ],
        }
    for profile in ("default", "stress", "extreme"):
        for base_case, exp_case in paired_cases(
            raw[f"expectation_baseline_{profile}"],
            raw[f"expectation_bytecopy_{profile}"],
        ):
            ratio = float(base_case["results"]["cuda_device_resident_seconds"]) / float(
                exp_case["results"]["cuda_device_resident_seconds"]
            )
            summary["retained_experiments"].append(
                {
                    "name": "statevector host byte-copy",
                    "profile": profile,
                    "scale": exp_case["scale"],
                    "device_resident_ratio": ratio,
                    "transfer_inclusive_ratio": float(
                        base_case["results"]["cuda_transfer_inclusive_seconds"]
                    )
                    / float(exp_case["results"]["cuda_transfer_inclusive_seconds"]),
                }
            )
        for base_case, spec_case in paired_cases(
            raw[f"comm_baseline_{profile}"],
            raw[f"comm_specialized_{profile}"],
        ):
            ratio = float(base_case["results"]["cuda_device_resident_preallocated_seconds"]) / float(
                spec_case["results"]["cuda_device_resident_preallocated_seconds"]
            )
            summary["rejected_experiments"].append(
                {
                    "name": "one/two-word 2D commutation grid specialization",
                    "profile": profile,
                    "scale": spec_case["scale"],
                    "preallocated_ratio": ratio,
                }
            )
    summary["ncu_selected_metrics"] = raw["ncu"]
    summary["competitive_cases"] = raw["competitive"]["cases"]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path, required=True)
    args = parser.parse_args()

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    if args.raw_dir.exists():
        raw = load_raw(args.raw_dir)
        default_report = raw["final_default"]
        scaling_reports = [raw["final_default"], raw["final_stress"], raw["final_extreme"]]
        render_optimization_deltas(
            raw,
            args.plot_dir / "cuda_deep_optimization_h100_optimization_deltas.svg",
        )
        ncu_metrics = raw["ncu"]
        summary = build_summary(raw)
        args.summary_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        summary = json.loads(args.summary_output.read_text(encoding="utf-8"))
        default_report = summary["final_scaling"]["default"]
        scaling_reports = [
            summary["final_scaling"]["default"],
            summary["final_scaling"]["stress"],
            summary["final_scaling"]["extreme"],
        ]
        render_optimization_deltas_from_summary(
            summary,
            args.plot_dir / "cuda_deep_optimization_h100_optimization_deltas.svg",
        )
        ncu_metrics = summary["ncu_selected_metrics"]

    render_path_speedups(
        default_report,
        args.plot_dir / "cuda_deep_optimization_h100_path_speedups.svg",
    )
    render_scaling(
        scaling_reports,
        args.plot_dir / "cuda_deep_optimization_h100_scaling.svg",
    )
    render_profiler_bottlenecks(
        ncu_metrics,
        args.plot_dir / "cuda_deep_optimization_h100_profiler_bottlenecks.svg",
    )
    render_architecture(args.plot_dir / "cuda_deep_optimization_architecture.svg")
    render_kernel_flows(args.plot_dir / "cuda_deep_optimization_kernel_flows.svg")


if __name__ == "__main__":
    main()

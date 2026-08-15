#!/usr/bin/env python3
"""Render Campaign 4 H100 CUDA optimization summary and SVG assets."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

RAW_FILES = {
    "baseline_stress": "baseline_cuda_scaling_stress.json",
    "baseline_extreme": "baseline_cuda_scaling_extreme.json",
    "baseline_materialization": "baseline_cuda_scaling_materialization.json",
    "workspace_default": "experiment_campaign4_workspace_default.json",
    "cub_reduce_absent": "experiment_campaign4_simplify_cub_radix_sort_reduce_absent.json",
    "cub_reduce_prereserved": "experiment_campaign4_simplify_cub_radix_sort_reduce_pre_reserved_outside_timing.json",
    "cub_rle_absent": "experiment_campaign4_simplify_cub_radix_sort_run_length_absent.json",
    "cub_rle_prereserved": "experiment_campaign4_simplify_cub_radix_sort_run_length_pre_reserved_outside_timing.json",
    "commutation_materialization": "experiment_commutation_materialization.json",
    "statevector_ncu_input": "experiment_statevector_ncu_input.json",
    "statevector_ncu_privileged_input": "experiment_statevector_ncu_privileged_input.json",
    "nsys_input": "experiment_nsys_workspace_input.json",
    "default_cross": "experiment_cuda_scaling_default.json",
    "competitive": "competitive_baselines_final.json",
}

PLOTS = {
    "workspace": "cuda_h100_campaign4_workspace_boundaries.svg",
    "duplicate": "cuda_h100_campaign4_duplicate_reduction.svg",
    "commutation": "cuda_h100_campaign4_commutation_materialization.svg",
    "cross": "cuda_h100_campaign4_cross_comparison.svg",
    "landscape": "cuda_h100_campaign4_performance_landscape.svg",
    "status": "cuda_h100_campaign4_evidence_status.svg",
}

COLORS = {
    "ink": "#111827",
    "muted": "#6b7280",
    "grid": "#d1d5db",
    "blue": "#2563eb",
    "green": "#059669",
    "teal": "#0f766e",
    "purple": "#7c3aed",
    "amber": "#b45309",
    "red": "#dc2626",
    "gray": "#4b5563",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_raw(raw_dir: Path) -> dict[str, dict[str, Any]]:
    missing = [filename for filename in RAW_FILES.values() if not (raw_dir / filename).exists()]
    if missing:
        raise FileNotFoundError("missing Campaign 4 raw artifact(s): " + ", ".join(missing))
    return {key: load_json(raw_dir / filename) for key, filename in RAW_FILES.items()}


def text(x: float, y: float, value: str, *, size: int = 12, weight: int = 400, color: str = COLORS["ink"], anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">'
        f"{html.escape(value)}</text>"
    )


def rect(x: float, y: float, width: float, height: float, fill: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(width, 0.0):.1f}" height="{height:.1f}" rx="2" fill="{fill}"/>'


def circle(x: float, y: float, radius: float, fill: str, *, stroke: str = "#ffffff") -> str:
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.5"/>'
    )


def svg_start(width: int, height: int, title: str, desc: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{html.escape(title)}</title>',
        f'<desc id="desc">{html.escape(desc)}</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text(32, 38, title, size=24, weight=700),
        text(32, 62, desc, size=13, color=COLORS["muted"]),
    ]


def seconds(value: float) -> str:
    if value < 0.001:
        return f"{value * 1e6:.0f} us"
    if value < 1.0:
        return f"{value * 1e3:.2f} ms"
    return f"{value:.2f} s"


def ratio(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}x"
    if value >= 10:
        return f"{value:.1f}x"
    if value < 0.01:
        return f"{value:.3f}x"
    if value < 0.1:
        return f"{value:.2f}x"
    return f"{value:.2f}x"


def find_case(report: dict[str, Any], name: str, scale: str) -> dict[str, Any]:
    for case in report["cases"]:
        if case["name"] == name and case["scale"] == scale:
            return case
    raise KeyError(f"case not found: {name} {scale}")


def case_by_scale(report: dict[str, Any], scale: str) -> dict[str, Any]:
    for case in report["cases"]:
        if case["scale"] == scale:
            return case
    raise KeyError(f"scale not found: {scale}")


def render_bar_chart(title: str, desc: str, rows: list[tuple[str, float, str]], output: Path) -> None:
    width = 1060
    left = 310
    top = 100
    row_h = 32
    max_bar = 520
    height = top + len(rows) * row_h + 44
    max_value = max(value for _, value, _ in rows)
    lines = svg_start(width, height, title, desc)
    lines.append(f'<line x1="{left}" y1="82" x2="{left + max_bar}" y2="82" stroke="{COLORS["grid"]}"/>')
    for index, (label, value, color) in enumerate(rows):
        y = top + index * row_h
        lines.append(text(32, y + 14, label, size=12))
        width_px = max_bar * (value / max_value if max_value else 0.0)
        lines.append(rect(left, y, width_px, 18, color))
        lines.append(text(left + width_px + 8, y + 14, seconds(value), size=12))
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_workspace(raw: dict[str, dict[str, Any]], output: Path) -> None:
    scale = "oneword_high_duplicate_terms_200000"
    rows = []
    for label, key, color in [
        ("default absent", "workspace_default", COLORS["gray"]),
        ("CUB reduce absent", "cub_reduce_absent", COLORS["blue"]),
        ("CUB reduce pre-reserved", "cub_reduce_prereserved", COLORS["green"]),
    ]:
        case = case_by_scale(raw[key], scale)
        rows.append((label, float(case["results"]["cuda_device_resident_seconds"]), color))
    render_bar_chart(
        "Campaign 4 Workspace Boundaries",
        f"Same simplify scale: {scale}; lower is better.",
        rows,
        output,
    )


def render_duplicate(raw: dict[str, dict[str, Any]], output: Path) -> None:
    rows = []
    for scale in [
        "oneword_low_duplicate_terms_200000",
        "oneword_medium_duplicate_terms_200000",
        "oneword_high_duplicate_terms_200000",
        "oneword_pathological_duplicate_terms_200000",
    ]:
        base = case_by_scale(raw["workspace_default"], scale)
        cub = case_by_scale(raw["cub_reduce_absent"], scale)
        speedup = float(base["results"]["cuda_device_resident_seconds"]) / float(
            cub["results"]["cuda_device_resident_seconds"]
        )
        rows.append((scale.replace("_", " "), speedup, COLORS["blue"]))
    width = 1060
    left = 420
    top = 100
    max_bar = 420
    height = top + len(rows) * 34 + 48
    max_value = max(value for _, value, _ in rows)
    lines = svg_start(width, height, "Campaign 4 Duplicate Reduction A/B", "CUB radix-sort prototype speedup versus default CUDA simplify; higher is better.")
    for index, (label, value, color) in enumerate(rows):
        y = top + index * 34
        lines.append(text(32, y + 14, label, size=12))
        width_px = max_bar * (value / max_value if max_value else 0.0)
        lines.append(rect(left, y, width_px, 18, color if value >= 1.0 else COLORS["amber"]))
        lines.append(text(left + width_px + 8, y + 14, ratio(value), size=12))
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_commutation(raw: dict[str, dict[str, Any]], output: Path) -> None:
    rows = []
    for case in raw["commutation_materialization"]["cases"]:
        target = str(case["dataset"]["output_target"]).replace("_", " ")
        results = case["results"]
        if case["dataset"]["output_target"] == "host_vector":
            value = float(results["cuda_device_resident_seconds"])
        elif case["dataset"]["output_target"] == "caller_owned_host_bytes":
            value = float(results["cuda_device_resident_preallocated_seconds"])
        else:
            value = float(results["cuda_device_resident_reused_device_output_seconds"])
        color = COLORS["green"] if "host" in target else COLORS["amber"]
        rows.append((target, value, color))
    render_bar_chart(
        "Campaign 4 Commutation Materialization",
        "Public host paths and private prototype labels on 8192x8192 dense output rows.",
        rows,
        output,
    )


def timings_for_cross(case: dict[str, Any]) -> list[tuple[str, float, str]]:
    results = case["results"]
    rows = [("CPU scalar", float(results["cpu_scalar_seconds"]), COLORS["gray"])]
    for backend, color in [("tbb", COLORS["purple"]), ("avx512", COLORS["teal"]), ("avx2", COLORS["amber"])]:
        timing = results.get("cpu_optimized_timings", {}).get(backend)
        if timing:
            rows.append((f"CPU {backend}", float(timing["seconds"]), color))
    rows.append(("CUDA transfer", float(results["cuda_transfer_inclusive_seconds"]), COLORS["blue"]))
    rows.append(("CUDA resident", float(results["cuda_device_resident_seconds"]), COLORS["green"]))
    return rows


SERIES_COLORS = {
    "CPU scalar": COLORS["gray"],
    "CPU TBB": COLORS["purple"],
    "CPU AVX2": COLORS["amber"],
    "CPU AVX-512": COLORS["teal"],
    "CUDA transfer-inclusive": COLORS["blue"],
    "CUDA device-resident": COLORS["green"],
    "CUDA operator-resident": "#16a34a",
    "External baseline": COLORS["red"],
}


OPERATION_LABELS = {
    "simplify_duplicate_pressure": "simplify",
    "statevector_expectation": "statevector",
    "pairwise_commutation": "commutation",
    "matmul_product_generation_simplify": "matmul+simplify",
}


def scale_label(scale: str) -> str:
    if scale.startswith("qubits_"):
        parts = scale.split("_")
        if len(parts) >= 4 and parts[0] == "qubits" and parts[2] == "terms":
            return f"{parts[1]}q {parts[3]} terms"
    if scale.startswith("terms_"):
        value = scale.removeprefix("terms_")
        if "x" in value:
            return value
        return f"{value} terms"
    return scale.replace("_", " ")


def add_speedup_point(points: list[dict[str, Any]], series: str, baseline: float, seconds_value: Any) -> None:
    if seconds_value is None:
        return
    value = float(seconds_value)
    if value <= 0.0:
        return
    points.append(
        {
            "series": series,
            "speedup_vs_cpu_scalar": baseline / value,
            "seconds": value,
        }
    )


def default_case_landscape_row(case: dict[str, Any]) -> dict[str, Any]:
    results = case["results"]
    baseline = float(results["cpu_scalar_seconds"])
    points: list[dict[str, Any]] = [
        {
            "series": "CPU scalar",
            "speedup_vs_cpu_scalar": 1.0,
            "seconds": baseline,
        }
    ]
    optimized = results.get("cpu_optimized_timings", {})
    for backend, label in [("tbb", "CPU TBB"), ("avx2", "CPU AVX2"), ("avx512", "CPU AVX-512")]:
        timing = optimized.get(backend)
        if timing:
            add_speedup_point(points, label, baseline, timing.get("seconds"))
    add_speedup_point(points, "CUDA transfer-inclusive", baseline, results.get("cuda_transfer_inclusive_seconds"))
    add_speedup_point(points, "CUDA device-resident", baseline, results.get("cuda_device_resident_seconds"))
    return {
        "category": "FastPauli default profile",
        "label": f"{OPERATION_LABELS.get(case['name'], case['name'])} {scale_label(case['scale'])}",
        "operation": case["name"],
        "scale": case["scale"],
        "points": points,
    }


def competitor_landscape_row(case: dict[str, Any]) -> dict[str, Any]:
    results = case["results"]
    dataset = case["dataset"]
    baseline = float(results["fastpauli_scalar_seconds"])
    points: list[dict[str, Any]] = [
        {
            "series": "CPU scalar",
            "speedup_vs_cpu_scalar": 1.0,
            "seconds": baseline,
        }
    ]
    if results.get("competitor_available"):
        add_speedup_point(points, "External baseline", baseline, results.get("competitor_seconds"))
    if case["name"] == "cuquantum_statevector_expectation":
        add_speedup_point(
            points,
            "CUDA operator-resident",
            baseline,
            results.get("fastpauli_cuda_operator_resident_host_statevector_seconds"),
        )
        add_speedup_point(
            points,
            "CUDA device-resident",
            baseline,
            results.get("fastpauli_cuda_device_resident_seconds"),
        )
    name = case["name"].replace("_", " ")
    if case["name"] == "simplify":
        name = "Qiskit simplify"
    elif case["name"] == "multiply":
        name = "OpenFermion multiply"
    elif case["name"] == "qiskit_grouping":
        name = "Qiskit grouping"
    elif case["name"] == "cuquantum_statevector_expectation":
        name = "cuStateVec statevector"
    label_bits = []
    if "num_terms" in dataset:
        label_bits.append(f"{dataset['num_terms']} terms")
    if "lhs_terms" in dataset and "rhs_terms" in dataset:
        label_bits.append(f"{dataset['lhs_terms']}x{dataset['rhs_terms']}")
    if "num_qubits" in dataset:
        label_bits.append(f"{dataset['num_qubits']}q")
    return {
        "category": "External package baseline",
        "label": f"{name} {' '.join(label_bits)}".strip(),
        "operation": case["name"],
        "scale": "external",
        "points": points,
    }


def build_performance_landscape(raw: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    default_rows = [
        default_case_landscape_row(case)
        for case in raw["default_cross"]["cases"]
        if case["name"]
        in {
            "simplify_duplicate_pressure",
            "statevector_expectation",
            "pairwise_commutation",
            "matmul_product_generation_simplify",
        }
    ]
    competitor_rows = [competitor_landscape_row(case) for case in raw["competitive"]["cases"]]
    return default_rows + competitor_rows


def render_performance_landscape(rows: list[dict[str, Any]], output: Path) -> None:
    width = 1320
    left = 360
    right = 1090
    top = 120
    row_h = 34
    legend_top = top + len(rows) * row_h + 36
    height = legend_top + 88
    speedups = [
        float(point["speedup_vs_cpu_scalar"])
        for row in rows
        for point in row["points"]
        if float(point["speedup_vs_cpu_scalar"]) > 0.0
    ]
    min_power = math.floor(math.log10(max(min(speedups), 1e-4)))
    max_power = math.ceil(math.log10(max(max(speedups), 1.01)))
    min_log = float(min(min_power, -3))
    max_log = float(max(max_power, 4))
    span = max_log - min_log

    def x_for(speedup: float) -> float:
        return left + ((math.log10(max(speedup, 10**min_log)) - min_log) / span) * (right - left)

    lines = svg_start(
        width,
        height,
        "FastPauli Performance Landscape",
        "Speedup versus FastPauli scalar CPU per row; values left of 1x are slower than scalar.",
    )
    lines.append(text(left, 88, "slower", size=11, color=COLORS["muted"], anchor="middle"))
    lines.append(text(x_for(1.0), 88, "CPU scalar baseline", size=11, color=COLORS["muted"], anchor="middle"))
    lines.append(text(right, 88, "faster", size=11, color=COLORS["muted"], anchor="middle"))

    tick_values = [10**power for power in range(int(min_log), int(max_log) + 1)]
    for tick in tick_values:
        x = x_for(tick)
        color = COLORS["ink"] if abs(tick - 1.0) < 1e-12 else COLORS["grid"]
        width_attr = "1.4" if abs(tick - 1.0) < 1e-12 else "0.8"
        lines.append(f'<line x1="{x:.1f}" y1="102" x2="{x:.1f}" y2="{legend_top - 16:.1f}" stroke="{color}" stroke-width="{width_attr}"/>')
        lines.append(text(x, 112, ratio(tick), size=10, color=COLORS["muted"], anchor="middle"))

    series_offsets = {
        "CPU scalar": -7,
        "CPU TBB": -4,
        "CPU AVX2": -1,
        "CPU AVX-512": 2,
        "CUDA transfer-inclusive": 5,
        "CUDA device-resident": 8,
        "CUDA operator-resident": 5,
        "External baseline": -8,
    }
    for index, row in enumerate(rows):
        y = top + index * row_h
        if index % 2 == 0:
            lines.append(f'<rect x="24" y="{y - 17:.1f}" width="{width - 48}" height="{row_h}" fill="#f9fafb"/>')
        lines.append(text(32, y + 5, str(row["label"]), size=11))
        best = max(float(point["speedup_vs_cpu_scalar"]) for point in row["points"])
        lines.append(text(1188, y + 5, f"best {ratio(best)}", size=10, color=COLORS["muted"], anchor="end"))
        for point in row["points"]:
            series = str(point["series"])
            speedup = float(point["speedup_vs_cpu_scalar"])
            x = x_for(speedup)
            offset = series_offsets.get(series, 0)
            lines.append(circle(x, y + offset, 5.0, SERIES_COLORS.get(series, COLORS["gray"])))

    legend_items = [
        "CPU scalar",
        "CPU TBB",
        "CPU AVX2",
        "CPU AVX-512",
        "CUDA transfer-inclusive",
        "CUDA operator-resident",
        "CUDA device-resident",
        "External baseline",
    ]
    legend_x = 32
    legend_y = legend_top
    for index, series in enumerate(legend_items):
        x = legend_x + (index % 4) * 290
        y = legend_y + (index // 4) * 28
        lines.append(circle(x, y - 4, 5.0, SERIES_COLORS[series]))
        lines.append(text(x + 14, y, series, size=11, color=COLORS["muted"]))
    lines.append(text(32, height - 24, "Source: Campaign 4 checked raw JSON; CUDA resident excludes host transfer/materialization where the benchmark defines that boundary.", size=11, color=COLORS["muted"]))
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_cross(raw: dict[str, dict[str, Any]], output: Path) -> None:
    cases = [
        find_case(raw["default_cross"], "simplify_duplicate_pressure", "terms_50000"),
        find_case(raw["default_cross"], "pairwise_commutation", "terms_2048x2048"),
        find_case(raw["default_cross"], "matmul_product_generation_simplify", "terms_256x256"),
    ]
    rows: list[tuple[str, float, str]] = []
    for case in cases:
        cpu = float(case["results"]["cpu_scalar_seconds"])
        for label, value, color in timings_for_cross(case):
            rows.append((f"{case['name'].split('_')[0]} {label}", cpu / value, color))
    width = 1180
    left = 430
    top = 100
    max_bar = 480
    row_h = 26
    height = top + len(rows) * row_h + 44
    max_value = max(value for _, value, _ in rows)
    denom = math.log10(max(max_value, 1.01))
    lines = svg_start(width, height, "Campaign 4 H100 Cross Comparison", "CPU scalar, optimized CPU selectors, and CUDA paths from the same default-profile evidence.")
    for index, (label, speedup, color) in enumerate(rows):
        y = top + index * row_h
        lines.append(text(32, y + 13, label, size=11))
        bar_width = max_bar * (math.log10(max(speedup, 1.0)) / denom if denom else 0.0)
        lines.append(rect(left, y, bar_width, 16, color))
        lines.append(text(left + bar_width + 8, y + 13, ratio(speedup), size=11))
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def log_contains(path: Path, needle: str) -> bool:
    return path.exists() and needle in path.read_text(encoding="utf-8", errors="replace")


def preferred_log_contains(preferred: Path, fallback: Path, needle: str) -> bool:
    if preferred.exists():
        return log_contains(preferred, needle)
    return log_contains(fallback, needle)


def collect_evidence_status(data_dir: Path) -> list[dict[str, Any]]:
    metadata = data_dir / "metadata"
    profiler = data_dir / "profiler"
    sanitizer_ok = all(
        preferred_log_contains(
            metadata / f"compute-sanitizer-{tool}-label-fix.log",
            metadata / f"compute-sanitizer-{tool}.log",
            expected,
        )
        for tool, expected in [
            ("memcheck", "ERROR SUMMARY: 0 errors"),
            ("racecheck", "RACECHECK SUMMARY: 0 hazards displayed"),
            ("initcheck", "ERROR SUMMARY: 0 errors"),
            ("synccheck", "ERROR SUMMARY: 0 errors"),
        ]
    )
    return [
        {
            "label": "baseline validation",
            "status": "passed"
            if log_contains(metadata / "baseline-validate.log", "Successfully built fastpauli")
            else "missing_or_failed",
        },
        {
            "label": "experiment validation",
            "status": "passed"
            if preferred_log_contains(
                metadata / "experiment-validate-final-label-fix.log",
                metadata / "experiment-validate-final.log",
                "Successfully built fastpauli",
            )
            else "missing_or_failed",
        },
        {
            "label": "phase 11 CUDA tests",
            "status": "passed"
            if (
                log_contains(metadata / "experiment-phase11-label-fix.log", "19 passed")
                if (metadata / "experiment-phase11-label-fix.log").exists()
                else log_contains(metadata / "experiment-phase11-post-label.log", "18 passed, 1 skipped")
            )
            else "missing_or_failed",
        },
        {
            "label": "compute-sanitizer ladder",
            "status": "passed" if sanitizer_ok else "missing_or_failed",
        },
        {
            "label": "unprivileged NCU permission note",
            "status": "expected_permission_denied"
            if log_contains(profiler / "ncu-statevector.log", "ERR_NVGPUCTRPERM")
            else "missing_or_unexpected",
        },
        {
            "label": "privileged Nsight Compute",
            "status": "passed"
            if log_contains(profiler / "ncu_statevector_campaign4_details.csv", "Metric Name")
            else "missing_or_failed",
        },
        {
            "label": "Nsight Systems summaries",
            "status": "passed"
            if log_contains(profiler / "nsys_campaign4_workspace_stats_cuda_api_sum.csv", "cudaMalloc")
            else "missing_or_failed",
        },
        {
            "label": "competitor package installs",
            "status": "passed"
            if (
                (metadata / "competitor-install-status.txt").exists()
                and "failed" not in (metadata / "competitor-install-status.txt").read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            )
            else "missing_or_failed",
        },
    ]


def render_status(statuses: list[dict[str, Any]], output: Path) -> None:
    width = 1060
    top = 100
    row_h = 32
    height = top + len(statuses) * row_h + 44
    lines = svg_start(
        width,
        height,
        "Campaign 4 Evidence Status",
        "Validation, profiler, sanitizer, and competitor gates derived from checked metadata logs.",
    )
    for index, item in enumerate(statuses):
        y = top + index * row_h
        status = str(item["status"])
        ok = status in {"passed", "expected_permission_denied"}
        color = COLORS["green"] if ok else COLORS["red"]
        lines.append(rect(32, y, 18, 18, color))
        lines.append(text(62, y + 14, str(item["label"]), size=12))
        lines.append(text(520, y + 14, status, size=12, color=color))
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(raw: dict[str, dict[str, Any]], data_dir: Path) -> dict[str, Any]:
    metadata_dir = data_dir / "metadata"
    baseline_revision = (metadata_dir / "baseline-revision.txt").read_text(encoding="utf-8").strip()
    experiment_revision = (metadata_dir / "experiment-revision.txt").read_text(encoding="utf-8").strip()
    build_info = raw["workspace_default"]["fastpauli_build_info"]
    high_case = case_by_scale(raw["workspace_default"], "oneword_high_duplicate_terms_200000")
    high_cub = case_by_scale(raw["cub_reduce_absent"], "oneword_high_duplicate_terms_200000")
    cub_speedup = float(high_case["results"]["cuda_device_resident_seconds"]) / float(
        high_cub["results"]["cuda_device_resident_seconds"]
    )
    evidence_status = collect_evidence_status(data_dir)
    return {
        "campaign": "h100_campaign4",
        "date": "2026-04-29",
        "baseline_revision": baseline_revision,
        "experiment_revision": experiment_revision,
        "hardware": {
            "gpu": (metadata_dir / "gpu.txt").read_text(encoding="utf-8").strip(),
            "cuda_toolkit": build_info["cuda_toolkit_version"],
            "compiled_cuda_architectures": build_info["cuda_architectures"],
            "compiler": build_info["compiler_build_config"],
            "available_cpu_backends": build_info["available_cpu_backends"],
        },
        "decisions": [
            {
                "experiment": "private_cuda_workspace",
                "status": "benchmark-only",
                "reason": "Implemented internally for scratch ownership and timing-boundary experiments; not public API.",
            },
            {
                "experiment": "cub_radix_sort_duplicate_reduction",
                "status": "rejected_for_production",
                "same_boundary_speedup_high_duplicate": cub_speedup,
                "reason": "Narrow CUB radix-sort prototype did not show broad enough same-boundary wins to replace the production Thrust path.",
            },
            {
                "experiment": "cub_run_length_duplicate_reduction",
                "status": "not_implemented_fallback",
                "reason": "The run-length selector is recorded as production fallback evidence; no DeviceRunLengthEncode path was retained.",
            },
            {
                "experiment": "commutation_device_output",
                "status": "deferred_to_api_review",
                "reason": "Host vector and caller-owned host bytes remain supported; device-byte and bit-packed outputs require public API design.",
            },
            {
                "experiment": "statevector_reduction_topology",
                "status": "retained_current",
                "reason": "Privileged NCU showed compute-heavy fused kernel behavior; no replacement topology was retained.",
            },
        ],
        "evidence": {
            "status": evidence_status,
            "baseline_json": sorted(path.name for path in (data_dir / "raw").glob("baseline_*.json")),
            "experiment_json": sorted(path.name for path in (data_dir / "raw").glob("experiment_*.json")),
            "competitive_baselines": "competitive_baselines_final.json",
            "sanitizers": ["memcheck", "racecheck", "initcheck", "synccheck"],
            "profilers": ["nsys", "ncu_privileged", "ncu_unprivileged_permission_denied"],
        },
        "competitors": raw["competitive"].get("competitors", {}),
        "readme_performance_landscape": build_performance_landscape(raw),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("docs/benchmarks/data/cuda_deep_optimization_h100_campaign4_2026-04-29"),
    )
    parser.add_argument("--plot-dir", type=Path, default=Path("docs/benchmarks/plots"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = load_raw(args.data_dir / "raw")
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(raw, args.data_dir)
    (args.data_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    render_workspace(raw, args.plot_dir / PLOTS["workspace"])
    render_duplicate(raw, args.plot_dir / PLOTS["duplicate"])
    render_commutation(raw, args.plot_dir / PLOTS["commutation"])
    render_cross(raw, args.plot_dir / PLOTS["cross"])
    render_performance_landscape(summary["readme_performance_landscape"], args.plot_dir / PLOTS["landscape"])
    render_status(summary["evidence"]["status"], args.plot_dir / PLOTS["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Render Campaign 5 H100 CUDA device-output summary and SVG assets."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

RAW_FILES = {
    "device_output_scaling": "experiment_campaign5_device_output_scaling.json",
    "default_cross": "experiment_cuda_kernels_default.json",
    "competitive": "competitive_baselines_final.json",
}

OPTIONAL_RAW_FILES = {
    "campaign4_commutation_reference": "campaign4_commutation_materialization_reference.json",
}

PLOTS = {
    "device_output": "cuda_h100_campaign5_device_output_boundaries.svg",
    "host_decomposition": "cuda_h100_campaign5_host_materialization_decomposition.svg",
    "landscape": "cuda_h100_campaign5_performance_landscape.svg",
    "status": "cuda_h100_campaign5_evidence_status.svg",
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
    "slate": "#334155",
}

SERIES_COLORS = {
    "CPU scalar": COLORS["gray"],
    "CPU TBB": COLORS["purple"],
    "CPU AVX2": COLORS["amber"],
    "CPU AVX-512": COLORS["teal"],
    "CUDA transfer-inclusive": COLORS["blue"],
    "CUDA device-resident": COLORS["green"],
    "CUDA device-output allocate": "#0891b2",
    "CUDA device-output reuse": "#16a34a",
    "CUDA device-output to_host": "#65a30d",
    "CUDA operator-resident": "#15803d",
    "External baseline": COLORS["red"],
}

OPERATION_LABELS = {
    "simplify_duplicate_pressure": "simplify",
    "statevector_expectation": "statevector",
    "pairwise_commutation": "commutation",
    "matmul_product_generation_simplify": "matmul+simplify",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_raw(data_dir: Path) -> dict[str, dict[str, Any]]:
    raw_dir = data_dir / "raw"
    missing = [filename for filename in RAW_FILES.values() if not (raw_dir / filename).exists()]
    if missing:
        raise FileNotFoundError("missing Campaign 5 raw artifact(s): " + ", ".join(missing))
    raw = {key: load_json(raw_dir / filename) for key, filename in RAW_FILES.items()}
    for key, filename in OPTIONAL_RAW_FILES.items():
        path = raw_dir / filename
        if path.exists():
            raw[key] = load_json(path)
    return raw


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 12,
    weight: int = 400,
    color: str = COLORS["ink"],
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">'
        f"{html.escape(value)}</text>"
    )


def rect(x: float, y: float, width: float, height: float, fill: str, *, rx: float = 2.0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(width, 0.0):.1f}" '
        f'height="{height:.1f}" rx="{rx:.1f}" fill="{fill}"/>'
    )


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
    if value < 1.0e-6:
        return f"{value * 1.0e9:.0f} ns"
    if value < 0.001:
        return f"{value * 1.0e6:.0f} us"
    if value < 1.0:
        return f"{value * 1.0e3:.2f} ms"
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


def scale_label(scale: str) -> str:
    if "terms_" in scale:
        value = scale.split("terms_", 1)[1]
        if "x" in value:
            return value
        return f"{value} terms"
    if scale.startswith("qubits_"):
        parts = scale.split("_")
        if len(parts) >= 4:
            return f"{parts[1]}q {parts[3]} terms"
    return scale.replace("_", " ")


def inferred_case_scale(case: dict[str, Any]) -> str:
    if "scale" in case:
        return str(case["scale"])
    dataset = case.get("dataset", {})
    name = case.get("name", "")
    if name == "statevector_expectation":
        return f"qubits_{dataset.get('num_qubits', 'unknown')}_terms_{dataset.get('num_terms', 'unknown')}"
    if "lhs_terms" in dataset and "rhs_terms" in dataset:
        return f"terms_{dataset['lhs_terms']}x{dataset['rhs_terms']}"
    if "num_terms" in dataset:
        return f"terms_{dataset['num_terms']}"
    return str(name)


def case_by_scale(report: dict[str, Any], scale: str) -> dict[str, Any]:
    for case in report["cases"]:
        if case["scale"] == scale:
            return case
    raise KeyError(f"scale not found: {scale}")


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
    for backend, label in [
        ("tbb", "CPU TBB"),
        ("avx2", "CPU AVX2"),
        ("avx512", "CPU AVX-512"),
    ]:
        timing = optimized.get(backend)
        if timing:
            add_speedup_point(points, label, baseline, timing.get("seconds"))
    add_speedup_point(points, "CUDA transfer-inclusive", baseline, results.get("cuda_transfer_inclusive_seconds"))
    add_speedup_point(points, "CUDA device-resident", baseline, results.get("cuda_device_resident_seconds"))
    add_speedup_point(points, "CUDA device-output allocate", baseline, results.get("cuda_device_output_allocate_seconds"))
    add_speedup_point(points, "CUDA device-output reuse", baseline, results.get("cuda_device_output_reuse_seconds"))
    add_speedup_point(points, "CUDA device-output to_host", baseline, results.get("cuda_device_output_to_host_seconds"))
    return {
        "category": "FastPauli default profile",
        "label": f"{OPERATION_LABELS.get(case['name'], case['name'])} {scale_label(inferred_case_scale(case))}",
        "operation": case["name"],
        "scale": inferred_case_scale(case),
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


def device_output_boundary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for case in report["cases"]:
        results = case["results"]
        rows.append(
            {
                "scale": case["scale"],
                "entries": case["dataset"]["entries"],
                "host_vector_seconds": results["cuda_device_resident_seconds"],
                "host_preallocated_seconds": results.get("cuda_device_resident_preallocated_seconds"),
                "private_reused_device_output_seconds": results.get(
                    "cuda_device_resident_reused_device_output_seconds"
                ),
                "public_device_output_allocate_seconds": results["cuda_device_output_allocate_seconds"],
                "public_device_output_reuse_seconds": results["cuda_device_output_reuse_seconds"],
                "public_device_output_to_host_seconds": results["cuda_device_output_to_host_seconds"],
                "public_cuda_array_interface_export_seconds": results[
                    "cuda_device_output_cuda_array_interface_export_seconds"
                ],
            }
        )
    return rows


def log_scale_mapper(values: list[float], left: float, right: float):
    positive = [value for value in values if value > 0.0]
    min_power = math.floor(math.log10(max(min(positive), 1.0e-9)))
    max_power = math.ceil(math.log10(max(max(positive), 1.01e-6)))
    min_log = min(float(min_power), -7.0)
    max_log = max(float(max_power), -2.0)
    span = max(max_log - min_log, 1.0)

    def x_for(value: float) -> float:
        return left + ((math.log10(max(value, 10**min_log)) - min_log) / span) * (right - left)

    return x_for, min_log, max_log


def render_device_output_boundaries(rows: list[dict[str, Any]], output: Path) -> None:
    series = [
        ("host vector", "host_vector_seconds", COLORS["blue"], -9),
        ("host preallocated", "host_preallocated_seconds", COLORS["purple"], -3),
        ("private device reuse", "private_reused_device_output_seconds", COLORS["amber"], 3),
        ("public device allocate", "public_device_output_allocate_seconds", "#0891b2", 9),
        ("public device reuse", "public_device_output_reuse_seconds", COLORS["green"], 15),
    ]
    values = [
        float(row[key])
        for row in rows
        for _, key, _, _ in series
        if row.get(key) is not None
    ]
    width = 1260
    left = 330
    right = 1010
    top = 120
    row_h = 48
    legend_top = top + len(rows) * row_h + 36
    height = legend_top + 72
    x_for, min_log, max_log = log_scale_mapper(values, left, right)
    lines = svg_start(
        width,
        height,
        "Campaign 5 Device-Output Boundaries",
        "Same pairwise commutation datasets; lower latency is farther left on the log axis.",
    )
    for power in range(int(min_log), int(max_log) + 1):
        tick = 10**power
        x = x_for(tick)
        lines.append(f'<line x1="{x:.1f}" y1="94" x2="{x:.1f}" y2="{legend_top - 20:.1f}" stroke="{COLORS["grid"]}" stroke-width="0.8"/>')
        lines.append(text(x, 108, seconds(tick), size=10, color=COLORS["muted"], anchor="middle"))
    for index, row in enumerate(rows):
        y = top + index * row_h
        if index % 2 == 0:
            lines.append(rect(24, y - 20, width - 48, row_h - 2, "#f9fafb", rx=0))
        lines.append(text(32, y + 3, scale_label(str(row["scale"])), size=12))
        lines.append(text(178, y + 3, f"{int(row['entries']):,} entries", size=10, color=COLORS["muted"]))
        for label, key, color, offset in series:
            value = row.get(key)
            if value is None:
                continue
            lines.append(circle(x_for(float(value)), y + offset, 5.3, color))
        best = min(float(row[key]) for _, key, _, _ in series if row.get(key) is not None)
        lines.append(text(1160, y + 3, f"best {seconds(best)}", size=10, color=COLORS["muted"], anchor="end"))
    for index, (label, _, color, _) in enumerate(series):
        x = 32 + index * 230
        y = legend_top
        lines.append(circle(x, y - 4, 5.3, color))
        lines.append(text(x + 14, y, label, size=11, color=COLORS["muted"]))
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_host_decomposition(rows: list[dict[str, Any]], output: Path) -> None:
    target = rows[-1]
    host = float(target["host_vector_seconds"])
    reuse = float(target["public_device_output_reuse_seconds"])
    to_host = float(target["public_device_output_to_host_seconds"])
    allocate = float(target["public_device_output_allocate_seconds"])
    allocation_delta = max(allocate - reuse, 0.0)
    data = [
        ("public device reuse", reuse, COLORS["green"]),
        ("device-to-host materialization", to_host, COLORS["blue"]),
        ("allocation overhead estimate", allocation_delta, COLORS["amber"]),
        ("legacy host-vector boundary", host, COLORS["purple"]),
    ]
    width = 1080
    left = 310
    top = 118
    row_h = 40
    max_bar = 560
    height = top + len(data) * row_h + 58
    max_value = max(value for _, value, _ in data)
    lines = svg_start(
        width,
        height,
        "Campaign 5 Host-Materialization Decomposition",
        f"Largest measured Campaign 5 matrix: {scale_label(str(target['scale']))}; lower is better.",
    )
    for index, (label, value, color) in enumerate(data):
        y = top + index * row_h
        lines.append(text(32, y + 14, label, size=12))
        width_px = max_bar * (value / max_value if max_value else 0.0)
        lines.append(rect(left, y, width_px, 18, color))
        lines.append(text(left + width_px + 10, y + 14, seconds(value), size=12))
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_performance_landscape(rows: list[dict[str, Any]], output: Path) -> None:
    width = 1380
    left = 370
    right = 1115
    top = 120
    row_h = 36
    legend_top = top + len(rows) * row_h + 36
    height = legend_top + 104
    speedups = [
        float(point["speedup_vs_cpu_scalar"])
        for row in rows
        for point in row["points"]
        if float(point["speedup_vs_cpu_scalar"]) > 0.0
    ]
    min_power = math.floor(math.log10(max(min(speedups), 1.0e-4)))
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
        "Campaign 5 broad comparison: CPU selectors, CUDA boundaries, device-output commutation, and external baselines.",
    )
    for tick in [10**power for power in range(int(min_log), int(max_log) + 1)]:
        x = x_for(tick)
        color = COLORS["ink"] if abs(tick - 1.0) < 1.0e-12 else COLORS["grid"]
        width_attr = "1.4" if abs(tick - 1.0) < 1.0e-12 else "0.8"
        lines.append(f'<line x1="{x:.1f}" y1="102" x2="{x:.1f}" y2="{legend_top - 16:.1f}" stroke="{color}" stroke-width="{width_attr}"/>')
        lines.append(text(x, 112, ratio(tick), size=10, color=COLORS["muted"], anchor="middle"))
    series_offsets = {
        "CPU scalar": -10,
        "CPU TBB": -6,
        "CPU AVX2": -2,
        "CPU AVX-512": 2,
        "CUDA transfer-inclusive": 6,
        "CUDA device-resident": 10,
        "CUDA device-output allocate": 14,
        "CUDA device-output reuse": 18,
        "CUDA device-output to_host": 22,
        "CUDA operator-resident": 10,
        "External baseline": -14,
    }
    for index, row in enumerate(rows):
        y = top + index * row_h
        if index % 2 == 0:
            lines.append(rect(24, y - 18, width - 48, row_h, "#f9fafb", rx=0))
        lines.append(text(32, y + 5, str(row["label"]), size=11))
        best = max(float(point["speedup_vs_cpu_scalar"]) for point in row["points"])
        lines.append(text(1240, y + 5, f"best {ratio(best)}", size=10, color=COLORS["muted"], anchor="end"))
        for point in row["points"]:
            series_name = str(point["series"])
            x = x_for(float(point["speedup_vs_cpu_scalar"]))
            lines.append(circle(x, y + series_offsets.get(series_name, 0), 5.0, SERIES_COLORS.get(series_name, COLORS["gray"])))
    legend_items = [
        "CPU scalar",
        "CPU TBB",
        "CPU AVX2",
        "CPU AVX-512",
        "CUDA transfer-inclusive",
        "CUDA device-resident",
        "CUDA device-output allocate",
        "CUDA device-output reuse",
        "CUDA device-output to_host",
        "CUDA operator-resident",
        "External baseline",
    ]
    for index, series_name in enumerate(legend_items):
        x = 32 + (index % 4) * 320
        y = legend_top + (index // 4) * 28
        lines.append(circle(x, y - 4, 5.0, SERIES_COLORS[series_name]))
        lines.append(text(x + 14, y, series_name, size=11, color=COLORS["muted"]))
    lines.append(text(32, height - 24, "Source: Campaign 5 checked raw JSON. CUDA device-output points apply to pairwise commutation rows and keep dense uint8 output on GPU.", size=11, color=COLORS["muted"]))
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def log_contains(path: Path, needle: str) -> bool:
    return needle in read_text_if_exists(path)


def collect_evidence_status(data_dir: Path) -> list[dict[str, str]]:
    metadata = data_dir / "metadata"
    profiler = data_dir / "profiler"
    sanitizer_ok = all(
        log_contains(metadata / f"compute-sanitizer-{tool}.log", expected)
        for tool, expected in [
            ("memcheck", "ERROR SUMMARY: 0 errors"),
            ("racecheck", "RACECHECK SUMMARY: 0 hazards displayed"),
            ("initcheck", "ERROR SUMMARY: 0 errors"),
            ("synccheck", "ERROR SUMMARY: 0 errors"),
        ]
    )
    privileged_ncu_ok = log_contains(
        profiler / "ncu_campaign5_commutation_details.csv",
        "Metric Name",
    )
    unprivileged_ncu_permission = log_contains(
        profiler / "ncu-campaign5-commutation.log",
        "ERR_NVGPUCTRPERM",
    )
    return [
        {
            "label": "experiment validation",
            "status": "passed" if log_contains(metadata / "experiment-validate-final.log", "Successfully built fastpauli") else "missing_or_failed",
        },
        {
            "label": "phase 11 CUDA tests",
            "status": "passed" if log_contains(metadata / "experiment-phase11-cuda.log", "19 passed") else "missing_or_failed",
        },
        {
            "label": "compute-sanitizer ladder",
            "status": "passed" if sanitizer_ok else "missing_or_failed",
        },
        {
            "label": "Campaign 5 scaling benchmark",
            "status": "passed" if (data_dir / "raw" / RAW_FILES["device_output_scaling"]).exists() else "missing_or_failed",
        },
        {
            "label": "default CUDA benchmark",
            "status": "passed" if (data_dir / "raw" / RAW_FILES["default_cross"]).exists() else "missing_or_failed",
        },
        {
            "label": "competitive baselines",
            "status": "passed" if (data_dir / "raw" / RAW_FILES["competitive"]).exists() else "missing_or_failed",
        },
        {
            "label": "Nsight Systems",
            "status": "passed" if any(path.exists() for path in profiler.glob("nsys_campaign5_device_output*")) else "missing_or_failed",
        },
        {
            "label": "Nsight Compute",
            "status": "passed"
            if privileged_ncu_ok
            else "expected_permission_denied"
            if unprivileged_ncu_permission
            else "missing_or_failed",
        },
    ]


def render_status(statuses: list[dict[str, str]], output: Path) -> None:
    width = 1060
    top = 100
    row_h = 32
    height = top + len(statuses) * row_h + 44
    lines = svg_start(
        width,
        height,
        "Campaign 5 Evidence Status",
        "Validation, benchmark, sanitizer, profiler, and competitor gates derived from checked artifacts.",
    )
    for index, item in enumerate(statuses):
        y = top + index * row_h
        status = item["status"]
        ok = status in {"passed", "passed_or_permission_recorded"}
        color = COLORS["green"] if ok else COLORS["red"]
        lines.append(rect(32, y, 18, 18, color))
        lines.append(text(62, y + 14, item["label"], size=12))
        lines.append(text(520, y + 14, status, size=12, color=color))
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def gpu_summary(data_dir: Path) -> str:
    gpu_csv = data_dir / "metadata" / "gpu.csv"
    if not gpu_csv.exists():
        return "unknown"
    with gpu_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        return gpu_csv.read_text(encoding="utf-8").strip()
    header = [item.strip() for item in rows[0]]
    values = [item.strip() for item in rows[1]]
    fields = dict(zip(header, values, strict=False))
    name = fields.get("name", values[0] if values else "unknown")
    compute = fields.get("compute_cap", "")
    driver = fields.get("driver_version", "")
    bits = [name]
    if compute:
        bits.append(f"SM {compute}")
    if driver:
        bits.append(f"driver {driver}")
    return ", ".join(bits)


def build_summary(raw: dict[str, dict[str, Any]], data_dir: Path) -> dict[str, Any]:
    metadata_dir = data_dir / "metadata"
    scaling = raw["device_output_scaling"]
    default_cross = raw["default_cross"]
    boundary_rows = device_output_boundary_rows(scaling)
    statuses = collect_evidence_status(data_dir)
    return {
        "campaign": "h100_campaign5",
        "date": "2026-04-29",
        "baseline_revision": read_text_if_exists(metadata_dir / "baseline-revision.txt").strip(),
        "experiment_revision": read_text_if_exists(metadata_dir / "experiment-revision.txt").strip()
        or str(default_cross.get("git_commit", "")),
        "hardware": {
            "gpu": gpu_summary(data_dir),
            "cuda_toolkit": default_cross.get("fastpauli_build_info", {}).get("cuda_toolkit_version"),
            "compiled_cuda_architectures": default_cross.get("fastpauli_build_info", {}).get("cuda_architectures"),
            "available_cpu_backends": default_cross.get("fastpauli_build_info", {}).get("available_cpu_backends", []),
            "compiler": default_cross.get("fastpauli_build_info", {}).get("compiler_build_config", {}),
        },
        "decisions": [
            {
                "experiment": "device_output_api",
                "status": "experimental_public_api",
                "reason": "Dense DeviceCommutationMatrix is retained as the supported GPU-resident commutation boundary.",
            },
            {
                "experiment": "output_storage_format",
                "status": "dense_uint8_retained",
                "reason": "Dense uint8 is directly consumable through CUDA Array Interface; bit-packed output remains design-deferred.",
            },
            {
                "experiment": "synchronization_policy",
                "status": "synchronize_before_return_retained",
                "reason": "Campaign 5 preserves the existing synchronous public CUDA semantics.",
            },
            {
                "experiment": "stream_or_async_api",
                "status": "deferred",
                "reason": "No public stream handle or async lifetime contract was added in this campaign.",
            },
            {
                "experiment": "readme_plot_policy",
                "status": "refreshed_broad_landscape",
                "reason": "Campaign 5 adds checked broad CPU/CUDA/external rows plus device-output commutation points.",
            },
        ],
        "device_output_boundaries": boundary_rows,
        "readme_performance_landscape": build_performance_landscape(raw),
        "competitors": raw["competitive"].get("competitors", {}),
        "evidence": {
            "status": statuses,
            "raw_json": sorted(path.name for path in (data_dir / "raw").glob("*.json")),
            "sanitizers": ["memcheck", "racecheck", "initcheck", "synccheck"],
            "profilers": ["nsys", "ncu"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("docs/benchmarks/data/cuda_deep_optimization_h100_campaign5_2026-04-29"),
    )
    parser.add_argument("--plot-dir", type=Path, default=Path("docs/benchmarks/plots"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = load_raw(args.data_dir)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(raw, args.data_dir)
    (args.data_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    boundary_rows = summary["device_output_boundaries"]
    render_device_output_boundaries(boundary_rows, args.plot_dir / PLOTS["device_output"])
    render_host_decomposition(boundary_rows, args.plot_dir / PLOTS["host_decomposition"])
    render_performance_landscape(summary["readme_performance_landscape"], args.plot_dir / PLOTS["landscape"])
    render_status(summary["evidence"]["status"], args.plot_dir / PLOTS["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

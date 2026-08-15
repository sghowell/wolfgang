#!/usr/bin/env python3
"""Render summary evidence and plots for ROCm MI300X Campaign 4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from render_rocm_campaign3_assets import (
    display_path,
    read_json,
    render_bar_svg,
    rocm_campaign2_landscape_points,
    timing_value,
    write_text,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "docs/benchmarks/data/rocm_mi300x_campaign4_2026-04-30"
DEFAULT_PLOT_DIR = ROOT / "docs/benchmarks/plots"
ROCM_CAMPAIGN3_SUMMARY = (
    ROOT / "docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30/summary.json"
)
CUDA_CAMPAIGN10_SUMMARY = (
    ROOT / "docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29/summary.json"
)

SIMPLIFY_PLOT = "rocm_mi300x_campaign4_simplify_hardening.svg"
LANDSCAPE_PLOT = "accelerator_landscape_with_rocm.svg"

TERMINAL_ITEMS = {
    "workspace",
    "custom packed key",
    "generic multi-word",
    "DLPack",
    "streams",
    "expectation",
    "matmul",
    "portability",
    "ROCm wheels",
    "multi-GPU",
    "simultaneous CUDA+HIP",
}

CAMPAIGN4_ROW_FIELDS = {
    "hip_simplify_strategy",
    "hip_simplify_strategy_status",
    "hip_simplify_strategy_reason",
    "hip_simplify_key_shape",
    "hip_workspace_mode",
    "hip_workspace_reserved_bytes",
    "hip_workspace_high_watermark_bytes",
    "hip_workspace_allocation_count",
    "hip_workspace_growth_count",
    "generic_multiword_parallelism",
    "campaign4_terminal_statuses",
}


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
            item["git_commit"] = report.get("git_commit")
            rows.append(item)
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Campaign 4 summary has no simplify rows")

    terminal_coverage: set[str] = set()
    retained_generic = False
    ok_rows = 0
    for row in rows:
        missing = sorted(field for field in CAMPAIGN4_ROW_FIELDS if field not in row)
        if missing:
            raise ValueError(f"Campaign 4 row omits {missing}: {row.get('source_file')}")

        terminal_coverage.update(row.get("campaign4_terminal_statuses", {}))
        if (
            row.get("hip_simplify_key_shape") == "generic_multiword"
            and row.get("generic_multiword_parallelism") == "reduce_by_key"
            and row.get("hip_simplify_strategy_status") == "retained"
        ):
            retained_generic = True

        if row.get("status") != "ok":
            continue
        ok_rows += 1
        required = (
            "hip_simplify_transfer_seconds",
            "hip_simplify_device_resident_seconds",
            "hip_simplify_to_host_seconds",
            "hip_simplify_output_terms",
            "hip_simplify_output_words",
            "correctness_digest",
        )
        missing_ok = [field for field in required if row.get(field) is None]
        if missing_ok:
            raise ValueError(f"ok simplify row omits {missing_ok}: {row.get('source_file')}")
        if row.get("correctness_passed") is not True:
            raise ValueError(f"ok simplify row failed correctness: {row.get('source_file')}")

    if ok_rows == 0:
        raise ValueError("Campaign 4 summary has no successful simplify rows")
    if not retained_generic:
        raise ValueError("Campaign 4 summary has no retained generic reduce_by_key row")
    missing_terminal = TERMINAL_ITEMS - terminal_coverage
    if missing_terminal:
        raise ValueError(f"Campaign 4 rows omit terminal statuses: {sorted(missing_terminal)}")


def summarize_row(row: dict[str, Any]) -> dict[str, Any]:
    dataset = row.get("dataset", {})
    return {
        "profile": row.get("report_profile", row.get("profile")),
        "case": row.get("case"),
        "status": row.get("status"),
        "correctness_passed": row.get("correctness_passed"),
        "git_commit": row.get("git_commit"),
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
        "hip_simplify_strategy_reason": row.get("hip_simplify_strategy_reason"),
        "hip_simplify_key_shape": row.get("hip_simplify_key_shape"),
        "generic_multiword_parallelism": row.get("generic_multiword_parallelism"),
        "hip_workspace_mode": row.get("hip_workspace_mode"),
        "hip_workspace_reserved_bytes": row.get("hip_workspace_reserved_bytes"),
        "hip_workspace_high_watermark_bytes": row.get("hip_workspace_high_watermark_bytes"),
        "hip_workspace_allocation_count": row.get("hip_workspace_allocation_count"),
        "hip_workspace_growth_count": row.get("hip_workspace_growth_count"),
        "hip_simplify_output_terms": row.get("hip_simplify_output_terms"),
        "hip_simplify_output_words": row.get("hip_simplify_output_words"),
        "correctness_digest": row.get("correctness_digest"),
        "source_file": row.get("source_file"),
    }


def profiler_artifacts(data_dir: Path) -> list[str]:
    profiler = data_dir / "profiler"
    if not profiler.exists():
        return []
    return [display_path(path) for path in sorted(profiler.rglob("*")) if path.is_file()]


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
        if row.get("report_profile") == "simplify-campaign4-profiler":
            continue
        case = str(row.get("case"))
        strategy = str(row.get("hip_simplify_strategy"))
        parallelism = str(row.get("generic_multiword_parallelism"))
        label_suffix = strategy
        if parallelism != "not_applicable":
            label_suffix = f"{strategy}/{parallelism}"
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
                    "series": label_suffix if series.startswith("HIP") else series,
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


def previous_rocm_landscape_points() -> list[dict[str, Any]]:
    if not ROCM_CAMPAIGN3_SUMMARY.exists():
        return []
    summary = read_json(ROCM_CAMPAIGN3_SUMMARY)
    points: list[dict[str, Any]] = []
    for point in summary.get("readme_performance_landscape", []):
        series = str(point.get("series", ""))
        if not (series.startswith("ROCm") or series == "CPU scalar"):
            continue
        seconds = timing_value(point, "seconds")
        if seconds is None:
            continue
        points.append(
            {
                "label": str(point.get("label", series)),
                "series": series,
                "seconds": seconds,
                "color": str(point.get("color", "#64748b")),
            }
        )
    return points


def campaign4_landscape_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        case = str(row.get("case"))
        for series, key, color in (
            ("CPU scalar", "cpu_scalar_seconds", "#334155"),
            ("ROCm HIP transfer", "hip_simplify_transfer_seconds", "#0e7490"),
            ("ROCm HIP resident", "hip_simplify_device_resident_seconds", "#2563eb"),
            ("ROCm HIP to_host", "hip_simplify_to_host_seconds", "#be123c"),
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
    return points


def landscape_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points = cuda_landscape_points()
    points.extend(rocm_campaign2_landscape_points())
    points.extend(previous_rocm_landscape_points())
    points.extend(campaign4_landscape_points(rows))
    return points


def strategy_decisions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decisions: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        strategy = str(row.get("hip_simplify_strategy", "unavailable"))
        key_shape = str(row.get("hip_simplify_key_shape", "unknown"))
        decisions[(strategy, key_shape)] = {
            "strategy": strategy,
            "key_shape": key_shape,
            "generic_multiword_parallelism": row.get("generic_multiword_parallelism"),
            "status": row.get("hip_simplify_strategy_status"),
            "reason": row.get("hip_simplify_strategy_reason"),
            "source_file": row.get("source_file"),
        }
    return [decisions[key] for key in sorted(decisions)]


def generic_ab_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    serial = next(
        (
            row
            for row in rows
            if row.get("case") == "campaign4_generic_serial_baseline"
            and row.get("status") == "ok"
        ),
        None,
    )
    parallel = next(
        (
            row
            for row in rows
            if row.get("case") == "campaign4_generic_parallel_baseline"
            and row.get("status") == "ok"
        ),
        None,
    )
    if serial is None or parallel is None:
        return {"available": False, "reason": "serial or parallel baseline row missing"}
    serial_seconds = timing_value(serial, "hip_simplify_device_resident_seconds")
    parallel_seconds = timing_value(parallel, "hip_simplify_device_resident_seconds")
    if serial_seconds is None or parallel_seconds is None:
        return {"available": False, "reason": "serial or parallel timing missing"}
    return {
        "available": True,
        "serial_device_resident_seconds": serial_seconds,
        "parallel_device_resident_seconds": parallel_seconds,
        "speedup": serial_seconds / parallel_seconds,
        "serial_cpu_scalar_seconds": serial.get("cpu_scalar_seconds"),
        "parallel_cpu_scalar_seconds": parallel.get("cpu_scalar_seconds"),
    }


def build_summary(data_dir: Path, plot_dir: Path) -> dict[str, Any]:
    reports = load_raw_reports(data_dir)
    rows = flatten_simplify_rows(reports)
    validate_rows(rows)
    plot_dir.mkdir(parents=True, exist_ok=True)

    render_bar_svg(
        title="MI300X ROCm Campaign 4 Simplify Hardening",
        subtitle="Generic reduce-by-key, packed-key probes, and workspace-status evidence",
        points=simplify_plot_points(rows),
        output=plot_dir / SIMPLIFY_PLOT,
    )
    render_bar_svg(
        title="FastPauli Accelerator Performance Landscape",
        subtitle="Checked CPU, CUDA, external, and ROCm evidence; log-scale seconds",
        points=landscape_points(rows),
        output=plot_dir / LANDSCAPE_PLOT,
    )

    return {
        "campaign": "rocm_mi300x_campaign4",
        "date": "2026-04-30",
        "data_dir": display_path(data_dir),
        "reports_loaded": [report.get("_source_file") for report in reports],
        "rows": [summarize_row(row) for row in rows],
        "strategy_decisions": strategy_decisions(rows),
        "generic_ab": generic_ab_summary(rows),
        "terminal_statuses": rows[0].get("campaign4_terminal_statuses", {}) if rows else {},
        "profiler_artifacts": profiler_artifacts(data_dir),
        "plots": {
            "simplify_hardening": display_path(plot_dir / SIMPLIFY_PLOT),
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

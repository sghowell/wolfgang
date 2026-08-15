#!/usr/bin/env python3
"""Render H100 campaign-3 benchmark plots from checked-in JSON evidence."""

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
    "auto": "#7c3aed",
    "transfer": "#2563eb",
    "resident": "#059669",
    "prealloc": "#0f766e",
    "external": "#d97706",
    "estimate": "#0891b2",
    "red": "#dc2626",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def stable_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


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
    if value >= 100.0:
        return f"{value:.0f}x"
    if value >= 10.0:
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


def case_key(case: dict[str, Any]) -> tuple[str, str]:
    return str(case["name"]), str(case["scale"])


def cases_by_key(report: dict[str, Any] | None) -> dict[tuple[str, str], dict[str, Any]]:
    if report is None:
        return {}
    return {case_key(case): case for case in report.get("cases", [])}


def timing(result: dict[str, Any], key: str) -> float | None:
    value = result.get(key)
    if value is None:
        return None
    return float(value)


def compare_reports(
    baseline: dict[str, Any] | None,
    experiment: dict[str, Any] | None,
    *,
    profile: str,
) -> list[dict[str, Any]]:
    baseline_cases = cases_by_key(baseline)
    experiment_cases = cases_by_key(experiment)
    rows: list[dict[str, Any]] = []
    for key, base_case in sorted(baseline_cases.items()):
        exp_case = experiment_cases.get(key)
        if exp_case is None:
            continue
        base_results = base_case["results"]
        exp_results = exp_case["results"]
        for boundary, result_key in (
            ("transfer_inclusive", "cuda_transfer_inclusive_seconds"),
            ("device_resident", "cuda_device_resident_seconds"),
        ):
            base_seconds = timing(base_results, result_key)
            exp_seconds = timing(exp_results, result_key)
            if base_seconds is None or exp_seconds is None or exp_seconds == 0.0:
                continue
            rows.append(
                {
                    "profile": profile,
                    "operation": key[0],
                    "scale": key[1],
                    "boundary": boundary,
                    "baseline_seconds": base_seconds,
                    "experiment_seconds": exp_seconds,
                    "speedup": base_seconds / exp_seconds,
                }
            )
    return rows


def path(name: str, seconds: float | None, kind: str) -> dict[str, Any] | None:
    if seconds is None:
        return None
    return {"name": name, "seconds": float(seconds), "kind": kind}


def selected_scaling_cases(default_report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if default_report is None:
        return []
    wanted = {
        "simplify_duplicate_pressure": "terms_50000",
        "statevector_expectation": "qubits_14_terms_4096",
        "pairwise_commutation": "terms_2048x2048",
        "matmul_product_generation_simplify": "terms_512x512",
    }
    selected: list[dict[str, Any]] = []
    by_name = {case["name"]: [] for case in default_report.get("cases", [])}
    for case in default_report.get("cases", []):
        by_name.setdefault(case["name"], []).append(case)
    for operation, scale in wanted.items():
        matches = [case for case in by_name.get(operation, []) if case.get("scale") == scale]
        if matches:
            selected.append(matches[0])
        elif by_name.get(operation):
            selected.append(by_name[operation][0])
    return selected


def scaling_paths(case: dict[str, Any]) -> list[dict[str, Any]]:
    results = case["results"]
    items = [
        path("FastPauli CPU scalar", timing(results, "cpu_scalar_seconds"), "scalar"),
        path("FastPauli CPU auto", timing(results, "cpu_default_seconds"), "auto"),
        path("FastPauli CUDA transfer", timing(results, "cuda_transfer_inclusive_seconds"), "transfer"),
        path("FastPauli CUDA resident", timing(results, "cuda_device_resident_seconds"), "resident"),
        path(
            "FastPauli CUDA preallocated",
            timing(results, "cuda_device_resident_preallocated_seconds"),
            "prealloc",
        ),
    ]
    for selector, selector_timing in sorted(results.get("cpu_optimized_timings", {}).items()):
        items.append(path(f"FastPauli CPU {selector}", timing(selector_timing, "seconds"), "auto"))
    return [item for item in items if item is not None]


def competitive_paths(case: dict[str, Any]) -> list[dict[str, Any]]:
    results = case["results"]
    items = [
        path("FastPauli CPU scalar", timing(results, "fastpauli_scalar_seconds"), "scalar"),
        path(
            "FastPauli CUDA host psi",
            timing(results, "fastpauli_cuda_operator_resident_host_statevector_seconds"),
            "transfer",
        ),
        path(
            "FastPauli CUDA device psi",
            timing(results, "fastpauli_cuda_device_resident_seconds"),
            "resident",
        ),
        path("External resident", timing(results, "competitor_seconds"), "external"),
        path(
            "External transfer",
            timing(results, "competitor_transfer_inclusive_seconds"),
            "external",
        ),
    ]
    return [item for item in items if item is not None]


def build_cross_comparison(
    default_report: dict[str, Any] | None,
    competitive_report: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in selected_scaling_cases(default_report):
        rows.append(
            {
                "case": f"{case['name'].replace('_', ' ')} {case['scale']}",
                "source": "cuda_scaling_default",
                "paths": scaling_paths(case),
            }
        )
    if competitive_report is not None:
        for case in competitive_report.get("cases", []):
            if case["name"] == "cuquantum_statevector_expectation":
                rows.append(
                    {
                        "case": "cuStateVec mapped statevector expectation",
                        "source": "competitive_baselines",
                        "paths": competitive_paths(case),
                    }
                )
            elif case["name"] in {"simplify", "multiply", "qiskit_grouping"}:
                results = case["results"]
                competitor_seconds = timing(results, "competitor_seconds")
                rows.append(
                    {
                        "case": f"{case['name'].replace('_', ' ')} external CPU baseline",
                        "source": "competitive_baselines",
                        "paths": [
                            item
                            for item in (
                                path(
                                    "FastPauli CPU scalar",
                                    timing(results, "fastpauli_scalar_seconds"),
                                    "scalar",
                                ),
                                path("External CPU", competitor_seconds, "external"),
                            )
                            if item is not None
                        ],
                    }
                )
    return [row for row in rows if len(row["paths"]) >= 2]


def materialization_rows(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if report is None:
        return []
    rows: list[dict[str, Any]] = []
    for case in report.get("cases", []):
        storage = case.get("instrumentation", {}).get("temporary_storage_bytes", {})
        if not storage.get("available"):
            continue
        rows.append(
            {
                "operation": case["name"],
                "scale": case["scale"],
                "implementation_path": storage.get("implementation_path"),
                "estimated_bytes": storage.get("estimated_bytes"),
                "result_materialization": case.get("instrumentation", {}).get(
                    "result_materialization"
                ),
            }
        )
    return rows


def privileged_ncu_inventory(raw_dir: Path) -> dict[str, Any]:
    inventory_path = raw_dir.parent / "metadata" / "privileged_ncu_inventory.txt"
    if not inventory_path.exists():
        return {
            "available": False,
            "report_files": [],
            "stdout_files": [],
            "inventory_path": stable_path(inventory_path),
        }

    report_files: list[dict[str, Any]] = []
    stdout_files: list[dict[str, Any]] = []
    for line in inventory_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        filename = parts[0]
        try:
            size_bytes = int(parts[1])
        except ValueError:
            continue
        entry = {"filename": filename, "size_bytes": size_bytes}
        if filename.endswith(".ncu-rep"):
            report_files.append(entry)
        elif filename.endswith(".stdout"):
            stdout_files.append(entry)

    return {
        "available": bool(report_files),
        "report_files": sorted(report_files, key=lambda item: item["filename"]),
        "stdout_files": sorted(stdout_files, key=lambda item: item["filename"]),
        "inventory_path": stable_path(inventory_path),
    }


def ncu_report_filename_for_step(step_name: str) -> str | None:
    if not step_name.startswith("ncu ") or not step_name.endswith(" detailed"):
        return None
    hot_path = step_name.removeprefix("ncu ").removesuffix(" detailed").replace(" ", "_")
    return f"ncu_{hot_path}_detailed.ncu-rep"


def normalize_profile_status(
    profile_report: dict[str, Any] | None,
    privileged_ncu: dict[str, Any],
) -> list[dict[str, Any]]:
    if profile_report is None:
        return []

    privileged_reports = {
        str(item["filename"]): item
        for item in privileged_ncu.get("report_files", [])
    }
    normalized: list[dict[str, Any]] = []
    for item in profile_report.get("results", []):
        status = dict(item)
        status["effective_status"] = status.get("status")
        expected_report = ncu_report_filename_for_step(str(status.get("name", "")))
        if (
            expected_report is not None
            and status.get("status") != "success"
            and expected_report in privileged_reports
        ):
            status["status"] = "expected_permission_denied_superseded"
            status["effective_status"] = "success"
            status["superseded_by_privileged_ncu_report"] = expected_report
            status["superseded_by_privileged_ncu_report_bytes"] = privileged_reports[
                expected_report
            ]["size_bytes"]
            status["nonprivileged_failure_reason"] = "ERR_NVGPUCTRPERM"
        normalized.append(status)
    return normalized


def profiler_evidence_status(profile_status: list[dict[str, Any]], privileged_ncu: dict[str, Any]) -> dict[str, Any]:
    nonprivileged_ncu = [
        item
        for item in profile_status
        if ncu_report_filename_for_step(str(item.get("name", ""))) is not None
    ]
    expected_permission_denied = [
        item["name"]
        for item in nonprivileged_ncu
        if item.get("status") == "expected_permission_denied_superseded"
    ]
    required_effective_failures = [
        item["name"]
        for item in profile_status
        if item.get("required") and item.get("effective_status", item.get("status")) != "success"
    ]
    return {
        "required_effective_failures": required_effective_failures,
        "nonprivileged_ncu": {
            "status": (
                "expected_permission_denied"
                if expected_permission_denied
                else ("success" if nonprivileged_ncu else "not_run")
            ),
            "superseded_steps": expected_permission_denied,
        },
        "privileged_ncu": {
            "status": "success" if privileged_ncu.get("available") else "not_available",
            "report_count": len(privileged_ncu.get("report_files", [])),
            "reports": [
                item["filename"]
                for item in privileged_ncu.get("report_files", [])
            ],
        },
    }


def build_summary(raw_dir: Path) -> dict[str, Any]:
    baseline_stress = load_optional_json(raw_dir / "baseline_cuda_scaling_stress.json")
    baseline_extreme = load_optional_json(raw_dir / "baseline_cuda_scaling_extreme.json")
    experiment_default = load_optional_json(raw_dir / "experiment_cuda_scaling_default.json")
    experiment_stress = load_optional_json(raw_dir / "experiment_cuda_scaling_stress.json")
    experiment_extreme = load_optional_json(raw_dir / "experiment_cuda_scaling_extreme.json")
    experiment_materialization = load_optional_json(
        raw_dir / "experiment_cuda_scaling_materialization.json"
    )
    competitive = load_optional_json(raw_dir / "competitive_baselines_final.json")
    profile_report = load_optional_json(raw_dir / "experiment_profile_report.json")
    privileged_ncu = privileged_ncu_inventory(raw_dir)
    profile_status = normalize_profile_status(profile_report, privileged_ncu)
    comparisons = [
        *compare_reports(baseline_stress, experiment_stress, profile="stress"),
        *compare_reports(baseline_extreme, experiment_extreme, profile="extreme"),
    ]
    return {
        "campaign": "h100_campaign3",
        "raw_files": sorted(path.name for path in raw_dir.glob("*.json")),
        "provenance": {
            "baseline_git_commit": None
            if baseline_stress is None
            else baseline_stress.get("git_commit"),
            "experiment_git_commit": None
            if experiment_default is None
            else experiment_default.get("git_commit"),
        },
        "baseline_vs_experiment": comparisons,
        "cross_comparison": build_cross_comparison(experiment_default, competitive),
        "materialization_boundaries": materialization_rows(experiment_materialization),
        "competitors": {} if competitive is None else competitive.get("competitors", {}),
        "profile_status": profile_status,
        "profiler_evidence_status": profiler_evidence_status(profile_status, privileged_ncu),
        "privileged_ncu": privileged_ncu,
    }


def render_cross_comparison(summary: dict[str, Any], output: Path) -> None:
    rows = summary["cross_comparison"]
    width = 1260
    left = 360
    bar_width = 620
    top = 116
    row_gap = 20
    section_gap = 28
    max_speedup = 1.0
    for row in rows:
        scalar = row["paths"][0]["seconds"]
        for item in row["paths"]:
            if item["seconds"] > 0:
                max_speedup = max(max_speedup, scalar / item["seconds"])
    height = top + sum(30 + len(row["paths"]) * row_gap + section_gap for row in rows) + 24
    lines = svg_header(
        width,
        height,
        "FastPauli H100 campaign 3 CPU CUDA external comparison",
        "README-facing cross-comparison across CPU, CUDA, and external baselines.",
    )
    lines.extend(
        [
            svg_text(32, 40, "H100 Campaign 3 Cross-Comparison", size=26, weight=700),
            svg_text(
                32,
                66,
                "Speedup versus the first FastPauli CPU scalar path in each group; log-scaled bars.",
                size=13,
                color=COLORS["muted"],
            ),
            svg_text(left, 96, "1.0x", size=11, color=COLORS["muted"]),
            svg_text(
                left + bar_width,
                96,
                format_ratio(max_speedup),
                size=11,
                color=COLORS["muted"],
                anchor="end",
            ),
        ]
    )
    y = top
    log_max = math.log10(max(max_speedup, 1.000001))
    for row in rows:
        scalar = row["paths"][0]["seconds"]
        lines.append(svg_text(32, y, row["case"], size=12, weight=700))
        lines.append(svg_text(32, y + 17, row["source"], size=10, color=COLORS["muted"]))
        for index, item in enumerate(row["paths"]):
            speedup = scalar / item["seconds"] if item["seconds"] > 0 else 0.0
            log_speedup = math.log10(max(speedup, 1.0))
            filled = max(2.0, bar_width * log_speedup / log_max) if log_speedup > 0 else 2.0
            color = COLORS.get(item["kind"], COLORS["scalar"])
            bar_y = y + 24 + index * row_gap
            lines.append(svg_rect(left, bar_y, filled, 12, color))
            lines.append(
                svg_text(
                    left + filled + 8,
                    bar_y + 10,
                    f"{item['name']} {format_ratio(speedup)} ({format_seconds(item['seconds'])})",
                    size=11,
                )
            )
        y += 30 + len(row["paths"]) * row_gap + section_gap
    lines.append("</svg>")
    write_svg(output, lines)


def render_ab_speedups(summary: dict[str, Any], output: Path) -> None:
    rows = [
        row
        for row in summary["baseline_vs_experiment"]
        if row["operation"] in {"simplify_duplicate_pressure", "matmul_product_generation_simplify"}
    ]
    width = 1120
    height = 130 + len(rows) * 34
    left = 360
    bar_width = 520
    max_speedup = max([1.0, *(float(row["speedup"]) for row in rows)])
    lines = svg_header(
        width,
        height,
        "FastPauli H100 campaign 3 simplify and matmul speedups",
        "Same-boundary baseline versus experiment speedups for duplicate reduction paths.",
    )
    lines.extend(
        [
            svg_text(32, 40, "Duplicate-Reduction A/B", size=25, weight=700),
            svg_text(
                32,
                64,
                "Same H100, same benchmark boundary; values above 1.0x favor Campaign 3.",
                size=13,
                color=COLORS["muted"],
            ),
        ]
    )
    for index, row in enumerate(rows):
        y = 100 + index * 34
        speedup = float(row["speedup"])
        color = COLORS["resident"] if speedup >= 1.0 else COLORS["red"]
        filled = max(2.0, bar_width * min(speedup, max_speedup) / max_speedup)
        label = f"{row['profile']} {row['operation'].replace('_', ' ')} {row['scale']} {row['boundary']}"
        lines.append(svg_text(32, y + 10, label, size=10, weight=700))
        lines.append(svg_rect(left, y, filled, 12, color))
        lines.append(svg_text(left + filled + 8, y + 10, format_ratio(speedup), size=11))
    lines.append("</svg>")
    write_svg(output, lines)


def render_materialization(summary: dict[str, Any], output: Path) -> None:
    rows = summary["materialization_boundaries"]
    width = 1120
    height = 130 + len(rows) * 32
    left = 420
    bar_width = 500
    max_bytes = max([1, *(int(row["estimated_bytes"] or 0) for row in rows)])
    lines = svg_header(
        width,
        height,
        "FastPauli H100 campaign 3 materialization boundaries",
        "Estimated allocation and materialization pressure by benchmark case.",
    )
    lines.extend(
        [
            svg_text(32, 40, "Allocation And Materialization Boundaries", size=25, weight=700),
            svg_text(
                32,
                64,
                "Static byte estimates from checked benchmark shapes and implementation paths.",
                size=13,
                color=COLORS["muted"],
            ),
        ]
    )
    for index, row in enumerate(rows):
        y = 100 + index * 32
        estimated = int(row["estimated_bytes"] or 0)
        filled = max(2.0, bar_width * math.log10(max(estimated, 1)) / math.log10(max_bytes))
        label = f"{row['operation'].replace('_', ' ')} {row['scale']}"
        lines.append(svg_text(32, y + 10, label, size=10, weight=700))
        lines.append(svg_rect(left, y, filled, 12, COLORS["estimate"]))
        lines.append(svg_text(left + filled + 8, y + 10, f"{estimated / 1.0e6:.1f} MB", size=11))
    lines.append("</svg>")
    write_svg(output, lines)


def render_evidence(summary: dict[str, Any], output: Path) -> None:
    statuses = summary["profile_status"]
    success = sum(
        1 for item in statuses if item.get("effective_status", item.get("status")) == "success"
    )
    failed = sum(
        1
        for item in statuses
        if item.get("effective_status", item.get("status")) not in {"success", "missing_executable"}
    )
    missing = sum(1 for item in statuses if item.get("status") == "missing_executable")
    privileged_ncu_reports = len(summary.get("privileged_ncu", {}).get("report_files", []))
    required_failures = summary.get("profiler_evidence_status", {}).get(
        "required_effective_failures",
        [],
    )
    profiler_status = "success" if not required_failures else "warning"
    profiler_detail = (
        f"{success} success, {privileged_ncu_reports} privileged NCU reports"
        if privileged_ncu_reports
        else f"{success} success, {failed} failed, {missing} missing"
    )
    cards = [
        ("Raw JSON files", "success", f"{len(summary['raw_files'])} checked evidence files"),
        ("A/B comparisons", "success", f"{len(summary['baseline_vs_experiment'])} same-key rows"),
        ("Cross comparison", "success", f"{len(summary['cross_comparison'])} README candidate groups"),
        ("Profiler ladder", profiler_status, profiler_detail),
    ]
    width = 1120
    height = 300
    lines = svg_header(
        width,
        height,
        "FastPauli H100 campaign 3 evidence status",
        "Validation, profiling, and benchmark evidence status.",
    )
    lines.extend(
        [
            svg_text(32, 40, "Campaign 3 Evidence Status", size=25, weight=700),
            svg_text(
                32,
                64,
                "Summary generated from checked raw benchmark and profiler reports.",
                size=13,
                color=COLORS["muted"],
            ),
        ]
    )
    for index, (title, status, detail) in enumerate(cards):
        x = 32 + index * 270
        y = 110
        color = COLORS["resident"] if status == "success" else COLORS["external"]
        lines.append(svg_rect(x, y, 240, 76, COLORS["panel"], rx=6))
        lines.append(svg_rect(x + 16, y + 18, 12, 12, color, rx=6))
        lines.append(svg_text(x + 40, y + 28, title, size=12, weight=700))
        lines.append(svg_text(x + 40, y + 52, detail, size=11, color=COLORS["muted"]))
    lines.append("</svg>")
    write_svg(output, lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary = build_summary(args.raw_dir)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    render_cross_comparison(
        summary,
        args.plot_dir / "cuda_h100_campaign3_readme_cross_comparison.svg",
    )
    render_ab_speedups(
        summary,
        args.plot_dir / "cuda_h100_campaign3_duplicate_reduction_speedups.svg",
    )
    render_materialization(
        summary,
        args.plot_dir / "cuda_h100_campaign3_materialization_boundaries.svg",
    )
    render_evidence(
        summary,
        args.plot_dir / "cuda_h100_campaign3_evidence_status.svg",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Render a concise Markdown CPU benchmark evidence report from JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def as_list(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "none"
    if isinstance(value, dict):
        return ", ".join(f"{key}={item}" for key, item in sorted(value.items())) or "none"
    if value is None:
        return "not_recorded"
    return str(value)


def seconds(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    return "not_recorded"


def table_cell(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def compact_dataset(dataset: dict[str, Any]) -> str:
    preferred_keys = (
        "num_qubits",
        "num_terms",
        "lhs_terms",
        "rhs_terms",
        "term_weight",
        "term_weight_distribution",
        "duplicate_rate",
        "lhs_duplicate_rate",
        "rhs_duplicate_rate",
        "matrix_entries",
        "statevector_length",
        "grouping_mode",
        "strategy",
        "random_seed",
        "operator_random_seed",
        "statevector_random_seed",
        "counts_random_seed",
        "coefficient_dtype",
        "competitor",
    )
    parts = []
    for key in preferred_keys:
        if key in dataset:
            parts.append(f"{key}={as_list(dataset[key])}")
    return table_cell("; ".join(parts) if parts else "not_recorded")


def optimized_timing_summary(optimized: Any) -> str:
    if not isinstance(optimized, dict) or not optimized:
        return "none"

    parts = []
    for backend, timings in sorted(optimized.items()):
        if isinstance(timings, dict):
            parts.append(f"{backend}={seconds(timings.get('seconds'))}")
        else:
            parts.append(str(backend))
    return table_cell(", ".join(parts))


def dispatch_rows(report: dict[str, Any]) -> list[str]:
    rows = [
        "| Case | Dataset | Matrix Entries | Backend Hint | Seconds | Correct |",
        "| --- | --- | ---: | --- | ---: | --- |",
    ]
    for case in report.get("cases", []):
        dataset = case.get("dataset", {})
        results = case.get("results", {})
        rows.append(
            "| {name} | {dataset} | {entries} | {hint} | {time} | {correct} |".format(
                name=table_cell(case.get("name", "unknown")),
                dataset=compact_dataset(dataset),
                entries=dataset.get("matrix_entries", "n/a"),
                hint=table_cell(
                    dataset.get("effective_backend_hint", dataset.get("active_cpu_backend", "n/a"))
                ),
                time=seconds(results.get("fastpauli_seconds")),
                correct=results.get("matches_forced_scalar", "n/a"),
            )
        )
    return rows


def threshold_rows(report: dict[str, Any]) -> list[str]:
    rows = [
        "| Case | Dataset | Entries | Region | Auto Hint | Scalar Seconds | Auto Seconds | Optimized Seconds | Correct |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |",
    ]
    for case in report.get("cases", []):
        dataset = case.get("dataset", {})
        results = case.get("results", {})
        optimized = results.get("optimized_backends", {})
        rows.append(
            "| {name} | {dataset} | {entries} | {region} | {hint} | {scalar} | {auto} | {optimized} | {correct} |".format(
                name=table_cell(case.get("name", "unknown")),
                dataset=compact_dataset(dataset),
                entries=dataset.get("matrix_entries", "n/a"),
                region=table_cell(dataset.get("threshold_region", "n/a")),
                hint=table_cell(dataset.get("auto_effective_backend_hint", "n/a")),
                scalar=seconds(results.get("scalar_seconds")),
                auto=seconds(results.get("auto_seconds")),
                optimized=optimized_timing_summary(optimized),
                correct=results.get("matches_forced_scalar", "n/a"),
            )
        )
    return rows


def hardening_rows(reports: list[dict[str, Any]]) -> list[str]:
    rows = [
        "| Profile | Operation | Case | Dataset | FastPauli Seconds | Baseline Seconds | Correctness Checked |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for report in reports:
        profile = report.get("profile", "unknown")
        for operation in report.get("operations", []):
            for case in operation.get("cases", []):
                rows.append(
                    "| {profile} | {operation} | {case} | {dataset} | {fastpauli} | {baseline} | {checked} |".format(
                        profile=table_cell(profile),
                        operation=table_cell(operation.get("benchmark", "unknown")),
                        case=table_cell(case.get("name", "unknown")),
                        dataset=compact_dataset(case.get("dataset", {})),
                        fastpauli=seconds(case.get("fastpauli_scalar_seconds")),
                        baseline=seconds(case.get("python_baseline_seconds")),
                        checked=operation.get("correctness_checked", "n/a"),
                    )
                )
    return rows


def competitive_rows(report: dict[str, Any] | None) -> list[str]:
    if report is None:
        return ["not provided"]
    rows = [
        "| Case | Dataset | Competitor | Competitor Available | FastPauli Seconds | Competitor Seconds | Correctness Checked |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for case in report.get("cases", []):
        dataset = case.get("dataset", {})
        results = case.get("results", {})
        rows.append(
            "| {case} | {dataset} | {competitor_name} | {available} | {fastpauli} | {competitor} | {checked} |".format(
                case=table_cell(case.get("name", "unknown")),
                dataset=compact_dataset(dataset),
                competitor_name=table_cell(dataset.get("competitor", "not_recorded")),
                available=results.get("competitor_available", "n/a"),
                fastpauli=seconds(results.get("fastpauli_scalar_seconds")),
                competitor=seconds(results.get("competitor_seconds")),
                checked=results.get("competitor_correctness_checked", "n/a"),
            )
        )
    return rows


def render_report(
    *,
    title: str,
    dispatch: dict[str, Any],
    hardening_reports: list[dict[str, Any]],
    thresholds: dict[str, Any],
    competitive: dict[str, Any] | None,
    notes: str,
) -> str:
    environment = dispatch.get("environment", {})
    build_info = dispatch.get("fastpauli_build_info", {})
    threshold_values = thresholds.get("thresholds", {})
    optimized_kernels = build_info.get("optimized_cpu_kernels", {})

    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        f"- Git commit: `{dispatch.get('git_commit', 'unknown')}`",
        f"- CPU: {environment.get('cpu_vendor_or_soc', 'not_recorded')}",
        f"- Architecture: {environment.get('cpu_architecture', 'not_recorded')}",
        f"- OS: {environment.get('operating_system', 'not_recorded')}",
        f"- Compiler: {environment.get('compiler', 'not_recorded')}",
        f"- Compiled CPU backends: {as_list(environment.get('compiled_fastpauli_cpu_backends'))}",
        f"- Available CPU backends: {as_list(environment.get('available_fastpauli_cpu_backends'))}",
        f"- Unavailable CPU backends: {as_list(environment.get('unavailable_fastpauli_cpu_backends'))}",
        f"- oneTBB: {as_list(environment.get('oneTBB'))}",
        f"- Thread settings: {as_list(environment.get('thread_settings'))}",
        f"- Auto-dispatch thresholds: {as_list(threshold_values)}",
        "",
        "## Optimized Kernel Coverage",
        "",
    ]
    apple_hardware = environment.get("apple_hardware")
    if isinstance(apple_hardware, dict):
        lines.insert(10, f"- Apple hardware: {as_list(apple_hardware)}")

    for backend, kernels in sorted(optimized_kernels.items()):
        lines.append(f"- {backend}: {as_list(kernels)}")

    lines.extend(
        [
            "",
            "## Commands",
            "",
            f"- Dispatch: `{dispatch.get('command', 'not_recorded')}`",
            f"- Thresholds: `{thresholds.get('command', 'not_recorded')}`",
        ]
    )
    for hardening in hardening_reports:
        lines.append(
            f"- Hardening {hardening.get('profile', 'unknown')}: "
            f"`{hardening.get('command', 'not_recorded')}`"
        )
    if competitive is not None:
        lines.append(f"- Competitive: `{competitive.get('command', 'not_recorded')}`")

    lines.extend(
        [
            "",
            "## Dispatch Benchmark",
            "",
            *dispatch_rows(dispatch),
            "",
            "## Threshold Characterization",
            "",
            *threshold_rows(thresholds),
            "",
            "## CPU Hardening",
            "",
            *hardening_rows(hardening_reports),
            "",
            "## Competitive Baselines",
            "",
            *competitive_rows(competitive),
            "",
            "## Limitations",
            "",
            notes.strip() or "No additional limitations recorded.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", required=True)
    parser.add_argument("--dispatch-json", type=Path, required=True)
    parser.add_argument("--hardening-json", type=Path, action="append", required=True)
    parser.add_argument("--threshold-json", type=Path, required=True)
    parser.add_argument("--competitive-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rendered = render_report(
        title=args.title,
        dispatch=read_json(args.dispatch_json) or {},
        hardening_reports=[
            report for report in (read_json(path) for path in args.hardening_json) if report is not None
        ],
        thresholds=read_json(args.threshold_json) or {},
        competitive=read_json(args.competitive_json),
        notes=args.notes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

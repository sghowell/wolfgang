#!/usr/bin/env python3
"""Render summary evidence and plots for ROCm MI300X Campaign 5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from render_rocm_campaign3_assets import (
    display_path,
    read_json,
    render_bar_svg,
    timing_value,
    write_text,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "docs/benchmarks/data/rocm_mi300x_campaign5_2026-04-30"
DEFAULT_PLOT_DIR = ROOT / "docs/benchmarks/plots"
ROCM_CAMPAIGN4_SUMMARY = (
    ROOT / "docs/benchmarks/data/rocm_mi300x_campaign4_2026-04-30/summary.json"
)

INTEROP_PLOT = "rocm_mi300x_campaign5_interop.svg"
LANDSCAPE_PLOT = "accelerator_landscape_with_rocm.svg"

TERMINAL_ITEMS = {
    "DLPack",
    "CUDA Array Interface guard",
    "streams",
    "graphs",
    "workspaces",
    "expectation",
    "matmul",
    "portability",
    "ROCm wheels",
    "multi-GPU",
    "simultaneous CUDA+HIP",
}

DLPACK_CONSUMER_FIELDS = {
    "consumer_library",
    "consumer_version",
    "consumer_backend",
    "consumer_available",
    "consumer_import_error",
    "consumer_correctness_passed",
    "consumer_read_only_enforced",
    "consumer_mutation_error",
}


def load_candidate_probe_artifacts(data_dir: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted((data_dir / "raw").glob("*candidate*probe*.json")):
        payload = read_json(path)
        if payload.get("artifact_type") != "candidate_dlpack_probe":
            continue
        payload["_source_file"] = display_path(path)
        artifacts.append(payload)
    return artifacts


def load_raw_reports(data_dir: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in sorted((data_dir / "raw").glob("*.json")):
        payload = read_json(path)
        payload["_source_file"] = display_path(path)
        reports.append(payload)
    return reports


def flatten_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in reports:
        for row in report.get("cases", []):
            if row.get("campaign") != "rocm_mi300x_campaign5":
                continue
            item = dict(row)
            item["report_profile"] = report.get("profile", row.get("profile", "unknown"))
            item["source_file"] = report.get("_source_file", "unknown")
            item["git_commit"] = report.get("git_commit")
            rows.append(item)
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Campaign 5 summary has no rows")

    terminal_coverage: set[str] = set()
    guard_retained = False
    for row in rows:
        if "final_status" not in row:
            raise ValueError(f"Campaign 5 row omits final_status: {row.get('source_file')}")
        terminal_coverage.update(row.get("campaign5_terminal_statuses", {}))

        if str(row.get("mode", "")).startswith("dlpack_"):
            missing = sorted(field for field in DLPACK_CONSUMER_FIELDS if field not in row)
            if missing:
                raise ValueError(f"DLPack row omits {missing}: {row.get('source_file')}")
            if row.get("final_status") == "retained":
                if row.get("hip_dlpack_device_type") != 10:
                    raise ValueError("retained HIP DLPack row must report device type 10")
                if row.get("hip_dlpack_device_type_name") != "kDLROCM":
                    raise ValueError("retained HIP DLPack row must report kDLROCM")
                if row.get("consumer_read_only_enforced") is not True:
                    raise ValueError("retained HIP DLPack row must enforce read-only consumer")

        if row.get("mode") == "cuda_array_interface_guard" and row.get("final_status") == "retained":
            guard_retained = True

    if guard_retained:
        raise ValueError("HIP CUDA Array Interface guard must not be retained")

    missing_terminal = TERMINAL_ITEMS - terminal_coverage
    if missing_terminal:
        raise ValueError(f"Campaign 5 rows omit terminal statuses: {sorted(missing_terminal)}")


def validate_candidate_artifacts(artifacts: list[dict[str, Any]]) -> None:
    if not artifacts:
        raise ValueError("Campaign 5 DLPack rejection requires a candidate probe artifact")
    torch_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.get("consumer_library") == "torch"
        and artifact.get("consumer_backend") == "rocm"
    ]
    if not torch_artifacts:
        raise ValueError("Campaign 5 candidate artifacts omit a PyTorch ROCm probe")
    for artifact in torch_artifacts:
        if artifact.get("candidate_probe_consumer_correctness_passed") is not True:
            raise ValueError("PyTorch ROCm candidate probe must pass correctness before mutation")
        if artifact.get("candidate_probe_consumer_read_only_enforced") is not False:
            raise ValueError("PyTorch ROCm candidate probe must record failed read-only enforcement")
        if artifact.get("candidate_probe_mutation_result") != "accepted_mutation":
            raise ValueError("PyTorch ROCm candidate probe must record accepted_mutation")
        if not artifact.get("temporary_candidate_patch"):
            raise ValueError("candidate probe artifact must describe the temporary candidate patch")


def summarize_row(row: dict[str, Any]) -> dict[str, Any]:
    dataset = row.get("dataset", {})
    return {
        "profile": row.get("report_profile", row.get("profile")),
        "case": row.get("case"),
        "operation": row.get("operation"),
        "mode": row.get("mode"),
        "status": row.get("status"),
        "final_status": row.get("final_status"),
        "timing_boundary": row.get("timing_boundary"),
        "git_commit": row.get("git_commit"),
        "num_qubits": dataset.get("num_qubits"),
        "lhs_terms": dataset.get("lhs_terms"),
        "rhs_terms": dataset.get("rhs_terms"),
        "entries": dataset.get("entries"),
        "device_name": row.get("device_name"),
        "gfx_target": row.get("gfx_target"),
        "consumer_library": row.get("consumer_library"),
        "consumer_version": row.get("consumer_version"),
        "consumer_backend": row.get("consumer_backend"),
        "consumer_available": row.get("consumer_available"),
        "consumer_import_error": row.get("consumer_import_error"),
        "consumer_correctness_passed": row.get("consumer_correctness_passed"),
        "consumer_read_only_enforced": row.get("consumer_read_only_enforced"),
        "consumer_mutation_error": row.get("consumer_mutation_error"),
        "candidate_probe_evidence_kind": row.get("candidate_probe_evidence_kind"),
        "candidate_probe_source_file": row.get("candidate_probe_source_file"),
        "candidate_probe_mutation_result": row.get("candidate_probe_mutation_result"),
        "hip_dlpack_device_type": row.get("hip_dlpack_device_type"),
        "hip_dlpack_device_type_name": row.get("hip_dlpack_device_type_name"),
        "dlpack_unavailable_error": row.get("dlpack_unavailable_error"),
        "cuda_array_interface_error": row.get("cuda_array_interface_error"),
        "hip_device_output_to_host_seconds": row.get("hip_device_output_to_host_seconds"),
        "hip_count_commuting_axis_none_seconds": row.get(
            "hip_count_commuting_axis_none_seconds"
        ),
        "correctness_digest": row.get("correctness_digest"),
        "decision_item": row.get("decision_item"),
        "decision_reason": row.get("decision_reason"),
        "retention_decision": row.get("retention_decision"),
        "source_file": row.get("source_file"),
    }


def profiler_artifacts(data_dir: Path) -> list[str]:
    profiler = data_dir / "profiler"
    if not profiler.exists():
        return []
    return [display_path(path) for path in sorted(profiler.rglob("*")) if path.is_file()]


def status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("final_status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def consumer_summary(
    rows: list[dict[str, Any]],
    candidate_artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_consumer: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not str(row.get("mode", "")).startswith("dlpack_"):
            continue
        library = str(row.get("consumer_library"))
        if library in by_consumer and by_consumer[library].get("consumer_available"):
            continue
        by_consumer[library] = {
            "consumer_library": row.get("consumer_library"),
            "consumer_version": row.get("consumer_version"),
            "consumer_backend": row.get("consumer_backend"),
            "consumer_available": row.get("consumer_available"),
            "consumer_import_error": row.get("consumer_import_error"),
            "final_status": row.get("final_status"),
            "consumer_read_only_enforced": row.get("consumer_read_only_enforced"),
            "candidate_probe_mutation_result": row.get("candidate_probe_mutation_result"),
            "candidate_probe_evidence_kind": row.get("candidate_probe_evidence_kind"),
            "candidate_probe_source_file": row.get("candidate_probe_source_file"),
            "candidate_probe_consumer_correctness_passed": row.get(
                "candidate_probe_consumer_correctness_passed"
            ),
            "candidate_probe_consumer_read_only_enforced": row.get(
                "candidate_probe_consumer_read_only_enforced"
            ),
        }
    for artifact in candidate_artifacts:
        library = str(artifact.get("consumer_library"))
        if library == "None":
            continue
        entry = by_consumer.setdefault(
            library,
            {
                "consumer_library": artifact.get("consumer_library"),
                "consumer_version": artifact.get("consumer_version"),
                "consumer_backend": artifact.get("consumer_backend"),
                "consumer_available": artifact.get("consumer_available"),
                "consumer_import_error": artifact.get("consumer_import_error", ""),
                "final_status": artifact.get("final_status", "rejected_with_evidence"),
                "consumer_read_only_enforced": artifact.get(
                    "candidate_probe_consumer_read_only_enforced"
                ),
            },
        )
        entry["candidate_probe_evidence_kind"] = artifact.get("evidence_kind")
        entry["candidate_probe_source_file"] = artifact.get("_source_file")
        entry["candidate_probe_git_commit"] = artifact.get("git_commit")
        entry["candidate_probe_command"] = artifact.get("command")
        entry["candidate_probe_consumer_correctness_passed"] = artifact.get(
            "candidate_probe_consumer_correctness_passed"
        )
        entry["candidate_probe_consumer_read_only_enforced"] = artifact.get(
            "candidate_probe_consumer_read_only_enforced"
        )
        entry["candidate_probe_mutation_result"] = artifact.get(
            "candidate_probe_mutation_result"
        )
    return [by_consumer[key] for key in sorted(by_consumer)]


def terminal_statuses(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        statuses = row.get("campaign5_terminal_statuses")
        if isinstance(statuses, dict):
            return statuses
    return {}


def interop_plot_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if row.get("operation") != "commutation_interop":
            continue
        case = str(row.get("case"))
        mode = str(row.get("mode"))
        for series, key, color in (
            ("HIP to_host", "hip_device_output_to_host_seconds", "#2563eb"),
            ("HIP compact count", "hip_count_commuting_axis_none_seconds", "#0f766e"),
        ):
            marker = (case, mode, series)
            if marker in seen:
                continue
            seconds = timing_value(row, key)
            if seconds is None:
                continue
            seen.add(marker)
            points.append(
                {
                    "label": f"{case} | {mode} | {series}",
                    "series": series,
                    "seconds": seconds,
                    "color": color,
                }
            )
    return points


def previous_landscape_points() -> list[dict[str, Any]]:
    if not ROCM_CAMPAIGN4_SUMMARY.exists():
        return []
    summary = read_json(ROCM_CAMPAIGN4_SUMMARY)
    points: list[dict[str, Any]] = []
    for point in summary.get("readme_performance_landscape", []):
        seconds = timing_value(point, "seconds")
        if seconds is None:
            continue
        points.append(
            {
                "label": str(point.get("label", point.get("series", "unknown"))),
                "series": str(point.get("series", "unknown")),
                "seconds": seconds,
                "color": str(point.get("color", "#64748b")),
            }
        )
    return points


def campaign5_retained_landscape_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in rows:
        if row.get("final_status") != "retained":
            continue
        seconds = timing_value(row, "consumer_sum_seconds")
        if seconds is None:
            continue
        points.append(
            {
                "label": f"ROCm HIP DLPack | {row.get('consumer_library')}",
                "series": "ROCm HIP DLPack",
                "seconds": seconds,
                "color": "#be123c",
            }
        )
    return points


def build_summary(data_dir: Path, plot_dir: Path) -> dict[str, Any]:
    reports = load_raw_reports(data_dir)
    candidate_artifacts = load_candidate_probe_artifacts(data_dir)
    rows = flatten_rows(reports)
    validate_rows(rows)
    validate_candidate_artifacts(candidate_artifacts)
    plot_dir.mkdir(parents=True, exist_ok=True)

    render_bar_svg(
        title="MI300X ROCm Campaign 5 Interop Boundary",
        subtitle="Public HIP DLPack rejected; retained dense to_host and compact count timings shown",
        points=interop_plot_points(rows),
        output=plot_dir / INTEROP_PLOT,
    )

    landscape = previous_landscape_points()
    retained_points = campaign5_retained_landscape_points(rows)
    landscape.extend(retained_points)
    if landscape:
        render_bar_svg(
            title="FastPauli Accelerator Performance Landscape",
            subtitle="Checked CPU, CUDA, external, and ROCm evidence; log-scale seconds",
            points=landscape,
            output=plot_dir / LANDSCAPE_PLOT,
        )

    return {
        "campaign": "rocm_mi300x_campaign5",
        "date": "2026-04-30",
        "data_dir": display_path(data_dir),
        "reports_loaded": [report.get("_source_file") for report in reports],
        "rows": [summarize_row(row) for row in rows],
        "status_counts": status_counts(rows),
        "consumer_summary": consumer_summary(rows, candidate_artifacts),
        "candidate_probe_artifacts": [
            {
                "source_file": artifact.get("_source_file"),
                "git_commit": artifact.get("git_commit"),
                "evidence_kind": artifact.get("evidence_kind"),
                "consumer_library": artifact.get("consumer_library"),
                "consumer_version": artifact.get("consumer_version"),
                "consumer_backend": artifact.get("consumer_backend"),
                "candidate_probe_consumer_correctness_passed": artifact.get(
                    "candidate_probe_consumer_correctness_passed"
                ),
                "candidate_probe_consumer_read_only_enforced": artifact.get(
                    "candidate_probe_consumer_read_only_enforced"
                ),
                "candidate_probe_mutation_result": artifact.get(
                    "candidate_probe_mutation_result"
                ),
                "temporary_candidate_patch": artifact.get("temporary_candidate_patch"),
            }
            for artifact in candidate_artifacts
        ],
        "terminal_statuses": terminal_statuses(rows),
        "landscape_refreshed_with_campaign5": bool(retained_points),
        "profiler_artifacts": profiler_artifacts(data_dir),
        "plots": {
            "interop": display_path(plot_dir / INTEROP_PLOT),
            "landscape": display_path(plot_dir / LANDSCAPE_PLOT),
        },
        "readme_performance_landscape": landscape,
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

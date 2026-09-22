from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "bench_metal_kernels.py"
APPLE_ACCELERATOR = ROOT / "docs/architecture/apple_accelerator.md"
PROTOCOL = ROOT / "docs/benchmarks/protocol.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("bench_metal_kernels", BENCHMARK)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wave1d_profile_lists_small_and_large_reuse_cases() -> None:
    completed = subprocess.run(
        [sys.executable, str(BENCHMARK), "--list-cases", "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["benchmark"] == "apple_metal_kernels"
    assert "wave1d" in payload["profiles"]
    cases = payload["profiles"]["wave1d"]
    assert {case["profile"] for case in cases} == {"wave1d"}
    assert {case["operation"] for case in cases} == {"commutes_with_device"}
    assert {
        "metal_wave1d_small_rows_128x128",
        "metal_wave1d_medium_rows_512x512",
        "metal_wave1d_large_rows_2048x2048",
    } <= {case["name"] for case in cases}
    assert {case["wave1d_gate"] for case in cases} == {"small_regression_guard", "retained_reuse_gate"}



def wave1d_reports(module):
    variants = (
        ("metal_transfer_inclusive", "transfer_inclusive", 0.004),
        ("metal_device_matrix", "device_output_allocating", 0.002),
        ("metal_device_matrix_reuse", "device_output_reused", 0.0015),
    )
    return [
        {
            "status": "ok",
            "metal_status": {"runtime_available": True},
            "cases": [
                {
                    "case": module.case_with_metadata(case, repeat=7),
                    "variant": variant,
                    "timing": {"median": duration * factor},
                    "transfer_boundary": boundary,
                    "status": "ok",
                    "correct": True,
                }
                for case in module.WAVE1D_CASES
                for variant, boundary, duration in variants
            ],
        }
        for factor in (0.9, 1.0, 1.1)
    ]


def test_wave1d_complete_evidence_and_small_regression_guard() -> None:
    module = load_benchmark_module()
    reports = wave1d_reports(module)
    summary = module.summarize_wave1d_evidence(reports, repeat=7)
    assert summary["status"] == "go"
    assert len(summary["aggregated_cases"]) == 3
    assert summary["measurement_methodology"]["independent_reruns"] == 3
    assert summary["measurement_methodology"]["promotion_metric"] == "mean_of_medians_seconds"
    small_name = module.WAVE1D_CASES[0]["name"]
    for report in reports:
        for row in report["cases"]:
            if row["case"]["name"] == small_name and row["variant"] == "metal_device_matrix_reuse":
                row["timing"]["median"] = 0.0022
    summary = module.summarize_wave1d_evidence(reports, repeat=7)
    assert summary["status"] == "reject_investigate"
    assert summary["small_row_regressions"][0]["case_name"] == small_name


@pytest.mark.parametrize("problem", [
    "empty", "one_rerun", "missing_case", "missing_variant", "duplicate", "incorrect",
    "nan", "infinite", "zero", "negative", "wrong_boundary", "wrong_shape", "wrong_repeat",
    "failed_row", "failed_report", "missing_runtime",
])
def test_wave1d_invalid_evidence_cannot_promote(problem: str) -> None:
    module = load_benchmark_module()
    reports = wave1d_reports(module)
    first = reports[0]["cases"][0]
    if problem == "empty":
        reports = []
    elif problem == "one_rerun":
        reports = reports[:1]
    elif problem == "missing_case":
        for report in reports:
            report["cases"] = report["cases"][3:]
    elif problem == "missing_variant":
        reports[0]["cases"].pop()
    elif problem == "duplicate":
        reports[0]["cases"].append(copy.deepcopy(first))
    elif problem == "incorrect":
        first["correct"] = False
    elif problem in {"nan", "infinite", "zero", "negative"}:
        first["timing"]["median"] = {"nan": float("nan"), "infinite": float("inf"), "zero": 0, "negative": -1}[problem]
    elif problem == "wrong_boundary":
        first["transfer_boundary"] = "device_output_reused"
    elif problem == "wrong_shape":
        first["case"]["lhs_terms"] = 1
    elif problem == "wrong_repeat":
        first["case"]["repeat"] = 1
    elif problem == "failed_row":
        first["status"] = "failed"
    elif problem == "failed_report":
        reports[0]["status"] = "failed"
    elif problem == "missing_runtime":
        reports[0].pop("metal_status")
    summary = module.summarize_wave1d_evidence(reports, repeat=7)
    assert summary["status"] in {"invalid_evidence", "insufficient_evidence"}
    assert summary["evidence_errors"]
    assert summary["aggregated_cases"] == []


def test_wave1d_cpu_only_report_is_a_skip(monkeypatch) -> None:
    module = load_benchmark_module()
    monkeypatch.setattr(module, "build_single_report", lambda **kwargs: {
        "status": "skipped", "skip_reason": "Metal unavailable",
        "metal_status": {"runtime_available": False}, "cases": [],
    })
    report = module.build_report(repeat=7, profile="wave1d", reruns=3)
    assert report["status"] == "skipped"
    assert report["wave1d_evidence"]["status"] == "skipped"


def test_wave1d_docs_require_same_boundary_comparisons() -> None:
    architecture = read(APPLE_ACCELERATOR)
    protocol = read(PROTOCOL)

    for token in (
        "Wave 1D",
        "mean-of-medians",
        "retained reused-output",
        "device_output_allocating",
        "transfer_inclusive",
        "small-row",
    ):
        assert token in architecture
        assert token in protocol


def test_wave1d_report_is_registered_as_latest_apple_metal_evidence() -> None:
    report_path = "docs/benchmarks/reports/apple_metal_wave1d_2026-08-21.md"

    roadmap = read(ROOT / "docs/roadmap.md")
    provenance = read(ROOT / "docs/research/provenance.md")
    campaign_plan = read(ROOT / "docs/plans/wolfgang-kernel-performance-campaign.md")

    assert "Latest Apple Metal report: " + report_path in roadmap
    assert report_path in provenance
    assert "Apple Metal Wave 1D" in provenance
    assert "latest Apple Metal report: " + report_path in campaign_plan

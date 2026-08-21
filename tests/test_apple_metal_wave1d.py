from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

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



def test_wave1d_evidence_summary_computes_mean_of_medians_and_regressions() -> None:
    module = load_benchmark_module()
    reports = [
        {
            "cases": [
                {
                    "case": {
                        "name": "metal_wave1d_small_rows_128x128",
                        "profile": "wave1d",
                        "wave1d_gate": "small_regression_guard",
                        "lhs_terms": 128,
                        "rhs_terms": 128,
                    },
                    "variant": "metal_transfer_inclusive",
                    "timing": {"median": 0.0040},
                    "transfer_boundary": "transfer_inclusive",
                    "status": "ok",
                },
                {
                    "case": {
                        "name": "metal_wave1d_small_rows_128x128",
                        "profile": "wave1d",
                        "wave1d_gate": "small_regression_guard",
                        "lhs_terms": 128,
                        "rhs_terms": 128,
                    },
                    "variant": "metal_device_matrix",
                    "timing": {"median": 0.0020},
                    "transfer_boundary": "device_output_allocating",
                    "status": "ok",
                },
                {
                    "case": {
                        "name": "metal_wave1d_small_rows_128x128",
                        "profile": "wave1d",
                        "wave1d_gate": "small_regression_guard",
                        "lhs_terms": 128,
                        "rhs_terms": 128,
                    },
                    "variant": "metal_device_matrix_reuse",
                    "timing": {"median": 0.00212},
                    "transfer_boundary": "device_output_reused",
                    "status": "ok",
                },
                {
                    "case": {
                        "name": "metal_wave1d_large_rows_2048x2048",
                        "profile": "wave1d",
                        "wave1d_gate": "retained_reuse_gate",
                        "lhs_terms": 2048,
                        "rhs_terms": 2048,
                    },
                    "variant": "metal_transfer_inclusive",
                    "timing": {"median": 0.120},
                    "transfer_boundary": "transfer_inclusive",
                    "status": "ok",
                },
                {
                    "case": {
                        "name": "metal_wave1d_large_rows_2048x2048",
                        "profile": "wave1d",
                        "wave1d_gate": "retained_reuse_gate",
                        "lhs_terms": 2048,
                        "rhs_terms": 2048,
                    },
                    "variant": "metal_device_matrix",
                    "timing": {"median": 0.080},
                    "transfer_boundary": "device_output_allocating",
                    "status": "ok",
                },
                {
                    "case": {
                        "name": "metal_wave1d_large_rows_2048x2048",
                        "profile": "wave1d",
                        "wave1d_gate": "retained_reuse_gate",
                        "lhs_terms": 2048,
                        "rhs_terms": 2048,
                    },
                    "variant": "metal_device_matrix_reuse",
                    "timing": {"median": 0.060},
                    "transfer_boundary": "device_output_reused",
                    "status": "ok",
                },
            ]
        },
        {
            "cases": [
                {
                    "case": {
                        "name": "metal_wave1d_small_rows_128x128",
                        "profile": "wave1d",
                        "wave1d_gate": "small_regression_guard",
                        "lhs_terms": 128,
                        "rhs_terms": 128,
                    },
                    "variant": "metal_transfer_inclusive",
                    "timing": {"median": 0.0038},
                    "transfer_boundary": "transfer_inclusive",
                    "status": "ok",
                },
                {
                    "case": {
                        "name": "metal_wave1d_small_rows_128x128",
                        "profile": "wave1d",
                        "wave1d_gate": "small_regression_guard",
                        "lhs_terms": 128,
                        "rhs_terms": 128,
                    },
                    "variant": "metal_device_matrix",
                    "timing": {"median": 0.0020},
                    "transfer_boundary": "device_output_allocating",
                    "status": "ok",
                },
                {
                    "case": {
                        "name": "metal_wave1d_small_rows_128x128",
                        "profile": "wave1d",
                        "wave1d_gate": "small_regression_guard",
                        "lhs_terms": 128,
                        "rhs_terms": 128,
                    },
                    "variant": "metal_device_matrix_reuse",
                    "timing": {"median": 0.00210},
                    "transfer_boundary": "device_output_reused",
                    "status": "ok",
                },
                {
                    "case": {
                        "name": "metal_wave1d_large_rows_2048x2048",
                        "profile": "wave1d",
                        "wave1d_gate": "retained_reuse_gate",
                        "lhs_terms": 2048,
                        "rhs_terms": 2048,
                    },
                    "variant": "metal_transfer_inclusive",
                    "timing": {"median": 0.118},
                    "transfer_boundary": "transfer_inclusive",
                    "status": "ok",
                },
                {
                    "case": {
                        "name": "metal_wave1d_large_rows_2048x2048",
                        "profile": "wave1d",
                        "wave1d_gate": "retained_reuse_gate",
                        "lhs_terms": 2048,
                        "rhs_terms": 2048,
                    },
                    "variant": "metal_device_matrix",
                    "timing": {"median": 0.079},
                    "transfer_boundary": "device_output_allocating",
                    "status": "ok",
                },
                {
                    "case": {
                        "name": "metal_wave1d_large_rows_2048x2048",
                        "profile": "wave1d",
                        "wave1d_gate": "retained_reuse_gate",
                        "lhs_terms": 2048,
                        "rhs_terms": 2048,
                    },
                    "variant": "metal_device_matrix_reuse",
                    "timing": {"median": 0.061},
                    "transfer_boundary": "device_output_reused",
                    "status": "ok",
                },
            ]
        },
        {
            "cases": [
                {
                    "case": {
                        "name": "metal_wave1d_small_rows_128x128",
                        "profile": "wave1d",
                        "wave1d_gate": "small_regression_guard",
                        "lhs_terms": 128,
                        "rhs_terms": 128,
                    },
                    "variant": "metal_transfer_inclusive",
                    "timing": {"median": 0.0039},
                    "transfer_boundary": "transfer_inclusive",
                    "status": "ok",
                },
                {
                    "case": {
                        "name": "metal_wave1d_small_rows_128x128",
                        "profile": "wave1d",
                        "wave1d_gate": "small_regression_guard",
                        "lhs_terms": 128,
                        "rhs_terms": 128,
                    },
                    "variant": "metal_device_matrix",
                    "timing": {"median": 0.0020},
                    "transfer_boundary": "device_output_allocating",
                    "status": "ok",
                },
                {
                    "case": {
                        "name": "metal_wave1d_small_rows_128x128",
                        "profile": "wave1d",
                        "wave1d_gate": "small_regression_guard",
                        "lhs_terms": 128,
                        "rhs_terms": 128,
                    },
                    "variant": "metal_device_matrix_reuse",
                    "timing": {"median": 0.00211},
                    "transfer_boundary": "device_output_reused",
                    "status": "ok",
                },
                {
                    "case": {
                        "name": "metal_wave1d_large_rows_2048x2048",
                        "profile": "wave1d",
                        "wave1d_gate": "retained_reuse_gate",
                        "lhs_terms": 2048,
                        "rhs_terms": 2048,
                    },
                    "variant": "metal_transfer_inclusive",
                    "timing": {"median": 0.121},
                    "transfer_boundary": "transfer_inclusive",
                    "status": "ok",
                },
                {
                    "case": {
                        "name": "metal_wave1d_large_rows_2048x2048",
                        "profile": "wave1d",
                        "wave1d_gate": "retained_reuse_gate",
                        "lhs_terms": 2048,
                        "rhs_terms": 2048,
                    },
                    "variant": "metal_device_matrix",
                    "timing": {"median": 0.081},
                    "transfer_boundary": "device_output_allocating",
                    "status": "ok",
                },
                {
                    "case": {
                        "name": "metal_wave1d_large_rows_2048x2048",
                        "profile": "wave1d",
                        "wave1d_gate": "retained_reuse_gate",
                        "lhs_terms": 2048,
                        "rhs_terms": 2048,
                    },
                    "variant": "metal_device_matrix_reuse",
                    "timing": {"median": 0.059},
                    "transfer_boundary": "device_output_reused",
                    "status": "ok",
                },
            ]
        },
    ]

    summary = module.summarize_wave1d_evidence(reports, repeat=7)

    assert summary["measurement_methodology"]["independent_reruns"] == 3
    assert summary["measurement_methodology"]["timed_repetitions_per_rerun"] == 7
    assert summary["measurement_methodology"]["promotion_metric"] == "mean_of_medians_seconds"
    assert summary["status"] == "reject_investigate"
    assert summary["small_row_regressions"]

    aggregated = {row["case_name"]: row for row in summary["aggregated_cases"]}
    small = aggregated["metal_wave1d_small_rows_128x128"]
    large = aggregated["metal_wave1d_large_rows_2048x2048"]

    assert small["comparisons"]["reused_vs_allocating"]["mean_of_medians_ratio"] > 1.05
    assert small["gate_decision"] == "reject_investigate"
    assert large["comparisons"]["reused_vs_allocating"]["mean_of_medians_ratio"] < 0.90
    assert large["gate_decision"] == "go"



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

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "docs/benchmarks/data/apple_metal_optimization_campaign8_2026-05-07"
PLOT_PATH = ROOT / "docs/benchmarks/plots/accelerator_landscape_with_rocm.svg"
RENDERER = ROOT / "scripts/render_apple_metal_assets.py"
REPORT_PATH = ROOT / "docs/benchmarks/reports/apple_metal_optimization_campaign8_2026-05-07.md"


def read(path: str | Path) -> str:
    return (ROOT / path if isinstance(path, str) else path).read_text(encoding="utf-8")


def test_metal_benchmark_declares_campaign8_profile() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks" / "bench_metal_kernels.py"),
            "--list-cases",
            "--json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["benchmark"] == "apple_metal_kernels"
    assert "campaign8" in payload["profiles"]
    cases = payload["profiles"]["campaign8"]
    assert {case["profile"] for case in cases} == {"campaign8"}
    assert {case["operation"] for case in cases} == {"simplify"}
    assert {
        "metal_campaign8_simplify_words1_duplicate_heavy_8192_terms",
        "metal_campaign8_simplify_words1_duplicate_light_8192_terms",
        "metal_campaign8_simplify_words1_cancellation_4096_terms",
        "metal_campaign8_simplify_words1_large_duplicate_heavy_16384_terms",
        "metal_campaign8_simplify_words2_status_only_4096_terms",
    } <= {case["name"] for case in cases}


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_campaign8_checked_assets_record_timing_decomposition() -> None:
    summary = json.loads((DATA_DIR / "summary.json").read_text(encoding="utf-8"))
    raw = json.loads((DATA_DIR / "raw" / "metal_benchmark_campaign8.json").read_text(encoding="utf-8"))
    report = read(REPORT_PATH)
    plot = read(PLOT_PATH)
    readme = read("docs/research/provenance.md")
    protocol = read("docs/benchmarks/protocol.md")

    assert summary["campaign"] == "apple_metal_optimization_campaign8"
    assert summary["status"] == "ok"
    assert summary["profile"] == "campaign8"
    assert summary["source_benchmark"].endswith("raw/metal_benchmark_campaign8.json")
    assert raw["profile"] == "campaign8"

    candidate_rows = [
        row for row in raw["cases"] if row.get("variant") == "metal_simplify_device_candidate"
    ]
    assert candidate_rows
    ok_rows = [row for row in candidate_rows if row["status"] == "ok"]
    assert ok_rows
    assert any(row["status"] in {"unavailable", "rejected_with_evidence"} for row in candidate_rows)

    for row in ok_rows:
        assert row["transfer_boundary"] == "device_resident"
        assert row["campaign8_timing_schema"] == "checked_device_resident_simplify_v1"
        assert row["timing_decomposition_seconds"]["total_observed"] >= (
            row["timing_decomposition_seconds"]["command_execution"]
        )
        assert row["pipeline_cache"]["boundary"] == "prewarmed_static_pipeline_cache"
        assert row["dispatch_counts"]["bitonic_sort"] == row["bitonic_passes"]
        assert row["dispatch_counts"]["prefix_sum"] == row["prefix_sum_passes"]
        assert row["performance_decision"]["candidate_status"] in {
            "experimental",
            "performance_relevant",
            "benchmark_only",
        }
        assert row["metal_execution"]["timing_decomposition_source"] == (
            "private_hook_internal_steady_clock"
        )

    for row in candidate_rows:
        if row["status"] != "ok":
            assert row["transfer_boundary"] == "status_only"
            assert "kernel_stack" not in row["metal_execution"]
            assert "timing_decomposition_seconds" not in row

    for token in (
        "Apple Metal Campaign 8",
        "timing_decomposition_seconds",
        "pipeline_cache",
        "performance_decision",
        "experimental",
    ):
        assert token in report
        assert token in readme
        assert token in protocol

    assert "Apple Metal simplify device candidate" in json.dumps(
        summary["readme_performance_landscape"],
        sort_keys=True,
    )
    assert "Apple Metal simplify device candidate" in plot


def test_campaign8_renderer_supports_latest_campaign() -> None:
    spec = importlib.util.spec_from_file_location("apple_metal_renderer", RENDERER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.LATEST_APPLE_METAL_CAMPAIGN == "apple_metal_optimization_campaign8"
    assert "apple_metal_optimization_campaign8" in module.CAMPAIGN_CONFIGS


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_campaign8_renderer_check_mode_validates_checked_assets() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(RENDERER),
            "--data-dir",
            str(DATA_DIR),
            "--plot-dir",
            str(ROOT / "docs" / "benchmarks" / "plots"),
            "--check-only",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Apple Metal Campaign 8 assets validated" in completed.stdout

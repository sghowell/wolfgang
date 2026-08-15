from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "docs/benchmarks/data/apple_metal_optimization_campaign7_2026-05-07"
PLOT_PATH = ROOT / "docs/benchmarks/plots/accelerator_landscape_with_rocm.svg"
RENDERER = ROOT / "scripts/render_apple_metal_assets.py"
REPORT_PATH = ROOT / "docs/benchmarks/reports/apple_metal_optimization_campaign7_2026-05-07.md"


def read(path: str | Path) -> str:
    return (ROOT / path if isinstance(path, str) else path).read_text(encoding="utf-8")


def test_metal_benchmark_declares_campaign7_profile() -> None:
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
    assert "campaign7" in payload["profiles"]
    cases = payload["profiles"]["campaign7"]
    assert {case["profile"] for case in cases} == {"campaign7"}
    assert {case["operation"] for case in cases} == {"simplify"}
    assert {
        "metal_campaign7_simplify_words1_duplicate_heavy_8192_terms",
        "metal_campaign7_simplify_words1_duplicate_light_8192_terms",
        "metal_campaign7_simplify_words1_cancellation_4096_terms",
        "metal_campaign7_simplify_words2_unavailable_4096_terms",
    } <= {case["name"] for case in cases}


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_campaign7_checked_assets_record_candidate_rows() -> None:
    summary = json.loads((DATA_DIR / "summary.json").read_text(encoding="utf-8"))
    raw = json.loads((DATA_DIR / "raw" / "metal_benchmark_campaign7.json").read_text(encoding="utf-8"))
    report = read(REPORT_PATH)
    plot = read(PLOT_PATH)
    readme = read("docs/research/provenance.md")
    protocol = read("docs/benchmarks/protocol.md")

    assert summary["campaign"] == "apple_metal_optimization_campaign7"
    assert summary["status"] == "ok"
    assert summary["profile"] == "campaign7"
    assert summary["source_benchmark"].endswith("raw/metal_benchmark_campaign7.json")
    assert raw["profile"] == "campaign7"

    candidate_rows = [
        row for row in raw["cases"] if row.get("variant") == "metal_simplify_device_candidate"
    ]
    assert candidate_rows
    assert any(row["status"] == "ok" and row["correct"] is True for row in candidate_rows)
    assert any(row["status"] == "unavailable" for row in candidate_rows)

    for row in candidate_rows:
        assert row["object_backend"] == "metal"
        assert row["operation"] == "simplify"
        assert row["metal_simplify_strategy"] == "device_candidate"
        if row["status"] == "ok":
            assert row["transfer_boundary"] == "device_resident"
            assert row["metal_simplify_strategy_status"] == "benchmark_only"
            assert row["output_terms"] == row["case"]["output_terms"]
            assert row["metal_execution"]["kernel_stack"] == [
                "fp_simplify_words1_init_keys",
                "fp_simplify_words1_bitonic_sort_step",
                "fp_simplify_words1_mark_heads",
                "fp_simplify_prefix_sum_step",
                "fp_simplify_words1_reduce_by_key",
                "fp_simplify_words1_compact_survivors",
            ]
            assert row["metal_simplify_primitive_stack"]["sort"] == "bitonic_sort_words1"
            assert row["metal_simplify_primitive_stack"]["prefix_sum"] == (
                "hillis_steele_inclusive_scan_uint32"
            )
            assert row["metal_simplify_primitive_stack"]["reduce_by_key"] == (
                "head_parallel_duplicate_sum_words1"
            )
        else:
            assert row["transfer_boundary"] == "status_only"
            assert row["metal_execution"]["command_buffer_synchronization"] == (
                "not_applicable_no_command_buffer"
            )
            assert row["metal_execution"]["kernel"] == (
                "not_applicable_simplify_candidate_not_executed"
            )
            assert "kernel_stack" not in row["metal_execution"]

    for token in (
        "Apple Metal Campaign 7",
        "metal_simplify_device_candidate",
        "checked device-resident simplify primitive stack",
    ):
        assert token in report
        assert token in readme
        assert token in protocol

    assert "Apple Metal simplify device candidate" in json.dumps(
        summary["readme_performance_landscape"],
        sort_keys=True,
    )
    assert "Apple Metal simplify device candidate" in plot


def test_campaign7_renderer_keeps_historical_campaign_config() -> None:
    spec = importlib.util.spec_from_file_location("apple_metal_renderer", RENDERER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.LATEST_APPLE_METAL_CAMPAIGN == "apple_metal_optimization_campaign8"
    assert "apple_metal_optimization_campaign7" in module.CAMPAIGN_CONFIGS


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_campaign7_renderer_check_mode_validates_checked_assets() -> None:
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
    assert "Apple Metal Campaign 7 assets validated" in completed.stdout

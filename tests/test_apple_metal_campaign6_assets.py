from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = "docs/plans/apple_metal_optimization_campaign6_plan.md"
REPORT_PATH = "docs/benchmarks/reports/apple_metal_optimization_campaign6_2026-05-07.md"
DATA_DIR = ROOT / "docs/benchmarks/data/apple_metal_optimization_campaign6_2026-05-07"
PLOT_PATH = ROOT / "docs/benchmarks/plots/accelerator_landscape_with_rocm.svg"
RENDERER = ROOT / "scripts/render_apple_metal_assets.py"


def read(path: str | Path) -> str:
    return (ROOT / path if isinstance(path, str) else path).read_text(encoding="utf-8")


def test_metal_benchmark_declares_campaign6_profile() -> None:
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
    assert "campaign6" in payload["profiles"]
    cases = payload["profiles"]["campaign6"]
    assert {case["profile"] for case in cases} == {"campaign6"}
    assert {case["operation"] for case in cases} == {"simplify"}
    assert {
        "metal_campaign6_simplify_words1_duplicate_heavy_8192_terms",
        "metal_campaign6_simplify_words1_duplicate_light_8192_terms",
        "metal_campaign6_simplify_words2_duplicate_heavy_4096_terms",
        "metal_campaign6_simplify_generic_multiword_2048_terms",
        "metal_campaign6_simplify_cancellation_4096_terms",
    } <= {case["name"] for case in cases}
    assert {case["packed_words"] for case in cases} >= {1, 2, 3}

    validate_source = read("scripts/validate.py")
    assert "Apple Metal Campaign 6 device-resident simplify groundwork smoke" in validate_source
    assert '"campaign6"' in validate_source


def test_campaign6_workspace_groundwork_is_private_and_registered() -> None:
    helper_hpp = read("src/detail/accelerator_host_helpers.hpp")
    workspace_hpp = read("src/metal/workspace_metal.hpp")
    workspace_mm = read("src/metal/workspace_metal.mm")
    protocol = read("docs/benchmarks/protocol.md")

    assert '#include "detail/accelerator_host_helpers.hpp"' in workspace_hpp
    for token in (
        "class MetalWorkspace",
        "struct WorkspaceSnapshot",
        "enum class WorkspaceTimingMode",
        "reserve_bytes",
        "high_watermark_bytes",
        "allocation_count",
        "workspace_timing_mode_from_env",
        "WOLFGANG_METAL_BENCH_WORKSPACE_TIMING",
    ):
        assert token in helper_hpp + workspace_hpp + workspace_mm

    assert "#include <Metal/Metal.h>" not in read("include/wolfgang/device_pauli_sum.hpp")
    assert "Campaign 6" in protocol
    assert "metal_simplify_workspace_probe" in protocol
    assert "private MetalWorkspace" in protocol


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_campaign6_checked_assets_record_workspace_probe_status() -> None:
    summary = json.loads((DATA_DIR / "summary.json").read_text(encoding="utf-8"))
    raw = json.loads((DATA_DIR / "raw" / "metal_benchmark_campaign6.json").read_text(encoding="utf-8"))
    report = read(REPORT_PATH)
    plot = read(PLOT_PATH)
    readme = read("docs/research/provenance.md")
    normalized_readme = " ".join(readme.split())
    plan = read(PLAN_PATH)

    assert summary["campaign"] == "apple_metal_optimization_campaign6"
    assert summary["status"] == "ok"
    assert summary["profile"] == "campaign6"
    assert summary["source_benchmark"].endswith("raw/metal_benchmark_campaign6.json")
    assert raw["profile"] == "campaign6"
    assert raw["benchmark"] == "apple_metal_kernels"

    for token in (
        "Apple Metal Campaign 6",
        "metal_simplify_workspace_probe",
        "device-resident simplify candidate remains blocked",
        "private MetalWorkspace",
    ):
        assert token in report
        assert token in normalized_readme
        assert token in plan

    assert "Apple Metal simplify transfer reference" in json.dumps(
        summary["readme_performance_landscape"],
        sort_keys=True,
    )
    assert "Apple Metal simplify transfer reference" in plot

    simplify_rows = [row for row in raw["cases"] if row.get("operation") == "simplify"]
    assert simplify_rows
    transfer_rows = [
        row for row in simplify_rows if row["variant"] == "metal_simplify_transfer_reference"
    ]
    workspace_rows = [
        row for row in simplify_rows if row["variant"] == "metal_simplify_workspace_probe"
    ]
    assert transfer_rows
    assert workspace_rows

    for row in workspace_rows:
        assert row["object_backend"] == "metal"
        assert row["status"] == "skipped"
        assert row["timing"] is None
        assert row["correct"] is None
        assert row["transfer_boundary"] == "status_only"
        assert row["metal_execution"]["kernel"] == (
            "not_applicable_device_resident_simplify_candidate_not_retained"
        )
        assert row["metal_simplify_strategy"] == "device_candidate"
        assert row["metal_simplify_strategy_status"] == "rejected_with_evidence"
        assert "Metal sort/prefix/reduce primitives" in row["metal_simplify_strategy_reason"]
        assert row["metal_simplify_workspace_model"]["status"] == "retained_private_model"
        assert row["metal_simplify_workspace_model"]["reserved_bytes_estimate"] > 0
        assert row["metal_simplify_workspace_model"]["workspace_timing_mode"] in {
            "absent",
            "grow_inside_timing",
            "pre_reserved_outside_timing",
        }
        assert row["output_terms"] == row["case"]["output_terms"]


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_campaign6_renderer_check_mode_validates_checked_assets() -> None:
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
    assert "Apple Metal Campaign 6 assets validated" in completed.stdout


def test_campaign6_renderer_keeps_historical_campaign_config() -> None:
    spec = importlib.util.spec_from_file_location("apple_metal_renderer", RENDERER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.LATEST_APPLE_METAL_CAMPAIGN == "apple_metal_optimization_campaign8"
    assert "apple_metal_optimization_campaign6" in module.CAMPAIGN_CONFIGS


def test_latest_apple_metal_renderer_command_advances_past_campaign6() -> None:
    expected_data_dir = "docs/benchmarks/data/apple_metal_optimization_campaign8_2026-05-07"
    stale_data_dir = "docs/benchmarks/data/apple_metal_optimization_campaign5_2026-05-06"

    readme = read("docs/research/provenance.md")
    protocol = read("docs/benchmarks/protocol.md")

    for document in (readme, protocol):
        assert "scripts/render_apple_metal_assets.py" in document
        assert expected_data_dir in document
        assert stale_data_dir not in document

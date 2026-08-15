from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = "docs/plans/apple_metal_optimization_campaign4_plan.md"
REPORT_PATH = "docs/benchmarks/reports/apple_metal_optimization_campaign4_2026-05-06.md"
DATA_DIR = ROOT / "docs/benchmarks/data/apple_metal_optimization_campaign4_2026-05-06"
PLOT_PATH = ROOT / "docs/benchmarks/plots/accelerator_landscape_with_rocm.svg"
RENDERER = ROOT / "scripts/render_apple_metal_assets.py"


def read(path: str | Path) -> str:
    return (ROOT / path if isinstance(path, str) else path).read_text(encoding="utf-8")


def normalized(path: str | Path) -> str:
    return " ".join(read(path).split())


def test_campaign4_plan_is_registered_as_source_of_truth() -> None:
    validate_path = ROOT / "scripts" / "validate.py"
    spec = importlib.util.spec_from_file_location("fastpauli_validate", validate_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert (ROOT / PLAN_PATH).exists()
    assert PLAN_PATH in module.SOURCE_OF_TRUTH_PATHS
    assert PLAN_PATH in read("docs/research/provenance.md")
    assert PLAN_PATH in read("AGENTS.md")
    assert PLAN_PATH in read("docs/roadmap.md")

    plan = normalized(PLAN_PATH)
    for required in (
        "two-word selector remains benchmark-only",
        "larger compact-consumer matrices",
        "parallel block-reduction compact total count",
        "private storage for device-only intermediate workflows",
        "sanitized derived shader-counter exports",
        "MPSGraph and PyTorch MPS remain skipped unless an exact sparse Pauli mapping exists",
        "PyPI publication, Windows support, and older macOS compatibility are out of scope",
        "no public Metal API expansion",
    ):
        assert required in plan


def test_metal_benchmark_declares_campaign4_profile() -> None:
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
    assert "campaign4" in payload["profiles"]
    cases = payload["profiles"]["campaign4"]
    assert {case["profile"] for case in cases} == {"campaign4"}
    assert {
        "metal_campaign4_words2_large_768x768",
        "metal_campaign4_compact_large_2048x2048",
        "metal_campaign4_private_device_boundary_2048x2048",
    } <= {case["name"] for case in cases}
    assert {case["packed_words"] for case in cases} >= {1, 2}
    assert all(case["matrix_entries"] == case["lhs_terms"] * case["rhs_terms"] for case in cases)

    validate_source = read("scripts/validate.py")
    assert "Apple Metal Campaign 4 experimental benchmark smoke" in validate_source
    assert '"campaign4"' in validate_source


def test_metal_source_declares_campaign4_parallel_compact_selector() -> None:
    matrix_source = read("src/metal/device_commutation_matrix_metal.mm")
    kernel_source = read("src/metal/kernels/commutation.metal")
    benchmark_source = read("benchmarks/bench_metal_kernels.py")
    combined = "\n".join([matrix_source, kernel_source, benchmark_source])

    for token in (
        "FASTPAULI_EXPERIMENTAL_METAL_COMPACT_CONSUMER",
        "gpu_parallel_total",
        "fp_count_commuting_total_block_sums",
        "metal_compact_consumer_gpu_parallel_total",
        "compact_consumer_gpu_parallel_block_reduction",
    ):
        assert token in combined


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_campaign4_checked_assets_record_headroom_closure() -> None:
    summary = json.loads((DATA_DIR / "summary.json").read_text(encoding="utf-8"))
    raw = json.loads((DATA_DIR / "raw" / "metal_benchmark_campaign4.json").read_text(encoding="utf-8"))
    report = read(REPORT_PATH)
    plot = read(PLOT_PATH)
    readme = read("docs/research/provenance.md")

    assert summary["campaign"] == "apple_metal_optimization_campaign4"
    assert summary["status"] == "ok"
    assert summary["profile"] == "campaign4"
    assert summary["source_benchmark"].endswith("raw/metal_benchmark_campaign4.json")
    assert summary["metal_status"]["runtime_available"] is True
    assert "Apple Metal Campaign 4" in report
    assert "parallel block-reduction compact total count" in report
    assert "two-word selector remains benchmark-only" in report
    assert "PyPI publication, Windows support, and older macOS compatibility are out of scope" in report
    assert "Apple Metal Campaign 4" in readme
    assert raw["profile"] == "campaign4"
    assert raw["benchmark"] == "apple_metal_kernels"

    external = summary.get("external_baselines", {})
    assert external.get("mpsgraph", {}).get("status") in {"ok", "skipped"}
    assert external.get("pytorch_mps", {}).get("status") in {"ok", "skipped"}
    assert "semantic_mapping" in external.get("mpsgraph", {})
    assert "semantic_mapping" in external.get("pytorch_mps", {})

    profiler = summary.get("profiler", {})
    assert profiler.get("status") in {"derived_counter_export_recorded", "derived_counter_export_blocked"}
    assert "raw trace bundles are not retained" in json.dumps(profiler, sort_keys=True)

    landscape_text = json.dumps(summary["readme_performance_landscape"], sort_keys=True)
    for token in (
        "CPU",
        "CUDA",
        "ROCm",
        "CuPy",
        "Apple Metal",
        "Apple Metal GPU parallel compact count",
        "metal_campaign4_compact_large_2048x2048",
        "metal_campaign4_words2_large_768x768",
    ):
        assert token in landscape_text
        assert token in plot


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_campaign4_rows_record_parallel_reduction_boundaries() -> None:
    raw = json.loads((DATA_DIR / "raw" / "metal_benchmark_campaign4.json").read_text(encoding="utf-8"))
    rows = [
        row
        for row in raw["cases"]
        if row.get("object_backend") == "metal" and row.get("status") == "ok"
    ]
    assert rows
    variants = {row["variant"] for row in rows}
    assert {
        "metal_device_matrix_reuse_words2_candidate",
        "metal_private_blit_host_output",
        "metal_compact_consumer_gpu_total",
        "metal_compact_consumer_gpu_parallel_total",
    } <= variants

    parallel_rows = [row for row in rows if row["variant"] == "metal_compact_consumer_gpu_parallel_total"]
    assert parallel_rows
    for row in parallel_rows:
        execution = row["metal_execution"]
        assert execution["kernel"] == "fp_count_commuting_total_block_sums"
        assert execution["transfer_boundary"] == "compact_consumer_gpu_parallel_block_reduction"
        assert execution["storage_mode"] == "shared_input_shared_partial_count_output"
        assert execution["threadgroup_size"] == [256, 1, 1]
        assert execution["output_entries"] > 1


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_campaign4_renderer_check_mode_validates_checked_assets() -> None:
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
    assert "Apple Metal Campaign 4 assets validated" in completed.stdout

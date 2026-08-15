from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = "docs/plans/apple_metal_optimization_campaign3_plan.md"
REPORT_PATH = "docs/benchmarks/reports/apple_metal_optimization_campaign3_2026-05-06.md"
DATA_DIR = ROOT / "docs/benchmarks/data/apple_metal_optimization_campaign3_2026-05-06"
PLOT_PATH = ROOT / "docs/benchmarks/plots/accelerator_landscape_with_rocm.svg"
RENDERER = ROOT / "scripts/render_apple_metal_assets.py"


def read(path: str | Path) -> str:
    return (ROOT / path if isinstance(path, str) else path).read_text(encoding="utf-8")


def normalized(path: str | Path) -> str:
    return " ".join(read(path).split())


def test_campaign3_plan_is_registered_as_source_of_truth() -> None:
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
        "shader-counter and shader-timeline profiler evidence",
        "two-word specialized selector decision",
        "offline `.metallib` compilation versus runtime NSString source compilation",
        "private storage plus blit staging",
        "workspace or heap reuse decision",
        "true Metal reduction kernels for compact consumers",
        "MPSGraph and PyTorch MPS external baselines",
        "no public Metal API expansion",
    ):
        assert required in plan


def test_metal_benchmark_declares_campaign3_profile() -> None:
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
    assert "campaign3" in payload["profiles"]
    cases = payload["profiles"]["campaign3"]
    assert {case["profile"] for case in cases} == {"campaign3"}
    assert {
        "metal_campaign3_words2_decision_384x384",
        "metal_campaign3_large_private_storage_1024x1024",
        "metal_campaign3_compact_reduction_512x512",
    } <= {case["name"] for case in cases}
    assert {case["packed_words"] for case in cases} >= {1, 2}
    assert all(case["matrix_entries"] == case["lhs_terms"] * case["rhs_terms"] for case in cases)

    validate_source = read("scripts/validate.py")
    assert "Apple Metal Campaign 3 experimental benchmark smoke" in validate_source
    assert '"campaign3"' in validate_source


def test_metal_source_declares_campaign3_experimental_paths() -> None:
    accelerator_source = read("src/metal/accelerator_metal.mm")
    objective_c_source = read("src/metal/commutation_metal.mm")
    matrix_source = read("src/metal/device_commutation_matrix_metal.mm")
    kernel_source = read("src/metal/kernels/commutation.metal")
    combined = "\n".join([accelerator_source, objective_c_source, matrix_source, kernel_source])

    for token in (
        "FASTPAULI_EXPERIMENTAL_METAL_LIBRARY_PATH",
        "FASTPAULI_EXPERIMENTAL_METAL_OUTPUT_STORAGE",
        "FASTPAULI_EXPERIMENTAL_METAL_COMPACT_CONSUMER",
        "newLibraryWithFile",
        "MTLResourceStorageModePrivate",
        "blitCommandEncoder",
        "fp_count_commuting_total_atomic",
        "fp_count_commuting_rows",
        "fp_count_commuting_cols",
    ):
        assert token in combined


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_campaign3_checked_assets_record_new_paths_and_external_baselines() -> None:
    summary = json.loads((DATA_DIR / "summary.json").read_text(encoding="utf-8"))
    raw = json.loads((DATA_DIR / "raw" / "metal_benchmark_campaign3.json").read_text(encoding="utf-8"))
    report = read(REPORT_PATH)
    plot = read(PLOT_PATH)
    readme = read("docs/research/provenance.md")

    assert summary["campaign"] == "apple_metal_optimization_campaign3"
    assert summary["status"] == "ok"
    assert summary["profile"] == "campaign3"
    assert summary["source_benchmark"].endswith("raw/metal_benchmark_campaign3.json")
    assert summary["metal_status"]["runtime_available"] is True
    assert "Apple Metal Campaign 3" in report
    assert "offline `.metallib`" in report
    assert "private storage plus blit staging" in report
    assert "GPU compact-consumer reductions" in report
    assert "Apple Metal Campaign 3" in readme
    assert raw["git_commit"].endswith("+dirty")
    assert raw["git_provenance"]["dirty"] is True
    assert raw["git_provenance"]["commit_label"] == raw["git_commit"]
    assert summary["git_provenance"]["commit_label"] == raw["git_commit"]
    assert "dirty working tree" in " ".join(report.split())

    external = summary.get("external_baselines", {})
    assert external.get("mpsgraph", {}).get("status") in {"ok", "skipped"}
    assert external.get("pytorch_mps", {}).get("status") in {"ok", "skipped"}
    assert "semantic_mapping" in external.get("mpsgraph", {})
    assert "semantic_mapping" in external.get("pytorch_mps", {})

    landscape_text = json.dumps(summary["readme_performance_landscape"], sort_keys=True)
    for token in (
        "CPU",
        "CUDA",
        "ROCm",
        "CuPy",
        "Apple Metal",
        "Apple Metal private blit host output",
        "Apple Metal GPU compact count",
        "Apple Metal GPU compact column counts",
        "Apple Metal GPU compact row counts",
        "metal_campaign3_large_private_storage_1024x1024",
        "metal_campaign3_compact_reduction_512x512",
    ):
        assert token in landscape_text
        assert token in plot


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_campaign3_rows_record_storage_library_and_reduction_boundaries() -> None:
    raw = json.loads((DATA_DIR / "raw" / "metal_benchmark_campaign3.json").read_text(encoding="utf-8"))
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
        "metal_compact_count_axis0_gpu",
        "metal_compact_count_axis1_gpu",
    } <= variants

    private_rows = [row for row in rows if row["variant"] == "metal_private_blit_host_output"]
    assert private_rows
    for row in private_rows:
        execution = row["metal_execution"]
        assert execution["storage_mode"] == "private_output_plus_shared_staging"
        assert execution["transfer_boundary"] == "device_resident_private_output_blit_to_shared_staging"
        assert execution["buffer_allocation_or_reuse_boundary"] == (
            "private_device_output_allocation_and_shared_staging_allocation_per_call"
        )

    reduction_rows = [
        row for row in rows if row["variant"].startswith("metal_compact_") and row["variant"].endswith("_gpu")
    ]
    assert reduction_rows
    for row in reduction_rows:
        execution = row["metal_execution"]
        assert execution["storage_mode"] == "shared_input_shared_count_output"
        assert execution["transfer_boundary"] == "compact_consumer_gpu_reduction"
        assert execution["kernel"] in {
            "fp_count_commuting_total_atomic",
            "fp_count_commuting_rows",
            "fp_count_commuting_cols",
        }

    metallib_rows = [
        row for row in rows if row.get("metal_execution", {}).get("library_source") == "offline_metallib"
    ]
    assert metallib_rows
    assert all(row["metal_execution"].get("metallib_path") for row in metallib_rows)


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_campaign3_renderer_check_mode_validates_checked_assets() -> None:
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
    assert "Apple Metal Campaign 3 assets validated" in completed.stdout

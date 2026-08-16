from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = "docs/plans/apple_metal_optimization_campaign2_plan.md"
REPORT_PATH = "docs/benchmarks/reports/apple_metal_optimization_campaign2_2026-05-05.md"
DATA_DIR = ROOT / "docs/benchmarks/data/apple_metal_optimization_campaign2_2026-05-05"
PLOT_PATH = ROOT / "docs/benchmarks/plots/accelerator_landscape_with_rocm.svg"
RENDERER = ROOT / "scripts/render_apple_metal_assets.py"


def read(path: str | Path) -> str:
    return (ROOT / path if isinstance(path, str) else path).read_text(encoding="utf-8")


def normalized(path: str | Path) -> str:
    return " ".join(read(path).split())


def test_campaign2_plan_is_registered_as_source_of_truth() -> None:
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
        "one-word specialized commutation kernel retained after A/B evidence",
        "two-word specialized commutation kernel retained as a benchmark-only candidate",
        "two-dimensional dispatch grid",
        "generic 2D fallback retained as the default commutation kernel for words >= 2",
        "poisoned reused-output correctness checks",
        "private storage, offline `.metallib`, Metal reductions, MPSGraph, and PyTorch MPS",
    ):
        assert required in plan


def test_metal_benchmark_declares_specialization_profile() -> None:
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
    assert "specialization" in payload["profiles"]
    cases = payload["profiles"]["specialization"]
    assert {case["profile"] for case in cases} == {"specialization"}
    assert {case["packed_words"] for case in cases} >= {1, 2, 3}
    assert all(case["matrix_entries"] == case["lhs_terms"] * case["rhs_terms"] for case in cases)


def test_metal_source_declares_specialized_2d_dispatch_kernels() -> None:
    objective_c_source = read("src/metal/commutation_metal.mm")
    kernel_source = read("src/metal/kernels/commutation.metal")
    combined = objective_c_source + "\n" + kernel_source

    for kernel_name in (
        "fp_pairwise_commutation_flat_generic",
        "fp_pairwise_commutation_words1",
        "fp_pairwise_commutation_words2",
        "fp_pairwise_commutation_generic",
    ):
        assert kernel_name in combined

    assert "uint2 pair [[thread_position_in_grid]]" in combined
    assert "WOLFGANG_EXPERIMENTAL_METAL_COMMUTATION_KERNEL" in objective_c_source
    assert "requires exactly one packed word" in objective_c_source
    assert "requires exactly two packed words" in objective_c_source
    assert "must be one of words1, words2, generic_2d, or flat_generic" in objective_c_source
    assert "dispatchThreads:MTLSizeMake(" in objective_c_source
    assert "static_cast<NSUInteger>(rhs.num_terms)" in objective_c_source
    assert "metal_detail::kCommutationThreadgroupX" in objective_c_source
    assert "metal_detail::kCommutationThreadgroupY" in objective_c_source
    retained_2d_kernels = kernel_source.split(
        "kernel void fp_pairwise_commutation_words1", maxsplit=1
    )[1]
    assert "uint entry [[thread_position_in_grid]]" not in retained_2d_kernels
    assert re.search(r"/\s*params\.rhs_terms", retained_2d_kernels) is None


def test_apple_campaign2_summary_report_and_landscape_are_checked() -> None:
    summary = json.loads((DATA_DIR / "summary.json").read_text(encoding="utf-8"))
    report = read(REPORT_PATH)
    plot = read(PLOT_PATH)
    readme = read("docs/research/provenance.md")

    assert summary["campaign"] == "apple_metal_optimization_campaign2"
    assert summary["status"] == "ok"
    assert summary["profile"] == "specialization"
    assert summary["source_benchmark"].endswith("raw/metal_benchmark_specialization.json")
    assert summary["metal_status"]["runtime_available"] is True
    assert "Apple Metal Campaign 2" in report
    assert "fp_pairwise_commutation_words1" in report
    assert "poisons the reused device-output matrix" in report
    assert "two-dimensional dispatch" in report
    assert "Apple Metal Campaign 2" in readme

    landscape_text = json.dumps(summary["readme_performance_landscape"], sort_keys=True)
    for token in (
        "CPU",
        "CUDA",
        "ROCm",
        "CuPy",
        "Apple Metal",
        "Apple Metal auto A/B reuse",
        "Apple Metal generic 2D baseline",
        "Apple Metal flat generic baseline",
        "Apple Metal words=1 specialized candidate",
        "Apple Metal words=2 specialized candidate",
        "metal_specialization_words1_512x512",
        "metal_specialization_words2_384x384",
        "metal_specialization_generic_words3_192x192",
    ):
        assert token in landscape_text
        assert token in plot


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_apple_campaign2_rows_record_kernel_selection_and_2d_grid() -> None:
    raw = json.loads((DATA_DIR / "raw" / "metal_benchmark_specialization.json").read_text(encoding="utf-8"))
    metal_rows = [
        row
        for row in raw["cases"]
        if row.get("object_backend") == "metal"
        and row.get("status") == "ok"
        and row.get("operation") in {"commutes_with", "commutes_with_device"}
    ]

    assert metal_rows
    assert {
        row["variant"]
        for row in metal_rows
        if row["variant"].startswith("metal_device_matrix_reuse_")
    } >= {
        "metal_device_matrix_reuse_auto_ab",
        "metal_device_matrix_reuse_generic2d_baseline",
        "metal_device_matrix_reuse_flat_generic_baseline",
        "metal_device_matrix_reuse_words1_candidate",
        "metal_device_matrix_reuse_words2_candidate",
    }
    for row in metal_rows:
        case = row["case"]
        execution = row["metal_execution"]
        if row["variant"] == "metal_device_matrix_reuse_flat_generic_baseline":
            assert execution["dispatch_api"] == "dispatchThreads_1d"
            assert execution["threadgroup_size"] == [256, 1, 1]
            assert execution["grid_shape"] == [case["matrix_entries"], 1, 1]
            assert execution["kernel"] == "fp_pairwise_commutation_flat_generic"
            assert execution["kernel_selector"] == "flat_generic"
        elif row["variant"] == "metal_device_matrix_reuse_generic2d_baseline":
            assert execution["dispatch_api"] == "dispatchThreads_2d"
            assert execution["threadgroup_size"] == [16, 16, 1]
            assert execution["grid_shape"] == [case["rhs_terms"], case["lhs_terms"], 1]
            assert execution["kernel"] == "fp_pairwise_commutation_generic"
            assert execution["kernel_selector"] == "generic_2d"
        elif row["variant"] == "metal_device_matrix_reuse_words1_candidate":
            assert execution["dispatch_api"] == "dispatchThreads_2d"
            assert execution["threadgroup_size"] == [16, 16, 1]
            assert execution["grid_shape"] == [case["rhs_terms"], case["lhs_terms"], 1]
            assert execution["kernel"] == "fp_pairwise_commutation_words1"
            assert execution["kernel_selector"] == "words1"
        elif row["variant"] == "metal_device_matrix_reuse_words2_candidate":
            assert execution["dispatch_api"] == "dispatchThreads_2d"
            assert execution["threadgroup_size"] == [16, 16, 1]
            assert execution["grid_shape"] == [case["rhs_terms"], case["lhs_terms"], 1]
            assert execution["kernel"] == "fp_pairwise_commutation_words2"
            assert execution["kernel_selector"] == "words2"
        else:
            assert execution["dispatch_api"] == "dispatchThreads_2d"
            assert execution["threadgroup_size"] == [16, 16, 1]
            assert execution["grid_shape"] == [case["rhs_terms"], case["lhs_terms"], 1]
            if case["packed_words"] == 1:
                assert execution["kernel"] == "fp_pairwise_commutation_words1"
            else:
                assert execution["kernel"] == "fp_pairwise_commutation_generic"
            assert execution["kernel_selector"] == "auto"
        assert execution["transfer_boundary"] == row["transfer_boundary"]


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_apple_campaign2_renderer_check_mode_validates_checked_assets() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(RENDERER),
            "--data-dir",
            str(DATA_DIR),
            "--plot-dir",
            str(ROOT / "docs/benchmarks/plots"),
            "--check-only",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Apple Metal Campaign 2 assets validated" in completed.stdout

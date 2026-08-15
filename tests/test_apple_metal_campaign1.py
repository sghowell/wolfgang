from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = "docs/plans/apple_metal_optimization_campaign1_plan.md"
REPORT_PATH = "docs/benchmarks/reports/apple_metal_optimization_campaign1_2026-05-05.md"
DATA_DIR = ROOT / "docs/benchmarks/data/apple_metal_optimization_campaign1_2026-05-05"
PLOT_PATH = ROOT / "docs/benchmarks/plots/accelerator_landscape_with_rocm.svg"
RENDERER = ROOT / "scripts/render_apple_metal_assets.py"


def read(path: str | Path) -> str:
    return (ROOT / path if isinstance(path, str) else path).read_text(encoding="utf-8")


def normalized(path: str | Path) -> str:
    return " ".join(read(path).split())


def test_campaign1_plan_is_registered_as_source_of_truth() -> None:
    import importlib.util

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
        "Metal benchmark scaling beyond the original 128x128 smoke case",
        "transfer-inclusive, device-resident, retained device matrix, reused-output, host-materialization, and compact-consumer timing boundaries",
        "broad README performance landscape rows that include Apple Metal next to CPU, CUDA, ROCm/HIP, and external baselines",
        "Private-storage buffers, reusable workspaces, command queue injection, compiled `.metallib` packaging",
    ):
        assert required in plan


def test_metal_benchmark_declares_smoke_and_scaling_profiles() -> None:
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
    assert payload["profiles"]["smoke"]
    assert len(payload["profiles"]["scaling"]) > len(payload["profiles"]["smoke"])
    assert {case["profile"] for case in payload["profiles"]["scaling"]} == {"scaling"}
    assert any(case["num_qubits"] > 64 for case in payload["profiles"]["scaling"])
    assert any(case["lhs_terms"] != case["rhs_terms"] for case in payload["profiles"]["scaling"])


def test_apple_campaign1_summary_and_broad_landscape_include_metal() -> None:
    summary = json.loads((DATA_DIR / "summary.json").read_text(encoding="utf-8"))
    report = read(REPORT_PATH)
    plot = read(PLOT_PATH)
    readme = read("docs/research/provenance.md")

    assert summary["campaign"] == "apple_metal_optimization_campaign1"
    assert summary["status"] == "ok"
    assert summary["source_benchmark"].endswith("raw/metal_benchmark_scaling.json")
    assert summary["metal_status"]["runtime_available"] is True
    assert summary["benchmark_rows"]
    assert "Apple Metal Campaign 1" in report
    assert "../plots/accelerator_landscape_with_rocm.svg" in report
    assert "should be added to the broad landscape" not in readme

    series: set[str] = set()
    labels: set[str] = set()
    for row in summary["readme_performance_landscape"]:
        if "seconds" in row:
            series.add(row["series"])
            labels.add(row["label"])
        for point in row.get("points", []):
            series.add(point["series"])
            labels.add(f'{row.get("gpu_name", "")} {point["series"]}')
    for required in (
        "CPU scalar",
        "CPU default",
        "CPU NEON",
        "Apple Metal transfer-inclusive",
        "Apple Metal device-resident host output",
        "Apple Metal device matrix allocate",
        "Apple Metal device matrix reuse",
        "Apple Metal device matrix to_host",
        "Apple Metal compact count",
    ):
        assert required in series

    landscape_text = " ".join(sorted(series | labels))
    for token in ("CPU", "CUDA", "ROCm", "CuPy", "Metal", "Apple Metal"):
        assert token in landscape_text
        assert token in plot


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_apple_campaign1_renderer_check_mode_validates_checked_assets() -> None:
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
    assert "Apple Metal Campaign 1 assets validated" in completed.stdout


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_apple_campaign1_rows_record_metal_execution_boundaries() -> None:
    raw = json.loads((DATA_DIR / "raw" / "metal_benchmark_scaling.json").read_text(encoding="utf-8"))
    summary = json.loads((DATA_DIR / "summary.json").read_text(encoding="utf-8"))
    raw_metal_rows = [
        row for row in raw["cases"] if row.get("object_backend") == "metal" and row.get("status") == "ok"
    ]
    summary_metal_rows = [
        row
        for row in summary["benchmark_rows"]
        if row.get("object_backend") == "metal" and row.get("status") == "ok"
    ]

    assert raw_metal_rows
    assert len(raw_metal_rows) == len(summary_metal_rows)
    for row in raw_metal_rows:
        execution = row.get("metal_execution")
        assert isinstance(execution, dict)
        assert execution["storage_mode"] == "MTLResourceStorageModeShared"
        assert execution["transfer_boundary"] == row["transfer_boundary"]
        assert execution["buffer_allocation_or_reuse_boundary"]
        assert execution["command_buffer_synchronization"]
        assert execution["threadgroup_size"]
        assert execution["grid_shape"]
        if row["operation"] in {"commutes_with", "commutes_with_device"}:
            assert execution["kernel"] == "fp_pairwise_commutation"
            assert execution["threadgroup_size"] == [256, 1, 1]
            assert execution["grid_shape"] == [row["case"]["matrix_entries"], 1, 1]
        else:
            assert execution["kernel"] == "not_applicable_no_metal_kernel_dispatch"


def test_apple_profiler_status_is_not_stale() -> None:
    architecture = normalized("docs/architecture/apple_accelerator.md")
    readme = normalized("docs/research/provenance.md")

    assert "profiler tooling: blocked until full Xcode/Instruments tooling is available" not in architecture
    assert "Metal System Trace evidence: captured" in architecture
    assert "GPU counter profile and shader timeline" in architecture
    assert "Apple Metal rows are included" in readme

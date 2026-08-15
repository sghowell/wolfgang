"""CUDA deep report asset renderer tests."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_cuda_deep_report_assets.py"
CAMPAIGN2_SCRIPT = ROOT / "scripts" / "render_cuda_campaign2_assets.py"
CAMPAIGN3_SCRIPT = ROOT / "scripts" / "render_cuda_campaign3_assets.py"
CAMPAIGN4_SCRIPT = ROOT / "scripts" / "render_cuda_campaign4_assets.py"
CAMPAIGN5_SCRIPT = ROOT / "scripts" / "render_cuda_campaign5_assets.py"
CAMPAIGN6_SCRIPT = ROOT / "scripts" / "render_cuda_campaign6_assets.py"
CAMPAIGN7_SCRIPT = ROOT / "scripts" / "render_cuda_campaign7_assets.py"
CAMPAIGN8_SCRIPT = ROOT / "scripts" / "render_cuda_campaign8_assets.py"
CAMPAIGN9_SCRIPT = ROOT / "scripts" / "render_cuda_campaign9_assets.py"
CAMPAIGN10_SCRIPT = ROOT / "scripts" / "render_cuda_campaign10_assets.py"
CAMPAIGN11_SCRIPT = ROOT / "scripts" / "render_cuda_campaign11_assets.py"
RAW_DIR = ROOT / "docs" / "benchmarks" / "data" / "cuda_deep_optimization_h100_2026-04-28" / "raw"
SUMMARY = ROOT / "docs" / "benchmarks" / "data" / "cuda_deep_optimization_h100_2026-04-28" / "summary.json"
CAMPAIGN2_DATA = ROOT / "docs" / "benchmarks" / "data" / "cuda_deep_optimization_h100_campaign2_2026-04-28"
CAMPAIGN2_RAW_DIR = CAMPAIGN2_DATA / "raw"
CAMPAIGN2_SUMMARY = CAMPAIGN2_DATA / "summary.json"
CAMPAIGN3_DATA = ROOT / "docs" / "benchmarks" / "data" / "cuda_deep_optimization_h100_campaign3_2026-04-28"
CAMPAIGN3_RAW_DIR = CAMPAIGN3_DATA / "raw"
CAMPAIGN3_SUMMARY = CAMPAIGN3_DATA / "summary.json"
CAMPAIGN4_DATA = ROOT / "docs" / "benchmarks" / "data" / "cuda_deep_optimization_h100_campaign4_2026-04-29"
CAMPAIGN5_DATA = ROOT / "docs" / "benchmarks" / "data" / "cuda_deep_optimization_h100_campaign5_2026-04-29"
CAMPAIGN6_DATA = ROOT / "docs" / "benchmarks" / "data" / "cuda_deep_optimization_h100_campaign6_2026-04-29"
CAMPAIGN7_DATA = ROOT / "docs" / "benchmarks" / "data" / "cuda_deep_optimization_h100_campaign7_2026-04-29"
CAMPAIGN8_DATA = ROOT / "docs" / "benchmarks" / "data" / "cuda_deep_optimization_h100_campaign8_2026-04-29"
CAMPAIGN9_DATA = ROOT / "docs" / "benchmarks" / "data" / "cuda_deferred_headroom_campaign9_2026-04-29"
CAMPAIGN10_DATA = ROOT / "docs" / "benchmarks" / "data" / "cuda_cross_architecture_campaign10_2026-04-29"
CAMPAIGN11_DATA = ROOT / "docs" / "benchmarks" / "data" / "cuda_residual_risk_campaign11_2026-04-29"
PLOT_DIR = ROOT / "docs" / "benchmarks" / "plots"
PLOT_NAMES = {
    "cuda_deep_optimization_architecture.svg",
    "cuda_deep_optimization_h100_optimization_deltas.svg",
    "cuda_deep_optimization_h100_path_speedups.svg",
    "cuda_deep_optimization_h100_profiler_bottlenecks.svg",
    "cuda_deep_optimization_h100_scaling.svg",
    "cuda_deep_optimization_kernel_flows.svg",
}
CAMPAIGN2_PLOT_NAMES = {
    "cuda_h100_campaign2_block_size_hillclimb.svg",
    "cuda_h100_campaign2_evidence_status.svg",
    "cuda_h100_campaign2_final_path_comparison.svg",
    "cuda_h100_campaign2_statevector_speedups.svg",
}
CAMPAIGN3_PLOT_NAMES = {
    "cuda_h100_campaign3_duplicate_reduction_speedups.svg",
    "cuda_h100_campaign3_evidence_status.svg",
    "cuda_h100_campaign3_materialization_boundaries.svg",
    "cuda_h100_campaign3_readme_cross_comparison.svg",
}
CAMPAIGN4_PLOT_NAMES = {
    "cuda_h100_campaign4_workspace_boundaries.svg",
    "cuda_h100_campaign4_duplicate_reduction.svg",
    "cuda_h100_campaign4_commutation_materialization.svg",
    "cuda_h100_campaign4_cross_comparison.svg",
    "cuda_h100_campaign4_performance_landscape.svg",
    "cuda_h100_campaign4_evidence_status.svg",
}
CAMPAIGN5_PLOT_NAMES = {
    "cuda_h100_campaign5_device_output_boundaries.svg",
    "cuda_h100_campaign5_host_materialization_decomposition.svg",
    "cuda_h100_campaign5_performance_landscape.svg",
    "cuda_h100_campaign5_evidence_status.svg",
}
CAMPAIGN6_PLOT_NAMES = {
    "cuda_h100_campaign6_consumer_pipeline.svg",
    "cuda_h100_campaign6_cupy_consumer.svg",
    "cuda_h100_campaign6_performance_landscape.svg",
    "cuda_h100_campaign6_evidence_status.svg",
}
CAMPAIGN7_PLOT_NAMES = {
    "cuda_h100_campaign7_fused_consumers.svg",
    "cuda_h100_campaign7_grouping_summaries.svg",
    "cuda_h100_campaign7_profiler_breakdown.svg",
    "cuda_h100_campaign7_portability.svg",
    "cuda_h100_campaign7_performance_landscape.svg",
    "cuda_h100_campaign7_evidence_status.svg",
}
CAMPAIGN8_PLOT_NAMES = {
    "cuda_h100_campaign8_device_resident_consumers.svg",
    "cuda_h100_campaign8_interop_consumers.svg",
    "cuda_h100_campaign8_stream_graph.svg",
    "cuda_h100_campaign8_scatter_ab.svg",
    "cuda_h100_campaign8_portability.svg",
    "cuda_h100_campaign8_performance_landscape.svg",
}
CAMPAIGN9_PLOT_NAMES = {
    "cuda_campaign9_deferred_headroom_status.svg",
    "cuda_campaign9_privileged_ncu.svg",
    "cuda_campaign9_portability.svg",
    "cuda_campaign9_performance_landscape.svg",
}
CAMPAIGN10_PLOT_NAMES = {
    "cuda_campaign10_cross_architecture.svg",
    "cuda_campaign10_dlpack_consumers.svg",
    "cuda_campaign10_headroom_status.svg",
    "cuda_campaign10_performance_landscape.svg",
}
CAMPAIGN11_ITEMS = {
    "non_h100_ncu_counters",
    "nanobind_refleak_investigation",
}
CAMPAIGN8_REQUIRED_STATUS_FIELDS = {
    "device_resident_graph_status",
    "public_grouping_api_status",
    "dlpack_interop_status",
    "non_h100_portability_status",
    "stream_graph_status",
    "scatter_tuning_status",
}


def _load_renderer_module():
    spec = importlib.util.spec_from_file_location("cuda_deep_report_renderer_under_test", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _campaign10_row(item: int, mode: str, status: str, *, gpu: str = "NVIDIA A100-SXM4-40GB", cc: str = "8.0") -> dict[str, object]:
    return {
        "campaign": "cuda_cross_architecture_campaign10",
        "mode": mode,
        "boundary": "test",
        "campaign9_headroom_item": item,
        "final_status": status,
        "deferred_status_allowed": False,
        "decision_doc": "docs/plans/cuda_cross_architecture_campaign10_plan.md",
        "provider_instance_type": "test-fixture",
        "gpu_name": gpu,
        "gpu_compute_capability": cc,
        "cuda_driver": "12.9",
        "cuda_runtime": "12.9",
        "cuda_toolkit": "12.9.86",
        "compiled_architectures": "80" if cc != "12.0" else "120",
        "architecture_compile_status": "compiled_and_ran",
        "git_revision": "f" * 40,
        "command": "bench --fixture",
        "correctness_digest": "{}",
        "unavailable_reason": "",
        "scale": f"fixture_{item}",
        "results": {
            "cpu_scalar_seconds": 1.0 + item,
            "cuda_device_resident_seconds": 0.1 + item / 100.0,
        },
    }


def _write_campaign10_fixture(data_dir: Path) -> None:
    raw = data_dir / "raw"
    raw.mkdir(parents=True)
    rows = [
        _campaign10_row(1, "cross_arch_portability", "passed"),
        _campaign10_row(2, "dlpack_pytorch", "implemented"),
        _campaign10_row(3, "public_grouping_api", "rejected_with_evidence"),
        _campaign10_row(4, "stream_graph_reprobe", "rejected_with_evidence"),
        _campaign10_row(5, "csr_scatter_reprobe", "rejected_with_evidence"),
    ]
    rows[1]["results"] = {
        "cupy_dlpack_from_dlpack_seconds": 0.0002,
        "torch_dlpack_from_dlpack_seconds": 0.0003,
        "cuda_device_resident_seconds": 0.002,
    }
    (raw / "fixture.json").write_text(json.dumps({"cases": rows}), encoding="utf-8")


def _campaign11_row(item: str, host: str, status: str) -> dict[str, object]:
    host_meta = {
        "a100": {
            "gpu_name": "NVIDIA A100-SXM4-80GB",
            "gpu_compute_capability": "8.0",
            "compiled_architectures": "80",
        },
        "rtxpro6000blackwell": {
            "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "gpu_compute_capability": "12.0",
            "compiled_architectures": "120",
        },
    }[host]
    row: dict[str, object] = {
        "campaign": "cuda_residual_risk_campaign11",
        "residual_item": item,
        "final_status": status,
        "deferred_status_allowed": False,
        "host_id": host,
        "gpu_name": host_meta["gpu_name"],
        "gpu_compute_capability": host_meta["gpu_compute_capability"],
        "cuda_driver": "13.0",
        "cuda_runtime": "12.8",
        "cuda_toolkit": "12.8.93",
        "compiled_architectures": host_meta["compiled_architectures"],
        "git_revision": "f" * 40,
        "command": "campaign11 --fixture",
        "artifact_paths": ["docs/benchmarks/data/cuda_residual_risk_campaign11_2026-04-29/logs/fixture.log"],
        "limitation": "",
        "decision": "fixture decision",
    }
    if item == "non_h100_ncu_counters":
        row.update(
            {
                "ncu_install_status": "installed",
                "ncu_version": "2025.3.0.0",
                "profiler_permission_status": "counters_captured",
            }
        )
    else:
        row.update(
            {
                "compute_sanitizer_status": "memcheck_passed",
                "nanobind_diagnostic_classification": "no_runtime_leak_detected",
            }
        )
    return row


def _write_campaign11_fixture(data_dir: Path) -> None:
    raw = data_dir / "raw"
    raw.mkdir(parents=True)
    rows = [
        _campaign11_row(item, host, "passed")
        for item in CAMPAIGN11_ITEMS
        for host in ("a100", "rtxpro6000blackwell")
    ]
    (raw / "fixture.json").write_text(json.dumps({"cases": rows}), encoding="utf-8")


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_cuda_deep_report_asset_renderer_outputs_summary_and_svgs(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    plots = tmp_path / "plots"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--raw-dir",
            str(RAW_DIR),
            "--summary-output",
            str(summary),
            "--plot-dir",
            str(plots),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    data = json.loads(summary.read_text(encoding="utf-8"))
    assert "final_scaling" in data
    assert "retained_experiments" in data
    assert "rejected_experiments" in data
    assert data["experiment_provenance"]["baseline_commit"].startswith("aeeebbaa")
    assert data["competitors"]["cuquantum"]["available"] is True
    assert {path.name for path in plots.glob("*.svg")} == PLOT_NAMES


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_cuda_deep_report_asset_renderer_matches_checked_in_outputs(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    plots = tmp_path / "plots"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--raw-dir",
            str(RAW_DIR),
            "--summary-output",
            str(summary),
            "--plot-dir",
            str(plots),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert summary.read_text(encoding="utf-8") == SUMMARY.read_text(encoding="utf-8")
    for name in PLOT_NAMES:
        assert (plots / name).read_text(encoding="utf-8") == (PLOT_DIR / name).read_text(
            encoding="utf-8"
        )


def test_cuda_deep_report_asset_renderer_rejects_mismatched_ab_cases() -> None:
    module = _load_renderer_module()
    left = {"cases": [{"name": "statevector_expectation", "scale": "a", "results": {}}]}
    right = {"cases": [{"name": "statevector_expectation", "scale": "b", "results": {}}]}

    with pytest.raises(ValueError, match="A/B case mismatch"):
        module.paired_cases(left, right)


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_cuda_campaign2_asset_renderer_matches_checked_in_outputs(tmp_path: Path) -> None:
    plots = tmp_path / "plots"
    completed = subprocess.run(
        [
            sys.executable,
            str(CAMPAIGN2_SCRIPT),
            "--summary",
            str(CAMPAIGN2_SUMMARY),
            "--raw-dir",
            str(CAMPAIGN2_RAW_DIR),
            "--plot-dir",
            str(plots),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert {path.name for path in plots.glob("*.svg")} == CAMPAIGN2_PLOT_NAMES
    for name in CAMPAIGN2_PLOT_NAMES:
        assert (plots / name).read_text(encoding="utf-8") == (PLOT_DIR / name).read_text(
            encoding="utf-8"
        )


def test_cuda_campaign2_summary_marks_privileged_ncu_as_required_evidence() -> None:
    summary = json.loads(CAMPAIGN2_SUMMARY.read_text(encoding="utf-8"))
    required_failures = [
        name
        for name, status in summary["profile_status"].items()
        if status["required"] and status["status"] != "success"
    ]

    assert required_failures == []
    assert summary["profiler_evidence_status"]["nonprivileged_ncu"]["status"] == (
        "expected_permission_denied"
    )
    assert summary["profiler_evidence_status"]["privileged_ncu"]["status"] == "success"
    assert set(summary["privileged_ncu_retry"]["reports"]) == set(
        summary["profiler_evidence_status"]["privileged_ncu"]["required_hot_paths"]
    )


def test_cuda_campaign3_asset_renderer_outputs_summary_and_svgs(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    base_case = {
        "name": "simplify_duplicate_pressure",
        "scale": "terms_50000",
        "results": {
            "cuda_transfer_inclusive_seconds": 4.0,
            "cuda_device_resident_seconds": 3.0,
        },
    }
    exp_case = {
        "name": "simplify_duplicate_pressure",
        "scale": "terms_50000",
        "dataset": {"num_qubits": 16, "num_terms": 50000, "survivor_count": 1024},
        "instrumentation": {
            "temporary_storage_bytes": {
                "available": True,
                "implementation_path": "packed_key32_sort_reduce",
                "estimated_bytes": 2400000,
            },
            "result_materialization": "device-resident sparse Pauli buffers",
        },
        "results": {
            "cpu_scalar_seconds": 10.0,
            "cpu_default_seconds": 9.0,
            "cpu_optimized_timings": {"avx512": {"seconds": 8.0}},
            "cuda_transfer_inclusive_seconds": 2.0,
            "cuda_device_resident_seconds": 1.0,
        },
    }
    default_report = {"git_commit": "experiment", "cases": [exp_case]}
    stress_baseline = {"git_commit": "baseline", "cases": [base_case]}
    stress_experiment = {"git_commit": "experiment", "cases": [exp_case]}
    materialization = {"git_commit": "experiment", "cases": [exp_case]}
    competitive = {
        "competitors": {"qiskit": {"available": True, "version": "test"}},
        "cases": [
            {
                "name": "simplify",
                "results": {
                    "fastpauli_scalar_seconds": 10.0,
                    "competitor_seconds": 20.0,
                },
            }
        ],
    }
    profile = {"results": [{"status": "success"}, {"status": "missing_executable"}]}
    payloads = {
        "baseline_cuda_scaling_stress.json": stress_baseline,
        "experiment_cuda_scaling_stress.json": stress_experiment,
        "experiment_cuda_scaling_default.json": default_report,
        "experiment_cuda_scaling_materialization.json": materialization,
        "competitive_baselines_final.json": competitive,
        "experiment_profile_report.json": profile,
    }
    for name, payload in payloads.items():
        (raw / name).write_text(json.dumps(payload), encoding="utf-8")

    summary = tmp_path / "summary.json"
    plots = tmp_path / "plots"
    completed = subprocess.run(
        [
            sys.executable,
            str(CAMPAIGN3_SCRIPT),
            "--raw-dir",
            str(raw),
            "--summary-output",
            str(summary),
            "--plot-dir",
            str(plots),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data["provenance"]["baseline_git_commit"] == "baseline"
    assert data["provenance"]["experiment_git_commit"] == "experiment"
    assert data["baseline_vs_experiment"][0]["speedup"] == 2.0
    assert {path.name for path in plots.glob("*.svg")} == CAMPAIGN3_PLOT_NAMES


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_cuda_campaign4_asset_renderer_matches_checked_in_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    plots = tmp_path / "plots"
    shutil.copytree(CAMPAIGN4_DATA, data_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(CAMPAIGN4_SCRIPT),
            "--data-dir",
            str(data_dir),
            "--plot-dir",
            str(plots),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (data_dir / "summary.json").read_text(encoding="utf-8") == (
        CAMPAIGN4_DATA / "summary.json"
    ).read_text(encoding="utf-8")
    assert {path.name for path in plots.glob("*.svg")} == CAMPAIGN4_PLOT_NAMES
    for name in CAMPAIGN4_PLOT_NAMES:
        assert (plots / name).read_text(encoding="utf-8") == (PLOT_DIR / name).read_text(
            encoding="utf-8"
        )


def test_cuda_campaign4_summary_tracks_readme_performance_landscape() -> None:
    summary = json.loads((CAMPAIGN4_DATA / "summary.json").read_text(encoding="utf-8"))
    rows = summary["readme_performance_landscape"]
    assert len(rows) == 16
    assert {row["category"] for row in rows} == {
        "FastPauli default profile",
        "External package baseline",
    }
    assert {
        "simplify_duplicate_pressure",
        "statevector_expectation",
        "pairwise_commutation",
        "matmul_product_generation_simplify",
    }.issubset({row["operation"] for row in rows})
    series = {point["series"] for row in rows for point in row["points"]}
    assert {
        "CPU scalar",
        "CPU TBB",
        "CPU AVX2",
        "CPU AVX-512",
        "CUDA transfer-inclusive",
        "CUDA operator-resident",
        "CUDA device-resident",
        "External baseline",
    }.issubset(series)


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_cuda_campaign5_asset_renderer_matches_checked_in_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    plots = tmp_path / "plots"
    shutil.copytree(CAMPAIGN5_DATA, data_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(CAMPAIGN5_SCRIPT),
            "--data-dir",
            str(data_dir),
            "--plot-dir",
            str(plots),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (data_dir / "summary.json").read_text(encoding="utf-8") == (
        CAMPAIGN5_DATA / "summary.json"
    ).read_text(encoding="utf-8")
    assert {path.name for path in plots.glob("*.svg")} == CAMPAIGN5_PLOT_NAMES
    for name in CAMPAIGN5_PLOT_NAMES:
        assert (plots / name).read_text(encoding="utf-8") == (PLOT_DIR / name).read_text(
            encoding="utf-8"
        )


def test_cuda_campaign5_summary_tracks_readme_performance_landscape() -> None:
    summary = json.loads((CAMPAIGN5_DATA / "summary.json").read_text(encoding="utf-8"))
    report = (
        ROOT
        / "docs"
        / "benchmarks"
        / "reports"
        / "cuda_deep_optimization_h100_campaign5_2026-04-29.md"
    ).read_text(encoding="utf-8")
    rows = summary["readme_performance_landscape"]
    assert "../plots/cuda_h100_campaign5_performance_landscape.svg" in report
    assert len(rows) >= 8
    assert {row["category"] for row in rows} == {
        "FastPauli default profile",
        "External package baseline",
    }
    series = {point["series"] for row in rows for point in row["points"]}
    assert {
        "CPU scalar",
        "CPU TBB",
        "CPU AVX2",
        "CPU AVX-512",
        "CUDA transfer-inclusive",
        "CUDA device-resident",
        "CUDA device-output allocate",
        "CUDA device-output reuse",
        "External baseline",
    }.issubset(series)
    status = {item["label"]: item["status"] for item in summary["evidence"]["status"]}
    assert status["phase 11 CUDA tests"] == "passed"
    assert status["compute-sanitizer ladder"] == "passed"
    assert status["Nsight Systems"] == "passed"


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_cuda_campaign6_asset_renderer_matches_checked_in_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    plots = tmp_path / "plots"
    shutil.copytree(CAMPAIGN6_DATA, data_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(CAMPAIGN6_SCRIPT),
            "--data-dir",
            str(data_dir),
            "--plot-dir",
            str(plots),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (data_dir / "summary.json").read_text(encoding="utf-8") == (
        CAMPAIGN6_DATA / "summary.json"
    ).read_text(encoding="utf-8")
    assert {path.name for path in plots.glob("*.svg")} == CAMPAIGN6_PLOT_NAMES
    for name in CAMPAIGN6_PLOT_NAMES:
        assert (plots / name).read_text(encoding="utf-8") == (PLOT_DIR / name).read_text(
            encoding="utf-8"
        )


def test_cuda_campaign6_summary_tracks_readme_performance_landscape() -> None:
    summary = json.loads((CAMPAIGN6_DATA / "summary.json").read_text(encoding="utf-8"))
    rows = summary["readme_performance_landscape"]
    assert (PLOT_DIR / "cuda_h100_campaign6_performance_landscape.svg").exists()
    assert len(summary["consumer_pipeline"]) == 3
    assert len(rows) >= 10
    assert {
        "FastPauli default profile",
        "Campaign 6 consumer scaling",
        "External package baseline",
    } == {row["category"] for row in rows}
    series = {point["series"] for row in rows for point in row["points"]}
    assert {
        "CPU scalar",
        "CPU TBB",
        "CPU AVX2",
        "CPU AVX-512",
        "CUDA transfer-inclusive",
        "CUDA device-resident",
        "CUDA compact count",
        "CUDA row count",
        "CUDA column count",
        "CuPy reduction",
        "External baseline",
    }.issubset(series)
    status = {item["label"]: item["status"] for item in summary["evidence"]["status"]}
    assert status["phase 11 CUDA tests"] == "passed"
    assert status["compute-sanitizer ladder"] == "passed"
    assert status["Campaign 6 consumer benchmark"] == "passed"
    assert status["Nsight Systems"] == "passed"
    assert status["Nsight Compute"] == "passed"


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_cuda_campaign7_renderer_matches_checked_in_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    plots = tmp_path / "plots"
    shutil.copytree(CAMPAIGN7_DATA, data_dir)
    shutil.copytree(
        CAMPAIGN6_DATA,
        tmp_path / "cuda_deep_optimization_h100_campaign6_2026-04-29",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(CAMPAIGN7_SCRIPT),
            "--data-dir",
            str(data_dir),
            "--plot-dir",
            str(plots),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (data_dir / "summary.json").read_text(encoding="utf-8") == (
        CAMPAIGN7_DATA / "summary.json"
    ).read_text(encoding="utf-8")
    assert {path.name for path in plots.glob("*.svg")} == CAMPAIGN7_PLOT_NAMES
    for name in CAMPAIGN7_PLOT_NAMES:
        assert (plots / name).read_text(encoding="utf-8") == (PLOT_DIR / name).read_text(
            encoding="utf-8"
        )


def test_cuda_campaign7_summary_tracks_readme_performance_landscape() -> None:
    summary = json.loads((CAMPAIGN7_DATA / "summary.json").read_text(encoding="utf-8"))
    report = (
        ROOT
        / "docs"
        / "benchmarks"
        / "reports"
        / "cuda_deep_optimization_h100_campaign7_2026-04-29.md"
    ).read_text(encoding="utf-8")
    rows = summary["readme_performance_landscape"]
    assert "../plots/cuda_h100_campaign7_performance_landscape.svg" in report
    assert len(summary["fused_consumers"]) == 3
    assert len(rows) >= 4
    assert "Campaign 7 fused consumers" in {row["category"] for row in rows}
    series = {point["series"] for row in rows for point in row["points"]}
    assert {
        "CUDA fused grouping",
        "CUDA fused degree summary",
        "CUDA fused CSR export",
    }.issubset(series)
    decisions = {item["experiment"]: item["status"] for item in summary["decisions"]}
    assert decisions["fused_commutation_consumers"] == "benchmark_only_retained"
    assert decisions["count_reduction_specialization"] == "rejected_not_dominant"
    assert decisions["async_stream_api"] == "deferred"
    assert decisions["bitpacked_output"] == "deferred_no_dense_capacity_or_bandwidth_trigger"
    assert decisions["non_h100_portability"] == "blocked_recorded"
    status = {item["label"]: item["status"] for item in summary["evidence"]["status"]}
    assert status["H100 repo validation"] == "passed"
    assert status["compute-sanitizer ladder"] == "passed"
    assert status["Campaign 7 fused benchmark"] == "passed"
    assert status["Nsight Systems"] == "passed"
    assert status["Nsight Compute"] == "passed"
    assert status["non-H100 NVIDIA portability"] == "blocked_recorded"


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_cuda_campaign8_renderer_matches_checked_in_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    plots = tmp_path / "plots"
    shutil.copytree(CAMPAIGN8_DATA, data_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(CAMPAIGN8_SCRIPT),
            "--data-dir",
            str(data_dir),
            "--plot-dir",
            str(plots),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (data_dir / "summary.json").read_text(encoding="utf-8") == (
        CAMPAIGN8_DATA / "summary.json"
    ).read_text(encoding="utf-8")
    assert {path.name for path in plots.glob("*.svg")} == CAMPAIGN8_PLOT_NAMES
    for name in CAMPAIGN8_PLOT_NAMES:
        assert (plots / name).read_text(encoding="utf-8") == (PLOT_DIR / name).read_text(
            encoding="utf-8"
        )


def test_cuda_campaign8_summary_tracks_remaining_headroom_fields() -> None:
    summary = json.loads((CAMPAIGN8_DATA / "summary.json").read_text(encoding="utf-8"))
    report = (
        ROOT
        / "docs"
        / "benchmarks"
        / "reports"
        / "cuda_deep_optimization_h100_campaign8_2026-04-29.md"
    ).read_text(encoding="utf-8")

    assert "../plots/cuda_h100_campaign8_performance_landscape.svg" in report
    assert set(summary["required_status_fields"]) == CAMPAIGN8_REQUIRED_STATUS_FIELDS
    for row in summary["device_resident_consumers"]:
        assert CAMPAIGN8_REQUIRED_STATUS_FIELDS.issubset(row)
    assert {item["experiment"] for item in summary["decisions"]} == {
        "device_resident_graph_consumers",
        "public_fused_grouping_api",
        "dlpack_or_framework_interop",
        "non_h100_portability",
        "stream_graph_execution",
        "csr_scatter_tuning",
    }
    decisions = {item["experiment"]: item["status"] for item in summary["decisions"]}
    assert decisions["device_resident_graph_consumers"] == "retained"
    assert decisions["public_fused_grouping_api"] == "deferred"
    assert decisions["dlpack_or_framework_interop"] in {"deferred", "accepted_private_probe"}
    assert decisions["non_h100_portability"] in {"passed", "blocked"}
    assert decisions["stream_graph_execution"] == "deferred"
    assert decisions["csr_scatter_tuning"] in {"rejected_no_consumer", "rejected_not_dominant"}
    status = {item["label"]: item["status"] for item in summary["evidence"]["status"]}
    assert status["H100 repo validation"] == "passed"
    assert status["phase 11 CUDA tests"] == "passed"
    assert status["compute-sanitizer ladder"] == "passed"
    assert status["Nsight Systems"] == "passed"
    assert status["Nsight Compute"] == "blocked_permission"
    assert status["non-H100 NVIDIA portability"] == "blocked_recorded"
    rows = summary["readme_performance_landscape"]
    series = {point["series"] for row in rows for point in row["points"]}
    assert {
        "CPU scalar",
        "CPU TBB",
        "CPU AVX2",
        "CPU AVX-512",
        "CUDA transfer-inclusive",
        "CUDA device-resident",
        "CUDA compact count",
        "CUDA fused grouping",
        "CUDA Campaign 8 graph compact",
        "CUDA Campaign 8 grouping compact",
        "External baseline",
    }.issubset(series)


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_cuda_campaign9_renderer_matches_checked_in_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    plots = tmp_path / "plots"
    shutil.copytree(CAMPAIGN9_DATA, data_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(CAMPAIGN9_SCRIPT),
            "--data-dir",
            str(data_dir),
            "--plot-dir",
            str(plots),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (data_dir / "summary.json").read_text(encoding="utf-8") == (
        CAMPAIGN9_DATA / "summary.json"
    ).read_text(encoding="utf-8")
    assert {path.name for path in plots.glob("*.svg")} == CAMPAIGN9_PLOT_NAMES
    for name in CAMPAIGN9_PLOT_NAMES:
        assert (plots / name).read_text(encoding="utf-8") == (PLOT_DIR / name).read_text(
            encoding="utf-8"
        )


def test_cuda_campaign9_summary_closes_every_deferred_item() -> None:
    summary = json.loads((CAMPAIGN9_DATA / "summary.json").read_text(encoding="utf-8"))

    assert summary["campaign"] == "cuda_deferred_headroom_campaign9"
    assert summary["deferred_status_allowed"] is False
    assert {item["campaign8_headroom_item"] for item in summary["decisions"]} == set(range(1, 7))
    assert all(item["final_status"] != "deferred" for item in summary["decisions"])
    decisions = {item["mode"]: item["final_status"] for item in summary["decisions"]}
    assert decisions["non_h100_portability"] == "blocked_external"
    assert decisions["privileged_ncu"] == "passed"
    assert decisions["public_grouping_api"] == "rejected_with_evidence"
    assert decisions["dlpack_interop"] == "implemented"
    assert decisions["stream_graph"] == "rejected_with_evidence"
    assert decisions["csr_scatter_reopen"] == "rejected_with_evidence"
    assert summary["ncu_summary"]
    assert all(
        row["mode"] != "privileged_ncu"
        for row in summary["readme_performance_landscape"]
    )
    series = {point["series"] for row in summary["readme_performance_landscape"] for point in row["points"]}
    assert {
        "CPU scalar",
        "CPU default",
        "CPU optimized",
        "CPU TBB",
        "CPU AVX2",
        "CPU AVX-512",
        "CUDA transfer-inclusive",
        "CUDA device-resident",
        "CUDA Campaign 8 graph compact",
        "CUDA Campaign 8 grouping compact",
        "CUDA CSR export baseline",
        "CUDA conflict degrees total",
        "CuPy CUDA Array Interface",
        "CuPy DLPack",
        "External baseline",
    }.issubset(series)


def test_cuda_campaign9_raw_rows_carry_required_schema_fields() -> None:
    required = {
        "campaign",
        "mode",
        "boundary",
        "campaign8_headroom_item",
        "final_status",
        "deferred_status_allowed",
        "decision_doc",
        "correctness_digest",
        "unavailable_reason",
        "git_revision",
        "cuda_driver",
        "cuda_runtime",
        "cuda_toolkit",
        "compiled_architectures",
        "gpu_name",
        "gpu_compute_capability",
    }
    for path in (CAMPAIGN9_DATA / "raw").glob("*.json"):
        if path.name.endswith("_smoke.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            assert required.issubset(case), path.name
            assert case["campaign"] == "cuda_deferred_headroom_campaign9"
            assert case["final_status"] != "deferred"
            assert case["deferred_status_allowed"] is False
            assert 1 <= int(case["campaign8_headroom_item"]) <= 6


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_cuda_campaign10_asset_renderer_matches_checked_in_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / CAMPAIGN10_DATA.name
    plots = tmp_path / "plots"
    shutil.copytree(CAMPAIGN10_DATA, data_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(CAMPAIGN10_SCRIPT),
            "--data-dir",
            str(data_dir),
            "--plot-dir",
            str(plots),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (data_dir / "summary.json").read_text(encoding="utf-8") == (
        CAMPAIGN10_DATA / "summary.json"
    ).read_text(encoding="utf-8")
    assert {path.name for path in plots.glob("*.svg")} == CAMPAIGN10_PLOT_NAMES
    for name in CAMPAIGN10_PLOT_NAMES:
        assert (plots / name).read_text(encoding="utf-8") == (PLOT_DIR / name).read_text(
            encoding="utf-8"
        )


def test_cuda_campaign10_checked_summary_closes_every_headroom_item() -> None:
    summary = json.loads((CAMPAIGN10_DATA / "summary.json").read_text(encoding="utf-8"))
    readme = (ROOT / "docs/research/provenance.md").read_text(encoding="utf-8")
    report = (
        ROOT
        / "docs"
        / "benchmarks"
        / "reports"
        / "cuda_cross_architecture_campaign10_2026-04-29.md"
    ).read_text(encoding="utf-8")

    assert "../benchmarks/plots/accelerator_landscape_with_rocm.svg" in readme
    assert "cuda_cross_architecture_campaign10_2026-04-29.md" in readme
    assert "../plots/cuda_campaign10_performance_landscape.svg" in report
    assert summary["campaign"] == "cuda_cross_architecture_campaign10"
    assert summary["deferred_status_allowed"] is False
    assert {item["campaign9_headroom_item"] for item in summary["decisions"]} == set(
        range(1, 6)
    )
    assert all(item["final_status"] != "deferred" for item in summary["decisions"])
    decisions = {item["mode"]: item["final_status"] for item in summary["decisions"]}
    assert decisions == {
        "cross_arch_portability": "passed",
        "dlpack_pytorch": "passed",
        "public_grouping_api": "rejected_with_evidence",
        "stream_graph_reprobe": "rejected_with_evidence",
        "csr_scatter_reprobe": "rejected_with_evidence",
    }
    hardware = {
        (row["gpu_compute_capability"], row["compiled_architectures"]): row[
            "architecture_compile_status"
        ]
        for row in summary["hardware"]
    }
    assert hardware[("8.0", "80")] == "compiled_and_ran"
    assert hardware[("12.0", "120")] == "compiled_and_ran"
    series = {
        point["series"]
        for row in summary["readme_performance_landscape"]
        for point in row["points"]
    }
    assert {
        "CPU scalar",
        "CPU optimized",
        "CPU AVX512",
        "CUDA transfer-inclusive",
        "CUDA device-resident",
        "CUDA compact graph consumer",
        "CUDA compact grouping consumer",
        "CUDA CSR export baseline",
        "CuPy DLPack",
        "PyTorch DLPack",
    }.issubset(series)


def test_cuda_campaign10_renderer_outputs_schema_checked_summary_and_svgs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    plots = tmp_path / "plots"
    _write_campaign10_fixture(data_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(CAMPAIGN10_SCRIPT),
            "--data-dir",
            str(data_dir),
            "--plot-dir",
            str(plots),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["campaign"] == "cuda_cross_architecture_campaign10"
    assert summary["deferred_status_allowed"] is False
    assert {item["campaign9_headroom_item"] for item in summary["decisions"]} == set(
        range(1, 6)
    )
    assert all(item["final_status"] != "deferred" for item in summary["decisions"])
    assert {path.name for path in plots.glob("*.svg")} == CAMPAIGN10_PLOT_NAMES
    assert summary["hardware"]
    series = {point["series"] for row in summary["readme_performance_landscape"] for point in row["points"]}
    assert {"CPU scalar", "CUDA device-resident", "PyTorch DLPack"}.issubset(series)
    landscape = (plots / "cuda_campaign10_performance_landscape.svg").read_text(
        encoding="utf-8"
    )
    widths = [
        float(width)
        for width in re.findall(r'x="820" y="[^"]+" width="([0-9.]+)" height="11"', landscape)
    ]
    assert widths
    assert all(width <= 380.0 for width in widths)


def test_cuda_campaign10_renderer_rejects_deferred_or_missing_items(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    plots = tmp_path / "plots"
    _write_campaign10_fixture(data_dir)
    payload_path = data_dir / "raw" / "fixture.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["cases"][0]["final_status"] = "deferred"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(CAMPAIGN10_SCRIPT),
            "--data-dir",
            str(data_dir),
            "--plot-dir",
            str(plots),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "final_status='deferred'" in completed.stderr


def test_cuda_campaign10_renderer_rejects_unknown_mode(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    plots = tmp_path / "plots"
    _write_campaign10_fixture(data_dir)
    payload_path = data_dir / "raw" / "fixture.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["cases"][0]["mode"] = "cross_arch_typo"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(CAMPAIGN10_SCRIPT),
            "--data-dir",
            str(data_dir),
            "--plot-dir",
            str(plots),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "invalid Campaign 10 mode" in completed.stderr


def test_cuda_campaign10_renderer_requires_blackwell_compile_outcome(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    plots = tmp_path / "plots"
    _write_campaign10_fixture(data_dir)
    payload_path = data_dir / "raw" / "fixture.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["cases"][0].update(
        {
            "gpu_name": "NVIDIA RTX PRO 6000 Blackwell",
            "gpu_compute_capability": "12.0",
            "compiled_architectures": "89",
            "architecture_compile_status": "not_checked",
        }
    )
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(CAMPAIGN10_SCRIPT),
            "--data-dir",
            str(data_dir),
            "--plot-dir",
            str(plots),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "Blackwell row lacks" in completed.stderr


def test_cuda_campaign11_renderer_outputs_schema_checked_summary(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_campaign11_fixture(data_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(CAMPAIGN11_SCRIPT),
            "--data-dir",
            str(data_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["campaign"] == "cuda_residual_risk_campaign11"
    assert summary["deferred_status_allowed"] is False
    assert set(summary["required_residual_items"]) == CAMPAIGN11_ITEMS
    assert set(summary["required_hosts"]) == {"a100", "rtxpro6000blackwell"}
    assert {item["residual_item"] for item in summary["decisions"]} == CAMPAIGN11_ITEMS
    assert all(item["final_status"] != "deferred" for item in summary["decisions"])
    assert {
        (row["residual_item"], row["host_id"])
        for row in summary["raw_rows"]
    } == {
        (item, host)
        for item in CAMPAIGN11_ITEMS
        for host in ("a100", "rtxpro6000blackwell")
    }
    ncu_rows = [
        row for row in summary["raw_rows"] if row["residual_item"] == "non_h100_ncu_counters"
    ]
    assert all(row["ncu_version"] for row in ncu_rows)
    assert all(row["profiler_permission_status"] == "counters_captured" for row in ncu_rows)


def test_cuda_campaign11_renderer_rejects_deferred_or_missing_host(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_campaign11_fixture(data_dir)
    payload_path = data_dir / "raw" / "fixture.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["cases"][0]["final_status"] = "deferred"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(CAMPAIGN11_SCRIPT),
            "--data-dir",
            str(data_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "final_status='deferred'" in completed.stderr

    payload["cases"] = [
        case for case in payload["cases"] if case["host_id"] != "rtxpro6000blackwell"
    ]
    payload["cases"][0]["final_status"] = "passed"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(CAMPAIGN11_SCRIPT),
            "--data-dir",
            str(data_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "does not cover all item/host pairs" in completed.stderr


def test_cuda_campaign11_checked_summary_closes_residual_risks() -> None:
    summary = json.loads((CAMPAIGN11_DATA / "summary.json").read_text(encoding="utf-8"))
    report = (
        ROOT
        / "docs"
        / "benchmarks"
        / "reports"
        / "cuda_residual_risk_campaign11_2026-04-29.md"
    ).read_text(encoding="utf-8")

    assert summary["campaign"] == "cuda_residual_risk_campaign11"
    assert summary["deferred_status_allowed"] is False
    assert {item["residual_item"] for item in summary["decisions"]} == CAMPAIGN11_ITEMS
    assert all(item["final_status"] != "deferred" for item in summary["decisions"])
    assert {row["host_id"] for row in summary["hardware"]} == {"a100", "rtxpro6000blackwell"}
    assert {
        (row["residual_item"], row["host_id"])
        for row in summary["raw_rows"]
    } == {
        (item, host)
        for item in CAMPAIGN11_ITEMS
        for host in ("a100", "rtxpro6000blackwell")
    }
    for row in summary["raw_rows"]:
        assert row["artifact_paths"]
        assert row["decision"]
        assert row["host_id"] in {"a100", "rtxpro6000blackwell"}
    assert "A10, L4, RTX 6000 Ada" in report
    assert "final_status: deferred" not in report


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_cuda_campaign8_raw_rows_carry_required_schema_fields() -> None:
    required = CAMPAIGN8_REQUIRED_STATUS_FIELDS | {
        "campaign",
        "mode",
        "boundary",
        "timing_boundary",
        "correctness_digest",
        "unavailable_reason",
        "git_revision",
        "cuda_driver",
        "cuda_runtime",
        "cuda_toolkit",
        "compiled_architectures",
        "gpu_name",
        "gpu_compute_capability",
    }
    raw_dir = CAMPAIGN8_DATA / "raw"
    for name in (
        "campaign8_device_graph.json",
        "campaign8_grouping_consumer.json",
        "campaign8_interop.json",
        "campaign8_stream_graph.json",
        "campaign8_scatter_ab.json",
    ):
        payload = json.loads((raw_dir / name).read_text(encoding="utf-8"))
        for case in payload["cases"]:
            assert required.issubset(case), name
            assert case["campaign"] == "h100_campaign8"
            assert len(case["git_revision"]) == 40
            assert case["gpu_name"]
            assert case["gpu_compute_capability"]


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_cuda_campaign4_evidence_status_prefers_latest_label_fix_logs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    plots = tmp_path / "plots"
    shutil.copytree(CAMPAIGN4_DATA, data_dir)
    metadata = data_dir / "metadata"
    (metadata / "experiment-validate-final-label-fix.log").write_text(
        "validation failed after label-fix rerun\n",
        encoding="utf-8",
    )
    (metadata / "experiment-phase11-label-fix.log").write_text(
        "18 passed, 1 skipped\n",
        encoding="utf-8",
    )
    (metadata / "compute-sanitizer-memcheck-label-fix.log").write_text(
        "========= ERROR SUMMARY: 1 errors\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(CAMPAIGN4_SCRIPT),
            "--data-dir",
            str(data_dir),
            "--plot-dir",
            str(plots),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    status = {item["label"]: item["status"] for item in summary["evidence"]["status"]}
    assert status["experiment validation"] == "missing_or_failed"
    assert status["phase 11 CUDA tests"] == "missing_or_failed"
    assert status["compute-sanitizer ladder"] == "missing_or_failed"


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_cuda_campaign3_asset_renderer_matches_checked_in_outputs(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    plots = tmp_path / "plots"
    completed = subprocess.run(
        [
            sys.executable,
            str(CAMPAIGN3_SCRIPT),
            "--raw-dir",
            str(CAMPAIGN3_RAW_DIR),
            "--summary-output",
            str(summary),
            "--plot-dir",
            str(plots),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert summary.read_text(encoding="utf-8") == CAMPAIGN3_SUMMARY.read_text(
        encoding="utf-8"
    )
    for name in CAMPAIGN3_PLOT_NAMES:
        assert (plots / name).read_text(encoding="utf-8") == (PLOT_DIR / name).read_text(
            encoding="utf-8"
        )


def test_cuda_campaign3_summary_marks_privileged_ncu_as_required_evidence() -> None:
    summary = json.loads(CAMPAIGN3_SUMMARY.read_text(encoding="utf-8"))
    required_failures = summary["profiler_evidence_status"]["required_effective_failures"]
    ncu_statuses = [
        item
        for item in summary["profile_status"]
        if item["name"].startswith("ncu ")
    ]

    assert required_failures == []
    assert len(ncu_statuses) == 4
    assert all(item["effective_status"] == "success" for item in ncu_statuses)
    assert all(
        item["status"] == "expected_permission_denied_superseded"
        for item in ncu_statuses
    )
    assert summary["profiler_evidence_status"]["nonprivileged_ncu"]["status"] == (
        "expected_permission_denied"
    )
    assert summary["profiler_evidence_status"]["privileged_ncu"]["status"] == "success"
    assert len(summary["profiler_evidence_status"]["privileged_ncu"]["reports"]) == 4

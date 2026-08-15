from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "docs/benchmarks/data/rocm_campaign8_architecture_readiness_2026-05-01"
REPORT = ROOT / "docs/benchmarks/reports/rocm_campaign8_architecture_readiness_2026-05-01.md"
RENDERER = ROOT / "scripts/render_rocm_campaign8_assets.py"

EXPECTED_TERMINAL_STATUSES = {
    "backend_neutral_object_model": "accepted_for_future_implementation",
    "simultaneous_cuda_hip_source_builds": "unavailable",
    "multi_gpu_rocm_execution": "out_of_scope_with_next_trigger",
    "non_mi300x_amd_portability": "blocked_external",
    "rocm_wheel_packaging_design": "accepted_for_future_implementation",
    "rocm_ci_hardware_policy": "accepted_for_future_implementation",
    "rocm_clean_machine_install_tests": "accepted_for_future_implementation",
    "rocprofv3_migration": "accepted_for_future_implementation",
    "legacy_rocprof_retention": "retained",
    "external_hip_statevector_contract": "accepted_for_future_implementation",
    "hip_dlpack_reconsideration_contract": "accepted_for_future_implementation",
    "hip_cuda_array_interface_policy": "rejected_with_evidence",
    "public_streams_policy": "rejected_with_evidence",
    "public_graphs_policy": "rejected_with_evidence",
    "public_workspaces_policy": "rejected_with_evidence",
    "targeted_rocm_performance_reopen": "accepted_for_future_implementation",
    "source_build_release_lane_retention": "retained",
}
REQUIRED_TERMINAL_KEYS = set(EXPECTED_TERMINAL_STATUSES)


def test_rocm_campaign8_checked_evidence_and_report_are_consistent() -> None:
    summary = json.loads((DATA_DIR / "summary.json").read_text(encoding="utf-8"))
    report = REPORT.read_text(encoding="utf-8")

    assert summary["campaign"] == "rocm_campaign8_architecture_readiness"
    assert summary["runtime_changes"] == "none"
    assert summary["campaign8_terminal_statuses"] == EXPECTED_TERMINAL_STATUSES
    assert summary["local_cpu_only_validation"]["status"] == "external_closeout_required"
    assert summary["cuda_hip_configure_rejection"]["status"] == "external_closeout_required"
    assert "No HIP kernel, public Python API, ROCm wheel, or multi-GPU runtime behavior changed" in report
    for key, status in summary["campaign8_terminal_statuses"].items():
        assert key in report
        assert status in report


def test_rocm_campaign8_renderer_validates_checked_assets() -> None:
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
    assert "Campaign 8 assets validated" in completed.stdout


def test_rocm_campaign8_renderer_rejects_status_key_drift(tmp_path: Path) -> None:
    source = json.loads((DATA_DIR / "summary.json").read_text(encoding="utf-8"))
    source["campaign8_terminal_statuses"]["unexpected"] = "retained"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "summary.json").write_text(json.dumps(source), encoding="utf-8")
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir()
    (plot_dir / "accelerator_landscape_with_rocm.svg").write_text(
        "<svg><text>CPU CUDA ROCm</text></svg>",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(RENDERER),
            "--data-dir",
            str(data_dir),
            "--plot-dir",
            str(plot_dir),
            "--check-only",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "key mismatch" in completed.stderr


def test_rocm_campaign8_renderer_rejects_status_value_drift(tmp_path: Path) -> None:
    source = json.loads((DATA_DIR / "summary.json").read_text(encoding="utf-8"))
    source["campaign8_terminal_statuses"]["simultaneous_cuda_hip_source_builds"] = (
        "accepted_for_future_implementation"
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "summary.json").write_text(json.dumps(source), encoding="utf-8")
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir()
    (plot_dir / "accelerator_landscape_with_rocm.svg").write_text(
        "<svg><text>CPU CUDA ROCm</text></svg>",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(RENDERER),
            "--data-dir",
            str(data_dir),
            "--plot-dir",
            str(plot_dir),
            "--check-only",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "terminal status mismatch" in completed.stderr

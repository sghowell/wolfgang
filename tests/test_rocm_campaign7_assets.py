"""ROCm Campaign 7 report asset checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from test_rocm_campaign7_plan import REQUIRED_TERMINAL_KEYS

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_rocm_campaign7_assets.py"

TERMINAL_STATUSES = {
    "mi300x_repeatability": "passed",
    "cpu_only_control": "passed",
    "rocm_source_build_runbook": "retained",
    "rocm_ci_or_release_lane": "retained",
    "rocm_packaging_policy": "retained",
    "rocm_wheel_support": "unavailable",
    "alternate_amd_gpu_portability": "blocked_external",
    "profiler_availability": "passed",
    "duplicate_pressure_simplify": "rejected_with_evidence",
    "duplicate_pressure_matmul": "rejected_with_evidence",
    "external_statevector_interop": "out_of_scope_with_next_trigger",
    "hip_dlpack": "rejected_with_evidence",
    "hip_cuda_array_interface": "rejected_with_evidence",
    "public_streams": "rejected_with_evidence",
    "public_graphs": "rejected_with_evidence",
    "public_workspaces": "rejected_with_evidence",
    "multi_gpu_rocm": "out_of_scope_with_next_trigger",
    "simultaneous_cuda_hip": "unavailable",
    "backend_neutral_accelerator_design": "out_of_scope_with_next_trigger",
}


def _base_row(
    *,
    case: str,
    operation: str,
    mode: str,
    final_status: str,
    timing_boundary: str,
) -> dict[str, Any]:
    return {
        "campaign": "rocm_mi300x_campaign7",
        "profile": "campaign7-fixture",
        "operation": operation,
        "mode": mode,
        "backend": "hip",
        "host_role": "primary_mi300x",
        "case": case,
        "status": "ok" if final_status in {"passed", "retained"} else final_status,
        "final_status": final_status,
        "timing_boundary": timing_boundary,
        "dataset": {"num_qubits": 24, "num_terms": 512, "words": 1},
        "cpu_scalar_seconds": 1.0e-4,
        "rocm_runtime_version": "fixture-runtime",
        "rocm_toolkit_version": "fixture-toolkit",
        "hip_compiler_version": "fixture-compiler",
        "gpu_name": "fixture MI300X",
        "gfx_target": "gfx942",
        "build_command": "fixture build",
        "validation_command": "fixture validate",
        "profiler_command": "fixture profiler",
        "correctness_passed": True,
        "correctness_digest": {"label_hash": case, "output_terms": 128},
        "campaign7_terminal_statuses": TERMINAL_STATUSES,
    }


def _fixture_rows() -> list[dict[str, Any]]:
    rows = [
        _base_row(
            case="fixture_runtime",
            operation="release_source_build",
            mode="mi300x_repeatability",
            final_status="passed",
            timing_boundary="source_build",
        ),
        _base_row(
            case="fixture_transfer",
            operation="retained_operation_smoke",
            mode="retained_transfer",
            final_status="retained",
            timing_boundary="transfer_inclusive",
        ),
        _base_row(
            case="fixture_commutation",
            operation="retained_operation_smoke",
            mode="retained_commutation",
            final_status="retained",
            timing_boundary="device_resident",
        ),
        _base_row(
            case="fixture_consumers",
            operation="retained_operation_smoke",
            mode="retained_device_consumers",
            final_status="retained",
            timing_boundary="compact_consumer",
        ),
        _base_row(
            case="fixture_simplify",
            operation="retained_operation_smoke",
            mode="retained_simplify",
            final_status="retained",
            timing_boundary="device_resident",
        ),
        _base_row(
            case="fixture_expectation",
            operation="retained_operation_smoke",
            mode="retained_expectation",
            final_status="retained",
            timing_boundary="device_resident",
        ),
        _base_row(
            case="fixture_matmul",
            operation="retained_operation_smoke",
            mode="retained_matmul",
            final_status="retained",
            timing_boundary="device_resident",
        ),
        _base_row(
            case="fixture_profiler",
            operation="profiler_smoke",
            mode="rocprof_availability",
            final_status="passed",
            timing_boundary="profiler_only",
        ),
        _base_row(
            case="fixture_simplify_pressure",
            operation="duplicate_pressure_probe",
            mode="simplify_duplicate_pressure",
            final_status="rejected_with_evidence",
            timing_boundary="benchmark_only",
        ),
        _base_row(
            case="fixture_matmul_pressure",
            operation="duplicate_pressure_probe",
            mode="matmul_duplicate_pressure",
            final_status="rejected_with_evidence",
            timing_boundary="benchmark_only",
        ),
    ]
    rows[1]["hip_transfer_seconds"] = 2.0e-5
    rows[2]["hip_device_output_reuse_seconds"] = 9.0e-6
    rows[2]["hip_count_commuting_axis_none_seconds"] = 4.0e-6
    rows[3]["hip_device_output_reuse_seconds"] = 9.0e-6
    rows[3]["hip_count_commuting_axis_none_seconds"] = 4.0e-6
    rows[4]["hip_simplify_device_resident_seconds"] = 3.0e-5
    rows[5]["hip_expectation_device_resident_seconds"] = 8.0e-6
    rows[6]["hip_matmul_device_resident_seconds"] = 6.0e-5
    rows[8]["hip_simplify_device_resident_seconds"] = 1.1e-4
    rows[9]["hip_matmul_device_resident_seconds"] = 2.2e-4
    return rows


def _write_raw_report(data_dir: Path, rows: list[dict[str, Any]]) -> None:
    raw = data_dir / "raw"
    raw.mkdir(parents=True)
    (raw / "fixture.json").write_text(
        json.dumps(
            {
                "benchmark": "rocm_kernels",
                "profile": "campaign7-fixture",
                "git_commit": "fixture",
                "cases": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_rocm_campaign7_renderer_writes_summary_and_plots(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    plot_dir = tmp_path / "plots"
    _write_raw_report(data_dir, _fixture_rows())

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--data-dir",
            str(data_dir),
            "--plot-dir",
            str(plot_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["campaign"] == "rocm_mi300x_campaign7"
    assert set(summary["terminal_statuses"]) == REQUIRED_TERMINAL_KEYS
    assert summary["status_counts"]["retained"] == 6
    assert summary["status_counts"]["passed"] == 2

    release_svg = (
        plot_dir / "rocm_mi300x_campaign7_release_support.svg"
    ).read_text(encoding="utf-8")
    assert "fixture_transfer" in release_svg
    assert "fixture_matmul_pressure" in release_svg
    assert "fixture_commutation | HIP compact consumer" not in release_svg
    assert "fixture_consumers | HIP commutation" not in release_svg
    assert (plot_dir / "accelerator_landscape_with_rocm.svg").exists()


def test_rocm_campaign7_renderer_rejects_missing_terminal_status(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    plot_dir = tmp_path / "plots"
    rows = _fixture_rows()
    statuses = dict(rows[0]["campaign7_terminal_statuses"])
    statuses.pop("backend_neutral_accelerator_design")
    for row in rows:
        row["campaign7_terminal_statuses"] = statuses
    _write_raw_report(data_dir, rows)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--data-dir",
            str(data_dir),
            "--plot-dir",
            str(plot_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "terminal statuses" in completed.stderr

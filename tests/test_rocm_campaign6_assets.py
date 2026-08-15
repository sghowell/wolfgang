"""ROCm Campaign 6 report asset checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_rocm_campaign6_assets.py"
TERMINAL_STATUSES = {
    "expectation": "retained",
    "matmul": "retained",
    "external device pointers": "unavailable",
    "DLPack": "rejected_with_evidence",
    "CUDA Array Interface guard": "rejected_with_evidence",
    "streams": "rejected_with_evidence",
    "graphs": "rejected_with_evidence",
    "workspaces": "rejected_with_evidence",
    "portability": "out_of_scope_with_next_trigger",
    "ROCm wheels": "out_of_scope_with_next_trigger",
    "multi-GPU": "out_of_scope_with_next_trigger",
    "simultaneous CUDA+HIP": "unavailable",
}


def _retained_expectation_row() -> dict[str, Any]:
    return {
        "campaign": "rocm_mi300x_campaign6",
        "operation": "expectation_statevector",
        "mode": "host_complex128",
        "backend": "hip",
        "case": "fixture_expectation",
        "status": "ok",
        "final_status": "retained",
        "timing_boundary": "device_resident_kernel",
        "dataset": {"num_qubits": 2, "num_terms": 5, "state_size": 4, "words": 1},
        "cpu_scalar_seconds": 1.0e-6,
        "hip_expectation_transfer_seconds": 2.0e-6,
        "hip_expectation_device_resident_seconds": 8.0e-7,
        "hip_expectation_result_copy_seconds": None,
        "correctness_passed": True,
        "correctness_digest": {
            "label_hash": "expectation",
            "coefficient_l1": 3.0,
            "result_real": 1.0,
            "result_imag": 0.0,
        },
        "campaign6_terminal_statuses": TERMINAL_STATUSES,
    }


def _retained_matmul_row() -> dict[str, Any]:
    return {
        "campaign": "rocm_mi300x_campaign6",
        "operation": "matmul",
        "mode": "matmul_simplify_true",
        "backend": "hip",
        "case": "fixture_matmul",
        "status": "ok",
        "final_status": "retained",
        "timing_boundary": "device_resident_kernel",
        "dataset": {"num_qubits": 24, "lhs_terms": 16, "rhs_terms": 16, "words": 1},
        "cpu_scalar_seconds": 1.0e-4,
        "hip_matmul_transfer_seconds": 2.0e-4,
        "hip_matmul_device_resident_seconds": 4.0e-5,
        "hip_matmul_to_host_seconds": 1.0e-5,
        "hip_matmul_simplify_output": True,
        "correctness_passed": True,
        "correctness_digest": {
            "label_hash": "matmul",
            "coefficient_l1": 17.0,
            "output_terms": 32,
        },
        "campaign6_terminal_statuses": TERMINAL_STATUSES,
    }


def _write_raw_report(data_dir: Path, rows: list[dict[str, Any]]) -> None:
    raw = data_dir / "raw"
    raw.mkdir(parents=True)
    (raw / "fixture.json").write_text(
        json.dumps(
            {
                "benchmark": "rocm_kernels",
                "profile": "campaign6-fixture",
                "git_commit": "fixture",
                "cases": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_rocm_campaign6_renderer_writes_summary_and_plots(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    plot_dir = tmp_path / "plots"
    _write_raw_report(data_dir, [_retained_expectation_row(), _retained_matmul_row()])

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
    assert summary["campaign"] == "rocm_mi300x_campaign6"
    assert {row["operation"] for row in summary["rows"]} == {
        "expectation_statevector",
        "matmul",
    }
    assert summary["status_counts"] == {"retained": 2}
    assert TERMINAL_STATUSES.items() <= summary["terminal_statuses"].items()

    parity_svg = (plot_dir / "rocm_mi300x_campaign6_parity.svg").read_text(encoding="utf-8")
    assert "fixture_expectation" in parity_svg
    assert "fixture_matmul" in parity_svg
    assert (plot_dir / "accelerator_landscape_with_rocm.svg").exists()


def test_rocm_campaign6_renderer_rejects_missing_retained_operation(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    plot_dir = tmp_path / "plots"
    missing = _retained_expectation_row()
    _write_raw_report(data_dir, [missing])

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
    assert "retained evidence is missing" in completed.stderr

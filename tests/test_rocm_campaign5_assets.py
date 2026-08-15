"""ROCm Campaign 5 report asset checks."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "docs/benchmarks/data/rocm_mi300x_campaign5_2026-04-30"
PLOT_DIR = ROOT / "docs/benchmarks/plots"
SCRIPT = ROOT / "scripts/render_rocm_campaign5_assets.py"
PLOT_NAMES = {
    "rocm_mi300x_campaign5_interop.svg",
    "accelerator_landscape_with_rocm.svg",
}
TERMINAL_ITEMS = {
    "DLPack",
    "CUDA Array Interface guard",
    "streams",
    "graphs",
    "workspaces",
    "expectation",
    "matmul",
    "portability",
    "ROCm wheels",
    "multi-GPU",
    "simultaneous CUDA+HIP",
}


def _normalize_summary_paths(
    value: Any,
    *,
    data_dir: Path,
    plot_dir: Path,
) -> Any:
    data_markers = {
        str(data_dir),
        str(DATA_DIR),
        str(DATA_DIR.relative_to(ROOT)),
    }
    plot_markers = {
        str(plot_dir),
        str(PLOT_DIR),
        str(PLOT_DIR.relative_to(ROOT)),
    }
    if isinstance(value, str):
        normalized = value
        for marker in sorted(data_markers, key=len, reverse=True):
            normalized = normalized.replace(marker, "<DATA_DIR>")
        for marker in sorted(plot_markers, key=len, reverse=True):
            normalized = normalized.replace(marker, "<PLOT_DIR>")
        return normalized
    if isinstance(value, list):
        return [
            _normalize_summary_paths(item, data_dir=data_dir, plot_dir=plot_dir)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _normalize_summary_paths(item, data_dir=data_dir, plot_dir=plot_dir)
            for key, item in value.items()
        }
    return value


def test_rocm_campaign5_checked_summary_and_plots() -> None:
    summary = json.loads((DATA_DIR / "summary.json").read_text(encoding="utf-8"))

    assert summary["campaign"] == "rocm_mi300x_campaign5"
    assert (PLOT_DIR / "rocm_mi300x_campaign5_interop.svg").exists()
    assert (PLOT_DIR / "accelerator_landscape_with_rocm.svg").exists()
    assert len(summary["rows"]) >= 12
    assert set(summary["terminal_statuses"]) >= TERMINAL_ITEMS
    assert summary["terminal_statuses"]["DLPack"] == "rejected_with_evidence"
    assert summary["landscape_refreshed_with_campaign5"] is False
    assert summary["profiler_artifacts"]

    rows = summary["rows"]
    assert all(row["final_status"] for row in rows)
    dlpack_rows = [row for row in rows if str(row["mode"]).startswith("dlpack_")]
    assert dlpack_rows
    for row in dlpack_rows:
        assert row["consumer_library"]
        assert "consumer_available" in row
        assert "consumer_read_only_enforced" in row
        if row["final_status"] == "retained":
            assert row["hip_dlpack_device_type"] == 10
            assert row["hip_dlpack_device_type_name"] == "kDLROCM"

    guard_rows = [row for row in rows if row["mode"] == "cuda_array_interface_guard"]
    assert guard_rows
    assert all(row["final_status"] != "retained" for row in guard_rows)

    consumer_status = {
        item["consumer_library"]: item for item in summary["consumer_summary"]
    }
    assert consumer_status["torch"]["consumer_backend"] == "rocm"
    assert consumer_status["torch"]["candidate_probe_mutation_result"] == "accepted_mutation"
    assert (
        consumer_status["torch"]["candidate_probe_evidence_kind"]
        == "temporary_candidate_build"
    )
    assert consumer_status["torch"]["candidate_probe_source_file"]
    assert consumer_status["torch"]["consumer_read_only_enforced"] is False
    assert summary["candidate_probe_artifacts"]

    series = {point["series"] for point in summary["readme_performance_landscape"]}
    assert {
        "CPU scalar",
        "CUDA device-resident",
        "ROCm HIP resident",
        "ROCm HIP commutation allocate",
    } <= series


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_rocm_campaign5_renderer_reproduces_summary_and_plots(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    plots = tmp_path / "plots"
    shutil.copytree(DATA_DIR, data_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
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
    assert summary["campaign"] == "rocm_mi300x_campaign5"
    assert summary["terminal_statuses"]
    assert {path.name for path in plots.glob("*.svg")} == PLOT_NAMES

    checked_summary = json.loads((DATA_DIR / "summary.json").read_text(encoding="utf-8"))
    assert _normalize_summary_paths(
        summary,
        data_dir=data_dir,
        plot_dir=plots,
    ) == _normalize_summary_paths(
        checked_summary,
        data_dir=DATA_DIR,
        plot_dir=PLOT_DIR,
    )

    assert (plots / "rocm_mi300x_campaign5_interop.svg").read_text(
        encoding="utf-8"
    ) == (PLOT_DIR / "rocm_mi300x_campaign5_interop.svg").read_text(
        encoding="utf-8"
    )

    # The shared README landscape filename is intentionally owned by the latest
    # campaign. Historical renderers must still emit a valid broad landscape,
    # but they should not force the checked canonical landscape back in time.
    landscape = (plots / "accelerator_landscape_with_rocm.svg").read_text(
        encoding="utf-8"
    )
    assert "FastPauli Accelerator Performance Landscape" in landscape

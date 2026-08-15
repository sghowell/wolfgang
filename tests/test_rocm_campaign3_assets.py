"""ROCm Campaign 3 report asset checks."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30"
PLOT_DIR = ROOT / "docs/benchmarks/plots"
SCRIPT = ROOT / "scripts/render_rocm_campaign3_assets.py"
PLOT_NAMES = {
    "rocm_mi300x_campaign3_simplify.svg",
    "accelerator_landscape_with_rocm.svg",
}
REPORT_LOCAL_PLOT_NAMES = {"rocm_mi300x_campaign3_simplify.svg"}


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


def test_rocm_campaign3_checked_summary_and_plots() -> None:
    summary = json.loads((DATA_DIR / "summary.json").read_text(encoding="utf-8"))

    assert summary["campaign"] == "rocm_mi300x_campaign3"
    assert (PLOT_DIR / "rocm_mi300x_campaign3_simplify.svg").exists()
    assert (PLOT_DIR / "accelerator_landscape_with_rocm.svg").exists()
    assert len(summary["rows"]) >= 6
    assert {
        "DLPack",
        "streams",
        "workspaces",
        "packed summaries",
        "expectation",
        "matmul",
        "portability",
        "ROCm wheels",
        "multi-GPU",
        "simultaneous CUDA+HIP",
    } <= set(summary["headroom_statuses"])

    decisions = {
        item["strategy"]: item["status"]
        for item in summary["strategy_decisions"]
    }
    assert decisions["rocthrust_default"] == "retained"
    assert decisions["hipcub_radix_sort_reduce"] == "rejected_with_evidence"

    series = {point["series"] for point in summary["readme_performance_landscape"]}
    assert {
        "CPU scalar",
        "CUDA device-resident",
        "ROCm HIP resident",
        "ROCm HIP commutation allocate",
    } <= series


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_rocm_campaign3_renderer_reproduces_summary_and_plots(tmp_path: Path) -> None:
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
    assert summary["campaign"] == "rocm_mi300x_campaign3"
    assert summary["headroom_statuses"]
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

    for name in REPORT_LOCAL_PLOT_NAMES:
        assert (plots / name).read_text(encoding="utf-8") == (
            PLOT_DIR / name
        ).read_text(encoding="utf-8")

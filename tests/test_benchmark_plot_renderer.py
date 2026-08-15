"""Benchmark plot renderer tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cuda_plot_renderer_includes_cpu_variants_and_cuda_paths(tmp_path: Path) -> None:
    report = tmp_path / "cuda_report.md"
    output = tmp_path / "plot.svg"
    report.write_text(
        "\n".join(
            [
                "# CUDA Evidence Test",
                "",
                "## Benchmark Default",
                "",
                "| Case | Dataset | CPU Scalar Seconds | CPU Optimized | CUDA Transfer-Inclusive Seconds | CUDA Device-Resident Seconds | Regime |",
                "| --- | --- | ---: | --- | ---: | ---: | --- |",
                "| pairwise_commutation | entries=1024 | 0.008 | avx512: 0.004; tbb: 0.002 | 0.002 | 0.001 | CUDA-faster |",
                "| simplify_duplicate_pressure | terms=4096 | 0.006 | n/a | 0.003 | 0.0015 | CUDA-faster |",
                "",
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "render_benchmark_plots.py"),
            "--cuda-report",
            str(report),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    rendered = output.read_text(encoding="utf-8")
    assert "FastPauli H100 CUDA backend speedups" in rendered
    assert "pairwise commutation" in rendered
    assert "CPU scalar 1.00x" in rendered
    assert "CPU avx512 2.00x" in rendered
    assert "CPU tbb 4.00x" in rendered
    assert "CUDA transfer 4.00x" in rendered
    assert "CUDA resident 8.00x" in rendered
    assert "simplify duplicate pressure" in rendered


def test_checked_in_cuda_plot_is_fresh(tmp_path: Path) -> None:
    generated = tmp_path / "generated.svg"
    report = Path("docs/benchmarks/reports/cuda_h100_nsight_hillclimb_2026-04-28.md")
    checked_in = ROOT / "docs" / "benchmarks" / "plots"
    checked_in = checked_in / "cuda_h100_nsight_hillclimb_default_backend_speedups.svg"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "render_benchmark_plots.py"),
            "--cuda-report",
            str(report),
            "--output",
            str(generated),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert generated.read_text(encoding="utf-8") == checked_in.read_text(encoding="utf-8")

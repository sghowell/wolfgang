from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = "docs/plans/apple_metal_optimization_campaign5_plan.md"
REPORT_PATH = "docs/benchmarks/reports/apple_metal_optimization_campaign5_2026-05-06.md"
DATA_DIR = ROOT / "docs/benchmarks/data/apple_metal_optimization_campaign5_2026-05-06"
PLOT_PATH = ROOT / "docs/benchmarks/plots/accelerator_landscape_with_rocm.svg"
RENDERER = ROOT / "scripts/render_apple_metal_assets.py"


def read(path: str | Path) -> str:
    return (ROOT / path if isinstance(path, str) else path).read_text(encoding="utf-8")


def test_metal_benchmark_declares_campaign5_profile() -> None:
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
    assert "campaign5" in payload["profiles"]
    cases = payload["profiles"]["campaign5"]
    assert {case["profile"] for case in cases} == {"campaign5"}
    assert {case["operation"] for case in cases} == {"simplify"}
    assert {
        "metal_campaign5_simplify_words1_duplicate_heavy_8192_terms",
        "metal_campaign5_simplify_words1_duplicate_light_8192_terms",
        "metal_campaign5_simplify_words2_duplicate_heavy_4096_terms",
        "metal_campaign5_simplify_generic_multiword_2048_terms",
        "metal_campaign5_simplify_cancellation_4096_terms",
    } <= {case["name"] for case in cases}
    assert {case["packed_words"] for case in cases} >= {1, 2, 3}
    assert all("matrix_entries" not in case for case in cases)

    validate_source = read("scripts/validate.py")
    assert "Apple Metal Campaign 5 simplify benchmark smoke" in validate_source
    assert '"campaign5"' in validate_source


def test_campaign5_boundary_is_registered_in_benchmark_metadata() -> None:
    spec = importlib.util.spec_from_file_location(
        "benchmark_metadata",
        ROOT / "benchmarks" / "_benchmark_metadata.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert (
        "device_to_host_cpu_simplify_host_to_device"
        in module.ACCELERATOR_TRANSFER_BOUNDARIES
    )


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_campaign5_checked_assets_record_transfer_reference_boundary() -> None:
    summary = json.loads((DATA_DIR / "summary.json").read_text(encoding="utf-8"))
    raw = json.loads((DATA_DIR / "raw" / "metal_benchmark_campaign5.json").read_text(encoding="utf-8"))
    report = read(REPORT_PATH)
    plot = read(PLOT_PATH)
    readme = read("docs/research/provenance.md")
    normalized_readme = " ".join(readme.split())
    protocol = read("docs/benchmarks/protocol.md")
    bindings = read("bindings/python/pauli_sum_py.cpp")
    package_init = read("python/fastpauli/__init__.py")

    assert summary["campaign"] == "apple_metal_optimization_campaign5"
    assert summary["status"] == "ok"
    assert summary["profile"] == "campaign5"
    assert summary["source_benchmark"].endswith("raw/metal_benchmark_campaign5.json")
    assert summary["metal_status"]["runtime_available"] is True
    assert raw["profile"] == "campaign5"
    assert raw["benchmark"] == "apple_metal_kernels"

    for token in (
        "Apple Metal Campaign 5",
        "metal_simplify_transfer_reference",
        "device_to_host_cpu_simplify_host_to_device",
        "correctness bridge, not a device-resident GPU duplicate-reduction path",
    ):
        assert token in report
        assert token in normalized_readme

    assert "Apple Metal simplify transfer reference" in json.dumps(
        summary["readme_performance_landscape"],
        sort_keys=True,
    )
    assert "Apple Metal simplify transfer reference" in plot
    assert "operation: simplify" in protocol
    assert "cpu_scalar" in protocol
    assert "metal_simplify_strategy_status" in protocol
    assert "Metal source builds" in bindings
    assert "transfer-reference correctness bridge" in bindings
    assert "source-build simplify transfer-reference correctness bridge" in package_init

    simplify_rows = [row for row in raw["cases"] if row.get("operation") == "simplify"]
    assert simplify_rows
    transfer_rows = [
        row for row in simplify_rows if row["variant"] == "metal_simplify_transfer_reference"
    ]
    assert transfer_rows
    for row in transfer_rows:
        assert row["object_backend"] == "metal"
        assert row["correct"] is True
        assert row["transfer_boundary"] == "device_to_host_cpu_simplify_host_to_device"
        assert row["metal_execution"]["kernel"] == "not_applicable_no_metal_kernel_dispatch"
        assert row["metal_simplify_strategy"] == "transfer_reference"
        assert row["metal_simplify_strategy_status"] == "retained"
        assert row["output_terms"] == row["case"]["output_terms"]


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_campaign5_renderer_check_mode_validates_checked_assets() -> None:
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
    assert "Apple Metal Campaign 5 assets validated" in completed.stdout

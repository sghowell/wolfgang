from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/plans/mi300x_rocm_optimization_campaign7_plan.md"
BENCHMARK = ROOT / "benchmarks/bench_rocm_kernels.py"
RELEASE_LANE = ROOT / "scripts/run_rocm_release_support_lane.py"


REQUIRED_TERMINAL_KEYS = {
    "mi300x_repeatability",
    "cpu_only_control",
    "rocm_source_build_runbook",
    "rocm_ci_or_release_lane",
    "rocm_packaging_policy",
    "rocm_wheel_support",
    "alternate_amd_gpu_portability",
    "profiler_availability",
    "duplicate_pressure_simplify",
    "duplicate_pressure_matmul",
    "external_statevector_interop",
    "hip_dlpack",
    "hip_cuda_array_interface",
    "public_streams",
    "public_graphs",
    "public_workspaces",
    "multi_gpu_rocm",
    "simultaneous_cuda_hip",
    "backend_neutral_accelerator_design",
}


def load_validate_module() -> ModuleType:
    validate_path = ROOT / "scripts/validate.py"
    spec = importlib.util.spec_from_file_location("fastpauli_validate", validate_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rocm_campaign7_plan_exists_and_covers_campaign6_headroom() -> None:
    text = PLAN.read_text(encoding="utf-8")
    protocol = (ROOT / "docs/benchmarks/protocol.md").read_text(encoding="utf-8")

    assert "Wave 5 ROCm portability, CI, packaging, and release-support evidence" in text
    assert "previous campaign report: docs/benchmarks/reports/rocm_mi300x_campaign6_2026-04-30.md" in text
    for key in REQUIRED_TERMINAL_KEYS:
        assert key in text
        assert key in protocol


def test_rocm_campaign7_plan_is_registered_as_completed_release_plan() -> None:
    plan_path = "docs/plans/mi300x_rocm_optimization_campaign7_plan.md"
    report_path = "docs/benchmarks/reports/rocm_mi300x_campaign7_2026-04-30.md"

    readme = (ROOT / "docs/research/provenance.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs/roadmap.md").read_text(encoding="utf-8")
    waves = (ROOT / "docs/plans/rocm_next_waves_plan.md").read_text(encoding="utf-8")
    backend = (ROOT / "docs/architecture/rocm_backend.md").read_text(encoding="utf-8")
    release = (ROOT / "docs/quality/release_and_packaging.md").read_text(encoding="utf-8")
    protocol = (ROOT / "docs/benchmarks/protocol.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert "../plans/mi300x_rocm_optimization_campaign7_plan.md" in readme
    assert report_path in roadmap
    assert "The MI300X ROCm Campaign 7 slice" in roadmap
    assert plan_path in roadmap
    assert plan_path in waves
    assert "../benchmarks/reports/rocm_mi300x_campaign7_2026-04-30.md" in readme
    assert report_path in waves
    assert report_path in agents
    assert plan_path in backend
    assert report_path in backend
    assert "Campaign 7" in release
    assert "ROCm Campaign 7 release-support rows" in protocol

    validate = load_validate_module()
    assert plan_path in validate.SOURCE_OF_TRUTH_PATHS


def test_rocm_campaign7_benchmark_profiles_are_registered() -> None:
    source = BENCHMARK.read_text(encoding="utf-8")

    for profile in (
        "campaign7-release-smoke",
        "campaign7-duplicate-pressure",
        "campaign7-profiler",
    ):
        assert profile in source

    assert "CAMPAIGN7_TERMINAL_STATUSES" in source
    for key in REQUIRED_TERMINAL_KEYS:
        assert key in source


def test_rocm_campaign7_release_lane_prints_expected_commands() -> None:
    completed = subprocess.run(
        [sys.executable, str(RELEASE_LANE), "--print-commands"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    output = completed.stdout
    for required in (
        "host-inventory",
        "package-metadata",
        "cpu-only-control",
        "hip-source-build",
        "hip-pytest",
        "cuda-hip-rejection",
        "campaign7-release-smoke",
        "campaign7-duplicate-pressure",
        "campaign7-profiler",
        "rocprof",
        "render-assets",
        "report-validation",
    ):
        assert required in output
    assert "FASTPAULI_ENABLE_HIP=ON" in output
    assert "FASTPAULI_HIP_ARCHITECTURES=gfx942" in output
    assert "|| true" not in output
    assert "test $status -ne 0" in output
    assert "grep -q 'cannot both be ON'" in output

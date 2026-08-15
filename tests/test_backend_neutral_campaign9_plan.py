from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = "docs/plans/backend_neutral_accelerator_campaign9_plan.md"
PLAN = ROOT / PLAN_PATH
REPORT_PATH = "docs/benchmarks/reports/backend_neutral_accelerator_campaign9_2026-05-01.md"
REPORT = ROOT / REPORT_PATH
SUMMARY_PATH = (
    "docs/benchmarks/data/backend_neutral_accelerator_campaign9_2026-05-01/summary.json"
)
SUMMARY = ROOT / SUMMARY_PATH

REQUIRED_KEYS = {
    "backend_neutral_status_schema",
    "object_local_backend_identity",
    "backend_construction_selector_contract",
    "device_commutation_matrix_backend_property",
    "ambiguous_dual_runtime_policy",
    "target_specific_accelerator_builds",
    "mixed_cuda_hip_build_rejection",
    "future_multi_runtime_design_gate",
    "same_backend_same_device_validation",
    "cpu_only_header_safety",
    "cuda_target_regression_lane",
    "hip_target_regression_lane",
    "benchmark_boundary_reporting",
    "no_wheel_or_portability_claim",
}

REQUIRED_OUT_OF_SCOPE = {
    "ROCm wheels",
    "non-MI300X AMD portability",
    "HIP DLPack",
    "HIP CUDA Array Interface",
    "multi-GPU ROCm",
    "Metal/MPS",
    "combined CUDA+ROCm wheels",
    "normal builds that link both CUDA and HIP runtimes",
    "mixed NVIDIA+AMD host validation as a completion requirement",
}


def load_validate_module() -> ModuleType:
    validate_path = ROOT / "scripts/validate.py"
    spec = importlib.util.spec_from_file_location("fastpauli_validate", validate_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize(text: str) -> str:
    return " ".join(text.split())


def test_backend_neutral_campaign9_plan_exists_and_is_executable() -> None:
    text = PLAN.read_text(encoding="utf-8")

    assert text.startswith("# Backend-Neutral Accelerator Campaign 9 Implementation Plan")
    assert "**Goal:**" in text
    assert "**Architecture:**" in text
    assert "**Tech Stack:**" in text
    assert "status: completed_target_specific_closeout_lanes" in text
    assert "trigger: accepted backend-neutral API scope for target-specific accelerator builds" in text
    assert "docs/architecture/backend_neutral_accelerators.md" in text
    assert "docs/benchmarks/reports/rocm_campaign8_architecture_readiness_2026-05-01.md" in text
    assert REPORT_PATH in text

    for key in REQUIRED_KEYS:
        assert key in text

    for phrase in REQUIRED_OUT_OF_SCOPE:
        assert phrase in text

    assert "does not require an uncommon mixed NVIDIA+AMD host" in normalize(text)
    assert "It is not a Campaign 9 completion gate" in normalize(text)
    assert "Current implementation and validation status" in text
    assert "implemented: PauliSum.to_device" in text
    assert "completed: CUDA-only remote regression refresh" in text
    assert "completed: HIP-only remote regression refresh" in text
    assert "completed: configure-time rejection evidence" in text
    assert SUMMARY_PATH in text
    assert 'PauliSum.to_device(backend="cuda")' in text
    assert 'PauliSum.to_device(backend="hip")' in text
    assert "future mixed-runtime build with both CUDA and HIP runtimes visible and omitted backend" in text
    assert "raise an ambiguous-backend error" in text


def test_backend_neutral_campaign9_plan_is_registered_as_source_of_truth() -> None:
    docs = {
        "docs/research/provenance.md": ROOT / "docs/research/provenance.md",
        "AGENTS.md": ROOT / "AGENTS.md",
        "docs/roadmap.md": ROOT / "docs/roadmap.md",
        "docs/plans/rocm_next_waves_plan.md": ROOT / "docs/plans/rocm_next_waves_plan.md",
        "docs/architecture/backend_neutral_accelerators.md": ROOT
        / "docs/architecture/backend_neutral_accelerators.md",
        "docs/architecture/cuda_backend.md": ROOT / "docs/architecture/cuda_backend.md",
        "docs/architecture/rocm_backend.md": ROOT / "docs/architecture/rocm_backend.md",
        "docs/architecture/hardware_targets_and_testing.md": ROOT
        / "docs/architecture/hardware_targets_and_testing.md",
        "docs/benchmarks/protocol.md": ROOT / "docs/benchmarks/protocol.md",
    }

    for label, path in docs.items():
        text = path.read_text(encoding="utf-8")
        expected = (
            "../plans/backend_neutral_accelerator_campaign9_plan.md"
            if label == "docs/research/provenance.md"
            else PLAN_PATH
        )
        assert expected in text, label

    assert "../benchmarks/reports/backend_neutral_accelerator_campaign9_2026-05-01.md" in (
        ROOT / "docs/research/provenance.md"
    ).read_text(encoding="utf-8")
    assert REPORT_PATH in (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert REPORT_PATH in (ROOT / "docs/roadmap.md").read_text(encoding="utf-8")

    validate = load_validate_module()
    assert PLAN_PATH in validate.SOURCE_OF_TRUTH_PATHS


def test_backend_neutral_campaign9_protocol_records_required_boundaries() -> None:
    protocol = (ROOT / "docs/benchmarks/protocol.md").read_text(encoding="utf-8")
    plan = PLAN.read_text(encoding="utf-8")

    assert "campaign: backend_neutral_accelerator_campaign9" in protocol
    assert "build_mode: cpu_only, cuda_only, hip_only, or metal_only" in protocol
    assert "transfer_boundary: transfer_inclusive, device_resident, host_materialized, compact_consumer, or status_only" in protocol

    for key in REQUIRED_KEYS:
        assert key in protocol
        assert key in plan

    assert "not as a speedup claim" in protocol
    assert "status_only" in protocol
    assert "Mixed CUDA+HIP source-build evidence is future-only" in protocol


def test_backend_neutral_campaign9_current_support_boundary_stays_explicit() -> None:
    readme = (ROOT / "docs/research/provenance.md").read_text(encoding="utf-8")
    backend = (ROOT / "docs/architecture/backend_neutral_accelerators.md").read_text(
        encoding="utf-8"
    )
    rocm_backend = (ROOT / "docs/architecture/rocm_backend.md").read_text(encoding="utf-8")
    waves = (ROOT / "docs/plans/rocm_next_waves_plan.md").read_text(encoding="utf-8")

    assert "CPU-only, CUDA-target, HIP-target, and Apple Metal-target builds are the supported normal modes" in normalize(
        readme
    )
    assert "CUDA Campaign 10 closes every H100 CUDA Campaign 9" in readme
    assert "configure-time error by policy" in backend
    assert "not a ROCm wheel, non-MI300X AMD" in normalize(rocm_backend)
    assert "does not claim ROCm wheels, non-MI300X AMD portability" in normalize(waves)
    assert "configure-time rejection of mixed CUDA+HIP requests" in waves


def test_backend_neutral_campaign9_checked_report_matches_summary() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    report = REPORT.read_text(encoding="utf-8")

    assert summary["campaign"] == "backend_neutral_accelerator_campaign9"
    assert summary["status"] == "completed_target_specific_closeout_lanes"
    assert summary["validation_git_revision"] == "ec8fd19"

    expected_lanes = {
        "local_cpu_only",
        "cuda_h100",
        "hip_mi300x",
        "dual_request_rejection",
    }
    assert set(summary["lanes"]) == expected_lanes
    assert all(lane["status"] == "passed" for lane in summary["lanes"].values())

    assert summary["lanes"]["cuda_h100"]["build_mode"] == "cuda_only"
    assert summary["lanes"]["cuda_h100"]["requested_cuda_architectures_built"] == "90"
    assert summary["lanes"]["hip_mi300x"]["build_mode"] == "hip_only"
    assert summary["lanes"]["hip_mi300x"]["hip_architectures"] == "gfx942"
    assert summary["lanes"]["dual_request_rejection"]["exit_code"] == 1

    for key, status in summary["terminal_statuses"].items():
        assert key in REQUIRED_KEYS
        assert key in report
        assert status in report

    assert "not new optimization claims" in report
    assert "combined CUDA+HIP runtime or wheel" in report


@pytest.mark.skip(reason="requires the private raw benchmark archive")
def test_backend_neutral_campaign9_summary_is_bound_to_raw_logs() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    logs = {
        key: (ROOT / path).read_text(encoding="utf-8")
        for key, path in summary["evidence_logs"].items()
    }

    assert "242 passed, 89 skipped in 18.06s" in logs["local_cpu"]
    assert "'compiled_backends': ['cpu']" in logs["local_cpu"]
    assert "'runtime_visible_backends': ['cpu']" in logs["local_cpu"]
    assert "Successfully built fastpauli-0.1.0.tar.gz" in logs["local_cpu"]

    assert "NVIDIA H100 80GB HBM3" in logs["cuda_h100"]
    assert "'accelerator_build_mode': 'cuda_only'" in logs["cuda_h100"]
    assert "'compiled_accelerator_backends': ['cuda']" in logs["cuda_h100"]
    assert "'runtime_visible_accelerator_backends': ['cuda']" in logs["cuda_h100"]
    assert "CUDA-enabled semantic pytest" in logs["cuda_h100"]
    assert "264 passed, 67 skipped in 11.64s" in logs["cuda_h100"]
    assert "30 passed, 8 skipped in 1.50s" in logs["cuda_h100"]

    assert "AMD Instinct MI300X VF" in logs["hip_mi300x"]
    assert "'accelerator_build_mode': 'hip_only'" in logs["hip_mi300x"]
    assert "'compiled_accelerator_backends': ['hip']" in logs["hip_mi300x"]
    assert "'runtime_visible_accelerator_backends': ['hip']" in logs["hip_mi300x"]
    assert "HIP-enabled semantic pytest" in logs["hip_mi300x"]
    assert "270 passed, 61 skipped in 13.44s" in logs["hip_mi300x"]
    assert '"correctness_passed": true' in logs["hip_mi300x"]

    assert "exit_code=1" in logs["dual_request_rejection"]
    assert "target-specific accelerator build policy" in logs["dual_request_rejection"]

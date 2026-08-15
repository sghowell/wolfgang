from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/plans/mi300x_rocm_optimization_campaign8_plan.md"
READINESS_LANE = ROOT / "scripts/run_rocm_campaign8_readiness_lane.py"
ARCHIVE_PORTABILITY = ROOT / "scripts/archive_portability.py"


REQUIRED_TERMINAL_KEYS = {
    "backend_neutral_object_model",
    "simultaneous_cuda_hip_source_builds",
    "multi_gpu_rocm_execution",
    "non_mi300x_amd_portability",
    "rocm_wheel_packaging_design",
    "rocm_ci_hardware_policy",
    "rocm_clean_machine_install_tests",
    "rocprofv3_migration",
    "legacy_rocprof_retention",
    "external_hip_statevector_contract",
    "hip_dlpack_reconsideration_contract",
    "hip_cuda_array_interface_policy",
    "public_streams_policy",
    "public_graphs_policy",
    "public_workspaces_policy",
    "targeted_rocm_performance_reopen",
    "source_build_release_lane_retention",
}


def extract_text_fence_after(text: str, marker: str) -> set[str]:
    marker_index = text.index(marker)
    fence_start = text.index("```text", marker_index)
    block_start = text.index("\n", fence_start) + 1
    block_end = text.index("```", block_start)
    return {
        line.strip()
        for line in text[block_start:block_end].splitlines()
        if line.strip()
    }


def load_validate_module() -> ModuleType:
    validate_path = ROOT / "scripts/validate.py"
    spec = importlib.util.spec_from_file_location("fastpauli_validate", validate_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rocm_campaign8_plan_exists_and_covers_campaign7_residual_risk() -> None:
    text = PLAN.read_text(encoding="utf-8")
    protocol = (ROOT / "docs/benchmarks/protocol.md").read_text(encoding="utf-8")

    assert "Wave 6 backend-neutral and long-horizon accelerator work" in text
    assert "previous campaign report: docs/benchmarks/reports/rocm_mi300x_campaign7_2026-04-30.md" in text
    assert "Campaign 8 should address every Campaign 7 residual-risk item" in text
    assert extract_text_fence_after(text, "Required terminal-status keys:") == REQUIRED_TERMINAL_KEYS
    assert (
        extract_text_fence_after(
            protocol,
            "`campaign8_terminal_statuses` must contain this exact residual-status key set:",
        )
        == REQUIRED_TERMINAL_KEYS
    )


def test_rocm_campaign8_plan_is_registered_as_latest_rocm_plan() -> None:
    plan_path = "docs/plans/mi300x_rocm_optimization_campaign8_plan.md"
    report_path = "docs/benchmarks/reports/rocm_campaign8_architecture_readiness_2026-05-01.md"

    readme = (ROOT / "docs/research/provenance.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs/roadmap.md").read_text(encoding="utf-8")
    waves = (ROOT / "docs/plans/rocm_next_waves_plan.md").read_text(encoding="utf-8")
    backend = (ROOT / "docs/architecture/rocm_backend.md").read_text(encoding="utf-8")
    release = (ROOT / "docs/quality/release_and_packaging.md").read_text(encoding="utf-8")
    protocol = (ROOT / "docs/benchmarks/protocol.md").read_text(encoding="utf-8")

    assert "../plans/mi300x_rocm_optimization_campaign8_plan.md" in readme
    assert plan_path in agents
    assert f"Latest ROCm/HIP campaign plan: {plan_path}" in roadmap
    assert f"Latest ROCm/HIP report: {report_path}" in roadmap
    assert "../benchmarks/reports/rocm_campaign8_architecture_readiness_2026-05-01.md" in readme
    assert report_path in agents
    assert report_path in waves
    assert plan_path in waves
    assert plan_path in backend
    assert plan_path in release
    assert "ROCm Campaign 8 architecture-readiness rows" in protocol

    validate = load_validate_module()
    assert plan_path in validate.SOURCE_OF_TRUTH_PATHS


def test_rocm_campaign8_contract_docs_are_linked() -> None:
    backend_neutral = "docs/architecture/backend_neutral_accelerators.md"
    profiler_decision = "docs/plans/rocm_profiler_migration_campaign8_decision.md"
    interop_decision = "docs/plans/rocm_hip_interop_reconsideration_campaign8_decision.md"

    for path in (backend_neutral, profiler_decision, interop_decision):
        assert (ROOT / path).exists()

    cuda_backend = (ROOT / "docs/architecture/cuda_backend.md").read_text(encoding="utf-8")
    rocm_backend = (ROOT / "docs/architecture/rocm_backend.md").read_text(encoding="utf-8")
    api_stability = (ROOT / "docs/architecture/api_stability.md").read_text(encoding="utf-8")
    hardware = (ROOT / "docs/architecture/hardware_targets_and_testing.md").read_text(encoding="utf-8")

    assert backend_neutral in cuda_backend
    assert backend_neutral in rocm_backend
    assert backend_neutral in api_stability
    assert profiler_decision in rocm_backend
    assert interop_decision in rocm_backend
    assert "docs/plans/mi300x_rocm_optimization_campaign8_plan.md" in hardware


def test_rocm_campaign8_readiness_lane_prints_and_writes_evidence(tmp_path: Path) -> None:
    required_labels = {
        "host-inventory",
        "cpu-only-control",
        "cuda-hip-rejection",
        "hip-source-build-mi300x",
        "hip-source-build-alternate-amd",
        "hip-retained-operation-tests",
        "rocm-release-smoke",
        "rocprof-legacy",
        "rocprofv3",
        "clean-machine-sdist-install",
        "packaging-policy-check",
        "render-assets",
        "report-validation",
    }

    printed = subprocess.run(
        [sys.executable, str(READINESS_LANE), "--print-commands"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert printed.returncode == 0, printed.stderr
    for label in required_labels:
        assert label in printed.stdout
    assert "requires a real non-MI300X AMD GPU host" in printed.stdout

    evidence_dir = tmp_path / "evidence"
    written = subprocess.run(
        [sys.executable, str(READINESS_LANE), "--write-evidence", str(evidence_dir)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert written.returncode == 0, written.stderr

    command_json = json.loads((evidence_dir / "raw/readiness_commands.json").read_text(encoding="utf-8"))
    assert {item["label"] for item in command_json["commands"]} == required_labels

    summary = json.loads((evidence_dir / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["campaign8_terminal_statuses"]) == REQUIRED_TERMINAL_KEYS
    assert summary["runtime_changes"] == "none"
    assert summary["local_cpu_only_validation"]["status"] == "external_closeout_required"
    assert summary["cuda_hip_configure_rejection"]["status"] == "external_closeout_required"
    assert "only records the command contract" in summary["local_cpu_only_validation"]["evidence"]
    assert "only records the expected configure failure" in summary["cuda_hip_configure_rejection"]["evidence"]


def test_rocm_campaign8_readiness_lane_archive_mode_requires_explicit_source_commit(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive-root"
    scripts_dir = archive_root / "scripts"
    scripts_dir.mkdir(parents=True)
    shutil.copy2(READINESS_LANE, scripts_dir / READINESS_LANE.name)
    shutil.copy2(ARCHIVE_PORTABILITY, scripts_dir / ARCHIVE_PORTABILITY.name)

    completed = subprocess.run(
        [sys.executable, str(scripts_dir / READINESS_LANE.name), "--write-evidence", str(tmp_path / "evidence")],
        cwd=archive_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "source" in (completed.stdout + completed.stderr).lower()
    assert "commit" in (completed.stdout + completed.stderr).lower()


def test_rocm_campaign8_readiness_lane_archive_mode_uses_explicit_source_commit(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive-root"
    scripts_dir = archive_root / "scripts"
    scripts_dir.mkdir(parents=True)
    shutil.copy2(READINESS_LANE, scripts_dir / READINESS_LANE.name)
    shutil.copy2(ARCHIVE_PORTABILITY, scripts_dir / ARCHIVE_PORTABILITY.name)
    source_commit = "89abcdef0123456789abcdef0123456789abcdef"
    evidence_dir = tmp_path / "evidence"

    completed = subprocess.run(
        [
            sys.executable,
            str(scripts_dir / READINESS_LANE.name),
            "--write-evidence",
            str(evidence_dir),
            "--source-commit",
            source_commit,
        ],
        cwd=archive_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads((evidence_dir / "summary.json").read_text(encoding="utf-8"))
    commands = json.loads((evidence_dir / "raw/readiness_commands.json").read_text(encoding="utf-8"))
    assert summary["git_revision"] == source_commit
    assert commands["git_revision"] == source_commit


def test_rocm_campaign8_completed_docs_do_not_retain_stale_future_routing() -> None:
    searched_docs = [
        ROOT / "docs/plans/rocm_next_waves_plan.md",
        ROOT / "docs/quality/release_and_packaging.md",
        ROOT / "docs/architecture/hardware_targets_and_testing.md",
    ]
    stale_phrases = [
        "Campaign 8 is the next planned",
        "The next packaging-facing ROCm plan is",
        "the next executable ROCm campaign is\n`docs/plans/mi300x_rocm_optimization_campaign8_plan.md`",
        "Campaign 8 must define",
        "until Wave 6 accepts a backend-neutral multi-accelerator design",
    ]

    for path in searched_docs:
        text = path.read_text(encoding="utf-8")
        for phrase in stale_phrases:
            assert phrase not in text, f"{path} still contains stale Campaign 8 routing: {phrase!r}"

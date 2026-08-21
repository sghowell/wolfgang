from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = "docs/plans/wolfgang-kernel-performance-campaign.md"
PLAN = ROOT / PLAN_PATH

REQUIRED_BRANCHES = {
    "exp/2026-08-16-xbackend-benchmark-contract",
    "exp/2026-08-16-cpu-small-shape-dispatch",
    "exp/2026-08-16-xbackend-workspace-contract",
    "exp/2026-08-16-metal-commutation-reuse",
    "exp/2026-08-16-cuda-retained-reuse",
    "exp/2026-08-16-hip-retained-reuse",
    "exp/2026-08-16-xbackend-expectation-residency",
    "exp/2026-08-16-xbackend-multiword-simplify",
    "exp/2026-08-16-metal-lower-pass-simplify",
}


def load_validate_module() -> ModuleType:
    validate_path = ROOT / "scripts/validate.py"
    spec = importlib.util.spec_from_file_location("wolfgang_validate", validate_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_cross_backend_campaign_plan_exists_and_freezes_the_evidence_boundary() -> None:
    text = PLAN.read_text(encoding="utf-8")

    assert text.startswith("# Wolfgang Cross-Backend Kernel Performance Campaign Implementation Plan")
    assert "**Goal:**" in text
    assert "**Architecture:**" in text
    assert "**Tech Stack:**" in text
    assert "d14b4960a5197485e41d81a5dc426af5fce7cbae" in text
    assert "Global ranking by expected user-visible speedup per engineering plus hardware cost" in text
    assert "Retained commutation result reuse and compact-consumer-first boundaries" in text
    assert "Workspace / scratch / output lifetime reuse for repeated accelerator calls" in text
    assert "CPU auto-dispatch selector split for small full-grouping and shape-sensitive cases" in text
    assert "Repeated-evidence promotion rule" in text
    assert "Affected-backend rerun rules" in text
    assert "MI300X operator rule" in text
    assert "Do not book paid hardware before first-wave prestaging is green and reviewed." in text
    assert "The first implementation branch should not touch production CUDA or HIP kernels." in text

    for branch in REQUIRED_BRANCHES:
        assert branch in text


def test_cross_backend_campaign_plan_is_registered_as_source_of_truth() -> None:
    plan_path = "docs/plans/wolfgang-kernel-performance-campaign.md"

    roadmap = read("docs/roadmap.md")
    provenance = read("docs/research/provenance.md")
    protocol = read("docs/benchmarks/protocol.md")

    assert plan_path in roadmap
    assert "Cross-backend kernel performance campaign" in roadmap
    assert "../plans/wolfgang-kernel-performance-campaign.md" in provenance
    assert "Cross-backend kernel performance campaign" in protocol

    validate = load_validate_module()
    assert plan_path in validate.SOURCE_OF_TRUTH_PATHS

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "docs/plans/apple_metal_optimization_campaign8_plan.md"


def read(path: str | Path) -> str:
    return (ROOT / path if isinstance(path, str) else path).read_text(encoding="utf-8")


def test_campaign8_plan_records_performance_relevance_boundary() -> None:
    text = PLAN_PATH.read_text(encoding="utf-8")

    for token in (
        "Apple Metal Campaign 8",
        "performance-relevant",
        "remain benchmark-only experimental",
        "timing_decomposition_seconds",
        "pipeline/library cache boundary",
        "public Metal `DevicePauliSum.simplify(atol, rtol)`",
        "transfer-reference path",
        "general FP64 Metal simplify",
        "Decision Rule",
    ):
        assert token in text


def test_campaign8_plan_is_registered_in_source_of_truth_docs() -> None:
    plan_path = "docs/plans/apple_metal_optimization_campaign8_plan.md"
    report_path = "docs/benchmarks/reports/apple_metal_optimization_campaign8_2026-05-07.md"

    for document_path in ("docs/research/provenance.md", "docs/roadmap.md", "AGENTS.md"):
        document = read(document_path)
        assert plan_path in document
        assert "Apple Metal Campaign 8" in document

    roadmap = read("docs/roadmap.md")
    assert report_path in roadmap
    assert "Latest Apple Metal report: " + report_path in roadmap

    architecture = read("docs/architecture/apple_accelerator.md")
    assert "Apple Metal Campaign 8" in architecture
    assert "timing decomposition" in architecture
    assert "public `DevicePauliSum.simplify(atol, rtol)` remains" in architecture

    protocol = read("docs/benchmarks/protocol.md")
    assert "Apple Metal Campaign 8" in protocol
    assert "timing_decomposition_seconds" in protocol
    assert "pipeline_cache" in protocol
    assert "performance_decision" in protocol

    validate_source = read("scripts/validate.py")
    assert plan_path in validate_source
    assert "Apple Metal Campaign 8 simplify performance relevance smoke" in validate_source
    assert '"campaign8"' in validate_source

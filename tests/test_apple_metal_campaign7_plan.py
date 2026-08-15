from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "docs/plans/apple_metal_optimization_campaign7_plan.md"


def read(path: str | Path) -> str:
    return (ROOT / path if isinstance(path, str) else path).read_text(encoding="utf-8")


def test_campaign7_plan_records_checked_primitive_stack_boundary() -> None:
    plan = PLAN_PATH.read_text(encoding="utf-8")

    required = (
        "Apple Metal Campaign 7 checked device-resident simplify primitive stack",
        "benchmark-only one-word packed-key sort primitive",
        "prefix-sum primitive for survivor compaction",
        "reduce-by-key primitive for duplicate coefficient summation",
        "deterministic canonical output order",
        "do not change public DevicePauliSum.simplify() behavior unless",
        "device_resident",
        "metal_simplify_device_candidate",
    )
    for token in required:
        assert token in plan

    assert "Campaign 6" in plan
    assert "Campaign 7" in read("AGENTS.md")
    assert "Apple Metal Campaign 7" in read("docs/roadmap.md")
    assert "Apple Metal Campaign 7" in read("docs/research/provenance.md")


def test_campaign7_plan_is_registered_in_validation_and_architecture() -> None:
    validate_source = read("scripts/validate.py")
    architecture = read("docs/architecture/apple_accelerator.md")
    protocol = read("docs/benchmarks/protocol.md")

    assert "docs/plans/apple_metal_optimization_campaign7_plan.md" in validate_source
    assert "Apple Metal Campaign 7 checked primitive stack smoke" in validate_source
    assert "Apple Metal Campaign 7" in architecture
    assert "metal_simplify_device_candidate" in protocol
    assert "checked Metal sort, prefix-sum, and reduce-by-key" in protocol

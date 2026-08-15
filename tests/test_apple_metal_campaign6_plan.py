from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = "docs/plans/apple_metal_optimization_campaign6_plan.md"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def normalized(path: str) -> str:
    return " ".join(read(path).split())


def test_campaign6_plan_is_registered_as_source_of_truth() -> None:
    validate_path = ROOT / "scripts" / "validate.py"
    spec = importlib.util.spec_from_file_location("fastpauli_validate", validate_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert (ROOT / PLAN_PATH).exists()
    assert PLAN_PATH in module.SOURCE_OF_TRUTH_PATHS
    assert PLAN_PATH in read("docs/research/provenance.md")
    assert PLAN_PATH in read("AGENTS.md")
    assert PLAN_PATH in read("docs/roadmap.md")
    assert PLAN_PATH in read("docs/architecture/apple_accelerator.md")
    assert PLAN_PATH in read("CHANGELOG.md")

    plan = normalized(PLAN_PATH)
    for required in (
        "private MetalWorkspace",
        "WorkspaceTimingMode",
        "FASTPAULI_EXPERIMENTAL_METAL_SIMPLIFY_STRATEGY",
        "FASTPAULI_METAL_BENCH_WORKSPACE_TIMING",
        "metal_simplify_workspace_probe",
        "device-resident simplify candidate remains blocked",
        "Metal sort, prefix-sum, and reduce-by-key primitives",
        "device_resident",
        "status_only",
        "Apple Metal Campaign 6 device-resident simplify groundwork",
        "docs/benchmarks/protocol.md",
    ):
        assert required in plan

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = "docs/plans/apple_metal_optimization_campaign5_plan.md"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def normalized(path: str) -> str:
    return " ".join(read(path).split())


def test_campaign5_plan_is_registered_as_source_of_truth() -> None:
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
        "Metal source-build DevicePauliSum.simplify(atol, rtol) behavior",
        "metal_simplify_transfer_reference",
        "device_to_host_cpu_simplify_host_to_device",
        "finite non-negative tolerance validation",
        "benchmark-only feasibility rows for device-resident simplify candidates",
        "#include <stdexcept>",
        "CAMPAIGN_CONFIGS[\"apple_metal_optimization_campaign5\"]",
        "infer_campaign()",
        "simplify-specific scale formatting",
        "docs/benchmarks/protocol.md",
        "metal_simplify_strategy",
        "output_terms",
        "PyPI publication, Windows support, and older macOS compatibility",
        "Metal statevector expectation",
        "Metal matmul",
    ):
        assert required in plan

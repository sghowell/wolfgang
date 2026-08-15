from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = "docs/plans/release_candidate_next_checkpoint_plan.md"
SUPPORT_MATRIX_PATH = "docs/release/support_matrix.md"
CHECKER_PATH = "scripts/check_release_readiness.py"


def load_module(path: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_next_checkpoint_docs_are_registered() -> None:
    validate = load_module("scripts/validate.py", "fastpauli_validate")

    for path in (PLAN_PATH, SUPPORT_MATRIX_PATH):
        assert (ROOT / path).exists(), path
        assert path in validate.SOURCE_OF_TRUTH_PATHS

    for path in (
        "docs/research/provenance.md",
        "AGENTS.md",
        "docs/roadmap.md",
        "docs/release/README.md",
        "docs/quality/release_and_packaging.md",
    ):
        text = (ROOT / path).read_text(encoding="utf-8")
        assert PLAN_PATH in text
        assert SUPPORT_MATRIX_PATH in text


def test_release_support_matrix_names_supported_and_unavailable_surfaces() -> None:
    matrix = (ROOT / SUPPORT_MATRIX_PATH).read_text(encoding="utf-8")

    required_rows = (
        "| CPU default package | CPU artifact target |",
        "| CUDA accelerator | Source-build support |",
        "| ROCm/HIP accelerator | Source-build support |",
        "| Apple Metal accelerator | Source-build evidence |",
        "| Combined accelerator binary | Unsupported by policy |",
        "| Windows | Unsupported release target |",
        "| TestPyPI validation | Published final dry run |",
        "| PyPI final release | Not published |",
    )
    for row in required_rows:
        assert row in matrix

    required_boundaries = (
        "CUDA wheels remain unavailable",
        "ROCm/HIP wheels remain unavailable",
        "Metal wheels remain unavailable",
        "Combined accelerator wheels remain unavailable",
        "Windows wheels remain unavailable",
        "PyPI publication is not claimed",
        "Generic Apple GPU support is unavailable",
        "Broader AMD GPU support remains unavailable",
    )
    for boundary in required_boundaries:
        assert boundary in matrix


def test_release_next_checkpoint_plan_defines_non_publication_scope() -> None:
    plan = (ROOT / PLAN_PATH).read_text(encoding="utf-8")
    normalized_plan = " ".join(plan.split())

    required_terms = (
        "does not change runtime behavior",
        "changing pyproject.toml or python/wolfgang_quantum/_version.py",
        "creating v0.1.0rc2 or v0.1.0 tags",
        "publishing GitHub or package-index artifacts",
        "scripts/check_release_readiness.py",
        "docs/release/support_matrix.md",
        "0.1.0rc2",
        "0.1.0",
    )
    for term in required_terms:
        assert term in normalized_plan


def test_release_readiness_checker_passes_current_docs() -> None:
    checker = load_module(CHECKER_PATH, "fastpauli_release_readiness")

    assert checker.check_release_readiness() == []


@pytest.mark.parametrize(
    "bad_claim",
    [
        "CUDA wheels are available",
        "ROCm/HIP wheels are available",
        "Metal wheels are available",
        "combined accelerator wheels are available",
        "Windows wheels are available",
        "PyPI publication is complete",
        "generic Apple GPU support is available",
        "broad AMD GPU support is available",
    ],
)
def test_release_readiness_checker_rejects_forbidden_claims(
    monkeypatch: pytest.MonkeyPatch,
    bad_claim: str,
) -> None:
    checker = load_module(CHECKER_PATH, "fastpauli_release_readiness")
    original_read_text = checker.read_text

    def read_text_with_bad_claim(path: str) -> str:
        text = original_read_text(path)
        if path == checker.README:
            return f"{text}\n{bad_claim}\n"
        return text

    monkeypatch.setattr(checker, "read_text", read_text_with_bad_claim)

    failures = checker.check_release_readiness()

    assert any("contains forbidden release claim" in failure for failure in failures)


def test_validate_runs_release_readiness_checker() -> None:
    validate = (ROOT / "scripts/validate.py").read_text(encoding="utf-8")

    assert "check release-readiness documentation" in validate
    assert "scripts/check_release_readiness.py" in validate

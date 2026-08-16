from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
RELEASE_VERSION = "0.2.3"
RELEASE_TAG = f"v{RELEASE_VERSION}"
RELEASE_LEDGER_PATH = "docs/release/0.2.3.md"
RELEASE_LEDGER_URL = f"https://github.com/sghowell/wolfgang/releases/tag/{RELEASE_TAG}"


def load_module(path: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v021_release_ledger_is_routed_on_active_release_surfaces() -> None:
    validate = load_module("scripts/validate.py", "wolfgang_validate")

    assert RELEASE_LEDGER_PATH in validate.SOURCE_OF_TRUTH_PATHS

    routed_docs = (
        "docs/research/provenance.md",
        "docs/roadmap.md",
        "docs/release/README.md",
        "docs/quality/release_and_packaging.md",
        "SUPPORT.md",
        "CHANGELOG.md",
    )
    for path in routed_docs:
        text = (ROOT / path).read_text(encoding="utf-8")
        assert RELEASE_LEDGER_PATH in text, path


def test_v021_support_matrix_tracks_pending_release_without_claiming_publication() -> None:
    support = (ROOT / "docs/release/support_matrix.md").read_text(encoding="utf-8")

    assert "Source version: 0.2.3" in support
    assert "Next intended release: v0.2.3 (pending PR, GitHub-only successor, not tagged or published)" in support
    assert "Latest tagged release: v0.2.2" in support
    assert f"Release under finalization: {RELEASE_TAG} pending GitHub publication" in support
    assert f"`{RELEASE_LEDGER_PATH}`" in support
    assert "PyPI publication: not part of v0.2.3 GitHub-only successor; no TestPyPI or PyPI run has been attempted" in support
    assert "TestPyPI validation: historical 0.1.0 dry-run evidence only" in support


def test_v021_release_ledger_records_required_release_notes_and_boundaries() -> None:
    ledger = (ROOT / RELEASE_LEDGER_PATH).read_text(encoding="utf-8")

    required_terms = (
        f"Status: prepared for GitHub-only patch successor `{RELEASE_TAG}`; not yet published.",
        f"Package version: {RELEASE_VERSION}",
        f"Git tag: {RELEASE_TAG}",
        f"Release tag URL: {RELEASE_LEDGER_URL}",
        "GitHub-only successor publication remains deferred for v0.2.3.",
        "Do not invoke TestPyPI or PyPI for this GitHub-only successor slice.",
        "corrected capabilities fix-forward",
        "immutable `v0.2.2` tag",
        "historical provenance",
        "No hardware rerun was required for v0.2.3 because no kernel or hardware claim changed.",
        "CUDA wheels remain unavailable",
        "ROCm/HIP wheels remain unavailable",
        "Metal wheels remain unavailable",
        "Windows wheels remain unavailable",
        "PyPI publication is not claimed",
    )
    for term in required_terms:
        assert term in ledger


def test_release_readiness_checker_targets_pending_v021_ledger() -> None:
    checker = load_module("scripts/check_release_readiness.py", "wolfgang_release_readiness")

    assert checker.LATEST_RELEASE_VERSION == RELEASE_VERSION
    assert checker.current_release_ledger_path(RELEASE_VERSION) == RELEASE_LEDGER_PATH
    assert checker.check_release_readiness() == []

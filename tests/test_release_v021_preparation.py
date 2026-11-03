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


def test_v021_support_matrix_tracks_published_github_release_without_claiming_package_index_publication() -> None:
    support = (ROOT / "docs/release/support_matrix.md").read_text(encoding="utf-8")

    assert "Source version: 0.2.3" in support
    assert f"Current published GitHub release: {RELEASE_TAG}" in support
    assert f"Latest tagged release: {RELEASE_TAG}" in support
    assert "Historical immutable checkpoint: v0.2.2" in support
    assert f"`{RELEASE_LEDGER_PATH}`" in support
    assert "and v0.2.2 and v0.2.3 made no TestPyPI or PyPI run" in support
    assert "not run for v0.2.2 or v0.2.3" in support


def test_support_matrix_states_each_package_index_status_once() -> None:
    # Each release once appended its own status line, which left the block with
    # two contradictory "TestPyPI validation:" and "PyPI publication:" lines.
    support = (ROOT / "docs/release/support_matrix.md").read_text(encoding="utf-8")
    block = support.split("## Current Package Version", 1)[1].split("## ", 1)[0]
    lines = [line.strip() for line in block.splitlines()]

    statuses = {}
    for key in ("TestPyPI validation:", "PyPI publication:"):
        matching = [line for line in lines if line.startswith(key)]
        assert len(matching) == 1, (key, matching)
        statuses[key] = matching[0]

    # One line must carry the whole status: the 0.1.0 runs and the later releases.
    assert "25462760923" in statuses["TestPyPI validation:"]
    assert "v0.2.3" in statuses["TestPyPI validation:"]
    assert "25462997972" in statuses["PyPI publication:"]
    assert "v0.2.3" in statuses["PyPI publication:"]


def test_v021_release_ledger_records_required_release_notes_and_boundaries() -> None:
    ledger = (ROOT / RELEASE_LEDGER_PATH).read_text(encoding="utf-8")

    required_terms = (
        f"Status: published as GitHub release `{RELEASE_TAG}` with one source distribution,",
        f"Package version: {RELEASE_VERSION}",
        f"Git tag: {RELEASE_TAG}",
        f"Release tag URL: {RELEASE_LEDGER_URL}",
        "GitHub release object: published",
        "Published release commit: b781cd428d94cd5328105f19837c6e19b3751b74",
        "Release wheelhouse workflow run: 31924676423",
        "GitHub publication state: published 2026-08-16T03:52:49Z",
        "does not invoke TestPyPI or PyPI for this successor slice.",
        "corrected capabilities fix-forward",
        "immutable `v0.2.2` tag",
        "historical provenance",
        "No hardware rerun was required for v0.2.3 because no kernel or hardware claim changed.",
        "The live GitHub release now serves the exact seven assets listed above.",
        "CUDA wheels remain unavailable",
        "ROCm/HIP wheels remain unavailable",
        "Metal wheels remain unavailable",
        "Windows wheels remain unavailable",
        "PyPI publication is not claimed",
    )
    for term in required_terms:
        assert term in ledger


def test_release_readiness_checker_targets_published_v021_ledger() -> None:
    checker = load_module("scripts/check_release_readiness.py", "wolfgang_release_readiness")

    assert checker.LATEST_RELEASE_VERSION == RELEASE_VERSION
    assert checker.current_release_ledger_path(RELEASE_VERSION) == RELEASE_LEDGER_PATH
    assert checker.check_release_readiness() == []

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
RELEASE_VERSION = "0.1.0rc2"
RELEASE_LABEL = "0.1.0-rc2"
RELEASE_TAG = "v0.1.0rc2"
RELEASE_LEDGER_PATH = f"docs/release/{RELEASE_LABEL}.md"
RELEASE_URL = f"https://github.com/sghowell/FastPauli/releases/tag/{RELEASE_TAG}"
RELEASE_REVISION = "504aa3f5726a930e3afde32c7c9f6b0346997248"
RELEASE_CI_RUN = "25415123969"
RELEASE_CI_URL = f"https://github.com/sghowell/FastPauli/actions/runs/{RELEASE_CI_RUN}"
PUBLISHED_ARTIFACTS = (
    (
        f"fastpauli-{RELEASE_VERSION}.tar.gz",
        "1088c74c9e7e152d3c399c7b16c8cd4100015e224b6dd374e9ce713efb61010f",
    ),
    (
        f"fastpauli-{RELEASE_VERSION}-cp312-cp312-macosx_26_0_arm64.whl",
        "beb3a7e7419adb42cb04c7341eaed49044af1d77551f071debee2266743447b9",
    ),
    (
        f"fastpauli-{RELEASE_VERSION}.checksums.txt",
        "57cbfad81bcdaef9040bf060df7681be63eb7c9a7412725acb11eca697bcb312",
    ),
)


def load_module(path: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def project_version() -> str:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_rc2_release_ledger_and_routing_remain_historical() -> None:
    validate = load_module("scripts/validate.py", "fastpauli_validate")

    assert RELEASE_LEDGER_PATH in validate.SOURCE_OF_TRUTH_PATHS

    routed_docs = (
        "docs/research/provenance.md",
        "AGENTS.md",
        "docs/roadmap.md",
        "docs/release/README.md",
        "docs/quality/release_and_packaging.md",
        "docs/release/support_matrix.md",
    )

    for path in routed_docs:
        text = (ROOT / path).read_text(encoding="utf-8")
        assert RELEASE_LEDGER_PATH in text


def test_rc2_support_matrix_history_remains_without_expanding_claims() -> None:
    matrix = (ROOT / "docs/release/support_matrix.md").read_text(encoding="utf-8")

    assert "Source version: 0.2.3" in matrix
    assert "Next intended release: v0.2.3 (pending PR, GitHub-only successor, not tagged or published)" in matrix
    assert "Latest tagged release: v0.2.2" in matrix
    assert f"Previous checkpoint: {RELEASE_TAG} GitHub prerelease" in matrix
    assert f"`{RELEASE_LEDGER_PATH}`" in matrix

    unavailable_claims = (
        "CUDA wheels remain unavailable",
        "ROCm/HIP wheels remain unavailable",
        "Metal wheels remain unavailable",
        "Combined accelerator wheels remain unavailable",
        "Windows wheels remain unavailable",
        "PyPI publication is not claimed",
        "Generic Apple GPU support is unavailable",
        "Broader AMD GPU support remains unavailable",
    )
    for claim in unavailable_claims:
        assert claim in matrix


def test_rc2_release_ledger_records_required_publication_evidence() -> None:
    ledger = (ROOT / RELEASE_LEDGER_PATH).read_text(encoding="utf-8")

    required_terms = (
        f"Status: published as GitHub prerelease `{RELEASE_TAG}`",
        f"Package version: {RELEASE_VERSION}",
        f"Release candidate display label: {RELEASE_LABEL}",
        f"Git tag: {RELEASE_TAG}",
        f"Release revision: {RELEASE_REVISION}",
        RELEASE_URL,
        f"Hosted CI run: {RELEASE_CI_RUN}",
        f"run: {RELEASE_CI_RUN}",
        f"head revision: {RELEASE_REVISION}",
        f"url: {RELEASE_CI_URL}",
        "source distribution:",
        "macOS arm64 CPU wheel:",
        "external checksum manifest:",
        "PyPI or another package-index publication is not claimed",
        "CUDA wheels are unavailable",
        "ROCm/HIP wheels are unavailable",
        "Metal wheels are unavailable",
        "Windows wheels are unavailable",
    )
    for term in required_terms:
        assert term in ledger

    for filename, sha256 in PUBLISHED_ARTIFACTS:
        assert f"filename: {filename}" in ledger
        assert f"sha256: {sha256}" in ledger

    assert "pending publication" not in ledger
    assert "recorded from tag v0.1.0rc2" not in ledger
    assert "branch pre-publication validation passed" not in ledger


def test_release_readiness_checker_rejects_published_placeholder_ledger(
    monkeypatch,
) -> None:
    checker = load_module("scripts/check_release_readiness.py", "fastpauli_release_readiness")
    original_read_text = checker.read_text

    def read_text_with_published_placeholders(path: str) -> str:
        text = original_read_text(path)
        if path == RELEASE_LEDGER_PATH:
            return (
                text.replace(RELEASE_REVISION, "recorded from the pushed release tag during closeout")
                .replace(f"Hosted CI run: {RELEASE_CI_RUN}", "Hosted CI run: recorded in closeout")
                .replace(f"run: {RELEASE_CI_RUN}", "run: recorded in closeout")
                .replace(f"url: {RELEASE_CI_URL}", "url: recorded in closeout")
            )
        if path == checker.SUPPORT_MATRIX:
            return text
        return text

    monkeypatch.setattr(checker, "read_text", read_text_with_published_placeholders)

    ledger = checker.read_text(RELEASE_LEDGER_PATH)
    failures: list[str] = []
    checker.check_concrete_publication_evidence(
        ledger=ledger,
        version=RELEASE_VERSION,
        ledger_path=RELEASE_LEDGER_PATH,
        support=f"Published checkpoint: {RELEASE_TAG} GitHub prerelease",
        failures=failures,
    )

    assert any("missing concrete release revision" in failure for failure in failures)
    assert any("missing concrete hosted CI run" in failure for failure in failures)
    assert any("still contains placeholder closeout wording" in failure for failure in failures)


def test_release_readiness_checker_tracks_current_rc2_ledger() -> None:
    checker = load_module("scripts/check_release_readiness.py", "fastpauli_release_readiness")

    assert checker.current_release_ledger_path(RELEASE_VERSION) == RELEASE_LEDGER_PATH

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = "docs/plans/release_candidate_foundation_plan.md"
RELEASE_INDEX_PATH = "docs/release/README.md"
RELEASE_LEDGER_PATH = "docs/release/0.1.0-rc1.md"
CHANGELOG_PATH = "CHANGELOG.md"
VALIDATOR_PATH = "scripts/validate_release_artifacts.py"
RELEASE_CANDIDATE_VERSION = "0.1.0rc1"
RELEASE_CANDIDATE_TAG = "v0.1.0rc1"
RELEASE_CANDIDATE_REVISION = "cc4132f93ac35d7bb0c9eb5eb8a5381f243d8caf"
RELEASE_CANDIDATE_CI_RUN = "25409288091"
RELEASE_CANDIDATE_RELEASE_URL = (
    "https://github.com/sghowell/FastPauli/releases/tag/v0.1.0rc1"
)
PUBLISHED_ARTIFACTS = (
    (
        "fastpauli-0.1.0rc1.tar.gz",
        "722a87155412ef59d738126672a7f456179a02ea17711ac006c6ebc718446ee0",
    ),
    (
        "fastpauli-0.1.0rc1-cp312-cp312-macosx_26_0_arm64.whl",
        "ae01419aa8e8e253d9b179d654eb5f5ea39ab6d7198b876ec522ba0ab7efd6b7",
    ),
    (
        "fastpauli-0.1.0rc1.checksums.txt",
        "ae6861aa2c0ca17559e7a9c7b570640521dd9c9d80da35204aa5d3239c9ef2e8",
    ),
)


def load_validate_module() -> ModuleType:
    validate_path = ROOT / "scripts" / "validate.py"
    spec = importlib.util.spec_from_file_location("fastpauli_validate", validate_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_candidate_foundation_docs_are_registered() -> None:
    validate = load_validate_module()
    readme = (ROOT / "docs/research/provenance.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs/roadmap.md").read_text(encoding="utf-8")
    release_standards = (ROOT / "docs/quality/release_and_packaging.md").read_text(
        encoding="utf-8"
    )

    for path in (PLAN_PATH, RELEASE_INDEX_PATH, RELEASE_LEDGER_PATH, CHANGELOG_PATH):
        assert (ROOT / path).exists(), path
        assert path in readme
        assert path in agents
        assert path in roadmap
        assert path in validate.SOURCE_OF_TRUTH_PATHS

    assert VALIDATOR_PATH in readme
    assert VALIDATOR_PATH in roadmap
    assert VALIDATOR_PATH in release_standards


def test_release_candidate_foundation_plan_defines_cpu_artifact_gate() -> None:
    plan = (ROOT / PLAN_PATH).read_text(encoding="utf-8")

    required_terms = (
        "Release Candidate Foundation",
        "CPU source distribution",
        "CPU wheel",
        "clean virtual environment",
        "FASTPAULI_ENABLE_CUDA=OFF",
        "FASTPAULI_ENABLE_HIP=OFF",
        "FASTPAULI_ENABLE_NATIVE=OFF",
        "scripts/validate_release_artifacts.py",
        "CHANGELOG.md",
        "docs/release/0.1.0-rc1.md",
        "Apple GPU implementation or Metal/MPS planning",
    )
    for term in required_terms:
        assert term in plan


def test_release_artifact_ci_job_is_declared() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "cpu-wheel-smoke" in ci
    assert "scripts/validate_release_artifacts.py" in ci
    assert "ubuntu-latest" in ci
    assert "macos-latest" in ci
    assert "FASTPAULI_ENABLE_CUDA: \"OFF\"" in ci
    assert "FASTPAULI_ENABLE_HIP: \"OFF\"" in ci
    assert "FASTPAULI_ENABLE_NATIVE: \"OFF\"" in ci
    assert "python -m pip install --upgrade pip build cmake ninja" in ci


def test_changelog_and_release_evidence_keep_support_boundaries() -> None:
    combined = "\n".join(
        [
            (ROOT / CHANGELOG_PATH).read_text(encoding="utf-8"),
            (ROOT / RELEASE_INDEX_PATH).read_text(encoding="utf-8"),
            (ROOT / RELEASE_LEDGER_PATH).read_text(encoding="utf-8"),
        ]
    )

    required_boundaries = (
        "CPU wheels",
        "CUDA wheels remain unavailable",
        "ROCm/HIP wheels remain unavailable",
        "combined accelerator wheels remain unavailable",
        "source-build-only",
        "scalar fallback",
    )
    for boundary in required_boundaries:
        assert boundary in combined


def test_release_candidate_rc1_ledger_keeps_pep440_metadata() -> None:
    changelog = (ROOT / CHANGELOG_PATH).read_text(encoding="utf-8")
    ledger = (ROOT / RELEASE_LEDGER_PATH).read_text(encoding="utf-8")

    assert f"## {RELEASE_CANDIDATE_VERSION}" in changelog
    assert f"Package version: {RELEASE_CANDIDATE_VERSION}" in ledger
    assert f"Git tag: {RELEASE_CANDIDATE_TAG}" in ledger
    assert f"GitHub release: {RELEASE_CANDIDATE_RELEASE_URL}" in ledger
    assert "PyPI or another package-index publication is not claimed" in ledger


def test_release_ledger_locks_publication_evidence() -> None:
    ledger = (ROOT / RELEASE_LEDGER_PATH).read_text(encoding="utf-8")

    assert f"Release revision: {RELEASE_CANDIDATE_REVISION}" in ledger
    assert f"Hosted CI run: {RELEASE_CANDIDATE_CI_RUN}" in ledger
    assert f"run: {RELEASE_CANDIDATE_CI_RUN}" in ledger
    assert f"head revision: {RELEASE_CANDIDATE_REVISION}" in ledger
    for filename, sha256 in PUBLISHED_ARTIFACTS:
        assert f"filename: {filename}" in ledger
        assert f"sha256: {sha256}" in ledger


def test_gitignore_excludes_release_artifacts() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    for pattern in ("*.whl", "*.tar.gz", "wheelhouse/", "release-artifacts/"):
        assert pattern in gitignore

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
RELEASE_VERSION = "0.1.0"
SOURCE_VERSION = "0.2.3"
RELEASE_LABEL = "0.1.0"
RELEASE_TAG = "v0.1.0"
RELEASE_LEDGER_PATH = "docs/release/0.1.0.md"
RELEASE_TAG_URL = f"https://github.com/sghowell/FastPauli/releases/tag/{RELEASE_TAG}"
RELEASE_REVISION = "a14869230743fbc0ad4e3b56305342f43139d31a"
HOSTED_CI_RUN = "25462644191"
WHEELHOUSE_RUN = "25452754832"
TESTPYPI_RUN = "25462760923"
PYPI_RUN = "25462997972"
VALID_RELEASE_REVISION = "0123456789abcdef0123456789abcdef01234567"
VALID_CI_RUN = "25420000000"
VALID_CI_URL = "https://github.com/sghowell/FastPauli/actions/runs/25420000000"


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


def test_final_release_evidence_is_routed_while_source_has_advanced() -> None:
    validate = load_module("scripts/validate.py", "fastpauli_validate")
    version_module = (ROOT / "python/wolfgang_quantum/_version.py").read_text(encoding="utf-8")

    assert project_version() == SOURCE_VERSION
    assert f'__version__ = "{SOURCE_VERSION}"' in version_module
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
        assert RELEASE_VERSION in text


def test_final_support_matrix_is_prepublication_without_expanding_claims() -> None:
    matrix = (ROOT / "docs/release/support_matrix.md").read_text(encoding="utf-8")

    assert f"Source version: {SOURCE_VERSION}" in matrix
    assert "Latest tagged release: v0.2.2" in matrix
    assert "Next intended release: v0.2.3 (pending PR, GitHub-only successor, not tagged or published)" in matrix
    assert "Previous checkpoint: v0.1.0rc2 GitHub prerelease" in matrix
    assert f"`{RELEASE_LEDGER_PATH}`" in matrix
    assert f"0.1.0 tag-ref wheelhouse run: {WHEELHOUSE_RUN}" in matrix
    assert f"Corrected 0.1.0 TestPyPI validation run: {TESTPYPI_RUN}" in matrix
    assert f"Latest 0.1.0 PyPI publication run: {PYPI_RUN}" in matrix
    assert "TestPyPI validation: final dry run published and smoke-tested" in matrix
    assert "PyPI publication: unavailable pending trusted-publisher configuration" in matrix

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


def test_final_release_ledger_records_required_prepublication_evidence() -> None:
    ledger = (ROOT / RELEASE_LEDGER_PATH).read_text(encoding="utf-8")

    required_terms = (
        f"Status: prepared for PyPI package-index release `{RELEASE_TAG}`; not yet published.",
        f"Package version: {RELEASE_VERSION}",
        f"Release display label: {RELEASE_LABEL}",
        f"Git tag: {RELEASE_TAG}",
        f"Release revision: {RELEASE_REVISION}",
        "GitHub release object: not created",
        f"Release tag URL: {RELEASE_TAG_URL}",
        f"Hosted CI run: {HOSTED_CI_RUN}",
        f"Corrected TestPyPI workflow run: {TESTPYPI_RUN}",
        f"Corrected TestPyPI workflow URL: https://github.com/sghowell/FastPauli/actions/runs/{TESTPYPI_RUN}",
        "Corrected TestPyPI workflow conclusion: success",
        f"PyPI workflow run: {PYPI_RUN}",
        f"PyPI workflow URL: https://github.com/sghowell/FastPauli/actions/runs/{PYPI_RUN}",
        "PyPI workflow conclusion: failure in PyPI trusted-publishing exchange after successful artifact build and collection",
        "The `25458693782` retry on 2026-05-06 passed the previous trusted-publisher",
        "corrected tag now points at",
        "The corrected `25462760923` TestPyPI run passed",
        "TestPyPI trusted publishing and upload: passed",
        "TestPyPI install smoke: passed",
        "The `25462997972` PyPI run rebuilt and revalidated the package set",
        "PyPI response: invalid-publisher",
        "TestPyPI status: published for final dry run; clean install smoke passed",
        "Rejected classifier: Programming Language :: C++ :: 20",
        "source distribution:",
        f"filename: fastpauli-{RELEASE_VERSION}.tar.gz",
        "manylinux x86_64 CPU wheels:",
        "macOS arm64 CPU wheels:",
        "external checksum manifest:",
        f"filename: fastpauli-{RELEASE_VERSION}.checksums.txt",
        "PyPI status: not published; blocked by PyPI trusted-publisher invalid-publisher",
        "TestPyPI trusted publisher status: configured for the observed tag-ref claims",
        "PyPI trusted publisher status: not configured for the observed pypi environment claims",
        "## Release-Tag Strategy",
        "The `v0.1.0` tag has been corrected",
        "Because TestPyPI has now accepted `0.1.0`, do not retag `v0.1.0` again",
        "configure PyPI trusted publisher for environment pypi",
        "PyPI publication is not claimed",
        "CUDA wheels are unavailable",
        "ROCm/HIP wheels are unavailable",
        "Metal wheels are unavailable",
        "Windows wheels are unavailable",
    )
    for term in required_terms:
        assert term in ledger

    wheel_filenames = (
        f"fastpauli-{RELEASE_VERSION}-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        f"fastpauli-{RELEASE_VERSION}-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        f"fastpauli-{RELEASE_VERSION}-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        f"fastpauli-{RELEASE_VERSION}-cp310-cp310-macosx_11_0_arm64.whl",
        f"fastpauli-{RELEASE_VERSION}-cp311-cp311-macosx_11_0_arm64.whl",
        f"fastpauli-{RELEASE_VERSION}-cp312-cp312-macosx_11_0_arm64.whl",
    )
    for filename in wheel_filenames:
        assert f"filename: {filename}" in ledger

    expected_hashes = (
        "sha256: 7e2e711d11f0f9c385ebb00d55892b927de266283fbcc35f8c68ef7902a5d619",
        "sha256: 12c177e7c8f34b3de60a50b8a5ee698d428a0753e6a25f4149d27d79fb218ce3",
        "sha256: 642c7a314de062cc73ea9024bef05d71c85dac2d3f0158fbf82571724a31e06f",
        "sha256: 0d463c541d6e9dde0e0b4a8b6995b19026999d48be9c14e7264154c1590d17f3",
        "sha256: d4cd8ef306121c8a50e619dddd9e9f2db6bf719ae5ae0b0332e7467070d3aea7",
        "sha256: 4340e43bcc41033130a34aa3d3acdc4aada78a7eb5a7c89e503e81acb49bd9d9",
        "sha256: 6b4a3e866cd5c38ac5a3c8de3b40aa90f689e6fe00f10a140e4f07b1e35886db",
        "sha256: 4aeb007cd2d1ea0bd2bb798ad851e587e46f097cc75a3c7ea0e2e32317544a04",
    )
    for digest in expected_hashes:
        assert digest in ledger

    assert "published as package-index release" not in ledger


def test_final_release_roadmap_tracks_current_pypi_blocker() -> None:
    roadmap = (ROOT / "docs/roadmap.md").read_text(encoding="utf-8")

    assert "TestPyPI upload and clean install smoke" in roadmap
    assert "PyPI trusted publishing is configured" in roadmap
    assert "observed `pypi` environment claims" in roadmap
    assert "TestPyPI trusted publishing is not yet configured" not in roadmap


def test_final_release_changelog_tracks_current_pypi_blocker() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert "TestPyPI trusted publishing and clean\ninstall smoke have passed" in changelog
    assert "PyPI publication is blocked by PyPI\ntrusted-publisher configuration" in changelog
    assert "missing matching PyPI\n  trusted-publisher configuration" in changelog
    assert "trusted publishing is not\nconfigured yet" not in changelog


def test_release_readiness_checker_tracks_current_final_ledger() -> None:
    checker = load_module("scripts/check_release_readiness.py", "fastpauli_release_readiness")

    assert checker.current_release_ledger_path(RELEASE_VERSION) == RELEASE_LEDGER_PATH
    assert checker.check_release_readiness() == []


def test_release_readiness_checker_rejects_prepublication_published_support_marker() -> None:
    checker = load_module("scripts/check_release_readiness.py", "fastpauli_release_readiness")
    ledger = (ROOT / RELEASE_LEDGER_PATH).read_text(encoding="utf-8")
    failures: list[str] = []

    checker.check_prepublication_evidence(
        ledger=ledger,
        version=RELEASE_VERSION,
        ledger_path=RELEASE_LEDGER_PATH,
        support="\n".join(
            (
                f"Release under finalization: {RELEASE_TAG} pending publication",
                f"Published release: {RELEASE_TAG}",
            )
        ),
        failures=failures,
    )

    assert any("must not call v0.1.0 published" in failure for failure in failures)


def test_release_readiness_checker_rejects_pypi_published_status_before_release() -> None:
    checker = load_module("scripts/check_release_readiness.py", "fastpauli_release_readiness")
    ledger = (ROOT / RELEASE_LEDGER_PATH).read_text(encoding="utf-8")
    failures: list[str] = []

    checker.check_prepublication_evidence(
        ledger=f"{ledger}\nPyPI status: published\n",
        version=RELEASE_VERSION,
        ledger_path=RELEASE_LEDGER_PATH,
        support=f"Release under finalization: {RELEASE_TAG} pending publication",
        failures=failures,
    )

    assert any("must not claim published artifacts" in failure for failure in failures)


def test_release_readiness_checker_requires_hash_for_every_final_artifact() -> None:
    checker = load_module("scripts/check_release_readiness.py", "fastpauli_release_readiness")
    required_artifacts = checker.required_published_artifact_filenames(RELEASE_VERSION)
    partial_artifacts = required_artifacts[:3]
    ledger = "\n".join(
        (
            f"Status: published as package-index release `{RELEASE_TAG}`",
            f"Release revision: {VALID_RELEASE_REVISION}",
            f"Hosted CI run: {VALID_CI_RUN}",
            f"url: {VALID_CI_URL}",
            "",
            *(
                f"filename: {filename}\n  sha256: {'a' * 64}"
                for filename in partial_artifacts
            ),
        )
    )
    failures: list[str] = []

    checker.check_concrete_publication_evidence(
        ledger=ledger,
        version=RELEASE_VERSION,
        ledger_path=RELEASE_LEDGER_PATH,
        support=f"Published release: {RELEASE_TAG}",
        failures=failures,
    )

    assert any("every expected artifact" in failure for failure in failures)
    assert any(required_artifacts[-1] in failure for failure in failures)

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
NEXT_VERSION = "0.2.2"
WORKFLOWS = tuple(sorted((ROOT / ".github/workflows").glob("*.yml")))
ACTION_PIN = re.compile(
    r"^\s*uses:\s+[^\s@]+@[0-9a-f]{40}\s+#\s+\S+\s*$",
    re.MULTILINE,
)


def load_module(path: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def project_metadata() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_post_release_source_identity_is_coherently_versioned() -> None:
    metadata = project_metadata()
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    canonical_module = (ROOT / "python/wolfgang_quantum/_version.py").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    standards = (ROOT / "docs/quality/release_and_packaging.md").read_text(encoding="utf-8")
    release_index = (ROOT / "docs/release/README.md").read_text(encoding="utf-8")
    support = (ROOT / "docs/release/support_matrix.md").read_text(encoding="utf-8")

    assert metadata["version"] == NEXT_VERSION
    assert re.search(rf"project\([\s\S]*?VERSION {re.escape(NEXT_VERSION)}\b", cmake)
    assert f'__version__ = "{NEXT_VERSION}"' in canonical_module
    assert not (ROOT / "python/fastpauli").exists()
    assert f"Next version: {NEXT_VERSION}" in changelog
    assert f"current development version: {NEXT_VERSION}" in standards
    assert f"current source version is `{NEXT_VERSION}`" in release_index
    assert f"Source version: {NEXT_VERSION}" in support
    assert "Latest tagged release: v0.2.1" in support


def test_project_metadata_exposes_public_project_urls() -> None:
    urls = project_metadata()["urls"]

    assert urls == {
        "Homepage": "https://github.com/sghowell/wolfgang",
        "Documentation": "https://sghowell.github.io/wolfgang/",
        "Repository": "https://github.com/sghowell/wolfgang.git",
        "Issues": "https://github.com/sghowell/wolfgang/issues",
        "Changelog": "https://github.com/sghowell/wolfgang/blob/main/CHANGELOG.md",
    }


def test_release_tag_checker_requires_exact_project_version_tag() -> None:
    checker_path = ROOT / "scripts/check_release_tag.py"
    assert checker_path.exists()
    checker = load_module("scripts/check_release_tag.py", "fastpauli_release_tag")

    assert checker.expected_release_tag() == f"v{NEXT_VERSION}"
    assert checker.check_release_tag("tag", f"v{NEXT_VERSION}") == []
    assert checker.check_release_tag("tag", "v0.1.0") == [
        f"release tag must exactly match project version: expected v{NEXT_VERSION}, got v0.1.0"
    ]
    assert checker.check_release_tag("branch", f"v{NEXT_VERSION}") == [
        "package-index publication requires a tag ref; got branch"
    ]


@pytest.mark.parametrize("workflow_path", WORKFLOWS, ids=lambda path: path.name)
def test_every_github_action_is_pinned_to_a_commented_full_sha(workflow_path: Path) -> None:
    workflow = workflow_path.read_text(encoding="utf-8")
    uses_lines = re.findall(r"^\s*uses:.*$", workflow, re.MULTILINE)

    assert uses_lines
    assert all(ACTION_PIN.fullmatch(line) for line in uses_lines), uses_lines


def test_workflows_have_read_only_defaults_and_concurrency() -> None:
    for workflow_path in WORKFLOWS:
        workflow = workflow_path.read_text(encoding="utf-8")
        assert re.search(r"^permissions:\n  contents: read$", workflow, re.MULTILINE)
        assert re.search(
            r"^concurrency:\n  group: .+\n  cancel-in-progress: (?:true|false)$",
            workflow,
            re.MULTILINE,
        )


def test_release_workflow_keeps_oidc_job_scoped_and_binds_exact_tag() -> None:
    workflow = (ROOT / ".github/workflows/release-wheelhouse.yml").read_text(encoding="utf-8")

    assert workflow.count("id-token: write") == 3
    assert "attestations: write" in workflow
    assert "python scripts/check_release_tag.py \"${GITHUB_REF_TYPE}\" \"${GITHUB_REF_NAME}\"" in workflow
    assert "startsWith(github.ref_name, 'v')" not in workflow
    assert "${GITHUB_REF_NAME}" in workflow
    assert "v*" not in workflow


def test_release_workflow_gates_dependencies_metadata_sbom_and_attestation() -> None:
    workflow = (ROOT / ".github/workflows/release-wheelhouse.yml").read_text(encoding="utf-8")

    for term in (
        "pip-audit",
        "cyclonedx-py",
        "python -m twine check dist/*",
        "Software bill of materials",
        "actions/attest-build-provenance",
    ):
        assert term in workflow


def test_release_workflow_audits_the_exact_artifacts_before_attestation() -> None:
    workflow = (ROOT / ".github/workflows/release-wheelhouse.yml").read_text(encoding="utf-8")

    assert "python scripts/audit_public_artifacts.py --tracked" in workflow
    assert "python scripts/audit_public_artifacts.py --history" in workflow
    assert 'python scripts/audit_public_artifacts.py --sdist "${artifact}"' in workflow
    assert 'python scripts/audit_public_artifacts.py --path "${audit_dir}"' in workflow
    assert workflow.index("Audit exact release artifacts") < workflow.index(
        "Upload collected release dist"
    )
    assert workflow.index("Audit source distribution before upload") < workflow.index(
        "Upload source distribution"
    )
    assert workflow.index("Audit wheels before upload") < workflow.index(
        "Upload wheel artifacts"
    )
    assert "cmake.define.WOLFGANG_ENABLE_INTERNAL_BINDINGS=OFF" in workflow


def test_ci_and_release_jobs_are_bounded() -> None:
    release = (ROOT / ".github/workflows/release-wheelhouse.yml").read_text(encoding="utf-8")
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert release.count("timeout-minutes:") >= 8
    assert ci.count("timeout-minutes:") >= 4


def test_dependabot_tracks_actions_and_python_dependencies() -> None:
    config = (ROOT / ".github/dependabot.yml").read_text(encoding="utf-8")

    assert "package-ecosystem: github-actions" in config
    assert "package-ecosystem: pip" in config
    assert config.count("interval: weekly") == 2
    assert config.count("default-days: 7") == 2

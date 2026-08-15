from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


REQUIRED_COMMUNITY_FILES = (
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "SUPPORT.md",
    "GOVERNANCE.md",
    "CITATION.cff",
    ".github/pull_request_template.md",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/performance_regression.yml",
    ".github/ISSUE_TEMPLATE/hardware_support.yml",
)

REQUIRED_DOC_FILES = (
    "mkdocs.yml",
    "docs/index.md",
    "docs/getting-started/installation.md",
    "docs/getting-started/quickstart.md",
    "docs/getting-started/conventions.md",
    "docs/guide/architecture.md",
    "docs/guide/agent-driven-engineering.md",
    "docs/guide/python-api.md",
    "docs/accelerators/overview.md",
    "docs/api/index.md",
    "docs/release/history_sanitization.md",
)


@pytest.mark.parametrize("relative_path", REQUIRED_COMMUNITY_FILES + REQUIRED_DOC_FILES)
def test_public_project_surface_exists(relative_path: str) -> None:
    path = ROOT / relative_path
    assert path.is_file(), f"missing public project surface: {relative_path}"
    assert path.stat().st_size > 100, f"public project surface is implausibly empty: {relative_path}"


def test_readme_is_user_first_and_links_public_surfaces() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    required_headings = (
        "## Why Wolfgang",
        "## Installation",
        "## Quickstart",
        "## Architecture",
        "## Performance",
        "## Platform support",
        "## Documentation",
        "## Contributing and security",
        "## Citation",
        "## License",
    )
    for heading in required_headings:
        assert heading in readme, f"README is missing {heading!r}"

    assert readme.index("## Installation") < readme.index("## Performance")
    assert readme.index("## Quickstart") < readme.index("## Performance")
    assert "SECURITY.md" in readme
    assert "CONTRIBUTING.md" in readme
    assert "CITATION.cff" in readme
    assert "docs/release/support_matrix.md" in readme


def test_readme_avoids_internal_campaign_ledger_as_primary_content() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert len(readme.splitlines()) <= 400
    assert readme.count("Campaign ") <= 6
    assert "Phase 11" not in readme[:3000]


def test_mkdocs_navigation_covers_user_and_engineering_journeys() -> None:
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    for required in (
        "Getting started:",
        "User guide:",
        "Accelerators:",
        "API reference:",
        "Engineering:",
        "Security:",
    ):
        assert required in config


def test_security_policy_has_private_reporting_and_supported_versions() -> None:
    policy = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert "Privately report" in policy
    assert "Supported versions" in policy
    assert "Do not open a public issue" in policy


def test_contributing_is_human_first() -> None:
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "## Development setup" in contributing
    assert "## Pull request checklist" in contributing
    assert "Codex-driven development" not in contributing[:500]


def test_public_project_surfaces_use_wolfgang_brand() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs_index = (ROOT / "docs/index.md").read_text(encoding="utf-8")
    mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")

    assert readme.startswith("# Wolfgang")
    assert docs_index.startswith("# Wolfgang")
    assert "site_name: Wolfgang" in mkdocs

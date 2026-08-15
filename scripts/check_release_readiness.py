#!/usr/bin/env python3
"""Check Wolfgang release-readiness documentation for claim drift.

This checker enforces the release support boundary. It does not build release
artifacts; `scripts/validate_release_artifacts.py` owns that heavier evidence.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SUPPORT_MATRIX = "docs/release/support_matrix.md"
NEXT_CHECKPOINT_PLAN = "docs/plans/release_candidate_next_checkpoint_plan.md"
RELEASE_STANDARDS = "docs/quality/release_and_packaging.md"
RELEASE_INDEX = "docs/release/README.md"
CHANGELOG = "CHANGELOG.md"
README = "README.md"
PROVENANCE = "docs/research/provenance.md"
ROADMAP = "docs/roadmap.md"
AGENTS = "AGENTS.md"
PYPROJECT = "pyproject.toml"
VERSION_MODULE = "python/wolfgang_quantum/_version.py"
CMAKE = "CMakeLists.txt"
VALIDATE = "scripts/validate.py"
LATEST_RELEASE_VERSION = "0.1.0"

LEGACY_RELEASE_ARTIFACT_PREFIXES = {
    "0.1.0rc1": "fastpauli",
    "0.1.0rc2": "fastpauli",
    "0.1.0": "fastpauli",
}
LEGACY_RELEASE_REPOSITORIES = (
    "https://github.com/sghowell/FastPauli",
    "https://github.com/wolfgang-quantum/wolfgang",
)


def release_display_label(version: str) -> str:
    """Return the release-ledger display label for a PEP 440 project version."""
    rc_match = re.fullmatch(r"(\d+\.\d+\.\d+)rc(\d+)", version)
    if rc_match is not None:
        return f"{rc_match.group(1)}-rc{rc_match.group(2)}"
    return version


def current_release_ledger_path(version: str) -> str:
    return f"docs/release/{release_display_label(version)}.md"


def release_ledger_is_published(ledger: str, version: str) -> bool:
    published_markers = (
        f"Status: published as GitHub prerelease `v{version}`",
        f"Status: published as package-index release `v{version}`",
    )
    return any(marker in ledger for marker in published_markers)


def version_is_release_candidate(version: str) -> bool:
    return re.fullmatch(r"\d+\.\d+\.\d+rc\d+", version) is not None


def project_distribution_name() -> str:
    text = read_text(PYPROJECT)
    match = re.search(r'^name = "([^"]+)"$', text, re.MULTILINE)
    if match is None:
        raise SystemExit("could not find project name in pyproject.toml")
    return match.group(1)


def legacy_release_artifact_prefix(version: str) -> str | None:
    return LEGACY_RELEASE_ARTIFACT_PREFIXES.get(version)


def release_sdist_prefix(version: str) -> str:
    legacy_prefix = legacy_release_artifact_prefix(version)
    if legacy_prefix is not None:
        return legacy_prefix
    return project_distribution_name()


def release_wheel_prefix(version: str) -> str:
    legacy_prefix = LEGACY_RELEASE_ARTIFACT_PREFIXES.get(version)
    if legacy_prefix is not None:
        return legacy_prefix
    return project_distribution_name().replace("-", "_")


def release_checksums_prefix(version: str) -> str:
    legacy_prefix = legacy_release_artifact_prefix(version)
    if legacy_prefix is not None:
        return legacy_prefix
    return project_distribution_name()


def final_wheelhouse_filenames(version: str) -> tuple[str, ...]:
    sdist_prefix = release_sdist_prefix(version)
    wheel_prefix = release_wheel_prefix(version)
    checksums_prefix = release_checksums_prefix(version)
    return (
        f"{sdist_prefix}-{version}.tar.gz",
        f"{wheel_prefix}-{version}-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        f"{wheel_prefix}-{version}-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        f"{wheel_prefix}-{version}-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        f"{wheel_prefix}-{version}-cp310-cp310-macosx_11_0_arm64.whl",
        f"{wheel_prefix}-{version}-cp311-cp311-macosx_11_0_arm64.whl",
        f"{wheel_prefix}-{version}-cp312-cp312-macosx_11_0_arm64.whl",
        f"{checksums_prefix}-{version}.checksums.txt",
    )


def release_candidate_artifact_filenames(version: str) -> tuple[str, ...]:
    sdist_prefix = release_sdist_prefix(version)
    wheel_prefix = release_wheel_prefix(version)
    checksums_prefix = release_checksums_prefix(version)
    return (
        f"{sdist_prefix}-{version}.tar.gz",
        f"{wheel_prefix}-{version}-cp312-cp312-macosx_26_0_arm64.whl",
        f"{checksums_prefix}-{version}.checksums.txt",
    )


def required_published_artifact_filenames(version: str) -> tuple[str, ...]:
    if version_is_release_candidate(version):
        return release_candidate_artifact_filenames(version)
    return final_wheelhouse_filenames(version)


def ledger_has_sha256_for_filename(ledger: str, filename: str) -> bool:
    pattern = rf"filename: {re.escape(filename)}\n\s+sha256: [0-9a-f]{{64}}"
    return re.search(pattern, ledger) is not None


def check_concrete_publication_evidence(
    *,
    ledger: str,
    version: str,
    ledger_path: str,
    support: str,
    failures: list[str],
) -> None:
    revision_match = re.search(r"^Release revision: ([0-9a-f]{40})$", ledger, re.MULTILINE)
    ci_match = re.search(r"^Hosted CI run: ([0-9]+)$", ledger, re.MULTILINE)
    repository_pattern = "|".join(re.escape(url) for url in LEGACY_RELEASE_REPOSITORIES)
    actions_url_match = re.search(
        rf"^url: (?:{repository_pattern})/actions/runs/[0-9]+$",
        ledger,
        re.MULTILINE,
    )
    sha256_matches = re.findall(r"sha256: ([0-9a-f]{64})", ledger)
    required_artifacts = required_published_artifact_filenames(version)

    require(revision_match is not None, f"{ledger_path} is missing concrete release revision", failures)
    require(ci_match is not None, f"{ledger_path} is missing concrete hosted CI run", failures)
    require(actions_url_match is not None, f"{ledger_path} is missing concrete CI run URL", failures)
    require(
        len(sha256_matches) >= len(required_artifacts),
        f"{ledger_path} must record concrete sha256 values for every expected artifact",
        failures,
    )
    for filename in required_artifacts:
        require(
            ledger_has_sha256_for_filename(ledger, filename),
            f"{ledger_path} must record a concrete sha256 next to {filename}",
            failures,
        )
    require(
        "recorded from" not in ledger and "recorded in closeout" not in ledger,
        f"{ledger_path} still contains placeholder closeout wording",
        failures,
    )
    require(
        (
            f"Published checkpoint: v{version} GitHub prerelease" in support
            or f"Published release: v{version}" in support
        ),
        f"{SUPPORT_MATRIX} does not name published release/checkpoint v{version}",
        failures,
    )


def check_prepublication_evidence(
    *,
    ledger: str,
    version: str,
    ledger_path: str,
    support: str,
    failures: list[str],
) -> None:
    expected_status = (
        f"Status: prepared for GitHub prerelease `v{version}`; not yet published."
        if version_is_release_candidate(version)
        else f"Status: prepared for PyPI package-index release `v{version}`; not yet published."
    )
    require(expected_status in ledger, f"{ledger_path} must use pre-publication status", failures)
    require(
        (
            f"Release candidate under finalization: v{version} pending publication" in support
            or f"Release under finalization: v{version} pending publication" in support
            or (
                f"Latest tagged release: v{version}" in support
                and "PyPI status: publication pending" in support
            )
        ),
        f"{SUPPORT_MATRIX} must mark v{version} as pending publication before release assets exist",
        failures,
    )
    for marker in (
        f"Published checkpoint: v{version} GitHub prerelease",
        f"Published release: v{version}",
    ):
        require(
            marker not in support,
            f"{SUPPORT_MATRIX} must not call v{version} published before release evidence is concrete",
            failures,
        )
    for pattern in (
        re.compile(rf"\bGitHub prerelease artifacts are published for v{re.escape(version)}\b"),
        re.compile(rf"\bpackage-index artifacts are published for v{re.escape(version)}\b", re.IGNORECASE),
        re.compile(rf"\bPyPI artifacts are published for v{re.escape(version)}\b", re.IGNORECASE),
        re.compile(r"^PyPI status: published\b", re.IGNORECASE | re.MULTILINE),
    ):
        require(
            pattern.search(ledger) is None,
            f"{ledger_path} must not claim published artifacts before release evidence is concrete",
            failures,
        )

ROUTED_DOCS = (
    PROVENANCE,
    ROADMAP,
    AGENTS,
    RELEASE_INDEX,
    RELEASE_STANDARDS,
    VALIDATE,
)

CLAIM_DOCS = (
    README,
    ROADMAP,
    CHANGELOG,
    RELEASE_INDEX,
    RELEASE_STANDARDS,
)

SUPPORT_MATRIX_TERMS = (
    "| CPU default package | CPU artifact target |",
    "| CUDA accelerator | Source-build support |",
    "| ROCm/HIP accelerator | Source-build support |",
    "| Apple Metal accelerator | Source-build evidence |",
    "| Combined accelerator binary | Unsupported by policy |",
    "| Windows | Unsupported release target |",
    "| TestPyPI validation | Published final dry run |",
    "| PyPI final release | Not published |",
    "CUDA wheels remain unavailable",
    "ROCm/HIP wheels remain unavailable",
    "Metal wheels remain unavailable",
    "Combined accelerator wheels remain unavailable",
    "Windows wheels remain unavailable",
    "PyPI publication is not claimed",
    "Generic Apple GPU support is unavailable",
    "Broader AMD GPU support remains unavailable",
)

NEXT_CHECKPOINT_TERMS = (
    "python scripts/check_release_readiness.py",
    "python scripts/validate.py",
    "python scripts/validate_release_artifacts.py --output-dir <artifact-dir>",
    "version metadata and changelog entry",
    "clean git status at release revision",
    "hosted CI run for the release revision",
    "benchmark report references for any performance claims",
)

BOUNDARY_TERMS = (
    "CPU wheels",
    "CUDA source-build support",
    "ROCm/HIP source-build support",
    "Apple Metal source-build evidence",
    "CUDA wheels remain unavailable",
    "ROCm/HIP wheels remain unavailable",
    "Metal wheels remain unavailable",
    "Combined accelerator wheels remain unavailable",
)

FORBIDDEN_CLAIM_PATTERNS = (
    re.compile(r"\bCUDA wheels (?:are |now )?available\b", re.IGNORECASE),
    re.compile(r"\bROCm/HIP wheels (?:are |now )?available\b", re.IGNORECASE),
    re.compile(r"\bMetal wheels (?:are |now )?available\b", re.IGNORECASE),
    re.compile(r"\bcombined accelerator wheels (?:are |now )?available\b", re.IGNORECASE),
    re.compile(r"\bWindows wheels (?:are |now )?available\b", re.IGNORECASE),
    re.compile(r"\bPyPI publication is complete\b", re.IGNORECASE),
    re.compile(r"\bpackage-index publication is complete\b", re.IGNORECASE),
    re.compile(r"\bgeneric Apple GPU support is available\b", re.IGNORECASE),
    re.compile(r"\bbroad AMD GPU support is available\b", re.IGNORECASE),
)


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def project_version() -> str:
    text = read_text(PYPROJECT)
    match = re.search(r'^version = "([^"]+)"$', text, re.MULTILINE)
    if match is None:
        raise SystemExit("could not find project version in pyproject.toml")
    return match.group(1)


def module_version() -> str:
    text = read_text(VERSION_MODULE)
    match = re.search(r'^__version__ = "([^"]+)"$', text, re.MULTILINE)
    if match is None:
        raise SystemExit(f"could not find __version__ in {VERSION_MODULE}")
    return match.group(1)


def cmake_version() -> str:
    text = read_text(CMAKE)
    match = re.search(r"project\([\s\S]*?\bVERSION\s+(\d+\.\d+\.\d+)\b", text)
    if match is None:
        raise SystemExit("could not find project version in CMakeLists.txt")
    return match.group(1)


def check_release_readiness() -> list[str]:
    failures: list[str] = []

    version = project_version()
    release_version = LATEST_RELEASE_VERSION
    current_ledger = current_release_ledger_path(release_version)

    for path in (
        SUPPORT_MATRIX,
        NEXT_CHECKPOINT_PLAN,
        current_ledger,
        RELEASE_STANDARDS,
        RELEASE_INDEX,
        CHANGELOG,
        README,
        ROADMAP,
        AGENTS,
        PROVENANCE,
        PYPROJECT,
        VERSION_MODULE,
        CMAKE,
        VALIDATE,
    ):
        require(
            (ROOT / path).exists(),
            f"missing required release-readiness path: {path}",
            failures,
        )

    require(
        module_version() == version,
        "pyproject.toml version and python/wolfgang_quantum/_version.py do not match",
        failures,
    )
    require(
        cmake_version() == version,
        "pyproject.toml version and CMakeLists.txt project version do not match",
        failures,
    )

    support = read_text(SUPPORT_MATRIX)
    for term in SUPPORT_MATRIX_TERMS:
        require(term in support, f"{SUPPORT_MATRIX} is missing required term: {term}", failures)
    for term in NEXT_CHECKPOINT_TERMS:
        require(term in support, f"{SUPPORT_MATRIX} is missing checkpoint term: {term}", failures)
    require(
        f"Source version: {version}" in support,
        f"{SUPPORT_MATRIX} does not name current source version {version}",
        failures,
    )
    require(
        f"Latest tagged release: v{release_version}" in support,
        f"{SUPPORT_MATRIX} does not name latest tagged release v{release_version}",
        failures,
    )
    require(
        current_ledger in support,
        f"{SUPPORT_MATRIX} does not reference current release ledger {current_ledger}",
        failures,
    )

    plan = read_text(NEXT_CHECKPOINT_PLAN)
    for term in (
        SUPPORT_MATRIX,
        "scripts/check_release_readiness.py",
        "scripts/validate_release_artifacts.py",
        "0.1.0rc2",
        "0.1.0",
    ):
        require(term in plan, f"{NEXT_CHECKPOINT_PLAN} is missing required term: {term}", failures)

    for path in ROUTED_DOCS:
        text = read_text(path)
        require(SUPPORT_MATRIX in text, f"{path} does not route to {SUPPORT_MATRIX}", failures)
        require(
            NEXT_CHECKPOINT_PLAN in text,
            f"{path} does not route to {NEXT_CHECKPOINT_PLAN}",
            failures,
        )
        require(current_ledger in text, f"{path} does not route to {current_ledger}", failures)

    ledger = read_text(current_ledger)
    common_ledger_terms = (
        f"Package version: {release_version}",
        f"Git tag: v{release_version}",
        f"/releases/tag/v{release_version}",
        "source distribution:",
        f"filename: {release_sdist_prefix(release_version)}-{release_version}.tar.gz",
        "external checksum manifest:",
        f"filename: {release_checksums_prefix(release_version)}-{release_version}.checksums.txt",
        "PyPI publication is not claimed",
    )
    release_candidate_terms = (
        "macOS arm64 CPU wheel:",
        f"filename: {release_wheel_prefix(release_version)}-{release_version}-cp312-cp312-macosx_26_0_arm64.whl",
    )
    final_release_terms = (
        "manylinux x86_64 CPU wheels:",
        "macOS arm64 CPU wheels:",
        "TestPyPI status: published for final dry run; clean install smoke passed",
        "PyPI status: not published; blocked by PyPI trusted-publisher invalid-publisher",
        "PyPI trusted publisher status: not configured for the observed pypi environment claims",
        *tuple(
            f"filename: {filename}"
            for filename in final_wheelhouse_filenames(release_version)
        ),
    )
    ledger_terms = (
        common_ledger_terms + release_candidate_terms
        if version_is_release_candidate(release_version)
        else common_ledger_terms + final_release_terms
    )
    for term in ledger_terms:
        require(term in ledger, f"{current_ledger} is missing required term: {term}", failures)
    if release_ledger_is_published(ledger, release_version):
        check_concrete_publication_evidence(
            ledger=ledger,
            version=release_version,
            ledger_path=current_ledger,
            support=support,
            failures=failures,
        )
    else:
        check_prepublication_evidence(
            ledger=ledger,
            version=release_version,
            ledger_path=current_ledger,
            support=support,
            failures=failures,
        )

    combined_claim_text = "\n".join(
        read_text(path)
        for path in (
            SUPPORT_MATRIX,
            RELEASE_STANDARDS,
            RELEASE_INDEX,
            CHANGELOG,
            README,
            ROADMAP,
        )
    )
    for term in BOUNDARY_TERMS:
        require(term in combined_claim_text, f"release docs are missing boundary term: {term}", failures)

    for path in CLAIM_DOCS:
        text = read_text(path)
        for pattern in FORBIDDEN_CLAIM_PATTERNS:
            match = pattern.search(text)
            require(
                match is None,
                f"{path} contains forbidden release claim: {pattern.pattern}",
                failures,
            )

    changelog = read_text(CHANGELOG)
    require("## Unreleased" in changelog, "CHANGELOG.md is missing an Unreleased section", failures)
    require(
        f"Next version: {version}" in changelog,
        f"CHANGELOG.md does not name next source version {version}",
        failures,
    )
    require(
        "Release-readiness hardening for the next public checkpoint" in changelog,
        "CHANGELOG.md does not describe the next-checkpoint release hardening",
        failures,
    )

    standards = read_text(RELEASE_STANDARDS)
    require(
        "scripts/check_release_readiness.py" in standards,
        f"{RELEASE_STANDARDS} does not require the release-readiness checker",
        failures,
    )
    require(
        SUPPORT_MATRIX in standards,
        f"{RELEASE_STANDARDS} does not reference the release support matrix",
        failures,
    )

    return failures


def main() -> None:
    failures = check_release_readiness()
    if failures:
        print("Release-readiness checks failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)
    print("release-readiness checks passed")


if __name__ == "__main__":
    main()

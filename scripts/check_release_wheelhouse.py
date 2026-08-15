#!/usr/bin/env python3
"""Check FastPauli final-release wheelhouse configuration."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PLAN = "docs/plans/release_0_1_0_wheelhouse_foundation_plan.md"
WORKFLOW = ".github/workflows/release-wheelhouse.yml"
PYPROJECT = "pyproject.toml"
README = "README.md"
PROVENANCE = "docs/research/provenance.md"
ROADMAP = "docs/roadmap.md"
AGENTS = "AGENTS.md"
RELEASE_INDEX = "docs/release/README.md"
RELEASE_STANDARDS = "docs/quality/release_and_packaging.md"
SUPPORT_MATRIX = "docs/release/support_matrix.md"

FORBIDDEN_CLASSIFIERS = (
    "Programming Language :: C++ :: 20",
)

ROUTED_DOCS = (
    PROVENANCE,
    ROADMAP,
    AGENTS,
    RELEASE_INDEX,
    RELEASE_STANDARDS,
    SUPPORT_MATRIX,
)

PLAN_TERMS = (
    "final 0.1.0 CPU wheelhouse",
    "manylinux x86_64",
    "macOS arm64",
    "Python 3.10, 3.11, and 3.12",
    "TestPyPI dry run",
    "PyPI trusted publishing",
    "CUDA wheels remain unavailable",
    "ROCm/HIP wheels remain unavailable",
    "Metal wheels remain unavailable",
    "Windows wheels remain unavailable",
    "package-index publication is unavailable until the explicit publish gate succeeds",
)

WORKFLOW_TERMS = (
    "workflow_dispatch:",
    "publish-target:",
    "pypa/cibuildwheel",
    "python -m build --sdist",
    "python -m twine check",
    "python -m twine check publish-dist/*",
    "python scripts/check_release_wheelhouse.py --require-trove-classifiers",
    "Set up Python for metadata checks",
    "scripts/write_release_checksums.py",
    "scripts/prepare_publish_dist.py",
    "--require-cpu-wheelhouse",
    "actions/upload-artifact",
    "pypa/gh-action-pypi-publish@",
    "repository-url: https://test.pypi.org/legacy/",
    "packages-dir: publish-dist",
    "GITHUB_REF_TYPE",
    "GITHUB_REF_NAME",
    "scripts/check_release_tag.py",
    "actions/attest-build-provenance@",
    "pip-audit",
    "cyclonedx-py",
    "id-token: write",
)


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def load_pyproject_toml() -> dict[str, object] | None:
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[import-not-found,no-redef]
        except ModuleNotFoundError:
            return None
    return tomllib.loads(read_text(PYPROJECT))


def check_pyproject_text(failures: list[str]) -> None:
    pyproject = read_text(PYPROJECT)
    required_terms = (
        '"cibuildwheel>=2.23,<3; python_version < \'3.11\'"',
        '"cibuildwheel>=3.0,<4; python_version >= \'3.11\'"',
        'build = ["cp310-*", "cp311-*", "cp312-*"]',
        'skip = ["*-win32", "*-win_amd64", "*-win_arm64", "*-musllinux*", "pp*"]',
        'test-command = "python {project}/scripts/wheel_smoke.py"',
        '"cmake.define.WOLFGANG_ENABLE_CUDA" = "OFF"',
        '"cmake.define.WOLFGANG_ENABLE_HIP" = "OFF"',
        '"cmake.define.WOLFGANG_ENABLE_METAL" = "OFF"',
        '"cmake.define.WOLFGANG_ENABLE_NATIVE" = "OFF"',
        "[tool.cibuildwheel.linux]",
        'archs = ["x86_64"]',
        "[tool.cibuildwheel.macos]",
        'archs = ["arm64"]',
    )
    for term in required_terms:
        require(term in pyproject, f"{PYPROJECT} is missing required term: {term}", failures)
    for classifier in FORBIDDEN_CLASSIFIERS:
        require(
            classifier not in pyproject,
            (
                f"{PYPROJECT} uses invalid Trove classifier {classifier!r}; "
                "use 'Programming Language :: C++' and document the C++20 baseline outside classifiers"
            ),
            failures,
        )


def check_project_classifiers(
    classifiers: object,
    failures: list[str],
    *,
    require_trove_classifiers: bool,
) -> None:
    if not isinstance(classifiers, list) or not all(
        isinstance(classifier, str) for classifier in classifiers
    ):
        failures.append(f"{PYPROJECT} project.classifiers must be a list of strings")
        return

    for classifier in FORBIDDEN_CLASSIFIERS:
        require(
            classifier not in classifiers,
            (
                f"{PYPROJECT} uses invalid Trove classifier {classifier!r}; "
                "TestPyPI rejected this classifier during release workflow run 25458693782"
            ),
            failures,
        )

    try:
        from trove_classifiers import classifiers as valid_classifiers
        from trove_classifiers import deprecated_classifiers
    except ModuleNotFoundError:
        if require_trove_classifiers:
            failures.append(
                "trove-classifiers must be installed when --require-trove-classifiers is used"
            )
        return

    for classifier in classifiers:
        if classifier in deprecated_classifiers:
            replacements = ", ".join(deprecated_classifiers[classifier])
            failures.append(
                f"{PYPROJECT} uses deprecated Trove classifier {classifier!r}; "
                f"use replacement(s): {replacements}"
            )
        elif classifier not in valid_classifiers:
            failures.append(f"{PYPROJECT} uses invalid Trove classifier {classifier!r}")


def check_pyproject(
    failures: list[str],
    *,
    require_trove_classifiers: bool = False,
) -> None:
    data = load_pyproject_toml()
    if data is None:
        check_pyproject_text(failures)
        if require_trove_classifiers:
            failures.append("tomllib is required for --require-trove-classifiers")
        return

    project = data.get("project", {})
    check_project_classifiers(
        project.get("classifiers", []),
        failures,
        require_trove_classifiers=require_trove_classifiers,
    )

    test_dependencies = project.get("optional-dependencies", {}).get("test", [])
    require(
        "cibuildwheel>=2.23,<3; python_version < '3.11'" in test_dependencies,
        "test extra must install a Python 3.10-compatible cibuildwheel 2.x release",
        failures,
    )
    require(
        "cibuildwheel>=3.0,<4; python_version >= '3.11'" in test_dependencies,
        "test extra must use cibuildwheel 3.x on Python 3.11+",
        failures,
    )

    cibw = data.get("tool", {}).get("cibuildwheel", {})
    require(
        cibw.get("build") == ["cp310-*", "cp311-*", "cp312-*"],
        "cibuildwheel build selector must target CPython 3.10-3.12",
        failures,
    )
    require(
        cibw.get("skip")
        == ["*-win32", "*-win_amd64", "*-win_arm64", "*-musllinux*", "pp*"],
        "cibuildwheel skip selector must keep Windows, musllinux, and PyPy out of the release lane",
        failures,
    )
    require(
        cibw.get("test-command") == "python {project}/scripts/wheel_smoke.py",
        "cibuildwheel must run the repo wheel smoke script",
        failures,
    )

    config = cibw.get("config-settings", {})
    for name in (
        "WOLFGANG_ENABLE_CUDA",
        "WOLFGANG_ENABLE_HIP",
        "WOLFGANG_ENABLE_METAL",
        "WOLFGANG_ENABLE_NATIVE",
    ):
        key = f"cmake.define.{name}"
        require(config.get(key) == "OFF", f"cibuildwheel config-setting {key}=OFF is required", failures)

    linux = cibw.get("linux", {})
    macos = cibw.get("macos", {})
    require(linux.get("archs") == ["x86_64"], "cibuildwheel Linux archs must be x86_64 only", failures)
    require(macos.get("archs") == ["arm64"], "cibuildwheel macOS archs must be arm64 only", failures)


def check_workflow(failures: list[str]) -> None:
    workflow = read_text(WORKFLOW)
    for term in WORKFLOW_TERMS:
        require(term in workflow, f"{WORKFLOW} is missing required term: {term}", failures)
    require(
        re.search(
            r"collect-wheelhouse:[\s\S]+?Set up Python for metadata checks"
            r"[\s\S]+?uses: actions/setup-python@[0-9a-f]{40}"
            r"[\s\S]+?python-version: \"3\.12\""
            r"[\s\S]+?python -m twine check dist/\*",
            workflow,
        )
        is not None,
        f"{WORKFLOW} must set up Python 3.12 in collect-wheelhouse before twine checks merged artifacts",
        failures,
    )
    require(
        "packages-dir: dist" not in workflow,
        f"{WORKFLOW} must not publish the checksum-containing dist directory",
        failures,
    )
    require(
        'python scripts/check_release_tag.py "${GITHUB_REF_TYPE}" "${GITHUB_REF_NAME}"'
        in workflow,
        f"{WORKFLOW} must bind publication to the exact project-version tag",
        failures,
    )
    require(
        "startsWith(github.ref_name, 'v')" not in workflow and "v*" not in workflow,
        f"{WORKFLOW} must not use a prefix-only release-tag gate",
        failures,
    )


def check_release_wheelhouse(*, require_trove_classifiers: bool = False) -> list[str]:
    failures: list[str] = []

    for path in (
        PLAN,
        WORKFLOW,
        PYPROJECT,
        "scripts/wheel_smoke.py",
        "scripts/write_release_checksums.py",
        "scripts/prepare_publish_dist.py",
    ):
        require((ROOT / path).exists(), f"missing release wheelhouse path: {path}", failures)

    if not failures:
        plan = read_text(PLAN)
        for term in PLAN_TERMS:
            require(term in plan, f"{PLAN} is missing required term: {term}", failures)

        for path in ROUTED_DOCS:
            require(PLAN in read_text(path), f"{path} does not route to {PLAN}", failures)

        check_pyproject(failures, require_trove_classifiers=require_trove_classifiers)
        check_workflow(failures)

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-trove-classifiers",
        action="store_true",
        help="fail unless trove-classifiers is installed and every project classifier is valid",
    )
    args = parser.parse_args()

    failures = check_release_wheelhouse(
        require_trove_classifiers=args.require_trove_classifiers
    )
    if failures:
        print("Release wheelhouse checks failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)
    print("release wheelhouse checks passed")


if __name__ == "__main__":
    main()

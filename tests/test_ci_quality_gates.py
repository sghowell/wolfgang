from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUALITY_WORKFLOW = ROOT / ".github/workflows/quality.yml"
DOCS_WORKFLOW = ROOT / ".github/workflows/docs.yml"
CI_WORKFLOW = ROOT / ".github/workflows/ci.yml"
CODEQL_WORKFLOW = ROOT / ".github/workflows/codeql.yml"
RELEASE_README = ROOT / "docs/release/README.md"
FULL_SHA_ACTION = re.compile(r"^\s*uses:\s+[^\s@]+@[0-9a-f]{40}\s+#\s+\S+\s*$", re.MULTILINE)
CODEQL_ACTION_REF = "github/codeql-action/init@c4dd10e44af883a891fe31ced449bcb4a6728b9b # v3.37.6"
CODEQL_ANALYZE_REF = (
    "github/codeql-action/analyze@c4dd10e44af883a891fe31ced449bcb4a6728b9b # v3.37.6"
)


def workflow_text(path: Path) -> str:
    assert path.exists(), f"missing workflow: {path.relative_to(ROOT)}"
    return path.read_text(encoding="utf-8")


def test_ci_matrix_keys_with_hyphens_use_bracket_expressions() -> None:
    workflow = workflow_text(CI_WORKFLOW)

    assert "matrix.qiskit-requirement" not in workflow
    assert "matrix.openfermion-requirement" not in workflow
    assert workflow.count("matrix['qiskit-requirement']") == 2
    assert workflow.count("matrix['openfermion-requirement']") == 2
    assert not re.search(
        r"^      RELEASE_ARTIFACT_DIR: \$\{\{ runner\.temp \}\}", workflow, re.MULTILINE
    )
    assert re.search(
        r"^          RELEASE_ARTIFACT_DIR: \$\{\{ runner\.temp \}\}", workflow, re.MULTILINE
    )


def test_codeql_uses_supported_per_language_build_modes() -> None:
    workflow = workflow_text(CODEQL_WORKFLOW)

    assert "if: ${{ !github.event.repository.private }}" in workflow
    assert "language: c-cpp\n            build-mode: manual" in workflow
    assert "language: python\n            build-mode: none" in workflow
    assert "languages: c-cpp,python" not in workflow


def test_codeql_pins_current_codeql_action_release_hashes() -> None:
    workflow = workflow_text(CODEQL_WORKFLOW)

    assert CODEQL_ACTION_REF in workflow
    assert CODEQL_ANALYZE_REF in workflow


def test_release_evidence_recipe_requires_out_of_tree_docs_build() -> None:
    readme = workflow_text(RELEASE_README)

    assert "mkdocs build --strict --site-dir <site-dir-outside-repo>" in readme
    assert "mkdocs build --strict --site-dir site" not in readme


def test_release_evidence_recipe_requires_pristine_snapshot_packaging() -> None:
    readme = workflow_text(RELEASE_README)

    assert "git worktree add --detach <snapshot-dir> HEAD" in readme
    assert (
        "python <snapshot-dir>/scripts/validate_release_artifacts.py --output-dir <artifact-dir>"
        in readme
    )


def test_release_evidence_recipe_twine_checks_only_distribution_files() -> None:
    readme = workflow_text(RELEASE_README)

    assert "python -m twine check <artifact-dir>/*.tar.gz <artifact-dir>/*.whl" in readme
    assert "python -m twine check <artifact-dir>/*\n" not in readme


def test_quality_workflow_has_hardened_job_boundaries() -> None:
    workflow = workflow_text(QUALITY_WORKFLOW)

    assert re.search(r"^permissions:\n  contents: read$", workflow, re.MULTILINE)
    assert re.search(
        r"^concurrency:\n  group: .+\n  cancel-in-progress: true$", workflow, re.MULTILINE
    )
    assert {"quality", "package", "sanitizers"} <= set(
        re.findall(r"^  ([a-z][a-z0-9-]+):$", workflow, re.MULTILINE)
    )
    assert workflow.count("timeout-minutes:") >= 3
    assert workflow.count("persist-credentials: false") >= 3
    uses_lines = re.findall(r"^\s*uses:.*$", workflow, re.MULTILINE)
    assert uses_lines
    assert all(FULL_SHA_ACTION.fullmatch(line) for line in uses_lines), uses_lines


def test_quality_job_gates_whole_repository_python_and_public_artifacts() -> None:
    workflow = workflow_text(QUALITY_WORKFLOW)

    for required in (
        "ruff check --config ruff.toml .",
        "pyright python/wolfgang_quantum",
        'codespell --skip="./docs/javascripts/vendor/mermaid-11.4.1.min.js" .',
        "python scripts/audit_public_artifacts.py --tracked",
    ):
        assert required in workflow
    for pinned_tool in (
        '"ruff==0.16.2"',
        "pyright==",
        "codespell==",
        "numpy==",
        "openfermion==",
        "qiskit==",
    ):
        assert pinned_tool in workflow
    assert "-DWOLFGANG_ENABLE_INTERNAL_BINDINGS=ON" in workflow


def test_ruff_config_has_no_obsolete_rule_exemptions() -> None:
    config = (ROOT / "ruff.toml").read_text(encoding="utf-8")

    assert "UP038" not in config


def test_package_job_builds_and_audits_both_distribution_types() -> None:
    workflow = workflow_text(QUALITY_WORKFLOW)

    assert "python -m build --sdist --wheel" in workflow
    assert "cmake.define.WOLFGANG_ENABLE_INTERNAL_BINDINGS=OFF" in workflow
    assert "python scripts/audit_public_artifacts.py --sdist" in workflow
    assert "python -m twine check dist/*" in workflow
    assert "build==" in workflow
    assert "twine==" in workflow


def test_sanitizer_job_configures_builds_and_runs_native_extension_tests() -> None:
    workflow = workflow_text(QUALITY_WORKFLOW)

    assert "runs-on: ubuntu-latest" in workflow
    assert "-fsanitize=address,undefined" in workflow
    assert "ASAN_OPTIONS:" in workflow
    assert "UBSAN_OPTIONS:" in workflow
    assert "cmake -S . -B build/sanitizers" in workflow
    assert (
        'CMAKE_LIBRARY_OUTPUT_DIRECTORY="${GITHUB_WORKSPACE}/python/wolfgang_quantum"' in workflow
    )
    assert "cmake --build build/sanitizers --target _wolfgang_core --parallel 2" in workflow
    assert "cmake --build build/sanitizers --target _fastpauli_core" not in workflow
    assert "PYTHONPATH=python" in workflow
    assert "python -m pytest" in workflow


def test_documentation_gate_is_strict_pinned_and_bounded() -> None:
    workflow = workflow_text(DOCS_WORKFLOW)

    assert 'python -m pip install -e ".[test,docs]"' in workflow
    assert "mkdocs build --strict" in workflow
    assert "python -m pytest tests/docs_mermaid_integration.py -q" in workflow
    assert workflow.count("timeout-minutes:") >= 2
    assert workflow.count("!github.event.repository.private") == 2
    uses_lines = re.findall(r"^\s*uses:.*$", workflow, re.MULTILINE)
    assert uses_lines
    assert all(FULL_SHA_ACTION.fullmatch(line) for line in uses_lines), uses_lines


def test_docs_extra_pins_the_documentation_toolchain() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "docs = [" in pyproject
    assert '"mkdocs-material==9.7.7"' in pyproject


def test_codespell_skips_vendored_mermaid_runtime() -> None:
    workflow = workflow_text(QUALITY_WORKFLOW)

    assert 'codespell --skip="./docs/javascripts/vendor/mermaid-11.4.1.min.js" .' in workflow

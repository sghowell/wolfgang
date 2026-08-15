from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ACTIVE_DOC_DIRECTORIES = (
    "docs/architecture",
    "docs/benchmarks",
    "docs/quality",
    "docs/guide",
    "docs/getting-started",
    "docs/user",
    "docs/api",
    "docs/accelerators",
    "docs/release",
)
ACTIVE_DOC_EXCEPTIONS = {
    "docs/guide/architecture.md",
    "docs/guide/python-api.md",
    "docs/getting-started/quickstart.md",
    "docs/api/index.md",
    "docs/release/README.md",
    "docs/release/0.1.0.md",
    "docs/release/0.1.0-rc1.md",
    "docs/release/0.1.0-rc2.md",
    "docs/release/0.1.0-wheelhouse-dry-run.md",
    "docs/release/history_sanitization.md",
}

ACTIVE_DOC_SUBTREE_EXCEPTIONS = (
    "docs/benchmarks/reports/",
)

ACTIVE_DOC_FILES = (
    "docs/roadmap.md",
)

ACTIVE_DOC_TOKEN_EXCEPTIONS = {
    "docs/architecture/adapter_contracts.md": ("legacy `fastpauli` package remains a compatibility shim",),
    "docs/quality/agent_harness.md": ("docs/plans/fastpauli_cpp_cuda_implementation_plan.md",),
    "docs/roadmap.md": ("docs/plans/fastpauli_cpp_cuda_implementation_plan.md",),
}

ACTIVE_DOC_FASTPAULI_TECHNICAL_TOKEN_RE = re.compile(
    r"FASTPAULI_[A-Z0-9_]+"
    r"|fastpauli_[a-z0-9_]+"
    r"|fastpauli-[0-9][^\s`|)]*"
    r"|docs/plans/fastpauli_[^\s`|)]+"
)


def test_pyproject_uses_wolfgang_distribution_and_public_urls() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'name = "wolfgang-quantum"' in pyproject
    assert '{ name = "Wolfgang contributors" }' in pyproject
    assert 'Homepage = "https://github.com/wolfgang-quantum/wolfgang"' in pyproject
    assert 'Documentation = "https://wolfgangquantum.com"' in pyproject
    assert 'Repository = "https://github.com/wolfgang-quantum/wolfgang.git"' in pyproject
    assert 'Issues = "https://github.com/wolfgang-quantum/wolfgang/issues"' in pyproject


def test_canonical_python_package_root_is_wolfgang_quantum() -> None:
    canonical = ROOT / "python" / "wolfgang_quantum"

    assert canonical.is_dir()
    assert (canonical / "__init__.py").is_file()
    assert (canonical / "__init__.pyi").is_file()
    assert (canonical / "_wolfgang_core.pyi").is_file()
    assert (canonical / "py.typed").is_file()


def test_public_readme_and_docs_use_wolfgang_identity() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs_index = (ROOT / "docs/index.md").read_text(encoding="utf-8")
    installation = (ROOT / "docs/getting-started/installation.md").read_text(encoding="utf-8")

    assert "# Wolfgang" in readme
    assert "wolfgang-quantum" in readme
    assert "wolfgang_quantum" in readme
    assert "# Wolfgang" in docs_index
    assert "Wolfgang" in installation


def test_historical_release_and_research_docs_preserve_fastpauli_provenance() -> None:
    provenance = (ROOT / "docs/research/provenance.md").read_text(encoding="utf-8")
    release_010 = (ROOT / "docs/release/0.1.0.md").read_text(encoding="utf-8")
    history = (ROOT / "docs/release/history_sanitization.md").read_text(encoding="utf-8")

    assert "Historical FastPauli engineering ledger" in provenance
    assert "FastPauli has completed" in provenance
    assert "FastPauli 0.1.0 Release Evidence" in release_010
    assert "FastPauli" in history


def test_active_docs_use_wolfgang_canonical_import_and_header_surfaces() -> None:
    active_docs = {
        "docs/api/index.md": ["wolfgang_quantum", "include/wolfgang/"],
        "docs/benchmarks/protocol.md": ["# Wolfgang Benchmark Protocol", "Wolfgang scalar CPU"],
        "docs/guide/python-api.md": ["wolfgang_quantum.capabilities()", "WolfgangCapabilities"],
        "docs/accelerators/overview.md": ["Wolfgang"],
        "docs/guide/architecture.md": ["include/wolfgang/", "python/wolfgang_quantum/"],
        "docs/getting-started/quickstart.md": ["from wolfgang_quantum import PauliSum"],
        "docs/user/expectation_values.md": ["from wolfgang_quantum import PauliSum"],
        "docs/user/performance.md": ["wolfgang_quantum._wolfgang_core", "import wolfgang_quantum"],
        "docs/architecture/api_stability.md": ["include/wolfgang"],
        "docs/architecture/adapter_contracts.md": ["Importing `wolfgang_quantum` must not import Qiskit or OpenFermion."],
        "docs/quality/python_binding_policy.md": ["wolfgang_quantum.capabilities()"],
        "docs/quality/code_standards.md": ["include/wolfgang"],
        "docs/quality/phase_quality_gates.md": ["import wolfgang_quantum", "wolfgang_quantum.__version__"],
        "docs/release/support_matrix.md": ["# Wolfgang Release Support Matrix", "public support boundary for Wolfgang release"],
    }

    forbidden = (
        "help(fastpauli",
        "include/fastpauli/`: documented native API declarations",
        "put documented user-facing C++ APIs in include/fastpauli",
        "promoting a helper header to include/fastpauli",
        "Call `fastpauli.capabilities()`,",
        "Call `fastpauli.capabilities()`",
    )

    for relative_path, required_fragments in active_docs.items():
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        for fragment in required_fragments:
            assert fragment in text, f"missing canonical Wolfgang fragment {fragment!r} in {relative_path}"
        for fragment in forbidden:
            assert fragment not in text, f"stale canonical FastPauli fragment {fragment!r} remains in {relative_path}"


def test_active_non_provenance_docs_do_not_present_fastpauli_as_current_product() -> None:
    disallowed_fragments = (
        "FastPauli's",
        "FastPauli is",
        "FastPauli uses",
        "FastPauli should",
        "FastPauli does",
        "FastPauli maintains",
        "FastPauli records",
        "FastPauli represents",
        "FastPauli ships",
        "FastPauli builds",
        "FastPauli ran",
        "# FastPauli",
    )

    for directory in ACTIVE_DOC_DIRECTORIES:
        for path in sorted((ROOT / directory).rglob("*.md")):
            relative = path.relative_to(ROOT).as_posix()
            if relative in ACTIVE_DOC_EXCEPTIONS or relative.startswith(ACTIVE_DOC_SUBTREE_EXCEPTIONS):
                continue
            text = path.read_text(encoding="utf-8")
            for fragment in disallowed_fragments:
                assert fragment not in text, f"stale FastPauli-first identity fragment {fragment!r} remains in {relative}"

    for relative in ACTIVE_DOC_FILES:
        text = (ROOT / relative).read_text(encoding="utf-8")
        for fragment in disallowed_fragments:
            assert fragment not in text, f"stale FastPauli-first identity fragment {fragment!r} remains in {relative}"


def test_active_release_tooling_uses_wolfgang_as_canonical_identity() -> None:
    active_release_surfaces = {
        "scripts/check_release_readiness.py": [
            'VERSION_MODULE = "python/wolfgang_quantum/_version.py"',
            '"""Check Wolfgang release-readiness documentation for claim drift.',
        ],
        "scripts/wheel_smoke.py": [
            'import wolfgang_quantum',
            'importlib_metadata.version("wolfgang-quantum")',
            '"package_import": "wolfgang_quantum"',
            '"project_distribution": "wolfgang-quantum"',
        ],
        "scripts/validate_release_artifacts.py": [
            'SDIST_ARTIFACT_PREFIX = "wolfgang-quantum-"',
            'WHEEL_ARTIFACT_PREFIX = "wolfgang_quantum-"',
            'PROJECT_DISTRIBUTION = "wolfgang-quantum"',
            'import wolfgang_quantum',
        ],
        "scripts/write_release_checksums.py": [
            'SDIST_ARTIFACT_PREFIX = "wolfgang-quantum-"',
            'WHEEL_ARTIFACT_PREFIX = "wolfgang_quantum-"',
        ],
        "tests/test_release_supply_chain.py": [
            'python/wolfgang_quantum/_version.py',
            'https://github.com/wolfgang-quantum/wolfgang',
            'https://wolfgangquantum.com',
        ],
        "tests/test_release_artifact_validation.py": [
            'SDIST_PREFIX = f"wolfgang-quantum-{RELEASE_VERSION}"',
            'WHEEL_PREFIX = f"wolfgang_quantum-{RELEASE_VERSION}"',
            'importlib_metadata.version(\'wolfgang-quantum\')',
        ],
    }

    forbidden_fragments = (
        'VERSION_MODULE = "python/fastpauli/_version.py"',
        'importlib_metadata.version("fastpauli")',
        'RELEASE_PREFIX = f"fastpauli-{RELEASE_VERSION}"',
        'SDIST_ARTIFACT_PREFIX = "fastpauli-"',
        'WHEEL_ARTIFACT_PREFIX = "fastpauli-"',
        'ARTIFACT_PREFIX = "fastpauli-"',
    )

    for relative_path, required_fragments in active_release_surfaces.items():
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        for fragment in required_fragments:
            assert fragment in text, f"missing canonical Wolfgang release fragment {fragment!r} in {relative_path}"
        for fragment in forbidden_fragments:
            assert fragment not in text, f"stale canonical FastPauli release fragment {fragment!r} remains in {relative_path}"


def test_active_docs_only_keep_allowlisted_fastpauli_tokens() -> None:
    tracked_tokens = ("FastPauli", "fastpauli", "FASTPAULI_")

    for directory in ACTIVE_DOC_DIRECTORIES:
        for path in sorted((ROOT / directory).rglob("*.md")):
            relative = path.relative_to(ROOT).as_posix()
            if relative in ACTIVE_DOC_EXCEPTIONS or relative.startswith(ACTIVE_DOC_SUBTREE_EXCEPTIONS):
                continue
            text = path.read_text(encoding="utf-8")
            allowed = ACTIVE_DOC_TOKEN_EXCEPTIONS.get(relative, ())
            cleaned = text
            for fragment in allowed:
                cleaned = cleaned.replace(fragment, "")
            cleaned = ACTIVE_DOC_FASTPAULI_TECHNICAL_TOKEN_RE.sub("", cleaned)
            assert not any(token in cleaned for token in tracked_tokens), (
                f"active doc retains non-allowlisted FastPauli token in {relative}"
            )

    for relative in ACTIVE_DOC_FILES:
        text = (ROOT / relative).read_text(encoding="utf-8")
        allowed = ACTIVE_DOC_TOKEN_EXCEPTIONS.get(relative, ())
        cleaned = text
        for fragment in allowed:
            cleaned = cleaned.replace(fragment, "")
        cleaned = ACTIVE_DOC_FASTPAULI_TECHNICAL_TOKEN_RE.sub("", cleaned)
        assert not any(token in cleaned for token in tracked_tokens), (
            f"active doc retains non-allowlisted FastPauli token in {relative}"
        )


def test_cpp_headers_use_wolfgang_as_canonical_surface_and_fastpauli_as_legacy_alias() -> None:
    canonical_headers = sorted((ROOT / "include" / "wolfgang").glob("*.hpp"))
    legacy_headers = sorted((ROOT / "include" / "fastpauli").glob("*.hpp"))

    assert canonical_headers
    assert {path.name for path in canonical_headers} == {path.name for path in legacy_headers}

    for canonical in canonical_headers:
        text = canonical.read_text(encoding="utf-8")
        assert '#include "fastpauli/' not in text, f"canonical header still depends on legacy include path: {canonical}"
        assert "namespace wolfgang {" in text, f"canonical declarations missing Wolfgang namespace in {canonical}"
        assert "namespace fastpauli {" not in text, f"canonical header still declares legacy namespace in {canonical}"
        assert "namespace fastpauli = wolfgang;" not in text, (
            f"canonical header should not carry legacy alias compatibility in {canonical}"
        )

    for legacy in legacy_headers:
        text = legacy.read_text(encoding="utf-8")
        assert f'#include "wolfgang/{legacy.name}"' in text, f"legacy header does not forward to canonical Wolfgang header: {legacy}"
        assert "namespace fastpauli {" not in text, f"legacy header should be a forwarder only: {legacy}"
        assert "namespace fastpauli = ::wolfgang;" in text, (
            f"legacy fastpauli alias missing from transition header {legacy}"
        )


def test_cuda_private_headers_use_wolfgang_as_canonical_namespace() -> None:
    private_headers = sorted((ROOT / "src" / "cuda").glob("*.cuh"))

    assert private_headers

    for header in private_headers:
        text = header.read_text(encoding="utf-8")
        assert "namespace wolfgang" in text, f"canonical CUDA private header missing Wolfgang namespace: {header}"
        assert "namespace fastpauli" not in text, (
            f"canonical CUDA private header still declares legacy FastPauli namespace: {header}"
        )

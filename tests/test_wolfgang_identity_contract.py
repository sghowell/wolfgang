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

ACTIVE_BENCHMARK_FILES = (
    "benchmarks/bench_competitive_baselines.py",
    "benchmarks/bench_cpu_dispatch.py",
    "benchmarks/bench_cpu_thresholds.py",
    "benchmarks/bench_cuda_kernels.py",
    "benchmarks/bench_cuda_scaling.py",
    "benchmarks/bench_expectation.py",
    "benchmarks/bench_grouping.py",
    "benchmarks/bench_metal_kernels.py",
    "benchmarks/bench_multiply.py",
    "benchmarks/bench_openfermion_conversion.py",
    "benchmarks/bench_rocm_kernels.py",
    "benchmarks/bench_simplify.py",
)

CURRENT_RENDERER_AND_REPORT_FILES = (
    "scripts/render_apple_metal_assets.py",
    "scripts/render_benchmark_plots.py",
    "scripts/render_cuda_deep_report_assets.py",
    "scripts/run_rocm_release_support_lane.py",
)

CURRENT_PLOT_FILES = (
    "docs/benchmarks/plots/accelerator_landscape_with_rocm.svg",
    "docs/benchmarks/plots/cuda_deep_optimization_architecture.svg",
    "docs/benchmarks/plots/cuda_deep_optimization_h100_optimization_deltas.svg",
    "docs/benchmarks/plots/cuda_deep_optimization_h100_path_speedups.svg",
    "docs/benchmarks/plots/cuda_deep_optimization_h100_profiler_bottlenecks.svg",
    "docs/benchmarks/plots/cuda_deep_optimization_h100_scaling.svg",
    "docs/benchmarks/plots/cuda_h100_nsight_hillclimb_default_backend_speedups.svg",
)


def test_pyproject_uses_wolfgang_distribution_and_public_urls() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'name = "wolfgang-quantum"' in pyproject
    assert '{ name = "Wolfgang contributors" }' in pyproject
    assert 'Homepage = "https://github.com/sghowell/wolfgang"' in pyproject
    assert 'Documentation = "https://sghowell.github.io/wolfgang/"' in pyproject
    assert 'Repository = "https://github.com/sghowell/wolfgang.git"' in pyproject
    assert 'Issues = "https://github.com/sghowell/wolfgang/issues"' in pyproject


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
        "scripts/validate.py": [
            'import wolfgang_quantum and report scalar CPU build info',
            'import wolfgang_quantum._wolfgang_core as core',
        ],
        "scripts/rocm_memory_probe.py": [
            'import wolfgang_quantum',
            'import wolfgang_quantum._wolfgang_core as core',
        ],
        "scripts/rocm_campaign5_candidate_probe.py": [
            'import wolfgang_quantum._wolfgang_core as core',
        ],
        "scripts/b300_blackwell_resume_runner.py": [
            'import wolfgang_quantum._wolfgang_core as core',
        ],
        "scripts/cuda_deep_profile.py": [
            'find_spec("wolfgang_quantum._wolfgang_core")',
            '<wolfgang_extension_path>',
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
            'https://github.com/sghowell/wolfgang',
            'https://sghowell.github.io/wolfgang/',
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


def test_active_benchmark_entrypoints_use_wolfgang_by_default() -> None:
    forbidden_fragments = (
        "import fastpauli",
        "from fastpauli import",
        '"""Compare FastPauli',
        '"""CUDA scaling benchmark for FastPauli hot paths.',
        'The benchmark compares FastPauli scalar CPU expectation kernels',
    )

    required_fragments = {
        "benchmarks/bench_competitive_baselines.py": ("import wolfgang_quantum", "from wolfgang_quantum import PauliSum"),
        "benchmarks/bench_cpu_dispatch.py": ("import wolfgang_quantum", "from wolfgang_quantum import PauliSum"),
        "benchmarks/bench_cpu_thresholds.py": ("import wolfgang_quantum", "from wolfgang_quantum import PauliSum"),
        "benchmarks/bench_cuda_kernels.py": ("import wolfgang_quantum",),
        "benchmarks/bench_cuda_scaling.py": ("import wolfgang_quantum",),
        "benchmarks/bench_expectation.py": ("import wolfgang_quantum", "from wolfgang_quantum import PauliSum"),
        "benchmarks/bench_grouping.py": ("import wolfgang_quantum", "from wolfgang_quantum import PauliSum"),
        "benchmarks/bench_metal_kernels.py": ("import wolfgang_quantum",),
        "benchmarks/bench_multiply.py": ("import wolfgang_quantum", "from wolfgang_quantum import PauliSum"),
        "benchmarks/bench_openfermion_conversion.py": ("import wolfgang_quantum", "from wolfgang_quantum import PauliSum"),
        "benchmarks/bench_rocm_kernels.py": ("import wolfgang_quantum",),
        "benchmarks/bench_simplify.py": ("import wolfgang_quantum", "from wolfgang_quantum import PauliSum"),
    }

    for relative_path in ACTIVE_BENCHMARK_FILES:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        for fragment in required_fragments[relative_path]:
            assert fragment in text, f"missing canonical Wolfgang benchmark import {fragment!r} in {relative_path}"
        for fragment in forbidden_fragments:
            assert fragment not in text, f"stale FastPauli benchmark surface {fragment!r} remains in {relative_path}"


def test_current_renderers_and_public_plot_artifacts_brand_wolfgang() -> None:
    forbidden_fragments = (
        "FastPauli accelerator performance landscape",
        "FastPauli H100 backend speedups",
        "FastPauli H100 CUDA backend speedups",
        "FastPauli H100 scaling",
        "FastPauli CUDA optimization deltas",
        "FastPauli H100 Nsight Compute bottlenecks",
        "FastPauli Execution And Hardware Architecture",
        "fastpauli-runtime.cdx.json",
    )
    required_fragments = {
        "scripts/render_apple_metal_assets.py": ("Wolfgang accelerator performance landscape",),
        "scripts/render_benchmark_plots.py": ("Wolfgang H100 CUDA backend speedups",),
        "scripts/render_cuda_deep_report_assets.py": ("Wolfgang H100 backend speedups", "Wolfgang Execution And Hardware Architecture"),
        "scripts/run_rocm_release_support_lane.py": ("# Wolfgang ROCm Campaign 7 Release-Support Lane",),
        "docs/benchmarks/plots/accelerator_landscape_with_rocm.svg": ("Wolfgang accelerator performance landscape",),
        "docs/benchmarks/plots/cuda_deep_optimization_architecture.svg": ("Wolfgang Execution And Hardware Architecture",),
        "docs/benchmarks/plots/cuda_deep_optimization_h100_optimization_deltas.svg": ("Wolfgang CUDA optimization deltas",),
        "docs/benchmarks/plots/cuda_deep_optimization_h100_path_speedups.svg": ("Wolfgang H100 backend speedups",),
        "docs/benchmarks/plots/cuda_deep_optimization_h100_profiler_bottlenecks.svg": ("Wolfgang H100 Nsight Compute bottlenecks",),
        "docs/benchmarks/plots/cuda_deep_optimization_h100_scaling.svg": ("Wolfgang H100 scaling",),
        "docs/benchmarks/plots/cuda_h100_nsight_hillclimb_default_backend_speedups.svg": ("Wolfgang H100 CUDA backend speedups",),
    }

    for relative_path in (*CURRENT_RENDERER_AND_REPORT_FILES, *CURRENT_PLOT_FILES):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        for fragment in required_fragments[relative_path]:
            assert fragment in text, f"missing canonical Wolfgang artifact branding {fragment!r} in {relative_path}"
        for fragment in forbidden_fragments:
            assert fragment not in text, f"stale FastPauli artifact branding {fragment!r} remains in {relative_path}"


def test_workflows_and_packaging_prefer_wolfgang_canonical_flags() -> None:
    quality = (ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8")
    codeql = (ROOT / ".github/workflows/codeql.yml").read_text(encoding="utf-8")
    release_wheelhouse = (ROOT / ".github/workflows/release-wheelhouse.yml").read_text(
        encoding="utf-8"
    )
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "pyright python/wolfgang_quantum" in quality
    for workflow_text in (quality, codeql, release_wheelhouse):
        assert "cmake.define.WOLFGANG_ENABLE_CUDA=OFF" in workflow_text or "-DWOLFGANG_ENABLE_CUDA=OFF" in workflow_text
        assert "cmake.define.WOLFGANG_ENABLE_HIP=OFF" in workflow_text or "-DWOLFGANG_ENABLE_HIP=OFF" in workflow_text
        assert "cmake.define.WOLFGANG_ENABLE_METAL=OFF" in workflow_text or "-DWOLFGANG_ENABLE_METAL=OFF" in workflow_text
        assert "cmake.define.WOLFGANG_ENABLE_NATIVE=OFF" in workflow_text or "-DWOLFGANG_ENABLE_NATIVE=OFF" in workflow_text

    assert "sbom/wolfgang-runtime.cdx.json" in release_wheelhouse
    assert "fastpauli-runtime.cdx.json" not in release_wheelhouse

    config_settings_section = pyproject.split("[tool.cibuildwheel.config-settings]", maxsplit=1)[1]
    config_settings_body = config_settings_section.split("[tool.cibuildwheel.linux]", maxsplit=1)[0]
    assert '"cmake.define.WOLFGANG_ENABLE_INTERNAL_BINDINGS" = "OFF"' in config_settings_body
    assert '"cmake.define.WOLFGANG_ENABLE_CUDA" = "OFF"' in config_settings_body
    assert '"cmake.define.WOLFGANG_ENABLE_HIP" = "OFF"' in config_settings_body
    assert '"cmake.define.WOLFGANG_ENABLE_METAL" = "OFF"' in config_settings_body
    assert '"cmake.define.WOLFGANG_ENABLE_NATIVE" = "OFF"' in config_settings_body
    assert '"cmake.define.FASTPAULI_ENABLE_CUDA"' not in config_settings_body
    assert '"cmake.define.FASTPAULI_ENABLE_HIP"' not in config_settings_body
    assert '"cmake.define.FASTPAULI_ENABLE_METAL"' not in config_settings_body
    assert '"cmake.define.FASTPAULI_ENABLE_NATIVE"' not in config_settings_body


def test_canonical_public_cmake_and_runtime_identity_prefers_wolfgang_names() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    cpu_backend = (ROOT / "src/cpu_backend.cpp").read_text(encoding="utf-8")
    package_init = (ROOT / "python/wolfgang_quantum/__init__.py").read_text(encoding="utf-8")
    internal_bindings = (ROOT / "bindings/python/internal_bindings.cpp").read_text(encoding="utf-8")

    assert 'WOLFGANG_ENABLE_CUDA' in cmake
    assert 'WOLFGANG_ENABLE_INTERNAL_BINDINGS' in cmake
    assert 'WOLFGANG_CUDA_ARCHITECTURES' in cmake
    assert 'WOLFGANG_ENABLE_CUDA' in cmake
    assert 'WOLFGANG_ENABLE_INTERNAL_BINDINGS' in cmake

    assert 'constexpr std::string_view kBackendEnvVar = "WOLFGANG_CPU_BACKEND";' in cpu_backend
    assert 'WOLFGANG_CPU_BACKEND' in cpu_backend
    assert 'WOLFGANG_CPU_BACKEND' in cpu_backend

    assert 'FastPauliCapabilities' not in package_init
    assert 'cpu_backend_env_var"] = "WOLFGANG_CPU_BACKEND"' in internal_bindings


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

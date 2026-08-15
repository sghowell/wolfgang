from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = "docs/plans/release_0_1_0_wheelhouse_foundation_plan.md"
WORKFLOW_PATH = ".github/workflows/release-wheelhouse.yml"
CHECKER_PATH = "scripts/check_release_wheelhouse.py"


def load_module(path: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_fake_wheel_smoke_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    build_info: dict[str, Any],
    version: str = "0.2.0",
) -> None:
    class FakePauliSum:
        def __init__(self, labels: list[str], coeffs: list[complex]) -> None:
            self._labels = labels
            self._coeffs = coeffs

        @classmethod
        def from_labels(cls, labels: list[str], coeffs: list[complex]) -> "FakePauliSum":
            return cls(labels, coeffs)

        def simplify(self) -> "FakePauliSum":
            assert self._labels == ["X", "X", "Z"]
            assert self._coeffs == [1.0, -0.5, 2.0]
            return FakePauliSum(["Z", "X"], [2.0 + 0.0j, 0.5 + 0.0j])

        def to_labels(self) -> tuple[list[str], list[complex]]:
            return self._labels, self._coeffs

    wolfgang_quantum = ModuleType("wolfgang_quantum")
    setattr(wolfgang_quantum, "__version__", version)
    setattr(wolfgang_quantum, "PauliSum", FakePauliSum)

    core = ModuleType("wolfgang_quantum._wolfgang_core")
    setattr(core, "_build_info", lambda: build_info)

    fastpauli = ModuleType("fastpauli")
    setattr(fastpauli, "__version__", version)

    monkeypatch.setitem(sys.modules, "wolfgang_quantum", wolfgang_quantum)
    monkeypatch.setitem(sys.modules, "wolfgang_quantum._wolfgang_core", core)
    monkeypatch.setitem(sys.modules, "fastpauli", fastpauli)


def test_release_wheelhouse_plan_is_registered_and_routed() -> None:
    validate = load_module("scripts/validate.py", "fastpauli_validate")

    assert (ROOT / PLAN_PATH).exists()
    assert PLAN_PATH in validate.SOURCE_OF_TRUTH_PATHS

    routed_docs = (
        "docs/research/provenance.md",
        "AGENTS.md",
        "docs/roadmap.md",
        "docs/release/README.md",
        "docs/quality/release_and_packaging.md",
        "docs/release/support_matrix.md",
    )
    for path in routed_docs:
        assert PLAN_PATH in (ROOT / path).read_text(encoding="utf-8")


def test_release_wheelhouse_plan_locks_public_boundary() -> None:
    plan = (ROOT / PLAN_PATH).read_text(encoding="utf-8")

    required_terms = (
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
    for term in required_terms:
        assert term in plan


def test_cibuildwheel_cpu_only_configuration_is_present() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "[tool.cibuildwheel]" in pyproject
    assert '"cibuildwheel>=2.23,<3; python_version < \'3.11\'"' in pyproject
    assert '"cibuildwheel>=3.0,<4; python_version >= \'3.11\'"' in pyproject
    assert '"trove-classifiers>=2026.4.28.13"' in pyproject
    assert 'build = ["cp310-*", "cp311-*", "cp312-*"]' in pyproject
    assert (
        'skip = ["*-win32", "*-win_amd64", "*-win_arm64", "*-musllinux*", "pp*"]'
        in pyproject
    )
    assert "cmake.define.WOLFGANG_ENABLE_CUDA" in pyproject
    assert "cmake.define.WOLFGANG_ENABLE_HIP" in pyproject
    assert "cmake.define.WOLFGANG_ENABLE_METAL" in pyproject
    assert "cmake.define.WOLFGANG_ENABLE_NATIVE" in pyproject
    assert 'test-command = "python {project}/scripts/wheel_smoke.py"' in pyproject
    assert "[tool.cibuildwheel.linux]" in pyproject
    assert 'archs = ["x86_64"]' in pyproject
    assert "[tool.cibuildwheel.macos]" in pyproject
    assert 'archs = ["arm64"]' in pyproject
    assert '"Programming Language :: C++ :: 20"' not in pyproject


def test_release_wheelhouse_workflow_builds_checksums_and_gates_publication() -> None:
    workflow = (ROOT / WORKFLOW_PATH).read_text(encoding="utf-8")

    required_terms = (
        "workflow_dispatch:",
        "publish-target:",
        "none",
        "testpypi",
        "pypi",
        "pypa/cibuildwheel",
        "python -m build --sdist",
        "python -m twine check",
        "python -m twine check publish-dist/*",
        "Set up Python for metadata checks",
        "python scripts/check_release_wheelhouse.py --require-trove-classifiers",
        "scripts/write_release_checksums.py",
        "scripts/prepare_publish_dist.py",
        "--require-cpu-wheelhouse",
        "actions/upload-artifact",
        "pypa/gh-action-pypi-publish@",
        "repository-url: https://test.pypi.org/legacy/",
        "packages-dir: publish-dist",
        "GITHUB_REF_TYPE",
        "GITHUB_REF_NAME",
        "id-token: write",
    )
    for term in required_terms:
        assert term in workflow

    assert "packages-dir: dist" not in workflow
    assert re.search(
        r"collect-wheelhouse:[\s\S]+?Set up Python for metadata checks"
        r"[\s\S]+?uses: actions/setup-python@[0-9a-f]{40}"
        r"[\s\S]+?python-version: \"3\.12\""
        r"[\s\S]+?python -m twine check dist/\*",
        workflow,
    )
    assert 'python scripts/check_release_tag.py "${GITHUB_REF_TYPE}" "${GITHUB_REF_NAME}"' in workflow
    assert "startsWith(github.ref_name, 'v')" not in workflow


def test_release_wheelhouse_checker_passes_current_config() -> None:
    checker = load_module(CHECKER_PATH, "fastpauli_release_wheelhouse")

    assert checker.check_release_wheelhouse() == []


def test_release_wheelhouse_checker_has_python310_pyproject_fallback() -> None:
    checker = load_module(CHECKER_PATH, "fastpauli_release_wheelhouse_fallback")
    failures: list[str] = []

    checker.check_pyproject_text(failures)

    assert failures == []


def test_release_wheelhouse_checker_rejects_invalid_trove_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = load_module(CHECKER_PATH, "fastpauli_release_wheelhouse_classifier")
    original_read_text = checker.read_text

    def read_text_with_invalid_classifier(path: str) -> str:
        text = original_read_text(path)
        if path == "pyproject.toml":
            return text.replace(
                '"Programming Language :: C++",',
                '"Programming Language :: C++",\n  "Programming Language :: C++ :: 20",',
            )
        return text

    monkeypatch.setattr(checker, "read_text", read_text_with_invalid_classifier)
    failures: list[str] = []

    checker.check_pyproject(failures)

    assert any("invalid Trove classifier" in failure for failure in failures)


def test_release_wheelhouse_checker_can_require_canonical_trove_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = load_module(CHECKER_PATH, "fastpauli_release_wheelhouse_trove")
    trove_module = ModuleType("trove_classifiers")
    trove_module.classifiers = {
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    }
    trove_module.deprecated_classifiers = {}
    monkeypatch.setitem(sys.modules, "trove_classifiers", trove_module)
    failures: list[str] = []

    checker.check_pyproject(failures, require_trove_classifiers=True)

    assert failures == []


def test_release_wheelhouse_checker_rejects_unknown_trove_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = load_module(CHECKER_PATH, "fastpauli_release_wheelhouse_unknown_trove")
    trove_module = ModuleType("trove_classifiers")
    trove_module.classifiers = {"Programming Language :: C++"}
    trove_module.deprecated_classifiers = {}
    monkeypatch.setitem(sys.modules, "trove_classifiers", trove_module)
    failures: list[str] = []

    checker.check_project_classifiers(
        ["Programming Language :: C++", "Fuzzy :: Wuzzy :: Was :: A :: Bear"],
        failures,
        require_trove_classifiers=True,
    )

    assert any("Fuzzy :: Wuzzy :: Was :: A :: Bear" in failure for failure in failures)


def test_wheel_smoke_script_reports_current_install_metadata_without_cpu_only_assumption() -> None:
    wheel_smoke = load_module("scripts/wheel_smoke.py", "fastpauli_wheel_smoke")

    summary = wheel_smoke.smoke_wolfgang_wheel(require_cpu_safe=False)

    assert summary["accelerator_build_mode"] in {"cpu_only", "cuda_only", "hip_only", "metal_only"}
    assert summary["project_distribution"] == "wolfgang-quantum"
    assert summary["package_import"] == "wolfgang_quantum"
    assert summary["fastpauli_compat_version"] == summary["wolfgang_version"]


def test_wheel_smoke_script_preserves_cpu_safe_contract_with_controlled_cpu_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel_smoke = load_module("scripts/wheel_smoke.py", "fastpauli_wheel_smoke_cpu_safe")
    cpu_build_info = {
        "accelerator_build_mode": "cpu_only",
        "compiled_backends": ["cpu"],
        "compiled_cpu_backends": ["scalar"],
        "cuda_enabled": False,
        "hip_enabled": False,
        "metal_enabled": False,
        "native_enabled": False,
    }
    install_fake_wheel_smoke_modules(monkeypatch, build_info=cpu_build_info)
    monkeypatch.setattr(wheel_smoke.importlib_metadata, "version", lambda _dist: "0.2.0")

    summary = wheel_smoke.smoke_wolfgang_wheel()

    assert summary["accelerator_build_mode"] == "cpu_only"
    assert summary["cuda_enabled"] is False
    assert summary["hip_enabled"] is False
    assert summary["metal_enabled"] is False
    assert summary["native_enabled"] is False
    assert summary["compiled_backends"] == ["cpu"]
    assert "scalar" in summary["compiled_cpu_backends"]


def test_wheel_smoke_script_rejects_accelerator_builds_when_cpu_safe_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel_smoke = load_module("scripts/wheel_smoke.py", "fastpauli_wheel_smoke_cpu_guard")
    install_fake_wheel_smoke_modules(
        monkeypatch,
        build_info={
            "accelerator_build_mode": "hip_only",
            "compiled_backends": ["cpu", "hip"],
            "compiled_cpu_backends": ["scalar"],
            "cuda_enabled": False,
            "hip_enabled": True,
            "metal_enabled": False,
            "native_enabled": False,
        },
    )
    monkeypatch.setattr(wheel_smoke.importlib_metadata, "version", lambda _dist: "0.2.0")

    with pytest.raises(AssertionError, match="hip_only"):
        wheel_smoke.smoke_wolfgang_wheel()


def test_release_checksum_writer_outputs_sorted_manifest(tmp_path: Path) -> None:
    checksum_writer = load_module(
        "scripts/write_release_checksums.py",
        "fastpauli_release_checksums",
    )
    (tmp_path / "wolfgang_quantum-0.1.0-cp310-cp310-manylinux_2_28_x86_64.whl").write_bytes(
        b"wheel"
    )
    (tmp_path / "wolfgang-quantum-0.1.0.tar.gz").write_bytes(b"sdist")

    manifest = checksum_writer.write_checksum_manifest(tmp_path)

    assert manifest.name == "wolfgang-quantum-0.1.0.checksums.txt"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert lines == sorted(lines, key=lambda line: line.split("  ", 1)[1])
    assert all(re.match(r"^[0-9a-f]{64}  wolfgang(?:_quantum|\-quantum)-0\.1\.0", line) for line in lines)


def test_release_checksum_writer_accepts_normalized_pep625_sdist_name(tmp_path: Path) -> None:
    checksum_writer = load_module(
        "scripts/write_release_checksums.py",
        "fastpauli_release_checksums_normalized_sdist",
    )
    (tmp_path / "wolfgang_quantum-0.2.0.tar.gz").write_bytes(b"sdist")
    (tmp_path / "wolfgang_quantum-0.2.0-cp312-cp312-macosx_26_0_arm64.whl").write_bytes(
        b"wheel"
    )

    manifest = checksum_writer.write_checksum_manifest(tmp_path)

    assert manifest.name == "wolfgang-quantum-0.2.0.checksums.txt"
    assert manifest.read_text(encoding="utf-8").splitlines() == [
        f"{checksum_writer.sha256_file(tmp_path / 'wolfgang_quantum-0.2.0-cp312-cp312-macosx_26_0_arm64.whl')}  wolfgang_quantum-0.2.0-cp312-cp312-macosx_26_0_arm64.whl",
        f"{checksum_writer.sha256_file(tmp_path / 'wolfgang_quantum-0.2.0.tar.gz')}  wolfgang_quantum-0.2.0.tar.gz",
    ]


def test_release_checksum_writer_can_require_complete_cpu_wheelhouse(
    tmp_path: Path,
) -> None:
    checksum_writer = load_module(
        "scripts/write_release_checksums.py",
        "fastpauli_release_checksums_complete",
    )
    (tmp_path / "wolfgang-quantum-0.1.0.tar.gz").write_bytes(b"sdist")
    for python_tag in ("cp310", "cp311", "cp312"):
        linux_wheel = (
            tmp_path
            / f"wolfgang_quantum-0.1.0-{python_tag}-{python_tag}-manylinux_2_28_x86_64.whl"
        )
        macos_wheel = (
            tmp_path
            / f"wolfgang_quantum-0.1.0-{python_tag}-{python_tag}-macosx_14_0_arm64.whl"
        )
        linux_wheel.write_bytes(b"linux-wheel")
        macos_wheel.write_bytes(b"macos-wheel")

    manifest = checksum_writer.write_checksum_manifest(
        tmp_path,
        require_cpu_wheelhouse=True,
    )

    assert manifest.name == "wolfgang-quantum-0.1.0.checksums.txt"
    manifest.unlink()
    (tmp_path / "wolfgang_quantum-0.1.0-cp312-cp312-macosx_14_0_arm64.whl").unlink()
    with pytest.raises(SystemExit, match="missing CPU wheel targets"):
        checksum_writer.write_checksum_manifest(tmp_path, require_cpu_wheelhouse=True)


def test_prepare_publish_dist_excludes_checksum_manifest(tmp_path: Path) -> None:
    prepare_publish = load_module(
        "scripts/prepare_publish_dist.py",
        "fastpauli_prepare_publish_dist",
    )
    dist_dir = tmp_path / "dist"
    publish_dir = tmp_path / "publish-dist"
    dist_dir.mkdir()
    (dist_dir / "wolfgang-quantum-0.1.0.tar.gz").write_bytes(b"sdist")
    (dist_dir / "wolfgang_quantum-0.1.0-cp310-cp310-manylinux_2_28_x86_64.whl").write_bytes(
        b"wheel"
    )
    (dist_dir / "wolfgang-quantum-0.1.0.checksums.txt").write_text(
        "checksum  artifact\n",
        encoding="utf-8",
    )

    copied = prepare_publish.prepare_publish_dist(dist_dir, publish_dir)

    assert [path.name for path in copied] == [
        "wolfgang-quantum-0.1.0.tar.gz",
        "wolfgang_quantum-0.1.0-cp310-cp310-manylinux_2_28_x86_64.whl",
    ]
    assert sorted(path.name for path in publish_dir.iterdir()) == [
        "wolfgang-quantum-0.1.0.tar.gz",
        "wolfgang_quantum-0.1.0-cp310-cp310-manylinux_2_28_x86_64.whl",
    ]

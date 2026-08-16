from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

RELEASE_VERSION = "0.2.2"
SDIST_PREFIX = f"wolfgang-quantum-{RELEASE_VERSION}"
WHEEL_PREFIX = f"wolfgang_quantum-{RELEASE_VERSION}"


def load_release_validator() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_release_artifacts.py"
    spec = importlib.util.spec_from_file_location("fastpauli_release_validator", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_artifact_command_pins_cpu_safe_config_settings(tmp_path: Path) -> None:
    validator = load_release_validator()

    command = validator.build_artifact_command(tmp_path, python_executable="/venv/bin/python")
    command_text = " ".join(command)

    assert command[:6] == [
        "/venv/bin/python",
        "-m",
        "build",
        "--sdist",
        "--wheel",
        "--outdir",
    ]
    assert str(tmp_path) in command
    assert "cmake.define.WOLFGANG_ENABLE_CUDA=OFF" in command_text
    assert "cmake.define.WOLFGANG_ENABLE_HIP=OFF" in command_text
    assert "cmake.define.WOLFGANG_ENABLE_METAL=OFF" in command_text
    assert "cmake.define.WOLFGANG_ENABLE_NATIVE=OFF" in command_text


def test_cpu_safe_build_env_disables_accelerators_and_native(monkeypatch) -> None:
    validator = load_release_validator()
    monkeypatch.setenv("WOLFGANG_ENABLE_CUDA", "ON")
    monkeypatch.setenv("WOLFGANG_ENABLE_HIP", "ON")
    monkeypatch.setenv("WOLFGANG_ENABLE_METAL", "ON")
    monkeypatch.setenv("WOLFGANG_ENABLE_NATIVE", "ON")
    monkeypatch.setenv("WOLFGANG_ENABLE_CUDA", "ON")
    monkeypatch.setenv("WOLFGANG_ENABLE_HIP", "ON")
    monkeypatch.setenv("WOLFGANG_ENABLE_METAL", "ON")
    monkeypatch.setenv("WOLFGANG_ENABLE_NATIVE", "ON")
    monkeypatch.setattr(validator, "cmake_executable_for_build", lambda: "/opt/cmake/bin/cmake")

    env = validator.cpu_safe_build_env()

    assert env["WOLFGANG_ENABLE_CUDA"] == "OFF"
    assert env["WOLFGANG_ENABLE_HIP"] == "OFF"
    assert env["WOLFGANG_ENABLE_METAL"] == "OFF"
    assert env["WOLFGANG_ENABLE_NATIVE"] == "OFF"
    assert env["WOLFGANG_ENABLE_CUDA"] == "OFF"
    assert env["WOLFGANG_ENABLE_HIP"] == "OFF"
    assert env["WOLFGANG_ENABLE_METAL"] == "OFF"
    assert env["WOLFGANG_ENABLE_NATIVE"] == "OFF"
    assert env["CMAKE_EXECUTABLE"] == "/opt/cmake/bin/cmake"


def test_wheel_smoke_script_checks_release_metadata() -> None:
    validator = load_release_validator()
    script = validator.wheel_smoke_script(expected_version=RELEASE_VERSION)

    assert "accelerator_build_mode'] == 'cpu_only'" in script
    assert "cuda_enabled'] is False" in script
    assert "hip_enabled'] is False" in script
    assert "metal_enabled'] is False" in script
    assert "native_enabled'] is False" in script
    assert "compiled_backends'] == ['cpu']" in script
    assert "compiled_cpu_backends" in script
    assert "PauliSum.from_labels" in script
    assert f"__version__ == '{RELEASE_VERSION}'" in script
    assert f"importlib_metadata.version('wolfgang-quantum') == '{RELEASE_VERSION}'" in script
    assert "find_spec('fastpauli') is None" in script


def test_project_version_reads_pyproject_version() -> None:
    validator = load_release_validator()

    assert validator.project_version() == RELEASE_VERSION


def test_build_artifacts_requires_exactly_one_wheel_and_sdist(tmp_path: Path) -> None:
    validator = load_release_validator()
    (tmp_path / f"{SDIST_PREFIX}.tar.gz").write_text("sdist", encoding="utf-8")
    (tmp_path / f"{WHEEL_PREFIX}-cp312-cp312-macosx_15_0_arm64.whl").write_text(
        "wheel",
        encoding="utf-8",
    )

    assert validator.find_single_artifact(tmp_path, ".tar.gz").name == f"{SDIST_PREFIX}.tar.gz"
    assert (
        validator.find_single_artifact(tmp_path, ".whl").name
        == f"{WHEEL_PREFIX}-cp312-cp312-macosx_15_0_arm64.whl"
    )

    (tmp_path / f"{WHEEL_PREFIX}-cp310-cp310-linux_x86_64.whl").write_text(
        "wheel",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="expected exactly one Wolfgang .whl artifact"):
        validator.find_single_artifact(tmp_path, ".whl")


def test_validate_artifact_versions_requires_current_artifact_names(tmp_path: Path) -> None:
    validator = load_release_validator()
    current_sdist = tmp_path / f"{SDIST_PREFIX}.tar.gz"
    current_wheel = tmp_path / f"{WHEEL_PREFIX}-cp312-cp312-macosx_26_0_arm64.whl"

    validator.validate_artifact_versions(current_sdist, current_wheel, RELEASE_VERSION)

    normalized_sdist = tmp_path / f"{WHEEL_PREFIX}.tar.gz"
    validator.validate_artifact_versions(normalized_sdist, current_wheel, RELEASE_VERSION)

    final_sdist = tmp_path / "fastpauli-0.1.0rc2.tar.gz"
    with pytest.raises(SystemExit, match="expected source distribution named"):
        validator.validate_artifact_versions(final_sdist, current_wheel, RELEASE_VERSION)

    final_wheel = tmp_path / "fastpauli-0.1.0rc2-cp312-cp312-macosx_26_0_arm64.whl"
    with pytest.raises(SystemExit, match="expected wheel basename"):
        validator.validate_artifact_versions(current_sdist, final_wheel, RELEASE_VERSION)


def test_sha256_file_reports_artifact_digest(tmp_path: Path) -> None:
    validator = load_release_validator()
    artifact = tmp_path / f"{SDIST_PREFIX}.tar.gz"
    artifact.write_bytes(b"release-artifact")

    assert (
        validator.sha256_file(artifact)
        == "55d98606526de0f88b30c309717deef32c0e061e1319fd1f20f866b49a226174"
    )


def test_release_summary_includes_artifact_hashes(monkeypatch, tmp_path: Path) -> None:
    validator = load_release_validator()
    sdist = tmp_path / f"{SDIST_PREFIX}.tar.gz"
    wheel = tmp_path / f"{WHEEL_PREFIX}-cp312-cp312-macosx_26_0_arm64.whl"
    sdist.write_bytes(b"sdist")
    wheel.write_bytes(b"wheel")

    monkeypatch.setattr(validator, "build_artifacts", lambda *_args, **_kwargs: (sdist, wheel))
    monkeypatch.setattr(
        validator,
        "install_and_smoke_wheel",
        lambda *_args, **_kwargs: tmp_path / "wheel-smoke-venv" / "bin" / "python",
    )

    summary = validator.validate_release_artifacts(
        tmp_path,
        python_executable="/venv/bin/python",
    )

    assert summary["sdist_sha256"] == validator.sha256_file(sdist)
    assert summary["wheel_sha256"] == validator.sha256_file(wheel)
    assert summary["sdist_name"] == sdist.name
    assert summary["wheel_name"] == wheel.name
    assert summary["expected_version"] == RELEASE_VERSION


def test_install_and_smoke_uses_clean_virtual_environment(monkeypatch, tmp_path: Path) -> None:
    validator = load_release_validator()
    calls: list[tuple[str, list[str]]] = []
    wheel = tmp_path / f"{WHEEL_PREFIX}-cp312-cp312-macosx_15_0_arm64.whl"
    wheel.write_text("wheel", encoding="utf-8")

    def capture_run_command(name: str, command: list[str], **kwargs: object) -> None:
        assert kwargs == {}
        calls.append((name, command))

    monkeypatch.setattr(validator, "run_command", capture_run_command)

    venv_python = validator.install_and_smoke_wheel(
        tmp_path,
        wheel,
        python_executable="/venv/bin/python",
    )

    assert venv_python == validator.venv_python_path(tmp_path / "wheel-smoke-venv")
    assert [name for name, _ in calls] == [
        "create clean wheel smoke virtual environment",
        "upgrade wheel smoke installer",
        "install produced Wolfgang wheel",
        "import produced Wolfgang wheel and verify CPU-safe metadata",
    ]
    assert calls[0][1] == [
        "/venv/bin/python",
        "-m",
        "venv",
        str(tmp_path / "wheel-smoke-venv"),
    ]
    assert str(wheel) in calls[2][1]
    assert "-c" in calls[3][1]

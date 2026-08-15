from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def load_validate_module() -> ModuleType:
    validate_path = Path(__file__).resolve().parents[1] / "scripts" / "validate.py"
    spec = importlib.util.spec_from_file_location("fastpauli_validate", validate_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cmake_configure_check_prefers_build_isolation_cmake(monkeypatch) -> None:
    validate = load_validate_module()
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(validate, "cmake_executable_for_build_isolation", lambda: "/opt/cmake/bin/cmake")
    monkeypatch.setattr(validate, "find_executable", lambda name: "/usr/bin/cmake")
    monkeypatch.setattr(validate.sys, "executable", "/usr/bin/python")

    def capture_run_check(name: str, command: list[str], **kwargs: object) -> None:
        assert not kwargs
        calls.append((name, command))

    monkeypatch.setattr(validate, "run_check", capture_run_check)

    validate.run_cmake_configure_check()

    assert len(calls) == 1
    _, command = calls[0]
    assert command[0] == "/opt/cmake/bin/cmake"
    assert "-DPython_EXECUTABLE=/usr/bin/python" in command


def test_cmake_configure_check_pins_the_active_python(monkeypatch) -> None:
    validate = load_validate_module()
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(validate, "cmake_executable_for_build_isolation", lambda: "/venv/bin/cmake")
    monkeypatch.setattr(validate.sys, "executable", "/venv/bin/python")

    def capture_run_check(name: str, command: list[str], **kwargs: object) -> None:
        assert not kwargs
        calls.append((name, command))

    monkeypatch.setattr(validate, "run_check", capture_run_check)

    validate.run_cmake_configure_check()

    assert len(calls) == 1
    _, command = calls[0]
    assert command[0] == "/venv/bin/cmake"
    assert "-DPython_EXECUTABLE=/venv/bin/python" in command
    assert "-DWOLFGANG_ENABLE_METAL=OFF" in command


def test_native_source_layout_check_rejects_top_level_backend_sources(
    monkeypatch,
    tmp_path: Path,
) -> None:
    validate = load_validate_module()
    monkeypatch.setattr(validate, "ROOT", tmp_path)

    required_paths = (
        *validate.DETAIL_HEADERS,
        *validate.ROOT_NATIVE_SOURCES,
        *validate.BACKEND_SPECIALIZED_SOURCES,
        *validate.CUDA_FOUNDATION_SOURCES,
        *validate.HIP_FOUNDATION_SOURCES,
        *validate.METAL_FOUNDATION_SOURCES,
    )
    for relative_path in required_paths:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// test fixture\n", encoding="utf-8")

    cmake_paths = "\n".join(
        validate.ROOT_NATIVE_SOURCES
        + validate.BACKEND_SPECIALIZED_SOURCES
        + tuple(path for path in validate.CUDA_FOUNDATION_SOURCES if path.endswith((".cpp", ".cu")))
        + tuple(path for path in validate.HIP_FOUNDATION_SOURCES if path.endswith(".hip.cpp"))
        + tuple(path for path in validate.METAL_FOUNDATION_SOURCES if path.endswith(".mm"))
    )
    (tmp_path / "CMakeLists.txt").write_text(cmake_paths, encoding="utf-8")

    validate.check_native_source_layout()

    misplaced = tmp_path / "src" / "commute_kernels_avx2.cpp"
    misplaced.write_text("// misplaced backend fixture\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="Backend-specialized native sources"):
        validate.check_native_source_layout()

    misplaced.unlink()
    generic_misplaced = tmp_path / "src" / "expectation_kernels_avx2.cpp"
    generic_misplaced.write_text("// misplaced backend fixture\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="expectation_kernels_avx2.cpp"):
        validate.check_native_source_layout()


def test_cuda_validation_path_is_explicit(monkeypatch) -> None:
    validate = load_validate_module()
    calls: list[tuple[str, list[str], dict[str, object]]] = []

    monkeypatch.delenv("WOLFGANG_CUDA_ARCHITECTURES", raising=False)
    monkeypatch.delenv("CMAKE_CUDA_HOST_COMPILER", raising=False)
    monkeypatch.delenv("CUDACXX", raising=False)
    monkeypatch.delenv("CUDAHOSTCXX", raising=False)
    monkeypatch.delenv("CXX", raising=False)

    def fake_find_executable(name: str) -> str | None:
        mapping = {
            "nvcc": "/usr/bin/nvcc",
            "cmake": "/usr/bin/cmake",
            "g++": "/usr/bin/g++",
            "c++": "/usr/bin/c++",
            "nvidia-smi": "/usr/bin/nvidia-smi",
        }
        return mapping.get(name)

    monkeypatch.setattr(validate, "find_executable", fake_find_executable)
    monkeypatch.setattr(validate, "ensure_pip_install_config_settings_support", lambda: None)
    monkeypatch.setattr(validate.sys, "executable", "/venv/bin/python")

    def capture_run_check(name: str, command: list[str], **kwargs: object) -> None:
        calls.append((name, command, kwargs))

    monkeypatch.setattr(validate, "run_check", capture_run_check)
    monkeypatch.setattr(validate, "print_check", lambda name: None)

    validate.run_cuda_validation_checks()

    command_text = "\n".join(" ".join(command) for _, command, _ in calls)
    assert "WOLFGANG_ENABLE_CUDA=ON" in command_text
    assert "WOLFGANG_CUDA_ARCHITECTURES" not in command_text
    assert "CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++" in command_text
    assert any("CUDA toolkit version from nvcc" in name for name, _, _ in calls)
    assert any("CUDA host compiler version" in name for name, _, _ in calls)
    assert any("CUDA source build with default architectures" in name for name, _, _ in calls)
    assert any("CUDA-enabled semantic pytest" in name for name, _, _ in calls)

    source_build = next(
        kwargs
        for name, _, kwargs in calls
        if name == "CUDA source build with default architectures"
    )
    env = source_build["env"]
    assert isinstance(env, dict)
    assert env["CUDACXX"] == "/usr/bin/nvcc"
    assert env["CUDAHOSTCXX"] == "/usr/bin/g++"


def test_cuda_validation_env_does_not_mutate_include_search_path() -> None:
    validate = load_validate_module()

    env = validate.cuda_validation_env(
        nvcc="/usr/local/cuda-12.8/bin/nvcc",
        host_compiler="/usr/bin/g++",
    )

    assert env["CUDACXX"] == "/usr/local/cuda-12.8/bin/nvcc"
    assert env["CUDAHOSTCXX"] == "/usr/bin/g++"
    assert env["CUDA_PATH"] == "/usr/local/cuda-12.8"
    assert "CPLUS_INCLUDE_PATH" not in env


def test_cuda_validation_installs_cupy_with_ctk_headers(monkeypatch) -> None:
    validate = load_validate_module()
    calls: list[tuple[str, list[str], dict[str, object]]] = []

    monkeypatch.delenv("WOLFGANG_CUDA_ARCHITECTURES", raising=False)
    monkeypatch.delenv("CMAKE_CUDA_HOST_COMPILER", raising=False)
    monkeypatch.delenv("CUDACXX", raising=False)
    monkeypatch.delenv("CUDAHOSTCXX", raising=False)
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.setattr(validate, "find_executable", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(validate.sys, "executable", "/venv/bin/python")

    def capture_run_check(name: str, command: list[str], **kwargs: object) -> None:
        calls.append((name, command, kwargs))

    monkeypatch.setattr(validate, "run_check", capture_run_check)
    monkeypatch.setattr(validate, "print_check", lambda name: None)

    validate.run_cuda_validation_checks()

    cupy_install = next(command for name, command, _ in calls if name == "install CuPy CUDA toolkit headers")
    assert "cupy-cuda12x[ctk]" in cupy_install


def test_cuda_validation_semantic_pytest_ignores_cpu_only_release_checks(monkeypatch) -> None:
    validate = load_validate_module()
    calls: list[tuple[str, list[str], dict[str, object]]] = []

    monkeypatch.delenv("WOLFGANG_CUDA_ARCHITECTURES", raising=False)
    monkeypatch.delenv("CMAKE_CUDA_HOST_COMPILER", raising=False)
    monkeypatch.delenv("CUDACXX", raising=False)
    monkeypatch.delenv("CUDAHOSTCXX", raising=False)
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.setattr(validate, "find_executable", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(validate.sys, "executable", "/venv/bin/python")

    def capture_run_check(name: str, command: list[str], **kwargs: object) -> None:
        calls.append((name, command, kwargs))

    monkeypatch.setattr(validate, "run_check", capture_run_check)
    monkeypatch.setattr(validate, "print_check", lambda name: None)

    validate.run_cuda_validation_checks()

    semantic_pytest = next(command for name, command, _ in calls if name == "CUDA-enabled semantic pytest")
    assert "--ignore" in semantic_pytest
    assert "tests/test_release_wheelhouse_foundation.py" in semantic_pytest
    assert "tests/test_release_artifact_validation.py" in semantic_pytest
    assert "tests/test_apple_metal_foundation.py" in semantic_pytest


def test_cuda_validation_requested_architectures_are_authoritative(monkeypatch) -> None:
    validate = load_validate_module()
    calls: list[tuple[str, list[str], dict[str, object]]] = []
    build_info_architectures: list[str] = []

    monkeypatch.setenv("WOLFGANG_CUDA_ARCHITECTURES", "100-real;120")
    monkeypatch.setenv("CUDAHOSTCXX", "/usr/bin/g++")
    monkeypatch.setattr(validate, "find_executable", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(validate.sys, "executable", "/venv/bin/python")

    def capture_run_check(name: str, command: list[str], **kwargs: object) -> None:
        calls.append((name, command, kwargs))

    def capture_build_info(expected_architectures: str) -> None:
        build_info_architectures.append(expected_architectures)

    monkeypatch.setattr(validate, "run_check", capture_run_check)
    monkeypatch.setattr(validate, "run_cuda_build_info_check", capture_build_info)
    monkeypatch.setattr(validate, "print_check", lambda name: None)

    validate.run_cuda_validation_checks()

    build_calls = [
        (name, command, kwargs)
        for name, command, kwargs in calls
        if "CUDA source build" in name
    ]
    assert [name for name, _, _ in build_calls] == [
        "CUDA source build with requested architecture override",
    ]
    requested_command = " ".join(build_calls[0][1])
    assert "WOLFGANG_CUDA_ARCHITECTURES=100-real;120" in requested_command
    assert "CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++" in requested_command
    assert build_info_architectures == ["100-real,120"]


def test_cuda_source_build_upgrades_python_pip_when_config_settings_are_missing(
    monkeypatch,
) -> None:
    validate = load_validate_module()
    calls: list[tuple[str, list[str], dict[str, object]]] = []
    support_checks = iter((False, True))

    monkeypatch.setattr(validate, "find_executable", lambda name: None)
    monkeypatch.setattr(
        validate,
        "pip_install_supports_config_settings",
        lambda: next(support_checks),
    )

    def capture_run_check(name: str, command: list[str], **kwargs: object) -> None:
        calls.append((name, command, kwargs))

    monkeypatch.setattr(validate, "run_check", capture_run_check)

    validate.run_cuda_source_build("80", nvcc="/usr/bin/nvcc", host_compiler="/usr/bin/c++")

    assert [name for name, _, _ in calls] == [
        "upgrade pip for PEP 517 config settings",
        "CUDA source build with requested architecture override",
    ]
    assert calls[0][1] == [
        validate.sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip",
        "setuptools",
        "wheel",
    ]
    source_build_command = " ".join(calls[1][1])
    assert "--config-settings=cmake.define.WOLFGANG_ENABLE_CUDA=ON" in source_build_command
    assert "WOLFGANG_CUDA_ARCHITECTURES=80" in source_build_command


def test_cuda_source_build_pins_cudatoolkit_root_from_nvcc_path(monkeypatch) -> None:
    validate = load_validate_module()
    calls: list[tuple[str, list[str], dict[str, object]]] = []

    monkeypatch.setattr(validate, "find_executable", lambda name: "/usr/bin/uv" if name == "uv" else None)

    def capture_run_check(name: str, command: list[str], **kwargs: object) -> None:
        calls.append((name, command, kwargs))

    monkeypatch.setattr(validate, "run_check", capture_run_check)

    validate.run_cuda_source_build(
        "90",
        nvcc="/usr/local/cuda-12.8/bin/nvcc",
        host_compiler="/usr/bin/g++",
    )

    assert len(calls) == 1
    source_build_command = " ".join(calls[0][1])
    assert "--config-settings=cmake.define.CUDAToolkit_ROOT=/usr/local/cuda-12.8" in source_build_command


def test_hip_validation_path_is_explicit(monkeypatch) -> None:
    validate = load_validate_module()
    calls: list[tuple[str, list[str], dict[str, object]]] = []

    monkeypatch.delenv("WOLFGANG_HIP_ARCHITECTURES", raising=False)
    monkeypatch.setattr(validate, "find_executable", lambda name: f"/opt/rocm/bin/{name}")
    monkeypatch.setattr(validate.sys, "executable", "/venv/bin/python")

    def capture_run_check(name: str, command: list[str], **kwargs: object) -> None:
        calls.append((name, command, kwargs))

    monkeypatch.setattr(validate, "run_check", capture_run_check)
    monkeypatch.setattr(validate, "print_check", lambda name: None)
    monkeypatch.setattr(validate, "ensure_pip_install_config_settings_support", lambda: None)

    validate.run_hip_validation_checks()

    command_text = "\n".join(" ".join(command) for _, command, _ in calls)
    assert "WOLFGANG_ENABLE_HIP=ON" in command_text
    assert "WOLFGANG_HIP_ARCHITECTURES=gfx942" in command_text
    assert any("HIP compiler version from hipcc" in name for name, _, _ in calls)
    assert any("HIP source build for gfx942" in name for name, _, _ in calls)
    assert any("HIP-enabled semantic pytest" in name for name, _, _ in calls)

    source_build = next(kwargs for name, _, kwargs in calls if name == "HIP source build for gfx942")
    env = source_build["env"]
    assert isinstance(env, dict)
    assert env["PATH"].startswith("/opt/rocm/bin:/opt/rocm/llvm/bin:")


def test_hip_validation_can_add_requested_architecture_lane(monkeypatch) -> None:
    validate = load_validate_module()
    calls: list[tuple[str, list[str], dict[str, object]]] = []

    monkeypatch.setenv("WOLFGANG_HIP_ARCHITECTURES", "gfx950")
    monkeypatch.setattr(validate, "find_executable", lambda name: f"/opt/rocm/bin/{name}")
    monkeypatch.setattr(validate.sys, "executable", "/venv/bin/python")

    def capture_run_check(name: str, command: list[str], **kwargs: object) -> None:
        calls.append((name, command, kwargs))

    monkeypatch.setattr(validate, "run_check", capture_run_check)
    monkeypatch.setattr(validate, "print_check", lambda name: None)
    monkeypatch.setattr(validate, "ensure_pip_install_config_settings_support", lambda: None)

    validate.run_hip_validation_checks()

    build_calls = [(name, command) for name, command, _ in calls if "HIP source build" in name]
    assert [name for name, _ in build_calls] == [
        "HIP source build for gfx942",
        "HIP source build for requested architecture override",
    ]
    assert "WOLFGANG_HIP_ARCHITECTURES=gfx942" in " ".join(build_calls[0][1])
    assert "WOLFGANG_HIP_ARCHITECTURES=gfx950" in " ".join(build_calls[1][1])


def test_hip_validation_normalizes_requested_architecture_lists(monkeypatch) -> None:
    validate = load_validate_module()
    calls: list[tuple[str, list[str], dict[str, object]]] = []
    build_info_architectures: list[str] = []

    monkeypatch.setenv("WOLFGANG_HIP_ARCHITECTURES", "gfx942;gfx950")
    monkeypatch.setattr(validate, "find_executable", lambda name: f"/opt/rocm/bin/{name}")
    monkeypatch.setattr(validate.sys, "executable", "/venv/bin/python")

    def capture_run_check(name: str, command: list[str], **kwargs: object) -> None:
        calls.append((name, command, kwargs))

    def capture_build_info(expected_architectures: str) -> None:
        build_info_architectures.append(expected_architectures)

    monkeypatch.setattr(validate, "run_check", capture_run_check)
    monkeypatch.setattr(validate, "print_check", lambda name: None)
    monkeypatch.setattr(validate, "ensure_pip_install_config_settings_support", lambda: None)
    monkeypatch.setattr(validate, "run_hip_build_info_check", capture_build_info)

    validate.run_hip_validation_checks()

    build_calls = [(name, command) for name, command, _ in calls if "HIP source build" in name]
    assert "WOLFGANG_HIP_ARCHITECTURES=gfx942;gfx950" in " ".join(build_calls[1][1])
    assert build_info_architectures == ["gfx942", "gfx942,gfx950"]


def test_metal_validation_path_is_explicit(monkeypatch) -> None:
    validate = load_validate_module()
    calls: list[tuple[str, list[str], dict[str, object]]] = []

    monkeypatch.setattr(validate.sys, "platform", "darwin")
    monkeypatch.setattr(validate, "find_executable", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(validate.sys, "executable", "/venv/bin/python")
    monkeypatch.setattr(
        validate,
        "discover_metal_toolchain_identifier",
        lambda: "com.apple.dt.toolchain.Metal.32023.883",
    )
    monkeypatch.setattr(validate, "command_succeeds", lambda command, **kwargs: True)

    def capture_run_check(name: str, command: list[str], **kwargs: object) -> None:
        calls.append((name, command, kwargs))

    monkeypatch.setattr(validate, "run_check", capture_run_check)
    monkeypatch.setattr(validate, "print_check", lambda name: None)

    validate.run_metal_validation_checks()

    command_text = "\n".join(" ".join(command) for _, command, _ in calls)
    assert "WOLFGANG_ENABLE_METAL=ON" in command_text
    assert "WOLFGANG_ENABLE_CUDA=OFF" in command_text
    assert "WOLFGANG_ENABLE_HIP=OFF" in command_text
    assert any("Apple macOS SDK version" in name for name, _, _ in calls)
    assert any(command[-3:] == ["--sdk", "macosx", "--show-sdk-version"] for _, command, _ in calls)
    assert any("Apple Metal compiler version" in name for name, _, _ in calls)
    assert all(
        kwargs.get("env", {}).get("TOOLCHAINS") == "com.apple.dt.toolchain.Metal.32023.883"
        for _, _, kwargs in calls
        if kwargs.get("env") is not None
    )
    assert all(
        str(kwargs.get("env", {}).get("PATH", "")).split(validate.os.pathsep)[0]
        == "/venv/bin"
        for _, _, kwargs in calls
        if kwargs.get("env") is not None
    )
    assert any("Metal source build" in name for name, _, _ in calls)
    assert any("Metal-enabled semantic pytest" in name for name, _, _ in calls)
    semantic_pytest = next(command for name, command, _ in calls if name == "Metal-enabled semantic pytest")
    assert "--ignore" in semantic_pytest
    assert "tests/test_release_wheelhouse_foundation.py" in semantic_pytest
    assert any("Apple Metal kernel benchmark smoke" in name for name, _, _ in calls)


def test_metal_toolchain_identifier_is_discovered_from_xcodebuild(monkeypatch) -> None:
    validate = load_validate_module()

    class Completed:
        returncode = 0
        stdout = "\n".join(
            [
                "Status: installed",
                "Toolchain Identifier: com.apple.dt.toolchain.Metal.32023.883",
            ]
        )
        stderr = ""

    monkeypatch.setattr(validate, "find_executable", lambda name: "/usr/bin/xcodebuild")
    monkeypatch.setattr(validate.subprocess, "run", lambda *args, **kwargs: Completed())

    assert (
        validate.discover_metal_toolchain_identifier()
        == "com.apple.dt.toolchain.Metal.32023.883"
    )


def test_metal_validation_rejects_non_macos(monkeypatch) -> None:
    validate = load_validate_module()

    monkeypatch.setattr(validate.sys, "platform", "linux")
    monkeypatch.setattr(validate, "print_check", lambda name: None)

    with pytest.raises(SystemExit, match="Metal validation requires macOS"):
        validate.run_metal_validation_checks()

from __future__ import annotations

import pytest

from benchmarks._benchmark_metadata import (
    ROOT,
    accelerator_build_mode,
    benchmark_environment,
    benchmark_row_boundary,
    command_string,
    compiler_cpu_flags,
    git_commit,
    git_provenance,
    preferred_cpuinfo_value,
)


def test_preferred_cpuinfo_value_reports_model_before_vendor() -> None:
    cpuinfo = "\n".join(
        [
            "processor\t: 0",
            "vendor_id\t: AuthenticAMD",
            "cpu family\t: 25",
            "model name\t: AMD EPYC 9654 96-Core Processor",
        ]
    )

    assert preferred_cpuinfo_value(cpuinfo) == "AMD EPYC 9654 96-Core Processor"


def test_git_provenance_honors_benchmark_commit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WOLFGANG_BENCHMARK_GIT_COMMIT", "abc1234")

    provenance = git_provenance()

    assert git_commit() == "abc1234"
    assert provenance == {
        "commit": "abc1234",
        "commit_label": "abc1234",
        "dirty": False,
        "source": "WOLFGANG_BENCHMARK_GIT_COMMIT",
        "working_tree_status": [],
    }


def test_command_string_redacts_private_python_and_repo_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark_script = ROOT / "benchmarks" / "bench_cpu_dispatch.py"
    monkeypatch.setattr(
        "sys.executable",
        str(ROOT / ".venv" / "bin" / "python3"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [str(benchmark_script), "--repeat", "5", "--json"],
    )

    assert command_string() == "python benchmarks/bench_cpu_dispatch.py --repeat 5 --json"


def test_compiler_flags_record_only_allowlisted_cpu_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CXXFLAGS", "-O3 -march=native -I/private/research/include")
    monkeypatch.setenv("API_TOKEN", "must-not-be-recorded")

    metadata = compiler_cpu_flags({"compiler_build_config": {}})

    assert metadata == {
        "detected": ["-march=native"],
        "sources_checked": [
            "ARCHFLAGS",
            "CFLAGS",
            "CMAKE_ARGS",
            "CMAKE_BUILD_TYPE",
            "CMAKE_CXX_FLAGS",
            "CPPFLAGS",
            "CXXFLAGS",
            "SKBUILD_CMAKE_ARGS",
        ],
    }


def test_accelerator_build_mode_reports_campaign9_boundaries() -> None:
    assert accelerator_build_mode({"cuda_enabled": False, "hip_enabled": False}) == "cpu_only"
    assert accelerator_build_mode({"cuda_enabled": True, "hip_enabled": False}) == "cuda_only"
    assert accelerator_build_mode({"cuda_enabled": False, "hip_enabled": True}) == "hip_only"
    assert accelerator_build_mode({"metal_enabled": True}) == "metal_only"
    with pytest.raises(ValueError, match="unsupported mixed accelerator build metadata"):
        accelerator_build_mode({"cuda_enabled": True, "hip_enabled": True, "metal_enabled": False})


def test_benchmark_environment_includes_backend_neutral_accelerator_metadata() -> None:
    build_info = {
        "cpu_backend": "scalar",
        "cuda_enabled": True,
        "hip_enabled": False,
        "compiled_accelerator_backends": ["cuda"],
        "runtime_visible_accelerator_backends": ["cuda"],
        "cpu_backend_build_flags": {"scalar": True},
        "compiler_build_config": {
            "CMAKE_CXX_COMPILER_ID": "Clang",
            "CMAKE_CXX_COMPILER_VERSION": "18.1",
            "CMAKE_CXX_FLAGS": "-I/private/research/include",
            "CMAKE_CUDA_HOST_COMPILER": "/private/toolchain/bin/clang++",
        },
    }

    environment = benchmark_environment(build_info, numpy_version="2.0.0")

    assert environment["compiler_build_config"] == {
        "CMAKE_CXX_COMPILER_ID": "Clang",
        "CMAKE_CXX_COMPILER_VERSION": "18.1",
    }
    assert environment["accelerator_build_mode"] == "cuda_only"
    assert environment["compiled_accelerator_backends"] == ["cuda"]
    assert environment["runtime_visible_accelerator_backends"] == ["cuda"]
    assert environment["CUDA"]["enabled"] is True
    assert environment["HIP"]["enabled"] is False
    assert environment["Metal"]["enabled"] is False


def test_benchmark_row_boundary_normalizes_campaign9_status_rows() -> None:
    boundary = benchmark_row_boundary(
        build_info={
            "cuda_enabled": False,
            "hip_enabled": False,
            "compiled_accelerator_backends": [],
            "runtime_visible_accelerator_backends": [],
        },
        object_backend="cpu",
        transfer_boundary="status_only",
    )

    assert boundary == {
        "build_mode": "cpu_only",
        "object_backend": "cpu",
        "compiled_backends": ["cpu"],
        "runtime_visible_backends": ["cpu"],
        "transfer_boundary": "status_only",
    }

"""Native source-layout guardrails."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_BACKEND_SOURCE_PATTERN = re.compile(
    r"(^cuda_|_cuda|_kernels|_avx2|_avx512|_neon|_sve|_tbb)\.(c|cc|cpp|cxx|cu)$"
)


def test_phase_six_native_modules_are_split_by_responsibility() -> None:
    expected_sources = {
        "src/arithmetic.cpp",
        "src/commute.cpp",
        "src/cpu_backend.cpp",
        "src/expectation.cpp",
        "src/export.cpp",
        "src/grouping.cpp",
        "src/multiply.cpp",
        "src/parse.cpp",
        "src/pauli_sum.cpp",
        "src/simplify.cpp",
    }
    expected_private_headers = {
        "src/detail/bitops.hpp",
        "src/detail/checked_arithmetic.hpp",
        "src/detail/commutation.hpp",
        "src/detail/commute_kernels.hpp",
        "src/detail/packed_key.hpp",
        "src/detail/phase.hpp",
    }

    for relative_path in expected_sources | expected_private_headers:
        assert (ROOT / relative_path).exists(), f"missing native file: {relative_path}"

    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    for relative_path in expected_sources:
        assert relative_path in cmake


def test_internal_helpers_are_split_by_responsibility() -> None:
    assert not (ROOT / "src/detail/pauli_sum_internal.hpp").exists()

    forbidden_include = '#include "detail/pauli_sum_internal.hpp"'
    offenders: list[str] = []
    native_implementation_files = (
        *sorted((ROOT / "src").rglob("*.cpp")),
        *sorted((ROOT / "src").rglob("*.hpp")),
    )
    for native_path in native_implementation_files:
        if native_path.name.startswith("._"):
            continue
        source = native_path.read_text(encoding="utf-8")
        if forbidden_include in source:
            offenders.append(str(native_path.relative_to(ROOT)))

    assert not offenders, "source files must include focused detail headers: " + ", ".join(offenders)


def test_backend_specialized_sources_live_in_dedicated_directories() -> None:
    """Keep CPU backend specialization files out of the top-level src/ directory."""

    expected_simd_sources = {
        "src/simd/commute_kernels_scalar.cpp",
        "src/simd/commute_kernels_avx2.cpp",
        "src/simd/commute_kernels_avx512.cpp",
        "src/simd/commute_kernels_neon.cpp",
    }
    expected_parallel_sources = {
        "src/parallel/commute_kernels_tbb.cpp",
    }
    forbidden_root_sources = {
        "src/commute_kernels.cpp",
        "src/commute_kernels_avx2.cpp",
        "src/commute_kernels_avx512.cpp",
        "src/commute_kernels_neon.cpp",
        "src/commute_kernels_tbb.cpp",
    }

    for relative_path in expected_simd_sources | expected_parallel_sources:
        assert (ROOT / relative_path).exists(), f"missing native backend file: {relative_path}"

    for relative_path in forbidden_root_sources:
        assert not (ROOT / relative_path).exists(), (
            f"backend file belongs in src/simd or src/parallel: {relative_path}"
        )
    for source_path in sorted((ROOT / "src").glob("*")):
        assert not ROOT_BACKEND_SOURCE_PATTERN.search(source_path.name), (
            f"backend-specialized source belongs in src/simd, src/parallel, or src/cuda: {source_path}"
        )

    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    for relative_path in expected_simd_sources | expected_parallel_sources:
        assert relative_path in cmake
    for relative_path in forbidden_root_sources:
        assert relative_path not in cmake


def test_cuda_foundation_sources_live_under_cuda_directory() -> None:
    expected_cuda_sources = {
        "src/cuda/device_commutation_matrix.cu",
        "src/cuda/device_pauli_sum.cu",
        "src/cuda/simplify_cuda.cu",
        "src/cuda/expectation_cuda.cu",
        "src/cuda/commutation_cuda.cu",
        "src/cuda/matmul_cuda.cu",
    }
    expected_cuda_private_headers = {
        "src/cuda/device_commutation_matrix.cuh",
        "src/cuda/device_pauli_sum.cuh",
    }
    expected_public_headers = {
        "include/fastpauli/device_commutation_matrix.hpp",
        "include/fastpauli/device_pauli_sum.hpp",
    }
    expected_cpu_only_sources = {
        "src/device_commutation_matrix_stub.cpp",
        "src/device_pauli_sum_stub.cpp",
    }

    for relative_path in (
        expected_cuda_sources
        | expected_cuda_private_headers
        | expected_public_headers
        | expected_cpu_only_sources
    ):
        assert (ROOT / relative_path).exists(), f"missing CUDA foundation file: {relative_path}"

    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    for relative_path in expected_cuda_sources | expected_cpu_only_sources:
        assert relative_path in cmake


def test_device_commutation_matrix_cuda_source_imports_copy_helper_from_canonical_private_header() -> None:
    source = (ROOT / "src/cuda/device_commutation_matrix.cu").read_text(encoding="utf-8")
    assert "using cuda_detail::copy_to_device;" in source


def test_backend_neutral_accelerator_sources_are_declared_as_disjoint_sets() -> None:
    """Keep CUDA and HIP sources separable for target-specific accelerator builds."""

    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "FASTPAULI_CUDA_EXTENSION_SOURCES" in cmake
    assert "FASTPAULI_HIP_EXTENSION_SOURCES" in cmake
    assert "FASTPAULI_ACCELERATOR_STUB_SOURCES" in cmake
    assert "elseif(FASTPAULI_BUILD_HIP_ENABLED)" not in cmake

    cuda_sources = {
        "src/cuda/device_commutation_matrix.cu",
        "src/cuda/device_pauli_sum.cu",
        "src/cuda/simplify_cuda.cu",
        "src/cuda/expectation_cuda.cu",
        "src/cuda/commutation_cuda.cu",
        "src/cuda/matmul_cuda.cu",
        "src/cuda/workspace.cu",
    }
    hip_sources = {
        "src/hip/commutation_hip.hip.cpp",
        "src/hip/device_commutation_matrix.hip.cpp",
        "src/hip/device_pauli_sum.hip.cpp",
        "src/hip/expectation_hip.hip.cpp",
        "src/hip/matmul_hip.hip.cpp",
        "src/hip/simplify_hip.hip.cpp",
        "src/hip/workspace_hip.hip.cpp",
    }

    assert cuda_sources.isdisjoint(hip_sources)
    for relative_path in cuda_sources | hip_sources:
        assert (ROOT / relative_path).exists(), f"missing accelerator source: {relative_path}"
        assert relative_path in cmake

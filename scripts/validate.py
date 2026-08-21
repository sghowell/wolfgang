#!/usr/bin/env python3
"""Repo-local validation entrypoint for Wolfgang.

The script intentionally starts with mechanical checks that do not need the
package installed, then installs the package and runs build/import/test checks.
Each check prints its name before running so CI and agent closeouts have a
clear evidence trail.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import MutableMapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCE_OF_TRUTH_PATHS = (
    "README.md",
    "AGENTS.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "docs/roadmap.md",
    "docs/plans/cpp_cuda_implementation_plan.md",
    "docs/plans/release_candidate_foundation_plan.md",
    "docs/plans/release_candidate_next_checkpoint_plan.md",
    "docs/plans/release_0_1_0_wheelhouse_foundation_plan.md",
    "docs/plans/apple_metal_mps_bringup_plan.md",
    "docs/plans/apple_metal_optimization_campaign1_plan.md",
    "docs/plans/apple_metal_optimization_campaign2_plan.md",
    "docs/plans/apple_metal_optimization_campaign3_plan.md",
    "docs/plans/apple_metal_optimization_campaign4_plan.md",
    "docs/plans/apple_metal_optimization_campaign5_plan.md",
    "docs/plans/apple_metal_optimization_campaign6_plan.md",
    "docs/plans/apple_metal_optimization_campaign7_plan.md",
    "docs/plans/apple_metal_optimization_campaign8_plan.md",
    "docs/plans/cuda_deep_optimization_plan.md",
    "docs/plans/mi300x_rocm_bringup_plan.md",
    "docs/plans/rocm_next_waves_plan.md",
    "docs/plans/mi300x_rocm_optimization_campaign2_plan.md",
    "docs/plans/mi300x_rocm_optimization_campaign3_plan.md",
    "docs/plans/mi300x_rocm_optimization_campaign4_plan.md",
    "docs/plans/mi300x_rocm_optimization_campaign5_plan.md",
    "docs/plans/mi300x_rocm_optimization_campaign6_plan.md",
    "docs/plans/mi300x_rocm_optimization_campaign7_plan.md",
    "docs/plans/mi300x_rocm_optimization_campaign8_plan.md",
    "docs/plans/backend_neutral_accelerator_campaign9_plan.md",
    "docs/plans/wolfgang-kernel-performance-campaign.md",
    "docs/architecture/semantic_contracts.md",
    "docs/architecture/cuda_backend.md",
    "docs/architecture/rocm_backend.md",
    "docs/architecture/apple_accelerator.md",
    "docs/architecture/backend_neutral_accelerators.md",
    "docs/architecture/hardware_targets_and_testing.md",
    "docs/architecture/testing_and_ci.md",
    "docs/architecture/adapter_contracts.md",
    "docs/architecture/api_stability.md",
    "docs/benchmarks/protocol.md",
    "docs/quality/phase_quality_gates.md",
    "docs/quality/agent_harness.md",
    "docs/quality/code_review.md",
    "docs/quality/code_standards.md",
    "docs/quality/documentation_standards.md",
    "docs/quality/security_and_supply_chain.md",
    "docs/quality/release_and_packaging.md",
    "docs/release/README.md",
    "docs/release/0.2.2.md",
    "docs/release/0.2.3.md",
    "docs/release/0.1.0.md",
    "docs/release/0.1.0-wheelhouse-dry-run.md",
    "docs/release/0.1.0-rc1.md",
    "docs/release/0.1.0-rc2.md",
    "docs/release/cloud_hardware_qualification_harness.md",
    "docs/release/support_matrix.md",
)

STALE_MARKERS = (
    "TBD",
    "TODO",
    "placeholder",
    "fill in",
    "future constructor",
    "CUDA should be optional",
    "first release should be CPU-only",
    "C++23",
    "GPU implementation before CPU correctness",
    "one sparse tuple",
)

ROOT_NATIVE_SOURCES = (
    "src/accelerator_status.cpp",
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
)

BACKEND_SPECIALIZED_SOURCES = (
    "src/simd/commute_kernels_scalar.cpp",
    "src/simd/commute_kernels_avx2.cpp",
    "src/simd/commute_kernels_avx512.cpp",
    "src/simd/commute_kernels_neon.cpp",
    "src/parallel/commute_kernels_tbb.cpp",
)

CUDA_FOUNDATION_SOURCES = (
    "include/wolfgang/device_commutation_matrix.hpp",
    "include/wolfgang/device_pauli_sum.hpp",
    "src/device_commutation_matrix_stub.cpp",
    "src/device_pauli_sum_stub.cpp",
    "src/cuda/device_commutation_matrix.cu",
    "src/cuda/device_pauli_sum.cu",
    "src/cuda/simplify_cuda.cu",
    "src/cuda/expectation_cuda.cu",
    "src/cuda/commutation_cuda.cu",
    "src/cuda/matmul_cuda.cu",
    "src/cuda/device_commutation_matrix.cuh",
    "src/cuda/device_pauli_sum.cuh",
)

HIP_FOUNDATION_SOURCES = (
    "src/hip/commutation_hip.hip.cpp",
    "src/hip/commutation_hip.hip.hpp",
    "src/hip/device_commutation_matrix.hip.cpp",
    "src/hip/device_commutation_matrix.hip.hpp",
    "src/hip/device_pauli_sum.hip.cpp",
    "src/hip/device_pauli_sum.hip.hpp",
    "src/hip/expectation_hip.hip.cpp",
    "src/hip/matmul_hip.hip.cpp",
    "src/hip/simplify_hip.hip.cpp",
    "src/hip/simplify_hip.hip.hpp",
)

METAL_FOUNDATION_SOURCES = (
    "src/metal/accelerator_metal.mm",
    "src/metal/device_commutation_matrix_metal.mm",
    "src/metal/device_pauli_sum_metal.mm",
    "src/metal/commutation_metal.mm",
    "src/metal/simplify_metal.mm",
    "src/metal/workspace_metal.mm",
    "src/metal/device_commutation_matrix_metal.hpp",
    "src/metal/device_pauli_sum_metal.hpp",
    "src/metal/simplify_metal.hpp",
    "src/metal/workspace_metal.hpp",
    "src/metal/kernels/commutation.metal",
    "src/metal/kernels/simplify.metal",
)

DEFAULT_CUDA_ARCHITECTURES = "80,86,89,90,100-real,120"
DEFAULT_HIP_ARCHITECTURES = "gfx942"

DETAIL_HEADERS = (
    "src/detail/bitops.hpp",
    "src/detail/checked_arithmetic.hpp",
    "src/detail/commutation.hpp",
    "src/detail/commute_kernels.hpp",
    "src/detail/packed_key.hpp",
    "src/detail/phase.hpp",
)

FORBIDDEN_ROOT_BACKEND_SOURCES = (
    "src/commute_kernels.cpp",
    "src/commute_kernels_avx2.cpp",
    "src/commute_kernels_avx512.cpp",
    "src/commute_kernels_neon.cpp",
    "src/commute_kernels_tbb.cpp",
)

ROOT_BACKEND_SOURCE_PATTERN = re.compile(
    r"(^cuda_|_cuda|_kernels|_avx2|_avx512|_neon|_sve|_tbb)\.(c|cc|cpp|cxx|cu)$"
)


def print_check(name: str) -> None:
    print(f"\n==> {name}", flush=True)


def fail(message: str) -> None:
    raise SystemExit(message)


def run_check(name: str, command: list[str], *, env: dict[str, str] | None = None) -> None:
    print_check(name)
    print("$ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if completed.returncode != 0:
        fail(f"{name} failed with exit code {completed.returncode}")


def command_succeeds(command: list[str], *, env: dict[str, str] | None = None) -> bool:
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return completed.returncode == 0


def pip_install_supports_config_settings() -> bool:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0 and "--config-settings" in completed.stdout


def ensure_pip_install_config_settings_support() -> None:
    if pip_install_supports_config_settings():
        return

    run_check(
        "upgrade pip for PEP 517 config settings",
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ],
    )
    if not pip_install_supports_config_settings():
        fail("python -m pip install still lacks --config-settings after upgrade")


def find_executable(name: str) -> str | None:
    sibling = Path(sys.executable).with_name(name)
    if sibling.exists():
        return str(sibling)

    path_value = shutil.which(name)
    if path_value is not None:
        return path_value
    return None


def prepend_executable_directory_to_path(env: MutableMapping[str, str]) -> None:
    executable_dir = str(Path(sys.executable).parent)
    path_entries = [entry for entry in env.get("PATH", "").split(os.pathsep) if entry]
    if executable_dir not in path_entries:
        env["PATH"] = os.pathsep.join([executable_dir, *path_entries])


def cmake_executable_for_build_isolation() -> str | None:
    try:
        import cmake as cmake_package  # type: ignore[import-not-found]
    except Exception:
        return find_executable("cmake")

    package_cmake = Path(cmake_package.CMAKE_BIN_DIR) / "cmake"
    if package_cmake.exists():
        return str(package_cmake)
    return find_executable("cmake")


def discover_metal_toolchain_identifier() -> str | None:
    xcodebuild = find_executable("xcodebuild")
    if xcodebuild is None:
        return None

    try:
        completed = subprocess.run(
            [xcodebuild, "-showComponent", "MetalToolchain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None

    if completed.returncode != 0 or "Status: installed" not in completed.stdout:
        return None

    for line in completed.stdout.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == "Toolchain Identifier":
            identifier = value.strip()
            if identifier:
                return identifier
    return None


def metal_validation_env() -> dict[str, str]:
    env = os.environ.copy()
    prepend_executable_directory_to_path(env)
    if "TOOLCHAINS" not in env:
        identifier = discover_metal_toolchain_identifier()
        if identifier is not None:
            env["TOOLCHAINS"] = identifier
    return env


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def check_source_docs_exist() -> None:
    print_check("source-of-truth files exist")
    missing = [path for path in SOURCE_OF_TRUTH_PATHS if not (ROOT / path).exists()]
    if missing:
        fail("Missing source-of-truth files:\n" + "\n".join(missing))
    for path in SOURCE_OF_TRUTH_PATHS:
        print(f"found {path}")


def check_markdown_links() -> None:
    print_check("README.md and AGENTS.md local links resolve")
    checked: list[str] = []

    markdown_link = re.compile(r"\[[^\]]+\]\(([^)#][^)]+)\)")
    for source in ("README.md", "AGENTS.md"):
        text = read_text(source)
        for match in markdown_link.finditer(text):
            target = match.group(1).split("#", 1)[0]
            if "://" in target or target.startswith("mailto:"):
                continue
            if not target:
                continue
            if not (ROOT / target).exists():
                fail(f"{source} links to missing path: {target}")
            checked.append(f"{source}: {target}")

    for path in SOURCE_OF_TRUTH_PATHS:
        if not (ROOT / path).exists():
            fail(f"AGENTS.md source-of-truth path is missing: {path}")
        checked.append(f"source-of-truth: {path}")

    for item in checked:
        print(f"checked {item}")


def check_stale_markers() -> None:
    print_check("stale planning marker scan")
    scan_roots = (
        ROOT / "README.md",
        ROOT / "AGENTS.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "docs",
    )
    findings: list[str] = []
    for root in scan_roots:
        paths = [root] if root.is_file() else sorted(root.rglob("*.md"))
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for marker in STALE_MARKERS:
                if marker in text:
                    rel = path.relative_to(ROOT)
                    findings.append(f"{rel}: unsupported marker {marker!r}")
    if findings:
        fail("Unsupported stale markers found:\n" + "\n".join(findings))
    print("no unsupported stale markers found")


def check_cmake_defaults() -> None:
    print_check("CMake CPU-safe defaults are declared")
    cmake = read_text("CMakeLists.txt")
    required_snippets = (
        'function(_wolfgang_bool_option canonical legacy doc default)',
        'function(_wolfgang_string_option canonical legacy default doc)',
        'option(${canonical} "${doc}" ${default})',
        'set(${canonical} "${default}" CACHE STRING "${doc}")',
        'message(DEPRECATION "${legacy} is deprecated; use ${canonical}.")',
        'set(${legacy} "${${canonical}}" CACHE BOOL "Deprecated alias for ${canonical}" FORCE)',
        'set(${legacy} "${${canonical}}" CACHE STRING "Deprecated alias for ${canonical}" FORCE)',
        '_wolfgang_bool_option(WOLFGANG_ENABLE_CUDA WOLFGANG_ENABLE_CUDA "Build CUDA backend support" OFF)',
        '_wolfgang_bool_option(WOLFGANG_ENABLE_HIP WOLFGANG_ENABLE_HIP "Build ROCm/HIP backend support" OFF)',
        '_wolfgang_bool_option(WOLFGANG_ENABLE_METAL WOLFGANG_ENABLE_METAL "Build Apple Metal backend support" OFF)',
        '_wolfgang_bool_option(WOLFGANG_ENABLE_NATIVE WOLFGANG_ENABLE_NATIVE "Allow native CPU tuning such as -march=native" OFF)',
        '_wolfgang_bool_option(WOLFGANG_ENABLE_OPENMP WOLFGANG_ENABLE_OPENMP "Build OpenMP-enabled CPU paths" OFF)',
        '_wolfgang_string_option(WOLFGANG_ENABLE_TBB WOLFGANG_ENABLE_TBB "auto" "Build oneTBB-enabled CPU paths: auto, ON, or OFF")',
        '_wolfgang_string_option(WOLFGANG_ENABLE_AVX2 WOLFGANG_ENABLE_AVX2 "auto" "Build AVX2-dispatched CPU paths: auto, ON, or OFF")',
        '_wolfgang_string_option(WOLFGANG_ENABLE_AVX512 WOLFGANG_ENABLE_AVX512 "auto" "Build AVX-512-dispatched CPU paths: auto, ON, or OFF")',
        '_wolfgang_string_option(WOLFGANG_ENABLE_ARM_NEON WOLFGANG_ENABLE_ARM_NEON "auto" "Build ARM NEON-dispatched CPU paths: auto, ON, or OFF")',
        '_wolfgang_string_option(WOLFGANG_ENABLE_ARM_SVE WOLFGANG_ENABLE_ARM_SVE "auto" "Build ARM SVE-dispatched CPU paths: auto, ON, or OFF")',
        '_wolfgang_string_option(WOLFGANG_CUDA_ARCHITECTURES WOLFGANG_CUDA_ARCHITECTURES "80;86;89;90;100-real;120" "CUDA architectures for WOLFGANG_ENABLE_CUDA=ON source builds")',
        '_wolfgang_string_option(WOLFGANG_HIP_ARCHITECTURES WOLFGANG_HIP_ARCHITECTURES "gfx942" "HIP architectures for WOLFGANG_ENABLE_HIP=ON source builds")',
        'WOLFGANG_ENABLE_CUDA and WOLFGANG_ENABLE_HIP cannot both be ON',
        'WOLFGANG_ENABLE_CUDA and WOLFGANG_ENABLE_METAL cannot both be ON',
        'WOLFGANG_ENABLE_HIP and WOLFGANG_ENABLE_METAL cannot both be ON',
        'Release-wheel-safe builds require WOLFGANG_ENABLE_NATIVE=OFF.',
        'Phase 10 CUDA transfers require WOLFGANG_CUDA_USE_THRUST=ON.',
        'WOLFGANG_ENABLE_METAL=ON requires Apple platforms with Metal.framework.',
        'WOLFGANG_ENABLE_TBB=ON requested, but oneTBB was not found by CMake.',
        'WOLFGANG_ENABLE_AVX2=ON requested, but this compiler/target cannot build the AVX2 object.',
        'WOLFGANG_ENABLE_AVX512=ON requested, but this compiler/target cannot build the AVX-512 VPOPCNTDQ object.',
        'WOLFGANG_ENABLE_ARM_NEON=ON requested, but this compiler/target cannot build the NEON object.',
        'WOLFGANG_ENABLE_ARM_SVE=ON requested, but ARM SVE kernels are not implemented yet.',
        'WOLFGANG_BUILD_CPU_BACKEND="scalar"',
        'WOLFGANG_BUILD_CUDA_ENABLED=${WOLFGANG_BUILD_CUDA_ENABLED}',
        'WOLFGANG_BUILD_CUDA_ARCHITECTURES="${WOLFGANG_BUILD_CUDA_ARCHITECTURES}"',
        'WOLFGANG_BUILD_HIP_ENABLED=${WOLFGANG_BUILD_HIP_ENABLED}',
        'WOLFGANG_BUILD_HIP_ARCHITECTURES="${WOLFGANG_BUILD_HIP_ARCHITECTURES}"',
        'WOLFGANG_BUILD_METAL_ENABLED=${WOLFGANG_BUILD_METAL_ENABLED}',
        'WOLFGANG_BUILD_TBB_ENABLED=${WOLFGANG_BUILD_TBB_ENABLED}',
        'WOLFGANG_BUILD_AVX2_ENABLED=${WOLFGANG_BUILD_AVX2_ENABLED}',
        'WOLFGANG_BUILD_AVX512_ENABLED=${WOLFGANG_BUILD_AVX512_ENABLED}',
        'WOLFGANG_BUILD_ARM_NEON_ENABLED=${WOLFGANG_BUILD_ARM_NEON_ENABLED}',
        'WOLFGANG_BUILD_ARM_SVE_ENABLED=${WOLFGANG_BUILD_ARM_SVE_ENABLED}',
    )
    missing = [snippet for snippet in required_snippets if snippet not in cmake]
    if missing:
        fail("CMakeLists.txt is missing required CPU-safe defaults:\n" + "\n".join(missing))
    for snippet in required_snippets:
        print(f"checked {snippet}")


def check_native_source_layout() -> None:
    print_check("native source layout matches implementation plan")
    required_paths = (
        *DETAIL_HEADERS,
        *ROOT_NATIVE_SOURCES,
        *BACKEND_SPECIALIZED_SOURCES,
        *CUDA_FOUNDATION_SOURCES,
        *HIP_FOUNDATION_SOURCES,
        *METAL_FOUNDATION_SOURCES,
    )
    missing = [path for path in required_paths if not (ROOT / path).exists()]
    if missing:
        fail("Missing required native layout paths:\n" + "\n".join(missing))

    retired_monolith = ROOT / "src/detail/pauli_sum_internal.hpp"
    if retired_monolith.exists():
        fail("Retired native detail header must not be restored: src/detail/pauli_sum_internal.hpp")

    misplaced = [path for path in FORBIDDEN_ROOT_BACKEND_SOURCES if (ROOT / path).exists()]
    misplaced.extend(
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "src").glob("*"))
        if path.is_file() and ROOT_BACKEND_SOURCE_PATTERN.search(path.name)
    )
    misplaced = sorted(set(misplaced))
    if misplaced:
        fail(
            "Backend-specialized native sources must live under src/simd or src/parallel:\n"
            + "\n".join(misplaced)
        )

    cmake = read_text("CMakeLists.txt")
    missing_from_cmake = [
        path
        for path in (
            *ROOT_NATIVE_SOURCES,
            *BACKEND_SPECIALIZED_SOURCES,
            *CUDA_FOUNDATION_SOURCES,
            *HIP_FOUNDATION_SOURCES,
            *METAL_FOUNDATION_SOURCES,
        )
        if path.endswith((".cpp", ".cu", ".hip.cpp", ".mm")) and path not in cmake
    ]
    if missing_from_cmake:
        fail("CMakeLists.txt is missing native source paths:\n" + "\n".join(missing_from_cmake))

    forbidden_in_cmake = [path for path in FORBIDDEN_ROOT_BACKEND_SOURCES if path in cmake]
    if forbidden_in_cmake:
        fail(
            "CMakeLists.txt still references top-level backend-specialized sources:\n"
            + "\n".join(forbidden_in_cmake)
        )

    monolithic_include = '#include "detail/pauli_sum_internal.hpp"'
    native_implementation_files = (
        *sorted((ROOT / "src").rglob("*.cpp")),
        *sorted((ROOT / "src").rglob("*.hpp")),
    )
    stale_includes = [
        str(path.relative_to(ROOT))
        for path in native_implementation_files
        if monolithic_include in path.read_text(encoding="utf-8")
    ]
    if stale_includes:
        fail(
            "Native sources must include focused detail headers instead of pauli_sum_internal.hpp:\n"
            + "\n".join(stale_includes)
        )

    print("native source layout is aligned")


def check_review_policy() -> None:
    print_check("review-policy existence and closeout checklist")
    code_review = read_text("docs/quality/code_review.md")
    agents = read_text("AGENTS.md")
    contributing = read_text("CONTRIBUTING.md")
    required_review_terms = (
        "reviewer type",
        "review scope",
        "finding counts by severity",
        "validation rerun after review fixes",
        "residual risk",
    )
    missing_terms = [term for term in required_review_terms if term not in code_review]
    if missing_terms:
        fail("docs/quality/code_review.md is missing closeout terms:\n" + "\n".join(missing_terms))
    if "docs/quality/code_review.md" not in agents:
        fail("AGENTS.md does not reference docs/quality/code_review.md")
    if "docs/quality/code_review.md" not in contributing:
        fail("CONTRIBUTING.md does not reference docs/quality/code_review.md")
    print("review policy and closeout terms are present")


def run_build_info_check() -> None:
    script = (
        "import wolfgang_quantum; "
        "import wolfgang_quantum._wolfgang_core as core; "
        "info = core._build_info(); "
        "print(info); "
        "assert info['cpu_backend'] == 'scalar'; "
        "assert info['cuda_enabled'] is False; "
        "assert info['hip_enabled'] is False; "
        "assert info['metal_enabled'] is False; "
        "assert 'metal_capability_summary' in info; "
        "assert info['native_enabled'] is False; "
        "assert info['accelerator_build_mode'] == 'cpu_only'; "
        "assert info['compiled_accelerator_backends'] == []; "
        "assert info['runtime_visible_accelerator_backends'] == []; "
        "assert info['compiled_backends'] == ['cpu']; "
        "assert info['runtime_visible_backends'] == ['cpu']; "
        "assert core._metal_status()['built'] is False; "
        "assert 'scalar' in info['compiled_cpu_backends']; "
        "assert 'oneTBB_version' in info; "
        "assert 'optimized_cpu_kernels' in info; "
        "print(wolfgang_quantum.__version__)"
    )
    run_check(
        "import wolfgang_quantum and report scalar CPU build info",
        [sys.executable, "-c", script],
    )


def run_optional_openfermion_checks() -> None:
    if not command_succeeds([sys.executable, "-c", "import openfermion"]):
        print_check("OpenFermion optional checks")
        print("skipping OpenFermion adapter tests and benchmark smoke: openfermion is not installed")
        return

    run_check(
        "OpenFermion adapter pytest",
        [sys.executable, "-m", "pytest", "tests/test_openfermion_adapter.py"],
    )
    run_check(
        "OpenFermion conversion benchmark smoke",
        [
            sys.executable,
            "benchmarks/bench_openfermion_conversion.py",
            "--smoke",
            "--repeat",
            "1",
            "--json",
        ],
    )


def run_cmake_configure_check() -> None:
    cmake = cmake_executable_for_build_isolation()
    if cmake is None:
        fail("cmake executable is not available after installing the test extra")
    build_dir = ROOT / "_skbuild" / "validate-cpu"
    run_check(
        "CPU-only CMake configure with CUDA and native tuning disabled",
        [
            cmake,
            "-S",
            str(ROOT),
            "-B",
            str(build_dir),
            "-DWOLFGANG_ENABLE_CUDA=OFF",
            "-DWOLFGANG_ENABLE_HIP=OFF",
            "-DWOLFGANG_ENABLE_METAL=OFF",
            "-DWOLFGANG_ENABLE_NATIVE=OFF",
            f"-DPython_EXECUTABLE={sys.executable}",
        ],
    )


def run_cuda_validation_checks() -> None:
    print_check("CUDA validation environment")
    nvcc = cuda_compiler()
    cmake = find_executable("cmake")
    if nvcc is None:
        fail("WOLFGANG_VALIDATE_CUDA=1 requested, but nvcc is not available")
    if cmake is None:
        fail("WOLFGANG_VALIDATE_CUDA=1 requested, but cmake is not available")

    requested_architectures = os.environ.get("WOLFGANG_CUDA_ARCHITECTURES")
    print(f"nvcc: {nvcc}")
    print(f"cmake: {cmake}")
    print(f"default WOLFGANG_CUDA_ARCHITECTURES={DEFAULT_CUDA_ARCHITECTURES}")
    if requested_architectures is not None:
        print(f"requested WOLFGANG_CUDA_ARCHITECTURES={requested_architectures}")

    run_check("CUDA toolkit version from nvcc", [nvcc, "--version"])

    host_compiler = cuda_host_compiler()
    if host_compiler is None:
        fail("WOLFGANG_VALIDATE_CUDA=1 requested, but no CUDA host C++ compiler is available")
    assert host_compiler is not None
    run_check("CUDA host compiler version", [host_compiler, "--version"])

    nvidia_smi = find_executable("nvidia-smi")
    if nvidia_smi is not None:
        run_check(
            "CUDA device summary from nvidia-smi",
            [
                nvidia_smi,
                "--query-gpu=name,compute_cap,driver_version",
                "--format=csv,noheader",
            ],
        )
    else:
        print("nvidia-smi is unavailable; CUDA runtime status will be reported by FastPauli")

    validation_env = cuda_validation_env(nvcc=nvcc, host_compiler=host_compiler)

    effective_architectures = requested_architectures or DEFAULT_CUDA_ARCHITECTURES

    run_cuda_source_build(effective_architectures, nvcc=nvcc, host_compiler=host_compiler)
    run_cuda_build_info_check(effective_architectures.replace(";", ","))
    run_check(
        "install CuPy CUDA toolkit headers",
        [sys.executable, "-m", "pip", "install", "cupy-cuda12x[ctk]"],
        env=validation_env,
    )

    run_check(
        "CUDA-enabled semantic pytest",
        [
            sys.executable,
            "-m",
            "pytest",
            "--ignore",
            "tests/test_release_wheelhouse_foundation.py",
            "--ignore",
            "tests/test_release_artifact_validation.py",
            "--ignore",
            "tests/test_apple_metal_foundation.py",
        ],
        env=validation_env,
    )
    run_check(
        "CUDA transfer pytest",
        [sys.executable, "-m", "pytest", "tests/test_phase10_cuda_foundation.py"],
        env=validation_env,
    )
    run_check(
        "CUDA kernel pytest",
        [sys.executable, "-m", "pytest", "tests/test_phase11_cuda_kernels.py"],
        env=validation_env,
    )
    run_check(
        "CUDA kernel benchmark smoke",
        [sys.executable, "benchmarks/bench_cuda_kernels.py", "--smoke", "--repeat", "1", "--json"],
        env=validation_env,
    )
    run_check(
        "CUDA scaling benchmark smoke",
        [
            sys.executable,
            "benchmarks/bench_cuda_scaling.py",
            "--profile",
            "smoke",
            "--repeat",
            "1",
            "--json",
        ],
        env=validation_env,
    )


def cuda_host_compiler() -> str | None:
    for variable in ("CMAKE_CUDA_HOST_COMPILER", "CUDAHOSTCXX", "CXX"):
        value = os.environ.get(variable)
        if value:
            return value
    return find_executable("g++") or find_executable("c++")


def cuda_compiler() -> str | None:
    value = os.environ.get("CUDACXX")
    if value:
        return value
    return find_executable("nvcc")


def cuda_validation_env(*, nvcc: str, host_compiler: str) -> dict[str, str]:
    env = os.environ.copy()
    cuda_root = str(Path(nvcc).resolve().parent.parent)
    env["CUDACXX"] = nvcc
    env["CUDAHOSTCXX"] = host_compiler
    env["CUDA_PATH"] = cuda_root
    env.pop("CPLUS_INCLUDE_PATH", None)
    return env


def run_cuda_source_build(architectures: str, *, nvcc: str, host_compiler: str) -> None:
    uv = find_executable("uv")
    if uv is not None:
        command = [uv, "pip", "install"]
    else:
        ensure_pip_install_config_settings_support()
        command = [sys.executable, "-m", "pip", "install"]

    command.extend(
        [
            "-e",
            ".[test]",
            "--config-settings=cmake.define.WOLFGANG_ENABLE_INTERNAL_BINDINGS=ON",
            "--config-settings=cmake.define.WOLFGANG_ENABLE_CUDA=ON",
            f"--config-settings=cmake.define.CMAKE_CUDA_HOST_COMPILER={host_compiler}",
            f"--config-settings=cmake.define.CUDAToolkit_ROOT={Path(nvcc).resolve().parent.parent}",
        ]
    )
    env = cuda_validation_env(nvcc=nvcc, host_compiler=host_compiler)
    normalized_architectures = architectures.replace(";", ",")
    check_name = "CUDA source build with default architectures"
    if normalized_architectures != DEFAULT_CUDA_ARCHITECTURES:
        check_name = "CUDA source build with requested architecture override"
        command.append(
            f"--config-settings=cmake.define.WOLFGANG_CUDA_ARCHITECTURES={architectures}"
        )

    run_check(check_name, command, env=env)


def run_cuda_build_info_check(expected_architectures: str) -> None:
    script = (
        "import wolfgang_quantum; "
        "import wolfgang_quantum._wolfgang_core as core; "
        "info = core._build_info(); "
        "status = core._cuda_status(); "
        "print(info); "
        "print(status); "
        "compiler = info['compiler_build_config']; "
        "assert info['cuda_enabled'] is True; "
        "assert info['accelerator_build_mode'] == 'cuda_only'; "
        f"assert info['cuda_architectures'] == {expected_architectures!r}; "
        "assert info['cuda_toolkit_version']; "
        "assert info['cuda_toolkit_version'] != 'not_available'; "
        "assert compiler['WOLFGANG_CUDA_HOST_COMPILER']; "
        "assert compiler['WOLFGANG_CUDA_HOST_COMPILER'] != 'not_available'; "
        "assert compiler['WOLFGANG_CUDA_HOST_COMPILER_SOURCE']; "
        "assert compiler['WOLFGANG_CUDA_HOST_COMPILER_SOURCE'] != 'not_available'; "
        "assert status['built'] is True; "
        "assert status['runtime_available'] is True; "
        "assert status['device_count'] >= 1; "
        "assert status['devices']; "
        "assert status['devices'][0]['compute_capability'][0] >= 7"
    )
    run_check(
        "CUDA build info, runtime status, and device metadata",
        [sys.executable, "-c", script],
    )


def hip_compiler() -> str | None:
    for name in ("hipcc", "amdclang++", "clang++"):
        found = find_executable(name)
        if found is not None:
            return found
    return None


def hip_validation_env() -> dict[str, str]:
    env = os.environ.copy()
    rocm_paths = ("/opt/rocm/bin", "/opt/rocm/llvm/bin")
    existing_path = env.get("PATH", "")
    env["PATH"] = ":".join((*rocm_paths, existing_path))
    return env


def run_hip_validation_checks() -> None:
    print_check("HIP validation environment")
    hip = hip_compiler()
    cmake = find_executable("cmake")
    if hip is None:
        fail("WOLFGANG_VALIDATE_HIP=1 requested, but hipcc/amdclang++ is not available")
    if cmake is None:
        fail("WOLFGANG_VALIDATE_HIP=1 requested, but cmake is not available")

    requested_architectures = os.environ.get("WOLFGANG_HIP_ARCHITECTURES")
    print(f"HIP compiler: {hip}")
    print(f"cmake: {cmake}")
    print(f"default WOLFGANG_HIP_ARCHITECTURES={DEFAULT_HIP_ARCHITECTURES}")
    if requested_architectures is not None:
        print(f"requested WOLFGANG_HIP_ARCHITECTURES={requested_architectures}")

    run_check("HIP compiler version from hipcc", [hip, "--version"], env=hip_validation_env())

    hip_smi = find_executable("amd-smi")
    if hip_smi is not None:
        run_check("ROCm device summary from amd-smi", [hip_smi, "static"], env=hip_validation_env())
    else:
        print("amd-smi is unavailable; HIP runtime status will be reported by FastPauli")

    run_hip_source_build(DEFAULT_HIP_ARCHITECTURES)
    run_hip_build_info_check(DEFAULT_HIP_ARCHITECTURES)

    if (
        requested_architectures
        and requested_architectures.replace(";", ",") != DEFAULT_HIP_ARCHITECTURES
    ):
        run_hip_source_build(requested_architectures, requested=True)
        run_hip_build_info_check(requested_architectures.replace(";", ","))

    run_check(
        "HIP-enabled semantic pytest",
        [sys.executable, "-m", "pytest"],
        env=hip_validation_env(),
    )
    run_check(
        "HIP foundation pytest",
        [sys.executable, "-m", "pytest", "tests/test_phase12_rocm_foundation.py"],
        env=hip_validation_env(),
    )
    run_check(
        "ROCm kernel benchmark smoke on HIP build",
        [sys.executable, "benchmarks/bench_rocm_kernels.py", "--smoke", "--repeat", "1", "--json"],
        env=hip_validation_env(),
    )


def run_hip_source_build(architectures: str, *, requested: bool = False) -> None:
    ensure_pip_install_config_settings_support()
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-e",
        ".[test]",
        "--config-settings=cmake.define.WOLFGANG_ENABLE_INTERNAL_BINDINGS=ON",
        "--config-settings=cmake.define.WOLFGANG_ENABLE_HIP=ON",
        f"--config-settings=cmake.define.WOLFGANG_HIP_ARCHITECTURES={architectures}",
    ]
    check_name = "HIP source build for gfx942"
    if requested:
        check_name = "HIP source build for requested architecture override"
    run_check(check_name, command, env=hip_validation_env())


def run_hip_build_info_check(expected_architectures: str) -> None:
    script = (
        "import wolfgang_quantum; "
        "import wolfgang_quantum._wolfgang_core as core; "
        "info = core._build_info(); "
        "status = core._hip_status(); "
        "print(info); "
        "print(status); "
        "compiler = info['compiler_build_config']; "
        "assert info['hip_enabled'] is True; "
        f"assert info['hip_architectures'] == {expected_architectures!r}; "
        "assert info['rocm_toolkit_version']; "
        "assert info['rocm_toolkit_version'] != 'not_available'; "
        "assert compiler['CMAKE_HIP_COMPILER']; "
        "assert compiler['CMAKE_HIP_COMPILER'] != 'not_available'; "
        "assert compiler['CMAKE_HIP_COMPILER_ID']; "
        "assert compiler['CMAKE_HIP_COMPILER_ID'] != 'not_available'; "
        "assert status['built'] is True; "
        "assert status['runtime_available'] is True; "
        "assert status['device_count'] >= 1; "
        "assert status['devices']; "
        "assert status['devices'][0]['name']"
    )
    run_check(
        "HIP build info, runtime status, and device metadata",
        [sys.executable, "-c", script],
        env=hip_validation_env(),
    )


def run_metal_validation_checks() -> None:
    print_check("Metal validation environment")
    if sys.platform != "darwin":
        fail("WOLFGANG_VALIDATE_METAL=1 requested, but Metal validation requires macOS")
    cmake = find_executable("cmake")
    if cmake is None:
        fail("WOLFGANG_VALIDATE_METAL=1 requested, but cmake is not available")
    print(f"cmake: {cmake}")

    validation_env = metal_validation_env()
    xcrun = find_executable("xcrun")
    if xcrun is not None:
        if command_succeeds([xcrun, "--find", "metal"], env=validation_env):
            run_check(
                "Apple Metal compiler path from xcrun",
                [xcrun, "--find", "metal"],
                env=validation_env,
            )
        else:
            print(
                "xcrun is available, but xcrun --find metal did not locate the standalone "
                "Metal compiler; continuing to validate the CMake source build"
            )
        if command_succeeds([xcrun, "metal", "-v"], env=validation_env):
            run_check("Apple Metal compiler version", [xcrun, "metal", "-v"], env=validation_env)
        else:
            print(
                "xcrun found the Metal compiler path, but the standalone compiler version "
                "probe did not run; continuing to validate the CMake source build"
            )
        run_check(
            "Apple macOS SDK version",
            [xcrun, "--sdk", "macosx", "--show-sdk-version"],
            env=validation_env,
        )
    else:
        print("xcrun is unavailable; Metal runtime status will be reported by FastPauli")

    run_metal_source_build(validation_env)
    run_metal_build_info_check(validation_env)
    run_check(
        "Metal-enabled semantic pytest",
        [
            sys.executable,
            "-m",
            "pytest",
            "--ignore",
            "tests/test_release_wheelhouse_foundation.py",
        ],
        env=validation_env,
    )
    run_check(
        "Apple Metal foundation pytest",
        [sys.executable, "-m", "pytest", "tests/test_apple_metal_foundation.py"],
        env=validation_env,
    )
    run_check(
        "Apple Metal kernel benchmark smoke",
        [sys.executable, "benchmarks/bench_metal_kernels.py", "--smoke", "--repeat", "1", "--json"],
        env=validation_env,
    )
    run_check(
        "Apple Metal Campaign 3 experimental benchmark smoke",
        [
            sys.executable,
            "benchmarks/bench_metal_kernels.py",
            "--profile",
            "campaign3",
            "--repeat",
            "1",
            "--json",
        ],
        env=validation_env,
    )
    run_check(
        "Apple Metal Campaign 4 experimental benchmark smoke",
        [
            sys.executable,
            "benchmarks/bench_metal_kernels.py",
            "--profile",
            "campaign4",
            "--repeat",
            "1",
            "--json",
        ],
        env=validation_env,
    )
    run_check(
        "Apple Metal Campaign 5 simplify benchmark smoke",
        [
            sys.executable,
            "benchmarks/bench_metal_kernels.py",
            "--profile",
            "campaign5",
            "--repeat",
            "1",
            "--json",
        ],
        env=validation_env,
    )
    run_check(
        "Apple Metal Campaign 6 device-resident simplify groundwork smoke",
        [
            sys.executable,
            "benchmarks/bench_metal_kernels.py",
            "--profile",
            "campaign6",
            "--repeat",
            "1",
            "--json",
        ],
        env=validation_env,
    )
    run_check(
        "Apple Metal Campaign 7 checked primitive stack smoke",
        [
            sys.executable,
            "benchmarks/bench_metal_kernels.py",
            "--profile",
            "campaign7",
            "--repeat",
            "1",
            "--json",
        ],
        env=validation_env,
    )
    run_check(
        "Apple Metal Campaign 8 simplify performance relevance smoke",
        [
            sys.executable,
            "benchmarks/bench_metal_kernels.py",
            "--profile",
            "campaign8",
            "--repeat",
            "1",
            "--json",
        ],
        env=validation_env,
    )


def run_metal_source_build(env: dict[str, str] | None = None) -> None:
    uv = find_executable("uv")
    if uv is not None:
        command = [uv, "pip", "install"]
    else:
        ensure_pip_install_config_settings_support()
        command = [sys.executable, "-m", "pip", "install"]

    command.extend(
        [
            "-e",
            ".[test]",
            "--config-settings=cmake.define.WOLFGANG_ENABLE_INTERNAL_BINDINGS=ON",
            "--config-settings=cmake.define.WOLFGANG_ENABLE_CUDA=OFF",
            "--config-settings=cmake.define.WOLFGANG_ENABLE_HIP=OFF",
            "--config-settings=cmake.define.WOLFGANG_ENABLE_METAL=ON",
        ]
    )
    run_check("Metal source build", command, env=env)


def run_metal_build_info_check(env: dict[str, str] | None = None) -> None:
    script = (
        "import wolfgang_quantum; "
        "import wolfgang_quantum._wolfgang_core as core; "
        "info = core._build_info(); "
        "status = core._metal_status(); "
        "print(info); "
        "print(status); "
        "assert info['metal_enabled'] is True; "
        "assert 'metal_capability_summary' in info; "
        "assert info['accelerator_build_mode'] == 'metal_only'; "
        "assert 'metal' in info['compiled_accelerator_backends']; "
        "assert 'metal' in info['compiled_backends']; "
        "assert status['built'] is True; "
        "assert status['runtime_available'] is True; "
        "assert status['device_count'] >= 1; "
        "assert status['devices']; "
        "assert status['metal_device_name']; "
        "assert status['macos_version']; "
        "assert status['xcode_or_clt_version']; "
        "assert status['storage_mode'] == 'MTLResourceStorageModeShared'"
    )
    run_check(
        "Metal build info, runtime status, and device metadata",
        [sys.executable, "-c", script],
        env=env,
    )


def run_editable_install_check() -> None:
    env = os.environ.copy()
    cmake_executable = cmake_executable_for_build_isolation()
    if cmake_executable is not None:
        env["CMAKE_EXECUTABLE"] = cmake_executable

    if command_succeeds([sys.executable, "-m", "pip", "--version"]):
        run_check(
            'editable install with test extra: python -m pip install -e ".[test]"',
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-e",
                ".[test]",
                "--config-settings=cmake.define.WOLFGANG_ENABLE_INTERNAL_BINDINGS=ON",
            ],
            env=env,
        )
        return

    uv = shutil.which("uv")
    if uv is not None:
        run_check(
            'editable install with test extra: uv pip install -e ".[test]"',
            [
                uv,
                "pip",
                "install",
                "-e",
                ".[test]",
                "--config-settings=cmake.define.WOLFGANG_ENABLE_INTERNAL_BINDINGS=ON",
            ],
            env=env,
        )
        return

    fail("Neither python -m pip nor uv is available for editable install validation")


def main() -> None:
    prepend_executable_directory_to_path(os.environ)

    check_source_docs_exist()
    check_markdown_links()
    check_stale_markers()
    check_cmake_defaults()
    check_native_source_layout()
    check_review_policy()
    run_check(
        "public artifact policy",
        [sys.executable, "scripts/audit_public_artifacts.py", "--tracked"],
    )
    run_check(
        "check release-readiness documentation",
        [sys.executable, "scripts/check_release_readiness.py"],
    )
    run_check(
        "check release wheelhouse readiness",
        [sys.executable, "scripts/check_release_wheelhouse.py"],
    )

    run_editable_install_check()
    run_cmake_configure_check()
    run_build_info_check()
    run_check("pytest", [sys.executable, "-m", "pytest"])
    run_check(
        "simplify benchmark smoke",
        [sys.executable, "benchmarks/bench_simplify.py", "--smoke", "--repeat", "1", "--json"],
    )
    run_check(
        "multiply benchmark smoke",
        [sys.executable, "benchmarks/bench_multiply.py", "--smoke", "--repeat", "1", "--json"],
    )
    run_check(
        "grouping benchmark smoke",
        [sys.executable, "benchmarks/bench_grouping.py", "--smoke", "--repeat", "1", "--json"],
    )
    run_check(
        "expectation benchmark smoke",
        [sys.executable, "benchmarks/bench_expectation.py", "--smoke", "--repeat", "1", "--json"],
    )
    run_check(
        "CUDA kernel benchmark smoke",
        [sys.executable, "benchmarks/bench_cuda_kernels.py", "--smoke", "--repeat", "1", "--json"],
    )
    run_check(
        "ROCm kernel benchmark smoke",
        [sys.executable, "benchmarks/bench_rocm_kernels.py", "--smoke", "--repeat", "1", "--json"],
    )
    run_check(
        "CPU dispatch benchmark smoke",
        [sys.executable, "benchmarks/bench_cpu_dispatch.py", "--smoke", "--repeat", "1", "--json"],
    )
    run_check(
        "CPU dispatch threshold benchmark smoke",
        [sys.executable, "benchmarks/bench_cpu_thresholds.py", "--smoke", "--repeat", "1", "--json"],
    )
    run_check(
        "CPU hardening benchmark smoke",
        [
            sys.executable,
            "benchmarks/bench_cpu_hardening.py",
            "--profile",
            "smoke",
            "--repeat",
            "1",
            "--json",
        ],
    )
    run_check(
        "competitive baseline benchmark smoke",
        [
            sys.executable,
            "benchmarks/bench_competitive_baselines.py",
            "--smoke",
            "--repeat",
            "1",
            "--json",
        ],
    )
    run_optional_openfermion_checks()
    if os.environ.get("WOLFGANG_VALIDATE_CUDA") == "1":
        run_cuda_validation_checks()
    if os.environ.get("WOLFGANG_VALIDATE_HIP") == "1":
        run_hip_validation_checks()
    if os.environ.get("WOLFGANG_VALIDATE_METAL") == "1":
        run_metal_validation_checks()
    run_check(
        "source distribution build smoke",
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--outdir",
            str(ROOT / "_skbuild" / "validate-dist"),
        ],
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_vendored_dlpack_header_has_license_and_pinned_version() -> None:
    header = ROOT / "third_party/dlpack/include/dlpack/dlpack.h"
    license_file = ROOT / "third_party/dlpack/LICENSE"
    assert header.is_file()
    assert license_file.is_file()
    text = header.read_text(encoding="utf-8")
    assert "#define DLPACK_MAJOR_VERSION 1" in text
    assert "#define DLPACK_MINOR_VERSION 1" in text
    assert "DLManagedTensorVersioned" in text


def test_dlpack_negotiation_compiles_for_lower_equal_and_future_versions(tmp_path: Path) -> None:
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")

    source = tmp_path / "dlpack_contract.cpp"
    source.write_text(
        r'''
#include "bindings/python/dlpack_interop.hpp"

using wolfgang::python::detail::kSupportedDLPackVersion;
using wolfgang::python::detail::negotiate_dlpack_version;

constexpr auto too_old = negotiate_dlpack_version(DLPackVersion{0, 8});
static_assert(!too_old.has_value());

constexpr auto minimum = negotiate_dlpack_version(DLPackVersion{1, 0});
static_assert(minimum.has_value());
static_assert(minimum->major == 1 && minimum->minor == 0);

constexpr auto equal = negotiate_dlpack_version(kSupportedDLPackVersion);
static_assert(equal.has_value());
static_assert(equal->major == kSupportedDLPackVersion.major);
static_assert(equal->minor == kSupportedDLPackVersion.minor);

constexpr auto future = negotiate_dlpack_version(DLPackVersion{99, 0});
static_assert(future.has_value());
static_assert(future->major == kSupportedDLPackVersion.major);
static_assert(future->minor == kSupportedDLPackVersion.minor);

int main() { return 0; }
''',
        encoding="utf-8",
    )
    output = tmp_path / "dlpack_contract"
    result = subprocess.run(
        [
            compiler,
            "-std=c++20",
            f"-I{ROOT}",
            f"-I{ROOT / 'third_party/dlpack/include'}",
            str(source),
            "-o",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    subprocess.run([str(output)], check=True)


def test_binding_uses_official_dlpack_types_not_handwritten_abi() -> None:
    source = (ROOT / "bindings/python/pauli_sum_py.cpp").read_text(encoding="utf-8")
    assert '#include "dlpack/dlpack.h"' in source
    assert "struct DLManagedTensorVersioned" not in source
    assert "DLPACK_FLAG_BITMASK_READ_ONLY" in source
    assert "managed->version = *negotiated" in source


def test_cuda_array_interface_checks_the_complete_allocation_extent() -> None:
    source = (ROOT / "src/cuda/expectation_cuda.cu").read_text(encoding="utf-8")
    assert "cuMemGetAddressRange" in source
    assert "required_bytes > allocation_bytes - byte_offset" in source
    assert "CUDA statevector byte range exceeds its backing allocation" in source


def test_cuda_array_interface_uses_driver_api_header_for_allocation_extent_query() -> None:
    source = (ROOT / "src/cuda/expectation_cuda.cu").read_text(encoding="utf-8")
    assert "#include <cuda.h>" in source


def test_cuda_array_interface_requires_exact_native_endian_complex_typestr() -> None:
    source = (ROOT / "bindings/python/pauli_sum_py.cpp").read_text(encoding="utf-8")
    assert 'typestr == std::string{native_order} + "c16"' in source
    assert 'typestr == std::string{native_order} + "c8"' in source
    assert 'typestr == "=c16"' in source
    assert 'typestr == "=c8"' in source
    assert "string_ends_with(typestr" not in source


def test_dlpack_capsule_construction_retains_raii_until_capsule_success() -> None:
    source = (ROOT / "bindings/python/pauli_sum_py.cpp").read_text(encoding="utf-8")
    assert "std::unique_ptr<std::int64_t[]> shape" in source
    assert "std::unique_ptr<std::int64_t[]> strides" in source
    assert "managed->manager_ctx = context.get()" in source
    assert "managed.get()," in source
    assert source.index("context.release();") > source.index("if (capsule == nullptr)")
    assert source.index("managed.release();") > source.index("if (capsule == nullptr)")

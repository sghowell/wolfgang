from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def normalized(path: str) -> str:
    return " ".join(read(path).split())


def test_cross_backend_docs_define_private_workspace_and_output_contract() -> None:
    backend_neutral = normalized("docs/architecture/backend_neutral_accelerators.md")
    cuda = normalized("docs/architecture/cuda_backend.md")
    rocm = normalized("docs/architecture/rocm_backend.md")
    apple = normalized("docs/architecture/apple_accelerator.md")

    for text in (backend_neutral, cuda, rocm, apple):
        assert "private reusable accelerator scratch and output buffers" in text
        assert "move-only" in text
        assert "same backend-local device ordinal" in text
        assert "reset retains the allocation for reuse" in text
        assert "release returns the allocation to the runtime" in text
        assert "must not expose raw device pointers or framework objects through the public API" in text


def test_backend_workspace_headers_share_private_contract_helpers() -> None:
    cuda_header = read("src/cuda/workspace.cuh")
    hip_header = read("src/hip/workspace_hip.hip.hpp")
    metal_header = read("src/metal/workspace_metal.hpp")

    for header in (cuda_header, hip_header, metal_header):
        assert '#include "detail/accelerator_host_helpers.hpp"' in header
        assert "using detail::WorkspaceSnapshot;" in header
        assert "using detail::WorkspaceTimingMode;" in header
        assert "using detail::workspace_timing_mode_name;" in header

    assert "struct WorkspaceSnapshot" not in cuda_header
    assert "struct WorkspaceSnapshot" not in hip_header
    assert "struct WorkspaceSnapshot" not in metal_header
    assert "enum class WorkspaceTimingMode" not in cuda_header
    assert "enum class WorkspaceTimingMode" not in hip_header
    assert "enum class WorkspaceTimingMode" not in metal_header
    assert "[[nodiscard]] const char* workspace_timing_mode_name(WorkspaceTimingMode mode) noexcept;" not in cuda_header
    assert "[[nodiscard]] const char* workspace_timing_mode_name(WorkspaceTimingMode mode) noexcept;" not in metal_header

    for token in (
        "int device_ordinal() const noexcept",
        "void ensure_device(int operand_device_ordinal) const",
        "void reset() noexcept",
        "WorkspaceSnapshot snapshot(WorkspaceTimingMode mode) const noexcept",
    ):
        assert token in hip_header


def test_metal_workspace_header_compiles_with_shared_helper_import(tmp_path: Path) -> None:
    if sys.platform != "darwin":
        pytest.skip("Metal header smoke coverage requires macOS")

    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("clang++ is required for Metal header smoke coverage")
    assert compiler is not None

    source = tmp_path / "metal_header_smoke.mm"
    source.write_text(
        r'''
#include "metal/workspace_metal.hpp"

int main() {
  using wolfgang::metal_detail::WorkspaceTimingMode;
  using wolfgang::metal_detail::workspace_timing_mode_name;

  return workspace_timing_mode_name(WorkspaceTimingMode::kAbsent) == nullptr;
}
''',
        encoding="utf-8",
    )

    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-fobjc-arc",
            "-fsyntax-only",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT}",
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr


def test_accelerator_host_helpers_compile_shared_workspace_contracts(tmp_path: Path) -> None:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for native helper contract coverage")
    assert compiler is not None

    source = tmp_path / "accelerator_workspace_contract_test.cpp"
    executable = tmp_path / "accelerator_workspace_contract_test"
    source.write_text(
        r'''
#include "detail/accelerator_host_helpers.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

int main() {
  using wolfgang::detail::WorkspaceSnapshot;
  using wolfgang::detail::WorkspaceTimingMode;
  using wolfgang::detail::ensure_workspace_device_match;
  using wolfgang::detail::validate_workspace_device_ordinal;
  using wolfgang::detail::workspace_timing_mode_name;

  validate_workspace_device_ordinal("HIP", 0);
  try {
    validate_workspace_device_ordinal("Metal", -1);
    return 1;
  } catch (const std::invalid_argument& error) {
    if (std::string(error.what()) !=
        "Metal reusable workspace device ordinal must be non-negative") {
      return 2;
    }
  }

  ensure_workspace_device_match("CUDA", 3, 3);
  try {
    ensure_workspace_device_match("HIP", 2, 4);
    return 3;
  } catch (const std::invalid_argument& error) {
    if (std::string(error.what()) !=
        "HIP reusable workspace device mismatch: workspace device ordinal 2, operand device ordinal 4") {
      return 4;
    }
  }

  WorkspaceSnapshot snapshot{};
  snapshot.device_ordinal = 7;
  snapshot.reserved_bytes = 4096;
  snapshot.high_watermark_bytes = 8192;
  snapshot.allocation_count = 1;
  snapshot.growth_count = 2;
  snapshot.timing_mode = workspace_timing_mode_name(WorkspaceTimingMode::kGrowInsideTiming);
  if (snapshot.device_ordinal != 7 || snapshot.reserved_bytes != 4096 ||
      snapshot.high_watermark_bytes != 8192 || snapshot.allocation_count != 1 ||
      snapshot.growth_count != 2) {
    return 5;
  }
  if (std::string(snapshot.timing_mode) != "grow_inside_timing") {
    return 6;
  }
  if (std::string(workspace_timing_mode_name(WorkspaceTimingMode::kAbsent)) != "absent") {
    return 7;
  }
  if (std::string(workspace_timing_mode_name(WorkspaceTimingMode::kPreReservedOutsideTiming)) !=
      "pre_reserved_outside_timing") {
    return 8;
  }
  return 0;
}
''',
        encoding="utf-8",
    )

    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{ROOT / 'src'}",
            str(source),
            "-o",
            str(executable),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    subprocess.run([str(executable)], check=True)

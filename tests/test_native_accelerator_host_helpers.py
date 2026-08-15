"""CPU-only compile coverage for native accelerator host helpers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_accelerator_host_validation_helpers_compile_and_preserve_checked_contracts(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for native helper contract coverage")

    source = tmp_path / "accelerator_host_helpers_test.cpp"
    executable = tmp_path / "accelerator_host_helpers_test"
    source.write_text(
        r'''
#include "detail/accelerator_host_helpers.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

int main() {
  using wolfgang::detail::checked_bytes;
  using wolfgang::detail::expected_statevector_length;
  using wolfgang::detail::validate_statevector_length;

  if (checked_bytes(4, sizeof(double), "device values") != 4 * sizeof(double)) {
    return 1;
  }

  try {
    (void)checked_bytes(std::numeric_limits<std::size_t>::max(), 2, "device values");
    return 2;
  } catch (const std::invalid_argument& error) {
    if (std::string(error.what()) != "device values size overflows size_t") {
      return 3;
    }
  }

  if (expected_statevector_length(3) != 8) {
    return 4;
  }
  validate_statevector_length(3, 8);

  try {
    (void)expected_statevector_length(64);
    return 5;
  } catch (const std::invalid_argument& error) {
    if (std::string(error.what()) !=
        "expectation_statevector requires num_qubits <= 63") {
      return 6;
    }
  }

  try {
    validate_statevector_length(3, 7);
    return 7;
  } catch (const std::invalid_argument& error) {
    if (std::string(error.what()) !=
        "expectation_statevector requires len(psi) == 2 ** num_qubits") {
      return 8;
    }
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


def test_cuda_and_hip_headers_share_host_validation_implementation() -> None:
    backend_headers = [
        ROOT / "src/cuda/device_pauli_sum.cuh",
        ROOT / "src/hip/device_pauli_sum.hip.hpp",
    ]

    for header in backend_headers:
        source = header.read_text(encoding="utf-8")
        assert '#include "detail/accelerator_host_helpers.hpp"' in source
        assert "using detail::checked_bytes;" in source
        assert "using detail::expected_statevector_length;" in source
        assert "using detail::validate_statevector_length;" in source
        assert "inline std::size_t checked_bytes(" not in source
        assert "inline std::size_t expected_statevector_length(" not in source
        assert "inline void validate_statevector_length(" not in source

#pragma once

#include "detail/checked_arithmetic.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace wolfgang::detail {

constexpr std::size_t kMaxAcceleratorStatevectorQubits = 63;

enum class WorkspaceTimingMode {
  kAbsent,
  kGrowInsideTiming,
  kPreReservedOutsideTiming,
};

struct WorkspaceSnapshot {
  int device_ordinal = -1;
  std::size_t reserved_bytes = 0;
  std::size_t high_watermark_bytes = 0;
  std::size_t allocation_count = 0;
  std::size_t growth_count = 0;
  const char* timing_mode = "absent";
};

inline std::size_t checked_bytes(
    std::size_t count,
    std::size_t element_size,
    const char* name) {
  return checked_product(count, element_size, name);
}

inline constexpr const char* workspace_timing_mode_name(WorkspaceTimingMode mode) noexcept {
  switch (mode) {
    case WorkspaceTimingMode::kAbsent:
      return "absent";
    case WorkspaceTimingMode::kGrowInsideTiming:
      return "grow_inside_timing";
    case WorkspaceTimingMode::kPreReservedOutsideTiming:
      return "pre_reserved_outside_timing";
  }
  return "absent";
}

inline void validate_workspace_device_ordinal(const char* backend_name, int device_ordinal) {
  if (device_ordinal < 0) {
    throw std::invalid_argument(
        std::string(backend_name) + " reusable workspace device ordinal must be non-negative");
  }
}

inline void ensure_workspace_device_match(
    const char* backend_name,
    int workspace_device_ordinal,
    int operand_device_ordinal) {
  if (workspace_device_ordinal != operand_device_ordinal) {
    throw std::invalid_argument(
        std::string(backend_name) +
        " reusable workspace device mismatch: workspace device ordinal " +
        std::to_string(workspace_device_ordinal) + ", operand device ordinal " +
        std::to_string(operand_device_ordinal));
  }
}

inline std::size_t expected_statevector_length(std::size_t num_qubits) {
  if (num_qubits > kMaxAcceleratorStatevectorQubits) {
    throw std::invalid_argument("expectation_statevector requires num_qubits <= 63");
  }
  return std::size_t{1} << num_qubits;
}

inline void validate_statevector_length(std::size_t num_qubits, std::size_t actual_size) {
  const std::size_t expected_size = expected_statevector_length(num_qubits);
  if (actual_size != expected_size) {
    throw std::invalid_argument(
        "expectation_statevector requires len(psi) == 2 ** num_qubits");
  }
}

}  // namespace wolfgang::detail

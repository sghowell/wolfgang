#pragma once

#include "detail/checked_arithmetic.hpp"

#include <cstddef>
#include <stdexcept>

namespace wolfgang::detail {

constexpr std::size_t kMaxAcceleratorStatevectorQubits = 63;

inline std::size_t checked_bytes(
    std::size_t count,
    std::size_t element_size,
    const char* name) {
  return checked_product(count, element_size, name);
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

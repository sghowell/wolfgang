#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace wolfgang::detail {

inline std::size_t checked_word_count(std::size_t num_qubits) {
  if (num_qubits == 0) {
    return 0;
  }
  if (num_qubits > std::numeric_limits<std::size_t>::max() - 63) {
    throw std::invalid_argument("num_qubits is too large");
  }
  return (num_qubits + 63) / 64;
}

inline std::size_t checked_product(std::size_t lhs, std::size_t rhs, const char* field_name) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::invalid_argument(std::string(field_name) + " size overflows size_t");
  }
  return lhs * rhs;
}

inline std::size_t checked_sum(std::size_t lhs, std::size_t rhs, const char* field_name) {
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
    throw std::invalid_argument(std::string(field_name) + " size overflows size_t");
  }
  return lhs + rhs;
}

inline std::size_t checked_matmul_intermediate_terms(
    std::size_t lhs_terms,
    std::size_t rhs_terms,
    std::size_t max_intermediate_terms) {
  if (lhs_terms != 0 && rhs_terms > std::numeric_limits<std::size_t>::max() / lhs_terms) {
    throw std::invalid_argument("matmul intermediate term count overflows size_t");
  }

  const std::size_t intermediate_terms = lhs_terms * rhs_terms;
  if (intermediate_terms > max_intermediate_terms) {
    throw std::invalid_argument("matmul intermediate term count exceeds max_intermediate_terms");
  }
  return intermediate_terms;
}

inline std::size_t checked_commutation_matrix_entries(
    std::size_t lhs_terms,
    std::size_t rhs_terms,
    std::size_t max_commutation_matrix_entries) {
  if (lhs_terms != 0 && rhs_terms > std::numeric_limits<std::size_t>::max() / lhs_terms) {
    throw std::invalid_argument("commutation matrix entry count overflows size_t");
  }

  const std::size_t entries = lhs_terms * rhs_terms;
  if (entries > max_commutation_matrix_entries) {
    throw std::invalid_argument(
        "commutation matrix entry count exceeds max_commutation_matrix_entries");
  }
  return entries;
}

}  // namespace wolfgang::detail

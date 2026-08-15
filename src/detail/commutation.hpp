#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace wolfgang::detail {

inline bool terms_commute(
    const std::vector<std::uint64_t>& lhs_x,
    const std::vector<std::uint64_t>& lhs_z,
    const std::vector<std::uint64_t>& rhs_x,
    const std::vector<std::uint64_t>& rhs_z,
    std::size_t words,
    std::size_t lhs_term,
    std::size_t rhs_term) noexcept {
  const std::size_t lhs_offset = lhs_term * words;
  const std::size_t rhs_offset = rhs_term * words;

  if (words == 1) {
    const std::uint64_t conflicts =
        (lhs_x[lhs_offset] & rhs_z[rhs_offset]) ^
        (lhs_z[lhs_offset] & rhs_x[rhs_offset]);
    return (std::popcount(conflicts) & 1U) == 0;
  }

  if (words == 2) {
    const std::uint64_t conflicts0 =
        (lhs_x[lhs_offset] & rhs_z[rhs_offset]) ^
        (lhs_z[lhs_offset] & rhs_x[rhs_offset]);
    const std::uint64_t conflicts1 =
        (lhs_x[lhs_offset + 1] & rhs_z[rhs_offset + 1]) ^
        (lhs_z[lhs_offset + 1] & rhs_x[rhs_offset + 1]);
    return ((std::popcount(conflicts0) + std::popcount(conflicts1)) & 1U) == 0;
  }

  bool odd_parity = false;
  for (std::size_t word = 0; word < words; ++word) {
    const std::uint64_t conflicts =
        (lhs_x[lhs_offset + word] & rhs_z[rhs_offset + word]) ^
        (lhs_z[lhs_offset + word] & rhs_x[rhs_offset + word]);
    odd_parity ^= (std::popcount(conflicts) & 1U) != 0;
  }
  return !odd_parity;
}

inline bool terms_qwc_compatible(
    const std::vector<std::uint64_t>& lhs_x,
    const std::vector<std::uint64_t>& lhs_z,
    const std::vector<std::uint64_t>& rhs_x,
    const std::vector<std::uint64_t>& rhs_z,
    std::size_t words,
    std::size_t lhs_term,
    std::size_t rhs_term) noexcept {
  const std::size_t lhs_offset = lhs_term * words;
  const std::size_t rhs_offset = rhs_term * words;
  for (std::size_t word = 0; word < words; ++word) {
    const std::uint64_t conflicts =
        (lhs_x[lhs_offset + word] & rhs_z[rhs_offset + word]) ^
        (lhs_z[lhs_offset + word] & rhs_x[rhs_offset + word]);
    if (conflicts != 0) {
      return false;
    }
  }
  return true;
}

}  // namespace wolfgang::detail

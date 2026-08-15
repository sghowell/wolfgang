#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace wolfgang::detail {

inline bool is_valid_pauli(char pauli) noexcept {
  return pauli == 'I' || pauli == 'X' || pauli == 'Y' || pauli == 'Z';
}

inline void set_pauli_bit(
    std::vector<std::uint64_t>& words,
    std::size_t words_per_term,
    std::size_t term,
    std::size_t qubit) {
  words[term * words_per_term + qubit / 64] |= std::uint64_t{1} << (qubit % 64);
}

inline char pauli_from_bits(bool x_bit, bool z_bit) noexcept {
  if (x_bit && z_bit) {
    return 'Y';
  }
  if (x_bit) {
    return 'X';
  }
  if (z_bit) {
    return 'Z';
  }
  return 'I';
}

inline std::size_t term_weight(
    const std::vector<std::uint64_t>& x,
    const std::vector<std::uint64_t>& z,
    std::size_t words,
    std::size_t term) noexcept {
  std::size_t weight = 0;
  const std::size_t offset = term * words;
  for (std::size_t word = 0; word < words; ++word) {
    weight += std::popcount(x[offset + word] | z[offset + word]);
  }
  return weight;
}

}  // namespace wolfgang::detail

#include "detail/commute_kernels.hpp"

#include <immintrin.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace wolfgang::detail {
namespace {

long long intrinsic_word(std::uint64_t word) noexcept {
  return std::bit_cast<long long>(word);
}

const std::array<std::array<std::uint8_t, 8>, 256>& expanded_byte_mask_table() noexcept {
  static const auto table = []() {
    std::array<std::array<std::uint8_t, 8>, 256> values{};
    for (std::size_t mask = 0; mask < values.size(); ++mask) {
      for (std::size_t lane = 0; lane < values[mask].size(); ++lane) {
        values[mask][lane] = static_cast<std::uint8_t>((mask >> lane) & 1U);
      }
    }
    return values;
  }();
  return table;
}

void store_commutation_octet(
    __m512i counts,
    std::vector<std::uint8_t>& out,
    std::size_t output_index) noexcept {
  const __mmask8 odd_mask = _mm512_test_epi64_mask(counts, _mm512_set1_epi64(1));
  const auto& expanded =
      expanded_byte_mask_table()[static_cast<std::uint8_t>((~odd_mask) & 0xffU)];
  std::memcpy(out.data() + output_index, expanded.data(), expanded.size());
}

}  // namespace

std::vector<std::uint8_t> commutes_with_avx512(
    const PauliSum& lhs,
    const PauliSum& rhs,
    std::size_t entries) {
  require_simd_commutation_words("avx512", lhs.words());

  std::vector<std::uint8_t> out(entries, 0);
  const std::vector<std::uint64_t>& lhs_x_words = lhs.x_words();
  const std::vector<std::uint64_t>& lhs_z_words = lhs.z_words();
  const std::vector<std::uint64_t>& rhs_x_words = rhs.x_words();
  const std::vector<std::uint64_t>& rhs_z_words = rhs.z_words();

  if (lhs.words() == 1) {
    for (std::size_t lhs_term = 0; lhs_term < lhs.num_terms(); ++lhs_term) {
      const __m512i lhs_x = _mm512_set1_epi64(intrinsic_word(lhs_x_words[lhs_term]));
      const __m512i lhs_z = _mm512_set1_epi64(intrinsic_word(lhs_z_words[lhs_term]));
      const std::size_t row_offset = lhs_term * rhs.num_terms();
      std::size_t rhs_term = 0;
      for (; rhs_term + 8 <= rhs.num_terms(); rhs_term += 8) {
        const __m512i rhs_z =
            _mm512_loadu_si512(reinterpret_cast<const void*>(rhs_z_words.data() + rhs_term));
        const __m512i rhs_x =
            _mm512_loadu_si512(reinterpret_cast<const void*>(rhs_x_words.data() + rhs_term));
        const __m512i conflicts =
            _mm512_xor_si512(_mm512_and_si512(lhs_x, rhs_z), _mm512_and_si512(lhs_z, rhs_x));
        store_commutation_octet(_mm512_popcnt_epi64(conflicts), out, row_offset + rhs_term);
      }
      for (; rhs_term < rhs.num_terms(); ++rhs_term) {
        const std::uint64_t conflicts =
            (lhs_x_words[lhs_term] & rhs_z_words[rhs_term]) ^
            (lhs_z_words[lhs_term] & rhs_x_words[rhs_term]);
        out[row_offset + rhs_term] = (std::popcount(conflicts) & 1U) == 0 ? 1 : 0;
      }
    }
    return out;
  }

  for (std::size_t lhs_term = 0; lhs_term < lhs.num_terms(); ++lhs_term) {
    const std::size_t lhs_offset = lhs_term * 2;
    const __m512i lhs_x0 = _mm512_set1_epi64(intrinsic_word(lhs_x_words[lhs_offset]));
    const __m512i lhs_z0 = _mm512_set1_epi64(intrinsic_word(lhs_z_words[lhs_offset]));
    const __m512i lhs_x1 = _mm512_set1_epi64(intrinsic_word(lhs_x_words[lhs_offset + 1]));
    const __m512i lhs_z1 = _mm512_set1_epi64(intrinsic_word(lhs_z_words[lhs_offset + 1]));
    const std::size_t row_offset = lhs_term * rhs.num_terms();
    std::size_t rhs_term = 0;
    for (; rhs_term + 8 <= rhs.num_terms(); rhs_term += 8) {
      const std::size_t rhs_offset = rhs_term * 2;
      const __m512i rhs_z0 = _mm512_set_epi64(
          intrinsic_word(rhs_z_words[rhs_offset + 14]),
          intrinsic_word(rhs_z_words[rhs_offset + 12]),
          intrinsic_word(rhs_z_words[rhs_offset + 10]),
          intrinsic_word(rhs_z_words[rhs_offset + 8]),
          intrinsic_word(rhs_z_words[rhs_offset + 6]),
          intrinsic_word(rhs_z_words[rhs_offset + 4]),
          intrinsic_word(rhs_z_words[rhs_offset + 2]),
          intrinsic_word(rhs_z_words[rhs_offset]));
      const __m512i rhs_x0 = _mm512_set_epi64(
          intrinsic_word(rhs_x_words[rhs_offset + 14]),
          intrinsic_word(rhs_x_words[rhs_offset + 12]),
          intrinsic_word(rhs_x_words[rhs_offset + 10]),
          intrinsic_word(rhs_x_words[rhs_offset + 8]),
          intrinsic_word(rhs_x_words[rhs_offset + 6]),
          intrinsic_word(rhs_x_words[rhs_offset + 4]),
          intrinsic_word(rhs_x_words[rhs_offset + 2]),
          intrinsic_word(rhs_x_words[rhs_offset]));
      const __m512i rhs_z1 = _mm512_set_epi64(
          intrinsic_word(rhs_z_words[rhs_offset + 15]),
          intrinsic_word(rhs_z_words[rhs_offset + 13]),
          intrinsic_word(rhs_z_words[rhs_offset + 11]),
          intrinsic_word(rhs_z_words[rhs_offset + 9]),
          intrinsic_word(rhs_z_words[rhs_offset + 7]),
          intrinsic_word(rhs_z_words[rhs_offset + 5]),
          intrinsic_word(rhs_z_words[rhs_offset + 3]),
          intrinsic_word(rhs_z_words[rhs_offset + 1]));
      const __m512i rhs_x1 = _mm512_set_epi64(
          intrinsic_word(rhs_x_words[rhs_offset + 15]),
          intrinsic_word(rhs_x_words[rhs_offset + 13]),
          intrinsic_word(rhs_x_words[rhs_offset + 11]),
          intrinsic_word(rhs_x_words[rhs_offset + 9]),
          intrinsic_word(rhs_x_words[rhs_offset + 7]),
          intrinsic_word(rhs_x_words[rhs_offset + 5]),
          intrinsic_word(rhs_x_words[rhs_offset + 3]),
          intrinsic_word(rhs_x_words[rhs_offset + 1]));
      const __m512i conflicts0 =
          _mm512_xor_si512(_mm512_and_si512(lhs_x0, rhs_z0), _mm512_and_si512(lhs_z0, rhs_x0));
      const __m512i conflicts1 =
          _mm512_xor_si512(_mm512_and_si512(lhs_x1, rhs_z1), _mm512_and_si512(lhs_z1, rhs_x1));
      store_commutation_octet(
          _mm512_add_epi64(_mm512_popcnt_epi64(conflicts0), _mm512_popcnt_epi64(conflicts1)),
          out,
          row_offset + rhs_term);
    }
    for (; rhs_term < rhs.num_terms(); ++rhs_term) {
      const std::size_t rhs_offset = rhs_term * 2;
      const std::uint64_t conflicts0 =
          (lhs_x_words[lhs_offset] & rhs_z_words[rhs_offset]) ^
          (lhs_z_words[lhs_offset] & rhs_x_words[rhs_offset]);
      const std::uint64_t conflicts1 =
          (lhs_x_words[lhs_offset + 1] & rhs_z_words[rhs_offset + 1]) ^
          (lhs_z_words[lhs_offset + 1] & rhs_x_words[rhs_offset + 1]);
      out[row_offset + rhs_term] =
          ((std::popcount(conflicts0) + std::popcount(conflicts1)) & 1U) == 0 ? 1 : 0;
    }
  }
  return out;
}

}  // namespace wolfgang::detail

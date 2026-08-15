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

const std::array<std::array<std::uint8_t, 4>, 16>& expanded_nibble_mask_table() noexcept {
  static const auto table = []() {
    std::array<std::array<std::uint8_t, 4>, 16> values{};
    for (std::size_t mask = 0; mask < values.size(); ++mask) {
      for (std::size_t lane = 0; lane < values[mask].size(); ++lane) {
        values[mask][lane] = static_cast<std::uint8_t>((mask >> lane) & 1U);
      }
    }
    return values;
  }();
  return table;
}

// AVX2 has no native 64-bit popcount. Count per nibble with pshufb, then
// horizontally sum each 64-bit lane with sad_epu8.
__m256i popcount64(__m256i values) noexcept {
  const __m256i lookup = _mm256_setr_epi8(
      0,
      1,
      1,
      2,
      1,
      2,
      2,
      3,
      1,
      2,
      2,
      3,
      2,
      3,
      3,
      4,
      0,
      1,
      1,
      2,
      1,
      2,
      2,
      3,
      1,
      2,
      2,
      3,
      2,
      3,
      3,
      4);
  const __m256i low_mask = _mm256_set1_epi8(0x0f);
  const __m256i low = _mm256_and_si256(values, low_mask);
  const __m256i high = _mm256_and_si256(_mm256_srli_epi16(values, 4), low_mask);
  const __m256i byte_counts =
      _mm256_add_epi8(_mm256_shuffle_epi8(lookup, low), _mm256_shuffle_epi8(lookup, high));
  return _mm256_sad_epu8(byte_counts, _mm256_setzero_si256());
}

void store_commutation_quad(
    __m256i counts,
    std::vector<std::uint8_t>& out,
    std::size_t output_index) noexcept {
  const __m256i even = _mm256_cmpeq_epi64(
      _mm256_and_si256(counts, _mm256_set1_epi64x(1)),
      _mm256_setzero_si256());
  const auto& expanded =
      expanded_nibble_mask_table()[static_cast<std::uint8_t>(_mm256_movemask_pd(
          _mm256_castsi256_pd(even)))];
  std::memcpy(out.data() + output_index, expanded.data(), expanded.size());
}

}  // namespace

std::vector<std::uint8_t> commutes_with_avx2(
    const PauliSum& lhs,
    const PauliSum& rhs,
    std::size_t entries) {
  require_simd_commutation_words("avx2", lhs.words());

  std::vector<std::uint8_t> out(entries, 0);
  const std::vector<std::uint64_t>& lhs_x_words = lhs.x_words();
  const std::vector<std::uint64_t>& lhs_z_words = lhs.z_words();
  const std::vector<std::uint64_t>& rhs_x_words = rhs.x_words();
  const std::vector<std::uint64_t>& rhs_z_words = rhs.z_words();

  if (lhs.words() == 1) {
    for (std::size_t lhs_term = 0; lhs_term < lhs.num_terms(); ++lhs_term) {
      const __m256i lhs_x = _mm256_set1_epi64x(intrinsic_word(lhs_x_words[lhs_term]));
      const __m256i lhs_z = _mm256_set1_epi64x(intrinsic_word(lhs_z_words[lhs_term]));
      const std::size_t row_offset = lhs_term * rhs.num_terms();
      std::size_t rhs_term = 0;
      for (; rhs_term + 4 <= rhs.num_terms(); rhs_term += 4) {
        const __m256i rhs_z = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(rhs_z_words.data() + rhs_term));
        const __m256i rhs_x = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(rhs_x_words.data() + rhs_term));
        const __m256i conflicts =
            _mm256_xor_si256(_mm256_and_si256(lhs_x, rhs_z), _mm256_and_si256(lhs_z, rhs_x));
        store_commutation_quad(popcount64(conflicts), out, row_offset + rhs_term);
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
    const __m256i lhs_x0 = _mm256_set1_epi64x(intrinsic_word(lhs_x_words[lhs_offset]));
    const __m256i lhs_z0 = _mm256_set1_epi64x(intrinsic_word(lhs_z_words[lhs_offset]));
    const __m256i lhs_x1 = _mm256_set1_epi64x(intrinsic_word(lhs_x_words[lhs_offset + 1]));
    const __m256i lhs_z1 = _mm256_set1_epi64x(intrinsic_word(lhs_z_words[lhs_offset + 1]));
    const std::size_t row_offset = lhs_term * rhs.num_terms();
    std::size_t rhs_term = 0;
    for (; rhs_term + 4 <= rhs.num_terms(); rhs_term += 4) {
      const std::size_t rhs_offset = rhs_term * 2;
      const __m256i rhs_z0 = _mm256_set_epi64x(
          intrinsic_word(rhs_z_words[rhs_offset + 6]),
          intrinsic_word(rhs_z_words[rhs_offset + 4]),
          intrinsic_word(rhs_z_words[rhs_offset + 2]),
          intrinsic_word(rhs_z_words[rhs_offset]));
      const __m256i rhs_x0 = _mm256_set_epi64x(
          intrinsic_word(rhs_x_words[rhs_offset + 6]),
          intrinsic_word(rhs_x_words[rhs_offset + 4]),
          intrinsic_word(rhs_x_words[rhs_offset + 2]),
          intrinsic_word(rhs_x_words[rhs_offset]));
      const __m256i rhs_z1 = _mm256_set_epi64x(
          intrinsic_word(rhs_z_words[rhs_offset + 7]),
          intrinsic_word(rhs_z_words[rhs_offset + 5]),
          intrinsic_word(rhs_z_words[rhs_offset + 3]),
          intrinsic_word(rhs_z_words[rhs_offset + 1]));
      const __m256i rhs_x1 = _mm256_set_epi64x(
          intrinsic_word(rhs_x_words[rhs_offset + 7]),
          intrinsic_word(rhs_x_words[rhs_offset + 5]),
          intrinsic_word(rhs_x_words[rhs_offset + 3]),
          intrinsic_word(rhs_x_words[rhs_offset + 1]));
      const __m256i conflicts0 =
          _mm256_xor_si256(_mm256_and_si256(lhs_x0, rhs_z0), _mm256_and_si256(lhs_z0, rhs_x0));
      const __m256i conflicts1 =
          _mm256_xor_si256(_mm256_and_si256(lhs_x1, rhs_z1), _mm256_and_si256(lhs_z1, rhs_x1));
      store_commutation_quad(
          _mm256_add_epi64(popcount64(conflicts0), popcount64(conflicts1)),
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

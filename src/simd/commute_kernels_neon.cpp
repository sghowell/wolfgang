#include "detail/commute_kernels.hpp"

#include <arm_neon.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace wolfgang::detail {
namespace {

uint64x2_t popcount64(uint64x2_t values) noexcept {
  const uint8x16_t byte_counts = vcntq_u8(vreinterpretq_u8_u64(values));
  const uint16x8_t sum16 = vpaddlq_u8(byte_counts);
  const uint32x4_t sum32 = vpaddlq_u16(sum16);
  return vpaddlq_u32(sum32);
}

void store_commutation_pair(
    uint64x2_t counts,
    std::vector<std::uint8_t>& out,
    std::size_t output_index) noexcept {
  alignas(16) std::uint64_t lanes[2];
  vst1q_u64(lanes, counts);
  out[output_index] = (lanes[0] & 1U) == 0 ? 1 : 0;
  out[output_index + 1] = (lanes[1] & 1U) == 0 ? 1 : 0;
}

}  // namespace

std::vector<std::uint8_t> commutes_with_neon(
    const PauliSum& lhs,
    const PauliSum& rhs,
    std::size_t entries) {
  require_simd_commutation_words("neon", lhs.words());

  std::vector<std::uint8_t> out(entries, 0);
  const std::vector<std::uint64_t>& lhs_x_words = lhs.x_words();
  const std::vector<std::uint64_t>& lhs_z_words = lhs.z_words();
  const std::vector<std::uint64_t>& rhs_x_words = rhs.x_words();
  const std::vector<std::uint64_t>& rhs_z_words = rhs.z_words();

  if (lhs.words() == 1) {
    for (std::size_t lhs_term = 0; lhs_term < lhs.num_terms(); ++lhs_term) {
      const uint64x2_t lhs_x = vdupq_n_u64(lhs_x_words[lhs_term]);
      const uint64x2_t lhs_z = vdupq_n_u64(lhs_z_words[lhs_term]);
      const std::size_t row_offset = lhs_term * rhs.num_terms();
      std::size_t rhs_term = 0;
      for (; rhs_term + 2 <= rhs.num_terms(); rhs_term += 2) {
        const uint64x2_t rhs_z = vld1q_u64(rhs_z_words.data() + rhs_term);
        const uint64x2_t rhs_x = vld1q_u64(rhs_x_words.data() + rhs_term);
        const uint64x2_t conflicts =
            veorq_u64(vandq_u64(lhs_x, rhs_z), vandq_u64(lhs_z, rhs_x));
        store_commutation_pair(popcount64(conflicts), out, row_offset + rhs_term);
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
    const uint64x2_t lhs_x0 = vdupq_n_u64(lhs_x_words[lhs_offset]);
    const uint64x2_t lhs_z0 = vdupq_n_u64(lhs_z_words[lhs_offset]);
    const uint64x2_t lhs_x1 = vdupq_n_u64(lhs_x_words[lhs_offset + 1]);
    const uint64x2_t lhs_z1 = vdupq_n_u64(lhs_z_words[lhs_offset + 1]);
    const std::size_t row_offset = lhs_term * rhs.num_terms();
    std::size_t rhs_term = 0;
    for (; rhs_term + 2 <= rhs.num_terms(); rhs_term += 2) {
      const std::size_t rhs_offset = rhs_term * 2;
      uint64x2_t rhs_z0 = vdupq_n_u64(0);
      uint64x2_t rhs_x0 = vdupq_n_u64(0);
      uint64x2_t rhs_z1 = vdupq_n_u64(0);
      uint64x2_t rhs_x1 = vdupq_n_u64(0);
      rhs_z0 = vsetq_lane_u64(rhs_z_words[rhs_offset], rhs_z0, 0);
      rhs_z0 = vsetq_lane_u64(rhs_z_words[rhs_offset + 2], rhs_z0, 1);
      rhs_x0 = vsetq_lane_u64(rhs_x_words[rhs_offset], rhs_x0, 0);
      rhs_x0 = vsetq_lane_u64(rhs_x_words[rhs_offset + 2], rhs_x0, 1);
      rhs_z1 = vsetq_lane_u64(rhs_z_words[rhs_offset + 1], rhs_z1, 0);
      rhs_z1 = vsetq_lane_u64(rhs_z_words[rhs_offset + 3], rhs_z1, 1);
      rhs_x1 = vsetq_lane_u64(rhs_x_words[rhs_offset + 1], rhs_x1, 0);
      rhs_x1 = vsetq_lane_u64(rhs_x_words[rhs_offset + 3], rhs_x1, 1);

      const uint64x2_t conflicts0 =
          veorq_u64(vandq_u64(lhs_x0, rhs_z0), vandq_u64(lhs_z0, rhs_x0));
      const uint64x2_t conflicts1 =
          veorq_u64(vandq_u64(lhs_x1, rhs_z1), vandq_u64(lhs_z1, rhs_x1));
      store_commutation_pair(
          vaddq_u64(popcount64(conflicts0), popcount64(conflicts1)),
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

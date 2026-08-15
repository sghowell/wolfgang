#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace wolfgang::detail {

inline int compare_word(std::uint64_t lhs, std::uint64_t rhs) noexcept {
  if (lhs < rhs) {
    return -1;
  }
  if (lhs > rhs) {
    return 1;
  }
  return 0;
}

struct PackedKey1 {
  std::uint64_t x;
  std::uint64_t z;

  bool operator==(const PackedKey1& rhs) const noexcept {
    return x == rhs.x && z == rhs.z;
  }
};

struct PackedKey2 {
  std::uint64_t x0;
  std::uint64_t z0;
  std::uint64_t x1;
  std::uint64_t z1;

  bool operator==(const PackedKey2& rhs) const noexcept {
    return x0 == rhs.x0 && z0 == rhs.z0 && x1 == rhs.x1 && z1 == rhs.z1;
  }
};

inline std::size_t mix_uint64(std::size_t seed, std::uint64_t value) noexcept {
  seed ^= static_cast<std::size_t>(
      value + 0x9e3779b97f4a7c15ULL + (static_cast<std::uint64_t>(seed) << 6U) +
      (static_cast<std::uint64_t>(seed) >> 2U));
  return seed;
}

struct PackedKey1Hash {
  std::size_t operator()(const PackedKey1& key) const noexcept {
    std::size_t seed = 0;
    seed = mix_uint64(seed, key.x);
    seed = mix_uint64(seed, key.z);
    return seed;
  }
};

struct PackedKey2Hash {
  std::size_t operator()(const PackedKey2& key) const noexcept {
    std::size_t seed = 0;
    seed = mix_uint64(seed, key.x0);
    seed = mix_uint64(seed, key.z0);
    seed = mix_uint64(seed, key.x1);
    seed = mix_uint64(seed, key.z1);
    return seed;
  }
};

inline bool less_packed_key1(const PackedKey1& lhs, const PackedKey1& rhs) noexcept {
  if (lhs.x != rhs.x) {
    return lhs.x < rhs.x;
  }
  return lhs.z < rhs.z;
}

inline bool less_packed_key2(const PackedKey2& lhs, const PackedKey2& rhs) noexcept {
  if (lhs.x0 != rhs.x0) {
    return lhs.x0 < rhs.x0;
  }
  if (lhs.z0 != rhs.z0) {
    return lhs.z0 < rhs.z0;
  }
  if (lhs.x1 != rhs.x1) {
    return lhs.x1 < rhs.x1;
  }
  return lhs.z1 < rhs.z1;
}

inline int compare_term_keys(
    const std::vector<std::uint64_t>& x,
    const std::vector<std::uint64_t>& z,
    std::size_t words,
    std::size_t lhs,
    std::size_t rhs) noexcept {
  if (words == 0) {
    return 0;
  }

  const std::size_t lhs_offset = lhs * words;
  const std::size_t rhs_offset = rhs * words;

  // Keep common compact operators branch-specialized until dedicated SIMD
  // dispatch lands; the general path below preserves the same key order.
  if (words == 1) {
    if (int comparison = compare_word(x[lhs_offset], x[rhs_offset]); comparison != 0) {
      return comparison;
    }
    return compare_word(z[lhs_offset], z[rhs_offset]);
  }

  if (words == 2) {
    for (std::size_t word = 0; word < 2; ++word) {
      if (int comparison = compare_word(x[lhs_offset + word], x[rhs_offset + word]);
          comparison != 0) {
        return comparison;
      }
      if (int comparison = compare_word(z[lhs_offset + word], z[rhs_offset + word]);
          comparison != 0) {
        return comparison;
      }
    }
    return 0;
  }

  for (std::size_t word = 0; word < words; ++word) {
    if (int comparison = compare_word(x[lhs_offset + word], x[rhs_offset + word]);
        comparison != 0) {
      return comparison;
    }
    if (int comparison = compare_word(z[lhs_offset + word], z[rhs_offset + word]);
        comparison != 0) {
      return comparison;
    }
  }
  return 0;
}

}  // namespace wolfgang::detail

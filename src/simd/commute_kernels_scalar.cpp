#include "detail/commute_kernels.hpp"

#include "detail/commutation.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace wolfgang::detail {

bool simd_commutation_supports_words(std::size_t words) noexcept {
  return words == 1 || words == 2;
}

void require_simd_commutation_words(std::string_view backend, std::size_t words) {
  if (simd_commutation_supports_words(words)) {
    return;
  }
  throw std::runtime_error(
      "WOLFGANG_CPU_BACKEND=" + std::string(backend) +
      " supports commutation kernels only for packed widths of 1 or 2 uint64 words; got " +
      std::to_string(words) + " words");
}

std::vector<std::uint8_t> commutes_with_scalar(
    const PauliSum& lhs,
    const PauliSum& rhs,
    std::size_t entries) {
  std::vector<std::uint8_t> out(entries, 0);

  if (lhs.words() == 1) {
    std::size_t output_index = 0;
    for (std::size_t lhs_term = 0; lhs_term < lhs.num_terms(); ++lhs_term) {
      const std::uint64_t lhs_x = lhs.x_words()[lhs_term];
      const std::uint64_t lhs_z = lhs.z_words()[lhs_term];
      for (std::size_t rhs_term = 0; rhs_term < rhs.num_terms(); ++rhs_term) {
        const std::uint64_t conflicts =
            (lhs_x & rhs.z_words()[rhs_term]) ^ (lhs_z & rhs.x_words()[rhs_term]);
        out[output_index] = (std::popcount(conflicts) & 1U) == 0 ? 1 : 0;
        ++output_index;
      }
    }
    return out;
  }

  if (lhs.words() == 2) {
    std::size_t output_index = 0;
    for (std::size_t lhs_term = 0; lhs_term < lhs.num_terms(); ++lhs_term) {
      const std::size_t lhs_offset = lhs_term * 2;
      const std::uint64_t lhs_x0 = lhs.x_words()[lhs_offset];
      const std::uint64_t lhs_z0 = lhs.z_words()[lhs_offset];
      const std::uint64_t lhs_x1 = lhs.x_words()[lhs_offset + 1];
      const std::uint64_t lhs_z1 = lhs.z_words()[lhs_offset + 1];
      for (std::size_t rhs_term = 0; rhs_term < rhs.num_terms(); ++rhs_term) {
        const std::size_t rhs_offset = rhs_term * 2;
        const std::uint64_t conflicts0 =
            (lhs_x0 & rhs.z_words()[rhs_offset]) ^ (lhs_z0 & rhs.x_words()[rhs_offset]);
        const std::uint64_t conflicts1 =
            (lhs_x1 & rhs.z_words()[rhs_offset + 1]) ^
            (lhs_z1 & rhs.x_words()[rhs_offset + 1]);
        out[output_index] =
            ((std::popcount(conflicts0) + std::popcount(conflicts1)) & 1U) == 0 ? 1 : 0;
        ++output_index;
      }
    }
    return out;
  }

  std::size_t output_index = 0;
  for (std::size_t lhs_term = 0; lhs_term < lhs.num_terms(); ++lhs_term) {
    for (std::size_t rhs_term = 0; rhs_term < rhs.num_terms(); ++rhs_term) {
      out[output_index] = terms_commute(
                              lhs.x_words(),
                              lhs.z_words(),
                              rhs.x_words(),
                              rhs.z_words(),
                              lhs.words(),
                              lhs_term,
                              rhs_term)
          ? 1
          : 0;
      ++output_index;
    }
  }
  return out;
}

std::vector<std::uint8_t> build_full_commutation_graph_scalar(const PauliSum& op) {
  const std::size_t terms = op.num_terms();
  std::vector<std::uint8_t> graph(terms * terms, 1);
  const std::vector<std::uint64_t>& x = op.x_words();
  const std::vector<std::uint64_t>& z = op.z_words();

  if (op.words() == 1) {
    for (std::size_t lhs = 0; lhs < terms; ++lhs) {
      const std::uint64_t lhs_x = x[lhs];
      const std::uint64_t lhs_z = z[lhs];
      for (std::size_t rhs = lhs + 1; rhs < terms; ++rhs) {
        const std::uint64_t conflicts = (lhs_x & z[rhs]) ^ (lhs_z & x[rhs]);
        const std::uint8_t commute = (std::popcount(conflicts) & 1U) == 0 ? 1 : 0;
        graph[lhs * terms + rhs] = commute;
        graph[rhs * terms + lhs] = commute;
      }
    }
    return graph;
  }

  if (op.words() == 2) {
    for (std::size_t lhs = 0; lhs < terms; ++lhs) {
      const std::size_t lhs_offset = lhs * 2;
      const std::uint64_t lhs_x0 = x[lhs_offset];
      const std::uint64_t lhs_z0 = z[lhs_offset];
      const std::uint64_t lhs_x1 = x[lhs_offset + 1];
      const std::uint64_t lhs_z1 = z[lhs_offset + 1];
      for (std::size_t rhs = lhs + 1; rhs < terms; ++rhs) {
        const std::size_t rhs_offset = rhs * 2;
        const std::uint64_t conflicts0 =
            (lhs_x0 & z[rhs_offset]) ^ (lhs_z0 & x[rhs_offset]);
        const std::uint64_t conflicts1 =
            (lhs_x1 & z[rhs_offset + 1]) ^ (lhs_z1 & x[rhs_offset + 1]);
        const std::uint8_t commute =
            ((std::popcount(conflicts0) + std::popcount(conflicts1)) & 1U) == 0 ? 1 : 0;
        graph[lhs * terms + rhs] = commute;
        graph[rhs * terms + lhs] = commute;
      }
    }
    return graph;
  }

  for (std::size_t lhs = 0; lhs < terms; ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs < terms; ++rhs) {
      const bool commute = terms_commute(
          op.x_words(),
          op.z_words(),
          op.x_words(),
          op.z_words(),
          op.words(),
          lhs,
          rhs);
      graph[lhs * terms + rhs] = commute ? 1 : 0;
      graph[rhs * terms + lhs] = commute ? 1 : 0;
    }
  }
  return graph;
}

}  // namespace wolfgang::detail

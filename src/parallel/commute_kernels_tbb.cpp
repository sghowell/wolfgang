#include "detail/commute_kernels.hpp"

#include "detail/commutation.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

namespace wolfgang::detail {

std::vector<std::uint8_t> commutes_with_tbb(
    const PauliSum& lhs,
    const PauliSum& rhs,
    std::size_t entries) {
  std::vector<std::uint8_t> out(entries, 0);

  oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range<std::size_t>(0, lhs.num_terms()),
      [&](const oneapi::tbb::blocked_range<std::size_t>& range) {
        for (std::size_t lhs_term = range.begin(); lhs_term != range.end(); ++lhs_term) {
          const std::size_t row_offset = lhs_term * rhs.num_terms();
          if (lhs.words() == 1) {
            const std::uint64_t lhs_x = lhs.x_words()[lhs_term];
            const std::uint64_t lhs_z = lhs.z_words()[lhs_term];
            for (std::size_t rhs_term = 0; rhs_term < rhs.num_terms(); ++rhs_term) {
              const std::uint64_t conflicts =
                  (lhs_x & rhs.z_words()[rhs_term]) ^ (lhs_z & rhs.x_words()[rhs_term]);
              out[row_offset + rhs_term] = (std::popcount(conflicts) & 1U) == 0 ? 1 : 0;
            }
            continue;
          }

          if (lhs.words() == 2) {
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
              out[row_offset + rhs_term] =
                  ((std::popcount(conflicts0) + std::popcount(conflicts1)) & 1U) == 0 ? 1 : 0;
            }
            continue;
          }

          for (std::size_t rhs_term = 0; rhs_term < rhs.num_terms(); ++rhs_term) {
            out[row_offset + rhs_term] = terms_commute(
                                             lhs.x_words(),
                                             lhs.z_words(),
                                             rhs.x_words(),
                                             rhs.z_words(),
                                             lhs.words(),
                                             lhs_term,
                                             rhs_term)
                ? 1
                : 0;
          }
        }
      });
  return out;
}

std::vector<std::uint8_t> build_full_commutation_graph_tbb(const PauliSum& op) {
  const std::size_t terms = op.num_terms();
  std::vector<std::uint8_t> graph(terms * terms, 1);

  oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range<std::size_t>(0, terms),
      [&](const oneapi::tbb::blocked_range<std::size_t>& range) {
        for (std::size_t lhs = range.begin(); lhs != range.end(); ++lhs) {
          for (std::size_t rhs = lhs + 1; rhs < terms; ++rhs) {
            const bool commute = terms_commute(
                op.x_words(),
                op.z_words(),
                op.x_words(),
                op.z_words(),
                op.words(),
                lhs,
                rhs);
            const std::uint8_t value = commute ? 1 : 0;
            graph[lhs * terms + rhs] = value;
            graph[rhs * terms + lhs] = value;
          }
        }
      });
  return graph;
}

}  // namespace wolfgang::detail

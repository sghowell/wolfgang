#include "detail/commute_kernels.hpp"

#include "wolfgang/cpu_backend.hpp"

#include "detail/commutation.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

namespace wolfgang::detail {

void commutes_with_tbb(
    const PauliSum& lhs,
    const PauliSum& rhs,
    std::uint8_t* out,
    CommutationRowsKernel rows) {
  note_tbb_kernel_use();
  // Each chunk writes its own rows, so the first touch of fresh output pages is
  // spread across the threads too.
  oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range<std::size_t>(0, lhs.num_terms()),
      [&](const oneapi::tbb::blocked_range<std::size_t>& range) {
        rows(lhs, rhs, range.begin(), range.end(), out);
      });
}

std::vector<std::uint8_t> build_full_commutation_graph_tbb(const PauliSum& op) {
  note_tbb_kernel_use();
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

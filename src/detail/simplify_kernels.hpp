#pragma once

#include "wolfgang/pauli_sum.hpp"

#include "detail/simplify_numeric.hpp"

#include <complex>
#include <cstdint>
#include <vector>

namespace wolfgang::detail {

// Packed output of a simplify kernel in canonical order.
struct SimplifiedTerms {
  std::vector<std::uint64_t> x;
  std::vector<std::uint64_t> z;
  std::vector<std::complex<double>> coeffs;
};

#if WOLFGANG_BUILD_TBB_ENABLED
// oneTBB simplify for words >= 1. Terms are sorted by (packed key, input
// index), so each duplicate run lists its coefficients in input order and is
// summed sequentially; runs are reduced in parallel and concatenated in key
// order. The result equals the scalar simplify bit for bit.
[[nodiscard]] SimplifiedTerms simplify_tbb(const PauliSum& op, const SimplifyDropRule& rule);
#endif

}  // namespace wolfgang::detail

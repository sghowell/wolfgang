#pragma once

#include <cmath>
#include <complex>
#include <stdexcept>

namespace wolfgang::detail {

// Keep standalone simplify and fused multiplication reductions consistent.
inline void require_finite_simplify_input(const std::complex<double>& coefficient) {
  if (!std::isfinite(coefficient.real()) || !std::isfinite(coefficient.imag())) {
    throw std::invalid_argument("simplify requires finite coefficients");
  }
}

inline void require_finite_simplify_accumulator(const std::complex<double>& coefficient) {
  if (!std::isfinite(coefficient.real()) || !std::isfinite(coefficient.imag())) {
    throw std::overflow_error("simplify coefficient accumulation overflowed complex128");
  }
}

}  // namespace wolfgang::detail

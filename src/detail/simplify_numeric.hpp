#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <span>
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

// atol + rtol * scale with the product rounded before the add. Compilers may
// otherwise contract it into one fused multiply-add (Apple clang and GCC do by
// default on arm64), which moves the threshold by an ulp and makes the drop
// rule differ across platforms and backends; every backend uses this form.
[[nodiscard]] inline double separately_rounded_threshold(double atol, double rtol, double scale) {
  const volatile double product = rtol * scale;
  return atol + product;
}

// Drop rule shared by the standalone CPU simplify paths (scalar and oneTBB) and
// the Metal hybrid; fused matmul applies the same rule at default tolerances: a combined coefficient c
// survives when |c| > atol + rtol * max_abs_input. Finite complex128 components
// can have a magnitude above DBL_MAX, so at rtol == 0 the threshold is atol
// itself (no 0 * infinity), and when the ordinary threshold overflows the
// comparison switches to half units, which needs no extended precision.
struct SimplifyDropRule {
  double drop_threshold = 0.0;
  double half_threshold = 0.0;

  // Validates that every input coefficient is finite.
  static SimplifyDropRule from_coefficients(
      std::span<const std::complex<double>> coeffs,
      double atol,
      double rtol) {
    double max_abs_input = 0.0;
    for (const std::complex<double>& coeff : coeffs) {
      require_finite_simplify_input(coeff);
      max_abs_input = std::max(max_abs_input, std::abs(coeff));
    }
    SimplifyDropRule rule;
    rule.drop_threshold = rtol == 0.0 ? atol : separately_rounded_threshold(atol, rtol, max_abs_input);
    if (!std::isfinite(rule.drop_threshold)) {
      double max_half_magnitude = 0.0;
      for (const auto& coeff : coeffs) {
        max_half_magnitude = std::max(
            max_half_magnitude, std::hypot(coeff.real() * 0.5, coeff.imag() * 0.5));
      }
      rule.half_threshold = separately_rounded_threshold(atol * 0.5, rtol, max_half_magnitude);
      rule.drop_threshold = rule.half_threshold * 2.0;
    }
    return rule;
  }

  // Validates that the accumulated coefficient did not overflow.
  [[nodiscard]] bool survives(const std::complex<double>& coeff) const {
    require_finite_simplify_accumulator(coeff);
    return std::isfinite(drop_threshold)
        ? std::abs(coeff) > drop_threshold
        : std::hypot(coeff.real() * 0.5, coeff.imag() * 0.5) > half_threshold;
  }
};

}  // namespace wolfgang::detail

#pragma once

#include <complex>
#include <cstdint>

namespace wolfgang::detail {

inline std::complex<double> phase_from_exponent(std::int64_t exponent) noexcept {
  const std::uint64_t normalized = static_cast<std::uint64_t>(exponent) & 3U;
  switch (normalized) {
    case 0:
      return {1.0, 0.0};
    case 1:
      return {0.0, 1.0};
    case 2:
      return {-1.0, 0.0};
    default:
      return {0.0, -1.0};
  }
}

inline std::complex<double> multiply_by_phase_exponent(
    std::complex<double> value,
    std::int64_t exponent) noexcept {
  const std::uint64_t normalized = static_cast<std::uint64_t>(exponent) & 3U;
  switch (normalized) {
    case 0:
      return value;
    case 1:
      return {-value.imag(), value.real()};
    case 2:
      return {-value.real(), -value.imag()};
    default:
      return {value.imag(), -value.real()};
  }
}

}  // namespace wolfgang::detail

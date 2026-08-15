#include "wolfgang/cpu_backend.hpp"
#include "wolfgang/pauli_sum.hpp"

#include "detail/commute_kernels.hpp"
#include "detail/checked_arithmetic.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace wolfgang {
namespace {

bool backend_available(const CpuBackendReport& report, std::string_view selector) noexcept {
  for (const CpuBackendCandidate& candidate : report.candidates) {
    if (candidate.name == selector) {
      return candidate.status == "available";
    }
  }
  return false;
}

}  // namespace

std::vector<std::uint8_t> PauliSum::commutes_with(
    const PauliSum& rhs,
    std::size_t max_commutation_matrix_entries) const {
  if (num_qubits_ != rhs.num_qubits_) {
    throw std::invalid_argument("PauliSum commutation requires the same num_qubits");
  }

  const std::size_t entries = detail::checked_commutation_matrix_entries(
      num_terms_,
      rhs.num_terms_,
      max_commutation_matrix_entries);

  const CpuBackendReport backend = cpu_backend_report_from_environment();
  if (backend.requested_backend == "auto") {
#if FASTPAULI_BUILD_TBB_ENABLED
    if (entries >= kAutoTbbPairwiseEntryThreshold && backend_available(backend, "tbb")) {
      return detail::commutes_with_tbb(*this, rhs, entries);
    }
#endif
#if FASTPAULI_BUILD_AVX512_ENABLED
    if (detail::simd_commutation_supports_words(words_) &&
        backend_available(backend, "avx512")) {
      return detail::commutes_with_avx512(*this, rhs, entries);
    }
#endif
#if FASTPAULI_BUILD_AVX2_ENABLED
    if (detail::simd_commutation_supports_words(words_) && backend_available(backend, "avx2")) {
      return detail::commutes_with_avx2(*this, rhs, entries);
    }
#endif
#if FASTPAULI_BUILD_ARM_NEON_ENABLED
    if (detail::simd_commutation_supports_words(words_) && backend_available(backend, "neon")) {
      return detail::commutes_with_neon(*this, rhs, entries);
    }
#endif
    return detail::commutes_with_scalar(*this, rhs, entries);
  }

  if (backend.active_backend == "tbb") {
#if FASTPAULI_BUILD_TBB_ENABLED
    return detail::commutes_with_tbb(*this, rhs, entries);
#endif
  }

  if (backend.active_backend == "avx512") {
#if FASTPAULI_BUILD_AVX512_ENABLED
    detail::require_simd_commutation_words("avx512", words_);
    return detail::commutes_with_avx512(*this, rhs, entries);
#endif
  }

  if (backend.active_backend == "avx2") {
#if FASTPAULI_BUILD_AVX2_ENABLED
    detail::require_simd_commutation_words("avx2", words_);
    return detail::commutes_with_avx2(*this, rhs, entries);
#endif
  }

  if (backend.active_backend == "neon") {
#if FASTPAULI_BUILD_ARM_NEON_ENABLED
    detail::require_simd_commutation_words("neon", words_);
    return detail::commutes_with_neon(*this, rhs, entries);
#endif
  }

  return detail::commutes_with_scalar(*this, rhs, entries);
}

std::size_t PauliSum::checked_commutation_matrix_entries_for_testing(
    std::size_t lhs_terms,
    std::size_t rhs_terms,
    std::size_t max_commutation_matrix_entries) {
  return detail::checked_commutation_matrix_entries(
      lhs_terms,
      rhs_terms,
      max_commutation_matrix_entries);
}

}  // namespace wolfgang

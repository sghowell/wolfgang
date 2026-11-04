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

#if WOLFGANG_BUILD_TBB_ENABLED
// The row kernel each oneTBB chunk runs: the widest SIMD kernel this build and
// CPU support for one- and two-word operators, otherwise scalar. Without it the
// oneTBB path ran a scalar loop per thread and needed many threads to match a
// single AVX-512 thread.
detail::CommutationRowsKernel tbb_commutation_rows(const CpuBackendReport& report, std::size_t words) {
  if (detail::simd_commutation_supports_words(words)) {
#if WOLFGANG_BUILD_AVX512_ENABLED
    if (backend_available(report, "avx512")) {
      return detail::commutes_with_avx512_rows;
    }
#endif
#if WOLFGANG_BUILD_AVX2_ENABLED
    if (backend_available(report, "avx2")) {
      return detail::commutes_with_avx2_rows;
    }
#endif
#if WOLFGANG_BUILD_ARM_NEON_ENABLED
    if (backend_available(report, "neon")) {
      return detail::commutes_with_neon_rows;
    }
#endif
  }
  return detail::commutes_with_scalar_rows;
}

// True for the AVX2 and AVX-512 row kernels, whose one-word form is bound by
// writing its output (kAutoTbbPairwiseOneWordX86SimdEntryThreshold).
bool x86_simd_rows(detail::CommutationRowsKernel rows) noexcept {
#if WOLFGANG_BUILD_AVX512_ENABLED
  if (rows == detail::commutes_with_avx512_rows) {
    return true;
  }
#endif
#if WOLFGANG_BUILD_AVX2_ENABLED
  if (rows == detail::commutes_with_avx2_rows) {
    return true;
  }
#endif
  (void)rows;
  return false;
}
#endif

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
#if WOLFGANG_BUILD_TBB_ENABLED
    if (entries >= kAutoTbbPairwiseEntryThreshold && backend_available(backend, "tbb")) {
      const detail::CommutationRowsKernel rows = tbb_commutation_rows(backend, words_);
      const bool x86_simd = x86_simd_rows(rows);
      // Two threads beat scalar and NEON, but matching AVX2 or AVX-512 takes three.
      const int minimum = x86_simd ? kAutoTbbPairwiseMinConcurrencyOverX86Simd : kAutoTbbPairwiseMinConcurrency;
      const std::size_t threshold = words_ == 1 && x86_simd
          ? kAutoTbbPairwiseOneWordX86SimdEntryThreshold
          : kAutoTbbPairwiseEntryThreshold;
      if (entries >= threshold && tbb_auto_has_min_concurrency(minimum)) {
        return detail::commutes_with_tbb(*this, rhs, entries, rows);
      }
    }
#endif
#if WOLFGANG_BUILD_AVX512_ENABLED
    if (detail::simd_commutation_supports_words(words_) &&
        backend_available(backend, "avx512")) {
      return detail::commutes_with_avx512(*this, rhs, entries);
    }
#endif
#if WOLFGANG_BUILD_AVX2_ENABLED
    if (detail::simd_commutation_supports_words(words_) && backend_available(backend, "avx2")) {
      return detail::commutes_with_avx2(*this, rhs, entries);
    }
#endif
#if WOLFGANG_BUILD_ARM_NEON_ENABLED
    if (detail::simd_commutation_supports_words(words_) && backend_available(backend, "neon")) {
      return detail::commutes_with_neon(*this, rhs, entries);
    }
#endif
    return detail::commutes_with_scalar(*this, rhs, entries);
  }

  if (backend.active_backend == "tbb") {
#if WOLFGANG_BUILD_TBB_ENABLED
    return detail::commutes_with_tbb(*this, rhs, entries, tbb_commutation_rows(backend, words_));
#endif
  }

  if (backend.active_backend == "avx512") {
#if WOLFGANG_BUILD_AVX512_ENABLED
    detail::require_simd_commutation_words("avx512", words_);
    return detail::commutes_with_avx512(*this, rhs, entries);
#endif
  }

  if (backend.active_backend == "avx2") {
#if WOLFGANG_BUILD_AVX2_ENABLED
    detail::require_simd_commutation_words("avx2", words_);
    return detail::commutes_with_avx2(*this, rhs, entries);
#endif
  }

  if (backend.active_backend == "neon") {
#if WOLFGANG_BUILD_ARM_NEON_ENABLED
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

#pragma once

#include "wolfgang/pauli_sum.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace wolfgang::detail {

[[nodiscard]] bool simd_commutation_supports_words(std::size_t words) noexcept;
void require_simd_commutation_words(std::string_view backend, std::size_t words);

[[nodiscard]] std::vector<std::uint8_t> commutes_with_scalar(
    const PauliSum& lhs,
    const PauliSum& rhs,
    std::size_t entries);

[[nodiscard]] std::vector<std::uint8_t> build_full_commutation_graph_scalar(
    const PauliSum& op);

#if FASTPAULI_BUILD_TBB_ENABLED
[[nodiscard]] std::vector<std::uint8_t> commutes_with_tbb(
    const PauliSum& lhs,
    const PauliSum& rhs,
    std::size_t entries);

[[nodiscard]] std::vector<std::uint8_t> build_full_commutation_graph_tbb(
    const PauliSum& op);
#endif

#if FASTPAULI_BUILD_AVX2_ENABLED
[[nodiscard]] std::vector<std::uint8_t> commutes_with_avx2(
    const PauliSum& lhs,
    const PauliSum& rhs,
    std::size_t entries);
#endif

#if FASTPAULI_BUILD_AVX512_ENABLED
[[nodiscard]] std::vector<std::uint8_t> commutes_with_avx512(
    const PauliSum& lhs,
    const PauliSum& rhs,
    std::size_t entries);
#endif

#if FASTPAULI_BUILD_ARM_NEON_ENABLED
[[nodiscard]] std::vector<std::uint8_t> commutes_with_neon(
    const PauliSum& lhs,
    const PauliSum& rhs,
    std::size_t entries);
#endif

}  // namespace wolfgang::detail

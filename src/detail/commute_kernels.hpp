#pragma once

#include "wolfgang/pauli_sum.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace wolfgang::detail {

[[nodiscard]] bool simd_commutation_supports_words(std::size_t words) noexcept;
void require_simd_commutation_words(std::string_view backend, std::size_t words);

// Row-range kernels fill rows [row_begin, row_end) of the dense
// lhs.num_terms() x rhs.num_terms() commutation matrix at `out` (row-major).
// PauliSum::commutes_with_into runs one over every row, or the oneTBB kernel
// runs one on each chunk of rows, so its threads use the same SIMD inner loop
// as the single-threaded path. The SIMD row kernels need one- or two-word
// operators.
using CommutationRowsKernel = void (*)(
    const PauliSum& lhs,
    const PauliSum& rhs,
    std::size_t row_begin,
    std::size_t row_end,
    std::uint8_t* out);

void commutes_with_scalar_rows(
    const PauliSum& lhs,
    const PauliSum& rhs,
    std::size_t row_begin,
    std::size_t row_end,
    std::uint8_t* out) noexcept;

[[nodiscard]] std::vector<std::uint8_t> build_full_commutation_graph_scalar(
    const PauliSum& op);

#if WOLFGANG_BUILD_TBB_ENABLED
// Runs `rows` on each oneTBB chunk of lhs terms, writing every entry of `out`.
void commutes_with_tbb(
    const PauliSum& lhs,
    const PauliSum& rhs,
    std::uint8_t* out,
    CommutationRowsKernel rows);

[[nodiscard]] std::vector<std::uint8_t> build_full_commutation_graph_tbb(
    const PauliSum& op);
#endif

#if WOLFGANG_BUILD_AVX2_ENABLED
void commutes_with_avx2_rows(
    const PauliSum& lhs,
    const PauliSum& rhs,
    std::size_t row_begin,
    std::size_t row_end,
    std::uint8_t* out) noexcept;
#endif

#if WOLFGANG_BUILD_AVX512_ENABLED
void commutes_with_avx512_rows(
    const PauliSum& lhs,
    const PauliSum& rhs,
    std::size_t row_begin,
    std::size_t row_end,
    std::uint8_t* out) noexcept;
#endif

#if WOLFGANG_BUILD_ARM_NEON_ENABLED
void commutes_with_neon_rows(
    const PauliSum& lhs,
    const PauliSum& rhs,
    std::size_t row_begin,
    std::size_t row_end,
    std::uint8_t* out) noexcept;
#endif

}  // namespace wolfgang::detail

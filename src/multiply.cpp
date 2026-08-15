#include "wolfgang/pauli_sum.hpp"

#include "detail/checked_arithmetic.hpp"
#include "detail/packed_key.hpp"
#include "detail/phase.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace wolfgang {
namespace {

using detail::PackedKey1;
using detail::PackedKey1Hash;
using detail::PackedKey2;
using detail::PackedKey2Hash;
using detail::less_packed_key1;
using detail::less_packed_key2;

bool sample_is_duplicate_heavy(std::size_t sample_size, std::size_t unique_terms) noexcept {
  if (sample_size < 128) {
    return false;
  }
  const double duplicate_rate =
      1.0 - (static_cast<double>(unique_terms) / static_cast<double>(sample_size));
  return duplicate_rate >= 0.25;
}

bool operand_duplicate_heavy_words1(
    const std::vector<std::uint64_t>& x,
    const std::vector<std::uint64_t>& z,
    std::size_t num_terms) {
  if (num_terms < 128) {
    return false;
  }
  const std::size_t sample_size = std::min<std::size_t>(num_terms, 512);
  std::unordered_set<PackedKey1, PackedKey1Hash> sample;
  sample.reserve(sample_size);
  for (std::size_t term = 0; term < sample_size; ++term) {
    sample.insert({x[term], z[term]});
  }
  return sample_is_duplicate_heavy(sample_size, sample.size());
}

bool operand_duplicate_heavy_words2(
    const std::vector<std::uint64_t>& x,
    const std::vector<std::uint64_t>& z,
    std::size_t num_terms) {
  if (num_terms < 128) {
    return false;
  }
  const std::size_t sample_size = std::min<std::size_t>(num_terms, 512);
  std::unordered_set<PackedKey2, PackedKey2Hash> sample;
  sample.reserve(sample_size);
  for (std::size_t term = 0; term < sample_size; ++term) {
    const std::size_t offset = term * 2;
    sample.insert({x[offset], z[offset], x[offset + 1], z[offset + 1]});
  }
  return sample_is_duplicate_heavy(sample_size, sample.size());
}

}  // namespace

PauliSum PauliSum::matmul(
    const PauliSum& rhs,
    bool simplify_output,
    std::size_t max_intermediate_terms) const {
  if (num_qubits_ != rhs.num_qubits_) {
    throw std::invalid_argument("PauliSum matmul requires the same num_qubits");
  }

  const std::size_t out_terms = detail::checked_matmul_intermediate_terms(
      num_terms_,
      rhs.num_terms_,
      max_intermediate_terms);
  if (out_terms == 0) {
    return PauliSum::empty(num_qubits_);
  }

  std::vector<std::int64_t> lhs_y_parity(num_terms_, 0);
  std::vector<std::int64_t> rhs_y_parity(rhs.num_terms_, 0);

  for (std::size_t lhs_term = 0; lhs_term < num_terms_; ++lhs_term) {
    const std::size_t lhs_offset = lhs_term * words_;
    for (std::size_t word = 0; word < words_; ++word) {
      lhs_y_parity[lhs_term] += static_cast<std::int64_t>(
          std::popcount(x_[lhs_offset + word] & z_[lhs_offset + word]));
    }
  }
  for (std::size_t rhs_term = 0; rhs_term < rhs.num_terms_; ++rhs_term) {
    const std::size_t rhs_offset = rhs_term * words_;
    for (std::size_t word = 0; word < words_; ++word) {
      rhs_y_parity[rhs_term] += static_cast<std::int64_t>(
          std::popcount(rhs.x_[rhs_offset + word] & rhs.z_[rhs_offset + word]));
    }
  }

  if (simplify_output && words_ == 1 &&
      (operand_duplicate_heavy_words1(x_, z_, num_terms_) ||
       operand_duplicate_heavy_words1(rhs.x_, rhs.z_, rhs.num_terms_))) {
    std::unordered_map<PackedKey1, std::complex<double>, PackedKey1Hash> accumulators;
    accumulators.max_load_factor(0.7F);
    accumulators.reserve(std::min<std::size_t>(out_terms, 1048576));

    for (std::size_t lhs_term = 0; lhs_term < num_terms_; ++lhs_term) {
      const std::uint64_t lhs_x = x_[lhs_term];
      const std::uint64_t lhs_z = z_[lhs_term];
      const std::complex<double> lhs_coeff = coeffs_[lhs_term];
      const std::int64_t lhs_y = lhs_y_parity[lhs_term];
      for (std::size_t rhs_term = 0; rhs_term < rhs.num_terms_; ++rhs_term) {
        const std::uint64_t rhs_x = rhs.x_[rhs_term];
        const std::uint64_t rhs_z = rhs.z_[rhs_term];
        const std::uint64_t out_word_x = lhs_x ^ rhs_x;
        const std::uint64_t out_word_z = lhs_z ^ rhs_z;
        const std::int64_t phase_exponent =
            static_cast<std::int64_t>(std::popcount(out_word_x & out_word_z)) -
            lhs_y -
            rhs_y_parity[rhs_term] +
            2 * static_cast<std::int64_t>(std::popcount(lhs_x & rhs_z));
        const std::complex<double> product_coeff = detail::multiply_by_phase_exponent(
            lhs_coeff * rhs.coeffs_[rhs_term],
            phase_exponent);
        auto [iterator, inserted] = accumulators.try_emplace(
            PackedKey1{out_word_x, out_word_z},
            product_coeff);
        if (!inserted) {
          iterator->second += product_coeff;
        }
      }
    }

    std::vector<std::pair<PackedKey1, std::complex<double>>> survivors;
    survivors.reserve(accumulators.size());
    for (const auto& [key, coeff] : accumulators) {
      if (std::abs(coeff) > 1.0e-12) {
        survivors.push_back({key, coeff});
      }
    }
    std::sort(
        survivors.begin(),
        survivors.end(),
        [](const auto& lhs, const auto& rhs) noexcept {
          return less_packed_key1(lhs.first, rhs.first);
        });

    std::vector<std::uint64_t> out_x_simplified;
    std::vector<std::uint64_t> out_z_simplified;
    std::vector<std::complex<double>> out_coeffs_simplified;
    out_x_simplified.reserve(survivors.size());
    out_z_simplified.reserve(survivors.size());
    out_coeffs_simplified.reserve(survivors.size());
    for (const auto& [key, coeff] : survivors) {
      out_x_simplified.push_back(key.x);
      out_z_simplified.push_back(key.z);
      out_coeffs_simplified.push_back(coeff);
    }
    if (out_coeffs_simplified.empty()) {
      return PauliSum::empty(num_qubits_);
    }
    const std::size_t simplified_terms = out_coeffs_simplified.size();
    return PauliSum(
        num_qubits_,
        words_,
        simplified_terms,
        std::move(out_x_simplified),
        std::move(out_z_simplified),
        std::move(out_coeffs_simplified));
  }

  if (simplify_output && words_ == 2 &&
      (operand_duplicate_heavy_words2(x_, z_, num_terms_) ||
       operand_duplicate_heavy_words2(rhs.x_, rhs.z_, rhs.num_terms_))) {
    std::unordered_map<PackedKey2, std::complex<double>, PackedKey2Hash> accumulators;
    accumulators.max_load_factor(0.7F);
    accumulators.reserve(std::min<std::size_t>(out_terms, 1048576));

    for (std::size_t lhs_term = 0; lhs_term < num_terms_; ++lhs_term) {
      const std::size_t lhs_offset = lhs_term * 2;
      const std::uint64_t lhs_x0 = x_[lhs_offset];
      const std::uint64_t lhs_z0 = z_[lhs_offset];
      const std::uint64_t lhs_x1 = x_[lhs_offset + 1];
      const std::uint64_t lhs_z1 = z_[lhs_offset + 1];
      const std::complex<double> lhs_coeff = coeffs_[lhs_term];
      const std::int64_t lhs_y = lhs_y_parity[lhs_term];
      for (std::size_t rhs_term = 0; rhs_term < rhs.num_terms_; ++rhs_term) {
        const std::size_t rhs_offset = rhs_term * 2;
        const std::uint64_t rhs_x0 = rhs.x_[rhs_offset];
        const std::uint64_t rhs_z0 = rhs.z_[rhs_offset];
        const std::uint64_t rhs_x1 = rhs.x_[rhs_offset + 1];
        const std::uint64_t rhs_z1 = rhs.z_[rhs_offset + 1];
        const std::uint64_t out_x0 = lhs_x0 ^ rhs_x0;
        const std::uint64_t out_z0 = lhs_z0 ^ rhs_z0;
        const std::uint64_t out_x1 = lhs_x1 ^ rhs_x1;
        const std::uint64_t out_z1 = lhs_z1 ^ rhs_z1;
        const std::int64_t phase_exponent =
            static_cast<std::int64_t>(std::popcount(out_x0 & out_z0)) +
            static_cast<std::int64_t>(std::popcount(out_x1 & out_z1)) -
            lhs_y -
            rhs_y_parity[rhs_term] +
            2 * static_cast<std::int64_t>(
                    std::popcount(lhs_x0 & rhs_z0) + std::popcount(lhs_x1 & rhs_z1));
        const std::complex<double> product_coeff = detail::multiply_by_phase_exponent(
            lhs_coeff * rhs.coeffs_[rhs_term],
            phase_exponent);
        auto [iterator, inserted] = accumulators.try_emplace(
            PackedKey2{out_x0, out_z0, out_x1, out_z1},
            product_coeff);
        if (!inserted) {
          iterator->second += product_coeff;
        }
      }
    }

    std::vector<std::pair<PackedKey2, std::complex<double>>> survivors;
    survivors.reserve(accumulators.size());
    for (const auto& [key, coeff] : accumulators) {
      if (std::abs(coeff) > 1.0e-12) {
        survivors.push_back({key, coeff});
      }
    }
    std::sort(
        survivors.begin(),
        survivors.end(),
        [](const auto& lhs, const auto& rhs) noexcept {
          return less_packed_key2(lhs.first, rhs.first);
        });

    std::vector<std::uint64_t> out_x_simplified;
    std::vector<std::uint64_t> out_z_simplified;
    std::vector<std::complex<double>> out_coeffs_simplified;
    out_x_simplified.reserve(survivors.size() * 2);
    out_z_simplified.reserve(survivors.size() * 2);
    out_coeffs_simplified.reserve(survivors.size());
    for (const auto& [key, coeff] : survivors) {
      out_x_simplified.push_back(key.x0);
      out_x_simplified.push_back(key.x1);
      out_z_simplified.push_back(key.z0);
      out_z_simplified.push_back(key.z1);
      out_coeffs_simplified.push_back(coeff);
    }
    if (out_coeffs_simplified.empty()) {
      return PauliSum::empty(num_qubits_);
    }
    const std::size_t simplified_terms = out_coeffs_simplified.size();
    return PauliSum(
        num_qubits_,
        words_,
        simplified_terms,
        std::move(out_x_simplified),
        std::move(out_z_simplified),
        std::move(out_coeffs_simplified));
  }

  std::vector<std::uint64_t> out_x(detail::checked_product(out_terms, words_, "x"), 0);
  std::vector<std::uint64_t> out_z(detail::checked_product(out_terms, words_, "z"), 0);
  std::vector<std::complex<double>> out_coeffs(out_terms);

  if (words_ == 1) {
    std::size_t output_term = 0;
    for (std::size_t lhs_term = 0; lhs_term < num_terms_; ++lhs_term) {
      const std::uint64_t lhs_x = x_[lhs_term];
      const std::uint64_t lhs_z = z_[lhs_term];
      const std::complex<double> lhs_coeff = coeffs_[lhs_term];
      const std::int64_t lhs_y = lhs_y_parity[lhs_term];
      for (std::size_t rhs_term = 0; rhs_term < rhs.num_terms_; ++rhs_term) {
        const std::uint64_t rhs_x = rhs.x_[rhs_term];
        const std::uint64_t rhs_z = rhs.z_[rhs_term];
        const std::uint64_t out_word_x = lhs_x ^ rhs_x;
        const std::uint64_t out_word_z = lhs_z ^ rhs_z;
        const std::int64_t phase_exponent =
            static_cast<std::int64_t>(std::popcount(out_word_x & out_word_z)) -
            lhs_y -
            rhs_y_parity[rhs_term] +
            2 * static_cast<std::int64_t>(std::popcount(lhs_x & rhs_z));

        out_x[output_term] = out_word_x;
        out_z[output_term] = out_word_z;
        out_coeffs[output_term] = detail::multiply_by_phase_exponent(
            lhs_coeff * rhs.coeffs_[rhs_term],
            phase_exponent);
        ++output_term;
      }
    }

    PauliSum product(
        num_qubits_,
        words_,
        out_terms,
        std::move(out_x),
        std::move(out_z),
        std::move(out_coeffs));
    if (simplify_output) {
      return product.simplify();
    }
    return product;
  }

  if (words_ == 2) {
    std::size_t output_term = 0;
    for (std::size_t lhs_term = 0; lhs_term < num_terms_; ++lhs_term) {
      const std::size_t lhs_offset = lhs_term * 2;
      const std::uint64_t lhs_x0 = x_[lhs_offset];
      const std::uint64_t lhs_z0 = z_[lhs_offset];
      const std::uint64_t lhs_x1 = x_[lhs_offset + 1];
      const std::uint64_t lhs_z1 = z_[lhs_offset + 1];
      const std::complex<double> lhs_coeff = coeffs_[lhs_term];
      const std::int64_t lhs_y = lhs_y_parity[lhs_term];
      for (std::size_t rhs_term = 0; rhs_term < rhs.num_terms_; ++rhs_term) {
        const std::size_t rhs_offset = rhs_term * 2;
        const std::size_t out_offset = output_term * 2;
        const std::uint64_t rhs_x0 = rhs.x_[rhs_offset];
        const std::uint64_t rhs_z0 = rhs.z_[rhs_offset];
        const std::uint64_t rhs_x1 = rhs.x_[rhs_offset + 1];
        const std::uint64_t rhs_z1 = rhs.z_[rhs_offset + 1];
        const std::uint64_t out_x0 = lhs_x0 ^ rhs_x0;
        const std::uint64_t out_z0 = lhs_z0 ^ rhs_z0;
        const std::uint64_t out_x1 = lhs_x1 ^ rhs_x1;
        const std::uint64_t out_z1 = lhs_z1 ^ rhs_z1;
        const std::int64_t phase_exponent =
            static_cast<std::int64_t>(std::popcount(out_x0 & out_z0)) +
            static_cast<std::int64_t>(std::popcount(out_x1 & out_z1)) -
            lhs_y -
            rhs_y_parity[rhs_term] +
            2 * static_cast<std::int64_t>(
                    std::popcount(lhs_x0 & rhs_z0) + std::popcount(lhs_x1 & rhs_z1));

        out_x[out_offset] = out_x0;
        out_x[out_offset + 1] = out_x1;
        out_z[out_offset] = out_z0;
        out_z[out_offset + 1] = out_z1;
        out_coeffs[output_term] = detail::multiply_by_phase_exponent(
            lhs_coeff * rhs.coeffs_[rhs_term],
            phase_exponent);
        ++output_term;
      }
    }

    PauliSum product(
        num_qubits_,
        words_,
        out_terms,
        std::move(out_x),
        std::move(out_z),
        std::move(out_coeffs));
    if (simplify_output) {
      return product.simplify();
    }
    return product;
  }

  std::size_t output_term = 0;
  for (std::size_t lhs_term = 0; lhs_term < num_terms_; ++lhs_term) {
    const std::size_t lhs_offset = lhs_term * words_;
    const std::complex<double> lhs_coeff = coeffs_[lhs_term];
    const std::int64_t lhs_y = lhs_y_parity[lhs_term];
    for (std::size_t rhs_term = 0; rhs_term < rhs.num_terms_; ++rhs_term) {
      const std::size_t rhs_offset = rhs_term * words_;
      const std::size_t out_offset = output_term * words_;
      std::int64_t phase_exponent = -lhs_y - rhs_y_parity[rhs_term];

      for (std::size_t word = 0; word < words_; ++word) {
        const std::uint64_t lhs_x = x_[lhs_offset + word];
        const std::uint64_t lhs_z = z_[lhs_offset + word];
        const std::uint64_t rhs_x = rhs.x_[rhs_offset + word];
        const std::uint64_t rhs_z = rhs.z_[rhs_offset + word];
        const std::uint64_t out_word_x = lhs_x ^ rhs_x;
        const std::uint64_t out_word_z = lhs_z ^ rhs_z;

        out_x[out_offset + word] = out_word_x;
        out_z[out_offset + word] = out_word_z;
        phase_exponent += static_cast<std::int64_t>(std::popcount(out_word_x & out_word_z));
        phase_exponent += 2 * static_cast<std::int64_t>(std::popcount(lhs_x & rhs_z));
      }

      out_coeffs[output_term] = detail::multiply_by_phase_exponent(
          lhs_coeff * rhs.coeffs_[rhs_term],
          phase_exponent);
      ++output_term;
    }
  }

  PauliSum product(
      num_qubits_,
      words_,
      out_terms,
      std::move(out_x),
      std::move(out_z),
      std::move(out_coeffs));
  if (simplify_output) {
    return product.simplify();
  }
  return product;
}

std::size_t PauliSum::checked_matmul_intermediate_terms_for_testing(
    std::size_t lhs_terms,
    std::size_t rhs_terms,
    std::size_t max_intermediate_terms) {
  return detail::checked_matmul_intermediate_terms(lhs_terms, rhs_terms, max_intermediate_terms);
}

}  // namespace wolfgang

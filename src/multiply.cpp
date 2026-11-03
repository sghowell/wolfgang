#include "wolfgang/pauli_sum.hpp"

#include "detail/checked_arithmetic.hpp"
#include "detail/packed_key.hpp"
#include "detail/phase.hpp"
#include "detail/simplify_numeric.hpp"

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
        detail::require_finite_simplify_input(product_coeff);
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
      detail::require_finite_simplify_accumulator(coeff);
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
        detail::require_finite_simplify_input(product_coeff);
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
      detail::require_finite_simplify_accumulator(coeff);
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

namespace {

// P and Q anticommute exactly when the symplectic form
// popcount(x_P & z_Q) + popcount(z_P & x_Q) is odd. A commutator keeps odd
// pairs and an anticommutator keeps even pairs, each as 2 c_i d_j P_i Q_j; the
// rest cancel exactly in PQ -/+ QP and are never materialized. kWords > 0 fixes
// the word count at compile time for the common one- and two-word layouts.
template <std::size_t kWords>
bool pair_kept(
    std::size_t runtime_words,
    const std::uint64_t* lhs_x,
    const std::uint64_t* lhs_z,
    const std::uint64_t* rhs_x,
    const std::uint64_t* rhs_z,
    std::uint64_t wanted_parity) noexcept {
  const std::size_t words = kWords == 0 ? runtime_words : kWords;
  std::uint64_t symplectic = 0;
  for (std::size_t word = 0; word < words; ++word) {
    symplectic += static_cast<std::uint64_t>(
        std::popcount((lhs_x[word] & rhs_z[word]) ^ (lhs_z[word] & rhs_x[word])));
  }
  return (symplectic & 1U) == wanted_parity;
}

// Two passes: count kept pairs, then write them into exactly sized buffers, so
// peak memory near max_intermediate_terms is the output itself.
template <std::size_t kWords>
void emit_parity_filtered_products(
    std::size_t runtime_words,
    const std::vector<std::uint64_t>& lhs_x,
    const std::vector<std::uint64_t>& lhs_z,
    const std::vector<std::complex<double>>& lhs_coeffs,
    const std::vector<std::int64_t>& lhs_y_parity,
    const std::vector<std::uint64_t>& rhs_x,
    const std::vector<std::uint64_t>& rhs_z,
    const std::vector<std::complex<double>>& rhs_coeffs,
    const std::vector<std::int64_t>& rhs_y_parity,
    bool keep_anticommuting,
    std::vector<std::uint64_t>& out_x,
    std::vector<std::uint64_t>& out_z,
    std::vector<std::complex<double>>& out_coeffs) {
  const std::size_t words = kWords == 0 ? runtime_words : kWords;
  const std::size_t lhs_terms = lhs_coeffs.size();
  const std::size_t rhs_terms = rhs_coeffs.size();
  const std::uint64_t wanted_parity = keep_anticommuting ? 1U : 0U;

  std::size_t kept = 0;
  for (std::size_t lhs_term = 0; lhs_term < lhs_terms; ++lhs_term) {
    for (std::size_t rhs_term = 0; rhs_term < rhs_terms; ++rhs_term) {
      kept += pair_kept<kWords>(
          words,
          lhs_x.data() + lhs_term * words,
          lhs_z.data() + lhs_term * words,
          rhs_x.data() + rhs_term * words,
          rhs_z.data() + rhs_term * words,
          wanted_parity);
    }
  }
  out_x.resize(detail::checked_product(kept, words, "commutator x"));
  out_z.resize(out_x.size());
  out_coeffs.resize(kept);

  std::size_t output_term = 0;
  for (std::size_t lhs_term = 0; lhs_term < lhs_terms; ++lhs_term) {
    const std::size_t lhs_offset = lhs_term * words;
    for (std::size_t rhs_term = 0; rhs_term < rhs_terms; ++rhs_term) {
      const std::size_t rhs_offset = rhs_term * words;
      if (!pair_kept<kWords>(
              words,
              lhs_x.data() + lhs_offset,
              lhs_z.data() + lhs_offset,
              rhs_x.data() + rhs_offset,
              rhs_z.data() + rhs_offset,
              wanted_parity)) {
        continue;
      }
      const std::size_t out_offset = output_term * words;
      std::int64_t phase_exponent = -lhs_y_parity[lhs_term] - rhs_y_parity[rhs_term];
      for (std::size_t word = 0; word < words; ++word) {
        const std::uint64_t lx = lhs_x[lhs_offset + word];
        const std::uint64_t lz = lhs_z[lhs_offset + word];
        const std::uint64_t rx = rhs_x[rhs_offset + word];
        const std::uint64_t rz = rhs_z[rhs_offset + word];
        const std::uint64_t ox = lx ^ rx;
        const std::uint64_t oz = lz ^ rz;
        out_x[out_offset + word] = ox;
        out_z[out_offset + word] = oz;
        phase_exponent += static_cast<std::int64_t>(std::popcount(ox & oz));
        phase_exponent += 2 * static_cast<std::int64_t>(std::popcount(lx & rz));
      }
      // Scale after multiplying so 2 * c_i alone can never overflow.
      out_coeffs[output_term] = detail::multiply_by_phase_exponent(
          2.0 * (lhs_coeffs[lhs_term] * rhs_coeffs[rhs_term]),
          phase_exponent);
      ++output_term;
    }
  }
}

std::vector<std::int64_t> y_parities(
    const std::vector<std::uint64_t>& x,
    const std::vector<std::uint64_t>& z,
    std::size_t num_terms,
    std::size_t words) {
  std::vector<std::int64_t> parity(num_terms, 0);
  for (std::size_t term = 0; term < num_terms; ++term) {
    for (std::size_t word = 0; word < words; ++word) {
      parity[term] += static_cast<std::int64_t>(
          std::popcount(x[term * words + word] & z[term * words + word]));
    }
  }
  return parity;
}

}  // namespace

PauliSum PauliSum::commutator(
    const PauliSum& rhs,
    bool simplify_output,
    std::size_t max_intermediate_terms) const {
  return parity_filtered_product(rhs, true, simplify_output, max_intermediate_terms);
}

PauliSum PauliSum::anticommutator(
    const PauliSum& rhs,
    bool simplify_output,
    std::size_t max_intermediate_terms) const {
  return parity_filtered_product(rhs, false, simplify_output, max_intermediate_terms);
}

PauliSum PauliSum::parity_filtered_product(
    const PauliSum& rhs,
    bool keep_anticommuting,
    bool simplify_output,
    std::size_t max_intermediate_terms) const {
  if (num_qubits_ != rhs.num_qubits_) {
    throw std::invalid_argument(
        keep_anticommuting ? "PauliSum commutator requires the same num_qubits"
                           : "PauliSum anticommutator requires the same num_qubits");
  }
  const std::size_t candidate_terms = detail::checked_matmul_intermediate_terms(
      num_terms_,
      rhs.num_terms_,
      max_intermediate_terms,
      keep_anticommuting ? "commutator" : "anticommutator");
  if (candidate_terms == 0) {
    return PauliSum::empty(num_qubits_);
  }
  // Pairs that cancel are never formed, so check every coefficient up front:
  // otherwise whether a NaN or infinity raised would depend on which pairs
  // commute.
  for (const auto* operand : {&coeffs_, &rhs.coeffs_}) {
    for (const std::complex<double>& coeff : *operand) {
      if (!std::isfinite(coeff.real()) || !std::isfinite(coeff.imag())) {
        throw std::invalid_argument(
            keep_anticommuting ? "commutator requires finite coefficients"
                               : "anticommutator requires finite coefficients");
      }
    }
  }

  const std::vector<std::int64_t> lhs_y = y_parities(x_, z_, num_terms_, words_);
  const std::vector<std::int64_t> rhs_y = y_parities(rhs.x_, rhs.z_, rhs.num_terms_, words_);
  std::vector<std::uint64_t> out_x;
  std::vector<std::uint64_t> out_z;
  std::vector<std::complex<double>> out_coeffs;
  switch (words_) {
    case 1:
      emit_parity_filtered_products<1>(
          words_, x_, z_, coeffs_, lhs_y, rhs.x_, rhs.z_, rhs.coeffs_, rhs_y,
          keep_anticommuting, out_x, out_z, out_coeffs);
      break;
    case 2:
      emit_parity_filtered_products<2>(
          words_, x_, z_, coeffs_, lhs_y, rhs.x_, rhs.z_, rhs.coeffs_, rhs_y,
          keep_anticommuting, out_x, out_z, out_coeffs);
      break;
    default:
      emit_parity_filtered_products<0>(
          words_, x_, z_, coeffs_, lhs_y, rhs.x_, rhs.z_, rhs.coeffs_, rhs_y,
          keep_anticommuting, out_x, out_z, out_coeffs);
      break;
  }

  const std::size_t out_terms = out_coeffs.size();
  PauliSum product(
      num_qubits_, words_, out_terms, std::move(out_x), std::move(out_z), std::move(out_coeffs));
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

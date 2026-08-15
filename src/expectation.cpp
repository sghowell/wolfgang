#include "fastpauli/pauli_sum.hpp"

#include "detail/checked_arithmetic.hpp"
#include "detail/phase.hpp"

#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace wolfgang {

namespace {

constexpr std::size_t kMaxInitialStatevectorQubits = 63;

std::size_t expected_statevector_length(std::size_t num_qubits) {
  if (num_qubits > kMaxInitialStatevectorQubits) {
    throw std::invalid_argument("expectation_statevector requires num_qubits <= 63");
  }
  return std::size_t{1} << num_qubits;
}

void validate_statevector_length(std::size_t num_qubits, std::size_t actual_size) {
  const std::size_t expected_size = expected_statevector_length(num_qubits);
  if (actual_size != expected_size) {
    throw std::invalid_argument("expectation_statevector requires len(psi) == 2 ** num_qubits");
  }
}

template <typename Complex>
std::complex<double> expectation_statevector_impl(
    const PauliSum& op,
    std::span<const Complex> psi) {
  validate_statevector_length(op.num_qubits(), psi.size());
  if (op.num_terms() == 0) {
    return {0.0, 0.0};
  }

  const std::size_t state_size = psi.size();
  const std::size_t words = op.words();
  const std::vector<std::uint64_t>& x_words = op.x_words();
  const std::vector<std::uint64_t>& z_words = op.z_words();
  const std::vector<std::complex<double>>& coeffs = op.coeffs();

  bool all_terms_diagonal = true;
  for (std::uint64_t word : x_words) {
    if (word != 0) {
      all_terms_diagonal = false;
      break;
    }
  }

  if (all_terms_diagonal) {
    std::vector<double> probabilities(state_size);
    for (std::size_t basis = 0; basis < state_size; ++basis) {
      probabilities[basis] = std::norm(static_cast<std::complex<double>>(psi[basis]));
    }

    if (op.num_terms() >= 64) {
      std::unordered_map<std::uint64_t, std::size_t> mask_indices;
      mask_indices.max_load_factor(0.7F);
      mask_indices.reserve(op.num_terms());
      std::vector<std::uint64_t> unique_masks;
      std::vector<std::complex<double>> coefficient_sums;
      unique_masks.reserve(op.num_terms());
      coefficient_sums.reserve(op.num_terms());

      for (std::size_t term = 0; term < op.num_terms(); ++term) {
        const std::size_t offset = term * words;
        const std::uint64_t z_mask = words == 0 ? 0 : z_words[offset];
        auto [iterator, inserted] = mask_indices.try_emplace(z_mask, unique_masks.size());
        if (inserted) {
          unique_masks.push_back(z_mask);
          coefficient_sums.push_back(coeffs[term]);
        } else {
          coefficient_sums[iterator->second] += coeffs[term];
        }
      }

      std::complex<double> result{0.0, 0.0};
      for (std::size_t mask_index = 0; mask_index < unique_masks.size(); ++mask_index) {
        const std::uint64_t z_mask = unique_masks[mask_index];
        double term_expectation = 0.0;
        for (std::size_t basis = 0; basis < state_size; ++basis) {
          const bool z_parity =
              (std::popcount(z_mask & static_cast<std::uint64_t>(basis)) & 1U) != 0;
          term_expectation += z_parity ? -probabilities[basis] : probabilities[basis];
        }
        result += coefficient_sums[mask_index] * term_expectation;
      }
      return result;
    }

    std::complex<double> result{0.0, 0.0};
    for (std::size_t term = 0; term < op.num_terms(); ++term) {
      const std::size_t offset = term * words;
      const std::uint64_t z_mask = words == 0 ? 0 : z_words[offset];
      double term_expectation = 0.0;
      for (std::size_t basis = 0; basis < state_size; ++basis) {
        const bool z_parity =
            (std::popcount(z_mask & static_cast<std::uint64_t>(basis)) & 1U) != 0;
        term_expectation += z_parity ? -probabilities[basis] : probabilities[basis];
      }
      result += coeffs[term] * term_expectation;
    }
    return result;
  }

  std::complex<double> result{0.0, 0.0};
  for (std::size_t term = 0; term < op.num_terms(); ++term) {
    const std::size_t offset = term * words;
    const std::uint64_t x_mask = words == 0 ? 0 : x_words[offset];
    const std::uint64_t z_mask = words == 0 ? 0 : z_words[offset];
    const std::complex<double> yz_phase =
        detail::phase_from_exponent(static_cast<int>(std::popcount(x_mask & z_mask)));

    if (x_mask == 0) {
      double term_expectation = 0.0;
      for (std::size_t basis = 0; basis < state_size; ++basis) {
        const bool z_parity =
            (std::popcount(z_mask & static_cast<std::uint64_t>(basis)) & 1U) != 0;
        const double probability =
            std::norm(static_cast<std::complex<double>>(psi[basis]));
        term_expectation += z_parity ? -probability : probability;
      }
      result += coeffs[term] * term_expectation;
      continue;
    }

    const std::uint64_t pivot_bit = x_mask & (std::uint64_t{0} - x_mask);
    double term_expectation = 0.0;
    for (std::size_t basis = 0; basis < state_size; ++basis) {
      if ((static_cast<std::uint64_t>(basis) & pivot_bit) != 0) {
        continue;
      }
      const std::size_t target = basis ^ static_cast<std::size_t>(x_mask);
      const bool z_parity =
          (std::popcount(z_mask & static_cast<std::uint64_t>(basis)) & 1U) != 0;
      const std::complex<double> phase = z_parity ? -yz_phase : yz_phase;
      const std::complex<double> amplitude =
          std::conj(static_cast<std::complex<double>>(psi[target])) *
          static_cast<std::complex<double>>(psi[basis]);
      term_expectation += 2.0 * (phase * amplitude).real();
    }
    result += coeffs[term] * term_expectation;
  }
  return result;
}

void validate_diagonal_terms(const PauliSum& op) {
  const std::vector<std::uint64_t>& x_words = op.x_words();
  for (std::uint64_t word : x_words) {
    if (word != 0) {
      throw std::invalid_argument("expectation_z_counts requires diagonal Pauli terms");
    }
  }
}

std::vector<std::uint64_t> pack_count_bitstrings(
    const std::vector<std::string>& bitstrings,
    std::size_t num_qubits,
    std::size_t words) {
  std::vector<std::uint64_t> packed(detail::checked_product(bitstrings.size(), words, "counts"), 0);
  for (std::size_t row = 0; row < bitstrings.size(); ++row) {
    const std::string& bitstring = bitstrings[row];
    if (bitstring.size() != num_qubits) {
      throw std::invalid_argument("Z-count bitstring length must equal num_qubits");
    }
    for (std::size_t char_index = 0; char_index < bitstring.size(); ++char_index) {
      const char bit = bitstring[char_index];
      if (bit != '0' && bit != '1') {
        throw std::invalid_argument("Z-count bitstrings may contain only 0 or 1");
      }
      if (bit == '1') {
        const std::size_t qubit = num_qubits - 1 - char_index;
        packed[row * words + qubit / 64] |= std::uint64_t{1} << (qubit % 64);
      }
    }
  }
  return packed;
}

double validate_counts_and_total(const std::vector<double>& counts) {
  double total = 0.0;
  for (double count : counts) {
    if (count < 0.0) {
      throw std::invalid_argument("Z-count values must be non-negative");
    }
    if (!std::isfinite(count)) {
      throw std::invalid_argument("Z-count values must be finite");
    }
    total += count;
  }
  if (total <= 0.0) {
    throw std::invalid_argument("Z-count total count must be positive");
  }
  return total;
}

}  // namespace

std::complex<double> PauliSum::expectation_statevector_complex128(
    std::span<const std::complex<double>> psi) const {
  return expectation_statevector_impl(*this, psi);
}

std::complex<double> PauliSum::expectation_statevector_complex64(
    std::span<const std::complex<float>> psi) const {
  return expectation_statevector_impl(*this, psi);
}

std::complex<double> PauliSum::expectation_z_counts(
    const std::vector<std::string>& bitstrings,
    const std::vector<double>& counts) const {
  if (bitstrings.size() != counts.size()) {
    throw std::invalid_argument("Z-count bitstring and count arrays must have the same length");
  }

  validate_diagonal_terms(*this);
  const double total_count = validate_counts_and_total(counts);
  const std::vector<std::uint64_t> packed_bitstrings =
      pack_count_bitstrings(bitstrings, num_qubits_, words_);

  std::complex<double> result{0.0, 0.0};
  for (std::size_t term = 0; term < num_terms_; ++term) {
    const std::size_t term_offset = term * words_;
    double weighted_sum = 0.0;
    for (std::size_t row = 0; row < bitstrings.size(); ++row) {
      bool odd_parity = false;
      const std::size_t row_offset = row * words_;
      for (std::size_t word = 0; word < words_; ++word) {
        const std::uint64_t active_z = z_[term_offset + word] & packed_bitstrings[row_offset + word];
        odd_parity ^= (std::popcount(active_z) & 1U) != 0;
      }
      weighted_sum += counts[row] * (odd_parity ? -1.0 : 1.0);
    }
    result += coeffs_[term] * (weighted_sum / total_count);
  }
  return result;
}

}  // namespace wolfgang

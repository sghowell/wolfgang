#include "wolfgang/pauli_sum.hpp"

#include "detail/checked_arithmetic.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace wolfgang {

PauliSum::PauliSum(std::size_t num_qubits, std::size_t num_terms)
    : num_qubits_(num_qubits),
      words_(detail::checked_word_count(num_qubits)),
      num_terms_(num_terms),
      x_(detail::checked_product(num_terms, words_, "x"), 0),
      z_(detail::checked_product(num_terms, words_, "z"), 0),
      coeffs_(num_terms, std::complex<double>{0.0, 0.0}) {}

PauliSum::PauliSum(
    std::size_t num_qubits,
    std::size_t words,
    std::size_t num_terms,
    std::vector<std::uint64_t> x,
    std::vector<std::uint64_t> z,
    std::vector<std::complex<double>> coeffs)
    : num_qubits_(num_qubits),
      words_(words),
      num_terms_(num_terms),
      x_(std::move(x)),
      z_(std::move(z)),
      coeffs_(std::move(coeffs)) {
  const std::size_t packed_size = detail::checked_product(num_terms_, words_, "packed");
  if (x_.size() != packed_size || z_.size() != packed_size || coeffs_.size() != num_terms_) {
    throw std::invalid_argument("PauliSum buffer sizes do not match metadata");
  }
}

PauliSum PauliSum::empty(std::size_t num_qubits) {
  return PauliSum(num_qubits, detail::checked_word_count(num_qubits), 0, {}, {}, {});
}

PauliSum PauliSum::from_words(
    std::size_t num_qubits,
    std::span<const std::uint64_t> x,
    std::span<const std::uint64_t> z,
    std::span<const std::complex<double>> coeffs) {
  const std::size_t words = detail::checked_word_count(num_qubits);
  const std::size_t num_terms = coeffs.size();
  const std::size_t packed_size = detail::checked_product(num_terms, words, "packed");
  if (x.size() != packed_size || z.size() != packed_size) {
    throw std::invalid_argument(
        "x and z must each hold num_terms * ceil(num_qubits / 64) packed words");
  }

  // Validate the owned copy, so a caller mutating its buffer concurrently
  // cannot change words between validation and use. Bits above num_qubits
  // would let two packed keys describe the same Pauli string and silently
  // break simplify, equality and commutation.
  std::vector<std::uint64_t> x_copy(x.begin(), x.end());
  std::vector<std::uint64_t> z_copy(z.begin(), z.end());
  const std::size_t tail_bits = num_qubits % 64;
  if (words > 0 && tail_bits != 0) {
    const std::uint64_t invalid_mask = ~((std::uint64_t{1} << tail_bits) - 1);
    for (std::size_t term = 0; term < num_terms; ++term) {
      const std::size_t last = term * words + words - 1;
      if (((x_copy[last] | z_copy[last]) & invalid_mask) != 0) {
        throw std::invalid_argument("packed words set bits at or above num_qubits");
      }
    }
  }

  return PauliSum(
      num_qubits,
      words,
      num_terms,
      std::move(x_copy),
      std::move(z_copy),
      std::vector<std::complex<double>>(coeffs.begin(), coeffs.end()));
}

std::size_t PauliSum::num_qubits() const noexcept { return num_qubits_; }

std::size_t PauliSum::num_terms() const noexcept { return num_terms_; }

std::size_t PauliSum::words() const noexcept { return words_; }

const std::vector<std::uint64_t>& PauliSum::x_words() const noexcept { return x_; }

const std::vector<std::uint64_t>& PauliSum::z_words() const noexcept { return z_; }

const std::vector<std::complex<double>>& PauliSum::coeffs() const noexcept { return coeffs_; }

}  // namespace wolfgang

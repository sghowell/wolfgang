#include "fastpauli/pauli_sum.hpp"

#include "detail/checked_arithmetic.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
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

std::size_t PauliSum::num_qubits() const noexcept { return num_qubits_; }

std::size_t PauliSum::num_terms() const noexcept { return num_terms_; }

std::size_t PauliSum::words() const noexcept { return words_; }

const std::vector<std::uint64_t>& PauliSum::x_words() const noexcept { return x_; }

const std::vector<std::uint64_t>& PauliSum::z_words() const noexcept { return z_; }

const std::vector<std::complex<double>>& PauliSum::coeffs() const noexcept { return coeffs_; }

}  // namespace wolfgang

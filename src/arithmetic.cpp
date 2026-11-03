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

PauliSum PauliSum::add(const PauliSum& rhs) const {
  if (num_qubits_ != rhs.num_qubits_) {
    throw std::invalid_argument("PauliSum addition requires the same num_qubits");
  }

  const std::size_t out_terms = detail::checked_sum(num_terms_, rhs.num_terms_, "terms");
  std::vector<std::uint64_t> out_x;
  std::vector<std::uint64_t> out_z;
  std::vector<std::complex<double>> out_coeffs;
  out_x.reserve(detail::checked_product(out_terms, words_, "x"));
  out_z.reserve(detail::checked_product(out_terms, words_, "z"));
  out_coeffs.reserve(out_terms);

  out_x.insert(out_x.end(), x_.begin(), x_.end());
  out_x.insert(out_x.end(), rhs.x_.begin(), rhs.x_.end());
  out_z.insert(out_z.end(), z_.begin(), z_.end());
  out_z.insert(out_z.end(), rhs.z_.begin(), rhs.z_.end());
  out_coeffs.insert(out_coeffs.end(), coeffs_.begin(), coeffs_.end());
  out_coeffs.insert(out_coeffs.end(), rhs.coeffs_.begin(), rhs.coeffs_.end());

  return PauliSum(
      num_qubits_,
      words_,
      out_terms,
      std::move(out_x),
      std::move(out_z),
      std::move(out_coeffs));
}

PauliSum PauliSum::scalar_multiply(std::complex<double> scalar) const {
  std::vector<std::complex<double>> out_coeffs;
  out_coeffs.reserve(coeffs_.size());
  for (const std::complex<double>& coeff : coeffs_) {
    out_coeffs.push_back(coeff * scalar);
  }

  return PauliSum(
      num_qubits_,
      words_,
      num_terms_,
      x_,
      z_,
      std::move(out_coeffs));
}

PauliSum PauliSum::negate() const {
  std::vector<std::complex<double>> out_coeffs;
  out_coeffs.reserve(coeffs_.size());
  for (const std::complex<double>& coeff : coeffs_) {
    out_coeffs.push_back(-coeff);
  }
  return PauliSum(num_qubits_, words_, num_terms_, x_, z_, std::move(out_coeffs));
}

PauliSum PauliSum::adjoint() const {
  std::vector<std::complex<double>> out_coeffs;
  out_coeffs.reserve(coeffs_.size());
  for (const std::complex<double>& coeff : coeffs_) {
    out_coeffs.push_back(std::conj(coeff));
  }
  return PauliSum(num_qubits_, words_, num_terms_, x_, z_, std::move(out_coeffs));
}

PauliSum PauliSum::select_terms(std::span<const std::size_t> indices) const {
  const std::size_t out_terms = indices.size();
  std::vector<std::uint64_t> out_x;
  std::vector<std::uint64_t> out_z;
  std::vector<std::complex<double>> out_coeffs;
  out_x.reserve(detail::checked_product(out_terms, words_, "x"));
  out_z.reserve(detail::checked_product(out_terms, words_, "z"));
  out_coeffs.reserve(out_terms);
  for (const std::size_t term : indices) {
    if (term >= num_terms_) {
      throw std::out_of_range("PauliSum term index out of range");
    }
    const std::size_t offset = term * words_;
    out_x.insert(out_x.end(), x_.begin() + offset, x_.begin() + offset + words_);
    out_z.insert(out_z.end(), z_.begin() + offset, z_.begin() + offset + words_);
    out_coeffs.push_back(coeffs_[term]);
  }
  return PauliSum(
      num_qubits_, words_, out_terms, std::move(out_x), std::move(out_z), std::move(out_coeffs));
}

bool PauliSum::structurally_equal(const PauliSum& rhs) const noexcept {
  return num_qubits_ == rhs.num_qubits_ && num_terms_ == rhs.num_terms_ && x_ == rhs.x_ &&
         z_ == rhs.z_ && coeffs_ == rhs.coeffs_;
}

}  // namespace wolfgang

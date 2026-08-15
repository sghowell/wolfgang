#include "wolfgang/pauli_sum.hpp"

#include "detail/bitops.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace wolfgang {

std::vector<std::string> PauliSum::to_labels() const {
  std::vector<std::string> labels;
  labels.reserve(num_terms_);

  for (std::size_t term = 0; term < num_terms_; ++term) {
    std::string label(num_qubits_, 'I');
    for (std::size_t qubit = 0; qubit < num_qubits_; ++qubit) {
      const std::size_t packed_index = term * words_ + qubit / 64;
      const std::uint64_t bit_mask = std::uint64_t{1} << (qubit % 64);
      const bool x_bit = (x_[packed_index] & bit_mask) != 0;
      const bool z_bit = (z_[packed_index] & bit_mask) != 0;
      label[num_qubits_ - 1 - qubit] = detail::pauli_from_bits(x_bit, z_bit);
    }
    labels.push_back(std::move(label));
  }

  return labels;
}

std::vector<PauliSum::SparseTerm> PauliSum::to_sparse_list() const {
  std::vector<SparseTerm> terms;
  terms.reserve(num_terms_);

  for (std::size_t term = 0; term < num_terms_; ++term) {
    SparseTerm sparse;
    sparse.coefficient = coeffs_[term];
    for (std::size_t qubit = 0; qubit < num_qubits_; ++qubit) {
      const std::size_t packed_index = term * words_ + qubit / 64;
      const std::uint64_t bit_mask = std::uint64_t{1} << (qubit % 64);
      const bool x_bit = (x_[packed_index] & bit_mask) != 0;
      const bool z_bit = (z_[packed_index] & bit_mask) != 0;
      const char pauli = detail::pauli_from_bits(x_bit, z_bit);
      if (pauli != 'I') {
        sparse.local_pauli_string.push_back(pauli);
        sparse.qubit_indices.push_back(qubit);
      }
    }
    terms.push_back(std::move(sparse));
  }

  return terms;
}

}  // namespace wolfgang

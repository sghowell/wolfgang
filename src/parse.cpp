#include "wolfgang/pauli_sum.hpp"

#include "detail/bitops.hpp"
#include "detail/checked_arithmetic.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace wolfgang {

PauliSum PauliSum::from_labels(
    const std::vector<std::string>& labels,
    const std::vector<std::complex<double>>& coeffs) {
  if (labels.empty()) {
    throw std::invalid_argument("PauliSum.from_labels([]) cannot infer num_qubits; use PauliSum.empty(num_qubits)");
  }
  if (coeffs.size() != labels.size()) {
    throw std::invalid_argument("coefficient count must match label count");
  }

  const std::size_t num_qubits = labels.front().size();
  const std::size_t words = detail::checked_word_count(num_qubits);
  const std::size_t num_terms = labels.size();
  std::vector<std::uint64_t> x(detail::checked_product(num_terms, words, "x"), 0);
  std::vector<std::uint64_t> z(detail::checked_product(num_terms, words, "z"), 0);

  for (std::size_t term = 0; term < num_terms; ++term) {
    const std::string& label = labels[term];
    if (label.size() != num_qubits) {
      throw std::invalid_argument("all dense labels must have the same length");
    }
    for (std::size_t label_offset = 0; label_offset < label.size(); ++label_offset) {
      const char pauli = label[label_offset];
      if (!detail::is_valid_pauli(pauli)) {
        throw std::invalid_argument("dense labels may contain only I, X, Y, or Z");
      }
      const std::size_t qubit = num_qubits - 1 - label_offset;
      if (pauli == 'X' || pauli == 'Y') {
        detail::set_pauli_bit(x, words, term, qubit);
      }
      if (pauli == 'Z' || pauli == 'Y') {
        detail::set_pauli_bit(z, words, term, qubit);
      }
    }
  }

  return PauliSum(num_qubits, words, num_terms, std::move(x), std::move(z), coeffs);
}

PauliSum PauliSum::from_sparse_list(
    const std::vector<SparseTerm>& terms,
    std::size_t num_qubits) {
  const std::size_t words = detail::checked_word_count(num_qubits);
  const std::size_t num_terms = terms.size();
  std::vector<std::uint64_t> x(detail::checked_product(num_terms, words, "x"), 0);
  std::vector<std::uint64_t> z(detail::checked_product(num_terms, words, "z"), 0);
  std::vector<std::complex<double>> coeffs;
  coeffs.reserve(num_terms);

  for (std::size_t term = 0; term < num_terms; ++term) {
    const SparseTerm& sparse = terms[term];
    if (sparse.local_pauli_string.size() != sparse.qubit_indices.size()) {
      throw std::invalid_argument("sparse local_pauli_string length must match qubit_indices length");
    }

    std::unordered_set<std::size_t> seen_indices;
    seen_indices.reserve(sparse.qubit_indices.size());
    for (std::size_t offset = 0; offset < sparse.local_pauli_string.size(); ++offset) {
      const char pauli = sparse.local_pauli_string[offset];
      if (!detail::is_valid_pauli(pauli)) {
        throw std::invalid_argument("sparse local_pauli_string may contain only I, X, Y, or Z");
      }
      const std::size_t qubit = sparse.qubit_indices[offset];
      if (qubit >= num_qubits) {
        throw std::invalid_argument("sparse qubit index is out of range");
      }
      if (!seen_indices.insert(qubit).second) {
        throw std::invalid_argument("sparse term contains a duplicate qubit index");
      }
      if (pauli == 'X' || pauli == 'Y') {
        detail::set_pauli_bit(x, words, term, qubit);
      }
      if (pauli == 'Z' || pauli == 'Y') {
        detail::set_pauli_bit(z, words, term, qubit);
      }
    }
    coeffs.push_back(sparse.coefficient);
  }

  return PauliSum(num_qubits, words, num_terms, std::move(x), std::move(z), std::move(coeffs));
}

}  // namespace wolfgang

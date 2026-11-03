#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace wolfgang {

class PauliSum {
public:
  // Sparse I/O terms use local_pauli_string[k] on qubit_indices[k].
  // Exported sparse terms are sorted by ascending qubit index.
  struct SparseTerm {
    std::string local_pauli_string;
    std::vector<std::size_t> qubit_indices;
    std::complex<double> coefficient;
  };

  // Metadata constructor retained for the Python scaffold surface. It creates
  // num_terms identity terms with zero coefficients.
  PauliSum(std::size_t num_qubits, std::size_t num_terms);

  // Construct a zero-term operator when num_qubits cannot be inferred from
  // labels. For num_qubits > 0, words() is ceil(num_qubits / 64).
  [[nodiscard]] static PauliSum empty(std::size_t num_qubits);

  // Dense labels use Qiskit display order: the right-most character is qubit 0.
  // All labels must have the same length and coeffs must match label count.
  [[nodiscard]] static PauliSum from_labels(
      const std::vector<std::string>& labels,
      const std::vector<std::complex<double>>& coeffs);

  // Sparse construction preserves input term order and validates duplicate and
  // out-of-range qubit indices before allocating externally visible output.
  [[nodiscard]] static PauliSum from_sparse_list(
      const std::vector<SparseTerm>& terms,
      std::size_t num_qubits);

  // Build from packed term-major words: term t occupies x[t * words(), (t + 1) *
  // words()) and the same range of z, with qubit q at bit q % 64 of word q / 64.
  // num_terms is coeffs.size(). Bits at or above num_qubits in each term's final
  // word must be zero so every term has exactly one packed representation.
  [[nodiscard]] static PauliSum from_words(
      std::size_t num_qubits,
      std::span<const std::uint64_t> x,
      std::span<const std::uint64_t> z,
      std::span<const std::complex<double>> coeffs);

  [[nodiscard]] std::size_t num_qubits() const noexcept;
  [[nodiscard]] std::size_t num_terms() const noexcept;
  [[nodiscard]] std::size_t words() const noexcept;

  // Export dense labels in Qiskit display order and preserve construction order.
  [[nodiscard]] std::vector<std::string> to_labels() const;

  // Export sparse terms in construction order; each term's qubit indices are
  // sorted ascending because the packed representation is scanned by qubit id.
  [[nodiscard]] std::vector<SparseTerm> to_sparse_list() const;

  // Combine duplicate packed Pauli keys, drop near-zero coefficients using
  // abs(c) <= atol + rtol * max_abs_input_coefficient, and return canonical
  // lexicographic packed-word order: x0, z0, x1, z1, ...
  [[nodiscard]] PauliSum simplify(double atol = 1.0e-12, double rtol = 0.0) const;

  // Concatenate terms in left-then-right order. This intentionally does not
  // simplify, so callers can choose when duplicate reduction is appropriate.
  [[nodiscard]] PauliSum add(const PauliSum& rhs) const;

  // Scale coefficients while preserving term order and packed Pauli words.
  // Multiplication by zero keeps zero-coefficient terms until simplify() runs.
  [[nodiscard]] PauliSum scalar_multiply(std::complex<double> scalar) const;

  // Flip every coefficient's sign exactly. Unlike scalar_multiply(-1), this
  // never turns an infinite component into NaN through 0 * inf cross terms.
  [[nodiscard]] PauliSum negate() const;

  // Conjugate every coefficient. Pauli strings are Hermitian, so this is the
  // operator adjoint; term order and packed words are preserved.
  [[nodiscard]] PauliSum adjoint() const;

  // Return the terms at `indices`, in that order. Indices may repeat; every
  // index must be below num_terms().
  [[nodiscard]] PauliSum select_terms(std::span<const std::size_t> indices) const;

  // True when both operands have the same num_qubits and the same terms in the
  // same order with coefficients comparing equal under ==. This is structural
  // equality: it never simplifies, so reordered or unmerged equal operators
  // compare unequal.
  [[nodiscard]] bool structurally_equal(const PauliSum& rhs) const noexcept;

  // Commutator [*this, rhs] = (*this)(rhs) - (rhs)(*this). Pauli strings either
  // commute or anticommute, so commuting term pairs contribute nothing and are
  // never generated; each anticommuting pair contributes 2 * c_i * d_j * P_i Q_j.
  // With simplify_output=false, surviving products keep lhs-outer, rhs-inner
  // order. max_intermediate_terms bounds the candidate pair count.
  [[nodiscard]] PauliSum commutator(
      const PauliSum& rhs,
      bool simplify_output = true,
      std::size_t max_intermediate_terms = 50000000) const;

  // Anticommutator {*this, rhs} = (*this)(rhs) + (rhs)(*this): only commuting
  // pairs contribute, each as 2 * c_i * d_j * P_i Q_j.
  [[nodiscard]] PauliSum anticommutator(
      const PauliSum& rhs,
      bool simplify_output = true,
      std::size_t max_intermediate_terms = 50000000) const;

  // Compose operators using matrix multiplication semantics: rhs acts first and
  // *this acts second. With simplify_output=false, products are emitted in
  // lhs-term outer, rhs-term inner nested-loop order.
  [[nodiscard]] PauliSum matmul(
      const PauliSum& rhs,
      bool simplify_output = true,
      std::size_t max_intermediate_terms = 50000000) const;

  // Return row-major pairwise commutation flags for this x rhs. Python shapes
  // this flat buffer into scalar, vector, or matrix results based on term count.
  [[nodiscard]] std::vector<std::uint8_t> commutes_with(
      const PauliSum& rhs,
      std::size_t max_commutation_matrix_entries = 100000000) const;

  // Greedily partition terms into deterministic internally compatible groups.
  // mode="qwc" requires qubit-wise compatibility; mode="full" requires global
  // Pauli commutation. strategy currently supports "largest_first".
  [[nodiscard]] std::vector<PauliSum> group_commuting(
      std::string_view mode = "qwc",
      std::string_view strategy = "largest_first",
      std::size_t max_terms_for_graph = 50000) const;

  // Compute <psi|H|psi> for a 1D statevector with dense-index convention
  // matching dense labels: the right-most label character is qubit 0.
  [[nodiscard]] std::complex<double> expectation_statevector_complex128(
      std::span<const std::complex<double>> psi) const;
  [[nodiscard]] std::complex<double> expectation_statevector_complex64(
      std::span<const std::complex<float>> psi) const;

  // Compute expectation over computational-basis counts. Input bitstrings use
  // the dense-label convention: the right-most bit is qubit 0.
  [[nodiscard]] std::complex<double> expectation_z_counts(
      const std::vector<std::string>& bitstrings,
      const std::vector<double>& counts) const;

  // Test-only guardrail helper exposed through the Python binding. It lets
  // overflow paths be validated without constructing impossible-sized buffers.
  [[nodiscard]] static std::size_t checked_matmul_intermediate_terms_for_testing(
      std::size_t lhs_terms,
      std::size_t rhs_terms,
      std::size_t max_intermediate_terms);

  // Test-only guardrail helper exposed through the Python binding. It lets
  // pairwise commutation overflow paths be validated without large buffers.
  [[nodiscard]] static std::size_t checked_commutation_matrix_entries_for_testing(
      std::size_t lhs_terms,
      std::size_t rhs_terms,
      std::size_t max_commutation_matrix_entries);

  // Read-only packed buffers are exposed for bindings, validation, and future
  // backend mirrors. They remain owned by PauliSum.
  [[nodiscard]] const std::vector<std::uint64_t>& x_words() const noexcept;
  [[nodiscard]] const std::vector<std::uint64_t>& z_words() const noexcept;
  [[nodiscard]] const std::vector<std::complex<double>>& coeffs() const noexcept;

private:
  friend class DevicePauliSum;

  // Shared kernel for commutator (keep_anticommuting=true) and anticommutator.
  [[nodiscard]] PauliSum parity_filtered_product(
      const PauliSum& rhs,
      bool keep_anticommuting,
      bool simplify_output,
      std::size_t max_intermediate_terms) const;

  PauliSum(
      std::size_t num_qubits,
      std::size_t words,
      std::size_t num_terms,
      std::vector<std::uint64_t> x,
      std::vector<std::uint64_t> z,
      std::vector<std::complex<double>> coeffs);

  std::size_t num_qubits_;
  std::size_t words_;
  std::size_t num_terms_;
  std::vector<std::uint64_t> x_;
  std::vector<std::uint64_t> z_;
  std::vector<std::complex<double>> coeffs_;
};

}  // namespace wolfgang



#include "wolfgang/pauli_sum.hpp"

#include "detail/phase.hpp"

#include <algorithm>
#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace wolfgang {

namespace {

constexpr std::size_t kMaxStatevectorQubits = 63;

std::size_t state_dimension(std::size_t num_qubits, const char* operation) {
  if (num_qubits > kMaxStatevectorQubits) {
    throw std::invalid_argument(std::string(operation) + " requires num_qubits <= 63");
  }
  return std::size_t{1} << num_qubits;
}

// Terms sharing an x mask move amplitude between the same basis pairs, so they
// are applied together. Within a group, duplicate z masks are pre-summed and
// each weight already includes the Y phase i^popcount(x & z).
struct XGroup {
  std::uint64_t x = 0;
  std::vector<std::uint64_t> z;
  std::vector<std::complex<double>> weight;
};

// Callers bound num_qubits <= 63 first, so every term has at most one word.
std::vector<XGroup> group_terms_by_x(const PauliSum& op) {
  std::map<std::uint64_t, std::map<std::uint64_t, std::complex<double>>> grouped;
  const std::size_t words = op.words();
  for (std::size_t term = 0; term < op.num_terms(); ++term) {
    const std::uint64_t x = words == 0 ? 0 : op.x_words()[term];
    const std::uint64_t z = words == 0 ? 0 : op.z_words()[term];
    grouped[x][z] += op.coeffs()[term] *
                     detail::phase_from_exponent(static_cast<int>(std::popcount(x & z)));
  }
  std::vector<XGroup> groups;
  groups.reserve(grouped.size());
  for (const auto& [x, by_z] : grouped) {
    XGroup group;
    group.x = x;
    for (const auto& [z, weight] : by_z) {
      group.z.push_back(z);
      group.weight.push_back(weight);
    }
    groups.push_back(std::move(group));
  }
  return groups;
}

// <b ^ x| H_x |b> for the terms of one x group, evaluated at input basis b.
inline std::complex<double> group_factor(const XGroup& group, std::uint64_t basis) noexcept {
  std::complex<double> factor{0.0, 0.0};
  for (std::size_t i = 0; i < group.z.size(); ++i) {
    if ((std::popcount(group.z[i] & basis) & 1U) != 0) {
      factor -= group.weight[i];
    } else {
      factor += group.weight[i];
    }
  }
  return factor;
}

// Accumulate H|psi> into out (length 2^n, zeroed by the caller) using
// precomputed groups, so batched calls group terms once.
template <typename Complex>
void apply_with_groups(
    const std::vector<XGroup>& groups,
    std::span<const Complex> psi,
    std::span<std::complex<double>> out) {
  for (const XGroup& group : groups) {
    for (std::size_t basis = 0; basis < psi.size(); ++basis) {
      out[basis ^ group.x] +=
          group_factor(group, basis) * static_cast<std::complex<double>>(psi[basis]);
    }
  }
}

template <typename Complex>
void apply_batch_impl(
    const PauliSum& op,
    std::span<const Complex> states,
    std::size_t rows,
    std::span<std::complex<double>> out) {
  const std::size_t dimension = state_dimension(op.num_qubits(), "apply_statevector");
  if (states.size() != rows * dimension || out.size() != states.size()) {
    throw std::invalid_argument("apply_statevector requires len(psi) == 2 ** num_qubits");
  }
  const std::vector<XGroup> groups = group_terms_by_x(op);
  std::fill(out.begin(), out.end(), std::complex<double>{0.0, 0.0});
  for (std::size_t row = 0; row < rows; ++row) {
    apply_with_groups(
        groups, states.subspan(row * dimension, dimension), out.subspan(row * dimension, dimension));
  }
}

template <typename Complex>
void variance_batch_impl(
    const PauliSum& op,
    std::span<const Complex> states,
    std::size_t rows,
    std::span<double> out) {
  const std::size_t dimension = state_dimension(op.num_qubits(), "variance_statevector");
  if (states.size() != rows * dimension || out.size() != rows) {
    throw std::invalid_argument("variance_statevector requires len(psi) == 2 ** num_qubits");
  }
  const std::vector<XGroup> groups = group_terms_by_x(op);
  std::vector<std::complex<double>> applied(dimension);
  for (std::size_t row = 0; row < rows; ++row) {
    const std::span<const Complex> psi = states.subspan(row * dimension, dimension);
    std::fill(applied.begin(), applied.end(), std::complex<double>{0.0, 0.0});
    apply_with_groups(groups, psi, std::span<std::complex<double>>(applied));
    double norm = 0.0;
    double applied_norm = 0.0;
    std::complex<double> expectation{0.0, 0.0};
    for (std::size_t basis = 0; basis < dimension; ++basis) {
      const auto amplitude = static_cast<std::complex<double>>(psi[basis]);
      norm += std::norm(amplitude);
      applied_norm += std::norm(applied[basis]);
      expectation += std::conj(amplitude) * applied[basis];
    }
    if (!(norm > 0.0)) {
      throw std::invalid_argument("variance_statevector requires a nonzero statevector");
    }
    // Variance of the normalized state psi / ||psi||.
    out[row] = std::max(0.0, applied_norm / norm - std::norm(expectation / norm));
  }
}

}  // namespace

void PauliSum::validate_statevector_length(std::size_t length) const {
  const std::size_t dimension = state_dimension(num_qubits_, "statevector input");
  if (length != dimension) {
    throw std::invalid_argument("statevector input requires len(psi) == 2 ** num_qubits");
  }
}

std::vector<std::complex<double>> PauliSum::apply_statevector_complex128(
    std::span<const std::complex<double>> psi) const {
  std::vector<std::complex<double>> out(psi.size());
  apply_batch_impl(*this, psi, 1, std::span<std::complex<double>>(out));
  return out;
}

std::vector<std::complex<double>> PauliSum::apply_statevector_complex64(
    std::span<const std::complex<float>> psi) const {
  std::vector<std::complex<double>> out(psi.size());
  apply_batch_impl(*this, psi, 1, std::span<std::complex<double>>(out));
  return out;
}

void PauliSum::apply_statevector_batch_complex128(
    std::span<const std::complex<double>> states,
    std::size_t rows,
    std::span<std::complex<double>> out) const {
  apply_batch_impl(*this, states, rows, out);
}

void PauliSum::apply_statevector_batch_complex64(
    std::span<const std::complex<float>> states,
    std::size_t rows,
    std::span<std::complex<double>> out) const {
  apply_batch_impl(*this, states, rows, out);
}

double PauliSum::variance_statevector_complex128(std::span<const std::complex<double>> psi) const {
  double value = 0.0;
  variance_batch_impl(*this, psi, 1, std::span<double>(&value, 1));
  return value;
}

double PauliSum::variance_statevector_complex64(std::span<const std::complex<float>> psi) const {
  double value = 0.0;
  variance_batch_impl(*this, psi, 1, std::span<double>(&value, 1));
  return value;
}

void PauliSum::variance_statevector_batch_complex128(
    std::span<const std::complex<double>> states,
    std::size_t rows,
    std::span<double> out) const {
  variance_batch_impl(*this, states, rows, out);
}

void PauliSum::variance_statevector_batch_complex64(
    std::span<const std::complex<float>> states,
    std::size_t rows,
    std::span<double> out) const {
  variance_batch_impl(*this, states, rows, out);
}

std::size_t PauliSum::dense_matrix_entries(std::size_t max_entries) const {
  const std::size_t dimension = state_dimension(num_qubits_, "to_matrix");
  // dimension^2 overflows size_t beyond 31 qubits; no such matrix is allocatable.
  if (num_qubits_ > 31 || dimension * dimension > max_entries) {
    throw std::invalid_argument(
        "to_matrix(sparse=False) would allocate 4 ** num_qubits entries, above max_entries");
  }
  return dimension * dimension;
}

std::vector<std::complex<double>> PauliSum::to_dense_matrix(std::size_t max_entries) const {
  std::vector<std::complex<double>> matrix(dense_matrix_entries(max_entries));
  to_dense_matrix_into(std::span<std::complex<double>>(matrix));
  return matrix;
}

void PauliSum::to_dense_matrix_into(std::span<std::complex<double>> out) const {
  const std::size_t dimension = state_dimension(num_qubits_, "to_matrix");
  if (num_qubits_ > 31 || out.size() != dimension * dimension) {
    throw std::invalid_argument("to_matrix output must hold 4 ** num_qubits entries");
  }
  std::fill(out.begin(), out.end(), std::complex<double>{0.0, 0.0});
  for (const XGroup& group : group_terms_by_x(*this)) {
    for (std::size_t column = 0; column < dimension; ++column) {
      out[(column ^ group.x) * dimension + column] += group_factor(group, column);
    }
  }
}

PauliSum::CsrMatrix PauliSum::to_csr_matrix(std::size_t max_nonzeros) const {
  const std::size_t dimension = state_dimension(num_qubits_, "to_matrix");
  const std::vector<XGroup> groups = group_terms_by_x(*this);
  // Budget the 2^n + 1 row pointers as well as the stored entries, so even an
  // operator with no terms cannot allocate an unbounded index array.
  const bool entries_overflow = !groups.empty() && dimension > max_nonzeros / groups.size();
  const std::size_t stored = entries_overflow ? 0 : dimension * groups.size();
  if (entries_overflow || dimension >= max_nonzeros || stored > max_nonzeros - dimension - 1) {
    throw std::invalid_argument(
        "to_matrix(sparse=True) would store 2 ** num_qubits * distinct_x_masks entries plus "
        "2 ** num_qubits + 1 row pointers, above max_entries");
  }

  CsrMatrix csr;
  csr.dimension = dimension;
  csr.indptr.reserve(dimension + 1);
  csr.data.reserve(stored);
  csr.indices.reserve(stored);
  csr.indptr.push_back(0);
  std::vector<std::pair<std::uint64_t, std::complex<double>>> row;
  row.reserve(groups.size());
  for (std::size_t r = 0; r < dimension; ++r) {
    row.clear();
    for (const XGroup& group : groups) {
      const std::uint64_t column = r ^ group.x;
      const std::complex<double> value = group_factor(group, column);
      if (value != std::complex<double>{0.0, 0.0}) {
        row.emplace_back(column, value);
      }
    }
    std::sort(row.begin(), row.end(), [](const auto& lhs, const auto& rhs) noexcept {
      return lhs.first < rhs.first;
    });
    for (const auto& [column, value] : row) {
      csr.indices.push_back(static_cast<std::int64_t>(column));
      csr.data.push_back(value);
    }
    csr.indptr.push_back(static_cast<std::int64_t>(csr.data.size()));
  }
  return csr;
}

}  // namespace wolfgang

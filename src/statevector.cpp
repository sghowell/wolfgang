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

template <typename Complex>
std::vector<std::complex<double>> apply_impl(const PauliSum& op, std::span<const Complex> psi) {
  const std::size_t dimension = state_dimension(op.num_qubits(), "apply_statevector");
  if (psi.size() != dimension) {
    throw std::invalid_argument("apply_statevector requires len(psi) == 2 ** num_qubits");
  }
  std::vector<std::complex<double>> out(dimension, std::complex<double>{0.0, 0.0});
  for (const XGroup& group : group_terms_by_x(op)) {
    for (std::size_t basis = 0; basis < dimension; ++basis) {
      out[basis ^ group.x] +=
          group_factor(group, basis) * static_cast<std::complex<double>>(psi[basis]);
    }
  }
  return out;
}

template <typename Complex>
double variance_impl(const PauliSum& op, std::span<const Complex> psi) {
  const std::vector<std::complex<double>> applied = apply_impl(op, psi);
  double applied_norm = 0.0;
  std::complex<double> expectation{0.0, 0.0};
  for (std::size_t basis = 0; basis < applied.size(); ++basis) {
    applied_norm += std::norm(applied[basis]);
    expectation += std::conj(static_cast<std::complex<double>>(psi[basis])) * applied[basis];
  }
  return std::max(0.0, applied_norm - std::norm(expectation));
}

}  // namespace

std::vector<std::complex<double>> PauliSum::apply_statevector_complex128(
    std::span<const std::complex<double>> psi) const {
  return apply_impl(*this, psi);
}

std::vector<std::complex<double>> PauliSum::apply_statevector_complex64(
    std::span<const std::complex<float>> psi) const {
  return apply_impl(*this, psi);
}

double PauliSum::variance_statevector_complex128(std::span<const std::complex<double>> psi) const {
  return variance_impl(*this, psi);
}

double PauliSum::variance_statevector_complex64(std::span<const std::complex<float>> psi) const {
  return variance_impl(*this, psi);
}

std::vector<std::complex<double>> PauliSum::to_dense_matrix(std::size_t max_entries) const {
  const std::size_t dimension = state_dimension(num_qubits_, "to_matrix");
  // dimension^2 overflows size_t beyond 31 qubits; no such matrix is allocatable.
  if (num_qubits_ > 31 || dimension * dimension > max_entries) {
    throw std::invalid_argument(
        "to_matrix(sparse=False) would allocate 4 ** num_qubits entries, above max_entries");
  }
  std::vector<std::complex<double>> matrix(dimension * dimension, std::complex<double>{0.0, 0.0});
  for (const XGroup& group : group_terms_by_x(*this)) {
    for (std::size_t column = 0; column < dimension; ++column) {
      matrix[(column ^ group.x) * dimension + column] += group_factor(group, column);
    }
  }
  return matrix;
}

PauliSum::CsrMatrix PauliSum::to_csr_matrix(std::size_t max_nonzeros) const {
  const std::size_t dimension = state_dimension(num_qubits_, "to_matrix");
  const std::vector<XGroup> groups = group_terms_by_x(*this);
  if (!groups.empty() && dimension > max_nonzeros / groups.size()) {
    throw std::invalid_argument(
        "to_matrix(sparse=True) would store 2 ** num_qubits * distinct_x_masks entries, "
        "above max_entries");
  }
  if (dimension - 1 > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument("to_matrix(sparse=True) index range exceeds int64");
  }

  CsrMatrix csr;
  csr.dimension = dimension;
  csr.indptr.reserve(dimension + 1);
  csr.data.reserve(dimension * groups.size());
  csr.indices.reserve(dimension * groups.size());
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

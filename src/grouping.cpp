#include "wolfgang/cpu_backend.hpp"
#include "wolfgang/pauli_sum.hpp"

#include "detail/bitops.hpp"
#include "detail/checked_arithmetic.hpp"
#include "detail/commutation.hpp"
#include "detail/commute_kernels.hpp"
#include "detail/packed_key.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace wolfgang {
namespace {

constexpr std::size_t kMaxPrecomputedFullGraphEntries = 10000000;

struct OrderedTerm {
  std::size_t index;
  std::size_t weight;
};

struct QwcGroup {
  std::vector<std::size_t> terms;
  std::vector<std::uint64_t> x_basis;
  std::vector<std::uint64_t> z_basis;
};

std::vector<OrderedTerm> deterministic_term_order(const PauliSum& op) {
  std::vector<OrderedTerm> ordered;
  ordered.reserve(op.num_terms());
  for (std::size_t term = 0; term < op.num_terms(); ++term) {
    ordered.push_back({
        term,
        detail::term_weight(op.x_words(), op.z_words(), op.words(), term),
    });
  }

  std::sort(ordered.begin(), ordered.end(), [&op](const OrderedTerm& lhs, const OrderedTerm& rhs) {
    if (lhs.weight != rhs.weight) {
      return lhs.weight > rhs.weight;
    }
    const int comparison = detail::compare_term_keys(
        op.x_words(),
        op.z_words(),
        op.words(),
        lhs.index,
        rhs.index);
    if (comparison != 0) {
      return comparison < 0;
    }
    return lhs.index < rhs.index;
  });
  return ordered;
}

bool qwc_group_accepts(const PauliSum& op, const QwcGroup& group, std::size_t term) noexcept {
  const std::size_t term_offset = term * op.words();
  for (std::size_t word = 0; word < op.words(); ++word) {
    const std::uint64_t conflicts =
        (op.x_words()[term_offset + word] & group.z_basis[word]) ^
        (op.z_words()[term_offset + word] & group.x_basis[word]);
    if (conflicts != 0) {
      return false;
    }
  }
  return true;
}

void add_to_qwc_group(const PauliSum& op, QwcGroup& group, std::size_t term) {
  const std::size_t term_offset = term * op.words();
  for (std::size_t word = 0; word < op.words(); ++word) {
    group.x_basis[word] |= op.x_words()[term_offset + word];
    group.z_basis[word] |= op.z_words()[term_offset + word];
  }
  group.terms.push_back(term);
}

bool should_precompute_full_graph(std::size_t terms, std::size_t max_terms_for_graph) noexcept {
  if (terms == 0 || terms > max_terms_for_graph) {
    return false;
  }
  if (terms > kMaxPrecomputedFullGraphEntries / terms) {
    return false;
  }
  return terms * terms <= kMaxPrecomputedFullGraphEntries;
}

bool full_group_accepts_streaming(
    const PauliSum& op,
    const std::vector<std::size_t>& group,
    std::size_t term) noexcept {
  for (std::size_t existing_term : group) {
    if (!detail::terms_commute(
            op.x_words(),
            op.z_words(),
            op.x_words(),
            op.z_words(),
            op.words(),
            term,
            existing_term)) {
      return false;
    }
  }
  return true;
}

bool full_group_accepts_graph(
    const std::vector<std::uint8_t>& graph,
    std::size_t total_terms,
    const std::vector<std::size_t>& group,
    std::size_t term) noexcept {
  for (std::size_t existing_term : group) {
    if (graph[term * total_terms + existing_term] == 0) {
      return false;
    }
  }
  return true;
}

bool backend_available(const CpuBackendReport& report, std::string_view selector) noexcept {
  for (const CpuBackendCandidate& candidate : report.candidates) {
    if (candidate.name == selector) {
      return candidate.status == "available";
    }
  }
  return false;
}

std::vector<std::uint8_t> build_full_commutation_graph(const PauliSum& op) {
  const CpuBackendReport backend = cpu_backend_report_from_environment();
  if (backend.requested_backend == "auto") {
#if WOLFGANG_BUILD_AVX512_ENABLED
    if (detail::simd_commutation_supports_words(op.words()) &&
        backend_available(backend, "avx512")) {
      return detail::commutes_with_avx512(op, op, op.num_terms() * op.num_terms());
    }
#endif
#if WOLFGANG_BUILD_AVX2_ENABLED
    if (detail::simd_commutation_supports_words(op.words()) &&
        backend_available(backend, "avx2")) {
      return detail::commutes_with_avx2(op, op, op.num_terms() * op.num_terms());
    }
#endif
#if WOLFGANG_BUILD_ARM_NEON_ENABLED
    if (detail::simd_commutation_supports_words(op.words()) &&
        backend_available(backend, "neon")) {
      return detail::commutes_with_neon(op, op, op.num_terms() * op.num_terms());
    }
#endif
    return detail::build_full_commutation_graph_scalar(op);
  }

  if (backend.active_backend == "tbb") {
#if WOLFGANG_BUILD_TBB_ENABLED
    return detail::build_full_commutation_graph_tbb(op);
#endif
  }

  if (backend.active_backend == "avx512") {
#if WOLFGANG_BUILD_AVX512_ENABLED
    detail::require_simd_commutation_words("avx512", op.words());
    return detail::commutes_with_avx512(op, op, op.num_terms() * op.num_terms());
#endif
  }

  if (backend.active_backend == "avx2") {
#if WOLFGANG_BUILD_AVX2_ENABLED
    detail::require_simd_commutation_words("avx2", op.words());
    return detail::commutes_with_avx2(op, op, op.num_terms() * op.num_terms());
#endif
  }

  if (backend.active_backend == "neon") {
#if WOLFGANG_BUILD_ARM_NEON_ENABLED
    detail::require_simd_commutation_words("neon", op.words());
    return detail::commutes_with_neon(op, op, op.num_terms() * op.num_terms());
#endif
  }

  return detail::build_full_commutation_graph_scalar(op);
}

}  // namespace

std::vector<PauliSum> PauliSum::group_commuting(
    std::string_view mode,
    std::string_view strategy,
    std::size_t max_terms_for_graph) const {
  if (strategy != "largest_first") {
    throw std::invalid_argument("group_commuting strategy must be 'largest_first'");
  }
  if (mode != "qwc" && mode != "full") {
    throw std::invalid_argument("group_commuting mode must be 'qwc' or 'full'");
  }
  if (num_terms_ == 0) {
    return {};
  }

  const std::vector<OrderedTerm> ordered = deterministic_term_order(*this);
  std::vector<std::vector<std::size_t>> term_groups;

  if (mode == "qwc") {
    std::vector<QwcGroup> qwc_groups;
    for (const OrderedTerm& ordered_term : ordered) {
      bool placed = false;
      for (QwcGroup& group : qwc_groups) {
        if (qwc_group_accepts(*this, group, ordered_term.index)) {
          add_to_qwc_group(*this, group, ordered_term.index);
          placed = true;
          break;
        }
      }
      if (!placed) {
        QwcGroup group;
        group.x_basis.assign(words_, 0);
        group.z_basis.assign(words_, 0);
        add_to_qwc_group(*this, group, ordered_term.index);
        qwc_groups.push_back(std::move(group));
      }
    }

    term_groups.reserve(qwc_groups.size());
    for (QwcGroup& group : qwc_groups) {
      term_groups.push_back(std::move(group.terms));
    }
  } else {
    const bool use_graph = should_precompute_full_graph(num_terms_, max_terms_for_graph);
    const std::vector<std::uint8_t> graph = use_graph
        ? build_full_commutation_graph(*this)
        : std::vector<std::uint8_t>{};

    for (const OrderedTerm& ordered_term : ordered) {
      bool placed = false;
      for (std::vector<std::size_t>& group : term_groups) {
        const bool accepts = use_graph
            ? full_group_accepts_graph(graph, num_terms_, group, ordered_term.index)
            : full_group_accepts_streaming(*this, group, ordered_term.index);
        if (accepts) {
          group.push_back(ordered_term.index);
          placed = true;
          break;
        }
      }
      if (!placed) {
        term_groups.push_back({ordered_term.index});
      }
    }
  }

  std::vector<PauliSum> groups;
  groups.reserve(term_groups.size());
  for (const std::vector<std::size_t>& term_group : term_groups) {
    std::vector<std::uint64_t> group_x;
    std::vector<std::uint64_t> group_z;
    std::vector<std::complex<double>> group_coeffs;
    group_x.reserve(detail::checked_product(term_group.size(), words_, "x"));
    group_z.reserve(detail::checked_product(term_group.size(), words_, "z"));
    group_coeffs.reserve(term_group.size());

    for (std::size_t term : term_group) {
      const std::size_t term_offset = term * words_;
      for (std::size_t word = 0; word < words_; ++word) {
        group_x.push_back(x_[term_offset + word]);
        group_z.push_back(z_[term_offset + word]);
      }
      group_coeffs.push_back(coeffs_[term]);
    }

    groups.push_back(PauliSum(
        num_qubits_,
        words_,
        term_group.size(),
        std::move(group_x),
        std::move(group_z),
        std::move(group_coeffs)));
  }
  return groups;
}

}  // namespace wolfgang

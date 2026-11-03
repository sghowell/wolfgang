#include "detail/simplify_kernels.hpp"

#include "detail/packed_key.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_sort.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace wolfgang::detail {

namespace {

// Sorting a contiguous key array keeps the comparison cache-friendly for the
// common one- and two-word layouts; wider operators sort term indices.
template <std::size_t kWords>
struct FixedKey {
  std::uint64_t x[kWords];
  std::uint64_t z[kWords];
  std::size_t index;
};

template <std::size_t kWords>
bool fixed_key_less(const FixedKey<kWords>& lhs, const FixedKey<kWords>& rhs) noexcept {
  for (std::size_t word = 0; word < kWords; ++word) {
    if (lhs.x[word] != rhs.x[word]) {
      return lhs.x[word] < rhs.x[word];
    }
    if (lhs.z[word] != rhs.z[word]) {
      return lhs.z[word] < rhs.z[word];
    }
  }
  return lhs.index < rhs.index;
}

template <std::size_t kWords>
bool fixed_key_same_term(const FixedKey<kWords>& lhs, const FixedKey<kWords>& rhs) noexcept {
  for (std::size_t word = 0; word < kWords; ++word) {
    if (lhs.x[word] != rhs.x[word] || lhs.z[word] != rhs.z[word]) {
      return false;
    }
  }
  return true;
}

// Reduce sorted entries in parallel. Each chunk owns the runs that start in it
// (a run may extend past the chunk end), sums each run sequentially in sorted
// order, and keeps survivors locally; chunks are then concatenated in order,
// so output order and every floating-point sum match the scalar loop.
template <typename SameTerm, typename EmitTerm>
SimplifiedTerms reduce_sorted_runs(
    std::size_t count,
    std::size_t words,
    const std::vector<std::complex<double>>& coeffs,
    const std::vector<std::size_t>& sorted_index,
    const SimplifyDropRule& rule,
    SameTerm same_term,
    EmitTerm emit_term) {
  struct ChunkOutput {
    std::vector<std::uint64_t> x;
    std::vector<std::uint64_t> z;
    std::vector<std::complex<double>> coeffs;
  };
  const std::size_t grain = 4096;
  const std::size_t chunks = (count + grain - 1) / grain;
  std::vector<ChunkOutput> outputs(chunks);
  oneapi::tbb::parallel_for(std::size_t{0}, chunks, [&](std::size_t chunk) {
    std::size_t position = chunk * grain;
    const std::size_t chunk_end = std::min(count, position + grain);
    // Skip the tail of a run that started in an earlier chunk.
    while (position < chunk_end && position > 0 && same_term(position - 1, position)) {
      ++position;
    }
    ChunkOutput& out = outputs[chunk];
    while (position < chunk_end) {
      std::complex<double> accumulated{0.0, 0.0};
      std::size_t next = position;
      while (next < count && (next == position || same_term(position, next))) {
        accumulated += coeffs[sorted_index[next]];
        ++next;
      }
      if (rule.survives(accumulated)) {
        emit_term(position, out.x, out.z);
        out.coeffs.push_back(accumulated);
      }
      position = next;
    }
  });

  std::vector<std::size_t> offsets(chunks + 1, 0);
  for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
    offsets[chunk + 1] = offsets[chunk] + outputs[chunk].coeffs.size();
  }
  SimplifiedTerms result;
  result.x.resize(offsets.back() * words);
  result.z.resize(offsets.back() * words);
  result.coeffs.resize(offsets.back());
  oneapi::tbb::parallel_for(std::size_t{0}, chunks, [&](std::size_t chunk) {
    const ChunkOutput& out = outputs[chunk];
    std::copy(out.x.begin(), out.x.end(), result.x.begin() + offsets[chunk] * words);
    std::copy(out.z.begin(), out.z.end(), result.z.begin() + offsets[chunk] * words);
    std::copy(out.coeffs.begin(), out.coeffs.end(), result.coeffs.begin() + offsets[chunk]);
  });
  return result;
}

template <std::size_t kWords>
SimplifiedTerms simplify_fixed(const PauliSum& op, const SimplifyDropRule& rule) {
  const std::size_t count = op.num_terms();
  const std::vector<std::uint64_t>& x = op.x_words();
  const std::vector<std::uint64_t>& z = op.z_words();
  std::vector<FixedKey<kWords>> keys(count);
  oneapi::tbb::parallel_for(std::size_t{0}, count, [&](std::size_t term) {
    FixedKey<kWords>& key = keys[term];
    for (std::size_t word = 0; word < kWords; ++word) {
      key.x[word] = x[term * kWords + word];
      key.z[word] = z[term * kWords + word];
    }
    key.index = term;
  });
  oneapi::tbb::parallel_sort(keys.begin(), keys.end(), fixed_key_less<kWords>);

  std::vector<std::size_t> sorted_index(count);
  oneapi::tbb::parallel_for(std::size_t{0}, count, [&](std::size_t position) {
    sorted_index[position] = keys[position].index;
  });
  return reduce_sorted_runs(
      count,
      kWords,
      op.coeffs(),
      sorted_index,
      rule,
      [&](std::size_t lhs, std::size_t rhs) { return fixed_key_same_term(keys[lhs], keys[rhs]); },
      [&](std::size_t position, std::vector<std::uint64_t>& out_x, std::vector<std::uint64_t>& out_z) {
        for (std::size_t word = 0; word < kWords; ++word) {
          out_x.push_back(keys[position].x[word]);
          out_z.push_back(keys[position].z[word]);
        }
      });
}

SimplifiedTerms simplify_generic(const PauliSum& op, const SimplifyDropRule& rule) {
  const std::size_t count = op.num_terms();
  const std::size_t words = op.words();
  const std::vector<std::uint64_t>& x = op.x_words();
  const std::vector<std::uint64_t>& z = op.z_words();
  std::vector<std::size_t> sorted_index(count);
  std::iota(sorted_index.begin(), sorted_index.end(), std::size_t{0});
  oneapi::tbb::parallel_sort(
      sorted_index.begin(), sorted_index.end(), [&](std::size_t lhs, std::size_t rhs) {
        const int comparison = compare_term_keys(x, z, words, lhs, rhs);
        return comparison != 0 ? comparison < 0 : lhs < rhs;
      });
  return reduce_sorted_runs(
      count,
      words,
      op.coeffs(),
      sorted_index,
      rule,
      [&](std::size_t lhs, std::size_t rhs) {
        return compare_term_keys(x, z, words, sorted_index[lhs], sorted_index[rhs]) == 0;
      },
      [&](std::size_t position, std::vector<std::uint64_t>& out_x, std::vector<std::uint64_t>& out_z) {
        const std::size_t offset = sorted_index[position] * words;
        out_x.insert(out_x.end(), x.begin() + offset, x.begin() + offset + words);
        out_z.insert(out_z.end(), z.begin() + offset, z.begin() + offset + words);
      });
}

}  // namespace

SimplifiedTerms simplify_tbb(const PauliSum& op, const SimplifyDropRule& rule) {
  switch (op.words()) {
    case 1:
      return simplify_fixed<1>(op, rule);
    case 2:
      return simplify_fixed<2>(op, rule);
    default:
      return simplify_generic(op, rule);
  }
}

}  // namespace wolfgang::detail

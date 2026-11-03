#include "detail/simplify_kernels.hpp"

#include "detail/packed_key.hpp"

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_sort.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
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

// Functors rather than function pointers so the sort's tasks can inline them.
template <std::size_t kWords>
struct FixedKeyLess {
  bool operator()(const FixedKey<kWords>& lhs, const FixedKey<kWords>& rhs) const noexcept {
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
};

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
// (a run may extend past the chunk end) and sums each run sequentially in
// sorted order, seeded with the run's first coefficient exactly as the scalar
// paths do, so output order and every floating-point sum, signed zeros
// included, match scalar simplify. Chunks keep only (sorted position,
// coefficient) pairs; packed words are written once, in the final copy.
template <typename SameTerm, typename CoeffAt, typename WriteTerm>
SimplifiedTerms reduce_sorted_runs(
    std::size_t count,
    std::size_t words,
    const SimplifyDropRule& rule,
    SameTerm same_term,
    CoeffAt coeff_at,
    WriteTerm write_term) {
  struct ChunkOutput {
    std::vector<std::size_t> positions;
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
      std::complex<double> accumulated = coeff_at(position);
      std::size_t next = position + 1;
      while (next < count && same_term(position, next)) {
        accumulated += coeff_at(next);
        ++next;
      }
      if (rule.survives(accumulated)) {
        out.positions.push_back(position);
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
    for (std::size_t i = 0; i < out.positions.size(); ++i) {
      const std::size_t term = offsets[chunk] + i;
      write_term(out.positions[i], result.x.data() + term * words, result.z.data() + term * words);
      result.coeffs[term] = out.coeffs[i];
    }
  });
  return result;
}

template <std::size_t kWords>
SimplifiedTerms simplify_fixed(const PauliSum& op, const SimplifyDropRule& rule) {
  const std::size_t count = op.num_terms();
  const std::vector<std::uint64_t>& x = op.x_words();
  const std::vector<std::uint64_t>& z = op.z_words();
  const std::vector<std::complex<double>>& coeffs = op.coeffs();
  // Default-initialized (not zeroed): the parallel fill writes every entry.
  std::unique_ptr<FixedKey<kWords>[]> keys(new FixedKey<kWords>[count]);
  oneapi::tbb::parallel_for(std::size_t{0}, count, [&](std::size_t term) {
    FixedKey<kWords>& key = keys[term];
    for (std::size_t word = 0; word < kWords; ++word) {
      key.x[word] = x[term * kWords + word];
      key.z[word] = z[term * kWords + word];
    }
    key.index = term;
  });
  oneapi::tbb::parallel_sort(keys.get(), keys.get() + count, FixedKeyLess<kWords>{});
  return reduce_sorted_runs(
      count,
      kWords,
      rule,
      [&](std::size_t lhs, std::size_t rhs) { return fixed_key_same_term(keys[lhs], keys[rhs]); },
      [&](std::size_t position) { return coeffs[keys[position].index]; },
      [&](std::size_t position, std::uint64_t* out_x, std::uint64_t* out_z) {
        for (std::size_t word = 0; word < kWords; ++word) {
          out_x[word] = keys[position].x[word];
          out_z[word] = keys[position].z[word];
        }
      });
}

SimplifiedTerms simplify_generic(const PauliSum& op, const SimplifyDropRule& rule) {
  const std::size_t count = op.num_terms();
  const std::size_t words = op.words();
  const std::vector<std::uint64_t>& x = op.x_words();
  const std::vector<std::uint64_t>& z = op.z_words();
  const std::vector<std::complex<double>>& coeffs = op.coeffs();
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
      rule,
      [&](std::size_t lhs, std::size_t rhs) {
        return compare_term_keys(x, z, words, sorted_index[lhs], sorted_index[rhs]) == 0;
      },
      [&](std::size_t position) { return coeffs[sorted_index[position]]; },
      [&](std::size_t position, std::uint64_t* out_x, std::uint64_t* out_z) {
        const std::size_t offset = sorted_index[position] * words;
        std::copy(x.begin() + offset, x.begin() + offset + words, out_x);
        std::copy(z.begin() + offset, z.begin() + offset + words, out_z);
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

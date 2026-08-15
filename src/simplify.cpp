#include "wolfgang/pauli_sum.hpp"

#include "detail/packed_key.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace wolfgang {
namespace {

struct TermKey1 {
  std::uint64_t x;
  std::uint64_t z;
  std::size_t index;
};

struct TermKey2 {
  std::uint64_t x0;
  std::uint64_t z0;
  std::uint64_t x1;
  std::uint64_t z1;
  std::size_t index;
};

using detail::PackedKey1;
using detail::PackedKey1Hash;
using detail::PackedKey2;
using detail::PackedKey2Hash;
using detail::less_packed_key1;
using detail::less_packed_key2;

bool less_term_key1(const TermKey1& lhs, const TermKey1& rhs) noexcept {
  if (lhs.x != rhs.x) {
    return lhs.x < rhs.x;
  }
  if (lhs.z != rhs.z) {
    return lhs.z < rhs.z;
  }
  return lhs.index < rhs.index;
}

bool less_term_key2(const TermKey2& lhs, const TermKey2& rhs) noexcept {
  if (lhs.x0 != rhs.x0) {
    return lhs.x0 < rhs.x0;
  }
  if (lhs.z0 != rhs.z0) {
    return lhs.z0 < rhs.z0;
  }
  if (lhs.x1 != rhs.x1) {
    return lhs.x1 < rhs.x1;
  }
  if (lhs.z1 != rhs.z1) {
    return lhs.z1 < rhs.z1;
  }
  return lhs.index < rhs.index;
}

double sample_duplicate_rate(std::size_t sample_size, std::size_t unique_terms) noexcept {
  if (sample_size == 0) {
    return 0.0;
  }
  return 1.0 - (static_cast<double>(unique_terms) / static_cast<double>(sample_size));
}

double sample_duplicate_rate_words1(
    const std::vector<std::uint64_t>& x,
    const std::vector<std::uint64_t>& z,
    std::size_t sample_size) {
  std::unordered_set<PackedKey1, PackedKey1Hash> sample;
  sample.reserve(sample_size);
  for (std::size_t term = 0; term < sample_size; ++term) {
    sample.insert({x[term], z[term]});
  }
  return sample_duplicate_rate(sample_size, sample.size());
}

double sample_duplicate_rate_words2(
    const std::vector<std::uint64_t>& x,
    const std::vector<std::uint64_t>& z,
    std::size_t sample_size) {
  std::unordered_set<PackedKey2, PackedKey2Hash> sample;
  sample.reserve(sample_size);
  for (std::size_t term = 0; term < sample_size; ++term) {
    const std::size_t offset = term * 2;
    sample.insert({x[offset], z[offset], x[offset + 1], z[offset + 1]});
  }
  return sample_duplicate_rate(sample_size, sample.size());
}

bool should_hash_accumulate_from_sample(double duplicate_rate) noexcept {
  return duplicate_rate >= 0.25;
}

bool should_skip_hash_accumulation_from_sample(double duplicate_rate) noexcept {
  return duplicate_rate <= 0.05;
}

bool should_hash_accumulate_words1(
    const std::vector<std::uint64_t>& x,
    const std::vector<std::uint64_t>& z,
    std::size_t num_terms) {
  if (num_terms < 4096) {
    return false;
  }
  const double initial_rate =
      sample_duplicate_rate_words1(x, z, std::min<std::size_t>(num_terms, 512));
  if (should_hash_accumulate_from_sample(initial_rate)) {
    return true;
  }
  if (should_skip_hash_accumulation_from_sample(initial_rate)) {
    return false;
  }
  return should_hash_accumulate_from_sample(
      sample_duplicate_rate_words1(x, z, std::min<std::size_t>(num_terms, 4096)));
}

bool should_hash_accumulate_words2(
    const std::vector<std::uint64_t>& x,
    const std::vector<std::uint64_t>& z,
    std::size_t num_terms) {
  if (num_terms < 4096) {
    return false;
  }
  const double initial_rate =
      sample_duplicate_rate_words2(x, z, std::min<std::size_t>(num_terms, 512));
  if (should_hash_accumulate_from_sample(initial_rate)) {
    return true;
  }
  if (should_skip_hash_accumulation_from_sample(initial_rate)) {
    return false;
  }
  return should_hash_accumulate_from_sample(
      sample_duplicate_rate_words2(x, z, std::min<std::size_t>(num_terms, 4096)));
}

}  // namespace

PauliSum PauliSum::simplify(double atol, double rtol) const {
  if (atol < 0.0 || rtol < 0.0 || !std::isfinite(atol) || !std::isfinite(rtol)) {
    throw std::invalid_argument("simplify tolerances must be non-negative finite values");
  }

  if (num_terms_ == 0) {
    return PauliSum::empty(num_qubits_);
  }

  double max_abs_input = 0.0;
  for (const std::complex<double>& coeff : coeffs_) {
    max_abs_input = std::max(max_abs_input, std::abs(coeff));
  }
  const double drop_threshold = atol + rtol * max_abs_input;

  if (words_ == 1) {
    if (should_hash_accumulate_words1(x_, z_, num_terms_)) {
      std::unordered_map<PackedKey1, std::complex<double>, PackedKey1Hash> accumulators;
      accumulators.max_load_factor(0.7F);
      accumulators.reserve(num_terms_);
      for (std::size_t term = 0; term < num_terms_; ++term) {
        auto [iterator, inserted] = accumulators.try_emplace(
            PackedKey1{x_[term], z_[term]},
            coeffs_[term]);
        if (!inserted) {
          iterator->second += coeffs_[term];
        }
      }

      std::vector<std::pair<PackedKey1, std::complex<double>>> survivors;
      survivors.reserve(accumulators.size());
      for (const auto& [key, coeff] : accumulators) {
        if (std::abs(coeff) > drop_threshold) {
          survivors.push_back({key, coeff});
        }
      }
      std::sort(
          survivors.begin(),
          survivors.end(),
          [](const auto& lhs, const auto& rhs) noexcept {
            return less_packed_key1(lhs.first, rhs.first);
          });

      std::vector<std::uint64_t> out_x;
      std::vector<std::uint64_t> out_z;
      std::vector<std::complex<double>> out_coeffs;
      out_x.reserve(survivors.size());
      out_z.reserve(survivors.size());
      out_coeffs.reserve(survivors.size());
      for (const auto& [key, coeff] : survivors) {
        out_x.push_back(key.x);
        out_z.push_back(key.z);
        out_coeffs.push_back(coeff);
      }

      if (out_coeffs.empty()) {
        return PauliSum::empty(num_qubits_);
      }
      const std::size_t out_terms = out_coeffs.size();
      return PauliSum(
          num_qubits_,
          words_,
          out_terms,
          std::move(out_x),
          std::move(out_z),
          std::move(out_coeffs));
    }

    std::vector<TermKey1> keys;
    keys.reserve(num_terms_);
    for (std::size_t term = 0; term < num_terms_; ++term) {
      keys.push_back({x_[term], z_[term], term});
    }
    std::sort(keys.begin(), keys.end(), less_term_key1);

    std::vector<std::uint64_t> out_x;
    std::vector<std::uint64_t> out_z;
    std::vector<std::complex<double>> out_coeffs;
    out_x.reserve(num_terms_);
    out_z.reserve(num_terms_);
    out_coeffs.reserve(num_terms_);

    std::size_t sorted_position = 0;
    while (sorted_position < keys.size()) {
      const std::uint64_t key_x = keys[sorted_position].x;
      const std::uint64_t key_z = keys[sorted_position].z;
      std::complex<double> accumulated{0.0, 0.0};

      std::size_t next_position = sorted_position;
      while (next_position < keys.size() &&
             keys[next_position].x == key_x &&
             keys[next_position].z == key_z) {
        accumulated += coeffs_[keys[next_position].index];
        ++next_position;
      }

      if (std::abs(accumulated) > drop_threshold) {
        out_x.push_back(key_x);
        out_z.push_back(key_z);
        out_coeffs.push_back(accumulated);
      }

      sorted_position = next_position;
    }

    if (out_coeffs.empty()) {
      return PauliSum::empty(num_qubits_);
    }
    const std::size_t out_terms = out_coeffs.size();
    return PauliSum(
        num_qubits_,
        words_,
        out_terms,
        std::move(out_x),
        std::move(out_z),
        std::move(out_coeffs));
  }

  if (words_ == 2) {
    if (should_hash_accumulate_words2(x_, z_, num_terms_)) {
      std::unordered_map<PackedKey2, std::complex<double>, PackedKey2Hash> accumulators;
      accumulators.max_load_factor(0.7F);
      accumulators.reserve(num_terms_);
      for (std::size_t term = 0; term < num_terms_; ++term) {
        const std::size_t offset = term * 2;
        auto [iterator, inserted] = accumulators.try_emplace(
            PackedKey2{x_[offset], z_[offset], x_[offset + 1], z_[offset + 1]},
            coeffs_[term]);
        if (!inserted) {
          iterator->second += coeffs_[term];
        }
      }

      std::vector<std::pair<PackedKey2, std::complex<double>>> survivors;
      survivors.reserve(accumulators.size());
      for (const auto& [key, coeff] : accumulators) {
        if (std::abs(coeff) > drop_threshold) {
          survivors.push_back({key, coeff});
        }
      }
      std::sort(
          survivors.begin(),
          survivors.end(),
          [](const auto& lhs, const auto& rhs) noexcept {
            return less_packed_key2(lhs.first, rhs.first);
          });

      std::vector<std::uint64_t> out_x;
      std::vector<std::uint64_t> out_z;
      std::vector<std::complex<double>> out_coeffs;
      out_x.reserve(survivors.size() * 2);
      out_z.reserve(survivors.size() * 2);
      out_coeffs.reserve(survivors.size());
      for (const auto& [key, coeff] : survivors) {
        out_x.push_back(key.x0);
        out_x.push_back(key.x1);
        out_z.push_back(key.z0);
        out_z.push_back(key.z1);
        out_coeffs.push_back(coeff);
      }

      if (out_coeffs.empty()) {
        return PauliSum::empty(num_qubits_);
      }
      const std::size_t out_terms = out_coeffs.size();
      return PauliSum(
          num_qubits_,
          words_,
          out_terms,
          std::move(out_x),
          std::move(out_z),
          std::move(out_coeffs));
    }

    std::vector<TermKey2> keys;
    keys.reserve(num_terms_);
    for (std::size_t term = 0; term < num_terms_; ++term) {
      const std::size_t offset = term * 2;
      keys.push_back({x_[offset], z_[offset], x_[offset + 1], z_[offset + 1], term});
    }
    std::sort(keys.begin(), keys.end(), less_term_key2);

    std::vector<std::uint64_t> out_x;
    std::vector<std::uint64_t> out_z;
    std::vector<std::complex<double>> out_coeffs;
    out_x.reserve(x_.size());
    out_z.reserve(z_.size());
    out_coeffs.reserve(num_terms_);

    std::size_t sorted_position = 0;
    while (sorted_position < keys.size()) {
      const TermKey2& key = keys[sorted_position];
      std::complex<double> accumulated{0.0, 0.0};

      std::size_t next_position = sorted_position;
      while (next_position < keys.size() &&
             keys[next_position].x0 == key.x0 &&
             keys[next_position].z0 == key.z0 &&
             keys[next_position].x1 == key.x1 &&
             keys[next_position].z1 == key.z1) {
        accumulated += coeffs_[keys[next_position].index];
        ++next_position;
      }

      if (std::abs(accumulated) > drop_threshold) {
        out_x.push_back(key.x0);
        out_x.push_back(key.x1);
        out_z.push_back(key.z0);
        out_z.push_back(key.z1);
        out_coeffs.push_back(accumulated);
      }

      sorted_position = next_position;
    }

    if (out_coeffs.empty()) {
      return PauliSum::empty(num_qubits_);
    }
    const std::size_t out_terms = out_coeffs.size();
    return PauliSum(
        num_qubits_,
        words_,
        out_terms,
        std::move(out_x),
        std::move(out_z),
        std::move(out_coeffs));
  }

  std::vector<std::size_t> indices(num_terms_);
  std::iota(indices.begin(), indices.end(), std::size_t{0});
  std::sort(indices.begin(), indices.end(), [this](std::size_t lhs, std::size_t rhs) {
    const int comparison = detail::compare_term_keys(x_, z_, words_, lhs, rhs);
    if (comparison != 0) {
      return comparison < 0;
    }
    // Equal keys are accumulated in construction order. This keeps floating
    // point summation deterministic across standard-library sort choices.
    return lhs < rhs;
  });

  std::vector<std::uint64_t> out_x;
  std::vector<std::uint64_t> out_z;
  std::vector<std::complex<double>> out_coeffs;
  out_x.reserve(x_.size());
  out_z.reserve(z_.size());
  out_coeffs.reserve(num_terms_);

  std::size_t sorted_position = 0;
  while (sorted_position < indices.size()) {
    const std::size_t first_term = indices[sorted_position];
    std::complex<double> accumulated{0.0, 0.0};

    std::size_t next_position = sorted_position;
    while (next_position < indices.size() &&
           detail::compare_term_keys(x_, z_, words_, first_term, indices[next_position]) == 0) {
      accumulated += coeffs_[indices[next_position]];
      ++next_position;
    }

    if (std::abs(accumulated) > drop_threshold) {
      const std::size_t first_offset = first_term * words_;
      for (std::size_t word = 0; word < words_; ++word) {
        out_x.push_back(x_[first_offset + word]);
        out_z.push_back(z_[first_offset + word]);
      }
      out_coeffs.push_back(accumulated);
    }

    sorted_position = next_position;
  }

  if (out_coeffs.empty()) {
    return PauliSum::empty(num_qubits_);
  }

  const std::size_t out_terms = out_coeffs.size();
  return PauliSum(
      num_qubits_,
      words_,
      out_terms,
      std::move(out_x),
      std::move(out_z),
      std::move(out_coeffs));
}

}  // namespace wolfgang

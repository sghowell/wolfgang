#include "simplify_hip.hip.hpp"
#include "hip/pool_allocator.hip.hpp"

#include <thrust/complex.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace wolfgang {

using hip_detail::check_hip;
using hip_detail::copy_device_to_device;
using hip_detail::copy_to_device;
using hip_detail::copy_to_host;
using hip_detail::pool_allocate;
using hip_detail::pooled_policy;
using hip_detail::PooledArray;
using hip_detail::ScopedHipDevice;
using hip_detail::validate_simplify_tolerances;

namespace {

enum class DuplicateReductionStrategy {
  kRocThrustDefault,
};

enum class GenericReductionStrategy {
  kParallelReduceByKey,
  kSerialKernel,
};

struct HipKey1 {
  std::uint64_t x;
  std::uint64_t z;
};

struct HipKey2 {
  std::uint64_t x0;
  std::uint64_t z0;
  std::uint64_t x1;
  std::uint64_t z1;
};

struct PackedKey32FromTerm {
  const std::uint64_t* x;
  const std::uint64_t* z;

  __host__ __device__ std::uint64_t operator()(std::size_t term) const noexcept {
    return (x[term] << 32U) | (z[term] & 0xFFFF'FFFFULL);
  }
};

struct PackedKey32ToX {
  __host__ __device__ std::uint64_t operator()(std::uint64_t key) const noexcept {
    return key >> 32U;
  }
};

struct PackedKey32ToZ {
  __host__ __device__ std::uint64_t operator()(std::uint64_t key) const noexcept {
    return key & 0xFFFF'FFFFULL;
  }
};

struct Key1Less {
  __host__ __device__ bool operator()(const HipKey1& lhs, const HipKey1& rhs) const noexcept {
    if (lhs.x != rhs.x) {
      return lhs.x < rhs.x;
    }
    return lhs.z < rhs.z;
  }
};

struct Key1Equal {
  __host__ __device__ bool operator()(const HipKey1& lhs, const HipKey1& rhs) const noexcept {
    return lhs.x == rhs.x && lhs.z == rhs.z;
  }
};

struct Key2Less {
  __host__ __device__ bool operator()(const HipKey2& lhs, const HipKey2& rhs) const noexcept {
    if (lhs.x0 != rhs.x0) {
      return lhs.x0 < rhs.x0;
    }
    if (lhs.z0 != rhs.z0) {
      return lhs.z0 < rhs.z0;
    }
    if (lhs.x1 != rhs.x1) {
      return lhs.x1 < rhs.x1;
    }
    return lhs.z1 < rhs.z1;
  }
};

struct Key2Equal {
  __host__ __device__ bool operator()(const HipKey2& lhs, const HipKey2& rhs) const noexcept {
    return lhs.x0 == rhs.x0 && lhs.z0 == rhs.z0 && lhs.x1 == rhs.x1 && lhs.z1 == rhs.z1;
  }
};

struct Key1FromTerm {
  const std::uint64_t* x;
  const std::uint64_t* z;

  __host__ __device__ HipKey1 operator()(std::size_t term) const noexcept {
    return {x[term], z[term]};
  }
};

struct Key2FromTerm {
  const std::uint64_t* x;
  const std::uint64_t* z;

  __host__ __device__ HipKey2 operator()(std::size_t term) const noexcept {
    const std::size_t offset = term * 2;
    return {x[offset], z[offset], x[offset + 1], z[offset + 1]};
  }
};

struct Key1ToX {
  __host__ __device__ std::uint64_t operator()(const HipKey1& key) const noexcept {
    return key.x;
  }
};

struct Key1ToZ {
  __host__ __device__ std::uint64_t operator()(const HipKey1& key) const noexcept {
    return key.z;
  }
};

struct Key2ToXWord {
  const HipKey2* keys;

  __host__ __device__ std::uint64_t operator()(std::size_t word_index) const noexcept {
    const HipKey2& key = keys[word_index / 2];
    return (word_index & 1U) == 0 ? key.x0 : key.x1;
  }
};

struct Key2ToZWord {
  const HipKey2* keys;

  __host__ __device__ std::uint64_t operator()(std::size_t word_index) const noexcept {
    const HipKey2& key = keys[word_index / 2];
    return (word_index & 1U) == 0 ? key.z0 : key.z1;
  }
};

struct ComplexHalfAbs {
  __host__ __device__ double operator()(thrust::complex<double> value) const noexcept {
    return ::hypot(value.real() * 0.5, value.imag() * 0.5);
  }
};

__host__ __device__ inline bool components_finite(thrust::complex<double> value) noexcept {
  return ::isfinite(value.real()) && ::isfinite(value.imag());
}

// |c|, or NaN when a component is not finite. With MaxPropagatingNaN one
// reduction yields both the drop-rule scale and the CPU's finite-input check.
// A finite coefficient whose magnitude overflows maps to +inf, not NaN.
struct ComplexAbsOrNaN {
  __host__ __device__ double operator()(thrust::complex<double> value) const noexcept {
    return components_finite(value) ? thrust::abs(value) : ::nan("");
  }
};

struct MaxPropagatingNaN {
  __host__ __device__ double operator()(double lhs, double rhs) const noexcept {
    if (lhs != lhs) {
      return lhs;
    }
    if (rhs != rhs) {
      return rhs;
    }
    return lhs < rhs ? rhs : lhs;
  }
};

// Mirrors the CPU drop rule (src/detail/simplify_numeric.hpp). Finite complex128
// components can have a magnitude above DBL_MAX: at rtol == 0 the threshold is
// atol itself, so 0 * inf never produces NaN, and when atol + rtol * max
// overflows, magnitudes and the threshold are compared in half units instead.
struct CoeffSurvives {
  double drop_threshold;
  double half_threshold = 0.0;
  bool use_half_units = false;
  // Device flag set when a reduced coefficient is not finite (the CPU raises
  // OverflowError). Tree-ordered sums can even turn inf + -inf into NaN, which
  // the magnitude test alone would drop silently. Every reduction path passes
  // its sums through this predicate, so one flag covers them all.
  int* overflowed = nullptr;

  __host__ __device__ bool operator()(thrust::complex<double> value) const noexcept {
    if (!components_finite(value)) {
      if (overflowed != nullptr) {
        *overflowed = 1;
      }
      return false;
    }
    return use_half_units ? ::hypot(value.real() * 0.5, value.imag() * 0.5) > half_threshold
                          : thrust::abs(value) > drop_threshold;
  }
};

template <typename HalfMagnitudeMax>
CoeffSurvives survival_rule(
    double atol,
    double rtol,
    double max_abs_input,
    HalfMagnitudeMax&& max_half_magnitude) {
  CoeffSurvives rule{rtol == 0.0 ? atol : atol + rtol * max_abs_input};
  if (!std::isfinite(rule.drop_threshold)) {
    rule.half_threshold = atol * 0.5 + rtol * max_half_magnitude();
    rule.drop_threshold = rule.half_threshold * 2.0;
    rule.use_half_units = true;
  }
  return rule;
}

[[noreturn]] void throw_accumulation_overflow() {
  throw std::overflow_error("simplify coefficient accumulation overflowed complex128");
}

DuplicateReductionStrategy duplicate_reduction_strategy_from_env() {
  const char* value = std::getenv("WOLFGANG_HIP_BENCH_DUPLICATE_REDUCTION");
  if (value == nullptr || std::string(value).empty() ||
      std::string(value) == hip_detail::kHipSimplifyDefaultStrategy) {
    return DuplicateReductionStrategy::kRocThrustDefault;
  }
  const std::string setting(value);
  if (setting == hip_detail::kHipSimplifyCustomPackedKeyStrategy ||
      setting == "hipcub_radix_sort_reduce") {
    throw std::invalid_argument(
        "WOLFGANG_HIP_BENCH_DUPLICATE_REDUCTION selected an unavailable "
        "Campaign 4 candidate; only rocthrust_default is executable");
  }
  throw std::invalid_argument(
      "WOLFGANG_HIP_BENCH_DUPLICATE_REDUCTION must be rocthrust_default, "
      "hipcub_radix_sort_reduce, or custom_packed_key");
}

GenericReductionStrategy generic_reduction_strategy_from_env() {
  const char* value = std::getenv("WOLFGANG_HIP_BENCH_GENERIC_MULTIWORD_REDUCTION");
  if (value == nullptr || std::string(value).empty() ||
      std::string(value) == hip_detail::kHipSimplifyGenericParallelStrategy ||
      std::string(value) == "reduce_by_key") {
    return GenericReductionStrategy::kParallelReduceByKey;
  }
  const std::string setting(value);
  if (setting == hip_detail::kHipSimplifyGenericSerialStrategy) {
    return GenericReductionStrategy::kSerialKernel;
  }
  throw std::invalid_argument(
      "WOLFGANG_HIP_BENCH_GENERIC_MULTIWORD_REDUCTION must be "
      "reduce_by_key, rocthrust_generic_parallel_reduce_by_key, or serial_kernel");
}

struct GenericTermIndexLess {
  const std::uint64_t* x;
  const std::uint64_t* z;
  std::size_t words;

  __host__ __device__ bool operator()(std::size_t lhs, std::size_t rhs) const noexcept {
    const std::size_t lhs_offset = lhs * words;
    const std::size_t rhs_offset = rhs * words;
    for (std::size_t word = 0; word < words; ++word) {
      const std::uint64_t lhs_x = x[lhs_offset + word];
      const std::uint64_t rhs_x = x[rhs_offset + word];
      if (lhs_x != rhs_x) {
        return lhs_x < rhs_x;
      }
      const std::uint64_t lhs_z = z[lhs_offset + word];
      const std::uint64_t rhs_z = z[rhs_offset + word];
      if (lhs_z != rhs_z) {
        return lhs_z < rhs_z;
      }
    }
    return lhs < rhs;
  }
};

struct GenericTermIndexEqual {
  const std::uint64_t* x;
  const std::uint64_t* z;
  std::size_t words;

  __host__ __device__ bool operator()(std::size_t lhs, std::size_t rhs) const noexcept {
    const std::size_t lhs_offset = lhs * words;
    const std::size_t rhs_offset = rhs * words;
    for (std::size_t word = 0; word < words; ++word) {
      if (x[lhs_offset + word] != x[rhs_offset + word] ||
          z[lhs_offset + word] != z[rhs_offset + word]) {
        return false;
      }
    }
    return true;
  }
};

struct GenericRepresentativeToXWord {
  const std::uint64_t* x;
  const std::size_t* representative_indices;
  std::size_t words;

  __host__ __device__ std::uint64_t operator()(std::size_t word_index) const noexcept {
    const std::size_t output_term = word_index / words;
    const std::size_t word = word_index - output_term * words;
    return x[representative_indices[output_term] * words + word];
  }
};

struct GenericRepresentativeToZWord {
  const std::uint64_t* z;
  const std::size_t* representative_indices;
  std::size_t words;

  __host__ __device__ std::uint64_t operator()(std::size_t word_index) const noexcept {
    const std::size_t output_term = word_index / words;
    const std::size_t word = word_index - output_term * words;
    return z[representative_indices[output_term] * words + word];
  }
};

__device__ bool generic_terms_equal(
    const std::uint64_t* x,
    const std::uint64_t* z,
    std::size_t words,
    std::size_t lhs,
    std::size_t rhs) {
  const std::size_t lhs_offset = lhs * words;
  const std::size_t rhs_offset = rhs * words;
  for (std::size_t word = 0; word < words; ++word) {
    if (x[lhs_offset + word] != x[rhs_offset + word] ||
        z[lhs_offset + word] != z[rhs_offset + word]) {
      return false;
    }
  }
  return true;
}

__global__ void reduce_sorted_generic_terms_kernel(
    const std::uint64_t* x,
    const std::uint64_t* z,
    const thrust::complex<double>* coeffs,
    const std::size_t* sorted_indices,
    std::size_t num_terms,
    std::size_t words,
    CoeffSurvives survives,
    std::uint64_t* out_x,
    std::uint64_t* out_z,
    thrust::complex<double>* out_coeffs,
    std::size_t* out_count) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  if (num_terms == 0) {
    *out_count = 0;
    return;
  }

  std::size_t output_term = 0;
  std::size_t group_start_index = sorted_indices[0];
  thrust::complex<double> accumulator = coeffs[group_start_index];
  for (std::size_t sorted_pos = 1; sorted_pos < num_terms; ++sorted_pos) {
    const std::size_t current_index = sorted_indices[sorted_pos];
    if (generic_terms_equal(x, z, words, group_start_index, current_index)) {
      accumulator += coeffs[current_index];
      continue;
    }

    if (survives(accumulator)) {
      const std::size_t out_offset = output_term * words;
      const std::size_t source_offset = group_start_index * words;
      for (std::size_t word = 0; word < words; ++word) {
        out_x[out_offset + word] = x[source_offset + word];
        out_z[out_offset + word] = z[source_offset + word];
      }
      out_coeffs[output_term] = accumulator;
      ++output_term;
    }

    group_start_index = current_index;
    accumulator = coeffs[current_index];
  }

  if (survives(accumulator)) {
    const std::size_t out_offset = output_term * words;
    const std::size_t source_offset = group_start_index * words;
    for (std::size_t word = 0; word < words; ++word) {
      out_x[out_offset + word] = x[source_offset + word];
      out_z[out_offset + word] = z[source_offset + word];
    }
    out_coeffs[output_term] = accumulator;
    ++output_term;
  }
  *out_count = output_term;
}

}  // namespace

DevicePauliSum DevicePauliSum::simplify(double atol, double rtol) const {
  if (!impl_) {
    throw std::runtime_error("DevicePauliSum is empty or moved-from");
  }
  validate_simplify_tolerances(atol, rtol);

  auto make_empty = [this]() {
    auto out = std::make_unique<Impl>();
    out->num_qubits = impl_->num_qubits;
    out->words = impl_->words;
    out->num_terms = 0;
    out->device_ordinal = impl_->device_ordinal;
    return DevicePauliSum(std::move(out));
  };

  if (impl_->num_terms == 0) {
    return make_empty();
  }

  ScopedHipDevice guard(impl_->device_ordinal);
  if (impl_->num_terms == 1) {
    // One term has nothing to merge; only the drop rule applies, and the
    // largest input magnitude is the term's own. Skipping sort and reduction
    // removes their fixed setup cost from this common degenerate case.
    thrust::complex<double> coeff;
    copy_to_host(&coeff, impl_->coeffs, 1, "single-term coefficient");
    if (!components_finite(coeff)) {
      throw std::invalid_argument("simplify requires finite coefficients");
    }
    const CoeffSurvives survives =
        survival_rule(atol, rtol, thrust::abs(coeff), [&] { return ComplexHalfAbs{}(coeff); });
    if (!survives(coeff)) {
      return make_empty();
    }
    auto out = std::make_unique<Impl>();
    out->num_qubits = impl_->num_qubits;
    out->words = impl_->words;
    out->num_terms = 1;
    out->device_ordinal = impl_->device_ordinal;
    if (impl_->words > 0) {
      pool_allocate(out->x, impl_->words, "simplified x words");
      pool_allocate(out->z, impl_->words, "simplified z words");
      copy_device_to_device(out->x, impl_->x, impl_->words, "simplified x words");
      copy_device_to_device(out->z, impl_->z, impl_->words, "simplified z words");
    }
    pool_allocate(out->coeffs, 1, "simplified coefficients");
    copy_device_to_device(out->coeffs, impl_->coeffs, 1, "simplified coefficients");
    check_hip(hipDeviceSynchronize(), "synchronize after HIP simplify");
    return DevicePauliSum(std::move(out));
  }

  auto coeff_ptr = thrust::device_pointer_cast(impl_->coeffs);
  const double max_abs_input = thrust::transform_reduce(pooled_policy(), 
      coeff_ptr,
      coeff_ptr + impl_->num_terms,
      ComplexAbsOrNaN{},
      0.0,
      MaxPropagatingNaN{});
  if (max_abs_input != max_abs_input) {
    throw std::invalid_argument("simplify requires finite coefficients");
  }
  CoeffSurvives survives = survival_rule(atol, rtol, max_abs_input, [&] {
    return thrust::transform_reduce(pooled_policy(), 
        coeff_ptr,
        coeff_ptr + impl_->num_terms,
        ComplexHalfAbs{},
        0.0,
        thrust::maximum<double>{});
  });
  const DuplicateReductionStrategy duplicate_strategy = duplicate_reduction_strategy_from_env();
  (void)duplicate_strategy;

  if (impl_->words == 0) {
    const thrust::complex<double> accumulator =
        thrust::reduce(pooled_policy(), coeff_ptr, coeff_ptr + impl_->num_terms, thrust::complex<double>{0.0, 0.0});
    if (!components_finite(accumulator)) {
      throw_accumulation_overflow();
    }
    if (!survives(accumulator)) {
      return make_empty();
    }
    auto out = std::make_unique<Impl>();
    out->num_qubits = impl_->num_qubits;
    out->words = impl_->words;
    out->num_terms = 1;
    out->device_ordinal = impl_->device_ordinal;
    pool_allocate(out->coeffs, 1, "simplified coefficients");
    copy_to_device(out->coeffs, &accumulator, 1, "simplified coefficients");
    check_hip(hipDeviceSynchronize(), "synchronize after HIP simplify");
    return DevicePauliSum(std::move(out));
  }

  PooledArray<int> overflow_flag(1);
  check_hip(
      hipMemsetAsync(thrust::raw_pointer_cast(overflow_flag.data()), 0, sizeof(int), hip_detail::kPoolStream),
      "clear simplify overflow flag");
  survives.overflowed = thrust::raw_pointer_cast(overflow_flag.data());
  // Called once the predicate has seen every reduced coefficient.
  const auto check_overflow = [&survives] {
    int overflowed = 0;
    copy_to_host(&overflowed, survives.overflowed, 1, "simplify overflow flag");
    if (overflowed != 0) {
      throw_accumulation_overflow();
    }
  };

  if (impl_->words == 1 && impl_->num_qubits <= 32) {
    PooledArray<std::uint64_t> keys(impl_->num_terms);
    PooledArray<thrust::complex<double>> values(impl_->num_terms);
    auto counting = thrust::make_counting_iterator<std::size_t>(0);
    thrust::transform(pooled_policy(), 
        counting,
        counting + impl_->num_terms,
        keys.begin(),
        PackedKey32FromTerm{impl_->x, impl_->z});
    thrust::copy_n(pooled_policy(), coeff_ptr, impl_->num_terms, values.begin());
    thrust::stable_sort_by_key(pooled_policy(), keys.begin(), keys.end(), values.begin());

    PooledArray<std::uint64_t> reduced_keys(impl_->num_terms);
    PooledArray<thrust::complex<double>> reduced_values(impl_->num_terms);
    auto reduced_end = thrust::reduce_by_key(pooled_policy(), 
        keys.begin(),
        keys.end(),
        values.begin(),
        reduced_keys.begin(),
        reduced_values.begin(),
        thrust::equal_to<std::uint64_t>{},
        thrust::plus<thrust::complex<double>>{});
    const std::size_t reduced_terms =
        static_cast<std::size_t>(reduced_end.first - reduced_keys.begin());

    PooledArray<std::uint64_t> survivor_keys(reduced_terms);
    PooledArray<thrust::complex<double>> survivor_values(reduced_terms);
    auto reduced_zip = thrust::make_zip_iterator(
        thrust::make_tuple(reduced_keys.begin(), reduced_values.begin()));
    auto survivor_zip = thrust::make_zip_iterator(
        thrust::make_tuple(survivor_keys.begin(), survivor_values.begin()));
    auto survivor_end = thrust::copy_if(pooled_policy(), 
        reduced_zip,
        reduced_zip + reduced_terms,
        reduced_values.begin(),
        survivor_zip,
        survives);
    const std::size_t survivors = static_cast<std::size_t>(survivor_end - survivor_zip);
    check_overflow();
    if (survivors == 0) {
      return make_empty();
    }

    auto out = std::make_unique<Impl>();
    out->num_qubits = impl_->num_qubits;
    out->words = impl_->words;
    out->num_terms = survivors;
    out->device_ordinal = impl_->device_ordinal;
    pool_allocate(out->x, survivors, "simplified x words");
    pool_allocate(out->z, survivors, "simplified z words");
    pool_allocate(out->coeffs, survivors, "simplified coefficients");
    thrust::transform(pooled_policy(), 
        survivor_keys.begin(),
        survivor_keys.begin() + survivors,
        thrust::device_pointer_cast(out->x),
        PackedKey32ToX{});
    thrust::transform(pooled_policy(), 
        survivor_keys.begin(),
        survivor_keys.begin() + survivors,
        thrust::device_pointer_cast(out->z),
        PackedKey32ToZ{});
    thrust::copy_n(pooled_policy(), survivor_values.begin(), survivors, thrust::device_pointer_cast(out->coeffs));
    check_hip(hipDeviceSynchronize(), "synchronize after HIP simplify");
    return DevicePauliSum(std::move(out));
  }

  if (impl_->words == 1) {
    PooledArray<HipKey1> keys(impl_->num_terms);
    PooledArray<thrust::complex<double>> values(impl_->num_terms);
    auto counting = thrust::make_counting_iterator<std::size_t>(0);
    thrust::transform(pooled_policy(), 
        counting,
        counting + impl_->num_terms,
        keys.begin(),
        Key1FromTerm{impl_->x, impl_->z});
    thrust::copy_n(pooled_policy(), coeff_ptr, impl_->num_terms, values.begin());
    thrust::stable_sort_by_key(pooled_policy(), keys.begin(), keys.end(), values.begin(), Key1Less{});

    PooledArray<HipKey1> reduced_keys(impl_->num_terms);
    PooledArray<thrust::complex<double>> reduced_values(impl_->num_terms);
    auto reduced_end = thrust::reduce_by_key(pooled_policy(), 
        keys.begin(),
        keys.end(),
        values.begin(),
        reduced_keys.begin(),
        reduced_values.begin(),
        Key1Equal{},
        thrust::plus<thrust::complex<double>>{});
    const std::size_t reduced_terms =
        static_cast<std::size_t>(reduced_end.first - reduced_keys.begin());

    PooledArray<HipKey1> survivor_keys(reduced_terms);
    PooledArray<thrust::complex<double>> survivor_values(reduced_terms);
    auto reduced_zip = thrust::make_zip_iterator(
        thrust::make_tuple(reduced_keys.begin(), reduced_values.begin()));
    auto survivor_zip = thrust::make_zip_iterator(
        thrust::make_tuple(survivor_keys.begin(), survivor_values.begin()));
    auto survivor_end = thrust::copy_if(pooled_policy(), 
        reduced_zip,
        reduced_zip + reduced_terms,
        reduced_values.begin(),
        survivor_zip,
        survives);
    const std::size_t survivors = static_cast<std::size_t>(survivor_end - survivor_zip);
    check_overflow();
    if (survivors == 0) {
      return make_empty();
    }

    auto out = std::make_unique<Impl>();
    out->num_qubits = impl_->num_qubits;
    out->words = impl_->words;
    out->num_terms = survivors;
    out->device_ordinal = impl_->device_ordinal;
    pool_allocate(out->x, survivors, "simplified x words");
    pool_allocate(out->z, survivors, "simplified z words");
    pool_allocate(out->coeffs, survivors, "simplified coefficients");
    thrust::transform(pooled_policy(), 
        survivor_keys.begin(),
        survivor_keys.begin() + survivors,
        thrust::device_pointer_cast(out->x),
        Key1ToX{});
    thrust::transform(pooled_policy(), 
        survivor_keys.begin(),
        survivor_keys.begin() + survivors,
        thrust::device_pointer_cast(out->z),
        Key1ToZ{});
    thrust::copy_n(pooled_policy(), survivor_values.begin(), survivors, thrust::device_pointer_cast(out->coeffs));
    check_hip(hipDeviceSynchronize(), "synchronize after HIP simplify");
    return DevicePauliSum(std::move(out));
  }

  if (impl_->words == 2) {
    PooledArray<HipKey2> keys(impl_->num_terms);
    PooledArray<thrust::complex<double>> values(impl_->num_terms);
    auto counting = thrust::make_counting_iterator<std::size_t>(0);
    thrust::transform(pooled_policy(), 
        counting,
        counting + impl_->num_terms,
        keys.begin(),
        Key2FromTerm{impl_->x, impl_->z});
    thrust::copy_n(pooled_policy(), coeff_ptr, impl_->num_terms, values.begin());
    thrust::stable_sort_by_key(pooled_policy(), keys.begin(), keys.end(), values.begin(), Key2Less{});

    PooledArray<HipKey2> reduced_keys(impl_->num_terms);
    PooledArray<thrust::complex<double>> reduced_values(impl_->num_terms);
    auto reduced_end = thrust::reduce_by_key(pooled_policy(), 
        keys.begin(),
        keys.end(),
        values.begin(),
        reduced_keys.begin(),
        reduced_values.begin(),
        Key2Equal{},
        thrust::plus<thrust::complex<double>>{});
    const std::size_t reduced_terms =
        static_cast<std::size_t>(reduced_end.first - reduced_keys.begin());

    PooledArray<HipKey2> survivor_keys(reduced_terms);
    PooledArray<thrust::complex<double>> survivor_values(reduced_terms);
    auto reduced_zip = thrust::make_zip_iterator(
        thrust::make_tuple(reduced_keys.begin(), reduced_values.begin()));
    auto survivor_zip = thrust::make_zip_iterator(
        thrust::make_tuple(survivor_keys.begin(), survivor_values.begin()));
    auto survivor_end = thrust::copy_if(pooled_policy(), 
        reduced_zip,
        reduced_zip + reduced_terms,
        reduced_values.begin(),
        survivor_zip,
        survives);
    const std::size_t survivors = static_cast<std::size_t>(survivor_end - survivor_zip);
    check_overflow();
    if (survivors == 0) {
      return make_empty();
    }

    auto out = std::make_unique<Impl>();
    out->num_qubits = impl_->num_qubits;
    out->words = impl_->words;
    out->num_terms = survivors;
    out->device_ordinal = impl_->device_ordinal;
    const std::size_t packed_words =
        detail::checked_product(survivors, impl_->words, "simplified packed words");
    pool_allocate(out->x, packed_words, "simplified x words");
    pool_allocate(out->z, packed_words, "simplified z words");
    pool_allocate(out->coeffs, survivors, "simplified coefficients");
    auto packed_counting = thrust::make_counting_iterator<std::size_t>(0);
    const HipKey2* survivor_key_ptr = thrust::raw_pointer_cast(survivor_keys.data());
    thrust::transform(pooled_policy(), 
        packed_counting,
        packed_counting + packed_words,
        thrust::device_pointer_cast(out->x),
        Key2ToXWord{survivor_key_ptr});
    thrust::transform(pooled_policy(), 
        packed_counting,
        packed_counting + packed_words,
        thrust::device_pointer_cast(out->z),
        Key2ToZWord{survivor_key_ptr});
    thrust::copy_n(pooled_policy(), survivor_values.begin(), survivors, thrust::device_pointer_cast(out->coeffs));
    check_hip(hipDeviceSynchronize(), "synchronize after HIP simplify");
    return DevicePauliSum(std::move(out));
  }

  const std::size_t packed_words =
      detail::checked_product(impl_->num_terms, impl_->words, "generic simplify packed words");
  PooledArray<std::size_t> sorted_indices(impl_->num_terms);
  thrust::sequence(pooled_policy(), sorted_indices.begin(), sorted_indices.end());
  thrust::sort(pooled_policy(), 
      sorted_indices.begin(),
      sorted_indices.end(),
      GenericTermIndexLess{impl_->x, impl_->z, impl_->words});

  if (generic_reduction_strategy_from_env() == GenericReductionStrategy::kSerialKernel) {
    PooledArray<std::uint64_t> temp_x(packed_words);
    PooledArray<std::uint64_t> temp_z(packed_words);
    PooledArray<thrust::complex<double>> temp_coeffs(impl_->num_terms);
    PooledArray<std::size_t> temp_count(1);
    reduce_sorted_generic_terms_kernel<<<1, 1>>>(
        impl_->x,
        impl_->z,
        impl_->coeffs,
        thrust::raw_pointer_cast(sorted_indices.data()),
        impl_->num_terms,
        impl_->words,
        survives,
        thrust::raw_pointer_cast(temp_x.data()),
        thrust::raw_pointer_cast(temp_z.data()),
        thrust::raw_pointer_cast(temp_coeffs.data()),
        thrust::raw_pointer_cast(temp_count.data()));
    check_hip(hipGetLastError(), "launch generic HIP simplify serial fallback");
    check_hip(hipDeviceSynchronize(), "synchronize after generic HIP simplify serial fallback");

    std::size_t survivors = 0;
    copy_to_host(
        &survivors,
        thrust::raw_pointer_cast(temp_count.data()),
        1,
        "generic simplify count");
    check_overflow();
    if (survivors == 0) {
      return make_empty();
    }

    auto out = std::make_unique<Impl>();
    out->num_qubits = impl_->num_qubits;
    out->words = impl_->words;
    out->num_terms = survivors;
    out->device_ordinal = impl_->device_ordinal;
    const std::size_t survivor_packed_words =
        detail::checked_product(survivors, impl_->words, "generic simplified packed words");
    pool_allocate(out->x, survivor_packed_words, "simplified x words");
    pool_allocate(out->z, survivor_packed_words, "simplified z words");
    pool_allocate(out->coeffs, survivors, "simplified coefficients");
    copy_device_to_device(
        out->x,
        thrust::raw_pointer_cast(temp_x.data()),
        survivor_packed_words,
        "simplified x words");
    copy_device_to_device(
        out->z,
        thrust::raw_pointer_cast(temp_z.data()),
        survivor_packed_words,
        "simplified z words");
    copy_device_to_device(
        out->coeffs,
        thrust::raw_pointer_cast(temp_coeffs.data()),
        survivors,
        "simplified coefficients");
    check_hip(hipDeviceSynchronize(), "synchronize after HIP simplify");
    return DevicePauliSum(std::move(out));
  }

  auto sorted_coeffs = thrust::make_permutation_iterator(coeff_ptr, sorted_indices.begin());
  PooledArray<std::size_t> reduced_indices(impl_->num_terms);
  PooledArray<thrust::complex<double>> reduced_values(impl_->num_terms);
  auto reduced_end = thrust::reduce_by_key(pooled_policy(), 
      sorted_indices.begin(),
      sorted_indices.end(),
      sorted_coeffs,
      reduced_indices.begin(),
      reduced_values.begin(),
      GenericTermIndexEqual{impl_->x, impl_->z, impl_->words},
      thrust::plus<thrust::complex<double>>{});
  const std::size_t reduced_terms =
      static_cast<std::size_t>(reduced_end.first - reduced_indices.begin());

  PooledArray<std::size_t> survivor_indices(reduced_terms);
  PooledArray<thrust::complex<double>> survivor_values(reduced_terms);
  auto reduced_zip = thrust::make_zip_iterator(
      thrust::make_tuple(reduced_indices.begin(), reduced_values.begin()));
  auto survivor_zip = thrust::make_zip_iterator(
      thrust::make_tuple(survivor_indices.begin(), survivor_values.begin()));
  auto survivor_end = thrust::copy_if(pooled_policy(), 
      reduced_zip,
      reduced_zip + reduced_terms,
      reduced_values.begin(),
      survivor_zip,
      survives);
  const std::size_t survivors = static_cast<std::size_t>(survivor_end - survivor_zip);
  check_overflow();
  if (survivors == 0) {
    return make_empty();
  }

  auto out = std::make_unique<Impl>();
  out->num_qubits = impl_->num_qubits;
  out->words = impl_->words;
  out->num_terms = survivors;
  out->device_ordinal = impl_->device_ordinal;
  const std::size_t survivor_packed_words =
      detail::checked_product(survivors, impl_->words, "generic simplified packed words");
  pool_allocate(out->x, survivor_packed_words, "simplified x words");
  pool_allocate(out->z, survivor_packed_words, "simplified z words");
  pool_allocate(out->coeffs, survivors, "simplified coefficients");
  auto survivor_word_indices = thrust::make_counting_iterator<std::size_t>(0);
  const std::size_t* survivor_index_ptr = thrust::raw_pointer_cast(survivor_indices.data());
  thrust::transform(pooled_policy(), 
      survivor_word_indices,
      survivor_word_indices + survivor_packed_words,
      thrust::device_pointer_cast(out->x),
      GenericRepresentativeToXWord{impl_->x, survivor_index_ptr, impl_->words});
  thrust::transform(pooled_policy(), 
      survivor_word_indices,
      survivor_word_indices + survivor_packed_words,
      thrust::device_pointer_cast(out->z),
      GenericRepresentativeToZWord{impl_->z, survivor_index_ptr, impl_->words});
  thrust::copy_n(pooled_policy(), survivor_values.begin(), survivors, thrust::device_pointer_cast(out->coeffs));
  check_hip(hipDeviceSynchronize(), "synchronize after HIP simplify");
  return DevicePauliSum(std::move(out));
}

}  // namespace wolfgang

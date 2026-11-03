#include "device_pauli_sum.cuh"
#include "cuda/workspace.cuh"

#include <cub/cub.cuh>
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
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace wolfgang {

using cuda_detail::check_cuda;
using cuda_detail::copy_device_to_device;
using cuda_detail::copy_to_device;
using cuda_detail::copy_to_host;
using cuda_detail::cuda_allocate;
using cuda_detail::ScopedCudaDevice;
using cuda_detail::validate_simplify_tolerances;

namespace {

enum class DuplicateReductionStrategy {
  kThrustDefault,
  kCubRadixSortReduce,
  kCubRadixSortRunLength,
};

// Operators wider than two words use the generic index-sort path. The
// production reduction is a parallel segmented reduce-by-key; the original
// one-thread kernel is retained only as a benchmark baseline.
enum class GenericReductionStrategy {
  kParallelReduceByKey,
  kSerialKernel,
};

struct CudaKey1 {
  std::uint64_t x;
  std::uint64_t z;
};

struct CudaKey2 {
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
  __host__ __device__ bool operator()(const CudaKey1& lhs, const CudaKey1& rhs) const noexcept {
    if (lhs.x != rhs.x) {
      return lhs.x < rhs.x;
    }
    return lhs.z < rhs.z;
  }
};

struct Key1Equal {
  __host__ __device__ bool operator()(const CudaKey1& lhs, const CudaKey1& rhs) const noexcept {
    return lhs.x == rhs.x && lhs.z == rhs.z;
  }
};

struct Key2Less {
  __host__ __device__ bool operator()(const CudaKey2& lhs, const CudaKey2& rhs) const noexcept {
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
  __host__ __device__ bool operator()(const CudaKey2& lhs, const CudaKey2& rhs) const noexcept {
    return lhs.x0 == rhs.x0 && lhs.z0 == rhs.z0 && lhs.x1 == rhs.x1 && lhs.z1 == rhs.z1;
  }
};

struct Key1FromTerm {
  const std::uint64_t* x;
  const std::uint64_t* z;

  __host__ __device__ CudaKey1 operator()(std::size_t term) const noexcept {
    return {x[term], z[term]};
  }
};

struct Key2FromTerm {
  const std::uint64_t* x;
  const std::uint64_t* z;

  __host__ __device__ CudaKey2 operator()(std::size_t term) const noexcept {
    const std::size_t offset = term * 2;
    return {x[offset], z[offset], x[offset + 1], z[offset + 1]};
  }
};

struct Key1ToX {
  __host__ __device__ std::uint64_t operator()(const CudaKey1& key) const noexcept {
    return key.x;
  }
};

struct Key1ToZ {
  __host__ __device__ std::uint64_t operator()(const CudaKey1& key) const noexcept {
    return key.z;
  }
};

struct Key2ToX0 {
  __host__ __device__ std::uint64_t operator()(const CudaKey2& key) const noexcept {
    return key.x0;
  }
};

struct Key2ToZ0 {
  __host__ __device__ std::uint64_t operator()(const CudaKey2& key) const noexcept {
    return key.z0;
  }
};

struct Key2ToX1 {
  __host__ __device__ std::uint64_t operator()(const CudaKey2& key) const noexcept {
    return key.x1;
  }
};

struct Key2ToZ1 {
  __host__ __device__ std::uint64_t operator()(const CudaKey2& key) const noexcept {
    return key.z1;
  }
};

struct Key2ToXWord {
  const CudaKey2* keys;

  __host__ __device__ std::uint64_t operator()(std::size_t word_index) const noexcept {
    const CudaKey2& key = keys[word_index / 2];
    return (word_index & 1U) == 0 ? key.x0 : key.x1;
  }
};

struct Key2ToZWord {
  const CudaKey2* keys;

  __host__ __device__ std::uint64_t operator()(std::size_t word_index) const noexcept {
    const CudaKey2& key = keys[word_index / 2];
    return (word_index & 1U) == 0 ? key.z0 : key.z1;
  }
};

struct ComplexAbs {
  __host__ __device__ double operator()(thrust::complex<double> value) const noexcept {
    return thrust::abs(value);
  }
};

struct ComplexHalfAbs {
  __host__ __device__ double operator()(thrust::complex<double> value) const noexcept {
    return ::hypot(value.real() * 0.5, value.imag() * 0.5);
  }
};

// Mirrors the CPU drop rule (src/simplify.cpp). Finite complex128 components can
// have a magnitude above DBL_MAX: at rtol == 0 the threshold is atol itself, so
// 0 * inf never produces NaN, and when atol + rtol * max overflows, magnitudes
// and the threshold are compared in half units instead.
struct CoeffSurvives {
  double drop_threshold;
  double half_threshold = 0.0;
  bool use_half_units = false;

  __host__ __device__ bool operator()(thrust::complex<double> value) const noexcept {
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

DuplicateReductionStrategy duplicate_reduction_strategy_from_env() {
  const char* value = std::getenv("WOLFGANG_CUDA_BENCH_DUPLICATE_REDUCTION");
  if (value == nullptr || std::string(value).empty() || std::string(value) == "thrust_default") {
    return DuplicateReductionStrategy::kThrustDefault;
  }
  const std::string setting(value);
  if (setting == "cub_radix_sort_reduce") {
    return DuplicateReductionStrategy::kCubRadixSortReduce;
  }
  if (setting == "cub_radix_sort_run_length") {
    return DuplicateReductionStrategy::kCubRadixSortRunLength;
  }
  throw std::invalid_argument(
      "WOLFGANG_CUDA_BENCH_DUPLICATE_REDUCTION must be thrust_default, "
      "cub_radix_sort_reduce, or cub_radix_sort_run_length");
}

GenericReductionStrategy generic_reduction_strategy_from_env() {
  const char* value = std::getenv("WOLFGANG_CUDA_BENCH_GENERIC_MULTIWORD_REDUCTION");
  if (value == nullptr || std::string(value).empty() ||
      std::string(value) == "reduce_by_key") {
    return GenericReductionStrategy::kParallelReduceByKey;
  }
  if (std::string(value) == "serial_kernel") {
    return GenericReductionStrategy::kSerialKernel;
  }
  throw std::invalid_argument(
      "WOLFGANG_CUDA_BENCH_GENERIC_MULTIWORD_REDUCTION must be reduce_by_key or serial_kernel");
}

int checked_cub_num_items(std::size_t count, const char* operation) {
  if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument(std::string(operation) + " exceeds CUB int item-count limits");
  }
  return static_cast<int>(count);
}

wolfgang::cuda::detail::CudaWorkspace& duplicate_reduction_workspace(int device_ordinal) {
  thread_local std::unique_ptr<wolfgang::cuda::detail::CudaWorkspace> workspace;
  if (!workspace || workspace->device_ordinal() != device_ordinal) {
    workspace = std::make_unique<wolfgang::cuda::detail::CudaWorkspace>(device_ordinal);
  }
  workspace->ensure_device(device_ordinal);
  return *workspace;
}

class CubTemporaryStorage {
public:
  CubTemporaryStorage(
      int device_ordinal,
      std::size_t bytes,
      wolfgang::cuda::detail::WorkspaceTimingMode workspace_mode)
      : device_ordinal_(device_ordinal), workspace_mode_(workspace_mode) {
    if (bytes == 0) {
      return;
    }
    if (workspace_mode_ == wolfgang::cuda::detail::WorkspaceTimingMode::kAbsent) {
      ScopedCudaDevice guard(device_ordinal_);
      check_cuda(cudaMalloc(&ptr_, bytes), "allocate CUB temporary storage");
      owns_allocation_ = true;
      return;
    }

    wolfgang::cuda::detail::CudaWorkspace& workspace =
        duplicate_reduction_workspace(device_ordinal_);
    if (workspace_mode_ == wolfgang::cuda::detail::WorkspaceTimingMode::kGrowInsideTiming) {
      workspace.release();
    }
    ptr_ = workspace.reserve_bytes(bytes, 256);
  }

  CubTemporaryStorage(const CubTemporaryStorage&) = delete;
  CubTemporaryStorage& operator=(const CubTemporaryStorage&) = delete;

  ~CubTemporaryStorage() {
    if (owns_allocation_ && ptr_ != nullptr) {
      ScopedCudaDevice guard(device_ordinal_);
      (void)cudaFree(ptr_);
    }
    if (workspace_mode_ != wolfgang::cuda::detail::WorkspaceTimingMode::kAbsent) {
      duplicate_reduction_workspace(device_ordinal_).reset();
    }
  }

  [[nodiscard]] void* data() const noexcept { return ptr_; }

private:
  void* ptr_ = nullptr;
  int device_ordinal_ = 0;
  wolfgang::cuda::detail::WorkspaceTimingMode workspace_mode_ =
      wolfgang::cuda::detail::WorkspaceTimingMode::kAbsent;
  bool owns_allocation_ = false;
};

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

// Gathers packed words for each surviving term from its representative input
// index, so output masks are written in one parallel pass per array.
struct GenericRepresentativeToWord {
  const std::uint64_t* words_in;
  const std::size_t* representative_indices;
  std::size_t words;

  __host__ __device__ std::uint64_t operator()(std::size_t word_index) const noexcept {
    const std::size_t output_term = word_index / words;
    const std::size_t word = word_index - output_term * words;
    return words_in[representative_indices[output_term] * words + word];
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

  // Validate the benchmark selector up front so a bad value fails on every
  // generic-width input, including the small-input fast paths below.
  const GenericReductionStrategy generic_strategy =
      impl_->words > 2 ? generic_reduction_strategy_from_env()
                       : GenericReductionStrategy::kParallelReduceByKey;
  const DuplicateReductionStrategy duplicate_strategy = duplicate_reduction_strategy_from_env();

  if (impl_->num_terms == 0) {
    return make_empty();
  }

  ScopedCudaDevice guard(impl_->device_ordinal);
  if (impl_->num_terms == 1) {
    // One term has nothing to merge; only the drop rule applies, and the
    // largest input magnitude is the term's own. Skipping sort and reduction
    // removes their fixed setup cost from this common degenerate case.
    thrust::complex<double> coeff;
    copy_to_host(&coeff, impl_->coeffs, 1, "single-term coefficient");
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
      cuda_allocate(out->x, impl_->words, "simplified x words");
      cuda_allocate(out->z, impl_->words, "simplified z words");
      copy_device_to_device(out->x, impl_->x, impl_->words, "simplified x words");
      copy_device_to_device(out->z, impl_->z, impl_->words, "simplified z words");
    }
    cuda_allocate(out->coeffs, 1, "simplified coefficients");
    copy_device_to_device(out->coeffs, impl_->coeffs, 1, "simplified coefficients");
    check_cuda(cudaDeviceSynchronize(), "synchronize after CUDA simplify");
    return DevicePauliSum(std::move(out));
  }

  auto coeff_ptr = thrust::device_pointer_cast(impl_->coeffs);
  const double max_abs_input = thrust::transform_reduce(
      coeff_ptr,
      coeff_ptr + impl_->num_terms,
      ComplexAbs{},
      0.0,
      thrust::maximum<double>{});
  const CoeffSurvives survives = survival_rule(atol, rtol, max_abs_input, [&] {
    return thrust::transform_reduce(
        coeff_ptr,
        coeff_ptr + impl_->num_terms,
        ComplexHalfAbs{},
        0.0,
        thrust::maximum<double>{});
  });

  if (impl_->words == 0) {
    const thrust::complex<double> accumulator =
        thrust::reduce(coeff_ptr, coeff_ptr + impl_->num_terms, thrust::complex<double>{0.0, 0.0});
    if (!survives(accumulator)) {
      return make_empty();
    }
    auto out = std::make_unique<Impl>();
    out->num_qubits = impl_->num_qubits;
    out->words = impl_->words;
    out->num_terms = 1;
    out->device_ordinal = impl_->device_ordinal;
    cuda_allocate(out->coeffs, 1, "simplified coefficients");
    copy_to_device(out->coeffs, &accumulator, 1, "simplified coefficients");
    check_cuda(cudaDeviceSynchronize(), "synchronize after CUDA simplify");
    return DevicePauliSum(std::move(out));
  }

  if (impl_->words == 1 && impl_->num_qubits <= 32) {
    if (duplicate_strategy == DuplicateReductionStrategy::kCubRadixSortReduce) {
      thrust::device_vector<std::uint64_t> keys(impl_->num_terms);
      thrust::device_vector<thrust::complex<double>> values(impl_->num_terms);
      thrust::device_vector<std::uint64_t> sorted_keys(impl_->num_terms);
      thrust::device_vector<thrust::complex<double>> sorted_values(impl_->num_terms);
      auto counting = thrust::make_counting_iterator<std::size_t>(0);
      thrust::transform(
          counting,
          counting + impl_->num_terms,
          keys.begin(),
          PackedKey32FromTerm{impl_->x, impl_->z});
      thrust::copy_n(coeff_ptr, impl_->num_terms, values.begin());

      const int num_items = checked_cub_num_items(impl_->num_terms, "CUDA CUB simplify");
      std::size_t sort_temp_bytes = 0;
      check_cuda(
          cub::DeviceRadixSort::SortPairs(
              nullptr,
              sort_temp_bytes,
              thrust::raw_pointer_cast(keys.data()),
              thrust::raw_pointer_cast(sorted_keys.data()),
              thrust::raw_pointer_cast(values.data()),
              thrust::raw_pointer_cast(sorted_values.data()),
              num_items),
          "query CUB radix sort temporary storage");
      CubTemporaryStorage sort_temp(
          impl_->device_ordinal,
          sort_temp_bytes,
          wolfgang::cuda::detail::workspace_timing_mode_from_env());
      check_cuda(
          cub::DeviceRadixSort::SortPairs(
              sort_temp.data(),
              sort_temp_bytes,
              thrust::raw_pointer_cast(keys.data()),
              thrust::raw_pointer_cast(sorted_keys.data()),
              thrust::raw_pointer_cast(values.data()),
              thrust::raw_pointer_cast(sorted_values.data()),
              num_items),
          "run CUB radix sort for simplify");

      // The Campaign 4 CUB prototypes deliberately keep the reduction and
      // tolerance compaction identical to the production Thrust path after the
      // radix-sort boundary. This isolates whether CCCL radix sort plus explicit
      // scratch ownership beats the existing comparison-sort boundary without
      // changing canonical ordering or floating-point summation semantics.
      thrust::device_vector<std::uint64_t> reduced_keys(impl_->num_terms);
      thrust::device_vector<thrust::complex<double>> reduced_values(impl_->num_terms);
      auto reduced_end = thrust::reduce_by_key(
          sorted_keys.begin(),
          sorted_keys.end(),
          sorted_values.begin(),
          reduced_keys.begin(),
          reduced_values.begin(),
          thrust::equal_to<std::uint64_t>{},
          thrust::plus<thrust::complex<double>>{});
      const std::size_t reduced_terms =
          static_cast<std::size_t>(reduced_end.first - reduced_keys.begin());

      thrust::device_vector<std::uint64_t> survivor_keys(reduced_terms);
      thrust::device_vector<thrust::complex<double>> survivor_values(reduced_terms);
      auto reduced_zip = thrust::make_zip_iterator(
          thrust::make_tuple(reduced_keys.begin(), reduced_values.begin()));
      auto survivor_zip = thrust::make_zip_iterator(
          thrust::make_tuple(survivor_keys.begin(), survivor_values.begin()));
      auto survivor_end = thrust::copy_if(
          reduced_zip,
          reduced_zip + reduced_terms,
          reduced_values.begin(),
          survivor_zip,
          survives);
      const std::size_t survivors = static_cast<std::size_t>(survivor_end - survivor_zip);
      if (survivors == 0) {
        return make_empty();
      }

      auto out = std::make_unique<Impl>();
      out->num_qubits = impl_->num_qubits;
      out->words = impl_->words;
      out->num_terms = survivors;
      out->device_ordinal = impl_->device_ordinal;
      cuda_allocate(out->x, survivors, "simplified x words");
      cuda_allocate(out->z, survivors, "simplified z words");
      cuda_allocate(out->coeffs, survivors, "simplified coefficients");
      thrust::transform(
          survivor_keys.begin(),
          survivor_keys.begin() + survivors,
          thrust::device_pointer_cast(out->x),
          PackedKey32ToX{});
      thrust::transform(
          survivor_keys.begin(),
          survivor_keys.begin() + survivors,
          thrust::device_pointer_cast(out->z),
          PackedKey32ToZ{});
      thrust::copy_n(survivor_values.begin(), survivors, thrust::device_pointer_cast(out->coeffs));
      check_cuda(cudaDeviceSynchronize(), "synchronize after CUB CUDA simplify");
      return DevicePauliSum(std::move(out));
    }

    // Most benchmark and near-term chemistry workloads in this campaign use
    // <=32 qubits.  In that regime x and z each occupy only the low 32 bits, so
    // packing them into one 64-bit key preserves the canonical x-then-z sort
    // order while cutting sort/reduce key traffic in half versus CudaKey1.
    thrust::device_vector<std::uint64_t> keys(impl_->num_terms);
    thrust::device_vector<thrust::complex<double>> values(impl_->num_terms);
    auto counting = thrust::make_counting_iterator<std::size_t>(0);
    thrust::transform(
        counting,
        counting + impl_->num_terms,
        keys.begin(),
        PackedKey32FromTerm{impl_->x, impl_->z});
    thrust::copy_n(coeff_ptr, impl_->num_terms, values.begin());
    thrust::stable_sort_by_key(keys.begin(), keys.end(), values.begin());

    thrust::device_vector<std::uint64_t> reduced_keys(impl_->num_terms);
    thrust::device_vector<thrust::complex<double>> reduced_values(impl_->num_terms);
    auto reduced_end = thrust::reduce_by_key(
        keys.begin(),
        keys.end(),
        values.begin(),
        reduced_keys.begin(),
        reduced_values.begin(),
        thrust::equal_to<std::uint64_t>{},
        thrust::plus<thrust::complex<double>>{});
    const std::size_t reduced_terms =
        static_cast<std::size_t>(reduced_end.first - reduced_keys.begin());

    thrust::device_vector<std::uint64_t> survivor_keys(reduced_terms);
    thrust::device_vector<thrust::complex<double>> survivor_values(reduced_terms);
    auto reduced_zip = thrust::make_zip_iterator(
        thrust::make_tuple(reduced_keys.begin(), reduced_values.begin()));
    auto survivor_zip = thrust::make_zip_iterator(
        thrust::make_tuple(survivor_keys.begin(), survivor_values.begin()));
    auto survivor_end = thrust::copy_if(
        reduced_zip,
        reduced_zip + reduced_terms,
        reduced_values.begin(),
        survivor_zip,
        survives);
    const std::size_t survivors = static_cast<std::size_t>(survivor_end - survivor_zip);
    if (survivors == 0) {
      return make_empty();
    }

    auto out = std::make_unique<Impl>();
    out->num_qubits = impl_->num_qubits;
    out->words = impl_->words;
    out->num_terms = survivors;
    out->device_ordinal = impl_->device_ordinal;
    cuda_allocate(out->x, survivors, "simplified x words");
    cuda_allocate(out->z, survivors, "simplified z words");
    cuda_allocate(out->coeffs, survivors, "simplified coefficients");
    thrust::transform(
        survivor_keys.begin(),
        survivor_keys.begin() + survivors,
        thrust::device_pointer_cast(out->x),
        PackedKey32ToX{});
    thrust::transform(
        survivor_keys.begin(),
        survivor_keys.begin() + survivors,
        thrust::device_pointer_cast(out->z),
        PackedKey32ToZ{});
    thrust::copy_n(survivor_values.begin(), survivors, thrust::device_pointer_cast(out->coeffs));
    check_cuda(cudaDeviceSynchronize(), "synchronize after CUDA simplify");
    return DevicePauliSum(std::move(out));
  }

  if (impl_->words == 1) {
    thrust::device_vector<CudaKey1> keys(impl_->num_terms);
    thrust::device_vector<thrust::complex<double>> values(impl_->num_terms);
    auto counting = thrust::make_counting_iterator<std::size_t>(0);
    thrust::transform(
        counting,
        counting + impl_->num_terms,
        keys.begin(),
        Key1FromTerm{impl_->x, impl_->z});
    thrust::copy_n(coeff_ptr, impl_->num_terms, values.begin());
    thrust::stable_sort_by_key(keys.begin(), keys.end(), values.begin(), Key1Less{});

    thrust::device_vector<CudaKey1> reduced_keys(impl_->num_terms);
    thrust::device_vector<thrust::complex<double>> reduced_values(impl_->num_terms);
    auto reduced_end = thrust::reduce_by_key(
        keys.begin(),
        keys.end(),
        values.begin(),
        reduced_keys.begin(),
        reduced_values.begin(),
        Key1Equal{},
        thrust::plus<thrust::complex<double>>{});
    const std::size_t reduced_terms =
        static_cast<std::size_t>(reduced_end.first - reduced_keys.begin());

    thrust::device_vector<CudaKey1> survivor_keys(reduced_terms);
    thrust::device_vector<thrust::complex<double>> survivor_values(reduced_terms);
    auto reduced_zip = thrust::make_zip_iterator(
        thrust::make_tuple(reduced_keys.begin(), reduced_values.begin()));
    auto survivor_zip = thrust::make_zip_iterator(
        thrust::make_tuple(survivor_keys.begin(), survivor_values.begin()));
    auto survivor_end = thrust::copy_if(
        reduced_zip,
        reduced_zip + reduced_terms,
        reduced_values.begin(),
        survivor_zip,
        survives);
    const std::size_t survivors = static_cast<std::size_t>(survivor_end - survivor_zip);
    if (survivors == 0) {
      return make_empty();
    }

    auto out = std::make_unique<Impl>();
    out->num_qubits = impl_->num_qubits;
    out->words = impl_->words;
    out->num_terms = survivors;
    out->device_ordinal = impl_->device_ordinal;
    cuda_allocate(out->x, survivors, "simplified x words");
    cuda_allocate(out->z, survivors, "simplified z words");
    cuda_allocate(out->coeffs, survivors, "simplified coefficients");
    thrust::transform(
        survivor_keys.begin(),
        survivor_keys.begin() + survivors,
        thrust::device_pointer_cast(out->x),
        Key1ToX{});
    thrust::transform(
        survivor_keys.begin(),
        survivor_keys.begin() + survivors,
        thrust::device_pointer_cast(out->z),
        Key1ToZ{});
    thrust::copy_n(survivor_values.begin(), survivors, thrust::device_pointer_cast(out->coeffs));
    check_cuda(cudaDeviceSynchronize(), "synchronize after CUDA simplify");
    return DevicePauliSum(std::move(out));
  }

  if (impl_->words == 2) {
    thrust::device_vector<CudaKey2> keys(impl_->num_terms);
    thrust::device_vector<thrust::complex<double>> values(impl_->num_terms);
    auto counting = thrust::make_counting_iterator<std::size_t>(0);
    thrust::transform(
        counting,
        counting + impl_->num_terms,
        keys.begin(),
        Key2FromTerm{impl_->x, impl_->z});
    thrust::copy_n(coeff_ptr, impl_->num_terms, values.begin());
    thrust::stable_sort_by_key(keys.begin(), keys.end(), values.begin(), Key2Less{});

    thrust::device_vector<CudaKey2> reduced_keys(impl_->num_terms);
    thrust::device_vector<thrust::complex<double>> reduced_values(impl_->num_terms);
    auto reduced_end = thrust::reduce_by_key(
        keys.begin(),
        keys.end(),
        values.begin(),
        reduced_keys.begin(),
        reduced_values.begin(),
        Key2Equal{},
        thrust::plus<thrust::complex<double>>{});
    const std::size_t reduced_terms =
        static_cast<std::size_t>(reduced_end.first - reduced_keys.begin());

    thrust::device_vector<CudaKey2> survivor_keys(reduced_terms);
    thrust::device_vector<thrust::complex<double>> survivor_values(reduced_terms);
    auto reduced_zip = thrust::make_zip_iterator(
        thrust::make_tuple(reduced_keys.begin(), reduced_values.begin()));
    auto survivor_zip = thrust::make_zip_iterator(
        thrust::make_tuple(survivor_keys.begin(), survivor_values.begin()));
    auto survivor_end = thrust::copy_if(
        reduced_zip,
        reduced_zip + reduced_terms,
        reduced_values.begin(),
        survivor_zip,
        survives);
    const std::size_t survivors = static_cast<std::size_t>(survivor_end - survivor_zip);
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
    cuda_allocate(out->x, packed_words, "simplified x words");
    cuda_allocate(out->z, packed_words, "simplified z words");
    cuda_allocate(out->coeffs, survivors, "simplified coefficients");
    auto packed_counting = thrust::make_counting_iterator<std::size_t>(0);
    const CudaKey2* survivor_key_ptr = thrust::raw_pointer_cast(survivor_keys.data());
    thrust::transform(
        packed_counting,
        packed_counting + packed_words,
        thrust::device_pointer_cast(out->x),
        Key2ToXWord{survivor_key_ptr});
    thrust::transform(
        packed_counting,
        packed_counting + packed_words,
        thrust::device_pointer_cast(out->z),
        Key2ToZWord{survivor_key_ptr});
    thrust::copy_n(survivor_values.begin(), survivors, thrust::device_pointer_cast(out->coeffs));
    check_cuda(cudaDeviceSynchronize(), "synchronize after CUDA simplify");
    return DevicePauliSum(std::move(out));
  }

  const std::size_t packed_words =
      detail::checked_product(impl_->num_terms, impl_->words, "generic simplify packed words");
  // The comparator breaks ties on input index, so each duplicate run lists its
  // operands in input order. reduce_by_key may still associate partial sums in
  // any order (cuda_backend.md allows implementation-defined reduction order).
  thrust::device_vector<std::size_t> sorted_indices(impl_->num_terms);
  thrust::sequence(sorted_indices.begin(), sorted_indices.end());
  thrust::sort(
      sorted_indices.begin(),
      sorted_indices.end(),
      GenericTermIndexLess{impl_->x, impl_->z, impl_->words});

  if (generic_strategy == GenericReductionStrategy::kParallelReduceByKey) {
    auto sorted_coeffs = thrust::make_permutation_iterator(coeff_ptr, sorted_indices.begin());
    thrust::device_vector<std::size_t> reduced_indices(impl_->num_terms);
    thrust::device_vector<thrust::complex<double>> reduced_values(impl_->num_terms);
    auto reduced_end = thrust::reduce_by_key(
        sorted_indices.begin(),
        sorted_indices.end(),
        sorted_coeffs,
        reduced_indices.begin(),
        reduced_values.begin(),
        GenericTermIndexEqual{impl_->x, impl_->z, impl_->words},
        thrust::plus<thrust::complex<double>>{});
    const std::size_t reduced_terms =
        static_cast<std::size_t>(reduced_end.first - reduced_indices.begin());

    thrust::device_vector<std::size_t> survivor_indices(reduced_terms);
    thrust::device_vector<thrust::complex<double>> survivor_values(reduced_terms);
    auto reduced_zip = thrust::make_zip_iterator(
        thrust::make_tuple(reduced_indices.begin(), reduced_values.begin()));
    auto survivor_zip = thrust::make_zip_iterator(
        thrust::make_tuple(survivor_indices.begin(), survivor_values.begin()));
    auto survivor_end = thrust::copy_if(
        reduced_zip,
        reduced_zip + reduced_terms,
        reduced_values.begin(),
        survivor_zip,
        survives);
    const std::size_t survivors = static_cast<std::size_t>(survivor_end - survivor_zip);
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
    cuda_allocate(out->x, survivor_packed_words, "simplified x words");
    cuda_allocate(out->z, survivor_packed_words, "simplified z words");
    cuda_allocate(out->coeffs, survivors, "simplified coefficients");
    auto survivor_word_indices = thrust::make_counting_iterator<std::size_t>(0);
    const std::size_t* survivor_index_ptr = thrust::raw_pointer_cast(survivor_indices.data());
    thrust::transform(
        survivor_word_indices,
        survivor_word_indices + survivor_packed_words,
        thrust::device_pointer_cast(out->x),
        GenericRepresentativeToWord{impl_->x, survivor_index_ptr, impl_->words});
    thrust::transform(
        survivor_word_indices,
        survivor_word_indices + survivor_packed_words,
        thrust::device_pointer_cast(out->z),
        GenericRepresentativeToWord{impl_->z, survivor_index_ptr, impl_->words});
    thrust::copy_n(survivor_values.begin(), survivors, thrust::device_pointer_cast(out->coeffs));
    check_cuda(cudaDeviceSynchronize(), "synchronize after generic CUDA simplify");
    return DevicePauliSum(std::move(out));
  }

  thrust::device_vector<std::uint64_t> temp_x(packed_words);
  thrust::device_vector<std::uint64_t> temp_z(packed_words);
  thrust::device_vector<thrust::complex<double>> temp_coeffs(impl_->num_terms);
  thrust::device_vector<std::size_t> temp_count(1);
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
  check_cuda(cudaGetLastError(), "launch generic CUDA simplify");
  check_cuda(cudaDeviceSynchronize(), "synchronize after generic CUDA simplify");

  std::size_t survivors = 0;
  copy_to_host(&survivors, thrust::raw_pointer_cast(temp_count.data()), 1, "generic simplify count");
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
  cuda_allocate(out->x, survivor_packed_words, "simplified x words");
  cuda_allocate(out->z, survivor_packed_words, "simplified z words");
  cuda_allocate(out->coeffs, survivors, "simplified coefficients");
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
  check_cuda(cudaDeviceSynchronize(), "synchronize after CUDA simplify");
  return DevicePauliSum(std::move(out));
}

}  // namespace wolfgang

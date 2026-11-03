#include "device_pauli_sum.cuh"
#include "cuda/pool_allocator.cuh"
#include "cuda/workspace.cuh"
#include "detail/device_simplify_pipeline.cuh"
#include "detail/simplify_numeric.hpp"
#include "detail/small_simplify_kernel.cuh"

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
using cuda_detail::pool_allocate;
using cuda_detail::pooled_policy;
using cuda_detail::PooledArray;
using cuda_detail::copy_device_to_device;
using cuda_detail::copy_to_device;
using cuda_detail::copy_to_host;
using cuda_detail::ScopedCudaDevice;
using cuda_detail::validate_simplify_tolerances;

namespace {

enum class DuplicateReductionStrategy {
  kThrustDefault,
  kCubRadixSortReduce,
  kCubRadixSortRunLength,
};

// Operators wider than two words sort term indices. The production reduction
// is the shared pipeline's parallel reduce-by-key; the original one-thread
// kernel is retained only as a benchmark baseline.
enum class GenericReductionStrategy {
  kParallelReduceByKey,
  kSerialKernel,
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

// Mirrors the CPU drop rule (src/simplify.cpp). Finite complex128 components can
// have a magnitude above DBL_MAX: at rtol == 0 the threshold is atol itself, so
// 0 * inf never produces NaN, and when atol + rtol * max overflows, magnitudes
// and the threshold are compared in half units instead.
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

void require_finite_input(double max_abs_or_nan) {
  if (max_abs_or_nan != max_abs_or_nan) {
    throw std::invalid_argument("simplify requires finite coefficients");
  }
}

// A per-thread value in pinned, device-mapped host memory: kernels write it
// and the host reads it after a synchronization, with no copy. It holds the
// fused small-simplify result, the pipeline result and the benchmark
// baselines' overflow flag. One allocation per type per thread, freed when the
// thread exits, so threads that come and go do not each keep pinned memory.
template <typename T>
struct MappedHostValue {
  volatile T* host = nullptr;
  T* device = nullptr;
};

template <typename T>
class ThreadMappedHostValue {
 public:
  explicit ThreadMappedHostValue(const char* name) {
    T* host = nullptr;
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&host), sizeof(T), cudaHostAllocMapped | cudaHostAllocPortable), name);
    T* device = nullptr;
    const cudaError_t status = cudaHostGetDevicePointer(reinterpret_cast<void**>(&device), host, 0);
    if (status != cudaSuccess) {
      (void)cudaFreeHost(host);
      check_cuda(status, name);
    }
    value_ = {host, device};
  }
  ThreadMappedHostValue(const ThreadMappedHostValue&) = delete;
  ThreadMappedHostValue& operator=(const ThreadMappedHostValue&) = delete;
  // The main thread's thread_local destructors run before the CUDA runtime's
  // exit handlers; any error from a runtime already unloading is ignored.
  ~ThreadMappedHostValue() { (void)cudaFreeHost(const_cast<T*>(value_.host)); }

  [[nodiscard]] const MappedHostValue<T>& get() const noexcept { return value_; }

 private:
  MappedHostValue<T> value_;
};

template <typename T>
MappedHostValue<T> mapped_host_value(const char* name) {
  thread_local const ThreadMappedHostValue<T> value(name);
  return value.get();
}

// Hooks for the shared multi-launch pipeline (detail/device_simplify_pipeline.cuh):
// pooled scratch, the legacy default stream, and a per-thread mapped result.
struct CudaPipelineBackend {
  static constexpr int kThreadsPerBlock = cuda_detail::kCudaThreadsPerBlock;

  static void check(cudaError_t status, const char* what) { check_cuda(status, what); }
  static void check_launch(const char* what) { check_cuda(cudaGetLastError(), what); }
  static void synchronize_stream(const char* what) { check_cuda(cudaStreamSynchronize(0), what); }
  static void synchronize_device(const char* what) { check_cuda(cudaDeviceSynchronize(), what); }
  static int launch_blocks(std::size_t threads, const char* what) {
    return cuda_detail::checked_launch_blocks(threads, what);
  }
  static auto thrust_policy() { return pooled_policy(); }

  class Scratch {
   public:
    explicit Scratch(std::size_t bytes) : buffer_(bytes, "simplify scratch") {}
    [[nodiscard]] std::byte* data() const noexcept { return buffer_.get(); }

   private:
    cuda_detail::PoolBuffer<std::byte> buffer_;
  };

  static wolfgang::accelerator_detail::PipelineResult* result_device() {
    return mapped_host_value<wolfgang::accelerator_detail::PipelineResult>("map simplify result").device;
  }
  static const volatile wolfgang::accelerator_detail::PipelineResult* result_host() {
    return mapped_host_value<wolfgang::accelerator_detail::PipelineResult>("map simplify result").host;
  }
};

[[noreturn]] void throw_accumulation_overflow() {
  throw std::overflow_error("simplify coefficient accumulation overflowed complex128");
}

template <typename HalfMagnitudeMax>
CoeffSurvives survival_rule(
    double atol,
    double rtol,
    double max_abs_input,
    HalfMagnitudeMax&& max_half_magnitude) {
  CoeffSurvives rule{
      rtol == 0.0 ? atol : detail::separately_rounded_threshold(atol, rtol, max_abs_input)};
  if (!std::isfinite(rule.drop_threshold)) {
    rule.half_threshold = detail::separately_rounded_threshold(atol * 0.5, rtol, max_half_magnitude());
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

// Benchmark-only: WOLFGANG_CUDA_BENCH_SMALL_SIMPLIFY=off keeps small operators on
// the multi-launch path for paired comparisons.
bool small_simplify_enabled_from_env() {
  const char* value = std::getenv("WOLFGANG_CUDA_BENCH_SMALL_SIMPLIFY");
  if (value == nullptr || std::string(value).empty() || std::string(value) == "fused") {
    return true;
  }
  if (std::string(value) == "off") {
    return false;
  }
  throw std::invalid_argument("WOLFGANG_CUDA_BENCH_SMALL_SIMPLIFY must be fused or off");
}

// Launches the fused kernel and returns false, leaving no error behind, when
// the device cannot run it as configured; the caller then uses the
// multi-launch path.
template <int Words>
bool launch_small_simplify(
    const std::uint64_t* in_x,
    const std::uint64_t* in_z,
    const thrust::complex<double>* in_coeffs,
    std::size_t terms,
    double atol,
    double rtol,
    std::uint64_t* out_x,
    std::uint64_t* out_z,
    thrust::complex<double>* out_coeffs,
    wolfgang::accelerator_detail::SmallSimplifyResult* device_result) {
  using wolfgang::accelerator_detail::small_simplify_kernel;
  const int threads = wolfgang::accelerator_detail::small_simplify_block_threads(terms);
  const std::size_t launch_bytes =
      static_cast<std::size_t>(threads) * wolfgang::accelerator_detail::small_simplify_bytes_per_position(Words);
  // More than 48 KiB of dynamic shared memory needs an opt-in, which the
  // runtime applies per device; it is cheap, so it is set before each launch
  // that needs it instead of being cached.
  if (launch_bytes > std::size_t{48} * 1024) {
    const cudaError_t attribute_result = cudaFuncSetAttribute(
        small_simplify_kernel<Words>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(launch_bytes));
    if (attribute_result != cudaSuccess) {
      (void)cudaGetLastError();
      return false;
    }
  }
  small_simplify_kernel<Words><<<1, threads, launch_bytes>>>(
      in_x,
      in_z,
      in_coeffs,
      static_cast<std::uint32_t>(terms),
      atol,
      rtol,
      out_x,
      out_z,
      out_coeffs,
      device_result);
  const cudaError_t launch_result = cudaGetLastError();
  if (launch_result == cudaErrorInvalidConfiguration || launch_result == cudaErrorInvalidValue ||
      launch_result == cudaErrorLaunchOutOfResources) {
    return false;  // configuration errors are not sticky; use the multi-launch path
  }
  check_cuda(launch_result, "launch small simplify");
  return true;
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
      const ScopedCudaDevice guard(device_ordinal_, std::nothrow);
      if (cudaFree(ptr_) != cudaSuccess) {
        (void)cudaGetLastError();
      }
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
  const bool small_simplify_enabled = small_simplify_enabled_from_env();

  if (impl_->num_terms == 0) {
    return make_empty();
  }

  ScopedCudaDevice guard(impl_->device_ordinal);
  if (small_simplify_enabled &&
      wolfgang::accelerator_detail::small_simplify_applies(impl_->words, impl_->num_terms)) {
    // One fused launch (src/detail/small_simplify_kernel.cuh). Outputs are
    // sized for every input term; num_terms records the survivors.
    namespace acc = wolfgang::accelerator_detail;
    auto out = std::make_unique<Impl>();
    out->num_qubits = impl_->num_qubits;
    out->words = impl_->words;
    out->device_ordinal = impl_->device_ordinal;
    const std::size_t packed_words =
        detail::checked_product(impl_->num_terms, impl_->words, "small simplify packed words");
    pool_allocate(out->x, packed_words, "simplified x words");
    pool_allocate(out->z, packed_words, "simplified z words");
    pool_allocate(out->coeffs, impl_->num_terms, "simplified coefficients");
    const MappedHostValue<acc::SmallSimplifyResult> result_value =
        mapped_host_value<acc::SmallSimplifyResult>("map small simplify result");
    acc::SmallSimplifyResult* result_ptr = result_value.device;
    bool launched = false;
    switch (impl_->words) {
      case 1:
        launched = launch_small_simplify<1>(
            impl_->x, impl_->z, impl_->coeffs, impl_->num_terms, atol, rtol, out->x, out->z, out->coeffs, result_ptr);
        break;
      case 2:
        launched = launch_small_simplify<2>(
            impl_->x, impl_->z, impl_->coeffs, impl_->num_terms, atol, rtol, out->x, out->z, out->coeffs, result_ptr);
        break;
      case 3:
        launched = launch_small_simplify<3>(
            impl_->x, impl_->z, impl_->coeffs, impl_->num_terms, atol, rtol, out->x, out->z, out->coeffs, result_ptr);
        break;
      default:
        launched = launch_small_simplify<4>(
            impl_->x, impl_->z, impl_->coeffs, impl_->num_terms, atol, rtol, out->x, out->z, out->coeffs, result_ptr);
        break;
    }
    if (launched) {
      check_cuda(cudaStreamSynchronize(0), "synchronize small simplify");
      const unsigned long long status = result_value.host->status;
      const unsigned long long survivors = result_value.host->survivors;
      if (status == static_cast<unsigned long long>(acc::kSmallSimplifyNonFiniteInput)) {
        throw std::invalid_argument("simplify requires finite coefficients");
      }
      if (status == static_cast<unsigned long long>(acc::kSmallSimplifyOverflow)) {
        throw_accumulation_overflow();
      }
      if (survivors == 0) {
        return make_empty();
      }
      out->num_terms = static_cast<std::size_t>(survivors);
      return DevicePauliSum(std::move(out));
    }
    // The device could not run the fused kernel; continue on the multi-launch
    // path (the unused outputs are freed with `out`).
  }

  if (impl_->num_terms == 1) {
    // Operators wider than the fused path: one term has nothing to merge, so
    // only the drop rule applies, with the term's own magnitude as the input
    // maximum. Skipping sort and reduction removes their fixed setup cost.
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
    check_cuda(cudaDeviceSynchronize(), "synchronize after CUDA simplify");
    return DevicePauliSum(std::move(out));
  }

  // The benchmark-only baselines below keep host-side counts for paired
  // comparisons; every other operator with packed words takes the shared
  // pipeline, which synchronizes once before sizing its output.
  const bool cub_baseline = impl_->words == 1 && impl_->num_qubits <= 32 &&
      duplicate_strategy == DuplicateReductionStrategy::kCubRadixSortReduce;
  const bool serial_baseline =
      impl_->words > 2 && generic_strategy == GenericReductionStrategy::kSerialKernel;
  if (impl_->words > 0 && !cub_baseline && !serial_baseline) {
    namespace acc = wolfgang::accelerator_detail;
    auto out = std::make_unique<Impl>();
    out->num_qubits = impl_->num_qubits;
    out->words = impl_->words;
    out->device_ordinal = impl_->device_ordinal;
    const std::size_t survivors = acc::device_simplify_pipeline<CudaPipelineBackend>(
        {impl_->x, impl_->z, impl_->coeffs, impl_->num_terms, impl_->words, impl_->num_qubits},
        atol,
        rtol,
        [&](std::size_t count) {
          const std::size_t packed_words =
              detail::checked_product(count, impl_->words, "simplified packed words");
          pool_allocate(out->x, packed_words, "simplified x words");
          pool_allocate(out->z, packed_words, "simplified z words");
          pool_allocate(out->coeffs, count, "simplified coefficients");
          return acc::PipelineOutput{out->x, out->z, out->coeffs};
        });
    if (survivors == 0) {
      return make_empty();
    }
    out->num_terms = survivors;
    return DevicePauliSum(std::move(out));
  }

  auto coeff_ptr = thrust::device_pointer_cast(impl_->coeffs);
  const double max_abs_input = thrust::transform_reduce(pooled_policy(),
      coeff_ptr,
      coeff_ptr + impl_->num_terms,
      ComplexAbsOrNaN{},
      0.0,
      MaxPropagatingNaN{});
  require_finite_input(max_abs_input);
  CoeffSurvives survives = survival_rule(atol, rtol, max_abs_input, [&] {
    return thrust::transform_reduce(pooled_policy(),
        coeff_ptr,
        coeff_ptr + impl_->num_terms,
        ComplexHalfAbs{},
        0.0,
        thrust::maximum<double>{});
  });

  if (impl_->words == 0) {
    const thrust::complex<double> accumulator =
        thrust::reduce(
            pooled_policy(),
            coeff_ptr,
            coeff_ptr + impl_->num_terms,
            thrust::complex<double>{0.0, 0.0});
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
    check_cuda(cudaDeviceSynchronize(), "synchronize after CUDA simplify");
    return DevicePauliSum(std::move(out));
  }

  const MappedHostValue<int> overflow_flag = mapped_host_value<int>("map simplify overflow flag");
  *overflow_flag.host = 0;
  survives.overflowed = overflow_flag.device;
  // Called after the compaction (or serial kernel) has synchronized, so every
  // predicate write is visible to the host.
  const auto check_overflow = [&overflow_flag] {
    if (*overflow_flag.host != 0) {
      throw_accumulation_overflow();
    }
  };

  if (cub_baseline) {
    PooledArray<std::uint64_t> keys(impl_->num_terms);
    PooledArray<thrust::complex<double>> values(impl_->num_terms);
    PooledArray<std::uint64_t> sorted_keys(impl_->num_terms);
    PooledArray<thrust::complex<double>> sorted_values(impl_->num_terms);
    auto counting = thrust::make_counting_iterator<std::size_t>(0);
    thrust::transform(pooled_policy(),
        counting,
        counting + impl_->num_terms,
        keys.begin(),
        PackedKey32FromTerm{impl_->x, impl_->z});
    thrust::copy_n(pooled_policy(), coeff_ptr, impl_->num_terms, values.begin());

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
    PooledArray<std::uint64_t> reduced_keys(impl_->num_terms);
    PooledArray<thrust::complex<double>> reduced_values(impl_->num_terms);
    auto reduced_end = thrust::reduce_by_key(pooled_policy(),
        sorted_keys.begin(),
        sorted_keys.end(),
        sorted_values.begin(),
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
    thrust::copy_n(
        pooled_policy(),
        survivor_values.begin(),
        survivors,
        thrust::device_pointer_cast(out->coeffs));
    check_cuda(cudaDeviceSynchronize(), "synchronize after CUB CUDA simplify");
    return DevicePauliSum(std::move(out));
  }

  // Benchmark-only serial baseline for operators wider than two words.
  const std::size_t packed_words =
      detail::checked_product(impl_->num_terms, impl_->words, "generic simplify packed words");
  // The comparator breaks ties on input index, so the one-thread kernel sums
  // each duplicate run in input order.
  PooledArray<std::size_t> sorted_indices(impl_->num_terms);
  thrust::sequence(pooled_policy(), sorted_indices.begin(), sorted_indices.end());
  thrust::sort(pooled_policy(),
      sorted_indices.begin(),
      sorted_indices.end(),
      GenericTermIndexLess{impl_->x, impl_->z, impl_->words});

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
  check_cuda(cudaGetLastError(), "launch generic CUDA simplify");
  check_cuda(cudaDeviceSynchronize(), "synchronize after generic CUDA simplify");

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
  check_cuda(cudaDeviceSynchronize(), "synchronize after CUDA simplify");
  return DevicePauliSum(std::move(out));
}

}  // namespace wolfgang

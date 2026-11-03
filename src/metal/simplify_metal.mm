#include "simplify_metal.hpp"

#include "device_pauli_sum_metal.hpp"

#include "detail/simplify_numeric.hpp"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace wolfgang::metal_detail {
namespace {

struct SimplifyWords1Params {
  std::uint32_t terms = 0;
  std::uint32_t padded_terms = 0;
  std::uint64_t drop_threshold_fixed32 = 0;
  std::uint64_t drop_threshold_square_fixed64 = 0;
  std::uint32_t use_magnitude_square_threshold = 0;
};

struct BitonicSortParams {
  std::uint32_t padded_terms = 0;
  std::uint32_t j = 0;
  std::uint32_t k = 0;
};

struct PrefixSumParams {
  std::uint32_t entries = 0;
  std::uint32_t offset = 0;
};

struct Fixed32CoefficientScan {
  std::uint64_t max_abs_component = 0;
};

enum class SimplifyKernel {
  InitKeys,
  BitonicSortStep,
  MarkHeads,
  ClearUint,
  PrefixSumStep,
  ReduceByKey,
  CompactSurvivors,
  HybridTileSort,
  HybridTileMerge,
  HybridGlobalStep,
  HybridGatherCoeffs,
};

// CMake defines the tile size once for the host code, the precompiled library
// and the run-time compile options, so the kernels cannot disagree with it.
#ifndef WOLFGANG_HYBRID_SORT_TILE
#error "WOLFGANG_HYBRID_SORT_TILE must be defined by the build"
#endif
constexpr std::uint32_t kHybridSortTile = WOLFGANG_HYBRID_SORT_TILE;
static_assert((kHybridSortTile & (kHybridSortTile - 1U)) == 0, "the tile must be a power of two");
constexpr std::uint32_t kHybridSortTileThreads = kHybridSortTile / 2;
// The CPU reduction runs in parallel from this many terms. On an M4 Pro it was
// neutral below 2^20 terms (the GPU sort dominates there) and 1.15x and 1.3x
// faster end to end at 4M and 16M terms. Each chunk gets at least
// kHybridReduceChunkMinTerms.
constexpr std::size_t kHybridParallelReduceMinTerms = std::size_t{1} << 20;
constexpr std::size_t kHybridReduceChunkMinTerms = 16384;
// Above this many sort positions each merge stage gets its own command buffer.
constexpr std::uint32_t kHybridSingleCommandBufferMaxPositions = std::uint32_t{1} << 22;

struct HybridSortParams {
  std::uint32_t terms = 0;
  std::uint32_t j = 0;
  std::uint32_t k = 0;
};

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

class ScopedMetalBuffer {
 public:
  explicit ScopedMetalBuffer(id<MTLBuffer> buffer = nil) noexcept : buffer_(buffer) {}
  ScopedMetalBuffer(const ScopedMetalBuffer&) = delete;
  ScopedMetalBuffer& operator=(const ScopedMetalBuffer&) = delete;

  ~ScopedMetalBuffer() {
    [buffer_ release];
  }

  [[nodiscard]] id<MTLBuffer> get() const noexcept {
    return buffer_;
  }

  [[nodiscard]] id<MTLBuffer> release() noexcept {
    id<MTLBuffer> buffer = buffer_;
    buffer_ = nil;
    return buffer;
  }

 private:
  id<MTLBuffer> buffer_ = nil;
};

NSString* simplify_kernel_name(SimplifyKernel kernel) {
  switch (kernel) {
    case SimplifyKernel::InitKeys:
      return @"fp_simplify_words1_init_keys";
    case SimplifyKernel::BitonicSortStep:
      return @"fp_simplify_words1_bitonic_sort_step";
    case SimplifyKernel::MarkHeads:
      return @"fp_simplify_words1_mark_heads";
    case SimplifyKernel::ClearUint:
      return @"fp_simplify_clear_uint";
    case SimplifyKernel::PrefixSumStep:
      return @"fp_simplify_prefix_sum_step";
    case SimplifyKernel::ReduceByKey:
      return @"fp_simplify_words1_reduce_by_key";
    case SimplifyKernel::CompactSurvivors:
      return @"fp_simplify_words1_compact_survivors";
    case SimplifyKernel::HybridTileSort:
      return @"fp_simplify_words1_hybrid_tile_sort";
    case SimplifyKernel::HybridTileMerge:
      return @"fp_simplify_words1_hybrid_tile_merge";
    case SimplifyKernel::HybridGlobalStep:
      return @"fp_simplify_words1_hybrid_global_step";
    case SimplifyKernel::HybridGatherCoeffs:
      return @"fp_simplify_words1_hybrid_gather_coeffs";
  }
  return @"fp_simplify_words1_init_keys";
}

// Embedded at configure time from kernels/simplify.metal (see CMakeLists.txt).
constexpr const char kSimplifyKernelSource[] =
#include "metal/simplify_kernel_source.inc"
    ;

#if WOLFGANG_BUILD_METAL_PRECOMPILED_KERNELS
// Built from kernels/simplify.metal at build time (see CMakeLists.txt).
alignas(16) constexpr unsigned char kSimplifyMetallib[] = {
#include "metal/simplify_metallib.inc"
};
#endif

// Benchmark-only: WOLFGANG_METAL_BENCH_RUNTIME_KERNELS=1 skips the precompiled
// library so first-call timings of both paths can come from one build.
bool runtime_kernels_forced() {
  const char* value = std::getenv("WOLFGANG_METAL_BENCH_RUNTIME_KERNELS");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

// Loads (or compiles) the simplify kernels once per process; every pipeline
// comes from this library. The caller holds the pipeline cache mutex.
id<MTLLibrary> simplify_library(id<MTLDevice> device) {
  static id<MTLLibrary> library = nil;
  if (library != nil) {
    return library;
  }
#if WOLFGANG_BUILD_METAL_PRECOMPILED_KERNELS
  if (!runtime_kernels_forced()) {
    dispatch_data_t data = dispatch_data_create(
        kSimplifyMetallib, sizeof(kSimplifyMetallib), nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NSError* load_error = nil;
    library = [device newLibraryWithData:data error:&load_error];
    dispatch_release(data);
    if (library != nil) {
      return library;
    }
    // Fall through to the embedded source if this device rejects the library.
  }
#endif
  NSString* source = [NSString stringWithUTF8String:kSimplifyKernelSource];
  if (source == nil) {
    throw std::runtime_error("failed to build Metal simplify kernel source string");
  }
  MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
  options.preprocessorMacros = @{@"WOLFGANG_HYBRID_SORT_TILE" : @(kHybridSortTile)};
  NSError* error = nil;
  library = [device newLibraryWithSource:source options:options error:&error];
  [options release];
  if (library == nil) {
    throw std::runtime_error(
        "Metal failed to compile simplify kernels: " + nsstring_to_string([error localizedDescription]));
  }
  return library;
}

id<MTLComputePipelineState> simplify_pipeline(id<MTLDevice> device, SimplifyKernel kernel) {
  // Public simplify reaches this cache, so concurrent callers must not race
  // on first use.
  static std::mutex mutex;
  const std::lock_guard<std::mutex> lock(mutex);
  static id<MTLComputePipelineState> init = nil;
  static id<MTLComputePipelineState> sort = nil;
  static id<MTLComputePipelineState> mark = nil;
  static id<MTLComputePipelineState> clear = nil;
  static id<MTLComputePipelineState> prefix = nil;
  static id<MTLComputePipelineState> reduce = nil;
  static id<MTLComputePipelineState> compact = nil;
  static id<MTLComputePipelineState> hybrid_tile_sort = nil;
  static id<MTLComputePipelineState> hybrid_tile_merge = nil;
  static id<MTLComputePipelineState> hybrid_global_step = nil;
  static id<MTLComputePipelineState> hybrid_gather_coeffs = nil;

  id<MTLComputePipelineState>* selected = nullptr;
  switch (kernel) {
    case SimplifyKernel::InitKeys:
      selected = &init;
      break;
    case SimplifyKernel::BitonicSortStep:
      selected = &sort;
      break;
    case SimplifyKernel::MarkHeads:
      selected = &mark;
      break;
    case SimplifyKernel::ClearUint:
      selected = &clear;
      break;
    case SimplifyKernel::PrefixSumStep:
      selected = &prefix;
      break;
    case SimplifyKernel::ReduceByKey:
      selected = &reduce;
      break;
    case SimplifyKernel::CompactSurvivors:
      selected = &compact;
      break;
    case SimplifyKernel::HybridTileSort:
      selected = &hybrid_tile_sort;
      break;
    case SimplifyKernel::HybridTileMerge:
      selected = &hybrid_tile_merge;
      break;
    case SimplifyKernel::HybridGlobalStep:
      selected = &hybrid_global_step;
      break;
    case SimplifyKernel::HybridGatherCoeffs:
      selected = &hybrid_gather_coeffs;
      break;
  }
  if (*selected == nil) {
    id<MTLFunction> function = [simplify_library(device) newFunctionWithName:simplify_kernel_name(kernel)];
    if (function == nil) {
      throw std::runtime_error(
          "Metal simplify kernel function is missing: " + nsstring_to_string(simplify_kernel_name(kernel)));
    }
    NSError* error = nil;
    *selected = [device newComputePipelineStateWithFunction:function error:&error];
    [function release];
    if (*selected == nil) {
      throw std::runtime_error(
          "Metal failed to create simplify pipeline: " + nsstring_to_string([error localizedDescription]));
    }
  }
  return *selected;
}

void prewarm_simplify_pipelines(id<MTLDevice> device) {
  (void)simplify_pipeline(device, SimplifyKernel::InitKeys);
  (void)simplify_pipeline(device, SimplifyKernel::BitonicSortStep);
  (void)simplify_pipeline(device, SimplifyKernel::MarkHeads);
  (void)simplify_pipeline(device, SimplifyKernel::ClearUint);
  (void)simplify_pipeline(device, SimplifyKernel::PrefixSumStep);
  (void)simplify_pipeline(device, SimplifyKernel::ReduceByKey);
  (void)simplify_pipeline(device, SimplifyKernel::CompactSurvivors);
}

double elapsed_seconds(TimePoint start, TimePoint stop) {
  return std::chrono::duration<double>(stop - start).count();
}

std::uint32_t checked_u32(std::size_t value, const char* name) {
  if (value > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::invalid_argument(std::string(name) + " exceeds Metal uint grid limits");
  }
  return static_cast<std::uint32_t>(value);
}

std::size_t next_power_of_two(std::size_t value) {
  std::size_t result = 1;
  while (result < value) {
    if (result > std::numeric_limits<std::uint32_t>::max() / 2U) {
      throw std::invalid_argument("Metal simplify candidate exceeds uint32 padded size limit");
    }
    result <<= 1U;
  }
  return result;
}

std::uint64_t fixed32_threshold(double threshold) {
  if (threshold <= 0.0) {
    return 0;
  }
  const double scaled = std::floor(std::ldexp(threshold, 32));
  if (!std::isfinite(scaled) || scaled >= static_cast<double>(std::numeric_limits<std::uint64_t>::max())) {
    throw std::invalid_argument("Metal simplify fixed32 threshold is out of range");
  }
  return static_cast<std::uint64_t>(std::max(0.0, scaled));
}

std::uint64_t fixed64_threshold_square(double threshold) {
  if (threshold <= 0.0) {
    return 0;
  }
  if (threshold >= 1.0) {
    throw std::invalid_argument(
        "Metal words1 simplify candidate requires the fixed32 magnitude threshold "
        "to fit exact uint64 square comparison");
  }
  const long double threshold_wide = static_cast<long double>(threshold);
  const long double scaled = std::floor(std::ldexp(threshold_wide * threshold_wide, 64));
  const long double first_unsafe_integer = std::ldexp(static_cast<long double>(1), 64);
  if (!std::isfinite(scaled) || scaled >= first_unsafe_integer) {
    throw std::invalid_argument(
        "Metal words1 simplify candidate requires the fixed32 magnitude threshold "
        "to fit exact uint64 square comparison");
  }
  return static_cast<std::uint64_t>(std::max(static_cast<long double>(0), scaled));
}

bool double_to_signed_fixed32(double value, std::int64_t& out) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint64_t exponent_bits = (bits >> 52U) & 0x7ffULL;
  const std::uint64_t fraction = bits & 0x000fffffffffffffULL;
  const bool negative = (bits >> 63U) != 0ULL;
  if (exponent_bits == 0ULL && fraction == 0ULL) {
    out = 0;
    return true;
  }
  if (exponent_bits == 0ULL || exponent_bits == 0x7ffULL) {
    return false;
  }

  const int exponent = static_cast<int>(exponent_bits) - 1023;
  const std::uint64_t mantissa = (1ULL << 52U) | fraction;
  const int shift = exponent + 32 - 52;
  unsigned __int128 magnitude = 0;
  if (shift >= 0) {
    if (shift >= 128) {
      return false;
    }
    magnitude = static_cast<unsigned __int128>(mantissa) << static_cast<unsigned>(shift);
  } else {
    const unsigned right_shift = static_cast<unsigned>(-shift);
    if (right_shift >= 64U) {
      return false;
    }
    const std::uint64_t discarded_mask = (1ULL << right_shift) - 1ULL;
    if ((mantissa & discarded_mask) != 0ULL) {
      return false;
    }
    magnitude = mantissa >> right_shift;
  }

  if (magnitude > static_cast<unsigned __int128>(std::numeric_limits<std::int64_t>::max())) {
    return false;
  }
  const auto signed_magnitude = static_cast<std::int64_t>(magnitude);
  out = negative ? -signed_magnitude : signed_magnitude;
  return true;
}

std::uint64_t abs_i64_as_u64(std::int64_t value) {
  return value < 0 ? static_cast<std::uint64_t>(-value) : static_cast<std::uint64_t>(value);
}

void dispatch_1d(id<MTLComputeCommandEncoder> encoder, std::size_t entries) {
  [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(entries), 1, 1)
      threadsPerThreadgroup:MTLSizeMake(kThreadsPerThreadgroup, 1, 1)];
}

id<MTLBuffer> inclusive_prefix_sum(
    id<MTLDevice> device,
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> input,
    id<MTLBuffer> temp_a,
    id<MTLBuffer> temp_b,
    std::uint32_t entries,
    std::size_t& pass_count) {
  id<MTLBuffer> current = input;
  id<MTLBuffer> next = temp_a;
  for (std::uint32_t offset = 1; offset < entries; offset <<= 1U) {
    PrefixSumParams params{entries, offset};
    [encoder setComputePipelineState:simplify_pipeline(device, SimplifyKernel::PrefixSumStep)];
    [encoder setBuffer:current offset:0 atIndex:0];
    [encoder setBuffer:next offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];
    dispatch_1d(encoder, entries);
    current = next;
    next = (next == temp_a) ? temp_b : temp_a;
    ++pass_count;
  }
  return current;
}

template <typename Impl>
double max_abs_shared_coefficients(const Impl& impl) {
  const auto* coeffs =
      static_cast<const std::complex<double>*>([impl.coeffs contents]);
  double max_abs = 0.0;
  for (std::size_t index = 0; index < impl.num_terms; ++index) {
    max_abs = std::max(max_abs, std::abs(coeffs[index]));
  }
  return max_abs;
}

template <typename Impl>
Fixed32CoefficientScan scan_shared_fixed32_coefficients(const Impl& impl) {
  const auto* coeffs =
      static_cast<const std::complex<double>*>([impl.coeffs contents]);
  Fixed32CoefficientScan scan;
  for (std::size_t index = 0; index < impl.num_terms; ++index) {
    std::int64_t real_fixed32 = 0;
    std::int64_t imag_fixed32 = 0;
    if (!double_to_signed_fixed32(coeffs[index].real(), real_fixed32) ||
        !double_to_signed_fixed32(coeffs[index].imag(), imag_fixed32)) {
      throw std::invalid_argument(
          "Metal words1 simplify candidate requires coefficients exactly "
          "representable as signed fixed32 dyadic values");
    }
    scan.max_abs_component =
        std::max(scan.max_abs_component, abs_i64_as_u64(real_fixed32));
    scan.max_abs_component =
        std::max(scan.max_abs_component, abs_i64_as_u64(imag_fixed32));
  }
  return scan;
}

std::size_t simplify_workspace_bytes(std::size_t padded_terms) {
  const std::size_t u64_bytes = checked_bytes<std::uint64_t>(padded_terms, "Metal simplify uint64 scratch");
  const std::size_t coeff_bytes =
      checked_bytes<std::complex<double>>(padded_terms, "Metal simplify coefficient scratch");
  const std::size_t u32_bytes = checked_bytes<std::uint32_t>(padded_terms, "Metal simplify uint32 scratch");
  std::size_t total = 0;
  auto add = [&](std::size_t bytes) {
    if (total > std::numeric_limits<std::size_t>::max() - bytes) {
      throw std::overflow_error("Metal simplify workspace byte estimate overflow");
    }
    total += bytes;
  };
  add(u64_bytes * 6U);    // sorted, reduced, and output x/z buffers.
  add(coeff_bytes * 3U);  // sorted, reduced, and output coefficient buffers.
  add(u32_bytes * 5U);    // valid, head, survivor, invalid, and prefix scratch.
  return total;
}

}  // namespace

MetalSimplifyCandidateResult simplify_words1_device_candidate_for_testing(
    const DevicePauliSum& input,
    double atol,
    double rtol) {
  metal_detail::ScopedAutoreleasePool pool;
  if (!input.impl_) {
    throw std::runtime_error("DevicePauliSum is empty or moved-from");
  }
  validate_simplify_tolerances(atol, rtol);
  if (input.impl_->words != 1) {
    throw std::invalid_argument(
        "Metal words1 simplify candidate requires exactly one packed word");
  }
  prewarm_simplify_pipelines(input.impl_->device);
  const TimePoint total_start = Clock::now();

  const std::size_t input_terms = input.impl_->num_terms;
  const std::size_t padded_terms = next_power_of_two(std::max<std::size_t>(input_terms, 1));
  const std::uint32_t padded_u32 = checked_u32(padded_terms, "Metal simplify padded terms");
  const double max_abs_input = rtol == 0.0 ? 0.0 : max_abs_shared_coefficients(*input.impl_);
  const double drop_threshold = atol + rtol * max_abs_input;
  const std::uint64_t drop_threshold_fixed32 = fixed32_threshold(drop_threshold);
  const std::uint64_t drop_threshold_square_fixed64 =
      fixed64_threshold_square(drop_threshold);
  const Fixed32CoefficientScan fixed32_scan = scan_shared_fixed32_coefficients(*input.impl_);
  if (input_terms > 0 && fixed32_scan.max_abs_component > 0U) {
    const unsigned __int128 worst_case_accumulator_component =
        static_cast<unsigned __int128>(fixed32_scan.max_abs_component) *
        static_cast<unsigned __int128>(input_terms);
    if (worst_case_accumulator_component >
        static_cast<unsigned __int128>(std::numeric_limits<std::int64_t>::max())) {
      throw std::invalid_argument(
          "Metal words1 simplify candidate fixed32 coefficient sum may overflow");
    }
    if (drop_threshold_fixed32 != 0) {
      // floor(sqrt((2^64 - 1) / 2)): if both real and imaginary components are
      // at or below this bound, real^2 + imag^2 fits exact uint64 arithmetic.
      constexpr std::uint64_t kMaxExactSquareComponent = 3037000499ULL;
      if (worst_case_accumulator_component >
          static_cast<unsigned __int128>(kMaxExactSquareComponent)) {
        throw std::invalid_argument(
            "Metal words1 simplify candidate requires fixed32 coefficient sums "
            "to fit exact uint64 square comparison");
      }
    }
  }
  const SimplifyWords1Params params{
      checked_u32(input_terms, "Metal simplify input terms"),
      padded_u32,
      drop_threshold_fixed32,
      drop_threshold_square_fixed64,
      drop_threshold_fixed32 != 0 ? 1U : 0U,
  };

  const std::size_t u64_bytes = checked_bytes<std::uint64_t>(padded_terms, "Metal simplify uint64 bytes");
  const std::size_t coeff_bytes =
      checked_bytes<std::complex<double>>(padded_terms, "Metal simplify coefficient bytes");
  const std::size_t u32_bytes = checked_bytes<std::uint32_t>(padded_terms, "Metal simplify uint32 bytes");
  const std::size_t invalid_bytes = checked_bytes<std::uint32_t>(1, "Metal simplify invalid flag bytes");
  const std::size_t workspace_bytes = simplify_workspace_bytes(padded_terms);
  const TimePoint preflight_end = Clock::now();

  if (input_terms == 0) {
    auto out = std::make_unique<DevicePauliSum::Impl>();
    out->num_qubits = input.impl_->num_qubits;
    out->words = input.impl_->words;
    out->num_terms = 0;
    out->device_ordinal = input.impl_->device_ordinal;
    out->device = [input.impl_->device retain];
    out->command_queue = [input.impl_->command_queue retain];
    MetalSimplifyCandidateResult result;
    result.output = DevicePauliSum(std::move(out));
    result.input_terms = 0;
    result.output_terms = 0;
    result.padded_terms = padded_terms;
    result.workspace_reserved_bytes = workspace_bytes;
    result.timing.host_preflight_seconds = elapsed_seconds(total_start, preflight_end);
    result.timing.total_observed_seconds = elapsed_seconds(total_start, Clock::now());
    return result;
  }

  const TimePoint allocation_start = Clock::now();
  ScopedMetalBuffer sort_x(make_shared_buffer(input.impl_->device, nullptr, u64_bytes, "Metal simplify sorted x"));
  ScopedMetalBuffer sort_z(make_shared_buffer(input.impl_->device, nullptr, u64_bytes, "Metal simplify sorted z"));
  ScopedMetalBuffer sort_coeffs(
      make_shared_buffer(input.impl_->device, nullptr, coeff_bytes, "Metal simplify sorted coefficients"));
  ScopedMetalBuffer sort_valid(
      make_shared_buffer(input.impl_->device, nullptr, u32_bytes, "Metal simplify sorted validity flags"));
  ScopedMetalBuffer invalid_coefficients(
      make_shared_buffer(input.impl_->device, nullptr, invalid_bytes, "Metal simplify invalid coefficient flag"));
  ScopedMetalBuffer head_flags(
      make_shared_buffer(input.impl_->device, nullptr, u32_bytes, "Metal simplify head flags"));
  ScopedMetalBuffer prefix_a(
      make_shared_buffer(input.impl_->device, nullptr, u32_bytes, "Metal simplify prefix scratch A"));
  ScopedMetalBuffer prefix_b(
      make_shared_buffer(input.impl_->device, nullptr, u32_bytes, "Metal simplify prefix scratch B"));
  ScopedMetalBuffer reduced_x(
      make_shared_buffer(input.impl_->device, nullptr, u64_bytes, "Metal simplify reduced x"));
  ScopedMetalBuffer reduced_z(
      make_shared_buffer(input.impl_->device, nullptr, u64_bytes, "Metal simplify reduced z"));
  ScopedMetalBuffer reduced_coeffs(
      make_shared_buffer(input.impl_->device, nullptr, coeff_bytes, "Metal simplify reduced coefficients"));
  ScopedMetalBuffer survivor_flags(
      make_shared_buffer(input.impl_->device, nullptr, u32_bytes, "Metal simplify survivor flags"));
  ScopedMetalBuffer out_x(
      make_shared_buffer(input.impl_->device, nullptr, u64_bytes, "Metal simplify output x"));
  ScopedMetalBuffer out_z(
      make_shared_buffer(input.impl_->device, nullptr, u64_bytes, "Metal simplify output z"));
  ScopedMetalBuffer out_coeffs(
      make_shared_buffer(input.impl_->device, nullptr, coeff_bytes, "Metal simplify output coefficients"));
  const TimePoint allocation_end = Clock::now();

  const TimePoint encoding_start = Clock::now();
  id<MTLCommandBuffer> command_buffer = [input.impl_->command_queue commandBuffer];
  if (command_buffer == nil) {
    throw std::runtime_error("Metal failed to create a simplify command buffer");
  }
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  if (encoder == nil) {
    throw std::runtime_error("Metal failed to create a simplify command encoder");
  }

  const PrefixSumParams clear_invalid_params{1, 0};
  std::size_t clear_invalid_dispatches = 0;
  std::size_t init_key_dispatches = 0;
  std::size_t mark_head_dispatches = 0;
  std::size_t clear_survivor_dispatches = 0;
  std::size_t reduce_by_key_dispatches = 0;
  std::size_t compact_survivor_dispatches = 0;
  [encoder setComputePipelineState:simplify_pipeline(input.impl_->device, SimplifyKernel::ClearUint)];
  [encoder setBuffer:invalid_coefficients.get() offset:0 atIndex:0];
  [encoder setBytes:&clear_invalid_params length:sizeof(clear_invalid_params) atIndex:1];
  dispatch_1d(encoder, 1);
  ++clear_invalid_dispatches;

  [encoder setComputePipelineState:simplify_pipeline(input.impl_->device, SimplifyKernel::InitKeys)];
  [encoder setBuffer:input.impl_->x offset:0 atIndex:0];
  [encoder setBuffer:input.impl_->z offset:0 atIndex:1];
  [encoder setBuffer:input.impl_->coeffs offset:0 atIndex:2];
  [encoder setBuffer:sort_x.get() offset:0 atIndex:3];
  [encoder setBuffer:sort_z.get() offset:0 atIndex:4];
  [encoder setBuffer:sort_coeffs.get() offset:0 atIndex:5];
  [encoder setBuffer:sort_valid.get() offset:0 atIndex:6];
  [encoder setBuffer:invalid_coefficients.get() offset:0 atIndex:7];
  [encoder setBytes:&params length:sizeof(params) atIndex:8];
  dispatch_1d(encoder, padded_terms);
  ++init_key_dispatches;

  std::size_t bitonic_passes = 0;
  for (std::uint32_t k = 2; k <= padded_u32; k <<= 1U) {
    for (std::uint32_t j = k >> 1U; j > 0; j >>= 1U) {
      const BitonicSortParams sort_params{padded_u32, j, k};
      [encoder setComputePipelineState:simplify_pipeline(input.impl_->device, SimplifyKernel::BitonicSortStep)];
      [encoder setBuffer:sort_x.get() offset:0 atIndex:0];
      [encoder setBuffer:sort_z.get() offset:0 atIndex:1];
      [encoder setBuffer:sort_coeffs.get() offset:0 atIndex:2];
      [encoder setBuffer:sort_valid.get() offset:0 atIndex:3];
      [encoder setBytes:&sort_params length:sizeof(sort_params) atIndex:4];
      dispatch_1d(encoder, padded_terms);
      ++bitonic_passes;
    }
  }

  [encoder setComputePipelineState:simplify_pipeline(input.impl_->device, SimplifyKernel::MarkHeads)];
  [encoder setBuffer:sort_x.get() offset:0 atIndex:0];
  [encoder setBuffer:sort_z.get() offset:0 atIndex:1];
  [encoder setBuffer:sort_valid.get() offset:0 atIndex:2];
  [encoder setBuffer:head_flags.get() offset:0 atIndex:3];
  [encoder setBytes:&params length:sizeof(params) atIndex:4];
  dispatch_1d(encoder, padded_terms);
  ++mark_head_dispatches;

  std::size_t prefix_passes = 0;
  id<MTLBuffer> head_prefix = inclusive_prefix_sum(
      input.impl_->device,
      encoder,
      head_flags.get(),
      prefix_a.get(),
      prefix_b.get(),
      padded_u32,
      prefix_passes);

  const PrefixSumParams clear_params{padded_u32, 0};
  [encoder setComputePipelineState:simplify_pipeline(input.impl_->device, SimplifyKernel::ClearUint)];
  [encoder setBuffer:survivor_flags.get() offset:0 atIndex:0];
  [encoder setBytes:&clear_params length:sizeof(clear_params) atIndex:1];
  dispatch_1d(encoder, padded_terms);
  ++clear_survivor_dispatches;

  [encoder setComputePipelineState:simplify_pipeline(input.impl_->device, SimplifyKernel::ReduceByKey)];
  [encoder setBuffer:sort_x.get() offset:0 atIndex:0];
  [encoder setBuffer:sort_z.get() offset:0 atIndex:1];
  [encoder setBuffer:sort_coeffs.get() offset:0 atIndex:2];
  [encoder setBuffer:sort_valid.get() offset:0 atIndex:3];
  [encoder setBuffer:head_flags.get() offset:0 atIndex:4];
  [encoder setBuffer:head_prefix offset:0 atIndex:5];
  [encoder setBuffer:reduced_x.get() offset:0 atIndex:6];
  [encoder setBuffer:reduced_z.get() offset:0 atIndex:7];
  [encoder setBuffer:reduced_coeffs.get() offset:0 atIndex:8];
  [encoder setBuffer:survivor_flags.get() offset:0 atIndex:9];
  [encoder setBytes:&params length:sizeof(params) atIndex:10];
  dispatch_1d(encoder, padded_terms);
  ++reduce_by_key_dispatches;

  id<MTLBuffer> survivor_prefix = inclusive_prefix_sum(
      input.impl_->device,
      encoder,
      survivor_flags.get(),
      prefix_a.get(),
      prefix_b.get(),
      padded_u32,
      prefix_passes);

  const PrefixSumParams compact_params{padded_u32, 0};
  [encoder setComputePipelineState:simplify_pipeline(input.impl_->device, SimplifyKernel::CompactSurvivors)];
  [encoder setBuffer:reduced_x.get() offset:0 atIndex:0];
  [encoder setBuffer:reduced_z.get() offset:0 atIndex:1];
  [encoder setBuffer:reduced_coeffs.get() offset:0 atIndex:2];
  [encoder setBuffer:survivor_flags.get() offset:0 atIndex:3];
  [encoder setBuffer:survivor_prefix offset:0 atIndex:4];
  [encoder setBuffer:out_x.get() offset:0 atIndex:5];
  [encoder setBuffer:out_z.get() offset:0 atIndex:6];
  [encoder setBuffer:out_coeffs.get() offset:0 atIndex:7];
  [encoder setBytes:&compact_params length:sizeof(compact_params) atIndex:8];
  dispatch_1d(encoder, padded_terms);
  ++compact_survivor_dispatches;

  [encoder endEncoding];
  const TimePoint encoding_end = Clock::now();
  const TimePoint execution_start = Clock::now();
  wait_for_completion(command_buffer, "simplify words1 candidate");
  const TimePoint execution_end = Clock::now();

  const TimePoint accounting_start = Clock::now();
  const auto* invalid_values =
      static_cast<const std::uint32_t*>([invalid_coefficients.get() contents]);
  if (invalid_values[0] != 0U) {
    throw std::invalid_argument(
        "Metal words1 simplify candidate requires coefficients exactly representable as signed fixed32 dyadic values");
  }
  const auto* survivor_prefix_values =
      static_cast<const std::uint32_t*>([survivor_prefix contents]);
  const std::size_t output_terms = survivor_prefix_values[padded_terms - 1U];
  if (output_terms > input_terms) {
    throw std::runtime_error("Metal simplify candidate produced an invalid survivor count");
  }

  auto out = std::make_unique<DevicePauliSum::Impl>();
  out->num_qubits = input.impl_->num_qubits;
  out->words = input.impl_->words;
  out->num_terms = output_terms;
  out->device_ordinal = input.impl_->device_ordinal;
  out->device = [input.impl_->device retain];
  out->command_queue = [input.impl_->command_queue retain];
  out->x = out_x.release();
  out->z = out_z.release();
  out->coeffs = out_coeffs.release();
  const TimePoint accounting_end = Clock::now();

  MetalSimplifyCandidateResult result;
  result.output = DevicePauliSum(std::move(out));
  result.input_terms = input_terms;
  result.output_terms = output_terms;
  result.padded_terms = padded_terms;
  result.bitonic_passes = bitonic_passes;
  result.prefix_sum_passes = prefix_passes;
  result.workspace_reserved_bytes = workspace_bytes;
  result.timing.host_preflight_seconds = elapsed_seconds(total_start, preflight_end);
  result.timing.scratch_and_output_allocation_seconds =
      elapsed_seconds(allocation_start, allocation_end);
  result.timing.command_encoding_seconds = elapsed_seconds(encoding_start, encoding_end);
  result.timing.command_execution_seconds = elapsed_seconds(execution_start, execution_end);
  result.timing.output_accounting_seconds = elapsed_seconds(accounting_start, accounting_end);
  result.timing.total_observed_seconds = elapsed_seconds(total_start, accounting_end);
  result.dispatch_counts.clear_invalid = clear_invalid_dispatches;
  result.dispatch_counts.init_keys = init_key_dispatches;
  result.dispatch_counts.bitonic_sort = bitonic_passes;
  result.dispatch_counts.mark_heads = mark_head_dispatches;
  result.dispatch_counts.prefix_sum = prefix_passes;
  result.dispatch_counts.clear_survivors = clear_survivor_dispatches;
  result.dispatch_counts.reduce_by_key = reduce_by_key_dispatches;
  result.dispatch_counts.compact_survivors = compact_survivor_dispatches;
  result.dispatch_counts.total_kernel_dispatches =
      clear_invalid_dispatches + init_key_dispatches + bitonic_passes +
      mark_head_dispatches + prefix_passes + clear_survivor_dispatches +
      reduce_by_key_dispatches + compact_survivor_dispatches;
  return result;
}

namespace {

std::atomic<std::uint64_t> g_hybrid_simplify_runs{0};

// Shared scratch for the hybrid. Allocation failure raises std::bad_alloc
// (MemoryError in Python), as CPU simplify would for the same size.
id<MTLBuffer> make_hybrid_buffer(id<MTLDevice> device, const void* source, std::size_t bytes) {
  id<MTLBuffer> buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    throw std::bad_alloc();
  }
  if (source != nullptr) {
    std::memcpy([buffer contents], source, bytes);
  }
  return buffer;
}

}  // namespace

bool hybrid_simplify_supported(id<MTLDevice> device) noexcept {
  // Any pipeline failure keeps the host bridge; tests assert through
  // hybrid_simplify_runs_for_testing that the hybrid really runs.
  try {
    ScopedAutoreleasePool pool;
    for (const SimplifyKernel kernel :
         {SimplifyKernel::HybridTileSort, SimplifyKernel::HybridTileMerge,
          SimplifyKernel::HybridGlobalStep, SimplifyKernel::HybridGatherCoeffs}) {
      if ([simplify_pipeline(device, kernel) maxTotalThreadsPerThreadgroup] < kHybridSortTileThreads) {
        return false;
      }
    }
    return true;
  } catch (...) {
    return false;
  }
}

std::uint64_t hybrid_simplify_runs_for_testing() noexcept {
  return g_hybrid_simplify_runs.load(std::memory_order_relaxed);
}


HybridSimplifyOutput simplify_words1_hybrid(
    id<MTLDevice> device,
    id<MTLCommandQueue> command_queue,
    id<MTLBuffer> x,
    id<MTLBuffer> z,
    id<MTLBuffer> coeffs,
    std::size_t terms,
    double atol,
    double rtol) {
  ScopedAutoreleasePool pool;
  // Validates finite inputs exactly as CPU simplify does.
  const detail::SimplifyDropRule rule = detail::SimplifyDropRule::from_coefficients(
      std::span<const std::complex<double>>(
          static_cast<const std::complex<double>*>([coeffs contents]), terms),
      atol,
      rtol);
  HybridSimplifyOutput out;
  g_hybrid_simplify_runs.fetch_add(1, std::memory_order_relaxed);
  if (terms == 0) {
    return out;
  }

  // The network spans next_power_of_two(terms) positions, but positions past
  // `terms` are never stored (see the kernel comment), so buffers hold `terms`
  // entries and only the dispatch shape depends on the padded size.
  const std::uint32_t terms_u32 = checked_u32(terms, "Metal hybrid simplify terms");
  const std::uint32_t padded_u32 =
      checked_u32(next_power_of_two(terms), "Metal hybrid simplify padded terms");
  const std::uint32_t tiles = (terms_u32 - 1U) / kHybridSortTile + 1U;
  id<MTLComputePipelineState> tile_sort = simplify_pipeline(device, SimplifyKernel::HybridTileSort);
  id<MTLComputePipelineState> tile_merge = simplify_pipeline(device, SimplifyKernel::HybridTileMerge);
  id<MTLComputePipelineState> global_step = simplify_pipeline(device, SimplifyKernel::HybridGlobalStep);
  id<MTLComputePipelineState> gather = simplify_pipeline(device, SimplifyKernel::HybridGatherCoeffs);
  if ([tile_sort maxTotalThreadsPerThreadgroup] < kHybridSortTileThreads ||
      [tile_merge maxTotalThreadsPerThreadgroup] < kHybridSortTileThreads) {
    throw std::runtime_error("Metal hybrid simplify tile kernels exceed this device's threadgroup limit");
  }
  ScopedMetalBuffer sort_x(make_hybrid_buffer(
      device, nullptr, checked_bytes<std::uint64_t>(terms, "Metal hybrid sort x")));
  ScopedMetalBuffer sort_z(make_hybrid_buffer(
      device, nullptr, checked_bytes<std::uint64_t>(terms, "Metal hybrid sort z")));
  ScopedMetalBuffer sort_index(make_hybrid_buffer(
      device, nullptr, checked_bytes<std::uint32_t>(terms, "Metal hybrid sort index")));
  ScopedMetalBuffer sort_coeffs(make_hybrid_buffer(
      device, nullptr, checked_bytes<std::complex<double>>(terms, "Metal hybrid sorted coefficients")));

  // Ordinary sizes encode the whole sort into one command buffer. Larger inputs
  // commit one buffer per merge stage, queued in order with only the last one
  // waited on, so no single buffer approaches the GPU watchdog on a Mac that
  // drives a display (one buffer ran about two seconds at 2^26 positions on an
  // M4 Pro).
  const bool split_stages = padded_u32 > kHybridSingleCommandBufferMaxPositions;
  std::vector<id<MTLCommandBuffer>> committed;
  id<MTLCommandBuffer> command_buffer = nil;
  id<MTLComputeCommandEncoder> encoder = nil;
  const auto begin_command_buffer = [&] {
    command_buffer = [command_queue commandBuffer];
    if (command_buffer == nil) {
      throw std::runtime_error("Metal failed to create a hybrid simplify command buffer");
    }
    encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      throw std::runtime_error("Metal failed to create a hybrid simplify command encoder");
    }
  };
  const auto commit_stage = [&] {
    if (!split_stages) {
      return;
    }
    [encoder endEncoding];
    [command_buffer commit];
    committed.push_back(command_buffer);
    begin_command_buffer();
  };
  begin_command_buffer();
  const MTLSize tile_grid = MTLSizeMake(tiles, 1, 1);
  const MTLSize tile_threads = MTLSizeMake(kHybridSortTileThreads, 1, 1);
  HybridSortParams params{terms_u32, 0, 0};
  [encoder setComputePipelineState:tile_sort];
  [encoder setBuffer:x offset:0 atIndex:0];
  [encoder setBuffer:z offset:0 atIndex:1];
  [encoder setBuffer:sort_x.get() offset:0 atIndex:2];
  [encoder setBuffer:sort_z.get() offset:0 atIndex:3];
  [encoder setBuffer:sort_index.get() offset:0 atIndex:4];
  [encoder setBytes:&params length:sizeof(params) atIndex:5];
  [encoder dispatchThreadgroups:tile_grid threadsPerThreadgroup:tile_threads];
  commit_stage();
  for (std::uint32_t k = 2U * kHybridSortTile; k != 0U && k <= padded_u32; k <<= 1U) {
    params.k = k;
    [encoder setBuffer:sort_x.get() offset:0 atIndex:0];
    [encoder setBuffer:sort_z.get() offset:0 atIndex:1];
    [encoder setBuffer:sort_index.get() offset:0 atIndex:2];
    [encoder setComputePipelineState:global_step];
    for (std::uint32_t j = k >> 1U; j >= kHybridSortTile; j >>= 1U) {
      params.j = j;
      [encoder setBytes:&params length:sizeof(params) atIndex:3];
      dispatch_1d(encoder, padded_u32 / 2U);
    }
    [encoder setComputePipelineState:tile_merge];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    [encoder dispatchThreadgroups:tile_grid threadsPerThreadgroup:tile_threads];
    commit_stage();
  }
  [encoder setComputePipelineState:gather];
  [encoder setBuffer:coeffs offset:0 atIndex:0];
  [encoder setBuffer:sort_index.get() offset:0 atIndex:1];
  [encoder setBuffer:sort_coeffs.get() offset:0 atIndex:2];
  [encoder setBytes:&params length:sizeof(params) atIndex:3];
  dispatch_1d(encoder, terms);
  [encoder endEncoding];
  wait_for_completion(command_buffer, "Metal hybrid simplify sort");
  // Earlier stage buffers ran before the last one on the same queue.
  for (id<MTLCommandBuffer> stage : committed) {
    if ([stage status] != MTLCommandBufferStatusCompleted) {
      NSError* error = [stage error];
      throw std::runtime_error(
          std::string("Metal hybrid simplify sort stage failed: ") +
          (error == nil ? std::string("unknown command buffer error")
                        : nsstring_to_string([error localizedDescription])));
    }
  }

  // The buffers hold every input term in canonical (x, z) order with
  // input-index tie-breaks, and its coefficient. Summing each run left to right,
  // seeded with its first coefficient, reproduces the CPU paths' input-order
  // sums bit for bit (signed zeros included).
  auto* keys_x = static_cast<std::uint64_t*>([sort_x.get() contents]);
  auto* keys_z = static_cast<std::uint64_t*>([sort_z.get() contents]);
  auto* sums = static_cast<std::complex<double>*>([sort_coeffs.get() contents]);

  if (terms < kHybridParallelReduceMinTerms) {
    // Sequential pass; survivors compact in place because the write position
    // never passes the read position.
    std::size_t survivors = 0;
    std::size_t position = 0;
    while (position < terms) {
      const std::uint64_t key_x = keys_x[position];
      const std::uint64_t key_z = keys_z[position];
      std::complex<double> accumulated = sums[position];
      std::size_t next = position + 1;
      while (next < terms && keys_x[next] == key_x && keys_z[next] == key_z) {
        accumulated += sums[next];
        ++next;
      }
      if (rule.survives(accumulated)) {
        keys_x[survivors] = key_x;
        keys_z[survivors] = key_z;
        sums[survivors] = accumulated;
        ++survivors;
      }
      position = next;
    }
    out.num_terms = survivors;
    if (survivors == 0) {
      return out;
    }
    // Keep the scratch buffers as the output when at most half of them would
    // be slack; otherwise copy survivors into exact-size buffers.
    if (survivors >= terms / 2) {
      out.x = sort_x.release();
      out.z = sort_z.release();
      out.coeffs = sort_coeffs.release();
    } else {
      out.x = make_hybrid_buffer(device, keys_x, survivors * sizeof(std::uint64_t));
      out.z = make_hybrid_buffer(device, keys_z, survivors * sizeof(std::uint64_t));
      out.coeffs = make_hybrid_buffer(device, sums, survivors * sizeof(std::complex<double>));
    }
    return out;
  }

  // Parallel pass over chunks that start on run boundaries, so no run is split
  // and every sum keeps its sequential input order. Pass 1 sums each run into
  // its head and marks surviving heads; pass 2 copies survivors to exact-size
  // outputs at per-chunk prefix offsets. In-place compaction is not safe across
  // chunks, because a later chunk's destination can overlap an earlier chunk's
  // unread runs.
  const std::size_t chunk_count = std::max<std::size_t>(
      2, std::min<std::size_t>(std::thread::hardware_concurrency(), terms / kHybridReduceChunkMinTerms));
  std::vector<std::size_t> chunk_start(chunk_count + 1, terms);
  chunk_start[0] = 0;
  for (std::size_t chunk = 1; chunk < chunk_count; ++chunk) {
    std::size_t start = std::max(chunk_start[chunk - 1], chunk * (terms / chunk_count));
    while (start < terms && start > 0 && keys_x[start] == keys_x[start - 1] &&
           keys_z[start] == keys_z[start - 1]) {
      ++start;
    }
    chunk_start[chunk] = start;
  }
  std::vector<std::uint8_t> keep(terms, 0);
  std::vector<std::size_t> chunk_survivors(chunk_count, 0);
  std::atomic<bool> overflowed{false};
  const std::size_t* starts = chunk_start.data();
  std::uint8_t* keep_flags = keep.data();
  std::size_t* counts = chunk_survivors.data();
  std::atomic<bool>* overflow_flag = &overflowed;
  const detail::SimplifyDropRule drop_rule = rule;
  dispatch_apply(chunk_count, DISPATCH_APPLY_AUTO, ^(std::size_t chunk) {
    std::size_t count = 0;
    std::size_t position = starts[chunk];
    const std::size_t end = starts[chunk + 1];
    while (position < end) {
      const std::uint64_t key_x = keys_x[position];
      const std::uint64_t key_z = keys_z[position];
      std::complex<double> accumulated = sums[position];
      std::size_t next = position + 1;
      while (next < end && keys_x[next] == key_x && keys_z[next] == key_z) {
        accumulated += sums[next];
        ++next;
      }
      // survives() throws on a non-finite sum; a block must not throw, so the
      // check is made here and raised after the pass.
      if (!std::isfinite(accumulated.real()) || !std::isfinite(accumulated.imag())) {
        overflow_flag->store(true, std::memory_order_relaxed);
      } else if (drop_rule.survives(accumulated)) {
        sums[position] = accumulated;
        keep_flags[position] = 1;
        ++count;
      }
      position = next;
    }
    counts[chunk] = count;
  });
  if (overflowed.load()) {
    detail::require_finite_simplify_accumulator(
        std::complex<double>(std::numeric_limits<double>::infinity(), 0.0));
  }
  std::vector<std::size_t> chunk_offset(chunk_count, 0);
  std::size_t survivors = 0;
  for (std::size_t chunk = 0; chunk < chunk_count; ++chunk) {
    chunk_offset[chunk] = survivors;
    survivors += chunk_survivors[chunk];
  }
  out.num_terms = survivors;
  if (survivors == 0) {
    return out;
  }
  ScopedMetalBuffer out_x(make_hybrid_buffer(device, nullptr, survivors * sizeof(std::uint64_t)));
  ScopedMetalBuffer out_z(make_hybrid_buffer(device, nullptr, survivors * sizeof(std::uint64_t)));
  ScopedMetalBuffer out_coeffs(
      make_hybrid_buffer(device, nullptr, survivors * sizeof(std::complex<double>)));
  auto* dst_x = static_cast<std::uint64_t*>([out_x.get() contents]);
  auto* dst_z = static_cast<std::uint64_t*>([out_z.get() contents]);
  auto* dst_coeffs = static_cast<std::complex<double>*>([out_coeffs.get() contents]);
  const std::size_t* offsets = chunk_offset.data();
  dispatch_apply(chunk_count, DISPATCH_APPLY_AUTO, ^(std::size_t chunk) {
    std::size_t write = offsets[chunk];
    for (std::size_t position = starts[chunk]; position < starts[chunk + 1]; ++position) {
      if (keep_flags[position] != 0) {
        dst_x[write] = keys_x[position];
        dst_z[write] = keys_z[position];
        dst_coeffs[write] = sums[position];
        ++write;
      }
    }
  });
  out.x = out_x.release();
  out.z = out_z.release();
  out.coeffs = out_coeffs.release();
  return out;
}

}  // namespace wolfgang::metal_detail

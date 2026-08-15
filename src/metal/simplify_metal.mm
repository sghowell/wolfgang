#include "simplify_metal.hpp"

#include "device_pauli_sum_metal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

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
  }
  return @"fp_simplify_words1_init_keys";
}

std::string simplify_kernel_source_text() {
  const std::string source_file = __FILE__;
  const std::string marker = "simplify_metal.mm";
  const std::size_t marker_pos = source_file.rfind(marker);
  if (marker_pos == std::string::npos) {
    throw std::runtime_error("cannot locate Metal simplify source directory");
  }
  const std::string kernel_path =
      source_file.substr(0, marker_pos) + "kernels/simplify.metal";
  std::ifstream input(kernel_path);
  if (!input) {
    throw std::runtime_error("failed to open Metal simplify kernel source: " + kernel_path);
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

NSString* simplify_kernel_source() {
  static const std::string source_text = simplify_kernel_source_text();
  NSString* source = [NSString stringWithUTF8String:source_text.c_str()];
  if (source == nil) {
    throw std::runtime_error("failed to build Metal simplify kernel source string");
  }
  return source;
}

id<MTLComputePipelineState> simplify_pipeline(id<MTLDevice> device, SimplifyKernel kernel) {
  static id<MTLComputePipelineState> init = nil;
  static id<MTLComputePipelineState> sort = nil;
  static id<MTLComputePipelineState> mark = nil;
  static id<MTLComputePipelineState> clear = nil;
  static id<MTLComputePipelineState> prefix = nil;
  static id<MTLComputePipelineState> reduce = nil;
  static id<MTLComputePipelineState> compact = nil;

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
  }
  if (*selected == nil) {
    *selected = make_compute_pipeline(device, simplify_kernel_name(kernel), simplify_kernel_source());
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

}  // namespace wolfgang::metal_detail

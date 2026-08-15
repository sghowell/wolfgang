#include "device_commutation_matrix_metal.hpp"
#include "device_pauli_sum_metal.hpp"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace wolfgang {

namespace {

struct CommutationKernelParams {
  std::uint64_t lhs_terms = 0;
  std::uint64_t rhs_terms = 0;
  std::uint64_t words = 0;
};

enum class CommutationKernelVariant {
  Words1,
  Words2,
  Generic,
  FlatGeneric,
};

constexpr const char* kMetalLibraryPathEnv = "FASTPAULI_EXPERIMENTAL_METAL_LIBRARY_PATH";
constexpr const char* kMetalOutputStorageEnv = "FASTPAULI_EXPERIMENTAL_METAL_OUTPUT_STORAGE";

NSString* commutation_kernel_source() {
  return @"#include <metal_stdlib>\n"
          "using namespace metal;\n"
          "struct CommutationKernelParams {\n"
          "  ulong lhs_terms;\n"
          "  ulong rhs_terms;\n"
          "  ulong words;\n"
          "};\n"
          "kernel void fp_pairwise_commutation_flat_generic(\n"
          "    device const ulong* lhs_x [[buffer(0)]],\n"
          "    device const ulong* lhs_z [[buffer(1)]],\n"
          "    device const ulong* rhs_x [[buffer(2)]],\n"
          "    device const ulong* rhs_z [[buffer(3)]],\n"
          "    device uchar* out [[buffer(4)]],\n"
          "    constant CommutationKernelParams& params [[buffer(5)]],\n"
          "    uint entry [[thread_position_in_grid]]) {\n"
          "  const ulong total = params.lhs_terms * params.rhs_terms;\n"
          "  if (static_cast<ulong>(entry) >= total) {\n"
          "    return;\n"
          "  }\n"
          "  const ulong lhs_term = static_cast<ulong>(entry) / params.rhs_terms;\n"
          "  const ulong rhs_term = static_cast<ulong>(entry) - lhs_term * params.rhs_terms;\n"
          "  const ulong lhs_offset = lhs_term * params.words;\n"
          "  const ulong rhs_offset = rhs_term * params.words;\n"
          "  uint parity = 0;\n"
          "  for (ulong word = 0; word < params.words; ++word) {\n"
          "    const ulong anti = (lhs_x[lhs_offset + word] & rhs_z[rhs_offset + word]) ^\n"
          "                       (lhs_z[lhs_offset + word] & rhs_x[rhs_offset + word]);\n"
          "    parity ^= static_cast<uint>(popcount(anti)) & 1u;\n"
          "  }\n"
          "  out[entry] = parity == 0u ? 1u : 0u;\n"
          "}\n"
          "kernel void fp_pairwise_commutation_words1(\n"
          "    device const ulong* lhs_x [[buffer(0)]],\n"
          "    device const ulong* lhs_z [[buffer(1)]],\n"
          "    device const ulong* rhs_x [[buffer(2)]],\n"
          "    device const ulong* rhs_z [[buffer(3)]],\n"
          "    device uchar* out [[buffer(4)]],\n"
          "    constant CommutationKernelParams& params [[buffer(5)]],\n"
          "    uint2 pair [[thread_position_in_grid]]) {\n"
          "  const ulong rhs_term = static_cast<ulong>(pair.x);\n"
          "  const ulong lhs_term = static_cast<ulong>(pair.y);\n"
          "  if (lhs_term >= params.lhs_terms || rhs_term >= params.rhs_terms) {\n"
          "    return;\n"
          "  }\n"
          "  const ulong anti = (lhs_x[lhs_term] & rhs_z[rhs_term]) ^\n"
          "                     (lhs_z[lhs_term] & rhs_x[rhs_term]);\n"
          "  const uint parity = static_cast<uint>(popcount(anti)) & 1u;\n"
          "  out[lhs_term * params.rhs_terms + rhs_term] = parity == 0u ? 1u : 0u;\n"
          "}\n"
          "kernel void fp_pairwise_commutation_words2(\n"
          "    device const ulong* lhs_x [[buffer(0)]],\n"
          "    device const ulong* lhs_z [[buffer(1)]],\n"
          "    device const ulong* rhs_x [[buffer(2)]],\n"
          "    device const ulong* rhs_z [[buffer(3)]],\n"
          "    device uchar* out [[buffer(4)]],\n"
          "    constant CommutationKernelParams& params [[buffer(5)]],\n"
          "    uint2 pair [[thread_position_in_grid]]) {\n"
          "  const ulong rhs_term = static_cast<ulong>(pair.x);\n"
          "  const ulong lhs_term = static_cast<ulong>(pair.y);\n"
          "  if (lhs_term >= params.lhs_terms || rhs_term >= params.rhs_terms) {\n"
          "    return;\n"
          "  }\n"
          "  const ulong lhs_offset = lhs_term * 2ul;\n"
          "  const ulong rhs_offset = rhs_term * 2ul;\n"
          "  const ulong anti0 = (lhs_x[lhs_offset] & rhs_z[rhs_offset]) ^\n"
          "                      (lhs_z[lhs_offset] & rhs_x[rhs_offset]);\n"
          "  const ulong anti1 = (lhs_x[lhs_offset + 1ul] & rhs_z[rhs_offset + 1ul]) ^\n"
          "                      (lhs_z[lhs_offset + 1ul] & rhs_x[rhs_offset + 1ul]);\n"
          "  const uint parity = (static_cast<uint>(popcount(anti0)) ^\n"
          "                       static_cast<uint>(popcount(anti1))) & 1u;\n"
          "  out[lhs_term * params.rhs_terms + rhs_term] = parity == 0u ? 1u : 0u;\n"
          "}\n"
          "kernel void fp_pairwise_commutation_generic(\n"
          "    device const ulong* lhs_x [[buffer(0)]],\n"
          "    device const ulong* lhs_z [[buffer(1)]],\n"
          "    device const ulong* rhs_x [[buffer(2)]],\n"
          "    device const ulong* rhs_z [[buffer(3)]],\n"
          "    device uchar* out [[buffer(4)]],\n"
          "    constant CommutationKernelParams& params [[buffer(5)]],\n"
          "    uint2 pair [[thread_position_in_grid]]) {\n"
          "  const ulong rhs_term = static_cast<ulong>(pair.x);\n"
          "  const ulong lhs_term = static_cast<ulong>(pair.y);\n"
          "  if (lhs_term >= params.lhs_terms || rhs_term >= params.rhs_terms) {\n"
          "    return;\n"
          "  }\n"
          "  const ulong lhs_offset = lhs_term * params.words;\n"
          "  const ulong rhs_offset = rhs_term * params.words;\n"
          "  uint parity = 0;\n"
          "  for (ulong word = 0; word < params.words; ++word) {\n"
          "    const ulong anti = (lhs_x[lhs_offset + word] & rhs_z[rhs_offset + word]) ^\n"
          "                       (lhs_z[lhs_offset + word] & rhs_x[rhs_offset + word]);\n"
          "    parity ^= static_cast<uint>(popcount(anti)) & 1u;\n"
          "  }\n"
          "  out[lhs_term * params.rhs_terms + rhs_term] = parity == 0u ? 1u : 0u;\n"
          "}\n";
}

std::string configured_metal_library_path() {
  const char* path = std::getenv(kMetalLibraryPathEnv);
  return path == nullptr ? std::string{} : std::string(path);
}

bool use_private_host_output_storage() {
  const char* value = std::getenv(kMetalOutputStorageEnv);
  if (value == nullptr || std::string(value).empty() || std::string(value) == "shared") {
    return false;
  }
  if (std::string(value) == "private") {
    return true;
  }
  throw std::invalid_argument(
      "FASTPAULI_EXPERIMENTAL_METAL_OUTPUT_STORAGE must be shared or private");
}

CommutationKernelVariant commutation_kernel_variant(std::size_t words) {
  const char* forced = std::getenv("FASTPAULI_EXPERIMENTAL_METAL_COMMUTATION_KERNEL");
  if (forced != nullptr) {
    const std::string forced_value(forced);
    if (forced_value == "words1") {
      if (words != 1) {
        throw std::invalid_argument(
            "FASTPAULI_EXPERIMENTAL_METAL_COMMUTATION_KERNEL=words1 requires exactly one packed word");
      }
      return CommutationKernelVariant::Words1;
    }
    if (forced_value == "words2") {
      if (words != 2) {
        throw std::invalid_argument(
            "FASTPAULI_EXPERIMENTAL_METAL_COMMUTATION_KERNEL=words2 requires exactly two packed words");
      }
      return CommutationKernelVariant::Words2;
    }
    if (forced_value == "flat_generic") {
      return CommutationKernelVariant::FlatGeneric;
    }
    if (forced_value == "generic_2d") {
      return CommutationKernelVariant::Generic;
    }
    throw std::invalid_argument(
        "FASTPAULI_EXPERIMENTAL_METAL_COMMUTATION_KERNEL must be one of words1, words2, generic_2d, or flat_generic");
  }
  if (words == 1) {
    return CommutationKernelVariant::Words1;
  }
  return CommutationKernelVariant::Generic;
}

NSString* commutation_kernel_name(CommutationKernelVariant variant) {
  switch (variant) {
    case CommutationKernelVariant::Words1:
      return @"fp_pairwise_commutation_words1";
    case CommutationKernelVariant::Words2:
      return @"fp_pairwise_commutation_words2";
    case CommutationKernelVariant::Generic:
      return @"fp_pairwise_commutation_generic";
    case CommutationKernelVariant::FlatGeneric:
      return @"fp_pairwise_commutation_flat_generic";
  }
  return @"fp_pairwise_commutation_generic";
}

id<MTLComputePipelineState> commutation_pipeline(
    id<MTLDevice> device,
    CommutationKernelVariant variant) {
  struct PipelineCache {
    id<MTLComputePipelineState> pipeline = nil;
    std::string library_path;
  };
  static PipelineCache words1_pipeline;
  static PipelineCache words2_pipeline;
  static PipelineCache generic_pipeline;
  static PipelineCache flat_generic_pipeline;
  PipelineCache* cache = nullptr;
  switch (variant) {
    case CommutationKernelVariant::Words1:
      cache = &words1_pipeline;
      break;
    case CommutationKernelVariant::Words2:
      cache = &words2_pipeline;
      break;
    case CommutationKernelVariant::Generic:
      cache = &generic_pipeline;
      break;
    case CommutationKernelVariant::FlatGeneric:
      cache = &flat_generic_pipeline;
      break;
  }
  const std::string library_path = configured_metal_library_path();
  if (cache->pipeline != nil && cache->library_path != library_path) {
    [cache->pipeline release];
    cache->pipeline = nil;
  }
  if (cache->pipeline == nil) {
    if (library_path.empty()) {
      cache->pipeline = metal_detail::make_compute_pipeline(
          device,
          commutation_kernel_name(variant),
          commutation_kernel_source());
    } else {
      cache->pipeline = metal_detail::make_compute_pipeline_from_metallib(
          device,
          commutation_kernel_name(variant),
          library_path);
    }
    cache->library_path = library_path;
  }
  return cache->pipeline;
}

template <typename Impl>
void validate_metal_operands(const Impl& lhs, const Impl& rhs, const char* operation) {
  validate_same_accelerator_context(
      operation,
      {AcceleratorBackend::Metal, lhs.device_ordinal},
      {AcceleratorBackend::Metal, rhs.device_ordinal});
  if (lhs.num_qubits != rhs.num_qubits) {
    throw std::invalid_argument("PauliSum commutes_with requires the same num_qubits");
  }
  if (lhs.words != rhs.words) {
    throw std::invalid_argument("PauliSum commutes_with requires matching packed word counts");
  }
}

void validate_metal_grid_size(std::size_t lhs_terms, std::size_t rhs_terms, const char* operation) {
  constexpr auto max_grid_dimension =
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
  if (lhs_terms > max_grid_dimension || rhs_terms > max_grid_dimension) {
    throw std::invalid_argument(
        std::string(operation) + " request exceeds Metal uint2 grid dimension limits");
  }
}

void validate_metal_flat_grid_size(std::size_t entries, const char* operation) {
  if (entries > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::invalid_argument(
        std::string(operation) + " request exceeds Metal uint grid limits");
  }
}

template <typename Impl>
void encode_commutation_kernel(
    const Impl& lhs,
    const Impl& rhs,
    id<MTLBuffer> output,
    std::size_t entries,
    const char* action) {
  if (entries == 0) {
    return;
  }

  id<MTLCommandBuffer> command_buffer = [lhs.command_queue commandBuffer];
  if (command_buffer == nil) {
    throw std::runtime_error("Metal failed to create a commutation command buffer");
  }
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  if (encoder == nil) {
    throw std::runtime_error("Metal failed to create a commutation command encoder");
  }

  CommutationKernelParams params{
      static_cast<std::uint64_t>(lhs.num_terms),
      static_cast<std::uint64_t>(rhs.num_terms),
      static_cast<std::uint64_t>(lhs.words),
  };
  const CommutationKernelVariant variant = commutation_kernel_variant(lhs.words);
  [encoder setComputePipelineState:commutation_pipeline(lhs.device, variant)];
  [encoder setBuffer:lhs.x offset:0 atIndex:0];
  [encoder setBuffer:lhs.z offset:0 atIndex:1];
  [encoder setBuffer:rhs.x offset:0 atIndex:2];
  [encoder setBuffer:rhs.z offset:0 atIndex:3];
  [encoder setBuffer:output offset:0 atIndex:4];
  [encoder setBytes:&params length:sizeof(params) atIndex:5];

  if (variant == CommutationKernelVariant::FlatGeneric) {
    validate_metal_flat_grid_size(entries, action);
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(entries), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(metal_detail::kThreadsPerThreadgroup, 1, 1)];
  } else {
    validate_metal_grid_size(lhs.num_terms, rhs.num_terms, action);
    [encoder dispatchThreads:MTLSizeMake(
                                 static_cast<NSUInteger>(rhs.num_terms),
                                 static_cast<NSUInteger>(lhs.num_terms),
                                 1)
        threadsPerThreadgroup:MTLSizeMake(
                                    metal_detail::kCommutationThreadgroupX,
                                    metal_detail::kCommutationThreadgroupY,
                                    1)];
  }
  [encoder endEncoding];
  metal_detail::wait_for_completion(command_buffer, action);
}

class ScopedMetalBuffer {
public:
  explicit ScopedMetalBuffer(id<MTLBuffer> buffer) : buffer_(buffer) {}
  ScopedMetalBuffer(const ScopedMetalBuffer&) = delete;
  ScopedMetalBuffer& operator=(const ScopedMetalBuffer&) = delete;
  ~ScopedMetalBuffer() { [buffer_ release]; }

private:
  id<MTLBuffer> buffer_ = nil;
};

void blit_to_shared_staging(
    id<MTLCommandQueue> command_queue,
    id<MTLBuffer> source,
    id<MTLBuffer> destination,
    std::size_t bytes,
    const char* action) {
  id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
  if (command_buffer == nil) {
    throw std::runtime_error("Metal failed to create a private-output blit command buffer");
  }
  id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
  if (encoder == nil) {
    throw std::runtime_error("Metal failed to create a private-output blit encoder");
  }
  [encoder copyFromBuffer:source
             sourceOffset:0
                 toBuffer:destination
        destinationOffset:0
                     size:bytes];
  [encoder endEncoding];
  metal_detail::wait_for_completion(command_buffer, action);
}

}  // namespace

std::vector<std::uint8_t> DevicePauliSum::commutes_with(
    const DevicePauliSum& rhs,
    std::size_t max_commutation_matrix_entries) const {
  if (!impl_ || !rhs.impl_) {
    throw std::runtime_error("DevicePauliSum is empty or moved-from");
  }
  const std::size_t entries = detail::checked_commutation_matrix_entries(
      impl_->num_terms,
      rhs.impl_->num_terms,
      max_commutation_matrix_entries);
  std::vector<std::uint8_t> output(entries);
  commutes_with_into(rhs, std::span<std::uint8_t>(output), max_commutation_matrix_entries);
  return output;
}

void DevicePauliSum::commutes_with_into(
    const DevicePauliSum& rhs,
    std::span<std::uint8_t> output,
    std::size_t max_commutation_matrix_entries) const {
  if (!impl_ || !rhs.impl_) {
    throw std::runtime_error("DevicePauliSum is empty or moved-from");
  }
  validate_metal_operands(*impl_, *rhs.impl_, "Metal commutes_with");

  const std::size_t entries = detail::checked_commutation_matrix_entries(
      impl_->num_terms,
      rhs.impl_->num_terms,
      max_commutation_matrix_entries);
  if (output.size() != entries) {
    throw std::invalid_argument("Metal commutes_with output buffer size does not match entry count");
  }
  if (entries == 0) {
    return;
  }

  if (use_private_host_output_storage()) {
    id<MTLBuffer> device_output = metal_detail::make_private_buffer(
        impl_->device,
        entries,
        "private commutation output");
    ScopedMetalBuffer output_guard(device_output);
    id<MTLBuffer> staging_output = metal_detail::make_shared_buffer(
        impl_->device,
        nullptr,
        entries,
        "private commutation output staging");
    ScopedMetalBuffer staging_guard(staging_output);
    encode_commutation_kernel(
        *impl_,
        *rhs.impl_,
        device_output,
        entries,
        "commutes_with private output");
    blit_to_shared_staging(
        impl_->command_queue,
        device_output,
        staging_output,
        entries,
        "commutes_with private output blit");
    std::memcpy(output.data(), [staging_output contents], entries);
    return;
  }

  id<MTLBuffer> device_output = metal_detail::make_shared_buffer(
      impl_->device,
      nullptr,
      entries,
      "commutation output");
  ScopedMetalBuffer output_guard(device_output);
  encode_commutation_kernel(
      *impl_,
      *rhs.impl_,
      device_output,
      entries,
      "commutes_with");
  std::memcpy(output.data(), [device_output contents], entries);
}

DeviceCommutationMatrix DevicePauliSum::commutes_with_device(
    const DevicePauliSum& rhs,
    std::size_t max_commutation_matrix_entries) const {
  if (!impl_ || !rhs.impl_) {
    throw std::runtime_error("DevicePauliSum is empty or moved-from");
  }
  validate_metal_operands(*impl_, *rhs.impl_, "Metal commutes_with_device");
  (void)detail::checked_commutation_matrix_entries(
      impl_->num_terms,
      rhs.impl_->num_terms,
      max_commutation_matrix_entries);

  DeviceCommutationMatrix output = DeviceCommutationMatrix::empty(
      impl_->num_terms,
      rhs.impl_->num_terms,
      AcceleratorBackend::Metal,
      impl_->device_ordinal);
  commutes_with_device_into(rhs, output, max_commutation_matrix_entries);
  return output;
}

void DevicePauliSum::commutes_with_device_into(
    const DevicePauliSum& rhs,
    DeviceCommutationMatrix& output,
    std::size_t max_commutation_matrix_entries) const {
  if (!impl_ || !rhs.impl_) {
    throw std::runtime_error("DevicePauliSum is empty or moved-from");
  }
  if (!output.impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }
  validate_metal_operands(*impl_, *rhs.impl_, "Metal commutes_with_device");
  validate_same_accelerator_context(
      "Metal commutes_with_device output",
      {AcceleratorBackend::Metal, impl_->device_ordinal},
      {AcceleratorBackend::Metal, output.impl_->device_ordinal});

  const std::size_t entries = detail::checked_commutation_matrix_entries(
      impl_->num_terms,
      rhs.impl_->num_terms,
      max_commutation_matrix_entries);
  if (output.impl_->rows != impl_->num_terms || output.impl_->cols != rhs.impl_->num_terms ||
      output.impl_->entries != entries) {
    throw std::invalid_argument(
        "Metal commutes_with_device output shape does not match operand term counts");
  }
  encode_commutation_kernel(
      *impl_,
      *rhs.impl_,
      output.impl_->data,
      entries,
      "commutes_with_device");
}

}  // namespace wolfgang

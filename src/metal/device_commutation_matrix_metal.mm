#include "device_commutation_matrix_metal.hpp"

#include "device_pauli_sum_metal.hpp"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace wolfgang {

namespace {

struct CountKernelParams {
  std::uint64_t entries = 0;
  std::uint64_t rows = 0;
  std::uint64_t cols = 0;
};

constexpr const char* kMetalCompactConsumerEnv = "WOLFGANG_EXPERIMENTAL_METAL_COMPACT_CONSUMER";
constexpr const char* kMetalLibraryPathEnv = "WOLFGANG_EXPERIMENTAL_METAL_LIBRARY_PATH";

enum class CompactConsumerMode {
  Cpu,
  GpuAtomic,
  GpuParallelTotal,
};

NSString* compact_consumer_kernel_source() {
  return @"#include <metal_stdlib>\n"
          "using namespace metal;\n"
          "struct CountKernelParams {\n"
          "  ulong entries;\n"
          "  ulong rows;\n"
          "  ulong cols;\n"
          "};\n"
          "kernel void fp_count_commuting_total_atomic(\n"
          "    device const uchar* data [[buffer(0)]],\n"
          "    device atomic_uint* total [[buffer(1)]],\n"
          "    constant CountKernelParams& params [[buffer(2)]],\n"
          "    uint count_index [[thread_position_in_grid]]) {\n"
          "  if (static_cast<ulong>(count_index) >= params.entries) {\n"
          "    return;\n"
          "  }\n"
          "  if (data[count_index] != 0u) {\n"
          "    atomic_fetch_add_explicit(total, 1u, memory_order_relaxed);\n"
          "  }\n"
          "}\n"
          "kernel void fp_count_commuting_total_block_sums(\n"
          "    device const uchar* data [[buffer(0)]],\n"
          "    device uint* partials [[buffer(1)]],\n"
          "    constant CountKernelParams& params [[buffer(2)]],\n"
          "    uint thread_index [[thread_index_in_threadgroup]],\n"
          "    uint block_index [[threadgroup_position_in_grid]]) {\n"
          "  threadgroup uint scratch[256];\n"
          "  const ulong entry = static_cast<ulong>(block_index) * 256ul +\n"
          "                      static_cast<ulong>(thread_index);\n"
          "  scratch[thread_index] = entry < params.entries && data[entry] != 0u ? 1u : 0u;\n"
          "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
          "  for (uint stride = 128u; stride > 0u; stride >>= 1u) {\n"
          "    if (thread_index < stride) {\n"
          "      scratch[thread_index] += scratch[thread_index + stride];\n"
          "    }\n"
          "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
          "  }\n"
          "  if (thread_index == 0u) {\n"
          "    partials[block_index] = scratch[0];\n"
          "  }\n"
          "}\n"
          "kernel void fp_count_commuting_rows(\n"
          "    device const uchar* data [[buffer(0)]],\n"
          "    device ulong* counts [[buffer(1)]],\n"
          "    constant CountKernelParams& params [[buffer(2)]],\n"
          "    uint row [[thread_position_in_grid]]) {\n"
          "  if (static_cast<ulong>(row) >= params.rows) {\n"
          "    return;\n"
          "  }\n"
          "  const ulong offset = static_cast<ulong>(row) * params.cols;\n"
          "  ulong count = 0;\n"
          "  for (ulong col = 0; col < params.cols; ++col) {\n"
          "    count += data[offset + col] != 0u ? 1ul : 0ul;\n"
          "  }\n"
          "  counts[row] = count;\n"
          "}\n"
          "kernel void fp_count_commuting_cols(\n"
          "    device const uchar* data [[buffer(0)]],\n"
          "    device ulong* counts [[buffer(1)]],\n"
          "    constant CountKernelParams& params [[buffer(2)]],\n"
          "    uint col [[thread_position_in_grid]]) {\n"
          "  if (static_cast<ulong>(col) >= params.cols) {\n"
          "    return;\n"
          "  }\n"
          "  ulong count = 0;\n"
          "  for (ulong row = 0; row < params.rows; ++row) {\n"
          "    count += data[row * params.cols + static_cast<ulong>(col)] != 0u ? 1ul : 0ul;\n"
          "  }\n"
          "  counts[col] = count;\n"
          "}\n";
}

void validate_count_result_fits(std::size_t entries) {
  if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
    if (entries > static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())) {
      throw std::overflow_error("Metal device commutation matrix count exceeds uint64 range");
    }
  }
}

void validate_gpu_count_grid_size(std::size_t count, const char* operation) {
  if (count > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::invalid_argument(
        std::string(operation) + " request exceeds Metal uint grid limits");
  }
}

std::string configured_metal_library_path() {
  const char* path = std::getenv(kMetalLibraryPathEnv);
  return path == nullptr ? std::string{} : std::string(path);
}

CompactConsumerMode compact_consumer_mode() {
  const char* value = std::getenv(kMetalCompactConsumerEnv);
  if (value == nullptr || std::string(value).empty() || std::string(value) == "cpu") {
    return CompactConsumerMode::Cpu;
  }
  if (std::string(value) == "gpu") {
    return CompactConsumerMode::GpuAtomic;
  }
  if (std::string(value) == "gpu_parallel_total") {
    return CompactConsumerMode::GpuParallelTotal;
  }
  throw std::invalid_argument(
      "WOLFGANG_EXPERIMENTAL_METAL_COMPACT_CONSUMER must be cpu, gpu, or gpu_parallel_total");
}

id<MTLComputePipelineState> compact_consumer_pipeline(id<MTLDevice> device, NSString* kernel_name) {
  struct PipelineCache {
    id<MTLComputePipelineState> pipeline = nil;
    std::string library_path;
  };
  static PipelineCache total_pipeline;
  static PipelineCache total_blocks_pipeline;
  static PipelineCache rows_pipeline;
  static PipelineCache cols_pipeline;

  PipelineCache* cache = nullptr;
  if ([kernel_name isEqualToString:@"fp_count_commuting_total_atomic"]) {
    cache = &total_pipeline;
  } else if ([kernel_name isEqualToString:@"fp_count_commuting_total_block_sums"]) {
    cache = &total_blocks_pipeline;
  } else if ([kernel_name isEqualToString:@"fp_count_commuting_rows"]) {
    cache = &rows_pipeline;
  } else {
    cache = &cols_pipeline;
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
          kernel_name,
          compact_consumer_kernel_source());
    } else {
      cache->pipeline = metal_detail::make_compute_pipeline_from_metallib(
          device,
          kernel_name,
          library_path);
    }
    cache->library_path = library_path;
  }
  return cache->pipeline;
}

void encode_count_kernel(
    const auto& impl,
    id<MTLComputePipelineState> pipeline,
    id<MTLBuffer> output,
    CountKernelParams params,
    std::size_t threads,
    const char* action) {
  if (threads == 0) {
    return;
  }
  validate_gpu_count_grid_size(threads, action);
  id<MTLCommandBuffer> command_buffer = [impl.command_queue commandBuffer];
  if (command_buffer == nil) {
    throw std::runtime_error("Metal failed to create a compact-consumer command buffer");
  }
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  if (encoder == nil) {
    throw std::runtime_error("Metal failed to create a compact-consumer command encoder");
  }
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:impl.data offset:0 atIndex:0];
  [encoder setBuffer:output offset:0 atIndex:1];
  [encoder setBytes:&params length:sizeof(params) atIndex:2];
  [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(threads), 1, 1)
      threadsPerThreadgroup:MTLSizeMake(metal_detail::kThreadsPerThreadgroup, 1, 1)];
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

std::uint64_t count_commuting_total_on_gpu(const auto& impl) {
  validate_gpu_count_grid_size(impl.entries, "count_commuting GPU reduction");
  if (impl.entries > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::invalid_argument("Metal GPU count_commuting reduction is limited to uint32 totals");
  }
  std::uint32_t zero = 0;
  id<MTLBuffer> result = metal_detail::make_shared_buffer(
      impl.device,
      &zero,
      sizeof(zero),
      "count_commuting GPU result");
  ScopedMetalBuffer result_guard(result);
  id<MTLComputePipelineState> pipeline =
      compact_consumer_pipeline(impl.device, @"fp_count_commuting_total_atomic");
  CountKernelParams params{
      static_cast<std::uint64_t>(impl.entries),
      static_cast<std::uint64_t>(impl.rows),
      static_cast<std::uint64_t>(impl.cols),
  };
  encode_count_kernel(
      impl,
      pipeline,
      result,
      params,
      impl.entries,
      "count_commuting GPU reduction");
  std::uint32_t host = 0;
  std::memcpy(&host, [result contents], sizeof(host));
  return static_cast<std::uint64_t>(host);
}

std::uint64_t count_commuting_total_blocks_on_gpu(const auto& impl) {
  if (impl.entries == 0) {
    return 0;
  }
  validate_gpu_count_grid_size(impl.entries, "count_commuting GPU parallel reduction");
  if (impl.entries > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::invalid_argument(
        "Metal GPU parallel count_commuting reduction is limited to uint32 totals");
  }
  const std::size_t threads_per_group =
      static_cast<std::size_t>(metal_detail::kThreadsPerThreadgroup);
  const std::size_t block_count = (impl.entries + threads_per_group - 1) / threads_per_group;
  const std::size_t dispatch_threads =
      detail::checked_product(block_count, threads_per_group, "Metal count block dispatch threads");
  validate_gpu_count_grid_size(dispatch_threads, "count_commuting GPU parallel reduction");
  const std::size_t bytes =
      metal_detail::checked_bytes<std::uint32_t>(block_count, "count_commuting GPU partial sums");
  id<MTLBuffer> partials = metal_detail::make_shared_buffer(
      impl.device,
      nullptr,
      bytes,
      "count_commuting GPU partial sums");
  ScopedMetalBuffer partials_guard(partials);
  id<MTLComputePipelineState> pipeline =
      compact_consumer_pipeline(impl.device, @"fp_count_commuting_total_block_sums");
  CountKernelParams params{
      static_cast<std::uint64_t>(impl.entries),
      static_cast<std::uint64_t>(impl.rows),
      static_cast<std::uint64_t>(impl.cols),
  };
  encode_count_kernel(
      impl,
      pipeline,
      partials,
      params,
      dispatch_threads,
      "count_commuting GPU parallel reduction");
  std::vector<std::uint32_t> host(block_count, 0);
  std::memcpy(host.data(), [partials contents], bytes);
  std::uint64_t total = 0;
  for (const std::uint32_t partial : host) {
    total += static_cast<std::uint64_t>(partial);
  }
  return total;
}

std::vector<std::uint64_t> count_commuting_axis_on_gpu(
    const auto& impl,
    bool rows) {
  const std::size_t count = rows ? impl.rows : impl.cols;
  std::vector<std::uint64_t> host(count, 0);
  if (count == 0 || impl.entries == 0) {
    return host;
  }
  validate_gpu_count_grid_size(count, rows ? "count_commuting_rows GPU reduction"
                                          : "count_commuting_cols GPU reduction");
  const std::size_t bytes = metal_detail::checked_bytes<std::uint64_t>(
      count,
      rows ? "Metal count_commuting_rows result bytes"
           : "Metal count_commuting_cols result bytes");
  id<MTLBuffer> result = metal_detail::make_shared_buffer(
      impl.device,
      nullptr,
      bytes,
      rows ? "count_commuting_rows GPU result" : "count_commuting_cols GPU result");
  ScopedMetalBuffer result_guard(result);
  id<MTLComputePipelineState> pipeline = compact_consumer_pipeline(
      impl.device,
      rows ? @"fp_count_commuting_rows" : @"fp_count_commuting_cols");
  CountKernelParams params{
      static_cast<std::uint64_t>(impl.entries),
      static_cast<std::uint64_t>(impl.rows),
      static_cast<std::uint64_t>(impl.cols),
  };
  encode_count_kernel(
      impl,
      pipeline,
      result,
      params,
      count,
      rows ? "count_commuting_rows GPU reduction" : "count_commuting_cols GPU reduction");
  std::memcpy(host.data(), [result contents], bytes);
  return host;
}

[[noreturn]] void throw_metal_interop_unavailable() {
  throw std::runtime_error(
      "Metal DeviceCommutationMatrix does not expose raw Metal buffers, CUDA Array Interface "
      "pointers, or DLPack capsules in this source-build lane.");
}

}  // namespace

DeviceCommutationMatrix::Impl::~Impl() {
  [data release];
  [command_queue release];
  [device release];
}

DeviceCommutationMatrix::DeviceCommutationMatrix() noexcept = default;

DeviceCommutationMatrix::DeviceCommutationMatrix(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

DeviceCommutationMatrix::~DeviceCommutationMatrix() = default;

DeviceCommutationMatrix::DeviceCommutationMatrix(DeviceCommutationMatrix&& other) noexcept =
    default;

DeviceCommutationMatrix& DeviceCommutationMatrix::operator=(
    DeviceCommutationMatrix&& other) noexcept = default;

DeviceCommutationMatrix DeviceCommutationMatrix::empty(
    std::size_t rows,
    std::size_t cols,
    int device) {
  return empty(rows, cols, AcceleratorBackend::None, device);
}

DeviceCommutationMatrix DeviceCommutationMatrix::empty(
    std::size_t rows,
    std::size_t cols,
    AcceleratorBackend backend,
    int device) {
  @autoreleasepool {
    const MetalStatus status = DevicePauliSum::metal_status();
    const AcceleratorBackend selected = select_accelerator_backend(
        backend,
        false,
        false,
        false,
        false,
        true,
        status.runtime_available);
    if (selected != AcceleratorBackend::Metal) {
      throw std::runtime_error(accelerator_not_built_message(selected));
    }
    metal_detail::validate_device_ordinal(device);

    auto impl = std::make_unique<Impl>();
    impl->rows = rows;
    impl->cols = cols;
    impl->entries = detail::checked_product(rows, cols, "Metal device commutation matrix entries");
    impl->device_ordinal = device;
    impl->device = metal_detail::create_default_device_or_throw();
    impl->command_queue = metal_detail::make_command_queue(impl->device);

    const std::size_t bytes =
        metal_detail::checked_bytes<std::uint8_t>(
            impl->entries,
            "Metal device commutation matrix bytes");
    impl->data = metal_detail::make_shared_buffer(
        impl->device,
        nullptr,
        bytes,
        "device commutation matrix");
    return DeviceCommutationMatrix(std::move(impl));
  }
}

std::vector<std::uint8_t> DeviceCommutationMatrix::to_host() const {
  if (!impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }

  std::vector<std::uint8_t> host(impl_->entries);
  if (impl_->entries == 0) {
    return host;
  }
  std::memcpy(host.data(), [impl_->data contents], host.size());
  return host;
}

std::uint64_t DeviceCommutationMatrix::count_commuting() const {
  if (!impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }
  validate_count_result_fits(impl_->entries);
  const CompactConsumerMode mode = compact_consumer_mode();
  if (mode == CompactConsumerMode::GpuAtomic) {
    return count_commuting_total_on_gpu(*impl_);
  }
  if (mode == CompactConsumerMode::GpuParallelTotal) {
    return count_commuting_total_blocks_on_gpu(*impl_);
  }
  const std::uint8_t* data = static_cast<const std::uint8_t*>([impl_->data contents]);
  std::uint64_t total = 0;
  for (std::size_t index = 0; index < impl_->entries; ++index) {
    total += data[index] != 0U ? 1U : 0U;
  }
  return total;
}

std::vector<std::uint64_t> DeviceCommutationMatrix::count_commuting_rows() const {
  if (!impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }
  validate_count_result_fits(impl_->entries);
  if (compact_consumer_mode() == CompactConsumerMode::GpuAtomic) {
    return count_commuting_axis_on_gpu(*impl_, true);
  }
  std::vector<std::uint64_t> counts(impl_->rows, 0);
  if (impl_->rows == 0 || impl_->cols == 0) {
    return counts;
  }

  const std::uint8_t* data = static_cast<const std::uint8_t*>([impl_->data contents]);
  for (std::size_t row = 0; row < impl_->rows; ++row) {
    const std::size_t row_offset = row * impl_->cols;
    std::uint64_t count = 0;
    for (std::size_t col = 0; col < impl_->cols; ++col) {
      count += data[row_offset + col] != 0U ? 1U : 0U;
    }
    counts[row] = count;
  }
  return counts;
}

std::vector<std::uint64_t> DeviceCommutationMatrix::count_commuting_cols() const {
  if (!impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }
  validate_count_result_fits(impl_->entries);
  if (compact_consumer_mode() == CompactConsumerMode::GpuAtomic) {
    return count_commuting_axis_on_gpu(*impl_, false);
  }
  std::vector<std::uint64_t> counts(impl_->cols, 0);
  if (impl_->rows == 0 || impl_->cols == 0) {
    return counts;
  }

  const std::uint8_t* data = static_cast<const std::uint8_t*>([impl_->data contents]);
  for (std::size_t row = 0; row < impl_->rows; ++row) {
    const std::size_t row_offset = row * impl_->cols;
    for (std::size_t col = 0; col < impl_->cols; ++col) {
      counts[col] += data[row_offset + col] != 0U ? 1U : 0U;
    }
  }
  return counts;
}

std::size_t DeviceCommutationMatrix::rows() const noexcept {
  return impl_ ? impl_->rows : 0;
}

std::size_t DeviceCommutationMatrix::cols() const noexcept {
  return impl_ ? impl_->cols : 0;
}

std::size_t DeviceCommutationMatrix::num_entries() const noexcept {
  return impl_ ? impl_->entries : 0;
}

int DeviceCommutationMatrix::device() const noexcept {
  return impl_ ? impl_->device_ordinal : -1;
}

std::string DeviceCommutationMatrix::backend() const {
  return std::string(accelerator_backend_name(
      impl_ ? AcceleratorBackend::Metal : AcceleratorBackend::None));
}

std::uintptr_t DeviceCommutationMatrix::data_pointer_for_cuda_array_interface() const {
  throw_metal_interop_unavailable();
}

std::uintptr_t DeviceCommutationMatrix::data_pointer_for_dlpack() const {
  throw_metal_interop_unavailable();
}

int DeviceCommutationMatrix::dlpack_device_type() const {
  throw_metal_interop_unavailable();
}

std::uint8_t* DeviceCommutationMatrix::mutable_data_for_device_write() {
  throw_metal_interop_unavailable();
}

void copy_device_commutation_matrix_from_host_for_testing(
    DeviceCommutationMatrix& matrix,
    std::span<const std::uint8_t> values) {
  if (!matrix.impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }
  if (values.size() != matrix.impl_->entries) {
    throw std::invalid_argument(
        "DeviceCommutationMatrix testing copy size does not match matrix entries");
  }
  if (values.empty()) {
    return;
  }
  std::memcpy([matrix.impl_->data contents], values.data(), values.size());
}

}  // namespace wolfgang

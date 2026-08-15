#include "device_commutation_matrix.hip.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace wolfgang {

using hip_detail::check_hip;
using hip_detail::copy_to_host;
using hip_detail::hip_allocate;
using hip_detail::kHipThreadsPerBlock;
using hip_detail::ScopedHipDevice;

namespace {

void validate_device_ordinal_for_matrix(int device) {
  int device_count = 0;
  const hipError_t count_result = hipGetDeviceCount(&device_count);
  if (count_result != hipSuccess) {
    throw std::runtime_error(
        std::string("HIP runtime library is unavailable or failed during device discovery: ") +
        hipGetErrorString(count_result));
  }
  if (device_count == 0) {
    throw std::runtime_error("no HIP device is available");
  }
  if (device < 0 || device >= device_count) {
    throw std::invalid_argument("HIP device ordinal is out of range");
  }
}

int checked_reduction_blocks(std::size_t count, const char* operation) {
  if (count == 0) {
    return 0;
  }
  if (count > hip_detail::kMaxHipLaunchBlocks) {
    throw std::invalid_argument(
        std::string(operation) + " request exceeds HIP launch grid limits");
  }
  return static_cast<int>(count);
}

void validate_count_result_fits(std::size_t entries) {
  if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
    if (entries > static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())) {
      throw std::overflow_error("HIP device commutation matrix count exceeds uint64 range");
    }
  }
}

__device__ std::uint64_t block_reduce_sum(std::uint64_t value) {
  __shared__ std::uint64_t scratch[kHipThreadsPerBlock];
  const int thread = threadIdx.x;
  scratch[thread] = value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (thread < stride) {
      scratch[thread] += scratch[thread + stride];
    }
    __syncthreads();
  }
  return scratch[0];
}

__global__ void count_total_kernel(
    const std::uint8_t* data,
    std::size_t entries,
    std::uint64_t* partials) {
  const std::size_t global_thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + static_cast<std::size_t>(threadIdx.x);
  std::uint64_t count = 0;
  if (global_thread < entries) {
    count = data[global_thread] != 0U ? 1U : 0U;
  }
  const std::uint64_t block_count = block_reduce_sum(count);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = block_count;
  }
}

__global__ void count_rows_kernel(
    const std::uint8_t* data,
    std::size_t rows,
    std::size_t cols,
    std::uint64_t* output) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  std::uint64_t count = 0;
  const std::size_t row_offset = row * cols;
  for (std::size_t col = static_cast<std::size_t>(threadIdx.x); col < cols;
       col += static_cast<std::size_t>(blockDim.x)) {
    count += data[row_offset + col] != 0U ? 1U : 0U;
  }
  const std::uint64_t block_count = block_reduce_sum(count);
  if (threadIdx.x == 0) {
    output[row] = block_count;
  }
}

__global__ void count_cols_kernel(
    const std::uint8_t* data,
    std::size_t rows,
    std::size_t cols,
    std::uint64_t* output) {
  const std::size_t col = static_cast<std::size_t>(blockIdx.x);
  if (col >= cols) {
    return;
  }

  std::uint64_t count = 0;
  for (std::size_t row = static_cast<std::size_t>(threadIdx.x); row < rows;
       row += static_cast<std::size_t>(blockDim.x)) {
    count += data[row * cols + col] != 0U ? 1U : 0U;
  }
  const std::uint64_t block_count = block_reduce_sum(count);
  if (threadIdx.x == 0) {
    output[col] = block_count;
  }
}

template <typename T>
std::unique_ptr<T, void (*)(T*)> device_allocation_guard(T* ptr) {
  return std::unique_ptr<T, void (*)(T*)>(
      ptr,
      [](T* value) {
        if (value != nullptr) {
          (void)hipFree(value);
        }
      });
}

[[noreturn]] void throw_hip_interop_unavailable() {
  throw std::runtime_error(
      "HIP DeviceCommutationMatrix does not expose CUDA Array Interface pointers; "
      "ROCm/HIP DLPack interop is unavailable because the current validated "
      "ROCm consumer did not enforce FastPauli's read-only export contract.");
}

}  // namespace

DeviceCommutationMatrix::Impl::~Impl() {
  if (data == nullptr) {
    return;
  }
  int previous_device = -1;
  const hipError_t current_result = hipGetDevice(&previous_device);
  (void)hipSetDevice(device_ordinal);
  (void)hipFree(data);
  if (current_result == hipSuccess && previous_device >= 0 && previous_device != device_ordinal) {
    (void)hipSetDevice(previous_device);
  }
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
  const HipStatus status = DevicePauliSum::hip_status();
  const AcceleratorBackend selected = select_accelerator_backend(
      backend,
      false,
      false,
      true,
      status.runtime_available,
      false,
      false);
  if (selected != AcceleratorBackend::Hip) {
    throw std::runtime_error(accelerator_not_built_message(selected));
  }
  validate_device_ordinal_for_matrix(device);

  auto impl = std::make_unique<Impl>();
  impl->rows = rows;
  impl->cols = cols;
  impl->entries = detail::checked_product(rows, cols, "HIP device commutation matrix entries");
  impl->device_ordinal = device;

  ScopedHipDevice guard(device);
  hip_allocate(impl->data, impl->entries, "HIP device commutation matrix");
  return DeviceCommutationMatrix(std::move(impl));
}

std::vector<std::uint8_t> DeviceCommutationMatrix::to_host() const {
  if (!impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }

  std::vector<std::uint8_t> host(impl_->entries);
  ScopedHipDevice guard(impl_->device_ordinal);
  copy_to_host(host.data(), impl_->data, impl_->entries, "HIP device commutation matrix");
  return host;
}

std::uint64_t DeviceCommutationMatrix::count_commuting() const {
  if (!impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }
  validate_count_result_fits(impl_->entries);
  if (impl_->entries == 0) {
    return 0;
  }

  const int blocks = hip_detail::checked_launch_blocks(
      impl_->entries,
      "HIP device commutation matrix total count");
  ScopedHipDevice guard(impl_->device_ordinal);
  std::uint64_t* device_partials = nullptr;
  hip_allocate(
      device_partials,
      static_cast<std::size_t>(blocks),
      "HIP device commutation count partials");
  auto partials_guard = device_allocation_guard(device_partials);

  count_total_kernel<<<blocks, kHipThreadsPerBlock>>>(
      impl_->data,
      impl_->entries,
      device_partials);
  check_hip(hipGetLastError(), "launch device commutation total count");

  std::vector<std::uint64_t> partials(static_cast<std::size_t>(blocks));
  copy_to_host(
      partials.data(),
      device_partials,
      partials.size(),
      "HIP device commutation count partials");

  std::uint64_t total = 0;
  for (std::uint64_t value : partials) {
    total += value;
  }
  return total;
}

std::vector<std::uint64_t> DeviceCommutationMatrix::count_commuting_rows() const {
  if (!impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }
  validate_count_result_fits(impl_->entries);

  std::vector<std::uint64_t> host(impl_->rows, 0);
  if (impl_->rows == 0 || impl_->cols == 0) {
    return host;
  }

  const int blocks = checked_reduction_blocks(
      impl_->rows,
      "HIP device commutation matrix row counts");
  ScopedHipDevice guard(impl_->device_ordinal);
  std::uint64_t* device_counts = nullptr;
  hip_allocate(device_counts, impl_->rows, "HIP device commutation row counts");
  auto counts_guard = device_allocation_guard(device_counts);

  count_rows_kernel<<<blocks, kHipThreadsPerBlock>>>(
      impl_->data,
      impl_->rows,
      impl_->cols,
      device_counts);
  check_hip(hipGetLastError(), "launch device commutation row counts");
  copy_to_host(host.data(), device_counts, host.size(), "HIP device commutation row counts");
  return host;
}

std::vector<std::uint64_t> DeviceCommutationMatrix::count_commuting_cols() const {
  if (!impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }
  validate_count_result_fits(impl_->entries);

  std::vector<std::uint64_t> host(impl_->cols, 0);
  if (impl_->rows == 0 || impl_->cols == 0) {
    return host;
  }

  const int blocks = checked_reduction_blocks(
      impl_->cols,
      "HIP device commutation matrix column counts");
  ScopedHipDevice guard(impl_->device_ordinal);
  std::uint64_t* device_counts = nullptr;
  hip_allocate(device_counts, impl_->cols, "HIP device commutation column counts");
  auto counts_guard = device_allocation_guard(device_counts);

  count_cols_kernel<<<blocks, kHipThreadsPerBlock>>>(
      impl_->data,
      impl_->rows,
      impl_->cols,
      device_counts);
  check_hip(hipGetLastError(), "launch device commutation column counts");
  copy_to_host(host.data(), device_counts, host.size(), "HIP device commutation column counts");
  return host;
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
      impl_ ? AcceleratorBackend::Hip : AcceleratorBackend::None));
}

std::uintptr_t DeviceCommutationMatrix::data_pointer_for_cuda_array_interface() const {
  if (!impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }
  throw_hip_interop_unavailable();
}

std::uintptr_t DeviceCommutationMatrix::data_pointer_for_dlpack() const {
  if (!impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }
  throw_hip_interop_unavailable();
}

int DeviceCommutationMatrix::dlpack_device_type() const {
  if (!impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }
  throw_hip_interop_unavailable();
}

std::uint8_t* DeviceCommutationMatrix::mutable_data_for_device_write() {
  if (!impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }
  return impl_->data;
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
  ScopedHipDevice guard(matrix.impl_->device_ordinal);
  hip_detail::copy_to_device(
      matrix.impl_->data,
      values.data(),
      values.size(),
      "DeviceCommutationMatrix testing copy");
}

}  // namespace wolfgang

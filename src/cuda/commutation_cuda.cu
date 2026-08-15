#include "device_pauli_sum.cuh"
#include "device_commutation_matrix.cuh"

#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace wolfgang {

using cuda_detail::check_cuda;
using cuda_detail::checked_launch_blocks;
using cuda_detail::copy_to_host;
using cuda_detail::cuda_allocate;
using cuda_detail::kCudaThreadsPerBlock;
using cuda_detail::ScopedCudaDevice;

namespace {

constexpr std::size_t kRegisteredHostOutputCopyThreshold = 32U * 1024U * 1024U;

class RegisteredHostBuffer {
public:
  RegisteredHostBuffer(std::uint8_t* ptr, std::size_t bytes) : ptr_(ptr) {
    const cudaError_t result = cudaHostRegister(ptr_, bytes, cudaHostRegisterDefault);
    if (result == cudaSuccess) {
      registered_ = true;
      return;
    }
    (void)cudaGetLastError();
  }

  RegisteredHostBuffer(const RegisteredHostBuffer&) = delete;
  RegisteredHostBuffer& operator=(const RegisteredHostBuffer&) = delete;

  ~RegisteredHostBuffer() {
    if (registered_) {
      (void)cudaHostUnregister(ptr_);
    }
  }

  [[nodiscard]] bool registered() const noexcept { return registered_; }

  void unregister_checked() {
    if (!registered_) {
      return;
    }
    check_cuda(cudaHostUnregister(ptr_), "unregister commutation output");
    registered_ = false;
  }

private:
  std::uint8_t* ptr_ = nullptr;
  bool registered_ = false;
};

class BenchmarkDeviceOutputWorkspace {
public:
  BenchmarkDeviceOutputWorkspace() = default;

  BenchmarkDeviceOutputWorkspace(const BenchmarkDeviceOutputWorkspace&) = delete;
  BenchmarkDeviceOutputWorkspace& operator=(const BenchmarkDeviceOutputWorkspace&) = delete;

  ~BenchmarkDeviceOutputWorkspace() { release(); }

  std::uint8_t* reserve(int device_ordinal, std::size_t bytes) {
    if (device_ordinal_ != -1 && device_ordinal_ != device_ordinal) {
      release();
    }
    if (bytes > capacity_) {
      release();
      device_ordinal_ = device_ordinal;
      cuda_allocate(ptr_, bytes, "benchmark commutation reusable device output");
      capacity_ = bytes;
    } else if (device_ordinal_ == -1) {
      device_ordinal_ = device_ordinal;
    }
    return ptr_;
  }

private:
  void release() noexcept {
    if (ptr_ == nullptr) {
      capacity_ = 0;
      device_ordinal_ = -1;
      return;
    }
    int previous_device = -1;
    const cudaError_t current_result = cudaGetDevice(&previous_device);
    (void)cudaSetDevice(device_ordinal_);
    (void)cudaFree(ptr_);
    if (current_result == cudaSuccess && previous_device >= 0 &&
        previous_device != device_ordinal_) {
      (void)cudaSetDevice(previous_device);
    }
    ptr_ = nullptr;
    capacity_ = 0;
    device_ordinal_ = -1;
  }

  std::uint8_t* ptr_ = nullptr;
  std::size_t capacity_ = 0;
  int device_ordinal_ = -1;
};

bool benchmark_reusable_device_output_enabled() {
  const char* value = std::getenv("FASTPAULI_CUDA_BENCH_REUSE_COMMUTATION_DEVICE_OUTPUT");
  if (value == nullptr) {
    return false;
  }
  const std::string setting(value);
  return setting == "1" || setting == "true" || setting == "TRUE" || setting == "on";
}

BenchmarkDeviceOutputWorkspace& benchmark_device_output_workspace() {
  thread_local BenchmarkDeviceOutputWorkspace workspace;
  return workspace;
}

__global__ void commutation_kernel(
    const std::uint64_t* lhs_x,
    const std::uint64_t* lhs_z,
    const std::uint64_t* rhs_x,
    const std::uint64_t* rhs_z,
    std::size_t lhs_terms,
    std::size_t rhs_terms,
    std::size_t words,
    std::uint8_t* out) {
  const std::size_t entry =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + static_cast<std::size_t>(threadIdx.x);
  const std::size_t total = lhs_terms * rhs_terms;
  if (entry >= total) {
    return;
  }

  const std::size_t lhs_term = entry / rhs_terms;
  const std::size_t rhs_term = entry - lhs_term * rhs_terms;
  const std::size_t lhs_offset = lhs_term * words;
  const std::size_t rhs_offset = rhs_term * words;
  unsigned int parity = 0;
  for (std::size_t word = 0; word < words; ++word) {
    const std::uint64_t anti_commuting_bits =
        (lhs_x[lhs_offset + word] & rhs_z[rhs_offset + word]) ^
        (lhs_z[lhs_offset + word] & rhs_x[rhs_offset + word]);
    parity ^= static_cast<unsigned int>(__popcll(anti_commuting_bits)) & 1U;
  }
  out[entry] = parity == 0 ? 1U : 0U;
}

void copy_commutation_output_to_host(std::uint8_t* dst, const std::uint8_t* src, std::size_t count) {
  if (count < kRegisteredHostOutputCopyThreshold) {
    copy_to_host(dst, src, count, "commutation output");
    return;
  }

  RegisteredHostBuffer registered_output(dst, count);
  if (!registered_output.registered()) {
    copy_to_host(dst, src, count, "commutation output");
    return;
  }
  copy_to_host(dst, src, count, "registered commutation output");
  registered_output.unregister_checked();
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
  std::vector<std::uint8_t> host_output(entries);
  commutes_with_into(rhs, std::span<std::uint8_t>(host_output), max_commutation_matrix_entries);
  return host_output;
}

void DevicePauliSum::commutes_with_into(
    const DevicePauliSum& rhs,
    std::span<std::uint8_t> output,
    std::size_t max_commutation_matrix_entries) const {
  if (!impl_ || !rhs.impl_) {
    throw std::runtime_error("DevicePauliSum is empty or moved-from");
  }
  validate_same_accelerator_context(
      "CUDA commutes_with",
      {AcceleratorBackend::Cuda, impl_->device_ordinal},
      {AcceleratorBackend::Cuda, rhs.impl_->device_ordinal});
  if (impl_->num_qubits != rhs.impl_->num_qubits) {
    throw std::invalid_argument("PauliSum commutes_with requires the same num_qubits");
  }

  const std::size_t entries = detail::checked_commutation_matrix_entries(
      impl_->num_terms,
      rhs.impl_->num_terms,
      max_commutation_matrix_entries);
  if (output.size() != entries) {
    throw std::invalid_argument("CUDA commutes_with output buffer size does not match entry count");
  }
  const int blocks = entries == 0 ? 0 : checked_launch_blocks(entries, "CUDA commutes_with");
  if (entries == 0) {
    return;
  }

  ScopedCudaDevice guard(impl_->device_ordinal);
  std::uint8_t* device_output = nullptr;
  std::unique_ptr<std::uint8_t, void (*)(std::uint8_t*)> output_guard(
      nullptr,
      [](std::uint8_t* ptr) {
        if (ptr != nullptr) {
          (void)cudaFree(ptr);
        }
      });
  if (benchmark_reusable_device_output_enabled()) {
    // Private benchmark-only path: reuse device output storage to quantify
    // allocation overhead without changing the public dense host-output API.
    device_output = benchmark_device_output_workspace().reserve(impl_->device_ordinal, entries);
  } else {
    cuda_allocate(device_output, entries, "commutation output");
    output_guard.reset(device_output);
  }
  commutation_kernel<<<blocks, kCudaThreadsPerBlock>>>(
      impl_->x,
      impl_->z,
      rhs.impl_->x,
      rhs.impl_->z,
      impl_->num_terms,
      rhs.impl_->num_terms,
      impl_->words,
      device_output);
  check_cuda(cudaGetLastError(), "launch CUDA commutation");
  copy_commutation_output_to_host(output.data(), device_output, entries);
}

DeviceCommutationMatrix DevicePauliSum::commutes_with_device(
    const DevicePauliSum& rhs,
    std::size_t max_commutation_matrix_entries) const {
  if (!impl_ || !rhs.impl_) {
    throw std::runtime_error("DevicePauliSum is empty or moved-from");
  }
  validate_same_accelerator_context(
      "CUDA commutes_with_device",
      {AcceleratorBackend::Cuda, impl_->device_ordinal},
      {AcceleratorBackend::Cuda, rhs.impl_->device_ordinal});
  if (impl_->num_qubits != rhs.impl_->num_qubits) {
    throw std::invalid_argument("PauliSum commutes_with requires the same num_qubits");
  }

  (void)detail::checked_commutation_matrix_entries(
      impl_->num_terms,
      rhs.impl_->num_terms,
      max_commutation_matrix_entries);
  DeviceCommutationMatrix output = DeviceCommutationMatrix::empty(
      impl_->num_terms,
      rhs.impl_->num_terms,
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
  validate_same_accelerator_context(
      "CUDA commutes_with_device",
      {AcceleratorBackend::Cuda, impl_->device_ordinal},
      {AcceleratorBackend::Cuda, rhs.impl_->device_ordinal});
  validate_same_accelerator_context(
      "CUDA commutes_with_device output",
      {AcceleratorBackend::Cuda, impl_->device_ordinal},
      {AcceleratorBackend::Cuda, output.impl_->device_ordinal});
  if (impl_->num_qubits != rhs.impl_->num_qubits) {
    throw std::invalid_argument("PauliSum commutes_with requires the same num_qubits");
  }

  const std::size_t entries = detail::checked_commutation_matrix_entries(
      impl_->num_terms,
      rhs.impl_->num_terms,
      max_commutation_matrix_entries);
  if (output.impl_->rows != impl_->num_terms || output.impl_->cols != rhs.impl_->num_terms ||
      output.impl_->entries != entries) {
    throw std::invalid_argument(
        "CUDA commutes_with_device output shape does not match operand term counts");
  }
  if (entries == 0) {
    return;
  }

  const int blocks = checked_launch_blocks(entries, "CUDA commutes_with_device");
  ScopedCudaDevice guard(impl_->device_ordinal);
  commutation_kernel<<<blocks, kCudaThreadsPerBlock>>>(
      impl_->x,
      impl_->z,
      rhs.impl_->x,
      rhs.impl_->z,
      impl_->num_terms,
      rhs.impl_->num_terms,
      impl_->words,
      output.mutable_data_for_device_write());
  check_cuda(cudaGetLastError(), "launch CUDA commutation device output");
  check_cuda(cudaDeviceSynchronize(), "synchronize CUDA commutation device output");
}

}  // namespace wolfgang

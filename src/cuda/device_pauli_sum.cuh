#pragma once

#include "wolfgang/device_pauli_sum.hpp"

#include "detail/accelerator_host_helpers.hpp"

#include <cuda_runtime_api.h>
#include <thrust/complex.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

namespace wolfgang {

struct DevicePauliSum::Impl {
  std::size_t num_qubits = 0;
  std::size_t words = 0;
  std::size_t num_terms = 0;
  std::uint64_t* x = nullptr;
  std::uint64_t* z = nullptr;
  thrust::complex<double>* coeffs = nullptr;
  int device_ordinal = 0;

  ~Impl();
};

namespace cuda_detail {

constexpr int kCudaThreadsPerBlock = 256;
constexpr std::size_t kMaxCudaLaunchBlocks = static_cast<std::size_t>(2147483647);

using detail::checked_bytes;
using detail::expected_statevector_length;
using detail::validate_statevector_length;

inline void check_cuda(cudaError_t result, const char* action) {
  if (result != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA ") + action + " failed: " + cudaGetErrorString(result));
  }
}

class ScopedCudaDevice {
public:
  explicit ScopedCudaDevice(int device) {
    check_cuda(cudaGetDevice(&previous_device_), "get current device");
    if (previous_device_ != device) {
      check_cuda(cudaSetDevice(device), "set device");
    }
  }

  ScopedCudaDevice(const ScopedCudaDevice&) = delete;
  ScopedCudaDevice& operator=(const ScopedCudaDevice&) = delete;

  ~ScopedCudaDevice() {
    if (previous_device_ >= 0) {
      (void)cudaSetDevice(previous_device_);
    }
  }

private:
  int previous_device_ = -1;
};

template <typename T>
void cuda_allocate(T*& pointer, std::size_t count, const char* name) {
  pointer = nullptr;
  const std::size_t bytes = checked_bytes(count, sizeof(T), name);
  if (bytes == 0) {
    return;
  }
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer), bytes), "allocate device buffer");
}

template <typename T>
void copy_to_device(T* dst, const T* src, std::size_t count, const char* name) {
  const std::size_t bytes = checked_bytes(count, sizeof(T), name);
  if (bytes == 0) {
    return;
  }
  check_cuda(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "copy host buffer to device");
}

inline void copy_bytes_to_device(void* dst, const void* src, std::size_t bytes, const char* name) {
  if (bytes == 0) {
    return;
  }
  (void)name;
  check_cuda(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "copy host buffer to device");
}

template <typename T>
void copy_to_host(T* dst, const T* src, std::size_t count, const char* name) {
  const std::size_t bytes = checked_bytes(count, sizeof(T), name);
  if (bytes == 0) {
    return;
  }
  check_cuda(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "copy device buffer to host");
}

template <typename T>
void copy_device_to_device(T* dst, const T* src, std::size_t count, const char* name) {
  const std::size_t bytes = checked_bytes(count, sizeof(T), name);
  if (bytes == 0) {
    return;
  }
  check_cuda(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice), "copy device buffer");
}

inline void validate_simplify_tolerances(double atol, double rtol) {
  if (atol < 0.0 || rtol < 0.0 || !std::isfinite(atol) || !std::isfinite(rtol)) {
    throw std::invalid_argument("simplify tolerances must be non-negative finite values");
  }
}

inline int checked_launch_blocks(std::size_t work_items, const char* operation) {
  const std::size_t blocks =
      (work_items + static_cast<std::size_t>(kCudaThreadsPerBlock) - 1) /
      static_cast<std::size_t>(kCudaThreadsPerBlock);
  if (blocks == 0 || blocks > kMaxCudaLaunchBlocks) {
    throw std::invalid_argument(
        std::string(operation) + " request exceeds CUDA launch grid limits");
  }
  return static_cast<int>(blocks);
}

}  // namespace cuda_detail

}  // namespace wolfgang

#pragma once

#include "wolfgang/device_pauli_sum.hpp"

#include "detail/accelerator_host_helpers.hpp"

#include <hip/hip_runtime.h>
#include <thrust/complex.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
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

namespace hip_detail {

constexpr int kHipThreadsPerBlock = 256;
constexpr std::size_t kMaxHipLaunchBlocks = static_cast<std::size_t>(2147483647);

using detail::checked_bytes;
using detail::expected_statevector_length;
using detail::validate_statevector_length;

inline void check_hip(hipError_t result, const char* action) {
  if (result != hipSuccess) {
    throw std::runtime_error(
        std::string("HIP ") + action + " failed: " + hipGetErrorString(result));
  }
}

inline void validate_simplify_tolerances(double atol, double rtol) {
  if (!std::isfinite(atol) || !std::isfinite(rtol) || atol < 0.0 || rtol < 0.0) {
    throw std::invalid_argument("simplify tolerances must be finite and non-negative");
  }
  if (atol > std::numeric_limits<double>::max() || rtol > std::numeric_limits<double>::max()) {
    throw std::invalid_argument("simplify tolerances must be finite and non-negative");
  }
}

class ScopedHipDevice {
public:
  explicit ScopedHipDevice(int device) {
    check_hip(hipGetDevice(&previous_device_), "get current device");
    if (previous_device_ != device) {
      check_hip(hipSetDevice(device), "set device");
    }
  }

  ScopedHipDevice(const ScopedHipDevice&) = delete;
  ScopedHipDevice& operator=(const ScopedHipDevice&) = delete;

  ~ScopedHipDevice() {
    if (previous_device_ >= 0) {
      (void)hipSetDevice(previous_device_);
    }
  }

private:
  int previous_device_ = -1;
};

template <typename T>
void hip_allocate(T*& pointer, std::size_t count, const char* name) {
  pointer = nullptr;
  const std::size_t bytes = checked_bytes(count, sizeof(T), name);
  if (bytes == 0) {
    return;
  }
  (void)name;
  check_hip(hipMalloc(reinterpret_cast<void**>(&pointer), bytes), "allocate device buffer");
}

template <typename T>
void copy_to_device(T* dst, const T* src, std::size_t count, const char* name) {
  const std::size_t bytes = checked_bytes(count, sizeof(T), name);
  if (bytes == 0) {
    return;
  }
  (void)name;
  check_hip(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice), "copy host buffer to device");
}

inline void copy_bytes_to_device(void* dst, const void* src, std::size_t bytes, const char* name) {
  if (bytes == 0) {
    return;
  }
  (void)name;
  check_hip(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice), "copy host buffer to device");
}

template <typename T>
void copy_to_host(T* dst, const T* src, std::size_t count, const char* name) {
  const std::size_t bytes = checked_bytes(count, sizeof(T), name);
  if (bytes == 0) {
    return;
  }
  (void)name;
  check_hip(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost), "copy device buffer to host");
}

template <typename T>
void copy_device_to_device(T* dst, const T* src, std::size_t count, const char* name) {
  const std::size_t bytes = checked_bytes(count, sizeof(T), name);
  if (bytes == 0) {
    return;
  }
  (void)name;
  check_hip(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice), "copy device buffer");
}

inline int checked_launch_blocks(std::size_t work_items, const char* operation) {
  const std::size_t blocks =
      (work_items + static_cast<std::size_t>(kHipThreadsPerBlock) - 1) /
      static_cast<std::size_t>(kHipThreadsPerBlock);
  if (blocks == 0 || blocks > kMaxHipLaunchBlocks) {
    throw std::invalid_argument(
        std::string(operation) + " request exceeds HIP launch grid limits");
  }
  return static_cast<int>(blocks);
}

}  // namespace hip_detail

}  // namespace wolfgang

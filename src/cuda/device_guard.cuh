#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <new>
#include <stdexcept>
#include <string>

namespace wolfgang::cuda_detail {

// True when `device`'s primary context exists, that is, when code in this
// process has already used the device. Only call it after the CUDA runtime has
// initialized the driver (any successful cudaSetDevice has).
inline bool device_has_primary_context(int device) noexcept {
  CUdevice handle = 0;
  unsigned int flags = 0;
  int active = 0;
  return cuDeviceGet(&handle, device) == CUDA_SUCCESS &&
         cuDevicePrimaryCtxGetState(handle, &flags, &active) == CUDA_SUCCESS && active != 0;
}

// Makes `device` current on the calling thread for the guard's lifetime, then
// switches back to the thread's previous device, unless that device has no
// primary context yet. Since CUDA 12, cudaSetDevice creates the primary context
// (about 500 MiB on an H100), so switching back to a device the process never
// used would charge a process that only works on device 1 for a context on
// device 0. In that case the thread stays on `device`; PyTorch's device guard
// makes the same choice.
class ScopedCudaDevice {
public:
  // Throws std::runtime_error when the device cannot be made current.
  explicit ScopedCudaDevice(int device) {
    if (!enter(device)) {
      throw std::runtime_error(
          std::string(previous_ < 0 ? "CUDA get current device failed: " : "CUDA set device failed: ") +
          cudaGetErrorString(error_));
    }
  }

  // For destructors and frees: a failure is cleared and leaves the thread where
  // it was; ok() reports whether `device` is current.
  ScopedCudaDevice(int device, std::nothrow_t) noexcept {
    if (!enter(device)) {
      (void)cudaGetLastError();
    }
  }

  ScopedCudaDevice(const ScopedCudaDevice&) = delete;
  ScopedCudaDevice& operator=(const ScopedCudaDevice&) = delete;

  ~ScopedCudaDevice() {
    if (switched_ && device_has_primary_context(previous_)) {
      (void)cudaSetDevice(previous_);
    }
  }

  [[nodiscard]] bool ok() const noexcept { return error_ == cudaSuccess; }

private:
  bool enter(int device) noexcept {
    error_ = cudaGetDevice(&previous_);
    if (error_ != cudaSuccess) {
      previous_ = -1;
      return false;
    }
    if (previous_ == device) {
      return true;
    }
    error_ = cudaSetDevice(device);
    if (error_ != cudaSuccess) {
      return false;
    }
    switched_ = true;
    return true;
  }

  int previous_ = -1;
  bool switched_ = false;
  cudaError_t error_ = cudaSuccess;
};

}  // namespace wolfgang::cuda_detail

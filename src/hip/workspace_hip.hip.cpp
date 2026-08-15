#include "workspace_hip.hip.hpp"

#include "device_pauli_sum.hip.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <utility>

namespace wolfgang::hip_detail {

HipTemporaryWorkspace::HipTemporaryWorkspace(int device_ordinal)
    : device_ordinal_(device_ordinal) {}

HipTemporaryWorkspace::~HipTemporaryWorkspace() {
  release();
}

HipTemporaryWorkspace::HipTemporaryWorkspace(HipTemporaryWorkspace&& other) noexcept
    : pointer_(std::exchange(other.pointer_, nullptr)),
      capacity_bytes_(std::exchange(other.capacity_bytes_, 0)),
      high_watermark_bytes_(std::exchange(other.high_watermark_bytes_, 0)),
      allocation_count_(std::exchange(other.allocation_count_, 0)),
      growth_count_(std::exchange(other.growth_count_, 0)),
      device_ordinal_(std::exchange(other.device_ordinal_, 0)) {}

HipTemporaryWorkspace& HipTemporaryWorkspace::operator=(HipTemporaryWorkspace&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  release();
  pointer_ = std::exchange(other.pointer_, nullptr);
  capacity_bytes_ = std::exchange(other.capacity_bytes_, 0);
  high_watermark_bytes_ = std::exchange(other.high_watermark_bytes_, 0);
  allocation_count_ = std::exchange(other.allocation_count_, 0);
  growth_count_ = std::exchange(other.growth_count_, 0);
  device_ordinal_ = std::exchange(other.device_ordinal_, 0);
  return *this;
}

void* HipTemporaryWorkspace::reserve(std::size_t bytes, const char* label) {
  (void)label;
  high_watermark_bytes_ = std::max(high_watermark_bytes_, bytes);
  if (bytes == 0) {
    return nullptr;
  }
  if (bytes <= capacity_bytes_) {
    return pointer_;
  }

  ScopedHipDevice guard(device_ordinal_);
  if (pointer_ != nullptr) {
    check_hip(hipFree(pointer_), "free temporary workspace before growth");
    ++growth_count_;
  }
  pointer_ = nullptr;
  check_hip(hipMalloc(&pointer_, bytes), "allocate temporary workspace");
  capacity_bytes_ = bytes;
  ++allocation_count_;
  return pointer_;
}

void HipTemporaryWorkspace::release() noexcept {
  if (pointer_ == nullptr) {
    capacity_bytes_ = 0;
    return;
  }

  int previous_device = -1;
  if (hipGetDevice(&previous_device) == hipSuccess && previous_device != device_ordinal_) {
    (void)hipSetDevice(device_ordinal_);
  }
  (void)hipFree(pointer_);
  if (previous_device >= 0 && previous_device != device_ordinal_) {
    (void)hipSetDevice(previous_device);
  }
  pointer_ = nullptr;
  capacity_bytes_ = 0;
}

std::size_t HipTemporaryWorkspace::capacity_bytes() const noexcept {
  return capacity_bytes_;
}

std::size_t HipTemporaryWorkspace::high_watermark_bytes() const noexcept {
  return high_watermark_bytes_;
}

std::size_t HipTemporaryWorkspace::allocation_count() const noexcept {
  return allocation_count_;
}

std::size_t HipTemporaryWorkspace::growth_count() const noexcept {
  return growth_count_;
}

}  // namespace wolfgang::hip_detail

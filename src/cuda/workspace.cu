#include "cuda/workspace.cuh"

#include "device_pauli_sum.cuh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace wolfgang::cuda::detail {

namespace {

bool is_power_of_two(std::size_t value) noexcept {
  return value != 0 && (value & (value - 1U)) == 0;
}

std::size_t checked_allocation_bytes(std::size_t bytes, std::size_t alignment) {
  const std::size_t padding = alignment > 0 ? alignment - 1U : 0;
  if (bytes > std::numeric_limits<std::size_t>::max() - padding) {
    throw std::overflow_error("CUDA workspace allocation byte count overflow");
  }
  return bytes + padding;
}

void* aligned_pointer(void* ptr, std::size_t alignment) noexcept {
  const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(ptr);
  const std::uintptr_t mask = static_cast<std::uintptr_t>(alignment - 1U);
  const std::uintptr_t aligned = (raw + mask) & ~mask;
  return reinterpret_cast<void*>(aligned);
}

[[noreturn]] void throw_invalid_workspace_mode(const char* value) {
  throw std::invalid_argument(
      std::string("FASTPAULI_CUDA_BENCH_WORKSPACE_MODE must be absent, grow_inside_timing, "
                  "or pre_reserved_outside_timing, got: ") +
      value);
}

}  // namespace

CudaWorkspace::CudaWorkspace(int device_ordinal) : device_ordinal_(device_ordinal) {
  if (device_ordinal < 0) {
    throw std::invalid_argument("CUDA workspace device ordinal must be non-negative");
  }
}

CudaWorkspace::CudaWorkspace(CudaWorkspace&& other) noexcept {
  steal_from(std::move(other));
}

CudaWorkspace& CudaWorkspace::operator=(CudaWorkspace&& other) noexcept {
  if (this != &other) {
    release();
    steal_from(std::move(other));
  }
  return *this;
}

CudaWorkspace::~CudaWorkspace() {
  release();
}

void CudaWorkspace::ensure_device(int operand_device_ordinal) const {
  if (operand_device_ordinal != device_ordinal_) {
    throw std::invalid_argument(
        "CUDA workspace device mismatch during workspace use: workspace device ordinal " +
        std::to_string(device_ordinal_) + ", operand device ordinal " +
        std::to_string(operand_device_ordinal));
  }
}

void* CudaWorkspace::reserve_bytes(std::size_t bytes, std::size_t alignment) {
  if (alignment == 0) {
    alignment = 1;
  }
  if (!is_power_of_two(alignment)) {
    throw std::invalid_argument("CUDA workspace alignment must be a power of two");
  }
  high_watermark_bytes_ = std::max(high_watermark_bytes_, bytes);
  if (bytes == 0) {
    return nullptr;
  }
  if (base_ptr_ != nullptr && bytes <= reserved_bytes_ && alignment <= alignment_) {
    return aligned_ptr_;
  }

  release();
  const std::size_t allocation_bytes = checked_allocation_bytes(bytes, alignment);
  cuda_detail::ScopedCudaDevice guard(device_ordinal_);
  cudaError_t result = cudaMalloc(&base_ptr_, allocation_bytes);
  if (result != cudaSuccess) {
    const std::string message =
        "CUDA workspace allocate failed during reserve_bytes: workspace device ordinal " +
        std::to_string(device_ordinal_) + ", requested bytes " + std::to_string(bytes) +
        ", allocation bytes " + std::to_string(allocation_bytes) + ": " +
        cudaGetErrorString(result);
    throw std::runtime_error(message);
  }

  aligned_ptr_ = aligned_pointer(base_ptr_, alignment);
  reserved_bytes_ = bytes;
  allocated_bytes_ = allocation_bytes;
  alignment_ = alignment;
  ++allocation_count_;
  ++growth_count_;
  return aligned_ptr_;
}

void CudaWorkspace::reset() noexcept {
  // A reset intentionally keeps the allocation.  It marks the scratch region as
  // reusable for the next operation without changing capacity or lifetime
  // counters; callers still need release() when they want to return memory.
}

void CudaWorkspace::release() noexcept {
  if (base_ptr_ != nullptr) {
    int previous_device = -1;
    const cudaError_t current_result = cudaGetDevice(&previous_device);
    (void)cudaSetDevice(device_ordinal_);
    (void)cudaFree(base_ptr_);
    if (current_result == cudaSuccess && previous_device >= 0 &&
        previous_device != device_ordinal_) {
      (void)cudaSetDevice(previous_device);
    }
  }
  base_ptr_ = nullptr;
  aligned_ptr_ = nullptr;
  reserved_bytes_ = 0;
  allocated_bytes_ = 0;
  alignment_ = 1;
}

WorkspaceSnapshot CudaWorkspace::snapshot() const noexcept {
  return {
      device_ordinal_,
      reserved_bytes_,
      high_watermark_bytes_,
      allocation_count_,
      growth_count_,
  };
}

void CudaWorkspace::steal_from(CudaWorkspace&& other) noexcept {
  base_ptr_ = other.base_ptr_;
  aligned_ptr_ = other.aligned_ptr_;
  reserved_bytes_ = other.reserved_bytes_;
  allocated_bytes_ = other.allocated_bytes_;
  alignment_ = other.alignment_;
  high_watermark_bytes_ = other.high_watermark_bytes_;
  allocation_count_ = other.allocation_count_;
  growth_count_ = other.growth_count_;
  device_ordinal_ = other.device_ordinal_;

  other.base_ptr_ = nullptr;
  other.aligned_ptr_ = nullptr;
  other.reserved_bytes_ = 0;
  other.allocated_bytes_ = 0;
  other.alignment_ = 1;
  other.high_watermark_bytes_ = 0;
  other.allocation_count_ = 0;
  other.growth_count_ = 0;
  other.device_ordinal_ = -1;
}

WorkspaceTimingMode workspace_timing_mode_from_env() {
  const char* value = std::getenv("FASTPAULI_CUDA_BENCH_WORKSPACE_MODE");
  if (value == nullptr || std::string(value).empty() || std::string(value) == "absent") {
    return WorkspaceTimingMode::kAbsent;
  }
  const std::string setting(value);
  if (setting == "grow_inside_timing") {
    return WorkspaceTimingMode::kGrowInsideTiming;
  }
  if (setting == "pre_reserved_outside_timing") {
    return WorkspaceTimingMode::kPreReservedOutsideTiming;
  }
  throw_invalid_workspace_mode(value);
}

const char* workspace_timing_mode_name(WorkspaceTimingMode mode) noexcept {
  switch (mode) {
    case WorkspaceTimingMode::kAbsent:
      return "absent";
    case WorkspaceTimingMode::kGrowInsideTiming:
      return "grow_inside_timing";
    case WorkspaceTimingMode::kPreReservedOutsideTiming:
      return "pre_reserved_outside_timing";
  }
  return "unknown";
}

}  // namespace wolfgang::cuda::detail

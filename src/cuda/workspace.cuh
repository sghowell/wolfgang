#pragma once

#include "detail/accelerator_host_helpers.hpp"

#include <cstddef>

namespace wolfgang::cuda::detail {

using detail::WorkspaceSnapshot;
using detail::WorkspaceTimingMode;
using detail::workspace_timing_mode_name;

class CudaWorkspace {
public:
  explicit CudaWorkspace(int device_ordinal);
  CudaWorkspace(const CudaWorkspace&) = delete;
  CudaWorkspace& operator=(const CudaWorkspace&) = delete;
  CudaWorkspace(CudaWorkspace&& other) noexcept;
  CudaWorkspace& operator=(CudaWorkspace&& other) noexcept;
  ~CudaWorkspace();

  [[nodiscard]] int device_ordinal() const noexcept { return device_ordinal_; }
  void ensure_device(int operand_device_ordinal) const;
  void* reserve_bytes(std::size_t bytes, std::size_t alignment);
  void reset() noexcept;
  void release() noexcept;
  [[nodiscard]] WorkspaceSnapshot snapshot() const noexcept;

private:
  void steal_from(CudaWorkspace&& other) noexcept;

  void* base_ptr_ = nullptr;
  void* aligned_ptr_ = nullptr;
  std::size_t reserved_bytes_ = 0;
  std::size_t allocated_bytes_ = 0;
  std::size_t alignment_ = 1;
  std::size_t high_watermark_bytes_ = 0;
  std::size_t allocation_count_ = 0;
  std::size_t growth_count_ = 0;
  int device_ordinal_ = -1;
};

[[nodiscard]] WorkspaceTimingMode workspace_timing_mode_from_env();
[[nodiscard]] const char* workspace_timing_mode_name(WorkspaceTimingMode mode) noexcept;

}  // namespace wolfgang::cuda::detail

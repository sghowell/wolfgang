#pragma once

#include "detail/accelerator_host_helpers.hpp"

#include <cstddef>

namespace wolfgang::hip_detail {

using detail::WorkspaceSnapshot;
using detail::WorkspaceTimingMode;
using detail::workspace_timing_mode_name;

class HipTemporaryWorkspace {
 public:
  explicit HipTemporaryWorkspace(int device_ordinal);
  ~HipTemporaryWorkspace();

  HipTemporaryWorkspace(HipTemporaryWorkspace&& other) noexcept;
  HipTemporaryWorkspace& operator=(HipTemporaryWorkspace&& other) noexcept;

  HipTemporaryWorkspace(const HipTemporaryWorkspace&) = delete;
  HipTemporaryWorkspace& operator=(const HipTemporaryWorkspace&) = delete;

  [[nodiscard]] int device_ordinal() const noexcept { return device_ordinal_; }
  void ensure_device(int operand_device_ordinal) const;
  void* reserve(std::size_t bytes, const char* label);
  void reset() noexcept;
  void release() noexcept;
  [[nodiscard]] WorkspaceSnapshot snapshot(WorkspaceTimingMode mode) const noexcept;

  [[nodiscard]] std::size_t capacity_bytes() const noexcept;
  [[nodiscard]] std::size_t high_watermark_bytes() const noexcept;
  [[nodiscard]] std::size_t allocation_count() const noexcept;
  [[nodiscard]] std::size_t growth_count() const noexcept;

 private:
  void* pointer_ = nullptr;
  std::size_t capacity_bytes_ = 0;
  std::size_t high_watermark_bytes_ = 0;
  std::size_t allocation_count_ = 0;
  std::size_t growth_count_ = 0;
  int device_ordinal_ = 0;
};

}  // namespace wolfgang::hip_detail

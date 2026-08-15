#pragma once

#include <Metal/Metal.h>

#include <cstddef>

namespace wolfgang::metal_detail {

enum class WorkspaceTimingMode {
  kAbsent,
  kGrowInsideTiming,
  kPreReservedOutsideTiming,
};

struct WorkspaceSnapshot {
  int device_ordinal = -1;
  std::size_t reserved_bytes = 0;
  std::size_t high_watermark_bytes = 0;
  std::size_t allocation_count = 0;
  std::size_t growth_count = 0;
  const char* timing_mode = "absent";
};

// Private scratch buffer owner for future device-resident Metal algorithms.
//
// The class intentionally stays inside src/metal. It does not expose command
// queues, Metal buffers, or lifetime handles through the public Python API.
// Campaign 6 uses it to pin down the allocation and timing vocabulary before a
// retained Metal sort/prefix/reduce simplify primitive exists.
class MetalWorkspace {
 public:
  explicit MetalWorkspace(int device_ordinal = -1) noexcept;
  MetalWorkspace(const MetalWorkspace&) = delete;
  MetalWorkspace& operator=(const MetalWorkspace&) = delete;
  MetalWorkspace(MetalWorkspace&& other) noexcept;
  MetalWorkspace& operator=(MetalWorkspace&& other) noexcept;
  ~MetalWorkspace();

  [[nodiscard]] id<MTLBuffer> buffer() const noexcept;
  [[nodiscard]] std::size_t reserved_bytes() const noexcept;
  [[nodiscard]] std::size_t high_watermark_bytes() const noexcept;
  [[nodiscard]] std::size_t allocation_count() const noexcept;
  [[nodiscard]] std::size_t growth_count() const noexcept;

  void reserve_bytes(id<MTLDevice> device, std::size_t bytes, std::size_t alignment = 256);
  void reset() noexcept;
  void release() noexcept;

  [[nodiscard]] WorkspaceSnapshot snapshot(WorkspaceTimingMode mode) const noexcept;

 private:
  int device_ordinal_ = -1;
  id<MTLBuffer> buffer_ = nil;
  std::size_t reserved_bytes_ = 0;
  std::size_t high_watermark_bytes_ = 0;
  std::size_t allocation_count_ = 0;
  std::size_t growth_count_ = 0;
};

[[nodiscard]] WorkspaceTimingMode workspace_timing_mode_from_env() noexcept;
[[nodiscard]] const char* workspace_timing_mode_name(WorkspaceTimingMode mode) noexcept;

}  // namespace wolfgang::metal_detail

#include "workspace_metal.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace wolfgang::metal_detail {

namespace {

constexpr const char* kWorkspaceTimingEnv = "FASTPAULI_METAL_BENCH_WORKSPACE_TIMING";

std::size_t align_up(std::size_t value, std::size_t alignment) {
  if (alignment == 0) {
    throw std::invalid_argument("Metal workspace alignment must be non-zero");
  }
  const std::size_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  const std::size_t delta = alignment - remainder;
  if (value > std::numeric_limits<std::size_t>::max() - delta) {
    throw std::overflow_error("Metal workspace reservation size overflow");
  }
  return value + delta;
}

void release_buffer(id<MTLBuffer> buffer) noexcept {
#if !__has_feature(objc_arc)
  [buffer release];
#else
  (void)buffer;
#endif
}

}  // namespace

MetalWorkspace::MetalWorkspace(int device_ordinal) noexcept : device_ordinal_(device_ordinal) {}

MetalWorkspace::MetalWorkspace(MetalWorkspace&& other) noexcept
    : device_ordinal_(other.device_ordinal_),
      buffer_(other.buffer_),
      reserved_bytes_(other.reserved_bytes_),
      high_watermark_bytes_(other.high_watermark_bytes_),
      allocation_count_(other.allocation_count_),
      growth_count_(other.growth_count_) {
  other.buffer_ = nil;
  other.reserved_bytes_ = 0;
  other.high_watermark_bytes_ = 0;
  other.allocation_count_ = 0;
  other.growth_count_ = 0;
}

MetalWorkspace& MetalWorkspace::operator=(MetalWorkspace&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  release();
  device_ordinal_ = other.device_ordinal_;
  buffer_ = other.buffer_;
  reserved_bytes_ = other.reserved_bytes_;
  high_watermark_bytes_ = other.high_watermark_bytes_;
  allocation_count_ = other.allocation_count_;
  growth_count_ = other.growth_count_;
  other.buffer_ = nil;
  other.reserved_bytes_ = 0;
  other.high_watermark_bytes_ = 0;
  other.allocation_count_ = 0;
  other.growth_count_ = 0;
  return *this;
}

MetalWorkspace::~MetalWorkspace() {
  release();
}

id<MTLBuffer> MetalWorkspace::buffer() const noexcept {
  return buffer_;
}

std::size_t MetalWorkspace::reserved_bytes() const noexcept {
  return reserved_bytes_;
}

std::size_t MetalWorkspace::high_watermark_bytes() const noexcept {
  return high_watermark_bytes_;
}

std::size_t MetalWorkspace::allocation_count() const noexcept {
  return allocation_count_;
}

std::size_t MetalWorkspace::growth_count() const noexcept {
  return growth_count_;
}

void MetalWorkspace::reserve_bytes(
    id<MTLDevice> device,
    std::size_t bytes,
    std::size_t alignment) {
  if (bytes == 0) {
    return;
  }
  if (device == nil) {
    throw std::invalid_argument("Metal workspace reservation requires a valid MTLDevice");
  }
  const std::size_t aligned_bytes = align_up(bytes, alignment);
  high_watermark_bytes_ = std::max(high_watermark_bytes_, aligned_bytes);
  if (buffer_ != nil && reserved_bytes_ >= aligned_bytes) {
    return;
  }

  id<MTLBuffer> next = [device newBufferWithLength:aligned_bytes
                                           options:MTLResourceStorageModeShared];
  if (next == nil) {
    throw std::runtime_error("failed to allocate Metal workspace buffer");
  }
  const bool grew_existing_buffer = buffer_ != nil;
  release_buffer(buffer_);
  buffer_ = next;
  reserved_bytes_ = aligned_bytes;
  allocation_count_ += 1;
  if (grew_existing_buffer) {
    growth_count_ += 1;
  }
}

void MetalWorkspace::reset() noexcept {
  // Campaign 6 tracks a whole-buffer reservation. Future algorithms can add an
  // allocation cursor here without changing the public accelerator API.
}

void MetalWorkspace::release() noexcept {
  release_buffer(buffer_);
  buffer_ = nil;
  reserved_bytes_ = 0;
}

WorkspaceSnapshot MetalWorkspace::snapshot(WorkspaceTimingMode mode) const noexcept {
  return WorkspaceSnapshot{
      device_ordinal_,
      reserved_bytes_,
      high_watermark_bytes_,
      allocation_count_,
      growth_count_,
      workspace_timing_mode_name(mode),
  };
}

WorkspaceTimingMode workspace_timing_mode_from_env() noexcept {
  const char* value = std::getenv(kWorkspaceTimingEnv);
  if (value == nullptr || std::string_view(value).empty()) {
    return WorkspaceTimingMode::kAbsent;
  }
  const std::string_view mode(value);
  if (mode == "grow_inside_timing") {
    return WorkspaceTimingMode::kGrowInsideTiming;
  }
  if (mode == "pre_reserved_outside_timing") {
    return WorkspaceTimingMode::kPreReservedOutsideTiming;
  }
  return WorkspaceTimingMode::kAbsent;
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
  return "absent";
}

}  // namespace wolfgang::metal_detail

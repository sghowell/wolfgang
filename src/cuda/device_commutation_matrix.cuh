#pragma once

#include "wolfgang/device_commutation_matrix.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace wolfgang {

struct DeviceCommutationMatrix::Impl {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t entries = 0;
  // Drawn from the Wolfgang pool. Once the pointer has been handed out through
  // DLPack or the CUDA Array Interface, a consumer may still have work queued
  // on another stream, so the buffer returns to the pool only after a device
  // synchronization.
  std::uint8_t* data = nullptr;
  int device_ordinal = 0;
  bool exported = false;

  ~Impl();
};

namespace detail {

// Device data for Wolfgang's own kernels, without marking the matrix exported.
struct DeviceCommutationMatrixAccess {
  static const std::uint8_t* data(const DeviceCommutationMatrix& matrix) noexcept {
    return matrix.impl_ ? matrix.impl_->data : nullptr;
  }
};

}  // namespace detail

namespace cuda::benchmark {

struct FusedCsrGraphResult {
  std::vector<std::uint64_t> row_offsets;
  std::vector<std::uint64_t> col_indices;
  std::uint64_t edge_count = 0;
  std::uint64_t col_index_checksum = 0;
  std::uint64_t row_offset_checksum = 0;
};

struct FusedConflictDegreesResult {
  std::vector<std::uint64_t> row_conflicts;
  std::vector<std::uint64_t> col_conflicts;
  std::uint64_t row_conflict_sum = 0;
  std::uint64_t col_conflict_sum = 0;
};

struct FusedGroupingSummaryResult {
  std::vector<std::uint64_t> row_conflicts;
  std::vector<std::uint64_t> col_conflicts;
  std::vector<std::uint64_t> top_row_indices;
  std::vector<std::uint64_t> top_row_conflicts;
  std::vector<std::uint64_t> top_col_indices;
  std::vector<std::uint64_t> top_col_conflicts;
  std::uint64_t row_conflict_sum = 0;
  std::uint64_t col_conflict_sum = 0;
};

[[nodiscard]] FusedCsrGraphResult fused_anticommutation_csr(
    const DeviceCommutationMatrix& matrix,
    bool include_outputs);

[[nodiscard]] FusedConflictDegreesResult fused_conflict_degrees(
    const DeviceCommutationMatrix& matrix,
    bool include_outputs);

[[nodiscard]] FusedGroupingSummaryResult fused_grouping_summary(
    const DeviceCommutationMatrix& matrix,
    std::size_t top_k,
    bool include_outputs);

}  // namespace cuda::benchmark

}  // namespace wolfgang

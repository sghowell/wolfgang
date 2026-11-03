#include "device_commutation_matrix.cuh"

#include "device_pauli_sum.cuh"
#include "cuda/pool_allocator.cuh"
#include "detail/checked_arithmetic.hpp"

#include <cuda_runtime_api.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace wolfgang {

using cuda_detail::check_cuda;
using cuda_detail::copy_to_device;
using cuda_detail::copy_to_host;
using cuda_detail::PoolBuffer;
using cuda_detail::pool_allocate;
using cuda_detail::pool_free_bytes_on_device;
using cuda_detail::ScopedCudaDevice;

namespace {

constexpr int kCountThreadsPerBlock = cuda_detail::kCudaThreadsPerBlock;
constexpr int kCsrThreadsPerBlock = cuda_detail::kCudaThreadsPerBlock;

void validate_device_ordinal_for_matrix(int device) {
  int device_count = 0;
  const cudaError_t count_result = cudaGetDeviceCount(&device_count);
  if (count_result != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA runtime library is unavailable or failed during device discovery: ") +
        cudaGetErrorString(count_result));
  }
  if (device_count == 0) {
    throw std::runtime_error("no CUDA device is available");
  }
  if (device < 0 || device >= device_count) {
    throw std::invalid_argument("CUDA device ordinal is out of range");
  }
}

int checked_reduction_blocks(std::size_t count, const char* operation) {
  if (count == 0) {
    return 0;
  }
  if (count > cuda_detail::kMaxCudaLaunchBlocks) {
    throw std::invalid_argument(
        std::string(operation) + " request exceeds CUDA launch grid limits");
  }
  return static_cast<int>(count);
}

void validate_count_result_fits(std::size_t entries) {
  if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
    if (entries > static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())) {
      throw std::overflow_error("device commutation matrix count exceeds uint64 range");
    }
  }
}

__device__ std::uint64_t block_reduce_sum(std::uint64_t value) {
  __shared__ std::uint64_t scratch[kCountThreadsPerBlock];
  const int thread = threadIdx.x;
  scratch[thread] = value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (thread < stride) {
      scratch[thread] += scratch[thread + stride];
    }
    __syncthreads();
  }
  return scratch[0];
}

__global__ void count_total_kernel(
    const std::uint8_t* data,
    std::size_t entries,
    std::uint64_t* partials) {
  const std::size_t global_thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + static_cast<std::size_t>(threadIdx.x);
  std::uint64_t count = 0;
  if (global_thread < entries) {
    count = data[global_thread] != 0U ? 1U : 0U;
  }
  const std::uint64_t block_count = block_reduce_sum(count);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = block_count;
  }
}

__global__ void count_rows_kernel(
    const std::uint8_t* data,
    std::size_t rows,
    std::size_t cols,
    std::uint64_t* output) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  std::uint64_t count = 0;
  const std::size_t row_offset = row * cols;
  for (std::size_t col = static_cast<std::size_t>(threadIdx.x); col < cols;
       col += static_cast<std::size_t>(blockDim.x)) {
    count += data[row_offset + col] != 0U ? 1U : 0U;
  }
  const std::uint64_t block_count = block_reduce_sum(count);
  if (threadIdx.x == 0) {
    output[row] = block_count;
  }
}

__global__ void count_cols_kernel(
    const std::uint8_t* data,
    std::size_t rows,
    std::size_t cols,
    std::uint64_t* output) {
  const std::size_t col = static_cast<std::size_t>(blockIdx.x);
  if (col >= cols) {
    return;
  }

  std::uint64_t count = 0;
  for (std::size_t row = static_cast<std::size_t>(threadIdx.x); row < rows;
       row += static_cast<std::size_t>(blockDim.x)) {
    count += data[row * cols + col] != 0U ? 1U : 0U;
  }
  const std::uint64_t block_count = block_reduce_sum(count);
  if (threadIdx.x == 0) {
    output[col] = block_count;
  }
}

__global__ void count_row_conflicts_kernel(
    const std::uint8_t* data,
    std::size_t rows,
    std::size_t cols,
    std::uint64_t* output) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  std::uint64_t count = 0;
  const std::size_t row_offset = row * cols;
  for (std::size_t col = static_cast<std::size_t>(threadIdx.x); col < cols;
       col += static_cast<std::size_t>(blockDim.x)) {
    count += data[row_offset + col] == 0U ? 1U : 0U;
  }
  const std::uint64_t block_count = block_reduce_sum(count);
  if (threadIdx.x == 0) {
    output[row] = block_count;
  }
}

__global__ void count_col_conflicts_kernel(
    const std::uint8_t* data,
    std::size_t rows,
    std::size_t cols,
    std::uint64_t* output) {
  const std::size_t col = static_cast<std::size_t>(blockIdx.x);
  if (col >= cols) {
    return;
  }

  std::uint64_t count = 0;
  for (std::size_t row = static_cast<std::size_t>(threadIdx.x); row < rows;
       row += static_cast<std::size_t>(blockDim.x)) {
    count += data[row * cols + col] == 0U ? 1U : 0U;
  }
  const std::uint64_t block_count = block_reduce_sum(count);
  if (threadIdx.x == 0) {
    output[col] = block_count;
  }
}

__global__ void set_final_csr_offset_kernel(
    const std::uint64_t* row_counts,
    std::size_t rows,
    std::uint64_t* row_offsets) {
  if (rows == 0) {
    row_offsets[0] = 0;
    return;
  }
  row_offsets[rows] = row_offsets[rows - 1] + row_counts[rows - 1];
}

__global__ void scatter_csr_conflicts_sorted_by_row_kernel(
    const std::uint8_t* data,
    std::size_t rows,
    std::size_t cols,
    const std::uint64_t* row_offsets,
    std::uint64_t* col_indices) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  __shared__ unsigned int prefix[kCsrThreadsPerBlock];
  __shared__ std::uint64_t tile_base;
  if (threadIdx.x == 0) {
    tile_base = 0;
  }
  __syncthreads();

  const std::size_t row_offset = row * cols;
  const std::uint64_t output_row_offset = row_offsets[row];
  for (std::size_t tile_start = 0; tile_start < cols;
       tile_start += static_cast<std::size_t>(blockDim.x)) {
    const std::size_t col = tile_start + static_cast<std::size_t>(threadIdx.x);
    const unsigned int is_conflict =
        (col < cols && data[row_offset + col] == 0U) ? 1U : 0U;
    prefix[threadIdx.x] = is_conflict;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
      unsigned int value = 0;
      if (threadIdx.x >= stride) {
        value = prefix[threadIdx.x - stride];
      }
      __syncthreads();
      prefix[threadIdx.x] += value;
      __syncthreads();
    }

    const unsigned int inclusive = prefix[threadIdx.x];
    const unsigned int exclusive = inclusive - is_conflict;
    const unsigned int tile_count = prefix[blockDim.x - 1];
    const std::uint64_t write_base = tile_base;
    if (is_conflict != 0U) {
      col_indices[output_row_offset + write_base + exclusive] =
          static_cast<std::uint64_t>(col);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      tile_base += static_cast<std::uint64_t>(tile_count);
    }
    __syncthreads();
  }
}

std::uint64_t checksum_uint64_values(const std::vector<std::uint64_t>& values) {
  std::uint64_t checksum = 1469598103934665603ULL;
  for (std::uint64_t value : values) {
    checksum ^= value;
    checksum *= 1099511628211ULL;
  }
  return checksum;
}

std::uint64_t sum_uint64_values(const std::vector<std::uint64_t>& values) {
  std::uint64_t total = 0;
  for (std::uint64_t value : values) {
    total += value;
  }
  return total;
}

std::vector<std::uint64_t> top_indices_by_value(
    const std::vector<std::uint64_t>& values,
    std::size_t top_k) {
  std::vector<std::uint64_t> indices(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    indices[index] = static_cast<std::uint64_t>(index);
  }
  const std::size_t limit = std::min(top_k, indices.size());
  std::partial_sort(
      indices.begin(),
      indices.begin() + static_cast<std::ptrdiff_t>(limit),
      indices.end(),
      [&values](std::uint64_t lhs, std::uint64_t rhs) {
        const std::uint64_t lhs_value = values[static_cast<std::size_t>(lhs)];
        const std::uint64_t rhs_value = values[static_cast<std::size_t>(rhs)];
        if (lhs_value == rhs_value) {
          return lhs < rhs;
        }
        return lhs_value > rhs_value;
      });
  indices.resize(limit);
  return indices;
}

std::vector<std::uint64_t> gather_values_by_index(
    const std::vector<std::uint64_t>& values,
    const std::vector<std::uint64_t>& indices) {
  std::vector<std::uint64_t> gathered;
  gathered.reserve(indices.size());
  for (std::uint64_t index : indices) {
    gathered.push_back(values[static_cast<std::size_t>(index)]);
  }
  return gathered;
}

}  // namespace

DeviceCommutationMatrix::Impl::~Impl() {
  if (data == nullptr) {
    return;
  }
  if (exported) {
    const cuda_detail::ScopedCudaDevice guard(device_ordinal, std::nothrow);
    if (guard.ok()) {
      (void)cudaDeviceSynchronize();
    }
    (void)cudaGetLastError();
  }
  pool_free_bytes_on_device(data, device_ordinal);
}

DeviceCommutationMatrix::DeviceCommutationMatrix() noexcept = default;

DeviceCommutationMatrix::DeviceCommutationMatrix(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

DeviceCommutationMatrix::~DeviceCommutationMatrix() = default;

DeviceCommutationMatrix::DeviceCommutationMatrix(DeviceCommutationMatrix&& other) noexcept =
    default;

DeviceCommutationMatrix& DeviceCommutationMatrix::operator=(
    DeviceCommutationMatrix&& other) noexcept = default;

DeviceCommutationMatrix DeviceCommutationMatrix::empty(
    std::size_t rows,
    std::size_t cols,
    int device) {
  return empty(rows, cols, AcceleratorBackend::None, device);
}

DeviceCommutationMatrix DeviceCommutationMatrix::empty(
    std::size_t rows,
    std::size_t cols,
    AcceleratorBackend backend,
    int device) {
  // commutes_with_device creates its output here on every call; a device count
  // selects the backend without querying every device's properties.
  int device_count = 0;
  const cudaError_t count_result = cudaGetDeviceCount(&device_count);
  if (count_result != cudaSuccess) {
    (void)cudaGetLastError();
  }
  const AcceleratorBackend selected = select_accelerator_backend(
      backend,
      true,
      count_result == cudaSuccess && device_count > 0,
      false,
      false,
      false,
      false);
  if (selected != AcceleratorBackend::Cuda) {
    throw std::runtime_error(accelerator_not_built_message(selected));
  }
  validate_device_ordinal_for_matrix(device);

  auto impl = std::make_unique<Impl>();
  impl->rows = rows;
  impl->cols = cols;
  impl->entries = detail::checked_product(rows, cols, "device commutation matrix entries");
  impl->device_ordinal = device;

  ScopedCudaDevice guard(device);
  pool_allocate(impl->data, impl->entries, "device commutation matrix");
  return DeviceCommutationMatrix(std::move(impl));
}

std::vector<std::uint8_t> DeviceCommutationMatrix::to_host() const {
  if (!impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }

  std::vector<std::uint8_t> host(impl_->entries);
  ScopedCudaDevice guard(impl_->device_ordinal);
  copy_to_host(host.data(), impl_->data, impl_->entries, "device commutation matrix");
  return host;
}

std::uint64_t DeviceCommutationMatrix::count_commuting() const {
  if (!impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }
  validate_count_result_fits(impl_->entries);
  if (impl_->entries == 0) {
    return 0;
  }

  const int blocks = cuda_detail::checked_launch_blocks(
      impl_->entries,
      "CUDA device commutation matrix total count");
  ScopedCudaDevice guard(impl_->device_ordinal);
  const PoolBuffer<std::uint64_t> partials_buffer(static_cast<std::size_t>(blocks), "device commutation count partials");
  std::uint64_t* device_partials = partials_buffer.get();

  count_total_kernel<<<blocks, kCountThreadsPerBlock>>>(
      impl_->data,
      impl_->entries,
      device_partials);
  check_cuda(cudaGetLastError(), "launch device commutation total count");

  std::vector<std::uint64_t> partials(static_cast<std::size_t>(blocks));
  copy_to_host(
      partials.data(),
      device_partials,
      partials.size(),
      "device commutation count partials");
  std::uint64_t total = 0;
  for (std::uint64_t value : partials) {
    total += value;
  }
  return total;
}

std::vector<std::uint64_t> DeviceCommutationMatrix::count_commuting_rows() const {
  if (!impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }
  validate_count_result_fits(impl_->entries);

  std::vector<std::uint64_t> host(impl_->rows, 0);
  if (impl_->rows == 0 || impl_->cols == 0) {
    return host;
  }

  const int blocks = checked_reduction_blocks(
      impl_->rows,
      "CUDA device commutation matrix row counts");
  ScopedCudaDevice guard(impl_->device_ordinal);
  const PoolBuffer<std::uint64_t> counts_buffer(impl_->rows, "device commutation row counts");
  std::uint64_t* device_counts = counts_buffer.get();

  count_rows_kernel<<<blocks, kCountThreadsPerBlock>>>(
      impl_->data,
      impl_->rows,
      impl_->cols,
      device_counts);
  check_cuda(cudaGetLastError(), "launch device commutation row counts");
  copy_to_host(host.data(), device_counts, host.size(), "device commutation row counts");
  return host;
}

std::vector<std::uint64_t> DeviceCommutationMatrix::count_commuting_cols() const {
  if (!impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }
  validate_count_result_fits(impl_->entries);

  std::vector<std::uint64_t> host(impl_->cols, 0);
  if (impl_->rows == 0 || impl_->cols == 0) {
    return host;
  }

  const int blocks = checked_reduction_blocks(
      impl_->cols,
      "CUDA device commutation matrix column counts");
  ScopedCudaDevice guard(impl_->device_ordinal);
  const PoolBuffer<std::uint64_t> counts_buffer(impl_->cols, "device commutation column counts");
  std::uint64_t* device_counts = counts_buffer.get();

  count_cols_kernel<<<blocks, kCountThreadsPerBlock>>>(
      impl_->data,
      impl_->rows,
      impl_->cols,
      device_counts);
  check_cuda(cudaGetLastError(), "launch device commutation column counts");
  copy_to_host(host.data(), device_counts, host.size(), "device commutation column counts");
  return host;
}

std::size_t DeviceCommutationMatrix::rows() const noexcept {
  return impl_ ? impl_->rows : 0;
}

std::size_t DeviceCommutationMatrix::cols() const noexcept {
  return impl_ ? impl_->cols : 0;
}

std::size_t DeviceCommutationMatrix::num_entries() const noexcept {
  return impl_ ? impl_->entries : 0;
}

int DeviceCommutationMatrix::device() const noexcept {
  return impl_ ? impl_->device_ordinal : -1;
}

std::string DeviceCommutationMatrix::backend() const {
  return std::string(accelerator_backend_name(
      impl_ ? AcceleratorBackend::Cuda : AcceleratorBackend::None));
}

std::uintptr_t DeviceCommutationMatrix::data_pointer_for_cuda_array_interface() const {
  if (!impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }
  impl_->exported = true;  // freeing now synchronizes the device first
  cuda_detail::grant_peer_access_for_export(impl_->device_ordinal);
  return reinterpret_cast<std::uintptr_t>(impl_->data);
}

std::uintptr_t DeviceCommutationMatrix::data_pointer_for_dlpack() const {
  return data_pointer_for_cuda_array_interface();
}

int DeviceCommutationMatrix::dlpack_device_type() const {
  if (!impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }
  return 2;
}

std::uint8_t* DeviceCommutationMatrix::mutable_data_for_device_write() {
  if (!impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }
  return impl_->data;
}

void copy_device_commutation_matrix_from_host_for_testing(
    DeviceCommutationMatrix& matrix,
    std::span<const std::uint8_t> values) {
  if (!matrix.impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }
  if (values.size() != matrix.impl_->entries) {
    throw std::invalid_argument(
        "DeviceCommutationMatrix testing copy size does not match matrix entries");
  }
  if (values.empty()) {
    return;
  }
  ScopedCudaDevice guard(matrix.impl_->device_ordinal);
  copy_to_device(
      matrix.impl_->data,
      values.data(),
      values.size(),
      "DeviceCommutationMatrix testing copy");
}

namespace cuda::benchmark {

FusedConflictDegreesResult fused_conflict_degrees(
    const DeviceCommutationMatrix& matrix,
    bool include_outputs) {
  if (matrix.rows() == 0 || matrix.cols() == 0) {
    return {};
  }

  const std::size_t rows = matrix.rows();
  const std::size_t cols = matrix.cols();
  const auto* data = detail::DeviceCommutationMatrixAccess::data(matrix);
  ScopedCudaDevice guard(matrix.device());

  const PoolBuffer<std::uint64_t> row_buffer(rows, "device row conflict counts");
  const PoolBuffer<std::uint64_t> col_buffer(cols, "device column conflict counts");
  std::uint64_t* device_row_conflicts = row_buffer.get();
  std::uint64_t* device_col_conflicts = col_buffer.get();

  count_row_conflicts_kernel<<<checked_reduction_blocks(rows, "CUDA row conflicts"), kCountThreadsPerBlock>>>(
      data,
      rows,
      cols,
      device_row_conflicts);
  check_cuda(cudaGetLastError(), "launch device commutation row conflict counts");
  count_col_conflicts_kernel<<<checked_reduction_blocks(cols, "CUDA column conflicts"), kCountThreadsPerBlock>>>(
      data,
      rows,
      cols,
      device_col_conflicts);
  check_cuda(cudaGetLastError(), "launch device commutation column conflict counts");

  FusedConflictDegreesResult result;
  result.row_conflicts.assign(rows, 0);
  result.col_conflicts.assign(cols, 0);
  copy_to_host(
      result.row_conflicts.data(),
      device_row_conflicts,
      result.row_conflicts.size(),
      "device row conflict counts");
  copy_to_host(
      result.col_conflicts.data(),
      device_col_conflicts,
      result.col_conflicts.size(),
      "device column conflict counts");
  result.row_conflict_sum = sum_uint64_values(result.row_conflicts);
  result.col_conflict_sum = sum_uint64_values(result.col_conflicts);
  if (!include_outputs) {
    result.row_conflicts.clear();
    result.col_conflicts.clear();
  }
  return result;
}

FusedCsrGraphResult fused_anticommutation_csr(
    const DeviceCommutationMatrix& matrix,
    bool include_outputs) {
  if (matrix.rows() == 0 || matrix.cols() == 0) {
    FusedCsrGraphResult result;
    result.row_offsets = {0};
    result.row_offset_checksum = checksum_uint64_values(result.row_offsets);
    if (!include_outputs) {
      result.row_offsets.clear();
    }
    return result;
  }

  const std::size_t rows = matrix.rows();
  const std::size_t cols = matrix.cols();
  const auto* data = detail::DeviceCommutationMatrixAccess::data(matrix);
  ScopedCudaDevice guard(matrix.device());

  const PoolBuffer<std::uint64_t> row_counts_buffer(rows, "device CSR row conflict counts");
  const PoolBuffer<std::uint64_t> row_offsets_buffer(rows + 1U, "device CSR row offsets");
  std::uint64_t* device_row_counts = row_counts_buffer.get();
  std::uint64_t* device_row_offsets = row_offsets_buffer.get();

  count_row_conflicts_kernel<<<checked_reduction_blocks(rows, "CUDA CSR row conflicts"), kCountThreadsPerBlock>>>(
      data,
      rows,
      cols,
      device_row_counts);
  check_cuda(cudaGetLastError(), "launch device CSR row conflict counts");

  thrust::exclusive_scan(
      thrust::device,
      thrust::device_pointer_cast(device_row_counts),
      thrust::device_pointer_cast(device_row_counts + rows),
      thrust::device_pointer_cast(device_row_offsets));
  check_cuda(cudaGetLastError(), "exclusive scan device CSR row offsets");
  set_final_csr_offset_kernel<<<1, 1>>>(device_row_counts, rows, device_row_offsets);
  check_cuda(cudaGetLastError(), "set final device CSR row offset");

  FusedCsrGraphResult result;
  result.row_offsets.assign(rows + 1U, 0);
  copy_to_host(
      result.row_offsets.data(),
      device_row_offsets,
      result.row_offsets.size(),
      "device CSR row offsets");
  result.edge_count = result.row_offsets.back();
  result.row_offset_checksum = checksum_uint64_values(result.row_offsets);

  if (result.edge_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error("CSR anti-commutation edge count exceeds host size_t range");
  }
  const std::size_t edge_count = static_cast<std::size_t>(result.edge_count);
  if (edge_count != 0) {
    const PoolBuffer<std::uint64_t> col_indices_buffer(edge_count, "device CSR column indices");
    std::uint64_t* device_col_indices = col_indices_buffer.get();
    scatter_csr_conflicts_sorted_by_row_kernel<<<checked_reduction_blocks(rows, "CUDA CSR scatter"), kCsrThreadsPerBlock>>>(
        data,
        rows,
        cols,
        device_row_offsets,
        device_col_indices);
    check_cuda(cudaGetLastError(), "launch device CSR column scatter");
    result.col_indices.assign(edge_count, 0);
    copy_to_host(
        result.col_indices.data(),
        device_col_indices,
        result.col_indices.size(),
        "device CSR column indices");
    result.col_index_checksum = checksum_uint64_values(result.col_indices);
  } else {
    result.col_index_checksum = checksum_uint64_values(result.col_indices);
  }

  if (!include_outputs) {
    result.row_offsets.clear();
    result.col_indices.clear();
  }
  return result;
}

FusedGroupingSummaryResult fused_grouping_summary(
    const DeviceCommutationMatrix& matrix,
    std::size_t top_k,
    bool include_outputs) {
  FusedConflictDegreesResult degrees = fused_conflict_degrees(matrix, true);
  FusedGroupingSummaryResult result;
  result.row_conflict_sum = degrees.row_conflict_sum;
  result.col_conflict_sum = degrees.col_conflict_sum;
  result.top_row_indices = top_indices_by_value(degrees.row_conflicts, top_k);
  result.top_col_indices = top_indices_by_value(degrees.col_conflicts, top_k);
  result.top_row_conflicts = gather_values_by_index(degrees.row_conflicts, result.top_row_indices);
  result.top_col_conflicts = gather_values_by_index(degrees.col_conflicts, result.top_col_indices);
  if (include_outputs) {
    result.row_conflicts = std::move(degrees.row_conflicts);
    result.col_conflicts = std::move(degrees.col_conflicts);
  }
  return result;
}

}  // namespace cuda::benchmark

}  // namespace wolfgang

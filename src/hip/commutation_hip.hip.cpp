#include "commutation_hip.hip.hpp"

#include "device_commutation_matrix.hip.hpp"
#include "device_pauli_sum.hip.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace wolfgang {

using hip_detail::check_hip;
using hip_detail::checked_launch_blocks;
using hip_detail::copy_to_host;
using hip_detail::hip_allocate;
using hip_detail::kHipThreadsPerBlock;
using hip_detail::ScopedHipDevice;

namespace {

__global__ void commutation_kernel(
    const std::uint64_t* lhs_x,
    const std::uint64_t* lhs_z,
    const std::uint64_t* rhs_x,
    const std::uint64_t* rhs_z,
    std::size_t lhs_terms,
    std::size_t rhs_terms,
    std::size_t words,
    std::uint8_t* out) {
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
  for (std::size_t entry =
           static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
           static_cast<std::size_t>(threadIdx.x);
       entry < lhs_terms * rhs_terms;
       entry += stride) {
    const std::size_t lhs_term = entry / rhs_terms;
    const std::size_t rhs_term = entry - lhs_term * rhs_terms;
    const std::size_t lhs_offset = lhs_term * words;
    const std::size_t rhs_offset = rhs_term * words;
    unsigned int parity = 0;
    for (std::size_t word = 0; word < words; ++word) {
      const std::uint64_t anti_commuting_bits =
          (lhs_x[lhs_offset + word] & rhs_z[rhs_offset + word]) ^
          (lhs_z[lhs_offset + word] & rhs_x[rhs_offset + word]);
      parity ^= static_cast<unsigned int>(__popcll(anti_commuting_bits)) & 1U;
    }
    out[entry] = parity == 0 ? 1U : 0U;
  }
}

void validate_commutation_operands(
    int lhs_device,
    int rhs_device,
    std::size_t lhs_num_qubits,
    std::size_t rhs_num_qubits) {
  validate_same_accelerator_context(
      "HIP commutes_with",
      {AcceleratorBackend::Hip, lhs_device},
      {AcceleratorBackend::Hip, rhs_device});
  if (lhs_num_qubits != rhs_num_qubits) {
    throw std::invalid_argument("PauliSum commutes_with requires the same num_qubits");
  }
}

}  // namespace

std::vector<std::uint8_t> DevicePauliSum::commutes_with(
    const DevicePauliSum& rhs,
    std::size_t max_commutation_matrix_entries) const {
  if (!impl_ || !rhs.impl_) {
    throw std::runtime_error("DevicePauliSum is empty or moved-from");
  }
  const std::size_t entries = detail::checked_commutation_matrix_entries(
      impl_->num_terms,
      rhs.impl_->num_terms,
      max_commutation_matrix_entries);
  std::vector<std::uint8_t> host_output(entries);
  commutes_with_into(rhs, std::span<std::uint8_t>(host_output), max_commutation_matrix_entries);
  return host_output;
}

void DevicePauliSum::commutes_with_into(
    const DevicePauliSum& rhs,
    std::span<std::uint8_t> output,
    std::size_t max_commutation_matrix_entries) const {
  if (!impl_ || !rhs.impl_) {
    throw std::runtime_error("DevicePauliSum is empty or moved-from");
  }
  validate_commutation_operands(
      impl_->device_ordinal,
      rhs.impl_->device_ordinal,
      impl_->num_qubits,
      rhs.impl_->num_qubits);
  const std::size_t entries = detail::checked_commutation_matrix_entries(
      impl_->num_terms,
      rhs.impl_->num_terms,
      max_commutation_matrix_entries);
  if (output.size() != entries) {
    throw std::invalid_argument("HIP commutes_with output buffer size does not match entry count");
  }
  if (entries == 0) {
    return;
  }

  const int blocks = checked_launch_blocks(entries, "HIP commutes_with");
  ScopedHipDevice guard(impl_->device_ordinal);
  std::uint8_t* device_output = nullptr;
  std::unique_ptr<std::uint8_t, void (*)(std::uint8_t*)> output_guard(
      nullptr,
      [](std::uint8_t* ptr) {
        if (ptr != nullptr) {
          (void)hipFree(ptr);
        }
      });
  hip_allocate(device_output, entries, "HIP commutation output");
  output_guard.reset(device_output);

  commutation_kernel<<<blocks, kHipThreadsPerBlock>>>(
      impl_->x,
      impl_->z,
      rhs.impl_->x,
      rhs.impl_->z,
      impl_->num_terms,
      rhs.impl_->num_terms,
      impl_->words,
      device_output);
  check_hip(hipGetLastError(), "launch commutation");
  check_hip(hipDeviceSynchronize(), "synchronize commutation");
  copy_to_host(output.data(), device_output, entries, "HIP commutation output");
}

DeviceCommutationMatrix DevicePauliSum::commutes_with_device(
    const DevicePauliSum& rhs,
    std::size_t max_commutation_matrix_entries) const {
  if (!impl_ || !rhs.impl_) {
    throw std::runtime_error("DevicePauliSum is empty or moved-from");
  }
  validate_commutation_operands(
      impl_->device_ordinal,
      rhs.impl_->device_ordinal,
      impl_->num_qubits,
      rhs.impl_->num_qubits);

  (void)detail::checked_commutation_matrix_entries(
      impl_->num_terms,
      rhs.impl_->num_terms,
      max_commutation_matrix_entries);
  DeviceCommutationMatrix output = DeviceCommutationMatrix::empty(
      impl_->num_terms,
      rhs.impl_->num_terms,
      impl_->device_ordinal);
  commutes_with_device_into(rhs, output, max_commutation_matrix_entries);
  return output;
}

void DevicePauliSum::commutes_with_device_into(
    const DevicePauliSum& rhs,
    DeviceCommutationMatrix& output,
    std::size_t max_commutation_matrix_entries) const {
  if (!impl_ || !rhs.impl_) {
    throw std::runtime_error("DevicePauliSum is empty or moved-from");
  }
  if (!output.impl_) {
    throw std::runtime_error("DeviceCommutationMatrix is empty or moved-from");
  }
  validate_same_accelerator_context(
      "HIP commutes_with_device",
      {AcceleratorBackend::Hip, impl_->device_ordinal},
      {AcceleratorBackend::Hip, rhs.impl_->device_ordinal});
  validate_same_accelerator_context(
      "HIP commutes_with_device output",
      {AcceleratorBackend::Hip, impl_->device_ordinal},
      {AcceleratorBackend::Hip, output.impl_->device_ordinal});
  if (impl_->num_qubits != rhs.impl_->num_qubits) {
    throw std::invalid_argument("PauliSum commutes_with requires the same num_qubits");
  }

  const std::size_t entries = detail::checked_commutation_matrix_entries(
      impl_->num_terms,
      rhs.impl_->num_terms,
      max_commutation_matrix_entries);
  if (output.impl_->rows != impl_->num_terms || output.impl_->cols != rhs.impl_->num_terms ||
      output.impl_->entries != entries) {
    throw std::invalid_argument(
        "HIP commutes_with_device output shape does not match operand term counts");
  }
  if (entries == 0) {
    return;
  }

  const int blocks = checked_launch_blocks(entries, "HIP commutes_with_device");
  ScopedHipDevice guard(impl_->device_ordinal);
  commutation_kernel<<<blocks, kHipThreadsPerBlock>>>(
      impl_->x,
      impl_->z,
      rhs.impl_->x,
      rhs.impl_->z,
      impl_->num_terms,
      rhs.impl_->num_terms,
      impl_->words,
      output.mutable_data_for_device_write());
  check_hip(hipGetLastError(), "launch commutation device output");
  check_hip(hipDeviceSynchronize(), "synchronize commutation device output");
}

}  // namespace wolfgang

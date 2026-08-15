#pragma once

#include "wolfgang/accelerator_status.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace wolfgang {

class DevicePauliSum;

class DeviceCommutationMatrix {
public:
  DeviceCommutationMatrix() noexcept;
  ~DeviceCommutationMatrix();

  DeviceCommutationMatrix(DeviceCommutationMatrix&& other) noexcept;
  DeviceCommutationMatrix& operator=(DeviceCommutationMatrix&& other) noexcept;

  DeviceCommutationMatrix(const DeviceCommutationMatrix&) = delete;
  DeviceCommutationMatrix& operator=(const DeviceCommutationMatrix&) = delete;

  [[nodiscard]] static DeviceCommutationMatrix empty(
      std::size_t rows,
      std::size_t cols,
      int device = 0);
  [[nodiscard]] static DeviceCommutationMatrix empty(
      std::size_t rows,
      std::size_t cols,
      AcceleratorBackend backend,
      int device = 0);

  [[nodiscard]] std::uint64_t count_commuting() const;
  [[nodiscard]] std::vector<std::uint64_t> count_commuting_rows() const;
  [[nodiscard]] std::vector<std::uint64_t> count_commuting_cols() const;
  [[nodiscard]] std::vector<std::uint8_t> to_host() const;
  [[nodiscard]] std::size_t rows() const noexcept;
  [[nodiscard]] std::size_t cols() const noexcept;
  [[nodiscard]] std::size_t num_entries() const noexcept;
  [[nodiscard]] int device() const noexcept;
  [[nodiscard]] std::string backend() const;
  [[nodiscard]] std::uintptr_t data_pointer_for_cuda_array_interface() const;
  [[nodiscard]] std::uintptr_t data_pointer_for_dlpack() const;
  [[nodiscard]] int dlpack_device_type() const;

private:
  struct Impl;

  explicit DeviceCommutationMatrix(std::unique_ptr<Impl> impl) noexcept;
  [[nodiscard]] std::uint8_t* mutable_data_for_device_write();

  std::unique_ptr<Impl> impl_;

  friend class DevicePauliSum;
  friend void copy_device_commutation_matrix_from_host_for_testing(
      DeviceCommutationMatrix& matrix,
      std::span<const std::uint8_t> values);
};

void copy_device_commutation_matrix_from_host_for_testing(
    DeviceCommutationMatrix& matrix,
    std::span<const std::uint8_t> values);

}  // namespace wolfgang



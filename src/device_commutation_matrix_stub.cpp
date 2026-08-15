#include "wolfgang/device_commutation_matrix.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace wolfgang {

namespace {

const std::string kAcceleratorNotBuiltMessage =
    accelerator_not_built_message(AcceleratorBackend::None);

}  // namespace

struct DeviceCommutationMatrix::Impl {};

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
    std::size_t,
    std::size_t,
    AcceleratorBackend backend,
    int) {
  (void)select_accelerator_backend(backend, false, false, false, false, false, false);
  throw std::runtime_error(kAcceleratorNotBuiltMessage);
}

std::uint64_t DeviceCommutationMatrix::count_commuting() const {
  throw std::runtime_error(kAcceleratorNotBuiltMessage);
}

std::vector<std::uint64_t> DeviceCommutationMatrix::count_commuting_rows() const {
  throw std::runtime_error(kAcceleratorNotBuiltMessage);
}

std::vector<std::uint64_t> DeviceCommutationMatrix::count_commuting_cols() const {
  throw std::runtime_error(kAcceleratorNotBuiltMessage);
}

std::vector<std::uint8_t> DeviceCommutationMatrix::to_host() const {
  throw std::runtime_error(kAcceleratorNotBuiltMessage);
}

std::size_t DeviceCommutationMatrix::rows() const noexcept {
  return 0;
}

std::size_t DeviceCommutationMatrix::cols() const noexcept {
  return 0;
}

std::size_t DeviceCommutationMatrix::num_entries() const noexcept {
  return 0;
}

int DeviceCommutationMatrix::device() const noexcept {
  return -1;
}

std::string DeviceCommutationMatrix::backend() const {
  return std::string(accelerator_backend_name(AcceleratorBackend::None));
}

std::uintptr_t DeviceCommutationMatrix::data_pointer_for_cuda_array_interface() const {
  throw std::runtime_error(kAcceleratorNotBuiltMessage);
}

std::uintptr_t DeviceCommutationMatrix::data_pointer_for_dlpack() const {
  throw std::runtime_error(kAcceleratorNotBuiltMessage);
}

int DeviceCommutationMatrix::dlpack_device_type() const {
  throw std::runtime_error(kAcceleratorNotBuiltMessage);
}

std::uint8_t* DeviceCommutationMatrix::mutable_data_for_device_write() {
  throw std::runtime_error(kAcceleratorNotBuiltMessage);
}

void copy_device_commutation_matrix_from_host_for_testing(
    DeviceCommutationMatrix&,
    std::span<const std::uint8_t>) {
  throw std::runtime_error(kAcceleratorNotBuiltMessage);
}

}  // namespace wolfgang

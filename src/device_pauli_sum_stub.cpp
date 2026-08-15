#include "wolfgang/device_pauli_sum.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace wolfgang {

namespace {

const std::string kCudaNotBuiltMessage =
    accelerator_not_built_message(AcceleratorBackend::Cuda);
const std::string kHipNotBuiltMessage =
    accelerator_not_built_message(AcceleratorBackend::Hip);
const std::string kMetalNotBuiltMessage =
    accelerator_not_built_message(AcceleratorBackend::Metal);
const std::string kAcceleratorNotBuiltMessage =
    accelerator_not_built_message(AcceleratorBackend::None);

}  // namespace

struct DevicePauliSum::Impl {};

DevicePauliSum::DevicePauliSum() noexcept = default;

DevicePauliSum::DevicePauliSum(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

DevicePauliSum::~DevicePauliSum() = default;

DevicePauliSum::DevicePauliSum(DevicePauliSum&& other) noexcept = default;

DevicePauliSum& DevicePauliSum::operator=(DevicePauliSum&& other) noexcept = default;

DevicePauliSum DevicePauliSum::from_host(const PauliSum& host, int device) {
  return from_host(host, AcceleratorBackend::None, device);
}

DevicePauliSum DevicePauliSum::from_host(
    const PauliSum&,
    AcceleratorBackend backend,
    int) {
  (void)select_accelerator_backend(backend, false, false, false, false, false, false);
  throw std::runtime_error(kAcceleratorNotBuiltMessage);
}

CudaStatus DevicePauliSum::cuda_status() {
  CudaStatus status;
  status.skip_reason = kCudaNotBuiltMessage;
  return status;
}

bool DevicePauliSum::cuda_available() {
  return false;
}

HipStatus DevicePauliSum::hip_status() {
  HipStatus status;
  status.skip_reason = kHipNotBuiltMessage;
  return status;
}

bool DevicePauliSum::hip_available() {
  return false;
}

MetalStatus DevicePauliSum::metal_status() {
  MetalStatus status;
  status.skip_reason = kMetalNotBuiltMessage;
  return status;
}

bool DevicePauliSum::metal_available() {
  return false;
}

PauliSum DevicePauliSum::to_host() const {
  throw std::runtime_error(kAcceleratorNotBuiltMessage);
}

DevicePauliSum DevicePauliSum::simplify(double, double) const {
  throw std::runtime_error(kAcceleratorNotBuiltMessage);
}

std::complex<double> DevicePauliSum::expectation_statevector_complex128(
    std::span<const std::complex<double>>) const {
  throw std::runtime_error(kAcceleratorNotBuiltMessage);
}

std::complex<double> DevicePauliSum::expectation_statevector_complex64(
    std::span<const std::complex<float>>) const {
  throw std::runtime_error(kAcceleratorNotBuiltMessage);
}

std::complex<double> DevicePauliSum::expectation_statevector_device_pointer(
    std::uintptr_t,
    DeviceStatevectorDtype,
    std::size_t) const {
  throw std::runtime_error(kAcceleratorNotBuiltMessage);
}

std::vector<std::uint8_t> DevicePauliSum::commutes_with(
    const DevicePauliSum&,
    std::size_t) const {
  throw std::runtime_error(kAcceleratorNotBuiltMessage);
}

void DevicePauliSum::commutes_with_into(
    const DevicePauliSum&,
    std::span<std::uint8_t>,
    std::size_t) const {
  throw std::runtime_error(kAcceleratorNotBuiltMessage);
}

DeviceCommutationMatrix DevicePauliSum::commutes_with_device(
    const DevicePauliSum&,
    std::size_t) const {
  throw std::runtime_error(kAcceleratorNotBuiltMessage);
}

void DevicePauliSum::commutes_with_device_into(
    const DevicePauliSum&,
    DeviceCommutationMatrix&,
    std::size_t) const {
  throw std::runtime_error(kAcceleratorNotBuiltMessage);
}

DevicePauliSum DevicePauliSum::matmul(
    const DevicePauliSum&,
    bool,
    std::size_t) const {
  throw std::runtime_error(kAcceleratorNotBuiltMessage);
}

std::size_t DevicePauliSum::num_qubits() const noexcept {
  return 0;
}

std::size_t DevicePauliSum::num_terms() const noexcept {
  return 0;
}

std::size_t DevicePauliSum::words() const noexcept {
  return 0;
}

int DevicePauliSum::device() const noexcept {
  return -1;
}

std::string DevicePauliSum::backend() const {
  return std::string(accelerator_backend_name(AcceleratorBackend::None));
}

}  // namespace wolfgang

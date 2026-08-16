#include "device_pauli_sum.hip.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace wolfgang {

namespace {

std::string hip_version_string(int version) {
  if (version <= 0) {
    return "unknown";
  }
  if (version >= 10000000) {
    const int major = version / 10000000;
    const int minor = (version / 100000) % 100;
    const int patch = version % 100000;
    return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
  }
  if (version >= 1000000) {
    const int major = version / 1000000;
    const int minor = (version / 10000) % 100;
    const int patch = version % 10000;
    return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
  }
  const int major = version / 1000;
  const int minor = (version % 1000) / 10;
  return std::to_string(major) + "." + std::to_string(minor);
}

HipStatus unavailable_status(hipError_t result) {
  HipStatus status;
  status.built = true;
  if (result == hipErrorNoDevice) {
    status.skip_reason = "no HIP device is available";
  } else {
    status.skip_reason =
        std::string("HIP runtime library is unavailable or failed during device discovery: ") +
        hipGetErrorString(result);
  }
  (void)hipGetLastError();
  return status;
}

std::vector<thrust::complex<double>> to_device_coefficients(
    const std::vector<std::complex<double>>& coeffs) {
  std::vector<thrust::complex<double>> converted;
  converted.reserve(coeffs.size());
  for (const std::complex<double>& coeff : coeffs) {
    converted.emplace_back(coeff.real(), coeff.imag());
  }
  return converted;
}

std::vector<std::complex<double>> to_host_coefficients(
    const std::vector<thrust::complex<double>>& coeffs) {
  std::vector<std::complex<double>> converted;
  converted.reserve(coeffs.size());
  for (const thrust::complex<double>& coeff : coeffs) {
    converted.emplace_back(coeff.real(), coeff.imag());
  }
  return converted;
}

void validate_device_ordinal_for_transfer(int device) {
  int device_count = 0;
  const hipError_t count_result = hipGetDeviceCount(&device_count);
  if (count_result != hipSuccess) {
    throw std::runtime_error(unavailable_status(count_result).skip_reason);
  }
  if (device_count == 0) {
    throw std::runtime_error("no HIP device is available");
  }
  if (device < 0 || device >= device_count) {
    throw std::invalid_argument("HIP device ordinal is out of range");
  }
}

}  // namespace

DevicePauliSum::Impl::~Impl() {
  if (x == nullptr && z == nullptr && coeffs == nullptr) {
    return;
  }
  int previous_device = -1;
  const hipError_t current_result = hipGetDevice(&previous_device);
  (void)hipSetDevice(device_ordinal);
  (void)hipFree(x);
  (void)hipFree(z);
  (void)hipFree(coeffs);
  if (current_result == hipSuccess && previous_device >= 0 && previous_device != device_ordinal) {
    (void)hipSetDevice(previous_device);
  }
}

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
    const PauliSum& host,
    AcceleratorBackend backend,
    int device) {
  const HipStatus status = hip_status();
  const AcceleratorBackend selected = select_accelerator_backend(
      backend,
      false,
      false,
      true,
      status.runtime_available,
      false,
      false);
  if (selected != AcceleratorBackend::Hip) {
    throw std::runtime_error(accelerator_not_built_message(selected));
  }
  validate_device_ordinal_for_transfer(device);

  auto impl = std::make_unique<Impl>();
  impl->num_qubits = host.num_qubits();
  impl->words = host.words();
  impl->num_terms = host.num_terms();
  impl->device_ordinal = device;

  const std::size_t packed_words =
      detail::checked_product(host.num_terms(), host.words(), "HIP device packed words");
  const std::vector<thrust::complex<double>> coeffs = to_device_coefficients(host.coeffs());

  hip_detail::ScopedHipDevice guard(device);
  hip_detail::hip_allocate(impl->x, packed_words, "HIP device x words");
  hip_detail::hip_allocate(impl->z, packed_words, "HIP device z words");
  hip_detail::hip_allocate(impl->coeffs, host.num_terms(), "HIP device coefficients");
  hip_detail::copy_to_device(impl->x, host.x_words().data(), packed_words, "HIP device x words");
  hip_detail::copy_to_device(impl->z, host.z_words().data(), packed_words, "HIP device z words");
  hip_detail::copy_to_device(
      impl->coeffs,
      coeffs.data(),
      coeffs.size(),
      "HIP device coefficients");

  return DevicePauliSum(std::move(impl));
}

CudaStatus DevicePauliSum::cuda_status() {
  CudaStatus status;
  status.skip_reason =
      "Wolfgang was built without CUDA support; rebuild from source with "
      "WOLFGANG_ENABLE_CUDA=ON to use PauliSum.to_device().";
  return status;
}

bool DevicePauliSum::cuda_available() {
  return false;
}

HipStatus DevicePauliSum::hip_status() {
  int runtime_version = 0;
  int driver_version = 0;
  (void)hipRuntimeGetVersion(&runtime_version);
  (void)hipDriverGetVersion(&driver_version);

  int device_count = 0;
  const hipError_t count_result = hipGetDeviceCount(&device_count);
  if (count_result != hipSuccess) {
    HipStatus status = unavailable_status(count_result);
    status.runtime_version = hip_version_string(runtime_version);
    status.driver_version = hip_version_string(driver_version);
    status.toolkit_version = WOLFGANG_ROCM_TOOLKIT_VERSION;
    return status;
  }

  HipStatus status;
  status.built = true;
  status.runtime_available = device_count > 0;
  status.device_count = device_count;
  status.runtime_version = hip_version_string(runtime_version);
  status.driver_version = hip_version_string(driver_version);
  status.toolkit_version = WOLFGANG_ROCM_TOOLKIT_VERSION;
  if (device_count == 0) {
    status.skip_reason = "no HIP device is available";
    return status;
  }

  status.devices.reserve(static_cast<std::size_t>(device_count));
  for (int ordinal = 0; ordinal < device_count; ++ordinal) {
    hipDeviceProp_t properties{};
    const hipError_t prop_result = hipGetDeviceProperties(&properties, ordinal);
    if (prop_result != hipSuccess) {
      status.skip_reason =
          std::string("HIP runtime failed during device property discovery: ") +
          hipGetErrorString(prop_result);
      status.runtime_available = false;
      status.devices.clear();
      (void)hipGetLastError();
      return status;
    }

    HipDeviceInfo device;
    device.ordinal = ordinal;
    device.name = properties.name;
    device.gfx_target = properties.gcnArchName;
    device.total_memory_bytes = properties.totalGlobalMem;
    status.devices.push_back(std::move(device));
  }
  return status;
}

bool DevicePauliSum::hip_available() {
  return hip_status().runtime_available;
}

MetalStatus DevicePauliSum::metal_status() {
  MetalStatus status;
  status.skip_reason =
      "Wolfgang was built without Metal support; rebuild from source on Apple Silicon with "
      "WOLFGANG_ENABLE_METAL=ON to use PauliSum.to_device().";
  return status;
}

bool DevicePauliSum::metal_available() {
  return false;
}

PauliSum DevicePauliSum::to_host() const {
  if (!impl_) {
    throw std::runtime_error("DevicePauliSum is empty or moved-from");
  }

  const std::size_t packed_words =
      detail::checked_product(impl_->num_terms, impl_->words, "HIP host packed words");
  std::vector<std::uint64_t> x(packed_words);
  std::vector<std::uint64_t> z(packed_words);
  std::vector<thrust::complex<double>> device_coeffs(impl_->num_terms);

  hip_detail::ScopedHipDevice guard(impl_->device_ordinal);
  hip_detail::copy_to_host(x.data(), impl_->x, packed_words, "HIP host x words");
  hip_detail::copy_to_host(z.data(), impl_->z, packed_words, "HIP host z words");
  hip_detail::copy_to_host(
      device_coeffs.data(),
      impl_->coeffs,
      device_coeffs.size(),
      "HIP host coefficients");

  return PauliSum(
      impl_->num_qubits,
      impl_->words,
      impl_->num_terms,
      std::move(x),
      std::move(z),
      to_host_coefficients(device_coeffs));
}

std::size_t DevicePauliSum::num_qubits() const noexcept {
  return impl_ ? impl_->num_qubits : 0;
}

std::size_t DevicePauliSum::num_terms() const noexcept {
  return impl_ ? impl_->num_terms : 0;
}

std::size_t DevicePauliSum::words() const noexcept {
  return impl_ ? impl_->words : 0;
}

int DevicePauliSum::device() const noexcept {
  return impl_ ? impl_->device_ordinal : -1;
}

std::string DevicePauliSum::backend() const {
  return std::string(accelerator_backend_name(
      impl_ ? AcceleratorBackend::Hip : AcceleratorBackend::None));
}

}  // namespace wolfgang

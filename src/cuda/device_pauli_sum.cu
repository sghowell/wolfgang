#include "device_pauli_sum.cuh"
#include "cuda/pool_allocator.cuh"

#include <cuda_runtime_api.h>
#include <thrust/complex.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace wolfgang {

using cuda_detail::check_cuda;
using cuda_detail::checked_bytes;
using cuda_detail::copy_to_device;
using cuda_detail::copy_to_host;
using cuda_detail::pool_allocate;
using cuda_detail::copy_bytes_to_device;
using cuda_detail::pool_free_bytes_on_device;
using cuda_detail::ScopedCudaDevice;

// std::complex<double> and thrust::complex<double> both store two adjacent
// doubles, so coefficients cross the host/device boundary as raw bytes.
static_assert(sizeof(std::complex<double>) == sizeof(thrust::complex<double>));

namespace {

std::string cuda_version_string(int version) {
  if (version <= 0) {
    return "unknown";
  }
  const int major = version / 1000;
  const int minor = (version % 1000) / 10;
  return std::to_string(major) + "." + std::to_string(minor);
}

CudaStatus unavailable_status(cudaError_t result) {
  CudaStatus status;
  status.built = true;
  if (result == cudaErrorNoDevice) {
    status.skip_reason = "no CUDA device is available";
  } else {
    status.skip_reason =
        std::string("CUDA runtime library is unavailable or failed during device discovery: ") +
        cudaGetErrorString(result);
  }
  (void)cudaGetLastError();
  return status;
}

void validate_device_ordinal_for_transfer(int device) {
  int device_count = 0;
  const cudaError_t count_result = cudaGetDeviceCount(&device_count);
  if (count_result != cudaSuccess) {
    throw std::runtime_error(unavailable_status(count_result).skip_reason);
  }
  if (device_count == 0) {
    throw std::runtime_error("no CUDA device is available");
  }
  if (device < 0 || device >= device_count) {
    throw std::invalid_argument("CUDA device ordinal is out of range");
  }
}

}  // namespace

DevicePauliSum::Impl::~Impl() {
  if (x == nullptr && z == nullptr && coeffs == nullptr) {
    return;
  }
  // Buffers come from the Wolfgang pool on device_ordinal.
  pool_free_bytes_on_device(x, device_ordinal);
  pool_free_bytes_on_device(z, device_ordinal);
  pool_free_bytes_on_device(coeffs, device_ordinal);
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
  // A device count is enough to select the backend; the full cuda_status()
  // queries every device's properties and dominated small transfers.
  int device_count = 0;
  const cudaError_t count_result = cudaGetDeviceCount(&device_count);
  if (count_result != cudaSuccess) {
    (void)cudaGetLastError();
  }
  const bool runtime_available = count_result == cudaSuccess && device_count > 0;
  const AcceleratorBackend selected = select_accelerator_backend(
      backend,
      true,
      runtime_available,
      false,
      false,
      false,
      false);
  if (selected != AcceleratorBackend::Cuda) {
    throw std::runtime_error(accelerator_not_built_message(selected));
  }
  validate_device_ordinal_for_transfer(device);

  auto impl = std::make_unique<Impl>();
  impl->num_qubits = host.num_qubits();
  impl->words = host.words();
  impl->num_terms = host.num_terms();
  impl->device_ordinal = device;

  const std::size_t packed_words =
      detail::checked_product(host.num_terms(), host.words(), "device packed words");

  ScopedCudaDevice guard(device);
  pool_allocate(impl->x, packed_words, "device x words");
  pool_allocate(impl->z, packed_words, "device z words");
  pool_allocate(impl->coeffs, host.num_terms(), "device coefficients");
  copy_to_device(impl->x, host.x_words().data(), packed_words, "device x words");
  copy_to_device(impl->z, host.z_words().data(), packed_words, "device z words");
  copy_bytes_to_device(
      impl->coeffs,
      host.coeffs().data(),
      checked_bytes(host.num_terms(), sizeof(std::complex<double>), "device coefficients"),
      "device coefficients");

  return DevicePauliSum(std::move(impl));
}

CudaStatus DevicePauliSum::cuda_status() {
  int runtime_version = 0;
  int driver_version = 0;
  (void)cudaRuntimeGetVersion(&runtime_version);
  (void)cudaDriverGetVersion(&driver_version);

  int device_count = 0;
  const cudaError_t count_result = cudaGetDeviceCount(&device_count);
  if (count_result != cudaSuccess) {
    CudaStatus status = unavailable_status(count_result);
    status.runtime_version = cuda_version_string(runtime_version);
    status.driver_version = cuda_version_string(driver_version);
    return status;
  }

  CudaStatus status;
  status.built = true;
  status.runtime_available = device_count > 0;
  status.device_count = device_count;
  status.runtime_version = cuda_version_string(runtime_version);
  status.driver_version = cuda_version_string(driver_version);
  if (device_count == 0) {
    status.skip_reason = "no CUDA device is available";
    return status;
  }

  status.devices.reserve(static_cast<std::size_t>(device_count));
  for (int ordinal = 0; ordinal < device_count; ++ordinal) {
    cudaDeviceProp properties{};
    const cudaError_t prop_result = cudaGetDeviceProperties(&properties, ordinal);
    if (prop_result != cudaSuccess) {
      status.skip_reason =
          std::string("CUDA runtime failed during device property discovery: ") +
          cudaGetErrorString(prop_result);
      status.runtime_available = false;
      status.devices.clear();
      (void)cudaGetLastError();
      return status;
    }

    CudaDeviceInfo device;
    device.ordinal = ordinal;
    device.name = properties.name;
    device.compute_capability_major = properties.major;
    device.compute_capability_minor = properties.minor;
    device.total_memory_bytes = properties.totalGlobalMem;
    status.devices.push_back(std::move(device));
  }
  return status;
}

bool DevicePauliSum::cuda_available() {
  return cuda_status().runtime_available;
}

HipStatus DevicePauliSum::hip_status() {
  HipStatus status;
  status.skip_reason =
      "Wolfgang was built without HIP support; rebuild from source with "
      "WOLFGANG_ENABLE_HIP=ON to use PauliSum.to_device().";
  return status;
}

bool DevicePauliSum::hip_available() {
  return false;
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
      detail::checked_product(impl_->num_terms, impl_->words, "host packed words");
  std::vector<std::uint64_t> x(packed_words);
  std::vector<std::uint64_t> z(packed_words);
  std::vector<std::complex<double>> coeffs(impl_->num_terms);

  ScopedCudaDevice guard(impl_->device_ordinal);
  copy_to_host(x.data(), impl_->x, packed_words, "host x words");
  copy_to_host(z.data(), impl_->z, packed_words, "host z words");
  if (!coeffs.empty()) {
    check_cuda(
        cudaMemcpy(
            coeffs.data(),
            impl_->coeffs,
            checked_bytes(coeffs.size(), sizeof(std::complex<double>), "host coefficients"),
            cudaMemcpyDeviceToHost),
        "copy device buffer to host");
  }

  return PauliSum(
      impl_->num_qubits,
      impl_->words,
      impl_->num_terms,
      std::move(x),
      std::move(z),
      std::move(coeffs));
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
      impl_ ? AcceleratorBackend::Cuda : AcceleratorBackend::None));
}

}  // namespace wolfgang

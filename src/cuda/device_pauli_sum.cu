#include "device_pauli_sum.cuh"
#include "cuda/pool_allocator.cuh"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/complex.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/count.h>

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
using cuda_detail::pooled_policy;
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

// True when a term's last word has bits at or above num_qubits in x or z.
struct TailBitsSet {
  const std::uint64_t* x;
  const std::uint64_t* z;
  std::size_t words;
  std::uint64_t mask;

  __host__ __device__ bool operator()(std::size_t term) const noexcept {
    const std::size_t index = term * words + words - 1;
    return ((x[index] | z[index]) & mask) != 0;
  }
};

// The device that owns `pointer`, after checking it is device (or managed)
// memory whose allocation covers `bytes`.
int validated_import_device(std::uintptr_t pointer, std::size_t bytes, const char* name) {
  cudaPointerAttributes attributes{};
  if (cudaPointerGetAttributes(&attributes, reinterpret_cast<const void*>(pointer)) != cudaSuccess) {
    (void)cudaGetLastError();
    throw std::invalid_argument(std::string(name) + " is not recognized CUDA memory");
  }
  if (attributes.type != cudaMemoryTypeDevice && attributes.type != cudaMemoryTypeManaged) {
    throw std::invalid_argument(std::string(name) + " must point to CUDA device memory");
  }
  if (attributes.device < 0) {
    throw std::invalid_argument(std::string(name) + " has no owning CUDA device");
  }
  // Query the extent in the owning device's context, as the statevector path
  // does; the current context may belong to another device or not exist yet.
  ScopedCudaDevice guard(attributes.device);
  CUdeviceptr base = 0;
  std::size_t allocation_bytes = 0;
  if (cuMemGetAddressRange(&base, &allocation_bytes, static_cast<CUdeviceptr>(pointer)) != CUDA_SUCCESS) {
    (void)cudaGetLastError();
    throw std::invalid_argument(std::string(name) + " allocation extent could not be determined");
  }
  const std::uintptr_t offset = pointer - static_cast<std::uintptr_t>(base);
  if (offset > allocation_bytes || bytes > allocation_bytes - offset) {
    throw std::invalid_argument(std::string(name) + " byte range exceeds its backing allocation");
  }
  return attributes.device;
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
  if (exported) {
    int previous = -1;
    (void)cudaGetDevice(&previous);
    if (cudaSetDevice(device_ordinal) == cudaSuccess) {
      (void)cudaDeviceSynchronize();
    }
    if (previous >= 0) {
      (void)cudaSetDevice(previous);
    }
    (void)cudaGetLastError();
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

DeviceArrayPointers DevicePauliSum::export_device_arrays() const {
  if (!impl_) {
    throw std::runtime_error("DevicePauliSum is empty or moved-from");
  }
  impl_->exported = true;
  // A pageable host-to-device cudaMemcpy (to_device) can return before its DMA
  // lands; finish the legacy stream so a consumer on any stream sees the data.
  {
    ScopedCudaDevice guard(impl_->device_ordinal);
    check_cuda(cudaStreamSynchronize(cudaStreamLegacy), "finish work before exporting device arrays");
  }
  DeviceArrayPointers pointers;
  pointers.x = reinterpret_cast<std::uintptr_t>(impl_->x);
  pointers.z = reinterpret_cast<std::uintptr_t>(impl_->z);
  pointers.coeffs = reinterpret_cast<std::uintptr_t>(impl_->coeffs);
  pointers.dlpack_device_type = 2;  // kDLCUDA
  pointers.device = impl_->device_ordinal;
  return pointers;
}

DevicePauliSum DevicePauliSum::from_device_arrays(
    std::size_t num_qubits,
    std::size_t num_terms,
    std::uintptr_t x,
    std::uintptr_t z,
    std::uintptr_t coeffs,
    AcceleratorBackend backend,
    int device) {
  int device_count = 0;
  const cudaError_t count_result = cudaGetDeviceCount(&device_count);
  if (count_result != cudaSuccess) {
    (void)cudaGetLastError();
  }
  const AcceleratorBackend selected = select_accelerator_backend(
      backend, true, count_result == cudaSuccess && device_count > 0, false, false, false, false);
  if (selected != AcceleratorBackend::Cuda) {
    throw std::runtime_error(accelerator_not_built_message(selected));
  }
  if (device < -1) {
    throw std::invalid_argument("device must be -1 or a CUDA device ordinal");
  }
  const std::size_t words = detail::checked_word_count(num_qubits);
  const std::size_t packed_words =
      detail::checked_product(num_terms, words, "device array packed words");
  const std::size_t word_bytes = checked_bytes(packed_words, sizeof(std::uint64_t), "device array words");
  const std::size_t coeff_bytes =
      checked_bytes(num_terms, sizeof(thrust::complex<double>), "device array coefficients");
  if (num_terms > 0) {
    if (coeffs == 0 || (words > 0 && (x == 0 || z == 0))) {
      throw std::invalid_argument("device array pointers must be non-null");
    }
    const int coeff_device = validated_import_device(coeffs, coeff_bytes, "coefficients");
    if (words > 0) {
      const int x_device = validated_import_device(x, word_bytes, "x words");
      const int z_device = validated_import_device(z, word_bytes, "z words");
      if (x_device != coeff_device || z_device != coeff_device) {
        throw std::invalid_argument("device arrays must be on the same CUDA device");
      }
    }
    if (device >= 0 && device != coeff_device) {
      throw std::invalid_argument("device arrays are not on the requested CUDA device");
    }
    device = coeff_device;
  }
  if (device < 0) {
    check_cuda(cudaGetDevice(&device), "query the current CUDA device");
  }
  validate_device_ordinal_for_transfer(device);

  auto impl = std::make_unique<Impl>();
  impl->num_qubits = num_qubits;
  impl->words = words;
  impl->num_terms = num_terms;
  impl->device_ordinal = device;
  ScopedCudaDevice guard(device);
  // A CUDA Array Interface producer may have written the inputs on a
  // non-blocking stream, which does not order with the legacy stream used for
  // the copies below; wait for all of the device's work first.
  if (num_terms > 0) {
    check_cuda(cudaDeviceSynchronize(), "synchronize device array producers");
  }
  pool_allocate(impl->x, packed_words, "device x words");
  pool_allocate(impl->z, packed_words, "device z words");
  pool_allocate(impl->coeffs, num_terms, "device coefficients");
  if (word_bytes != 0) {
    check_cuda(cudaMemcpy(impl->x, reinterpret_cast<const void*>(x), word_bytes, cudaMemcpyDefault),
               "copy device x words");
    check_cuda(cudaMemcpy(impl->z, reinterpret_cast<const void*>(z), word_bytes, cudaMemcpyDefault),
               "copy device z words");
  }
  if (coeff_bytes != 0) {
    check_cuda(
        cudaMemcpy(impl->coeffs, reinterpret_cast<const void*>(coeffs), coeff_bytes, cudaMemcpyDefault),
        "copy device coefficients");
  }
  const std::size_t tail = num_qubits % 64;
  if (num_terms > 0 && tail != 0) {
    // count_if reduces a plain size_t; any_of's internal (bool, index) result
    // returns padding bytes that compute-sanitizer initcheck reports.
    const auto counting = thrust::make_counting_iterator<std::size_t>(0);
    const std::size_t bad = thrust::count_if(
        pooled_policy(),
        counting,
        counting + num_terms,
        TailBitsSet{impl->x, impl->z, words, ~((std::uint64_t{1} << tail) - 1)});
    if (bad != 0) {
      throw std::invalid_argument("packed words set bits at or above num_qubits");
    }
  }
  // Device-to-device cudaMemcpy returns before the copy completes. Finish it,
  // so the caller may reuse the inputs and any stream may read the result.
  check_cuda(cudaStreamSynchronize(cudaStreamLegacy), "finish device array copies");
  return DevicePauliSum(std::move(impl));
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

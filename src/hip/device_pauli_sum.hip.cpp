#include "device_pauli_sum.hip.hpp"
#include "hip/pool_allocator.hip.hpp"
#include "detail/staged_transfer.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace wolfgang {

// Transfers copy coefficients as raw bytes between the two layouts.
static_assert(sizeof(std::complex<double>) == sizeof(thrust::complex<double>));

namespace {

// Per-thread pinned buffer for device-to-host copies
// (detail/staged_transfer.hpp). It grows to the largest transfer seen, up to
// kPinnedStagingMaxBytes, and is freed when the thread exits. If pinned
// memory cannot be allocated, to_host falls back to pageable copies.
constexpr std::size_t kPinnedStagingMaxBytes = std::size_t{16} << 20;
constexpr std::size_t kHipStagedMaxBytes = std::size_t{256} << 10;

class ThreadPinnedStaging {
 public:
  ThreadPinnedStaging() = default;
  ThreadPinnedStaging(const ThreadPinnedStaging&) = delete;
  ThreadPinnedStaging& operator=(const ThreadPinnedStaging&) = delete;
  // The main thread's thread_local destructors run before the HIP runtime's
  // exit handlers; any error from a runtime already unloading is ignored.
  ~ThreadPinnedStaging() {
    if (pointer_ != nullptr) {
      (void)hipHostFree(pointer_);
    }
  }

  // At least `bytes` of pinned memory, or nullptr when it cannot be had.
  std::byte* reserve(std::size_t bytes) noexcept {
    if (bytes > capacity_) {
      if (pointer_ != nullptr) {
        (void)hipHostFree(pointer_);
        pointer_ = nullptr;
        capacity_ = 0;
      }
      void* pointer = nullptr;
      if (hipHostMalloc(&pointer, bytes, hipHostMallocPortable) != hipSuccess) {
        (void)hipGetLastError();
        return nullptr;
      }
      pointer_ = static_cast<std::byte*>(pointer);
      capacity_ = bytes;
    }
    return pointer_;
  }
  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

 private:
  std::byte* pointer_ = nullptr;
  std::size_t capacity_ = 0;
};

ThreadPinnedStaging& thread_pinned_staging() {
  thread_local ThreadPinnedStaging staging;
  return staging;
}

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
  // Buffers come from the Wolfgang pool on device_ordinal.
  hip_detail::pool_free_bytes_on_device(x, device_ordinal);
  hip_detail::pool_free_bytes_on_device(z, device_ordinal);
  hip_detail::pool_free_bytes_on_device(coeffs, device_ordinal);
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
  // A device count is enough to select the backend; the full hip_status()
  // queries every device's properties on each transfer.
  int device_count = 0;
  const hipError_t count_result = hipGetDeviceCount(&device_count);
  if (count_result != hipSuccess) {
    (void)hipGetLastError();
  }
  const AcceleratorBackend selected = select_accelerator_backend(
      backend,
      false,
      false,
      true,
      count_result == hipSuccess && device_count > 0,
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
  hip_detail::ScopedHipDevice guard(device);
  hip_detail::pool_allocate(impl->x, packed_words, "HIP device x words");
  hip_detail::pool_allocate(impl->z, packed_words, "HIP device z words");
  hip_detail::pool_allocate(impl->coeffs, host.num_terms(), "HIP device coefficients");
  hip_detail::copy_to_device(impl->x, host.x_words().data(), packed_words, "HIP device x words");
  hip_detail::copy_to_device(impl->z, host.z_words().data(), packed_words, "HIP device z words");
  // std::complex<double> and thrust::complex<double> share layout, so the
  // coefficients copy as raw bytes with no host-side conversion buffer.
  hip_detail::copy_bytes_to_device(
      impl->coeffs,
      host.coeffs().data(),
      hip_detail::checked_bytes(host.num_terms(), sizeof(std::complex<double>), "HIP device coefficients"),
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

DeviceArrayPointers DevicePauliSum::export_device_arrays() const {
  throw std::runtime_error(
      "DevicePauliSum DLPack export is CUDA-only; no ROCm/HIP consumer has been validated for it yet (docs/architecture/rocm_backend.md).");
}

DevicePauliSum DevicePauliSum::from_device_arrays(
    std::size_t,
    std::size_t,
    std::uintptr_t,
    std::uintptr_t,
    std::uintptr_t,
    AcceleratorBackend,
    int) {
  throw std::runtime_error("DevicePauliSum.from_device_arrays is CUDA-only; HIP builds do not import device arrays yet.");
}

PauliSum DevicePauliSum::to_host() const {
  if (!impl_) {
    throw std::runtime_error("DevicePauliSum is empty or moved-from");
  }

  namespace acc = wolfgang::accelerator_detail;
  const std::size_t packed_words =
      detail::checked_product(impl_->num_terms, impl_->words, "HIP host packed words");
  const std::size_t total_bytes =
      2 * hip_detail::checked_bytes(packed_words, sizeof(std::uint64_t), "HIP host packed words") +
      hip_detail::checked_bytes(impl_->num_terms, sizeof(std::complex<double>), "HIP host coefficients");
  std::vector<std::uint64_t> x;
  std::vector<std::uint64_t> z;
  std::vector<std::complex<double>> coeffs;

  hip_detail::ScopedHipDevice guard(impl_->device_ordinal);
  // On the MI300X, ROCm's pageable copies beat staging from about 2 MB, so only
  // small operators stage, where the gain is one wait instead of three; larger
  // destinations are copied directly, with huge pages requested first.
  ThreadPinnedStaging& staging = thread_pinned_staging();
  std::byte* staged = total_bytes == 0 || total_bytes > kHipStagedMaxBytes
      ? nullptr
      : staging.reserve(acc::staging_bytes_for(total_bytes, kPinnedStagingMaxBytes));
  if (staged != nullptr) {
    // Same layout on both sides: copy raw bytes, no conversion buffer.
    const std::vector<acc::DeviceToHostSegment> segments{
        acc::device_to_host_segment(x, impl_->x, packed_words),
        acc::device_to_host_segment(z, impl_->z, packed_words),
        acc::device_to_host_segment(
            coeffs, reinterpret_cast<const std::complex<double>*>(impl_->coeffs), impl_->num_terms)};
    acc::staged_device_to_host(
        staged,
        staging.capacity(),
        segments,
        [](std::byte* dst, const std::byte* src, std::size_t bytes) {
          hip_detail::check_hip(
              hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, nullptr), "copy device buffer to host");
        },
        [] { hip_detail::check_hip(hipStreamSynchronize(nullptr), "synchronize device-to-host copy"); });
  } else if (total_bytes != 0) {
    acc::resize_for_direct_copy(x, packed_words);
    acc::resize_for_direct_copy(z, packed_words);
    acc::resize_for_direct_copy(coeffs, impl_->num_terms);
    hip_detail::copy_to_host(x.data(), impl_->x, packed_words, "HIP host x words");
    hip_detail::copy_to_host(z.data(), impl_->z, packed_words, "HIP host z words");
    hip_detail::check_hip(
        hipMemcpy(coeffs.data(), impl_->coeffs, coeffs.size() * sizeof(std::complex<double>), hipMemcpyDeviceToHost),
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
      impl_ ? AcceleratorBackend::Hip : AcceleratorBackend::None));
}

}  // namespace wolfgang

#include "device_pauli_sum_metal.hpp"

#include <complex>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace wolfgang {

namespace {

std::string objcxx_compiler_version() {
  return std::string(WOLFGANG_CMAKE_OBJCXX_COMPILER_ID) + " " +
         std::string(WOLFGANG_CMAKE_OBJCXX_COMPILER_VERSION);
}

std::string bool_text(bool value) {
  return value ? "true" : "false";
}

MetalDeviceInfo metal_device_info(id<MTLDevice> device) {
  MetalDeviceInfo info;
  info.ordinal = 0;
  info.name = metal_detail::nsstring_to_string([device name]);
  info.registry_id = static_cast<std::uint64_t>([device registryID]);
  info.recommended_max_working_set_size =
      static_cast<std::size_t>([device recommendedMaxWorkingSetSize]);
  info.low_power = [device isLowPower];
  info.headless = [device isHeadless];
  info.removable = [device isRemovable];
  if ([device respondsToSelector:@selector(hasUnifiedMemory)]) {
    info.unified_memory = [device hasUnifiedMemory];
  }
  info.capability_summary =
      "unified_memory=" + bool_text(info.unified_memory) +
      "; low_power=" + bool_text(info.low_power) +
      "; headless=" + bool_text(info.headless) +
      "; removable=" + bool_text(info.removable) +
      "; recommended_max_working_set_size_bytes=" +
      std::to_string(info.recommended_max_working_set_size);
  return info;
}

}  // namespace

DevicePauliSum::Impl::~Impl() {
  [x release];
  [z release];
  [coeffs release];
  [command_queue release];
  [device release];
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
  @autoreleasepool {
    const MetalStatus status = metal_status();
    const AcceleratorBackend selected = select_accelerator_backend(
        backend,
        false,
        false,
        false,
        false,
        true,
        status.runtime_available);
    if (selected != AcceleratorBackend::Metal) {
      throw std::runtime_error(accelerator_not_built_message(selected));
    }
    metal_detail::validate_device_ordinal(device);

    auto impl = std::make_unique<Impl>();
    impl->num_qubits = host.num_qubits();
    impl->words = host.words();
    impl->num_terms = host.num_terms();
    impl->device_ordinal = device;
    impl->device = metal_detail::create_default_device_or_throw();
    impl->command_queue = metal_detail::make_command_queue(impl->device);

    const std::size_t packed_words =
        detail::checked_product(host.num_terms(), host.words(), "Metal device packed words");
    const std::size_t packed_bytes =
        metal_detail::checked_bytes<std::uint64_t>(packed_words, "Metal packed word bytes");
    const std::size_t coeff_bytes =
        metal_detail::checked_bytes<std::complex<double>>(
            host.coeffs().size(),
            "Metal coefficient bytes");

    impl->x = metal_detail::make_shared_buffer(
        impl->device,
        host.x_words().data(),
        packed_bytes,
        "x words");
    impl->z = metal_detail::make_shared_buffer(
        impl->device,
        host.z_words().data(),
        packed_bytes,
        "z words");
    impl->coeffs = metal_detail::make_shared_buffer(
        impl->device,
        host.coeffs().data(),
        coeff_bytes,
        "coefficients");

    return DevicePauliSum(std::move(impl));
  }
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
  @autoreleasepool {
    MetalStatus status;
    status.built = true;
    status.storage_mode = "MTLResourceStorageModeShared";
    status.xcode_or_clt_version = objcxx_compiler_version();
    status.macos_version = metal_detail::nsstring_to_string(
        [[NSProcessInfo processInfo] operatingSystemVersionString]);

    id<MTLDevice> device = metal_detail::create_default_device();
    if (device == nil) {
      status.skip_reason = "no Metal device is available";
      return status;
    }

    status.runtime_available = true;
    status.device_count = 1;
    status.devices.push_back(metal_device_info(device));
    status.metal_device_name = status.devices.front().name;
    status.capability_summary = status.devices.front().capability_summary;
    [device release];
    return status;
  }
}

bool DevicePauliSum::metal_available() {
  return metal_status().runtime_available;
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

  const std::size_t packed_bytes =
      metal_detail::checked_bytes<std::uint64_t>(packed_words, "host packed word bytes");
  const std::size_t coeff_bytes =
      metal_detail::checked_bytes<std::complex<double>>(coeffs.size(), "host coefficient bytes");
  if (packed_bytes != 0) {
    std::memcpy(x.data(), [impl_->x contents], packed_bytes);
    std::memcpy(z.data(), [impl_->z contents], packed_bytes);
  }
  if (coeff_bytes != 0) {
    std::memcpy(coeffs.data(), [impl_->coeffs contents], coeff_bytes);
  }

  return PauliSum(
      impl_->num_qubits,
      impl_->words,
      impl_->num_terms,
      std::move(x),
      std::move(z),
      std::move(coeffs));
}

DevicePauliSum DevicePauliSum::simplify(double atol, double rtol) const {
  if (!impl_) {
    throw std::runtime_error("DevicePauliSum is empty or moved-from");
  }
  metal_detail::validate_simplify_tolerances(atol, rtol);

  PauliSum simplified = to_host().simplify(atol, rtol);
  return DevicePauliSum::from_host(
      simplified,
      AcceleratorBackend::Metal,
      impl_->device_ordinal);
}

std::complex<double> DevicePauliSum::expectation_statevector_complex128(
    std::span<const std::complex<double>>) const {
  metal_detail::throw_unsupported_operation("expectation_statevector");
}

std::complex<double> DevicePauliSum::expectation_statevector_complex64(
    std::span<const std::complex<float>>) const {
  metal_detail::throw_unsupported_operation("expectation_statevector");
}

std::complex<double> DevicePauliSum::expectation_statevector_device_pointer(
    std::uintptr_t,
    DeviceStatevectorDtype,
    std::size_t) const {
  metal_detail::throw_unsupported_operation("expectation_statevector device pointers");
}

DevicePauliSum DevicePauliSum::matmul(
    const DevicePauliSum&,
    bool,
    std::size_t) const {
  metal_detail::throw_unsupported_operation("matmul");
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
      impl_ ? AcceleratorBackend::Metal : AcceleratorBackend::None));
}

}  // namespace wolfgang

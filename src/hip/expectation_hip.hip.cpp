#include "device_pauli_sum.hip.hpp"

#include <hip/hip_runtime.h>
#include <thrust/complex.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace wolfgang {

using hip_detail::check_hip;
using hip_detail::checked_bytes;
using hip_detail::copy_bytes_to_device;
using hip_detail::copy_to_host;
using hip_detail::hip_allocate;
using hip_detail::ScopedHipDevice;
using hip_detail::validate_statevector_length;

namespace {

constexpr int kExpectationDefaultThreadsPerBlock = 256;
constexpr int kExpectationLargeStateThreadsPerBlock = 128;
constexpr std::size_t kExpectationLargeStateThreshold = std::size_t{1} << 17U;

// Host NumPy buffers are copied byte-for-byte into hipMalloc storage and read
// by kernels as thrust::complex<T>. These guards keep that zero-conversion path
// tied to the storage assumptions used by the CUDA implementation.
static_assert(
    std::is_trivially_copyable_v<std::complex<double>>,
    "std::complex<double> must be trivially copyable for HIP byte-copy statevectors");
static_assert(
    std::is_trivially_copyable_v<std::complex<float>>,
    "std::complex<float> must be trivially copyable for HIP byte-copy statevectors");
static_assert(
    std::is_trivially_copyable_v<thrust::complex<double>>,
    "thrust::complex<double> must be trivially copyable for HIP byte-copy statevectors");
static_assert(
    std::is_trivially_copyable_v<thrust::complex<float>>,
    "thrust::complex<float> must be trivially copyable for HIP byte-copy statevectors");
static_assert(
    sizeof(std::complex<double>) == sizeof(thrust::complex<double>),
    "std::complex<double> and thrust::complex<double> must have identical storage size");
static_assert(
    sizeof(std::complex<float>) == sizeof(thrust::complex<float>),
    "std::complex<float> and thrust::complex<float> must have identical storage size");

__device__ thrust::complex<double> phase_from_exponent_device(std::int64_t exponent) {
  switch (static_cast<std::uint64_t>(exponent) & 3U) {
    case 0:
      return {1.0, 0.0};
    case 1:
      return {0.0, 1.0};
    case 2:
      return {-1.0, 0.0};
    default:
      return {0.0, -1.0};
  }
}

__device__ thrust::complex<double> to_complex_double(thrust::complex<double> value) {
  return value;
}

__device__ thrust::complex<double> to_complex_double(thrust::complex<float> value) {
  return {static_cast<double>(value.real()), static_cast<double>(value.imag())};
}

template <typename StateComplex>
__global__ void expectation_statevector_terms_kernel(
    const std::uint64_t* x,
    const std::uint64_t* z,
    const thrust::complex<double>* coeffs,
    std::size_t num_terms,
    std::size_t state_size,
    const StateComplex* psi,
    double* result_parts) {
  const std::size_t term = static_cast<std::size_t>(blockIdx.x);
  if (term >= num_terms) {
    return;
  }

  const std::uint64_t x_mask = x == nullptr ? 0 : x[term];
  const std::uint64_t z_mask = z == nullptr ? 0 : z[term];
  const thrust::complex<double> yz_phase =
      phase_from_exponent_device(static_cast<std::int64_t>(__popcll(x_mask & z_mask)));

  extern __shared__ unsigned char raw_shared[];
  auto* partials = reinterpret_cast<thrust::complex<double>*>(raw_shared);
  thrust::complex<double> partial{0.0, 0.0};
  for (std::size_t basis = static_cast<std::size_t>(threadIdx.x);
       basis < state_size;
       basis += static_cast<std::size_t>(blockDim.x)) {
    const std::size_t target = basis ^ static_cast<std::size_t>(x_mask);
    const bool z_parity = (__popcll(z_mask & static_cast<std::uint64_t>(basis)) & 1U) != 0;
    const thrust::complex<double> phase = z_parity ? -yz_phase : yz_phase;
    partial += thrust::conj(to_complex_double(psi[target])) * phase * to_complex_double(psi[basis]);
  }

  partials[threadIdx.x] = partial;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1U) {
    if (threadIdx.x < stride) {
      partials[threadIdx.x] += partials[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const thrust::complex<double> weighted = coeffs[term] * partials[0];
    atomicAdd(result_parts, weighted.real());
    atomicAdd(result_parts + 1, weighted.imag());
  }
}

template <typename T>
std::unique_ptr<T, void (*)(T*)> hip_allocation_guard(T* ptr) {
  return std::unique_ptr<T, void (*)(T*)>(
      ptr,
      [](T* value) {
        if (value != nullptr) {
          (void)hipFree(value);
        }
      });
}

}  // namespace

std::complex<double> DevicePauliSum::expectation_statevector_complex128(
    std::span<const std::complex<double>> psi) const {
  if (!impl_) {
    throw std::runtime_error("DevicePauliSum is empty or moved-from");
  }
  validate_statevector_length(impl_->num_qubits, psi.size());
  ScopedHipDevice guard(impl_->device_ordinal);

  thrust::complex<double>* device_psi = nullptr;
  hip_allocate(device_psi, psi.size(), "HIP device statevector");
  auto psi_guard = hip_allocation_guard(device_psi);
  copy_bytes_to_device(
      device_psi,
      psi.data(),
      checked_bytes(psi.size(), sizeof(thrust::complex<double>), "HIP device statevector"),
      "HIP device statevector");
  return expectation_statevector_device_pointer_impl(
      reinterpret_cast<std::uintptr_t>(device_psi),
      DeviceStatevectorDtype::Complex128,
      psi.size(),
      false);
}

std::complex<double> DevicePauliSum::expectation_statevector_complex64(
    std::span<const std::complex<float>> psi) const {
  if (!impl_) {
    throw std::runtime_error("DevicePauliSum is empty or moved-from");
  }
  validate_statevector_length(impl_->num_qubits, psi.size());
  ScopedHipDevice guard(impl_->device_ordinal);

  thrust::complex<float>* device_psi = nullptr;
  hip_allocate(device_psi, psi.size(), "HIP device statevector");
  auto psi_guard = hip_allocation_guard(device_psi);
  copy_bytes_to_device(
      device_psi,
      psi.data(),
      checked_bytes(psi.size(), sizeof(thrust::complex<float>), "HIP device statevector"),
      "HIP device statevector");
  return expectation_statevector_device_pointer_impl(
      reinterpret_cast<std::uintptr_t>(device_psi),
      DeviceStatevectorDtype::Complex64,
      psi.size(),
      false);
}

std::complex<double> DevicePauliSum::expectation_statevector_device_pointer(
    std::uintptr_t,
    DeviceStatevectorDtype,
    std::size_t) const {
  throw std::runtime_error(
      "HIP expectation_statevector does not accept external device pointers; "
      "ROCm/HIP statevector interop requires a separate ownership and stream contract");
}

std::complex<double> DevicePauliSum::expectation_statevector_device_pointer_impl(
    std::uintptr_t device_pointer,
    DeviceStatevectorDtype dtype,
    std::size_t length,
    bool synchronize_input) const {
  if (!impl_) {
    throw std::runtime_error("DevicePauliSum is empty or moved-from");
  }
  validate_statevector_length(impl_->num_qubits, length);
  if (impl_->words > 1) {
    throw std::invalid_argument("HIP expectation_statevector requires num_qubits <= 63");
  }
  if (device_pointer == 0) {
    throw std::invalid_argument("HIP statevector data pointer must be non-null");
  }
  if (impl_->num_terms == 0) {
    return {0.0, 0.0};
  }

  ScopedHipDevice guard(impl_->device_ordinal);
  if (synchronize_input) {
    check_hip(hipDeviceSynchronize(), "synchronize before HIP statevector expectation");
  }

  double* result_parts = nullptr;
  hip_allocate(result_parts, 2, "HIP expectation result accumulator");
  auto result_guard = hip_allocation_guard(result_parts);
  check_hip(
      hipMemset(result_parts, 0, 2 * sizeof(double)),
      "initialize HIP statevector expectation accumulator");

  const int threads_per_block = length >= kExpectationLargeStateThreshold
      ? kExpectationLargeStateThreadsPerBlock
      : kExpectationDefaultThreadsPerBlock;
  const std::size_t shared_bytes =
      static_cast<std::size_t>(threads_per_block) * sizeof(thrust::complex<double>);
  if (dtype == DeviceStatevectorDtype::Complex128) {
    expectation_statevector_terms_kernel<<<
        static_cast<unsigned int>(impl_->num_terms),
        threads_per_block,
        shared_bytes>>>(
        impl_->x,
        impl_->z,
        impl_->coeffs,
        impl_->num_terms,
        length,
        reinterpret_cast<const thrust::complex<double>*>(device_pointer),
        result_parts);
  } else {
    expectation_statevector_terms_kernel<<<
        static_cast<unsigned int>(impl_->num_terms),
        threads_per_block,
        shared_bytes>>>(
        impl_->x,
        impl_->z,
        impl_->coeffs,
        impl_->num_terms,
        length,
        reinterpret_cast<const thrust::complex<float>*>(device_pointer),
        result_parts);
  }
  check_hip(hipGetLastError(), "launch HIP statevector expectation");

  double host_parts[2] = {0.0, 0.0};
  copy_to_host(host_parts, result_parts, 2, "HIP expectation result accumulator");
  return {host_parts[0], host_parts[1]};
}

}  // namespace wolfgang

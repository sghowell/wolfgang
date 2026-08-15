#include "device_pauli_sum.cuh"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/complex.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace wolfgang {

using cuda_detail::check_cuda;
using cuda_detail::checked_bytes;
using cuda_detail::copy_bytes_to_device;
using cuda_detail::copy_to_host;
using cuda_detail::cuda_allocate;
using cuda_detail::kCudaThreadsPerBlock;
using cuda_detail::ScopedCudaDevice;
using cuda_detail::validate_statevector_length;

namespace {

constexpr int kExpectationDefaultThreadsPerBlock = 256;
constexpr int kExpectationLargeStateThreadsPerBlock = 128;
constexpr std::size_t kExpectationLargeStateThreshold = std::size_t{1} << 17U;

// The host-statevector fast path copies bytes into cudaMalloc storage that
// kernels read as thrust::complex<T>. C++ guarantees array-oriented
// std::complex<T> storage as adjacent real/imag scalars; these guards catch
// representation drift that would invalidate the no-temporary conversion path
// on supported toolchains. Type alignment is intentionally not compared:
// cudaMalloc supplies the destination alignment, and the host source is read
// only by cudaMemcpy as bytes.
static_assert(
    std::is_trivially_copyable_v<std::complex<double>>,
    "std::complex<double> must be trivially copyable for CUDA byte-copy statevectors");
static_assert(
    std::is_trivially_copyable_v<std::complex<float>>,
    "std::complex<float> must be trivially copyable for CUDA byte-copy statevectors");
static_assert(
    std::is_trivially_copyable_v<thrust::complex<double>>,
    "thrust::complex<double> must be trivially copyable for CUDA byte-copy statevectors");
static_assert(
    std::is_trivially_copyable_v<thrust::complex<float>>,
    "thrust::complex<float> must be trivially copyable for CUDA byte-copy statevectors");
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
    // One atomic complex accumulation avoids the temporary term array and the
    // follow-up Thrust reduction. The public result tolerance already permits
    // parallel floating-point reduction order for CUDA expectation.
    atomicAdd(result_parts, weighted.real());
    atomicAdd(result_parts + 1, weighted.imag());
  }
}

}  // namespace

std::complex<double> DevicePauliSum::expectation_statevector_complex128(
    std::span<const std::complex<double>> psi) const {
  if (!impl_) {
    throw std::runtime_error("DevicePauliSum is empty or moved-from");
  }
  validate_statevector_length(impl_->num_qubits, psi.size());
  ScopedCudaDevice guard(impl_->device_ordinal);

  thrust::complex<double>* device_psi = nullptr;
  cuda_allocate(device_psi, psi.size(), "device statevector");
  std::unique_ptr<thrust::complex<double>, void (*)(thrust::complex<double>*)> psi_guard(
      device_psi,
      [](thrust::complex<double>* ptr) {
        if (ptr != nullptr) {
          (void)cudaFree(ptr);
        }
      });
  copy_bytes_to_device(
      device_psi,
      psi.data(),
      checked_bytes(psi.size(), sizeof(thrust::complex<double>), "device statevector"),
      "device statevector");
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
  ScopedCudaDevice guard(impl_->device_ordinal);

  thrust::complex<float>* device_psi = nullptr;
  cuda_allocate(device_psi, psi.size(), "device statevector");
  std::unique_ptr<thrust::complex<float>, void (*)(thrust::complex<float>*)> psi_guard(
      device_psi,
      [](thrust::complex<float>* ptr) {
        if (ptr != nullptr) {
          (void)cudaFree(ptr);
        }
      });
  copy_bytes_to_device(
      device_psi,
      psi.data(),
      checked_bytes(psi.size(), sizeof(thrust::complex<float>), "device statevector"),
      "device statevector");
  return expectation_statevector_device_pointer_impl(
      reinterpret_cast<std::uintptr_t>(device_psi),
      DeviceStatevectorDtype::Complex64,
      psi.size(),
      false);
}

std::complex<double> DevicePauliSum::expectation_statevector_device_pointer(
    std::uintptr_t device_pointer,
    DeviceStatevectorDtype dtype,
    std::size_t length) const {
  return expectation_statevector_device_pointer_impl(device_pointer, dtype, length, true);
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
    throw std::invalid_argument("CUDA expectation_statevector requires num_qubits <= 63");
  }
  if (device_pointer == 0) {
    throw std::invalid_argument("CUDA statevector data pointer must be non-null");
  }

  ScopedCudaDevice guard(impl_->device_ordinal);
  cudaPointerAttributes attributes{};
  const cudaError_t attr_result =
      cudaPointerGetAttributes(&attributes, reinterpret_cast<const void*>(device_pointer));
  if (attr_result != cudaSuccess) {
    (void)cudaGetLastError();
    throw std::invalid_argument("CUDA statevector data pointer is not recognized device memory");
  }
  if (attributes.type != cudaMemoryTypeDevice) {
    throw std::invalid_argument("CUDA statevector must point to device memory");
  }
  if (attributes.device != impl_->device_ordinal) {
    throw std::invalid_argument("CUDA statevector must be on the same CUDA device as DevicePauliSum");
  }

  const std::size_t item_size = dtype == DeviceStatevectorDtype::Complex128
      ? sizeof(thrust::complex<double>)
      : sizeof(thrust::complex<float>);
  const std::size_t required_bytes =
      checked_bytes(length, item_size, "CUDA statevector allocation extent");
  CUdeviceptr allocation_base = 0;
  std::size_t allocation_bytes = 0;
  const CUresult range_result = cuMemGetAddressRange(
      &allocation_base,
      &allocation_bytes,
      static_cast<CUdeviceptr>(device_pointer));
  if (range_result != CUDA_SUCCESS) {
    (void)cudaGetLastError();
    throw std::invalid_argument(
        "CUDA statevector allocation extent could not be determined");
  }
  const auto base_address = static_cast<std::uintptr_t>(allocation_base);
  if (device_pointer < base_address) {
    throw std::invalid_argument("CUDA statevector data pointer is outside its allocation");
  }
  const std::uintptr_t byte_offset = device_pointer - base_address;
  if (byte_offset > allocation_bytes || required_bytes > allocation_bytes - byte_offset) {
    throw std::invalid_argument(
        "CUDA statevector byte range exceeds its backing allocation");
  }
  if (impl_->num_terms == 0) {
    return {0.0, 0.0};
  }

  if (synchronize_input) {
    // A CUDA-array-interface producer may have used a non-default stream. A device
    // sync makes FastPauli's first interop path conservative and deterministic.
    check_cuda(cudaDeviceSynchronize(), "synchronize before CUDA statevector expectation");
  }

  double* result_parts = nullptr;
  cuda_allocate(result_parts, 2, "expectation result accumulator");
  std::unique_ptr<double, void (*)(double*)> result_guard(
      result_parts,
      [](double* ptr) {
        if (ptr != nullptr) {
          (void)cudaFree(ptr);
        }
      });
  check_cuda(
      cudaMemset(result_parts, 0, 2 * sizeof(double)),
      "initialize CUDA statevector expectation accumulator");
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
  check_cuda(cudaGetLastError(), "launch CUDA statevector expectation");

  double host_parts[2] = {0.0, 0.0};
  copy_to_host(host_parts, result_parts, 2, "expectation result accumulator");
  return {host_parts[0], host_parts[1]};
}

}  // namespace wolfgang

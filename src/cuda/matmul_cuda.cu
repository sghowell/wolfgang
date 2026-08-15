#include "device_pauli_sum.cuh"

#include <cuda_runtime_api.h>
#include <thrust/complex.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace wolfgang {

using cuda_detail::check_cuda;
using cuda_detail::cuda_allocate;
using cuda_detail::checked_launch_blocks;
using cuda_detail::kCudaThreadsPerBlock;
using cuda_detail::ScopedCudaDevice;

namespace {

__device__ thrust::complex<double> multiply_by_phase_exponent_device(
    thrust::complex<double> value,
    std::int64_t exponent) {
  switch (static_cast<std::uint64_t>(exponent) & 3U) {
    case 0:
      return value;
    case 1:
      return {-value.imag(), value.real()};
    case 2:
      return {-value.real(), -value.imag()};
    default:
      return {value.imag(), -value.real()};
  }
}

__global__ void matmul_product_kernel(
    const std::uint64_t* lhs_x,
    const std::uint64_t* lhs_z,
    const thrust::complex<double>* lhs_coeffs,
    const std::uint64_t* rhs_x,
    const std::uint64_t* rhs_z,
    const thrust::complex<double>* rhs_coeffs,
    std::size_t lhs_terms,
    std::size_t rhs_terms,
    std::size_t words,
    std::uint64_t* out_x,
    std::uint64_t* out_z,
    thrust::complex<double>* out_coeffs) {
  const std::size_t output_term =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + static_cast<std::size_t>(threadIdx.x);
  const std::size_t total = lhs_terms * rhs_terms;
  if (output_term >= total) {
    return;
  }

  const std::size_t lhs_term = output_term / rhs_terms;
  const std::size_t rhs_term = output_term - lhs_term * rhs_terms;
  const std::size_t lhs_offset = lhs_term * words;
  const std::size_t rhs_offset = rhs_term * words;
  const std::size_t out_offset = output_term * words;

  std::int64_t lhs_y = 0;
  std::int64_t rhs_y = 0;
  std::int64_t out_y = 0;
  std::int64_t lhs_rhs_cross = 0;
  for (std::size_t word = 0; word < words; ++word) {
    const std::uint64_t lhs_word_x = lhs_x[lhs_offset + word];
    const std::uint64_t lhs_word_z = lhs_z[lhs_offset + word];
    const std::uint64_t rhs_word_x = rhs_x[rhs_offset + word];
    const std::uint64_t rhs_word_z = rhs_z[rhs_offset + word];
    const std::uint64_t out_word_x = lhs_word_x ^ rhs_word_x;
    const std::uint64_t out_word_z = lhs_word_z ^ rhs_word_z;

    out_x[out_offset + word] = out_word_x;
    out_z[out_offset + word] = out_word_z;

    lhs_y += static_cast<std::int64_t>(__popcll(lhs_word_x & lhs_word_z));
    rhs_y += static_cast<std::int64_t>(__popcll(rhs_word_x & rhs_word_z));
    out_y += static_cast<std::int64_t>(__popcll(out_word_x & out_word_z));
    lhs_rhs_cross += static_cast<std::int64_t>(__popcll(lhs_word_x & rhs_word_z));
  }

  const std::int64_t phase_exponent = out_y - lhs_y - rhs_y + 2 * lhs_rhs_cross;
  out_coeffs[output_term] = multiply_by_phase_exponent_device(
      lhs_coeffs[lhs_term] * rhs_coeffs[rhs_term],
      phase_exponent);
}

}  // namespace

DevicePauliSum DevicePauliSum::matmul(
    const DevicePauliSum& rhs,
    bool simplify_output,
    std::size_t max_intermediate_terms) const {
  if (!impl_ || !rhs.impl_) {
    throw std::runtime_error("DevicePauliSum is empty or moved-from");
  }
  validate_same_accelerator_context(
      "CUDA matmul",
      {AcceleratorBackend::Cuda, impl_->device_ordinal},
      {AcceleratorBackend::Cuda, rhs.impl_->device_ordinal});
  if (impl_->num_qubits != rhs.impl_->num_qubits) {
    throw std::invalid_argument("PauliSum matmul requires the same num_qubits");
  }

  const std::size_t out_terms = detail::checked_matmul_intermediate_terms(
      impl_->num_terms,
      rhs.impl_->num_terms,
      max_intermediate_terms);
  const int blocks = out_terms == 0 ? 0 : checked_launch_blocks(out_terms, "CUDA matmul");
  auto out = std::make_unique<Impl>();
  out->num_qubits = impl_->num_qubits;
  out->words = impl_->words;
  out->num_terms = out_terms;
  out->device_ordinal = impl_->device_ordinal;
  if (out_terms == 0) {
    return DevicePauliSum(std::move(out));
  }

  ScopedCudaDevice guard(impl_->device_ordinal);
  const std::size_t packed_words =
      detail::checked_product(out_terms, impl_->words, "matmul packed words");
  cuda_allocate(out->x, packed_words, "matmul x words");
  cuda_allocate(out->z, packed_words, "matmul z words");
  cuda_allocate(out->coeffs, out_terms, "matmul coefficients");
  matmul_product_kernel<<<blocks, kCudaThreadsPerBlock>>>(
      impl_->x,
      impl_->z,
      impl_->coeffs,
      rhs.impl_->x,
      rhs.impl_->z,
      rhs.impl_->coeffs,
      impl_->num_terms,
      rhs.impl_->num_terms,
      impl_->words,
      out->x,
      out->z,
      out->coeffs);
  check_cuda(cudaGetLastError(), "launch CUDA matmul");

  DevicePauliSum product(std::move(out));
  if (simplify_output) {
    return product.simplify();
  }
  check_cuda(cudaDeviceSynchronize(), "synchronize after CUDA matmul");
  return product;
}

}  // namespace wolfgang

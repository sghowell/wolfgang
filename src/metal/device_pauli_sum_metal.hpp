#pragma once

#include "wolfgang/device_pauli_sum.hpp"

#include "detail/checked_arithmetic.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace wolfgang {

struct DevicePauliSum::Impl {
  std::size_t num_qubits = 0;
  std::size_t words = 0;
  std::size_t num_terms = 0;
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> command_queue = nil;
  id<MTLBuffer> x = nil;
  id<MTLBuffer> z = nil;
  id<MTLBuffer> coeffs = nil;
  int device_ordinal = 0;

  ~Impl();
};

namespace metal_detail {

constexpr NSUInteger kThreadsPerThreadgroup = 256;
constexpr NSUInteger kCommutationThreadgroupX = 16;
constexpr NSUInteger kCommutationThreadgroupY = 16;

[[nodiscard]] std::string nsstring_to_string(NSString* value);
[[nodiscard]] id<MTLDevice> create_default_device();
[[nodiscard]] id<MTLDevice> create_default_device_or_throw();
void validate_device_ordinal(int device);
[[nodiscard]] id<MTLCommandQueue> make_command_queue(id<MTLDevice> device);
[[nodiscard]] id<MTLBuffer> make_shared_buffer(
    id<MTLDevice> device,
    const void* source,
    std::size_t bytes,
    const char* name);
[[nodiscard]] id<MTLBuffer> make_private_buffer(
    id<MTLDevice> device,
    std::size_t bytes,
    const char* name);
[[nodiscard]] id<MTLComputePipelineState> make_compute_pipeline(
    id<MTLDevice> device,
    NSString* kernel_name,
    NSString* source);
[[nodiscard]] id<MTLComputePipelineState> make_compute_pipeline_from_metallib(
    id<MTLDevice> device,
    NSString* kernel_name,
    const std::string& library_path);
void wait_for_completion(id<MTLCommandBuffer> command_buffer, const char* action);
[[noreturn]] void throw_unsupported_operation(const char* operation);

inline void validate_simplify_tolerances(double atol, double rtol) {
  if (atol < 0.0 || rtol < 0.0 || !std::isfinite(atol) || !std::isfinite(rtol)) {
    throw std::invalid_argument("simplify tolerances must be non-negative finite values");
  }
}

template <typename T>
[[nodiscard]] std::size_t checked_bytes(std::size_t count, const char* name) {
  return detail::checked_product(count, sizeof(T), name);
}

}  // namespace metal_detail

}  // namespace wolfgang

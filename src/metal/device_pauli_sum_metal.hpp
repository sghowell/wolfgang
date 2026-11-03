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
#include <vector>

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

// Python and native worker threads need not provide an Objective-C pool. Keep
// temporary command buffers, encoders, and errors within each synchronous call,
// including exception unwinding; retained device objects outlive this scope.
class ScopedAutoreleasePool {
 public:
  ScopedAutoreleasePool();
  ~ScopedAutoreleasePool();
  ScopedAutoreleasePool(const ScopedAutoreleasePool&) = delete;
  ScopedAutoreleasePool& operator=(const ScopedAutoreleasePool&) = delete;

 private:
  void* pool_;
};

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

// Public Metal simplify runs this hybrid for one-word operators of at least
// kHybridSimplifyMinTerms terms: the GPU sorts (x, z, input index) and the CPU
// sums each duplicate run in float64, in input order, over the shared buffers.
// The result equals CPU simplify bit for bit; smaller inputs use the host bridge.
// On an M4 Pro the hybrid beat the bridge in every measured case from 12,288
// terms and broke even near 8,192, where its fixed command-buffer round trip
// (about 0.3 ms) dominates.
inline constexpr std::size_t kHybridSimplifyMinTerms = 12288;

// False when the hybrid's 512-thread tile kernels exceed the device's
// threadgroup limit; simplify then keeps the host bridge.
[[nodiscard]] bool hybrid_simplify_supported(id<MTLDevice> device);
// Owns +1 references to the simplified operator's buffers until the caller
// moves them into a DevicePauliSum::Impl. Buffers are nil when num_terms is 0
// and may be longer than num_terms entries.
struct HybridSimplifyOutput {
  id<MTLBuffer> x = nil;
  id<MTLBuffer> z = nil;
  id<MTLBuffer> coeffs = nil;
  std::size_t num_terms = 0;

  HybridSimplifyOutput() = default;
  HybridSimplifyOutput(const HybridSimplifyOutput&) = delete;
  HybridSimplifyOutput& operator=(const HybridSimplifyOutput&) = delete;
  HybridSimplifyOutput(HybridSimplifyOutput&& other) noexcept
      : x(other.x), z(other.z), coeffs(other.coeffs), num_terms(other.num_terms) {
    other.x = nil;
    other.z = nil;
    other.coeffs = nil;
  }
  HybridSimplifyOutput& operator=(HybridSimplifyOutput&&) = delete;
  ~HybridSimplifyOutput() {
    [x release];
    [z release];
    [coeffs release];
  }
};

[[nodiscard]] HybridSimplifyOutput simplify_words1_hybrid(
    id<MTLDevice> device,
    id<MTLCommandQueue> command_queue,
    id<MTLBuffer> x,
    id<MTLBuffer> z,
    id<MTLBuffer> coeffs,
    std::size_t terms,
    double atol,
    double rtol);
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

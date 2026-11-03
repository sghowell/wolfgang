#pragma once

#include "hip/pool_allocator_hooks.hpp"

#include <hip/hip_runtime.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/system/hip/execution_policy.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

namespace wolfgang::hip_detail {

// Wolfgang-private stream-ordered device memory pool.
//
// Every Wolfgang HIP operation runs on the null (default) stream and
// synchronizes before returning, so a block freed here is never still in use
// when the pool hands it out again. The pool keeps up to
// kPoolReleaseThresholdBytes of freed blocks cached per device, which removes
// hipMalloc/hipFree from the per-call cost of small operations; above that,
// it returns memory to the driver at the next synchronization. Cached blocks
// are memory other libraries in the process cannot use until the pool trims,
// which it does on any Wolfgang allocation failure (before one retry) and at
// process exit. Devices without memory-pool support, or runs with
// WOLFGANG_HIP_BENCH_DISABLE_POOL=1, use hipMalloc/hipFree instead. The
// pool handle is not revalidated after hipDeviceReset.
//
// Buffers exported to other frameworks (DLPack, the CUDA Array Interface) must
// not come from this pool unless the export keeps the owner alive: a consumer
// on another stream could still read a block after Wolfgang frees it.
inline constexpr std::uint64_t kPoolReleaseThresholdBytes = std::uint64_t{1} << 30;

// rocThrust's default policy, hipMemcpy and Wolfgang's kernels all use the null
// stream, so pooled allocations and frees are ordered with every other call.
inline const hipStream_t kPoolStream = nullptr;

// Allocation failures (after a trim and one retry) throw std::bad_alloc.
[[nodiscard]] void* pool_allocate_bytes(std::size_t bytes, const char* name);
void pool_free_bytes(void* pointer) noexcept;
void pool_free_bytes_on_device(void* pointer, int device) noexcept;

// Synchronize pooled devices and return cached blocks to the driver; used
// before retrying an allocation that failed for lack of memory.
void trim_pools_after_allocation_failure() noexcept;

template <typename T>
void pool_allocate(T*& pointer, std::size_t count, const char* name) {
  pointer = nullptr;
  if (count == 0) {
    return;
  }
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::bad_alloc();
  }
  pointer = static_cast<T*>(pool_allocate_bytes(count * sizeof(T), name));
}

// Owning pooled scratch buffer for the duration of one operation. It records
// its device so the free needs no pointer-attribute query.
template <typename T>
class PoolBuffer {
public:
  PoolBuffer() = default;
  PoolBuffer(std::size_t count, const char* name) {
    (void)hipGetDevice(&device_);
    pool_allocate(pointer_, count, name);
  }
  PoolBuffer(const PoolBuffer&) = delete;
  PoolBuffer& operator=(const PoolBuffer&) = delete;
  PoolBuffer(PoolBuffer&& other) noexcept
      : pointer_(std::exchange(other.pointer_, nullptr)), device_(other.device_) {}
  PoolBuffer& operator=(PoolBuffer&& other) noexcept {
    if (this != &other) {
      pool_free_bytes_on_device(pointer_, device_);
      pointer_ = std::exchange(other.pointer_, nullptr);
      device_ = other.device_;
    }
    return *this;
  }
  ~PoolBuffer() { pool_free_bytes_on_device(pointer_, device_); }

  [[nodiscard]] T* get() const noexcept { return pointer_; }

private:
  T* pointer_ = nullptr;
  int device_ = 0;
};

// Uninitialized pooled device array for scratch that every caller fully writes
// before reading. Unlike thrust::device_vector it launches no value-initializing
// fill kernel. Element types must be trivially copyable (implicit-lifetime).
template <typename T>
class PooledArray {
  static_assert(std::is_trivially_copyable_v<T>, "PooledArray holds raw device storage");

public:
  explicit PooledArray(std::size_t count) : buffer_(count, "device scratch"), count_(count) {}

  [[nodiscard]] thrust::device_ptr<T> begin() const noexcept {
    return thrust::device_pointer_cast(buffer_.get());
  }
  [[nodiscard]] thrust::device_ptr<T> end() const noexcept { return begin() + count_; }
  [[nodiscard]] thrust::device_ptr<T> data() const noexcept { return begin(); }

private:
  PoolBuffer<T> buffer_;
  std::size_t count_ = 0;
};

// Allocator for thrust::device_vector storage drawn from the pool.
template <typename T>
class PoolAllocator : public thrust::device_malloc_allocator<T> {
public:
  using base = thrust::device_malloc_allocator<T>;
  using pointer = typename base::pointer;
  using size_type = typename base::size_type;

  template <typename U>
  struct rebind {
    using other = PoolAllocator<U>;
  };

  PoolAllocator() = default;
  template <typename U>
  PoolAllocator(const PoolAllocator<U>&) noexcept {}

  pointer allocate(size_type count) {
    T* raw = nullptr;
    pool_allocate(raw, count, "thrust device vector");
    return pointer(raw);
  }

  void deallocate(pointer p, size_type) noexcept { pool_free_bytes(thrust::raw_pointer_cast(p)); }
};

template <typename T>
using PooledVector = thrust::device_vector<T, PoolAllocator<T>>;

// Byte allocator for Thrust algorithm temporaries (sort and reduce scratch).
struct PoolTemporaryAllocator {
  using value_type = char;

  char* allocate(std::ptrdiff_t bytes) {
    return static_cast<char*>(
        pool_allocate_bytes(static_cast<std::size_t>(bytes), "thrust scratch"));
  }

  void deallocate(char* pointer, std::size_t) noexcept { pool_free_bytes(pointer); }
};

// Execution policy whose algorithm temporaries come from the pool.
// par_nosync skips the stream synchronization Thrust otherwise performs after
// every algorithm. That is safe here: all work is ordered on the null
// stream, algorithms that return a value to the host still synchronize, and
// each public operation synchronizes before returning.
inline auto pooled_policy() {
  static PoolTemporaryAllocator allocator;
  return thrust::hip::par_nosync(allocator);
}

}  // namespace wolfgang::hip_detail

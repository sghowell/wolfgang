#pragma once

#include <thrust/device_malloc_allocator.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/execution_policy.h>

#include <cstddef>
#include <limits>
#include <new>
#include <utility>

namespace wolfgang::cuda_detail {

// Wolfgang-private stream-ordered device memory pool.
//
// Every Wolfgang CUDA operation runs on the legacy default stream and
// synchronizes before returning, so a block freed here is never still in use
// when the pool hands it out again. The pool keeps freed blocks cached instead
// of returning them to the driver, which removes cudaMalloc/cudaFree from the
// per-call cost of small operations. The pool is separate from the device's
// default pool, so other libraries' allocations are unaffected. Devices or
// drivers without memory-pool support fall back to cudaMalloc/cudaFree.
//
// Buffers exported to other frameworks (DLPack, CUDA Array Interface) must not
// come from this pool: a consumer on another stream could still read a block
// after Wolfgang frees it.
[[nodiscard]] void* pool_allocate_bytes(std::size_t bytes, const char* name);
void pool_free_bytes(void* pointer) noexcept;

template <typename T>
void pool_allocate(T*& pointer, std::size_t count, const char* name) {
  pointer = nullptr;
  if (count == 0) {
    return;
  }
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::bad_array_new_length();
  }
  pointer = static_cast<T*>(pool_allocate_bytes(count * sizeof(T), name));
}

// Owning pooled scratch buffer for the duration of one operation.
template <typename T>
class PoolBuffer {
public:
  PoolBuffer() = default;
  PoolBuffer(std::size_t count, const char* name) { pool_allocate(pointer_, count, name); }
  PoolBuffer(const PoolBuffer&) = delete;
  PoolBuffer& operator=(const PoolBuffer&) = delete;
  PoolBuffer(PoolBuffer&& other) noexcept : pointer_(std::exchange(other.pointer_, nullptr)) {}
  PoolBuffer& operator=(PoolBuffer&& other) noexcept {
    if (this != &other) {
      pool_free_bytes(pointer_);
      pointer_ = std::exchange(other.pointer_, nullptr);
    }
    return *this;
  }
  ~PoolBuffer() { pool_free_bytes(pointer_); }

  [[nodiscard]] T* get() const noexcept { return pointer_; }

private:
  T* pointer_ = nullptr;
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
    return static_cast<char*>(pool_allocate_bytes(static_cast<std::size_t>(bytes), "thrust scratch"));
  }

  void deallocate(char* pointer, std::size_t) noexcept { pool_free_bytes(pointer); }
};

// Execution policy whose algorithm temporaries come from the pool.
inline auto pooled_policy() {
  static PoolTemporaryAllocator allocator;
  return thrust::cuda::par(allocator);
}

}  // namespace wolfgang::cuda_detail

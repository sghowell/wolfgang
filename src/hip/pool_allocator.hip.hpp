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

// Device allocation for Wolfgang's HIP backend: hipMalloc/hipFree by default,
// with an optional stream-ordered pool behind a benchmark-only selector.
//
// Unlike CUDA (where the pool removed most of the small-operation cost on an
// H100), ROCm's hipMalloc/hipFree already cache small blocks. On an MI300X with
// HIP 7.16, small simplify and matmul calls ran 9-37% faster with plain
// hipMalloc than with hipMallocFromPoolAsync/hipFreeAsync
// (docs/benchmarks/reports/hip_parity_mi300x_2026-11-03.md), so the pool is
// off unless WOLFGANG_HIP_BENCH_ENABLE_POOL=1. The interface stays shared so
// the comparison can be rerun on later ROCm releases.
//
// When enabled, every Wolfgang HIP operation runs on the null (default) stream
// and synchronizes before returning, so a block freed to the pool is never
// still in use when the pool hands it out again. The pool keeps up to
// kPoolReleaseThresholdBytes of freed blocks cached per device and trims on
// any Wolfgang allocation failure (before one retry) and at process exit. The
// pool handle is not revalidated after hipDeviceReset.
//
// Buffers exported to other frameworks (DLPack, the CUDA Array Interface) must
// not come from this pool: a consumer on another stream could still read a
// block after Wolfgang frees it.
inline constexpr std::uint64_t kPoolReleaseThresholdBytes = std::uint64_t{1} << 30;

// rocThrust's default policy, hipMemcpy and Wolfgang's kernels all use the null
// stream, so pooled allocations and frees are ordered with every other call.
// par_nosync (below) and uninitialized scratch apply whichever allocator runs.
inline const hipStream_t kPoolStream = nullptr;

// Allocation failures (after a trim and one retry) throw std::bad_alloc.
[[nodiscard]] void* pool_allocate_bytes(std::size_t bytes, const char* name);
void pool_free_bytes(void* pointer) noexcept;
void pool_free_bytes_on_device(void* pointer, int device) noexcept;

// Synchronize pooled devices and return cached blocks to the driver; used
// before retrying an allocation that failed for lack of memory.
void trim_pools_after_allocation_failure() noexcept;

// Per-thread reusable blocks for operation scratch and rocThrust algorithm
// temporaries. On an MI300X, reusing the simplify pipeline's multi-megabyte
// scratch instead of a hipMalloc/hipFree pair per call made 50,000-term
// simplify about twice as fast, while ROCm already caches small blocks. Each
// thread keeps up to kThreadCacheBlocks freed blocks, shared by every call
// site, and reuses them. Every Wolfgang HIP operation runs on the null stream,
// so work that reuses a released block is ordered after the work that used
// it. Requests above kThreadCacheLimitBytes, requests while every cached block
// is in use, and devices using the optional pool (which caches by itself) go
// straight to the allocator. A block on another device is never handed out;
// it is freed when a later allocation evicts it. Freeing a block relies on
// hipFree waiting for queued work on its device. Blocks idle in a thread are
// freed when that thread exits and before any Wolfgang allocation on it is
// retried after running out of memory; other threads keep theirs, so each
// thread can hold up to kThreadCacheBlocks * kThreadCacheLimitBytes. Blocks
// are not revalidated after hipDeviceReset.
inline constexpr std::size_t kThreadCacheLimitBytes = std::size_t{256} << 20;
inline constexpr std::size_t kThreadCacheBlocks = 2;

[[nodiscard]] void* thread_cached_allocate(std::size_t bytes, const char* name);
void thread_cached_free(void* pointer) noexcept;
// Frees the calling thread's idle cached blocks.
void release_idle_thread_cache() noexcept;

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

// Byte allocator for Thrust algorithm temporaries (sort and reduce scratch),
// served from the per-thread block cache.
struct PoolTemporaryAllocator {
  using value_type = char;

  char* allocate(std::ptrdiff_t bytes) {
    return static_cast<char*>(
        thread_cached_allocate(static_cast<std::size_t>(bytes), "thrust scratch"));
  }

  void deallocate(char* pointer, std::size_t) noexcept { thread_cached_free(pointer); }
};

// Owning operation scratch from the per-thread block cache.
class ThreadCachedBuffer {
public:
  ThreadCachedBuffer(std::size_t bytes, const char* name)
      : pointer_(static_cast<std::byte*>(thread_cached_allocate(bytes, name))) {}
  ThreadCachedBuffer(const ThreadCachedBuffer&) = delete;
  ThreadCachedBuffer& operator=(const ThreadCachedBuffer&) = delete;
  ~ThreadCachedBuffer() { thread_cached_free(pointer_); }

  [[nodiscard]] std::byte* data() const noexcept { return pointer_; }

private:
  std::byte* pointer_ = nullptr;
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

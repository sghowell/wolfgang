#include "hip/pool_allocator.hip.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace wolfgang::hip_detail {

namespace {

struct DevicePool {
  hipMemPool_t pool = nullptr;  // null when the device lacks pool support or pooling is off
};

// Both live for the whole process (never destroyed) so frees from objects with
// static storage duration, or from the exit handler, never touch a destroyed
// mutex or map.
std::mutex& pools_mutex() {
  static auto* mutex = new std::mutex();
  return *mutex;
}

std::unordered_map<int, DevicePool>& pools() {
  static auto* table = new std::unordered_map<int, DevicePool>();
  return *table;
}

// Benchmark/test-only switch that turns on the optional pool so its path stays
// exercised and can be compared with the hipMalloc default.
bool pooling_enabled_by_environment() {
  const char* value = std::getenv("WOLFGANG_HIP_BENCH_ENABLE_POOL");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

// Synchronize each pooled device and return every cached block to the driver.
// Callers hold pools_mutex.
void trim_pools_locked() {
  int previous = -1;
  (void)hipGetDevice(&previous);
  for (auto& [device, entry] : pools()) {
    if (entry.pool != nullptr && hipSetDevice(device) == hipSuccess) {
      (void)hipStreamSynchronize(kPoolStream);
      (void)hipMemPoolTrimTo(entry.pool, 0);
    }
  }
  if (previous >= 0) {
    (void)hipSetDevice(previous);
  }
}

// Run pending stream-ordered frees and hand cached blocks back to the driver
// before the process exits. Objects still alive at exit are not freed here. The
// HIP runtime installs its exit handlers on first use, before any pool exists,
// so this handler runs ahead of runtime teardown.
void release_pools_at_exit() {
  std::lock_guard<std::mutex> lock(pools_mutex());
  trim_pools_locked();
}

DevicePool device_pool(int device) {
  std::lock_guard<std::mutex> lock(pools_mutex());
  auto found = pools().find(device);
  if (found != pools().end()) {
    return found->second;
  }
  DevicePool entry;
  int supported = 0;
  const hipError_t attribute_result =
      hipDeviceGetAttribute(&supported, hipDeviceAttributeMemoryPoolsSupported, device);
  if (attribute_result != hipSuccess) {
    (void)hipGetLastError();  // clear only the error this query raised
  } else if (supported != 0 && pooling_enabled_by_environment()) {
    hipMemPoolProps props{};
    props.allocType = hipMemAllocationTypePinned;
    props.handleTypes = hipMemHandleTypeNone;
    props.location.type = hipMemLocationTypeDevice;
    props.location.id = device;
    if (hipMemPoolCreate(&entry.pool, &props) != hipSuccess) {
      (void)hipGetLastError();
      entry.pool = nullptr;
    } else {
      // Cache up to kPoolReleaseThresholdBytes of freed blocks for reuse; above
      // that, the pool returns memory to the driver at the next synchronization,
      // so one very large operation does not stay reserved for the process.
      std::uint64_t threshold = kPoolReleaseThresholdBytes;
      if (hipMemPoolSetAttribute(entry.pool, hipMemPoolAttrReleaseThreshold, &threshold) !=
          hipSuccess) {
        (void)hipGetLastError();
        (void)hipMemPoolDestroy(entry.pool);
        entry.pool = nullptr;
      }
    }
  }
  if (entry.pool != nullptr) {
    static const bool registered = std::atexit(release_pools_at_exit) == 0;
    (void)registered;
  }
  pools().emplace(device, entry);
  return entry;
}

hipError_t allocate_on_device(int device, void** pointer, std::size_t bytes) {
  const DevicePool pool = device_pool(device);
  return pool.pool != nullptr
      ? hipMallocFromPoolAsync(pointer, bytes, pool.pool, kPoolStream)
      : hipMalloc(pointer, bytes);
}

// Process-wide totals over every thread's cache, for the test hook.
std::atomic<std::size_t> g_thread_cache_bytes{0};
std::atomic<std::size_t> g_thread_cache_allocations{0};

struct CachedBlock {
  void* pointer = nullptr;
  std::size_t capacity = 0;
  int device = -1;
  bool in_use = false;
};

void release_cached_block(CachedBlock& block) noexcept {
  if (block.pointer != nullptr) {
    pool_free_bytes_on_device(block.pointer, block.device);
    g_thread_cache_bytes.fetch_sub(block.capacity, std::memory_order_relaxed);
  }
  block = CachedBlock{};
}

// One per thread. Its destructor runs at thread exit; for the main thread,
// thread_local destructors run before any atexit handler, including the HIP
// runtime's.
struct ThreadBlockCache {
  std::array<CachedBlock, kThreadCacheBlocks> blocks{};

  ThreadBlockCache() = default;
  ThreadBlockCache(const ThreadBlockCache&) = delete;
  ThreadBlockCache& operator=(const ThreadBlockCache&) = delete;
  ~ThreadBlockCache() {
    for (CachedBlock& block : blocks) {
      release_cached_block(block);
    }
  }

  void release_idle() noexcept {
    for (CachedBlock& block : blocks) {
      if (!block.in_use) {
        release_cached_block(block);
      }
    }
  }
};

ThreadBlockCache& thread_block_cache() {
  thread_local ThreadBlockCache cache;
  return cache;
}

// Capacity for a new block: the next power of two, so a slowly growing size
// regrows only logarithmically often, capped at the cache limit.
std::size_t cached_block_capacity(std::size_t bytes) {
  std::size_t capacity = std::size_t{1} << 20;
  while (capacity < bytes) {
    capacity <<= 1;
  }
  return std::max(bytes, std::min(capacity, kThreadCacheLimitBytes));
}

}  // namespace

void* thread_cached_allocate(std::size_t bytes, const char* name) {
  if (bytes == 0) {
    return nullptr;
  }
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess) {
    (void)hipGetLastError();
    throw std::runtime_error(std::string("HIP get current device failed for ") + name);
  }
  if (bytes > kThreadCacheLimitBytes || device_pool(device).pool != nullptr) {
    return pool_allocate_bytes(bytes, name);
  }
  ThreadBlockCache& cache = thread_block_cache();
  CachedBlock* fit = nullptr;
  CachedBlock* replace = nullptr;
  for (CachedBlock& block : cache.blocks) {
    if (block.in_use) {
      continue;
    }
    if (block.pointer != nullptr && block.device == device && block.capacity >= bytes &&
        (fit == nullptr || block.capacity < fit->capacity)) {
      fit = &block;
    }
    if (replace == nullptr || block.capacity < replace->capacity) {
      replace = &block;
    }
  }
  if (fit != nullptr) {
    fit->in_use = true;
    return fit->pointer;
  }
  if (replace == nullptr) {
    return pool_allocate_bytes(bytes, name);  // every cached block is in use
  }
  release_cached_block(*replace);
  const std::size_t capacity = cached_block_capacity(bytes);
  void* pointer = nullptr;
  try {
    pointer = pool_allocate_bytes(capacity, name);
  } catch (const std::bad_alloc&) {
    // Idle cached blocks may be what is short; return them and retry with the
    // exact size.
    cache.release_idle();
    pointer = pool_allocate_bytes(bytes, name);
    replace->capacity = bytes;
  }
  if (replace->capacity == 0) {
    replace->capacity = capacity;
  }
  replace->pointer = pointer;
  replace->device = device;
  replace->in_use = true;
  g_thread_cache_bytes.fetch_add(replace->capacity, std::memory_order_relaxed);
  g_thread_cache_allocations.fetch_add(1, std::memory_order_relaxed);
  return pointer;
}

void release_idle_thread_cache() noexcept {
  thread_block_cache().release_idle();
}

void thread_cached_free(void* pointer) noexcept {
  if (pointer == nullptr) {
    return;
  }
  for (CachedBlock& block : thread_block_cache().blocks) {
    if (block.pointer == pointer) {
      block.in_use = false;
      return;
    }
  }
  pool_free_bytes(pointer);  // allocated outside the cache
}

ThreadCacheCounters thread_cache_counters_for_testing() {
  return {g_thread_cache_bytes.load(), g_thread_cache_allocations.load(), kThreadCacheLimitBytes};
}

void trim_pools_after_allocation_failure() noexcept {
  try {
    std::lock_guard<std::mutex> lock(pools_mutex());
    trim_pools_locked();
  } catch (...) {
    // A failed lock leaves the retry to report the original allocation error.
  }
}

void* pool_allocate_bytes(std::size_t bytes, const char* name) {
  if (bytes == 0) {
    return nullptr;
  }
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess) {
    (void)hipGetLastError();
    throw std::runtime_error(std::string("HIP get current device failed for ") + name);
  }
  void* pointer = nullptr;
  hipError_t result = allocate_on_device(device, &pointer, bytes);
  if (result == hipErrorOutOfMemory) {
    // Cached blocks may be what is short: return the pools' and this thread's
    // idle blocks to the driver and retry once.
    (void)hipGetLastError();
    trim_pools_after_allocation_failure();
    release_idle_thread_cache();
    result = allocate_on_device(device, &pointer, bytes);
  }
  if (result == hipErrorOutOfMemory) {
    (void)hipGetLastError();
    throw std::bad_alloc();
  }
  if (result != hipSuccess) {
    (void)hipGetLastError();
    throw std::runtime_error(
        std::string("HIP allocate device buffer failed for ") + name + ": " +
        hipGetErrorString(result));
  }
  return pointer;
}

void pool_free_bytes_on_device(void* pointer, int device) noexcept {
  if (pointer == nullptr) {
    return;
  }
  bool pooled = false;
  try {
    pooled = device_pool(device).pool != nullptr;
  } catch (...) {
    pooled = false;
  }
  int previous = -1;
  (void)hipGetDevice(&previous);
  if (device != previous) {
    (void)hipSetDevice(device);
  }
  // Pool support is fixed per device, so this matches the allocation path.
  const hipError_t result =
      pooled ? hipFreeAsync(pointer, kPoolStream) : hipFree(pointer);
  if (result != hipSuccess) {
    (void)hipGetLastError();  // a destructor must not leave its own error behind
  }
  if (device != previous && previous >= 0) {
    (void)hipSetDevice(previous);
  }
}

void pool_free_bytes(void* pointer) noexcept {
  if (pointer == nullptr) {
    return;
  }
  int device = -1;
  hipPointerAttribute_t attributes{};
  if (hipPointerGetAttributes(&attributes, pointer) == hipSuccess) {
    device = attributes.device;
  } else {
    (void)hipGetLastError();
    (void)hipGetDevice(&device);
  }
  pool_free_bytes_on_device(pointer, device);
}

std::size_t pool_used_bytes_for_testing(int device) {
  const DevicePool pool = device_pool(device);
  if (pool.pool == nullptr) {
    return 0;
  }
  std::uint64_t used = 0;
  if (hipMemPoolGetAttribute(pool.pool, hipMemPoolAttrUsedMemCurrent, &used) != hipSuccess) {
    (void)hipGetLastError();
    throw std::runtime_error("HIP memory pool usage query failed");
  }
  return static_cast<std::size_t>(used);
}

bool pool_enabled_for_testing(int device) {
  return device_pool(device).pool != nullptr;
}

}  // namespace wolfgang::hip_detail

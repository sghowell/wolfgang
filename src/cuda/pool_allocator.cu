#include "cuda/pool_allocator.cuh"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace wolfgang::cuda_detail {

namespace {

struct DevicePool {
  cudaMemPool_t pool = nullptr;  // null when the device lacks pool support or pooling is off
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

// Benchmark/test-only switch that forces the cudaMalloc fallback so it stays
// exercised on hardware with pool support.
bool pooling_disabled_by_environment() {
  const char* value = std::getenv("WOLFGANG_CUDA_BENCH_DISABLE_POOL");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

// Synchronize each pooled device and return every cached block to the driver.
// Callers hold pools_mutex.
void trim_pools_locked() {
  int previous = -1;
  (void)cudaGetDevice(&previous);
  for (auto& [device, entry] : pools()) {
    if (entry.pool != nullptr && cudaSetDevice(device) == cudaSuccess) {
      (void)cudaStreamSynchronize(cudaStreamLegacy);
      (void)cudaMemPoolTrimTo(entry.pool, 0);
    }
  }
  if (previous >= 0) {
    (void)cudaSetDevice(previous);
  }
}

// Run pending stream-ordered frees and hand cached blocks back to the driver
// before the process exits. Objects still alive at exit are not freed here. The
// CUDA runtime installs its exit handlers on first use, before any pool exists,
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
  const cudaError_t attribute_result =
      cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, device);
  if (attribute_result != cudaSuccess) {
    (void)cudaGetLastError();  // clear only the error this query raised
  } else if (supported != 0 && !pooling_disabled_by_environment()) {
    cudaMemPoolProps props{};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = device;
    if (cudaMemPoolCreate(&entry.pool, &props) != cudaSuccess) {
      (void)cudaGetLastError();
      entry.pool = nullptr;
    } else {
      // Cache up to kPoolReleaseThresholdBytes of freed blocks for reuse; above
      // that, the pool returns memory to the driver at the next synchronization,
      // so one very large operation does not stay reserved for the process.
      std::uint64_t threshold = kPoolReleaseThresholdBytes;
      if (cudaMemPoolSetAttribute(entry.pool, cudaMemPoolAttrReleaseThreshold, &threshold) !=
          cudaSuccess) {
        (void)cudaGetLastError();
        (void)cudaMemPoolDestroy(entry.pool);
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

cudaError_t allocate_on_device(int device, void** pointer, std::size_t bytes) {
  const DevicePool pool = device_pool(device);
  return pool.pool != nullptr
      ? cudaMallocFromPoolAsync(pointer, bytes, pool.pool, cudaStreamLegacy)
      : cudaMalloc(pointer, bytes);
}

}  // namespace

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
  if (cudaGetDevice(&device) != cudaSuccess) {
    (void)cudaGetLastError();
    throw std::runtime_error(std::string("CUDA get current device failed for ") + name);
  }
  void* pointer = nullptr;
  cudaError_t result = allocate_on_device(device, &pointer, bytes);
  if (result == cudaErrorMemoryAllocation) {
    // Cached blocks may be what is short: return them to the driver and retry once.
    (void)cudaGetLastError();
    trim_pools_after_allocation_failure();
    result = allocate_on_device(device, &pointer, bytes);
  }
  if (result == cudaErrorMemoryAllocation) {
    (void)cudaGetLastError();
    throw std::bad_alloc();
  }
  if (result != cudaSuccess) {
    (void)cudaGetLastError();
    throw std::runtime_error(
        std::string("CUDA allocate device buffer failed for ") + name + ": " +
        cudaGetErrorString(result));
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
  (void)cudaGetDevice(&previous);
  if (device != previous) {
    (void)cudaSetDevice(device);
  }
  // Pool support is fixed per device, so this matches the allocation path.
  const cudaError_t result =
      pooled ? cudaFreeAsync(pointer, cudaStreamLegacy) : cudaFree(pointer);
  if (result != cudaSuccess) {
    (void)cudaGetLastError();  // a destructor must not leave its own error behind
  }
  if (device != previous && previous >= 0) {
    (void)cudaSetDevice(previous);
  }
}

void pool_free_bytes(void* pointer) noexcept {
  if (pointer == nullptr) {
    return;
  }
  int device = -1;
  cudaPointerAttributes attributes{};
  if (cudaPointerGetAttributes(&attributes, pointer) == cudaSuccess) {
    device = attributes.device;
  } else {
    (void)cudaGetLastError();
    (void)cudaGetDevice(&device);
  }
  pool_free_bytes_on_device(pointer, device);
}

std::size_t pool_used_bytes_for_testing(int device) {
  const DevicePool pool = device_pool(device);
  if (pool.pool == nullptr) {
    return 0;
  }
  std::uint64_t used = 0;
  if (cudaMemPoolGetAttribute(pool.pool, cudaMemPoolAttrUsedMemCurrent, &used) != cudaSuccess) {
    (void)cudaGetLastError();
    throw std::runtime_error("CUDA memory pool usage query failed");
  }
  return static_cast<std::size_t>(used);
}

bool pool_enabled_for_testing(int device) {
  return device_pool(device).pool != nullptr;
}

}  // namespace wolfgang::cuda_detail

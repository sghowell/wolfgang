#include "cuda/pool_allocator.cuh"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace wolfgang::cuda_detail {

namespace {

struct DevicePool {
  cudaMemPool_t pool = nullptr;  // null when the device lacks pool support
};

std::mutex& pools_mutex() {
  static std::mutex mutex;
  return mutex;
}

// Pools live for the process; the driver reclaims them at exit.
std::unordered_map<int, DevicePool>& pools() {
  static auto* table = new std::unordered_map<int, DevicePool>();
  return *table;
}

DevicePool device_pool(int device) {
  std::lock_guard<std::mutex> lock(pools_mutex());
  auto found = pools().find(device);
  if (found != pools().end()) {
    return found->second;
  }
  DevicePool entry;
  int supported = 0;
  if (cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, device) == cudaSuccess &&
      supported != 0) {
    cudaMemPoolProps props{};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = device;
    if (cudaMemPoolCreate(&entry.pool, &props) == cudaSuccess) {
      // Keep every freed block cached for reuse by later operations.
      std::uint64_t threshold = std::numeric_limits<std::uint64_t>::max();
      if (cudaMemPoolSetAttribute(entry.pool, cudaMemPoolAttrReleaseThreshold, &threshold) !=
          cudaSuccess) {
        (void)cudaMemPoolDestroy(entry.pool);
        entry.pool = nullptr;
      }
    } else {
      entry.pool = nullptr;
    }
  }
  (void)cudaGetLastError();
  pools().emplace(device, entry);
  return entry;
}

}  // namespace

void* pool_allocate_bytes(std::size_t bytes, const char* name) {
  if (bytes == 0) {
    return nullptr;
  }
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA get current device failed for ") + name);
  }
  void* pointer = nullptr;
  const DevicePool pool = device_pool(device);
  const cudaError_t result = pool.pool != nullptr
      ? cudaMallocFromPoolAsync(&pointer, bytes, pool.pool, cudaStreamLegacy)
      : cudaMalloc(&pointer, bytes);
  if (result != cudaSuccess) {
    (void)cudaGetLastError();
    throw std::runtime_error(
        std::string("CUDA allocate device buffer failed for ") + name + ": " +
        cudaGetErrorString(result));
  }
  return pointer;
}

void pool_free_bytes(void* pointer) noexcept {
  if (pointer == nullptr) {
    return;
  }
  // Free on the owning device through the same path that allocated the block;
  // pool support is fixed per device, so the device decides the path.
  int device = -1;
  cudaPointerAttributes attributes{};
  if (cudaPointerGetAttributes(&attributes, pointer) == cudaSuccess) {
    device = attributes.device;
  }
  (void)cudaGetLastError();
  int previous = -1;
  (void)cudaGetDevice(&previous);
  if (device < 0) {
    device = previous;
  }
  bool pooled = false;
  try {
    pooled = device_pool(device).pool != nullptr;
  } catch (...) {
    pooled = false;
  }
  if (device != previous) {
    (void)cudaSetDevice(device);
  }
  if (pooled) {
    (void)cudaFreeAsync(pointer, cudaStreamLegacy);
  } else {
    (void)cudaFree(pointer);
  }
  if (device != previous && previous >= 0) {
    (void)cudaSetDevice(previous);
  }
  (void)cudaGetLastError();
}

}  // namespace wolfgang::cuda_detail

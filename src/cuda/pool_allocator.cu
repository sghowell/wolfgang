#include "cuda/pool_allocator.cuh"

#include "cuda/device_guard.cuh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#if !defined(_WIN32)
#include <pthread.h>
#endif

namespace wolfgang::cuda_detail {

namespace {

struct DevicePool {
  cudaMemPool_t pool = nullptr;  // null when the device lacks pool support or pooling is off
  std::vector<int> mapped_peers;  // peers grant_peer_access_for_export() has mapped it into
  bool peers_settled = false;     // every peer that can access the pool is mapped
};

// Set in a child forked from this process: the CUDA state, and any lock a
// parent thread held at fork, are not usable there, so the exit handler skips.
std::atomic<bool>& forked_child() {
  static auto* flag = new std::atomic<bool>(false);
  return *flag;
}

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
  for (auto& [device, entry] : pools()) {
    if (entry.pool == nullptr) {
      continue;
    }
    const ScopedCudaDevice guard(device, std::nothrow);
    if (guard.ok()) {
      (void)cudaStreamSynchronize(cudaStreamLegacy);
      (void)cudaMemPoolTrimTo(entry.pool, 0);
    }
  }
}

// Run pending stream-ordered frees and hand cached blocks back to the driver
// before the process exits. Objects still alive at exit are not freed here. The
// CUDA runtime installs its exit handlers on first use, before any pool exists,
// so this handler runs ahead of runtime teardown.
void release_pools_at_exit() {
  if (forked_child().load()) {
    return;
  }
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
    static const bool registered = [] {
#if !defined(_WIN32)
      (void)pthread_atfork(nullptr, nullptr, [] { forked_child().store(true); });
#endif
      return std::atexit(release_pools_at_exit) == 0;
    }();
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

void grant_peer_access_for_export(int device) noexcept {
  try {
    std::lock_guard<std::mutex> lock(pools_mutex());
    auto found = pools().find(device);
    if (found == pools().end() || found->second.pool == nullptr || found->second.peers_settled) {
      return;  // cudaMalloc memory needs no grant: consumers enable peer access themselves
    }
    DevicePool& entry = found->second;
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
      (void)cudaGetLastError();
      return;
    }
    // Peers are mapped one at a time, so one that fails does not keep the
    // others unmapped; any failure leaves the pool unsettled for the next export.
    bool settled = true;
    for (int peer = 0; peer < device_count; ++peer) {
      if (peer == device ||
          std::find(entry.mapped_peers.begin(), entry.mapped_peers.end(), peer) != entry.mapped_peers.end()) {
        continue;
      }
      int can_access = 0;
      if (cudaDeviceCanAccessPeer(&can_access, peer, device) != cudaSuccess) {
        (void)cudaGetLastError();
        settled = false;
        continue;
      }
      if (can_access == 0) {
        continue;
      }
      cudaMemAccessDesc desc{};
      desc.location.type = cudaMemLocationTypeDevice;
      desc.location.id = peer;
      // Exports are read-only by contract, but consumers do not enforce it;
      // read-write matches cudaMalloc memory under peer access.
      desc.flags = cudaMemAccessFlagsProtReadWrite;
      // The mapping covers the pool's current and future allocations.
      if (cudaMemPoolSetAccess(entry.pool, &desc, 1) != cudaSuccess) {
        (void)cudaGetLastError();
        settled = false;
        continue;
      }
      entry.mapped_peers.push_back(peer);
    }
    entry.peers_settled = settled;
  } catch (...) {
    // Leave the pool as it is; the next export retries.
  }
}

void synchronize_export_readers(int device) noexcept {
  {
    const ScopedCudaDevice guard(device, std::nothrow);
    if (guard.ok()) {
      (void)cudaDeviceSynchronize();
    }
  }
  std::vector<int> peers;
  try {
    std::lock_guard<std::mutex> lock(pools_mutex());
    auto found = pools().find(device);
    if (found != pools().end()) {
      peers = found->second.mapped_peers;
    }
  } catch (...) {
    peers.clear();
  }
  for (const int peer : peers) {
    // A peer without a primary context has run no kernel that could read the
    // buffer, and synchronizing it would create one.
    if (!device_has_primary_context(peer)) {
      continue;
    }
    const ScopedCudaDevice guard(peer, std::nothrow);
    if (guard.ok()) {
      (void)cudaDeviceSynchronize();
    }
  }
  (void)cudaGetLastError();  // a destructor must not leave its own error behind
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
  const ScopedCudaDevice guard(device, std::nothrow);
  // Pool support is fixed per device, so this matches the allocation path.
  const cudaError_t result =
      pooled ? cudaFreeAsync(pointer, cudaStreamLegacy) : cudaFree(pointer);
  if (result != cudaSuccess) {
    (void)cudaGetLastError();  // a destructor must not leave its own error behind
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

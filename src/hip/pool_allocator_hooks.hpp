#pragma once

#include <cstddef>

// Plain C++ declarations for the Wolfgang HIP memory pool, usable from host
// translation units that cannot include Thrust (the Python bindings).
namespace wolfgang::hip_detail {

// Bytes currently handed out by the device's pool (0 when the device does not
// use the pool), and whether the device uses the pool at all. Test hooks only.
[[nodiscard]] std::size_t pool_used_bytes_for_testing(int device);
[[nodiscard]] bool pool_enabled_for_testing(int device);

// Per-thread block cache, summed over every thread: bytes held in cached
// blocks, and blocks the cache has allocated so far. Test hooks only.
struct ThreadCacheCounters {
  std::size_t cached_bytes = 0;
  std::size_t block_allocations = 0;
  std::size_t limit_bytes = 0;
};
[[nodiscard]] ThreadCacheCounters thread_cache_counters_for_testing();

}  // namespace wolfgang::hip_detail

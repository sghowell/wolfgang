#pragma once

// Device-to-host copies into new host vectors, shared by the CUDA and HIP
// backends (plain C++; the backend supplies the copy and wait calls).
//
// Copying a pageable destination costs more than the copy itself:
//
// - each buffer was its own blocking copy, so a small operator waited for the
//   GPU three times;
// - std::vector(n) zero-fills its storage first, and the driver stages a
//   pageable copy through its own buffer;
// - fresh host memory is faulted in 4 KiB at a time. On the KVM hosts behind
//   docs/benchmarks/reports/host_transfers_2026-11-03.md, faults dominate
//   large copies.
//
// Here every buffer is copied asynchronously into a pinned staging buffer in
// batches. While one half of the staging buffer is filling, the other half is
// appended to vectors that were reserved rather than zero-filled. One wait
// covers a whole batch, so an operator that fits in one half waits once.
// Large destinations ask the kernel for huge pages before their first touch,
// as NumPy does for large arrays.

#include <algorithm>
#include <atomic>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#endif

#if !defined(_WIN32)
#include <pthread.h>
#endif

namespace wolfgang::accelerator_detail {

// Huge pages are requested for destinations of at least this size, the
// threshold NumPy uses on Linux.
inline constexpr std::size_t kHugePageAdviceMinBytes = std::size_t{4} << 20;

// Linux only, and advisory: when transparent huge pages are set to "madvise"
// (the Ubuntu default), memory faulted in 2 MiB at a time costs a fraction of
// 4 KiB faults. Only whole 2 MiB pages inside the range are advised.
inline void advise_huge_pages(const void* pointer, std::size_t bytes) noexcept {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
  constexpr std::uintptr_t kHugePage = std::uintptr_t{2} << 20;
  if (bytes < kHugePageAdviceMinBytes) {
    return;
  }
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  const std::uintptr_t begin = (address + kHugePage - 1) & ~(kHugePage - 1);
  const std::uintptr_t end = (address + bytes) & ~(kHugePage - 1);
  if (end > begin) {
    (void)madvise(reinterpret_cast<void*>(begin), end - begin, MADV_HUGEPAGE);
  }
#else
  (void)pointer;
  (void)bytes;
#endif
}

// Sizes `destination` to `count` elements for a direct device-to-host copy,
// requesting huge pages before the zero-fill touches the storage.
template <typename T>
void resize_for_direct_copy(std::vector<T>& destination, std::size_t count) {
  destination.clear();
  destination.reserve(count);
  advise_huge_pages(destination.data(), count * sizeof(T));
  destination.resize(count);
}

// One device buffer copied into a host vector of T.
struct DeviceToHostSegment {
  const void* device = nullptr;
  std::size_t bytes = 0;
  void* vector = nullptr;
  void (*append)(void* vector, const std::byte* data, std::size_t bytes) = nullptr;
  std::size_t element_size = 1;
};

template <typename T>
DeviceToHostSegment device_to_host_segment(std::vector<T>& destination, const T* device, std::size_t count) {
  destination.clear();
  destination.reserve(count);
  advise_huge_pages(destination.data(), count * sizeof(T));
  return {device, count * sizeof(T), &destination,
          [](void* vector, const std::byte* data, std::size_t bytes) {
            auto& out = *static_cast<std::vector<T>*>(vector);
            const auto* first = reinterpret_cast<const T*>(data);
            out.insert(out.end(), first, first + bytes / sizeof(T));
          },
          sizeof(T)};
}

// Process-wide pool of pinned staging buffers. A call borrows one (the
// smallest idle buffer that fits, else a new one) and returns it afterwards,
// so threads that come and go reuse buffers instead of each paying for a
// pinned allocation (about 80 us for 64 KiB on the MI300X). At most
// kMaxIdle buffers stay idle; extras are freed on return. The pool itself is
// never destroyed, because static destruction could run after the runtime has
// shut down. Instead an exit handler, registered after the first pinned
// allocation and so after the CUDA or HIP runtime installed its own exit
// handlers, frees the idle buffers first; a buffer still leased at exit is
// reclaimed by the driver. Like the other host-side buffers, the pool is not
// revalidated after a device reset.
// `Allocate(bytes)` returns pinned memory or nullptr; `Free(pointer)` frees it.
template <void* (*Allocate)(std::size_t), void (*Free)(void*)>
class PinnedStagingPool {
 public:
  static constexpr std::size_t kMaxIdle = 4;

  class Lease {
   public:
    Lease() = default;
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& other) noexcept
        : pointer_(std::exchange(other.pointer_, nullptr)), bytes_(std::exchange(other.bytes_, 0)) {}
    Lease& operator=(Lease&&) = delete;
    ~Lease() {
      if (pointer_ != nullptr) {
        instance().give_back(pointer_, bytes_);
      }
    }
    [[nodiscard]] std::byte* data() const noexcept { return pointer_; }
    [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

   private:
    friend class PinnedStagingPool;
    Lease(std::byte* pointer, std::size_t bytes) : pointer_(pointer), bytes_(bytes) {}
    std::byte* pointer_ = nullptr;
    std::size_t bytes_ = 0;
  };

  // A buffer of at least `bytes` if one can be had; failing that, the largest
  // idle buffer (the staged copy works with any size of at least 32 bytes);
  // failing that, an empty lease.
  static Lease borrow(std::size_t bytes) {
    PinnedStagingPool& pool = instance();
    {
      const std::lock_guard<std::mutex> lock(pool.mutex_);
      auto best = pool.idle_.end();
      for (auto it = pool.idle_.begin(); it != pool.idle_.end(); ++it) {
        if (it->second >= bytes && (best == pool.idle_.end() || it->second < best->second)) {
          best = it;
        }
      }
      if (best != pool.idle_.end()) {
        Lease lease(best->first, best->second);
        pool.idle_.erase(best);
        return lease;
      }
    }
    if (void* pointer = Allocate(bytes)) {
      static const bool registered = [] {
#if !defined(_WIN32)
        (void)pthread_atfork(nullptr, nullptr, [] { instance().forked_child_.store(true); });
#endif
        return std::atexit(release_idle_at_exit) == 0;
      }();
      (void)registered;
      return Lease(static_cast<std::byte*>(pointer), bytes);
    }
    const std::lock_guard<std::mutex> lock(pool.mutex_);
    if (pool.idle_.empty()) {
      return Lease();
    }
    auto largest = std::max_element(pool.idle_.begin(), pool.idle_.end(),
                                    [](const auto& a, const auto& b) { return a.second < b.second; });
    Lease lease(largest->first, largest->second);
    pool.idle_.erase(largest);
    return lease;
  }

 private:
  PinnedStagingPool() { idle_.reserve(kMaxIdle + 1); }  // give_back never allocates

  static PinnedStagingPool& instance() {
    static auto* pool = new PinnedStagingPool();
    return *pool;
  }

  static void release_idle_at_exit() {
    PinnedStagingPool& pool = instance();
    if (pool.forked_child_.load()) {
      return;  // a forked child cannot use the runtime, or a lock held at fork
    }
    const std::lock_guard<std::mutex> lock(pool.mutex_);
    for (const auto& [pointer, bytes] : pool.idle_) {
      Free(pointer);
    }
    pool.idle_.clear();
  }

  void give_back(std::byte* pointer, std::size_t bytes) noexcept {
    std::byte* evicted = nullptr;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      idle_.emplace_back(pointer, bytes);
      if (idle_.size() > kMaxIdle) {
        auto smallest = std::min_element(idle_.begin(), idle_.end(),
                                         [](const auto& a, const auto& b) { return a.second < b.second; });
        evicted = smallest->first;
        idle_.erase(smallest);
      }
    }
    if (evicted != nullptr) {
      Free(evicted);
    }
  }

  std::mutex mutex_;
  std::vector<std::pair<std::byte*, std::size_t>> idle_;
  std::atomic<bool> forked_child_{false};
};

// Staging capacity for `total_bytes`: twice the total (two halves), as a
// power of two from 64 KiB, capped at `max_bytes`.
inline std::size_t staging_bytes_for(std::size_t total_bytes, std::size_t max_bytes) {
  std::size_t bytes = std::size_t{64} << 10;
  while (bytes < max_bytes && bytes / 2 < total_bytes) {
    bytes <<= 1;
  }
  return std::min(bytes, max_bytes);
}

// Copies every segment through `staging` (pinned, at least 32 bytes).
// `copy_async(dst, src, bytes)` queues one device-to-host copy on the stream
// Wolfgang uses and `wait()` waits for that stream. If a copy throws, the
// function waits for copies already queued before rethrowing, so none is
// still writing into the staging buffer when the next call reuses it.
template <typename CopyAsync, typename Wait>
void staged_device_to_host(
    std::byte* staging,
    std::size_t staging_bytes,
    const std::vector<DeviceToHostSegment>& segments,
    CopyAsync&& copy_async,
    Wait&& wait) {
  struct Chunk {
    std::size_t segment;
    std::size_t offset;
    std::size_t bytes;
    std::size_t staging_offset;
  };
  // Cut segments into chunks no larger than half the staging buffer (whole
  // elements; 16-byte aligned in staging) and pack consecutive chunks into
  // batches that fit in one half.
  const std::size_t half = staging_bytes / 2 / 16 * 16;  // keeps chunks 16-byte aligned
  std::vector<std::vector<Chunk>> batches(1);
  std::size_t used = 0;
  for (std::size_t s = 0; s < segments.size(); ++s) {
    const DeviceToHostSegment& segment = segments[s];
    const std::size_t step = half / segment.element_size * segment.element_size;
    if (step == 0) {
      throw std::logic_error("staging buffer smaller than one element");
    }
    for (std::size_t offset = 0; offset < segment.bytes; offset += step) {
      const std::size_t bytes = std::min(step, segment.bytes - offset);
      const std::size_t at = (used + 15) / 16 * 16;
      if (at + bytes > half && !batches.back().empty()) {
        batches.emplace_back();
        used = 0;
      }
      const std::size_t start = batches.back().empty() ? 0 : (used + 15) / 16 * 16;
      batches.back().push_back({s, offset, bytes, start});
      used = start + bytes;
    }
  }
  if (batches.back().empty()) {
    return;
  }
  const auto issue = [&](std::size_t b) {
    std::byte* base = staging + (b % 2) * half;
    for (const Chunk& chunk : batches[b]) {
      copy_async(base + chunk.staging_offset,
                 static_cast<const std::byte*>(segments[chunk.segment].device) + chunk.offset,
                 chunk.bytes);
    }
  };
  const auto drain = [&](std::size_t b) {
    const std::byte* base = staging + (b % 2) * half;
    for (const Chunk& chunk : batches[b]) {
      const DeviceToHostSegment& segment = segments[chunk.segment];
      segment.append(segment.vector, base + chunk.staging_offset, chunk.bytes);
    }
  };
  // Batch b lands in half b % 2. A half is reused two batches later, after
  // the wait that precedes the next issue and after its own drain.
  try {
    issue(0);
    for (std::size_t b = 1; b < batches.size(); ++b) {
      wait();
      issue(b);
      drain(b - 1);
    }
    wait();
  } catch (...) {
    try {
      wait();
    } catch (...) {
    }
    throw;
  }
  drain(batches.size() - 1);
}

}  // namespace wolfgang::accelerator_detail

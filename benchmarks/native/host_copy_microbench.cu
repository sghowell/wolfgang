// Host-device copy microbenchmark behind the to_host design in
// src/detail/staged_transfer.hpp (docs/benchmarks/reports/host_transfers_2026-11-03.md).
// Not built by CMake:
//
//   nvcc -O2 -std=c++17 -o host_copy_microbench benchmarks/native/host_copy_microbench.cu
//   ./host_copy_microbench
//
// For each size it times, as medians in microseconds:
//   h2d_pageable     cudaMemcpy from an existing std::vector
//   h2d_staged       memcpy into an 8 MiB pinned half, async copy, double-buffered
//   d2h_pageable     cudaMemcpy into a new std::vector(n) (zero-filled first)
//   d2h_staged       async copies into pinned halves, appended to a reserved vector
//   d2h_staged_huge  as d2h_staged, with MADV_HUGEPAGE on the reserved storage
//   d2h_direct_huge  cudaMemcpy into a reserved vector advised for huge pages,
//                    then sized (the zero-fill faults the huge pages in)
// The huge-page columns advise only whole 2 MiB pages, so below 4 MiB they
// match the columns without advice (reported as n/a there).
#include <cuda_runtime.h>
#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double micros(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

void advise_huge(void* p, size_t bytes) {
  const uintptr_t huge = uintptr_t{2} << 20;
  const uintptr_t begin = (reinterpret_cast<uintptr_t>(p) + huge - 1) & ~(huge - 1);
  const uintptr_t end = (reinterpret_cast<uintptr_t>(p) + bytes) & ~(huge - 1);
  if (end > begin) {
    madvise(reinterpret_cast<void*>(begin), end - begin, MADV_HUGEPAGE);
  }
}

}  // namespace

int main() {
  const size_t half = size_t{8} << 20;
  char* pinned = nullptr;
  cudaHostAlloc(reinterpret_cast<void**>(&pinned), 2 * half, cudaHostAllocDefault);
  std::printf("bytes,h2d_pageable,h2d_staged,d2h_pageable,d2h_staged,d2h_staged_huge,d2h_direct_huge\n");
  for (size_t bytes : {size_t{4096}, size_t{65536}, size_t{2} << 20, size_t{8} << 20, size_t{32} << 20,
                       size_t{128} << 20}) {
    char* dev = nullptr;
    cudaMalloc(&dev, bytes);
    cudaMemset(dev, 1, bytes);
    std::vector<char> src(bytes, 3);
    auto staged_d2h = [&](std::vector<char>& dst) {
      size_t n = std::min(half, bytes);
      cudaMemcpyAsync(pinned, dev, n, cudaMemcpyDeviceToHost, 0);
      size_t prev = n, off = n;
      int h = 0;
      while (off < bytes) {
        cudaStreamSynchronize(0);
        const size_t m = std::min(half, bytes - off);
        cudaMemcpyAsync(pinned + (h ^ 1) * half, dev + off, m, cudaMemcpyDeviceToHost, 0);
        dst.insert(dst.end(), pinned + h * half, pinned + h * half + prev);
        prev = m;
        off += m;
        h ^= 1;
      }
      cudaStreamSynchronize(0);
      dst.insert(dst.end(), pinned + h * half, pinned + h * half + prev);
    };
    const int reps = bytes <= (size_t{2} << 20) ? 200 : 12;
    std::vector<double> t[6];
    for (int r = 0; r < reps; ++r) {
      auto a = Clock::now();
      cudaMemcpy(dev, src.data(), bytes, cudaMemcpyHostToDevice);
      t[0].push_back(micros(a, Clock::now()));

      a = Clock::now();
      for (size_t off = 0, h = 0; off < bytes; off += half, h ^= 1) {
        const size_t n = std::min(half, bytes - off);
        std::memcpy(pinned + h * half, src.data() + off, n);
        if (off != 0) {
          cudaStreamSynchronize(0);
        }
        cudaMemcpyAsync(dev + off, pinned + h * half, n, cudaMemcpyHostToDevice, 0);
      }
      cudaStreamSynchronize(0);
      t[1].push_back(micros(a, Clock::now()));

      a = Clock::now();
      {
        std::vector<char> dst(bytes);
        cudaMemcpy(dst.data(), dev, bytes, cudaMemcpyDeviceToHost);
      }
      t[2].push_back(micros(a, Clock::now()));

      a = Clock::now();
      {
        std::vector<char> dst;
        dst.reserve(bytes);
        staged_d2h(dst);
      }
      t[3].push_back(micros(a, Clock::now()));

      a = Clock::now();
      {
        std::vector<char> dst;
        dst.reserve(bytes);
        advise_huge(dst.data(), bytes);
        staged_d2h(dst);
      }
      t[4].push_back(micros(a, Clock::now()));

      a = Clock::now();
      {
        std::vector<char> dst;
        dst.reserve(bytes);
        advise_huge(dst.data(), bytes);
        dst.resize(bytes);
        cudaMemcpy(dst.data(), dev, bytes, cudaMemcpyDeviceToHost);
      }
      t[5].push_back(micros(a, Clock::now()));
    }
    std::printf("%zu,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n", bytes, median(t[0]), median(t[1]), median(t[2]),
                median(t[3]), median(t[4]), median(t[5]));
    cudaFree(dev);
  }
  cudaFreeHost(pinned);
}

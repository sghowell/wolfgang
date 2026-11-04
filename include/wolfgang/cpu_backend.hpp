#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace wolfgang {

inline constexpr std::size_t kAutoTbbPairwiseEntryThreshold = 331776;
inline constexpr std::size_t kAutoNeonFullGroupingScalarMinEntries = 1024;
// Auto dispatch sends simplify to the oneTBB kernel at or above this term count;
// below it, thread start-up costs more than the parallel sort saves (crossover
// measured between 4k and 8k terms on a 16-vCPU EPYC 9654; a 26-vCPU host won
// in every case from 10k).
inline constexpr std::size_t kAutoTbbSimplifyTermThreshold = 8192;
// Auto dispatch also needs at least this many oneTBB threads. Two threads beat
// scalar simplify from the term threshold upward on every host measured (one
// core's two hyperthreads and two separate cores on an EPYC 9654 and a Xeon
// 8480+, and an M4 Pro); one thread did not
// (docs/benchmarks/reports/cpu_thread_floor_2026-11-03.md).
inline constexpr int kAutoTbbSimplifyMinConcurrency = 2;
// Pairwise commutation from kAutoTbbPairwiseEntryThreshold entries runs on
// oneTBB with at least kAutoTbbPairwiseMinConcurrency threads, or
// kAutoTbbPairwiseMinConcurrencyOverX86Simd when an AVX2 or AVX-512 kernel
// covers the operator. Each oneTBB thread runs that kernel, and one core's two
// hyperthreads share its vector units, so two threads were 0.82-1.12 times one
// AVX-512 thread there. Arm cores have no SMT: two threads beat NEON by
// 1.54-1.92x on an M4 Pro.
inline constexpr int kAutoTbbPairwiseMinConcurrency = 2;
inline constexpr int kAutoTbbPairwiseMinConcurrencyOverX86Simd = 3;
// One-word operators switch at kAutoTbbPairwiseEntryThreshold too. Once the
// Python commutes_with and commutes_with_into wrote the kernels' output
// straight into the NumPy result, rather than into a zeroed vector copied
// afterwards, the serial page faults and copy were gone, and three oneTBB
// threads beat one AVX-512 thread from 576 x 576 on an EPYC 9654
// (docs/benchmarks/reports/host_commutation_into_2026-11-03.md). The C++
// PauliSum::commutes_with still zero-fills its vector on the calling thread.

struct CpuBackendCandidate {
  std::string name;
  std::string status;
};

struct CpuBackendReport {
  std::string requested_backend;
  std::string active_backend;
  std::vector<std::string> compiled_backends;
  std::vector<std::string> available_backends;
  std::vector<CpuBackendCandidate> candidates;
};

[[nodiscard]] CpuBackendReport cpu_backend_report_from_environment();
[[nodiscard]] CpuBackendReport cpu_backend_report_for_selector(std::string_view selector);

// Raises when WOLFGANG_CPU_BACKEND requests a path that cannot execute in the
// current build/runtime.
// Call this at Python compute boundaries so forced optimized selectors never
// silently fall back to scalar execution.
void ensure_cpu_backend_available_from_environment();

// Raises when WOLFGANG_CPU_BACKEND forces an optimized backend for an
// operation that has no optimized kernel coverage. Auto and scalar selectors
// are accepted because they truthfully execute the portable scalar
// implementation.
void ensure_cpu_backend_supports_scalar_operation(std::string_view operation);

// Like ensure_cpu_backend_supports_scalar_operation, but also accepts a forced
// WOLFGANG_CPU_BACKEND=tbb because simplify has a oneTBB kernel.
void ensure_cpu_backend_supports_simplify();

// True when simplify of num_terms terms should run the oneTBB kernel: auto
// selects it at or above kAutoTbbSimplifyTermThreshold when oneTBB is compiled,
// available and may use at least kAutoTbbSimplifyMinConcurrency threads, and
// WOLFGANG_CPU_BACKEND=tbb forces it. Below the threshold it only reads the
// selector environment variable, so small simplify calls stay cheap.
[[nodiscard]] bool cpu_backend_selects_tbb_simplify(std::size_t num_terms);

// True when oneTBB is available and may use at least `minimum` threads
// (cpu_thread_limit()). Auto dispatch prefers the scalar or SIMD kernels below
// an operation's floor: kAutoTbbSimplifyMinConcurrency for simplify (the
// default), kAutoTbbPairwiseMinConcurrency or
// kAutoTbbPairwiseMinConcurrencyOverX86Simd for pairwise commutation.
[[nodiscard]] bool tbb_auto_has_min_concurrency(int minimum = kAutoTbbSimplifyMinConcurrency);

// Threads the oneTBB CPU kernels may use: the smaller of oneTBB's default
// concurrency (hardware threads, respecting the process affinity mask) and any
// cap from WOLFGANG_NUM_THREADS or set_cpu_thread_limit(). 1 when oneTBB is not
// compiled or not usable in this process (for example after fork()). Throws
// std::invalid_argument when WOLFGANG_NUM_THREADS is set but is not a positive
// integer.
[[nodiscard]] std::size_t cpu_thread_limit();

// Caps the oneTBB CPU kernels at `threads` for the whole process, replacing any
// earlier cap, including one from WOLFGANG_NUM_THREADS. Throws
// std::invalid_argument when threads is 0. Without a usable oneTBB the cap is
// recorded but has nothing to limit.
void set_cpu_thread_limit(std::size_t threads);

// oneTBB does not support fork(): a child forked after the parent ran a oneTBB
// kernel can crash or hang inside oneTBB (macOS libdispatch traps outright).
// Every oneTBB kernel calls this on entry. In a process forked after that, the
// tbb backend reports "unavailable_after_fork": auto dispatch runs the scalar
// or SIMD kernels, whose results are bit-identical, and a forced
// WOLFGANG_CPU_BACKEND=tbb raises.
void note_tbb_kernel_use() noexcept;

}  // namespace wolfgang



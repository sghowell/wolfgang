#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace wolfgang {

inline constexpr std::size_t kAutoTbbPairwiseEntryThreshold = 331776;
inline constexpr std::size_t kAutoNeonFullGroupingScalarMinEntries = 1024;
// Auto dispatch sends simplify to the oneTBB kernel at or above this term count.
inline constexpr std::size_t kAutoTbbSimplifyTermThreshold = 65536;

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
// selects it at or above kAutoTbbSimplifyTermThreshold when oneTBB is compiled
// and available, and WOLFGANG_CPU_BACKEND=tbb forces it.
[[nodiscard]] bool cpu_backend_selects_tbb_simplify(std::size_t num_terms);

}  // namespace wolfgang



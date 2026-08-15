#include "fastpauli/cpu_backend.hpp"

#include <array>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace wolfgang {

namespace {

constexpr std::string_view kBackendEnvVar = "FASTPAULI_CPU_BACKEND";
constexpr std::array<std::string_view, 7> kSelectors = {
    "auto",
    "scalar",
    "tbb",
    "avx2",
    "avx512",
    "neon",
    "sve",
};
constexpr std::array<std::string_view, 6> kBackendCandidates = {
    "scalar",
    "tbb",
    "avx2",
    "avx512",
    "neon",
    "sve",
};

bool selector_is_known(std::string_view selector) noexcept {
  for (std::string_view known : kSelectors) {
    if (selector == known) {
      return true;
    }
  }
  return false;
}

std::string selector_list() {
  std::string values;
  for (std::size_t index = 0; index < kSelectors.size(); ++index) {
    if (index != 0) {
      values += ", ";
    }
    values += kSelectors[index];
  }
  return values;
}

bool backend_is_compiled(std::string_view backend) noexcept {
  if (backend == "scalar") {
    return true;
  }
  if (backend == "tbb") {
    return FASTPAULI_BUILD_TBB_ENABLED != 0;
  }
  if (backend == "avx2") {
    return FASTPAULI_BUILD_AVX2_ENABLED != 0;
  }
  if (backend == "avx512") {
    return FASTPAULI_BUILD_AVX512_ENABLED != 0;
  }
  if (backend == "neon") {
    return FASTPAULI_BUILD_ARM_NEON_ENABLED != 0;
  }
  if (backend == "sve") {
    return FASTPAULI_BUILD_ARM_SVE_ENABLED != 0;
  }
  return false;
}

bool backend_hardware_available(std::string_view backend) noexcept {
  if (backend == "scalar") {
    return true;
  }
  if (backend == "tbb") {
    return true;
  }

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  __builtin_cpu_init();
  if (backend == "avx2") {
    return __builtin_cpu_supports("avx2");
  }
  if (backend == "avx512") {
    return __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vl") &&
           __builtin_cpu_supports("avx512vpopcntdq");
  }
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  if (backend == "neon") {
    return true;
  }
#endif

  return false;
}

std::string backend_status(std::string_view backend) {
  if (!backend_is_compiled(backend)) {
    return "not_compiled";
  }
  if (!backend_hardware_available(backend)) {
    return "hardware_unavailable";
  }
  return "available";
}

std::string requested_backend_from_environment() {
  const char* value = std::getenv(kBackendEnvVar.data());
  if (value == nullptr || value[0] == '\0') {
    return "auto";
  }
  return value;
}

}  // namespace

CpuBackendReport cpu_backend_report_for_selector(std::string_view selector) {
  if (!selector_is_known(selector)) {
    throw std::invalid_argument(
        "FASTPAULI_CPU_BACKEND must be one of: " + selector_list());
  }

  CpuBackendReport report;
  report.requested_backend = std::string(selector);

  for (std::string_view backend : kBackendCandidates) {
    const std::string status = backend_status(backend);
    report.candidates.push_back({std::string(backend), status});
    if (backend_is_compiled(backend)) {
      report.compiled_backends.push_back(std::string(backend));
    }
    if (status == "available") {
      report.available_backends.push_back(std::string(backend));
    }
  }

  if (selector == "auto") {
    // The selector-level report stays scalar for import-time safety. Hot
    // operations apply their own runtime dispatch once dimensions and kernel
    // coverage are known.
    report.active_backend = "scalar";
    return report;
  }

  const std::string forced_status = backend_status(selector);
  if (forced_status != "available") {
    throw std::runtime_error(
        "FASTPAULI_CPU_BACKEND=" + std::string(selector) +
        " requested but the " + std::string(selector) +
        " backend is " + forced_status);
  }

  report.active_backend = std::string(selector);
  return report;
}

CpuBackendReport cpu_backend_report_from_environment() {
  return cpu_backend_report_for_selector(requested_backend_from_environment());
}

void ensure_cpu_backend_available_from_environment() {
  (void)cpu_backend_report_from_environment();
}

void ensure_cpu_backend_supports_scalar_operation(std::string_view operation) {
  const CpuBackendReport report = cpu_backend_report_from_environment();
  if (report.requested_backend == "auto" || report.active_backend == "scalar") {
    return;
  }

  throw std::runtime_error(
      "FASTPAULI_CPU_BACKEND=" + report.active_backend +
      " requested for " + std::string(operation) +
      ", but that operation currently has only scalar CPU coverage; use "
      "FASTPAULI_CPU_BACKEND=auto or FASTPAULI_CPU_BACKEND=scalar unless a "
      "named optimized kernel is documented for the operation.");
}

}  // namespace wolfgang

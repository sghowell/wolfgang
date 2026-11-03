#include "wolfgang/cpu_backend.hpp"

#include <array>
#include <atomic>
#if WOLFGANG_BUILD_TBB_ENABLED
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/task_arena.h>
#endif
#if !defined(_WIN32)
#include <pthread.h>
#endif
#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>

namespace wolfgang {

namespace {

constexpr std::string_view kBackendEnvVar = "WOLFGANG_CPU_BACKEND";
constexpr std::string_view kThreadsEnvVar = "WOLFGANG_NUM_THREADS";

// The process-wide thread cap. The oneTBB global_control that enforces it is
// heap-allocated and never destroyed at exit: destroying it after oneTBB's own
// teardown is unsafe, and the cap lives as long as the process anyway.
struct ThreadLimitState {
  std::mutex mutex;
  bool environment_read = false;
  std::size_t cap = 0;  // 0: no cap
#if WOLFGANG_BUILD_TBB_ENABLED
  oneapi::tbb::global_control* control = nullptr;
#endif
};

ThreadLimitState& thread_limit_state() {
  static auto* state = new ThreadLimitState();
  return *state;
}

std::size_t parse_thread_count(std::string_view text) {
  std::size_t value = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || value == 0) {
    throw std::invalid_argument(std::string(kThreadsEnvVar) + " must be a positive integer, got '" +
                                std::string(text) + "'");
  }
  return value;
}

// Set by the first oneTBB kernel in this process (or inherited across fork).
std::atomic<bool> g_tbb_used{false};
// Set in a child forked after oneTBB ran: its worker threads did not survive.
std::atomic<bool> g_tbb_unusable_after_fork{false};

void mark_tbb_unusable_in_forked_child() noexcept {
  if (g_tbb_used.load(std::memory_order_relaxed)) {
    g_tbb_unusable_after_fork.store(true, std::memory_order_relaxed);
  }
}
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
    return WOLFGANG_BUILD_TBB_ENABLED != 0;
  }
  if (backend == "avx2") {
    return WOLFGANG_BUILD_AVX2_ENABLED != 0;
  }
  if (backend == "avx512") {
    return WOLFGANG_BUILD_AVX512_ENABLED != 0;
  }
  if (backend == "neon") {
    return WOLFGANG_BUILD_ARM_NEON_ENABLED != 0;
  }
  if (backend == "sve") {
    return WOLFGANG_BUILD_ARM_SVE_ENABLED != 0;
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
  if (backend == "tbb" && g_tbb_unusable_after_fork.load(std::memory_order_relaxed)) {
    return "unavailable_after_fork";
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
    throw std::invalid_argument("WOLFGANG_CPU_BACKEND must be one of: " + selector_list());
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
        "WOLFGANG_CPU_BACKEND=" + std::string(selector) +
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
      "WOLFGANG_CPU_BACKEND=" + report.active_backend +
      " requested for " + std::string(operation) +
      ", but that operation currently has only scalar CPU coverage; use "
      "WOLFGANG_CPU_BACKEND=auto or WOLFGANG_CPU_BACKEND=scalar unless a "
      "named optimized kernel is documented for the operation.");
}

void ensure_cpu_backend_supports_simplify() {
  const CpuBackendReport report = cpu_backend_report_from_environment();
  if (report.active_backend == "tbb") {
    return;
  }
  ensure_cpu_backend_supports_scalar_operation("simplify");
}

namespace {

// Applies the cap to oneTBB when oneTBB is usable here. Callers hold the
// state mutex.
void apply_thread_cap_locked(ThreadLimitState& state) {
#if WOLFGANG_BUILD_TBB_ENABLED
  if (state.cap == 0 || backend_status("tbb") != "available") {
    return;
  }
  delete state.control;
  state.control = new oneapi::tbb::global_control(
      oneapi::tbb::global_control::max_allowed_parallelism, state.cap);
#else
  (void)state;
#endif
}

// Reads WOLFGANG_NUM_THREADS once. An invalid value throws on every call
// until it is fixed, as an invalid WOLFGANG_CPU_BACKEND does.
void read_thread_cap_from_environment_locked(ThreadLimitState& state) {
  if (state.environment_read) {
    return;
  }
  const char* raw = std::getenv(kThreadsEnvVar.data());
  if (raw != nullptr && *raw != '\0') {
    state.cap = parse_thread_count(raw);
    apply_thread_cap_locked(state);
  }
  state.environment_read = true;
}

void apply_thread_cap_from_environment() {
  ThreadLimitState& state = thread_limit_state();
  const std::lock_guard<std::mutex> lock(state.mutex);
  read_thread_cap_from_environment_locked(state);
}

}  // namespace

std::size_t cpu_thread_limit() {
  ThreadLimitState& state = thread_limit_state();
  const std::lock_guard<std::mutex> lock(state.mutex);
  read_thread_cap_from_environment_locked(state);
#if WOLFGANG_BUILD_TBB_ENABLED
  if (backend_status("tbb") != "available") {
    return 1;
  }
  const std::size_t allowed = oneapi::tbb::global_control::active_value(
      oneapi::tbb::global_control::max_allowed_parallelism);
  const auto arena = static_cast<std::size_t>(oneapi::tbb::this_task_arena::max_concurrency());
  return std::max<std::size_t>(1, std::min(allowed, arena));
#else
  return 1;
#endif
}

void set_cpu_thread_limit(std::size_t threads) {
  if (threads == 0) {
    throw std::invalid_argument("the CPU thread limit must be at least 1");
  }
  ThreadLimitState& state = thread_limit_state();
  const std::lock_guard<std::mutex> lock(state.mutex);
  state.environment_read = true;  // an explicit cap replaces WOLFGANG_NUM_THREADS
  state.cap = threads;
  apply_thread_cap_locked(state);
}

bool cpu_backend_selects_tbb_simplify(std::size_t num_terms) {
#if WOLFGANG_BUILD_TBB_ENABLED
  const char* raw_selector = std::getenv(kBackendEnvVar.data());
  const std::string_view selector = raw_selector == nullptr ? std::string_view{} : raw_selector;
  if (selector == "tbb") {
    apply_thread_cap_from_environment();
    return cpu_backend_report_from_environment().active_backend == "tbb";
  }
  if (!(selector.empty() || selector == "auto") || num_terms < kAutoTbbSimplifyTermThreshold) {
    return false;
  }
  return tbb_auto_has_min_concurrency();
#else
  (void)num_terms;
  return false;
#endif
}

bool tbb_auto_has_min_concurrency() {
#if WOLFGANG_BUILD_TBB_ENABLED
  // Check availability first: after a fork, querying oneTBB itself is unsafe.
  if (backend_status("tbb") != "available") {
    return false;
  }
  return cpu_thread_limit() >= static_cast<std::size_t>(kAutoTbbSimplifyMinConcurrency);
#else
  return false;
#endif
}

void note_tbb_kernel_use() noexcept {
#if !defined(_WIN32)
  static const bool registered = [] {
    (void)pthread_atfork(nullptr, nullptr, &mark_tbb_unusable_in_forked_child);
    return true;
  }();
  (void)registered;
#endif
  g_tbb_used.store(true, std::memory_order_relaxed);
  // Forced and commutation kernels reach oneTBB without auto dispatch, so the
  // cap from WOLFGANG_NUM_THREADS is applied here too; an invalid value has
  // already raised at the dispatch or capability check that preceded the
  // kernel, or is reported by get_num_threads().
  try {
    apply_thread_cap_from_environment();
  } catch (...) {
  }
}

}  // namespace wolfgang

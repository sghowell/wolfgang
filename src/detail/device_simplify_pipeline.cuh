#pragma once

// Multi-launch simplify for operators the fused kernel (small_simplify_kernel.cuh)
// does not take, shared by the CUDA and HIP backends and compiled only by nvcc
// or hipcc. Every intermediate count stays on the device:
//
//   1. one reduction yields the largest magnitude (NaN when a component is not
//      finite) and the largest half magnitude;
//   2. a pack kernel writes the sort keys; its first thread turns that
//      reduction into the CPU drop rule (src/detail/simplify_numeric.hpp);
//   3. a stable sort orders terms canonically, x word then z word per packed
//      word, with equal terms in input order;
//   4. reduce-by-key sums each duplicate run and writes the run count to
//      device memory;
//   5. an inclusive scan over "run survives" flags assigns each surviving run
//      its output position. It covers the whole input length, with runs past
//      the device-side run count flagged 0, so no stage needs a count on the
//      host;
//   6. a one-thread kernel publishes the counts and status flags to mapped
//      host memory.
//
// The host then synchronizes once, allocates exactly `survivors` output terms,
// and one kernel scatters the surviving runs into them. The Thrust pipeline
// this replaces read the maximum, the run count and the survivor count back
// between stages, eight synchronizations per call (see
// docs/benchmarks/reports/cuda_small_op_profile_h100_2026-11-03.md).
//
// Duplicate runs are summed by reduce-by-key in an implementation-defined tree
// order (docs/architecture/cuda_backend.md); canonical order, the drop rule and
// the error contract match CPU simplify exactly.

#include <thrust/complex.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>

#if defined(__HIPCC__)
#include <hipcub/hipcub.hpp>
#else
#include <cub/cub.cuh>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>

namespace wolfgang::accelerator_detail {

#if defined(__HIPCC__)
namespace gpucub = ::hipcub;
#else
namespace gpucub = ::cub;
#endif

using PipelineComplex = thrust::complex<double>;

__host__ __device__ inline bool pipeline_finite(PipelineComplex value) noexcept {
  return ::isfinite(value.real()) && ::isfinite(value.imag());
}

// Inputs to the drop rule. max_abs is NaN when any component is not finite,
// which reports invalid input; a finite coefficient whose magnitude overflows
// contributes +inf. max_half_abs feeds the half-unit rule used when
// atol + rtol * max_abs overflows.
struct CoeffStats {
  double max_abs;
  double max_half_abs;
};

struct CoeffStatsOf {
  __host__ __device__ CoeffStats operator()(PipelineComplex value) const noexcept {
    return {pipeline_finite(value) ? thrust::abs(value) : ::nan(""),
            ::hypot(value.real() * 0.5, value.imag() * 0.5)};
  }
};

struct CoeffStatsMax {
  __host__ __device__ CoeffStats operator()(const CoeffStats& lhs, const CoeffStats& rhs) const noexcept {
    double max_abs = lhs.max_abs;
    if (max_abs == max_abs && (rhs.max_abs != rhs.max_abs || max_abs < rhs.max_abs)) {
      max_abs = rhs.max_abs;  // NaN propagates from either side
    }
    return {max_abs, lhs.max_half_abs < rhs.max_half_abs ? rhs.max_half_abs : lhs.max_half_abs};
  }
};

struct DeviceDropRule {
  double drop_threshold;
  double half_threshold;
  int use_half_units;
};

// Mirrors SimplifyDropRule::from_coefficients. The volatile products keep
// nvcc and HIP clang from contracting atol + rtol * scale into a fused
// multiply-add, which would move the threshold by an ulp.
__host__ __device__ inline DeviceDropRule make_drop_rule(const CoeffStats& stats, double atol, double rtol) {
  DeviceDropRule rule{atol, 0.0, 0};
  if (rtol != 0.0) {
    const volatile double scaled = rtol * stats.max_abs;
    rule.drop_threshold = atol + scaled;
  }
  if (!::isfinite(rule.drop_threshold)) {
    const double half_atol = atol * 0.5;
    const volatile double half_scaled = rtol * stats.max_half_abs;
    rule.half_threshold = half_atol + half_scaled;
    rule.drop_threshold = rule.half_threshold * 2.0;
    rule.use_half_units = 1;
  }
  return rule;
}

// A sum that is not finite raises OverflowError on the CPU. Tree-ordered sums
// can even meet as inf + -inf = NaN, which the magnitude test alone would drop
// silently, so every sum passes through here and sets the overflow flag.
__host__ __device__ inline bool run_survives(const DeviceDropRule& rule, PipelineComplex sum, int* overflow) {
  if (!pipeline_finite(sum)) {
    *overflow = 1;
    return false;
  }
  return rule.use_half_units ? ::hypot(sum.real() * 0.5, sum.imag() * 0.5) > rule.half_threshold
                             : thrust::abs(sum) > rule.drop_threshold;
}

// Device-resident state of one call, inside its scratch allocation.
struct PipelineState {
  CoeffStats stats;
  DeviceDropRule rule;
  unsigned long long runs;
  int non_finite_input;
  int overflow;
};

// What the host reads after the one synchronization (mapped host memory).
struct PipelineResult {
  unsigned long long runs;
  unsigned long long survivors;
  int non_finite_input;
  int overflow;
};

// Sort key kinds. Each loads the key of input term `term` and stores a run's
// key at output term `out`.

// One word and at most 32 qubits: key = x << num_qubits | z keeps the
// canonical x-then-z order in 2 * num_qubits bits, so radix sort processes
// only those bits. Bits above num_qubits are zero in every stored word.
struct PackedKeys {
  using Key = std::uint64_t;
  using RunKey = Key;
  unsigned num_qubits;

  __device__ Key load(const std::uint64_t* x, const std::uint64_t* z, std::size_t term) const noexcept {
    return (x[term] << num_qubits) | z[term];
  }
  __device__ void store(Key key, std::size_t out, std::uint64_t* out_x, std::uint64_t* out_z) const noexcept {
    out_x[out] = key >> num_qubits;
    out_z[out] = key & ((Key{1} << num_qubits) - 1);
  }
};

struct WordPairKey {
  std::uint64_t x;
  std::uint64_t z;

  friend __host__ __device__ bool operator==(const WordPairKey& lhs, const WordPairKey& rhs) noexcept {
    return lhs.x == rhs.x && lhs.z == rhs.z;
  }
};

struct WordPairLess {
  __host__ __device__ bool operator()(const WordPairKey& lhs, const WordPairKey& rhs) const noexcept {
    return lhs.x != rhs.x ? lhs.x < rhs.x : lhs.z < rhs.z;
  }
};

struct OneWordKeys {
  using Key = WordPairKey;
  using RunKey = Key;
  using Less = WordPairLess;

  __device__ Key load(const std::uint64_t* x, const std::uint64_t* z, std::size_t term) const noexcept {
    return {x[term], z[term]};
  }
  __device__ void store(const Key& key, std::size_t out, std::uint64_t* out_x, std::uint64_t* out_z) const noexcept {
    out_x[out] = key.x;
    out_z[out] = key.z;
  }
};

struct TwoWordKey {
  std::uint64_t x0;
  std::uint64_t z0;
  std::uint64_t x1;
  std::uint64_t z1;

  friend __host__ __device__ bool operator==(const TwoWordKey& lhs, const TwoWordKey& rhs) noexcept {
    return lhs.x0 == rhs.x0 && lhs.z0 == rhs.z0 && lhs.x1 == rhs.x1 && lhs.z1 == rhs.z1;
  }
};

struct TwoWordLess {
  __host__ __device__ bool operator()(const TwoWordKey& lhs, const TwoWordKey& rhs) const noexcept {
    if (lhs.x0 != rhs.x0) {
      return lhs.x0 < rhs.x0;
    }
    if (lhs.z0 != rhs.z0) {
      return lhs.z0 < rhs.z0;
    }
    return lhs.x1 != rhs.x1 ? lhs.x1 < rhs.x1 : lhs.z1 < rhs.z1;
  }
};

struct TwoWordKeys {
  using Key = TwoWordKey;
  using RunKey = Key;
  using Less = TwoWordLess;

  __device__ Key load(const std::uint64_t* x, const std::uint64_t* z, std::size_t term) const noexcept {
    const std::size_t offset = term * 2;
    return {x[offset], z[offset], x[offset + 1], z[offset + 1]};
  }
  __device__ void store(const Key& key, std::size_t out, std::uint64_t* out_x, std::uint64_t* out_z) const noexcept {
    const std::size_t offset = out * 2;
    out_x[offset] = key.x0;
    out_z[offset] = key.z0;
    out_x[offset + 1] = key.x1;
    out_z[offset + 1] = key.z1;
  }
};

// Wider operators sort term indices. CUB's reduce-by-key compares keys with
// operator==, so runs are keyed by a reference that carries the word arrays
// and compares the referenced terms; the first reference of a run names its
// representative (lowest-index) term.
struct GenericTermRef {
  const std::uint64_t* x;
  const std::uint64_t* z;
  std::size_t words;
  std::size_t index;

  friend __host__ __device__ bool operator==(const GenericTermRef& lhs, const GenericTermRef& rhs) noexcept {
    const std::size_t lhs_offset = lhs.index * lhs.words;
    const std::size_t rhs_offset = rhs.index * rhs.words;
    for (std::size_t word = 0; word < lhs.words; ++word) {
      if (lhs.x[lhs_offset + word] != rhs.x[rhs_offset + word] ||
          lhs.z[lhs_offset + word] != rhs.z[rhs_offset + word]) {
        return false;
      }
    }
    return true;
  }
};

struct MakeGenericTermRef {
  const std::uint64_t* x;
  const std::uint64_t* z;
  std::size_t words;

  __host__ __device__ GenericTermRef operator()(std::size_t index) const noexcept {
    return {x, z, words, index};
  }
};

// Ties break on input index, so equal terms keep input order and the
// comparison sort is deterministic.
struct GenericIndexLess {
  const std::uint64_t* x;
  const std::uint64_t* z;
  std::size_t words;

  __host__ __device__ bool operator()(std::size_t lhs, std::size_t rhs) const noexcept {
    const std::size_t lhs_offset = lhs * words;
    const std::size_t rhs_offset = rhs * words;
    for (std::size_t word = 0; word < words; ++word) {
      const std::uint64_t lhs_x = x[lhs_offset + word];
      const std::uint64_t rhs_x = x[rhs_offset + word];
      if (lhs_x != rhs_x) {
        return lhs_x < rhs_x;
      }
      const std::uint64_t lhs_z = z[lhs_offset + word];
      const std::uint64_t rhs_z = z[rhs_offset + word];
      if (lhs_z != rhs_z) {
        return lhs_z < rhs_z;
      }
    }
    return lhs < rhs;
  }
};

struct GenericKeys {
  using Key = std::size_t;
  using RunKey = GenericTermRef;

  __device__ Key load(const std::uint64_t*, const std::uint64_t*, std::size_t term) const noexcept {
    return term;
  }
};

template <typename Kind>
__global__ void pipeline_pack_kernel(
    const std::uint64_t* x,
    const std::uint64_t* z,
    const PipelineComplex* coeffs,
    std::size_t num_terms,
    Kind kind,
    typename Kind::Key* keys,
    PipelineComplex* values,
    PipelineState* state,
    double atol,
    double rtol) {
  const std::size_t term =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + static_cast<std::size_t>(threadIdx.x);
  if (term == 0) {
    // The reduction finished before this launch (same stream), and every
    // later stage runs after it, so one thread can publish the rule.
    const CoeffStats stats = state->stats;
    state->rule = make_drop_rule(stats, atol, rtol);
    state->non_finite_input = stats.max_abs != stats.max_abs ? 1 : 0;
    state->overflow = 0;
  }
  if (term >= num_terms) {
    return;
  }
  keys[term] = kind.load(x, z, term);
  if (values != nullptr) {
    values[term] = coeffs[term];
  }
}

struct RunSurvivesFlag {
  const PipelineComplex* sums;
  PipelineState* state;

  __host__ __device__ unsigned long long operator()(unsigned long long run) const {
    if (run >= state->runs) {
      return 0;
    }
    return run_survives(state->rule, sums[run], &state->overflow) ? 1 : 0;
  }
};

__global__ inline void pipeline_publish_kernel(
    const PipelineState* state,
    const unsigned long long* positions,
    std::size_t num_terms,
    PipelineResult* result) {
  result->runs = state->runs;
  result->survivors = positions[num_terms - 1];
  result->non_finite_input = state->non_finite_input;
  result->overflow = state->overflow;
}

// A run survives when the scan steps at its position; its output term is the
// scan value minus one.
template <typename Kind>
__global__ void pipeline_scatter_kernel(
    const typename Kind::RunKey* run_keys,
    const PipelineComplex* run_sums,
    const unsigned long long* positions,
    std::size_t runs,
    Kind kind,
    std::uint64_t* out_x,
    std::uint64_t* out_z,
    PipelineComplex* out_coeffs) {
  const std::size_t run =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + static_cast<std::size_t>(threadIdx.x);
  if (run >= runs) {
    return;
  }
  const unsigned long long end = positions[run];
  if (end == (run == 0 ? 0ULL : positions[run - 1])) {
    return;
  }
  const std::size_t out = static_cast<std::size_t>(end - 1);
  kind.store(run_keys[run], out, out_x, out_z);
  out_coeffs[out] = run_sums[run];
}

// One thread per (run, word), so wide terms copy with coalesced accesses.
__global__ inline void pipeline_scatter_generic_kernel(
    const GenericTermRef* run_keys,
    const PipelineComplex* run_sums,
    const unsigned long long* positions,
    std::size_t runs,
    std::size_t words,
    std::uint64_t* out_x,
    std::uint64_t* out_z,
    PipelineComplex* out_coeffs) {
  const std::size_t thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + static_cast<std::size_t>(threadIdx.x);
  const std::size_t run = thread / words;
  if (run >= runs) {
    return;
  }
  const std::size_t word = thread - run * words;
  const unsigned long long end = positions[run];
  if (end == (run == 0 ? 0ULL : positions[run - 1])) {
    return;
  }
  const std::size_t out = static_cast<std::size_t>(end - 1);
  const GenericTermRef ref = run_keys[run];
  out_x[out * words + word] = ref.x[ref.index * words + word];
  out_z[out * words + word] = ref.z[ref.index * words + word];
  if (word == 0) {
    out_coeffs[out] = run_sums[run];
  }
}

struct PipelineInput {
  const std::uint64_t* x;
  const std::uint64_t* z;
  const PipelineComplex* coeffs;
  std::size_t num_terms;
  std::size_t words;
  std::size_t num_qubits;
};

struct PipelineOutput {
  std::uint64_t* x;
  std::uint64_t* z;
  PipelineComplex* coeffs;
};

// Byte offsets of the scratch arrays inside one allocation.
class PipelineScratchPlan {
 public:
  template <typename T>
  std::size_t add(std::size_t count) {
    constexpr std::size_t kAlignment = 256;
    const std::size_t offset = (bytes_ + kAlignment - 1) / kAlignment * kAlignment;
    if (offset < bytes_ || count > (std::numeric_limits<std::size_t>::max() - offset) / sizeof(T)) {
      throw std::bad_alloc();
    }
    bytes_ = offset + count * sizeof(T);
    return offset;
  }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

 private:
  std::size_t bytes_ = 0;
};

// A Backend supplies:
//   static void check(<runtime error type>, const char*);  // throws on failure
//   static void check_launch(const char*);
//   static void synchronize_stream(const char*);  // the stream Wolfgang uses
//   static void synchronize_device(const char*);  // before returning results
//   static int launch_blocks(std::size_t threads, const char*);
//   static constexpr int kThreadsPerBlock;
//   class Scratch { explicit Scratch(std::size_t bytes); std::byte* data() const; };
//   static auto thrust_policy();
//   static PipelineResult* result_device();       // mapped, per thread
//   static const volatile PipelineResult* result_host();
template <typename Backend, typename Kind, typename Allocate>
std::size_t run_pipeline(const PipelineInput& in, Kind kind, double atol, double rtol, Allocate&& allocate) {
  using Key = typename Kind::Key;
  using RunKey = typename Kind::RunKey;
  constexpr bool kPacked = std::is_same_v<Kind, PackedKeys>;
  constexpr bool kGeneric = std::is_same_v<Kind, GenericKeys>;
  const std::size_t n = in.num_terms;

  const auto stats_in = thrust::make_transform_iterator(in.coeffs, CoeffStatsOf{});
  const auto reduce_inputs = [&](Key* keys, PipelineComplex* values) {
    if constexpr (kGeneric) {
      return thrust::make_tuple(
          thrust::make_transform_iterator(keys, MakeGenericTermRef{in.x, in.z, in.words}),
          thrust::make_permutation_iterator(in.coeffs, keys));
    } else {
      return thrust::make_tuple(keys, static_cast<const PipelineComplex*>(values));
    }
  };
  const auto flags = [](const PipelineComplex* sums, PipelineState* state) {
    return thrust::make_transform_iterator(thrust::make_counting_iterator(0ULL), RunSurvivesFlag{sums, state});
  };
  const int end_bit = kPacked ? static_cast<int>(2 * in.num_qubits) : 0;

  // Size every CUB stage first (no device work), then carve one allocation.
  std::size_t temp_bytes = 0;
  {
    std::size_t bytes = 0;
    Backend::check(
        gpucub::DeviceReduce::Reduce(
            nullptr, bytes, stats_in, static_cast<CoeffStats*>(nullptr), n, CoeffStatsMax{}, CoeffStats{0.0, 0.0}),
        "size simplify input reduction");
    temp_bytes = std::max(temp_bytes, bytes);
    if constexpr (kPacked) {
      Backend::check(
          gpucub::DeviceRadixSort::SortPairs(
              nullptr, bytes, static_cast<const Key*>(nullptr), static_cast<Key*>(nullptr),
              static_cast<const PipelineComplex*>(nullptr), static_cast<PipelineComplex*>(nullptr), n, 0, end_bit),
          "size simplify radix sort");
      temp_bytes = std::max(temp_bytes, bytes);
    }
    const auto inputs = reduce_inputs(nullptr, nullptr);
    Backend::check(
        gpucub::DeviceReduce::ReduceByKey(
            nullptr, bytes, thrust::get<0>(inputs), static_cast<RunKey*>(nullptr), thrust::get<1>(inputs),
            static_cast<PipelineComplex*>(nullptr), static_cast<unsigned long long*>(nullptr),
            thrust::plus<PipelineComplex>{}, n),
        "size simplify reduce-by-key");
    temp_bytes = std::max(temp_bytes, bytes);
    Backend::check(
        gpucub::DeviceScan::InclusiveSum(
            nullptr, bytes, flags(nullptr, nullptr), static_cast<unsigned long long*>(nullptr), n),
        "size simplify survivor scan");
    temp_bytes = std::max(temp_bytes, bytes);
  }

  // Positions reuse the key array, which no stage reads after reduce-by-key.
  static_assert(sizeof(Key) >= sizeof(unsigned long long));
  PipelineScratchPlan plan;
  const std::size_t keys_at = plan.add<Key>(n);
  const std::size_t sorted_keys_at = kPacked ? plan.add<Key>(n) : 0;
  const std::size_t values_at = kGeneric ? 0 : plan.add<PipelineComplex>(n);
  const std::size_t run_keys_at = plan.add<RunKey>(n);
  const std::size_t run_sums_at = plan.add<PipelineComplex>(n);
  const std::size_t state_at = plan.add<PipelineState>(1);
  const std::size_t temp_at = plan.add<std::byte>(temp_bytes);
  typename Backend::Scratch scratch(plan.bytes());
  std::byte* base = scratch.data();
  Key* keys = reinterpret_cast<Key*>(base + keys_at);
  Key* sorted_keys = kPacked ? reinterpret_cast<Key*>(base + sorted_keys_at) : keys;
  PipelineComplex* values = kGeneric ? nullptr : reinterpret_cast<PipelineComplex*>(base + values_at);
  RunKey* run_keys = reinterpret_cast<RunKey*>(base + run_keys_at);
  PipelineComplex* run_sums = reinterpret_cast<PipelineComplex*>(base + run_sums_at);
  PipelineState* state = reinterpret_cast<PipelineState*>(base + state_at);
  void* temp = temp_bytes == 0 ? nullptr : static_cast<void*>(base + temp_at);
  unsigned long long* positions = reinterpret_cast<unsigned long long*>(keys);

  std::size_t bytes = temp_bytes;
  Backend::check(
      gpucub::DeviceReduce::Reduce(temp, bytes, stats_in, &state->stats, n, CoeffStatsMax{}, CoeffStats{0.0, 0.0}),
      "reduce simplify input");

  const int pack_blocks = Backend::launch_blocks(n, "simplify pack");
  pipeline_pack_kernel<Kind><<<pack_blocks, Backend::kThreadsPerBlock>>>(
      in.x, in.z, in.coeffs, n, kind, keys, kPacked ? nullptr : values, state, atol, rtol);
  Backend::check_launch("launch simplify pack");

  if constexpr (kPacked) {
    bytes = temp_bytes;
    Backend::check(
        gpucub::DeviceRadixSort::SortPairs(temp, bytes, keys, sorted_keys, in.coeffs, values, n, 0, end_bit),
        "radix sort simplify keys");
  } else if constexpr (kGeneric) {
    thrust::sort(Backend::thrust_policy(), keys, keys + n, GenericIndexLess{in.x, in.z, in.words});
  } else {
    thrust::stable_sort_by_key(Backend::thrust_policy(), keys, keys + n, values, typename Kind::Less{});
  }

  const auto inputs = reduce_inputs(sorted_keys, values);
  bytes = temp_bytes;
  Backend::check(
      gpucub::DeviceReduce::ReduceByKey(
          temp, bytes, thrust::get<0>(inputs), run_keys, thrust::get<1>(inputs), run_sums, &state->runs,
          thrust::plus<PipelineComplex>{}, n),
      "reduce simplify duplicate runs");

  bytes = temp_bytes;
  Backend::check(
      gpucub::DeviceScan::InclusiveSum(temp, bytes, flags(run_sums, state), positions, n),
      "scan surviving simplify runs");

  pipeline_publish_kernel<<<1, 1>>>(state, positions, n, Backend::result_device());
  Backend::check_launch("launch simplify publish");
  Backend::synchronize_stream("synchronize simplify counts");

  const volatile PipelineResult* result = Backend::result_host();
  if (result->non_finite_input != 0) {
    throw std::invalid_argument("simplify requires finite coefficients");
  }
  if (result->overflow != 0) {
    throw std::overflow_error("simplify coefficient accumulation overflowed complex128");
  }
  const std::size_t survivors = static_cast<std::size_t>(result->survivors);
  const std::size_t runs = static_cast<std::size_t>(result->runs);
  if (survivors == 0) {
    return 0;
  }

  const PipelineOutput out = allocate(survivors);
  if constexpr (kGeneric) {
    const int blocks = Backend::launch_blocks(runs * in.words, "simplify scatter");
    pipeline_scatter_generic_kernel<<<blocks, Backend::kThreadsPerBlock>>>(
        run_keys, run_sums, positions, runs, in.words, out.x, out.z, out.coeffs);
  } else {
    const int blocks = Backend::launch_blocks(runs, "simplify scatter");
    pipeline_scatter_kernel<Kind><<<blocks, Backend::kThreadsPerBlock>>>(
        run_keys, run_sums, positions, runs, kind, out.x, out.z, out.coeffs);
  }
  Backend::check_launch("launch simplify scatter");
  Backend::synchronize_device("synchronize after simplify");
  return survivors;
}

// Simplifies `in` (at least one term, at least one packed word) and returns the
// survivor count. `allocate(survivors)` is called once, only when that count is
// nonzero, and returns output buffers for that many terms. Throws
// std::invalid_argument for non-finite input and std::overflow_error when a
// duplicate sum overflows, as CPU simplify does.
template <typename Backend, typename Allocate>
std::size_t device_simplify_pipeline(const PipelineInput& in, double atol, double rtol, Allocate&& allocate) {
  if (in.words == 1 && in.num_qubits <= 32) {
    return run_pipeline<Backend>(in, PackedKeys{static_cast<unsigned>(in.num_qubits)}, atol, rtol, allocate);
  }
  if (in.words == 1) {
    return run_pipeline<Backend>(in, OneWordKeys{}, atol, rtol, allocate);
  }
  if (in.words == 2) {
    return run_pipeline<Backend>(in, TwoWordKeys{}, atol, rtol, allocate);
  }
  return run_pipeline<Backend>(in, GenericKeys{}, atol, rtol, allocate);
}

}  // namespace wolfgang::accelerator_detail

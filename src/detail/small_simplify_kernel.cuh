#pragma once

// Fused simplify for small operators, shared by the CUDA and HIP backends and
// compiled only by nvcc or hipcc. One thread block of
// small_simplify_capacity(words) threads handles up to that many terms:
//
//   1. load the packed keys into shared memory and reduce the largest
//      magnitude (NaN when any component is not finite, which reports invalid
//      input, as CPU simplify's ValueError);
//   2. sort (key, input index) with an all-ascending bitonic network in shared
//      memory; positions past the term count are +inf sentinels;
//   3. gather the coefficients into sorted order;
//   4. each run head sums its run sequentially in input order, which is the
//      CPU's order, so coefficients match CPU simplify bit for bit, and applies
//      the CPU drop rule (an overflowing sum reports OverflowError);
//   5. a block scan compacts survivors into the outputs.
//
// One launch and one small result copy replace the separate sort, reduce and
// compaction launches and the synchronizations between them, which set the
// floor for small operators on both backends.

#include <thrust/complex.h>

#include <cstddef>
#include <cstdint>

namespace wolfgang::accelerator_detail {

inline constexpr int kSmallSimplifyMaxWords = 4;

// Threads per block, and the most terms the fused path accepts (a power of
// two). Two or fewer words fit 1,024 terms in 56 KiB of shared memory.
__host__ __device__ constexpr int small_simplify_capacity(int words) noexcept {
  return words <= 2 ? 1024 : 512;
}

// Shared memory per block position: keys, coefficient, index and scan slot.
__host__ __device__ constexpr std::size_t small_simplify_bytes_per_position(int words) noexcept {
  return 2 * static_cast<std::size_t>(words) * sizeof(std::uint64_t) +
      sizeof(thrust::complex<double>) + 2 * sizeof(std::uint32_t);
}

__host__ __device__ constexpr std::size_t small_simplify_shared_bytes(int words) noexcept {
  return static_cast<std::size_t>(small_simplify_capacity(words)) *
      small_simplify_bytes_per_position(words);
}

// Block size for `terms` inputs: the next power of two, at least one AMD
// wavefront (64), so small operators run fewer sort stages in less shared
// memory than the full capacity.
__host__ __device__ constexpr int small_simplify_block_threads(std::size_t terms) noexcept {
  int threads = 64;
  while (static_cast<std::size_t>(threads) < terms) {
    threads <<= 1;
  }
  return threads;
}

[[nodiscard]] constexpr bool small_simplify_applies(std::size_t words, std::size_t terms) noexcept {
  return words >= 1 && words <= kSmallSimplifyMaxWords && terms >= 1 &&
      terms <= static_cast<std::size_t>(small_simplify_capacity(static_cast<int>(words)));
}

enum SmallSimplifyStatus : int {
  kSmallSimplifyOk = 0,
  kSmallSimplifyNonFiniteInput = 1,
  kSmallSimplifyOverflow = 2,
};

// Two 8-byte fields so the kernel writes every byte the host copies back.
struct SmallSimplifyResult {
  unsigned long long survivors;
  unsigned long long status;
};

namespace small_simplify {

template <int Words>
__device__ inline bool key_less(
    const std::uint64_t* kx,
    const std::uint64_t* kz,
    const std::uint32_t* index,
    int lhs,
    int rhs) {
  // Canonical order: word by word, x before z, then input index.
  for (int word = 0; word < Words; ++word) {
    const std::uint64_t lhs_x = kx[lhs * Words + word];
    const std::uint64_t rhs_x = kx[rhs * Words + word];
    if (lhs_x != rhs_x) {
      return lhs_x < rhs_x;
    }
    const std::uint64_t lhs_z = kz[lhs * Words + word];
    const std::uint64_t rhs_z = kz[rhs * Words + word];
    if (lhs_z != rhs_z) {
      return lhs_z < rhs_z;
    }
  }
  return index[lhs] < index[rhs];
}

template <int Words>
__device__ inline bool keys_equal(const std::uint64_t* kx, const std::uint64_t* kz, int lhs, int rhs) {
  for (int word = 0; word < Words; ++word) {
    if (kx[lhs * Words + word] != kx[rhs * Words + word] ||
        kz[lhs * Words + word] != kz[rhs * Words + word]) {
      return false;
    }
  }
  return true;
}

template <int Words>
__device__ inline void swap_entries(std::uint64_t* kx, std::uint64_t* kz, std::uint32_t* index, int lhs, int rhs) {
  for (int word = 0; word < Words; ++word) {
    const std::uint64_t x = kx[lhs * Words + word];
    kx[lhs * Words + word] = kx[rhs * Words + word];
    kx[rhs * Words + word] = x;
    const std::uint64_t z = kz[lhs * Words + word];
    kz[lhs * Words + word] = kz[rhs * Words + word];
    kz[rhs * Words + word] = z;
  }
  const std::uint32_t i = index[lhs];
  index[lhs] = index[rhs];
  index[rhs] = i;
}

__device__ inline bool components_finite(thrust::complex<double> value) {
  return ::isfinite(value.real()) && ::isfinite(value.imag());
}

__device__ inline double max_propagating_nan(double lhs, double rhs) {
  if (lhs != lhs) {
    return lhs;
  }
  if (rhs != rhs) {
    return rhs;
  }
  return lhs < rhs ? rhs : lhs;
}

}  // namespace small_simplify

// Launch with one block of small_simplify_block_threads(terms) threads (a power
// of two, at most small_simplify_capacity(Words)) and that many positions of
// small_simplify_bytes_per_position(Words) dynamic shared memory, for
// 1 <= terms <= capacity. Outputs hold `terms` entries; result->survivors of
// them are written.
template <int Words>
__global__ void small_simplify_kernel(
    const std::uint64_t* in_x,
    const std::uint64_t* in_z,
    const thrust::complex<double>* in_coeffs,
    std::uint32_t terms,
    double atol,
    double rtol,
    std::uint64_t* out_x,
    std::uint64_t* out_z,
    thrust::complex<double>* out_coeffs,
    SmallSimplifyResult* result) {
  using namespace small_simplify;
  extern __shared__ double2 small_simplify_shared[];
  const int capacity = static_cast<int>(blockDim.x);
  const int t = static_cast<int>(threadIdx.x);
  const int n = static_cast<int>(terms);
  auto* coef = reinterpret_cast<thrust::complex<double>*>(small_simplify_shared);
  auto* kx = reinterpret_cast<std::uint64_t*>(coef + capacity);
  auto* kz = kx + capacity * Words;
  auto* index = reinterpret_cast<std::uint32_t*>(kz + capacity * Words);
  auto* scan = index + capacity;

  // 1. Keys, input indices and the largest magnitudes. Sentinels (~0 words,
  // index ~0) order after every real entry: real indices are below terms.
  const bool valid = t < n;
  for (int word = 0; word < Words; ++word) {
    kx[t * Words + word] = valid ? in_x[t * Words + word] : ~std::uint64_t{0};
    kz[t * Words + word] = valid ? in_z[t * Words + word] : ~std::uint64_t{0};
  }
  index[t] = valid ? static_cast<std::uint32_t>(t) : 0xffffffffu;
  double magnitude = 0.0;
  double half_magnitude = 0.0;
  if (valid) {
    const thrust::complex<double> c = in_coeffs[t];
    if (components_finite(c)) {
      magnitude = thrust::abs(c);
      half_magnitude = ::hypot(c.real() * 0.5, c.imag() * 0.5);
    } else {
      magnitude = ::nan("");
    }
  }
  coef[t] = thrust::complex<double>(magnitude, half_magnitude);
  __syncthreads();
  for (int stride = capacity / 2; stride > 0; stride >>= 1) {
    if (t < stride) {
      const thrust::complex<double> a = coef[t];
      const thrust::complex<double> b = coef[t + stride];
      coef[t] = thrust::complex<double>(
          max_propagating_nan(a.real(), b.real()), a.imag() < b.imag() ? b.imag() : a.imag());
    }
    __syncthreads();
  }
  const double max_abs = coef[0].real();
  const double max_half = coef[0].imag();
  __syncthreads();
  if (max_abs != max_abs) {  // uniform across the block
    if (t == 0) {
      result->survivors = 0;
      result->status = static_cast<unsigned long long>(kSmallSimplifyNonFiniteInput);
    }
    return;
  }
  // The CPU drop rule (src/detail/simplify_numeric.hpp): no 0 * inf at
  // rtol == 0, and half units when the ordinary threshold overflows.
  double drop_threshold = rtol == 0.0 ? atol : atol + rtol * max_abs;
  double half_threshold = 0.0;
  if (!::isfinite(drop_threshold)) {
    half_threshold = atol * 0.5 + rtol * max_half;
    drop_threshold = half_threshold * 2.0;
  }

  // 2. All-ascending bitonic sort over `capacity` positions.
  const int pairs = capacity / 2;
  for (int k = 2; k <= capacity; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      if (t < pairs) {
        const int lower = ((t & ~(j - 1)) << 1) | (t & (j - 1));
        const int upper = j == (k >> 1) ? (lower ^ (k - 1)) : (lower + j);
        if (key_less<Words>(kx, kz, index, upper, lower)) {
          swap_entries<Words>(kx, kz, index, lower, upper);
        }
      }
      __syncthreads();
    }
  }

  // 3. Coefficients in sorted order, so run sums read shared memory.
  coef[t] = valid ? in_coeffs[index[t]] : thrust::complex<double>(0.0, 0.0);
  __syncthreads();

  // 4. Run heads sum their runs in input order (index tie-breaks keep it).
  int keep = 0;
  bool overflowed = false;
  thrust::complex<double> accumulated(0.0, 0.0);
  if (valid && (t == 0 || !keys_equal<Words>(kx, kz, t, t - 1))) {
    accumulated = coef[t];
    for (int next = t + 1; next < n && keys_equal<Words>(kx, kz, next, t); ++next) {
      accumulated += coef[next];
    }
    if (!components_finite(accumulated)) {
      overflowed = true;
    } else if (::isfinite(drop_threshold)) {
      keep = thrust::abs(accumulated) > drop_threshold ? 1 : 0;
    } else {
      keep = ::hypot(accumulated.real() * 0.5, accumulated.imag() * 0.5) > half_threshold ? 1 : 0;
    }
  }
  if (__syncthreads_or(overflowed ? 1 : 0) != 0) {
    if (t == 0) {
      result->survivors = 0;
      result->status = static_cast<unsigned long long>(kSmallSimplifyOverflow);
    }
    return;
  }

  // 5. Inclusive scan of survivor flags, then compaction.
  scan[t] = static_cast<std::uint32_t>(keep);
  __syncthreads();
  for (int offset = 1; offset < capacity; offset <<= 1) {
    const std::uint32_t add = t >= offset ? scan[t - offset] : 0U;
    __syncthreads();
    scan[t] += add;
    __syncthreads();
  }
  if (keep != 0) {
    const int out = static_cast<int>(scan[t]) - 1;
    for (int word = 0; word < Words; ++word) {
      out_x[out * Words + word] = kx[t * Words + word];
      out_z[out * Words + word] = kz[t * Words + word];
    }
    out_coeffs[out] = accumulated;
  }
  if (t == capacity - 1) {
    result->survivors = scan[t];
    result->status = static_cast<unsigned long long>(kSmallSimplifyOk);
  }
}

}  // namespace wolfgang::accelerator_detail

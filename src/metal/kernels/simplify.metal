#include <metal_stdlib>
using namespace metal;

struct SimplifyWords1Params {
  uint terms;
  uint padded_terms;
  ulong drop_threshold_fixed32;
  ulong drop_threshold_square_fixed64;
  uint use_magnitude_square_threshold;
};

struct BitonicSortParams {
  uint padded_terms;
  uint j;
  uint k;
};

struct PrefixSumParams {
  uint entries;
  uint offset;
};

static inline bool fp_key_equal(ulong lhs_x, ulong lhs_z, ulong rhs_x, ulong rhs_z) {
  return lhs_x == rhs_x && lhs_z == rhs_z;
}

static inline bool fp_key_less(ulong lhs_x, ulong lhs_z, ulong rhs_x, ulong rhs_z) {
  if (lhs_x != rhs_x) {
    return lhs_x < rhs_x;
  }
  return lhs_z < rhs_z;
}

static inline bool fp_entry_less(
    uint lhs_valid,
    ulong lhs_x,
    ulong lhs_z,
    uint rhs_valid,
    ulong rhs_x,
    ulong rhs_z) {
  if (lhs_valid != rhs_valid) {
    return lhs_valid != 0u;
  }
  if (lhs_valid == 0u) {
    return false;
  }
  return fp_key_less(lhs_x, lhs_z, rhs_x, rhs_z);
}

static inline ulong fp_abs_ulong(long value) {
  return value < 0l ? static_cast<ulong>(-value) : static_cast<ulong>(value);
}

static inline uint fp_survives_drop_threshold(
    long2 accumulated,
    constant SimplifyWords1Params& params) {
  const ulong real_abs = fp_abs_ulong(accumulated.x);
  const ulong imag_abs = fp_abs_ulong(accumulated.y);
  if (params.use_magnitude_square_threshold == 0u) {
    return real_abs != 0ul || imag_abs != 0ul ? 1u : 0u;
  }
  const ulong magnitude_square = real_abs * real_abs + imag_abs * imag_abs;
  return magnitude_square > params.drop_threshold_square_fixed64 ? 1u : 0u;
}

static inline long fp_double_bits_to_fixed32(ulong bits, thread bool& ok) {
  const ulong exponent_bits = (bits >> 52u) & 0x7fful;
  const ulong fraction = bits & 0x000ffffffffffffful;
  const bool negative = (bits >> 63u) != 0ul;
  if (exponent_bits == 0ul && fraction == 0ul) {
    return 0l;
  }
  if (exponent_bits == 0ul || exponent_bits == 0x7fful) {
    ok = false;
    return 0l;
  }

  const long exponent = static_cast<long>(exponent_bits) - 1023l;
  const ulong mantissa = (1ul << 52u) | fraction;
  const long shift = exponent + 32l - 52l;
  ulong magnitude = 0ul;
  if (shift >= 0l) {
    if (shift >= 63l) {
      ok = false;
      return 0l;
    }
    magnitude = mantissa << static_cast<uint>(shift);
    if ((magnitude >> static_cast<uint>(shift)) != mantissa) {
      ok = false;
      return 0l;
    }
  } else {
    const uint right_shift = static_cast<uint>(-shift);
    if (right_shift >= 64u) {
      ok = false;
      return 0l;
    }
    const ulong discarded_mask = (1ul << right_shift) - 1ul;
    if ((mantissa & discarded_mask) != 0ul) {
      ok = false;
      return 0l;
    }
    magnitude = mantissa >> right_shift;
  }

  if (magnitude > 0x7ffffffffffffffful) {
    ok = false;
    return 0l;
  }
  const long signed_magnitude = static_cast<long>(magnitude);
  return negative ? -signed_magnitude : signed_magnitude;
}

static inline ulong fp_fixed32_to_double_bits(long value) {
  if (value == 0l) {
    return 0ul;
  }
  const bool negative = value < 0l;
  const ulong magnitude = static_cast<ulong>(negative ? -value : value);
  const uint leading_zeros = clz(magnitude);
  const uint msb = 63u - leading_zeros;
  const long exponent_unbiased = static_cast<long>(msb) - 32l;
  const ulong significand =
      msb <= 52u ? (magnitude << (52u - msb)) : (magnitude >> (msb - 52u));
  const ulong fraction = significand & 0x000ffffffffffffful;
  const ulong exponent_bits = static_cast<ulong>(exponent_unbiased + 1023l);
  return (negative ? (1ul << 63u) : 0ul) | (exponent_bits << 52u) | fraction;
}

kernel void fp_simplify_words1_init_keys(
    device const ulong* in_x [[buffer(0)]],
    device const ulong* in_z [[buffer(1)]],
    device const ulong2* in_coeff_bits [[buffer(2)]],
    device ulong* sort_x [[buffer(3)]],
    device ulong* sort_z [[buffer(4)]],
    device long2* sort_coeffs_fixed32 [[buffer(5)]],
    device uint* sort_valid [[buffer(6)]],
    device atomic_uint* invalid_coefficients [[buffer(7)]],
    constant SimplifyWords1Params& params [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= params.padded_terms) {
    return;
  }
  if (index < params.terms) {
    bool real_ok = true;
    bool imag_ok = true;
    const ulong2 coeff_bits = in_coeff_bits[index];
    sort_x[index] = in_x[index];
    sort_z[index] = in_z[index];
    sort_coeffs_fixed32[index] = long2(
        fp_double_bits_to_fixed32(coeff_bits.x, real_ok),
        fp_double_bits_to_fixed32(coeff_bits.y, imag_ok));
    sort_valid[index] = 1u;
    if (!real_ok || !imag_ok) {
      atomic_store_explicit(invalid_coefficients, 1u, memory_order_relaxed);
    }
    return;
  }
  sort_x[index] = 0xfffffffffffffffful;
  sort_z[index] = 0xfffffffffffffffful;
  sort_coeffs_fixed32[index] = long2(0l, 0l);
  sort_valid[index] = 0u;
}

kernel void fp_simplify_words1_bitonic_sort_step(
    device ulong* sort_x [[buffer(0)]],
    device ulong* sort_z [[buffer(1)]],
    device long2* sort_coeffs_fixed32 [[buffer(2)]],
    device uint* sort_valid [[buffer(3)]],
    constant BitonicSortParams& params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= params.padded_terms) {
    return;
  }
  const uint partner = index ^ params.j;
  if (partner <= index || partner >= params.padded_terms) {
    return;
  }

  const bool ascending = (index & params.k) == 0u;
  const ulong lhs_x = sort_x[index];
  const ulong lhs_z = sort_z[index];
  const ulong rhs_x = sort_x[partner];
  const ulong rhs_z = sort_z[partner];
  const uint lhs_valid = sort_valid[index];
  const uint rhs_valid = sort_valid[partner];
  const bool should_swap =
      ascending ? fp_entry_less(rhs_valid, rhs_x, rhs_z, lhs_valid, lhs_x, lhs_z)
                : fp_entry_less(lhs_valid, lhs_x, lhs_z, rhs_valid, rhs_x, rhs_z);
  if (!should_swap) {
    return;
  }

  const long2 lhs_coeff = sort_coeffs_fixed32[index];
  sort_x[index] = rhs_x;
  sort_z[index] = rhs_z;
  sort_coeffs_fixed32[index] = sort_coeffs_fixed32[partner];
  sort_valid[index] = sort_valid[partner];
  sort_x[partner] = lhs_x;
  sort_z[partner] = lhs_z;
  sort_coeffs_fixed32[partner] = lhs_coeff;
  sort_valid[partner] = lhs_valid;
}

kernel void fp_simplify_words1_mark_heads(
    device const ulong* sort_x [[buffer(0)]],
    device const ulong* sort_z [[buffer(1)]],
    device const uint* sort_valid [[buffer(2)]],
    device uint* head_flags [[buffer(3)]],
    constant SimplifyWords1Params& params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= params.padded_terms) {
    return;
  }
  if (sort_valid[index] == 0u) {
    head_flags[index] = 0u;
    return;
  }
  if (index == 0u) {
    head_flags[index] = 1u;
    return;
  }
  head_flags[index] =
      fp_key_equal(sort_x[index], sort_z[index], sort_x[index - 1u], sort_z[index - 1u])
          ? 0u
          : 1u;
}

kernel void fp_simplify_clear_uint(
    device uint* values [[buffer(0)]],
    constant PrefixSumParams& params [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  if (index < params.entries) {
    values[index] = 0u;
  }
}

kernel void fp_simplify_prefix_sum_step(
    device const uint* in_values [[buffer(0)]],
    device uint* out_values [[buffer(1)]],
    constant PrefixSumParams& params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= params.entries) {
    return;
  }
  uint value = in_values[index];
  if (index >= params.offset) {
    value += in_values[index - params.offset];
  }
  out_values[index] = value;
}

kernel void fp_simplify_words1_reduce_by_key(
    device const ulong* sort_x [[buffer(0)]],
    device const ulong* sort_z [[buffer(1)]],
    device const long2* sort_coeffs_fixed32 [[buffer(2)]],
    device const uint* sort_valid [[buffer(3)]],
    device const uint* head_flags [[buffer(4)]],
    device const uint* head_prefix [[buffer(5)]],
    device ulong* reduced_x [[buffer(6)]],
    device ulong* reduced_z [[buffer(7)]],
    device long2* reduced_coeffs_fixed32 [[buffer(8)]],
    device uint* survivor_flags [[buffer(9)]],
    constant SimplifyWords1Params& params [[buffer(10)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= params.padded_terms || head_flags[index] == 0u) {
    return;
  }

  const ulong key_x = sort_x[index];
  const ulong key_z = sort_z[index];
  long2 accumulated = sort_coeffs_fixed32[index];
  uint next = index + 1u;
  while (next < params.padded_terms && sort_valid[next] != 0u && head_flags[next] == 0u) {
    accumulated += sort_coeffs_fixed32[next];
    ++next;
  }

  const uint unique_index = head_prefix[index] - 1u;
  reduced_x[unique_index] = key_x;
  reduced_z[unique_index] = key_z;
  reduced_coeffs_fixed32[unique_index] = accumulated;
  survivor_flags[unique_index] = fp_survives_drop_threshold(accumulated, params);
}

kernel void fp_simplify_words1_compact_survivors(
    device const ulong* reduced_x [[buffer(0)]],
    device const ulong* reduced_z [[buffer(1)]],
    device const long2* reduced_coeffs_fixed32 [[buffer(2)]],
    device const uint* survivor_flags [[buffer(3)]],
    device const uint* survivor_prefix [[buffer(4)]],
    device ulong* out_x [[buffer(5)]],
    device ulong* out_z [[buffer(6)]],
    device ulong2* out_coeff_bits [[buffer(7)]],
    constant PrefixSumParams& params [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= params.entries || survivor_flags[index] == 0u) {
    return;
  }
  const uint out_index = survivor_prefix[index] - 1u;
  const long2 coeff = reduced_coeffs_fixed32[index];
  out_x[out_index] = reduced_x[index];
  out_z[out_index] = reduced_z[index];
  out_coeff_bits[out_index] =
      ulong2(fp_fixed32_to_double_bits(coeff.x), fp_fixed32_to_double_bits(coeff.y));
}

// Hybrid simplify: sort (x, z, input index) on the GPU so each duplicate run
// lists its coefficients in input order; the float64 reduction then runs on
// the CPU over shared memory, which Metal kernels cannot do in double.
//
// The sort is an all-ascending bitonic network over next_power_of_two(terms)
// positions. Stage k (block size k) starts with a flip step pairing lower with
// lower ^ (k - 1), then half-cleaner steps pair lower with lower + j. Every
// compare-exchange keeps the smaller entry at the lower position, so positions
// at or past `terms` act as +inf sentinels that never move: buffers hold only
// `terms` entries and pairs whose upper position is out of range are skipped.
// Input indices are unique, so the order is total and any correct sort gives
// the same permutation.
//
// Steps with j <= kHybridSortTile / 2 stay inside an aligned tile and run in
// threadgroup memory, so a 2^20-position sort needs 66 dispatches, not 210.
// The host compiles this file with WOLFGANG_HYBRID_SORT_TILE set from
// kHybridSortTile in simplify_metal.mm, so the two cannot drift apart.
#ifndef WOLFGANG_HYBRID_SORT_TILE
#error "WOLFGANG_HYBRID_SORT_TILE must be defined by the host compile options"
#endif
constant uint kHybridSortTile = WOLFGANG_HYBRID_SORT_TILE;
constant uint kHybridSortTileThreads = kHybridSortTile / 2u;

struct HybridSortParams {
  uint terms;
  uint j;
  uint k;
};

static inline bool fp_index_entry_less(
    ulong lhs_x,
    ulong lhs_z,
    uint lhs_index,
    ulong rhs_x,
    ulong rhs_z,
    uint rhs_index) {
  if (lhs_x != rhs_x || lhs_z != rhs_z) {
    return fp_key_less(lhs_x, lhs_z, rhs_x, rhs_z);
  }
  return lhs_index < rhs_index;
}

// Lower position of pair `pair` in a step with partner distance j (a power of
// two): pair's bits below j stay, the rest shift up past bit j.
static inline uint fp_bitonic_lower(uint pair, uint j) {
  return ((pair & ~(j - 1u)) << 1u) | (pair & (j - 1u));
}

static inline void fp_tile_compare_exchange(
    threadgroup ulong* tile_x,
    threadgroup ulong* tile_z,
    threadgroup uint* tile_index,
    uint lower,
    uint upper) {
  const ulong lower_x = tile_x[lower];
  const ulong lower_z = tile_z[lower];
  const uint lower_index = tile_index[lower];
  const ulong upper_x = tile_x[upper];
  const ulong upper_z = tile_z[upper];
  const uint upper_index = tile_index[upper];
  if (fp_index_entry_less(upper_x, upper_z, upper_index, lower_x, lower_z, lower_index)) {
    tile_x[lower] = upper_x;
    tile_z[lower] = upper_z;
    tile_index[lower] = upper_index;
    tile_x[upper] = lower_x;
    tile_z[upper] = lower_z;
    tile_index[upper] = lower_index;
  }
}

// Tile positions at or past `terms` load as (~0, ~0, ~0u). Valid input
// indices are below terms <= UINT32_MAX, so the sentinel orders after every
// real entry, including an all-Y key.
static inline void fp_store_tile(
    device ulong* sort_x,
    device ulong* sort_z,
    device uint* sort_index,
    threadgroup const ulong* tile_x,
    threadgroup const ulong* tile_z,
    threadgroup const uint* tile_index,
    uint base,
    uint terms,
    uint thread_index) {
  for (uint local = thread_index; local < kHybridSortTile; local += kHybridSortTileThreads) {
    const uint position = base + local;
    if (position < terms) {
      sort_x[position] = tile_x[local];
      sort_z[position] = tile_z[local];
      sort_index[position] = tile_index[local];
    }
  }
}

// Stages k = 2 .. kHybridSortTile: sorts each tile of the input completely and
// tags entries with their input index. One threadgroup of
// kHybridSortTileThreads threads per tile.
kernel void fp_simplify_words1_hybrid_tile_sort(
    device const ulong* in_x [[buffer(0)]],
    device const ulong* in_z [[buffer(1)]],
    device ulong* sort_x [[buffer(2)]],
    device ulong* sort_z [[buffer(3)]],
    device uint* sort_index [[buffer(4)]],
    constant HybridSortParams& params [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint tile [[threadgroup_position_in_grid]]) {
  threadgroup ulong tile_x[kHybridSortTile];
  threadgroup ulong tile_z[kHybridSortTile];
  threadgroup uint tile_index[kHybridSortTile];
  const uint base = tile * kHybridSortTile;
  for (uint local = thread_index; local < kHybridSortTile; local += kHybridSortTileThreads) {
    const uint position = base + local;
    const bool valid = position < params.terms;
    tile_x[local] = valid ? in_x[position] : 0xfffffffffffffffful;
    tile_z[local] = valid ? in_z[position] : 0xfffffffffffffffful;
    tile_index[local] = valid ? position : 0xffffffffu;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 2u; k <= kHybridSortTile; k <<= 1u) {
    for (uint j = k >> 1u; j > 0u; j >>= 1u) {
      const uint lower = fp_bitonic_lower(thread_index, j);
      const uint upper = j == (k >> 1u) ? (lower ^ (k - 1u)) : (lower + j);
      fp_tile_compare_exchange(tile_x, tile_z, tile_index, lower, upper);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  fp_store_tile(sort_x, sort_z, sort_index, tile_x, tile_z, tile_index, base, params.terms, thread_index);
}

// Half-cleaner steps j = kHybridSortTile / 2 .. 1 that finish a stage after its
// global steps.
kernel void fp_simplify_words1_hybrid_tile_merge(
    device ulong* sort_x [[buffer(0)]],
    device ulong* sort_z [[buffer(1)]],
    device uint* sort_index [[buffer(2)]],
    constant HybridSortParams& params [[buffer(3)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint tile [[threadgroup_position_in_grid]]) {
  threadgroup ulong tile_x[kHybridSortTile];
  threadgroup ulong tile_z[kHybridSortTile];
  threadgroup uint tile_index[kHybridSortTile];
  const uint base = tile * kHybridSortTile;
  for (uint local = thread_index; local < kHybridSortTile; local += kHybridSortTileThreads) {
    const uint position = base + local;
    const bool valid = position < params.terms;
    tile_x[local] = valid ? sort_x[position] : 0xfffffffffffffffful;
    tile_z[local] = valid ? sort_z[position] : 0xfffffffffffffffful;
    tile_index[local] = valid ? sort_index[position] : 0xffffffffu;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = kHybridSortTile / 2u; j > 0u; j >>= 1u) {
    const uint lower = fp_bitonic_lower(thread_index, j);
    fp_tile_compare_exchange(tile_x, tile_z, tile_index, lower, lower + j);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  fp_store_tile(sort_x, sort_z, sort_index, tile_x, tile_z, tile_index, base, params.terms, thread_index);
}

// One step with j >= kHybridSortTile, one thread per pair.
kernel void fp_simplify_words1_hybrid_global_step(
    device ulong* sort_x [[buffer(0)]],
    device ulong* sort_z [[buffer(1)]],
    device uint* sort_index [[buffer(2)]],
    constant HybridSortParams& params [[buffer(3)]],
    uint pair [[thread_position_in_grid]]) {
  const uint lower = fp_bitonic_lower(pair, params.j);
  const uint upper = params.j == (params.k >> 1u) ? (lower ^ (params.k - 1u)) : (lower + params.j);
  if (upper >= params.terms) {
    return;
  }
  const ulong lower_x = sort_x[lower];
  const ulong lower_z = sort_z[lower];
  const uint lower_index = sort_index[lower];
  const ulong upper_x = sort_x[upper];
  const ulong upper_z = sort_z[upper];
  const uint upper_index = sort_index[upper];
  if (fp_index_entry_less(upper_x, upper_z, upper_index, lower_x, lower_z, lower_index)) {
    sort_x[lower] = upper_x;
    sort_z[lower] = upper_z;
    sort_index[lower] = upper_index;
    sort_x[upper] = lower_x;
    sort_z[upper] = lower_z;
    sort_index[upper] = lower_index;
  }
}

// Copies each sorted entry's coefficient bits into sorted order so the CPU
// reduction reads sequentially. A raw 16-byte copy: no double arithmetic.
kernel void fp_simplify_words1_hybrid_gather_coeffs(
    device const ulong2* coeffs [[buffer(0)]],
    device const uint* sort_index [[buffer(1)]],
    device ulong2* sorted_coeffs [[buffer(2)]],
    constant HybridSortParams& params [[buffer(3)]],
    uint position [[thread_position_in_grid]]) {
  if (position < params.terms) {
    sorted_coeffs[position] = coeffs[sort_index[position]];
  }
}

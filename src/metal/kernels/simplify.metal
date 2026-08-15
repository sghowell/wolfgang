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

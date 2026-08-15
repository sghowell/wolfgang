#include <metal_stdlib>
using namespace metal;

struct CommutationKernelParams {
  ulong lhs_terms;
  ulong rhs_terms;
  ulong words;
};

struct CountKernelParams {
  ulong entries;
  ulong rows;
  ulong cols;
};

kernel void fp_pairwise_commutation_flat_generic(
    device const ulong* lhs_x [[buffer(0)]],
    device const ulong* lhs_z [[buffer(1)]],
    device const ulong* rhs_x [[buffer(2)]],
    device const ulong* rhs_z [[buffer(3)]],
    device uchar* out [[buffer(4)]],
    constant CommutationKernelParams& params [[buffer(5)]],
    uint entry [[thread_position_in_grid]]) {
  const ulong total = params.lhs_terms * params.rhs_terms;
  if (static_cast<ulong>(entry) >= total) {
    return;
  }

  const ulong lhs_term = static_cast<ulong>(entry) / params.rhs_terms;
  const ulong rhs_term = static_cast<ulong>(entry) - lhs_term * params.rhs_terms;
  const ulong lhs_offset = lhs_term * params.words;
  const ulong rhs_offset = rhs_term * params.words;

  uint parity = 0;
  for (ulong word = 0; word < params.words; ++word) {
    const ulong anti =
        (lhs_x[lhs_offset + word] & rhs_z[rhs_offset + word]) ^
        (lhs_z[lhs_offset + word] & rhs_x[rhs_offset + word]);
    parity ^= static_cast<uint>(popcount(anti)) & 1u;
  }
  out[entry] = parity == 0u ? 1u : 0u;
}

kernel void fp_pairwise_commutation_words1(
    device const ulong* lhs_x [[buffer(0)]],
    device const ulong* lhs_z [[buffer(1)]],
    device const ulong* rhs_x [[buffer(2)]],
    device const ulong* rhs_z [[buffer(3)]],
    device uchar* out [[buffer(4)]],
    constant CommutationKernelParams& params [[buffer(5)]],
    uint2 pair [[thread_position_in_grid]]) {
  const ulong rhs_term = static_cast<ulong>(pair.x);
  const ulong lhs_term = static_cast<ulong>(pair.y);
  if (lhs_term >= params.lhs_terms || rhs_term >= params.rhs_terms) {
    return;
  }

  const ulong anti =
      (lhs_x[lhs_term] & rhs_z[rhs_term]) ^
      (lhs_z[lhs_term] & rhs_x[rhs_term]);
  const uint parity = static_cast<uint>(popcount(anti)) & 1u;
  out[lhs_term * params.rhs_terms + rhs_term] = parity == 0u ? 1u : 0u;
}

kernel void fp_pairwise_commutation_words2(
    device const ulong* lhs_x [[buffer(0)]],
    device const ulong* lhs_z [[buffer(1)]],
    device const ulong* rhs_x [[buffer(2)]],
    device const ulong* rhs_z [[buffer(3)]],
    device uchar* out [[buffer(4)]],
    constant CommutationKernelParams& params [[buffer(5)]],
    uint2 pair [[thread_position_in_grid]]) {
  const ulong rhs_term = static_cast<ulong>(pair.x);
  const ulong lhs_term = static_cast<ulong>(pair.y);
  if (lhs_term >= params.lhs_terms || rhs_term >= params.rhs_terms) {
    return;
  }

  const ulong lhs_offset = lhs_term * 2ul;
  const ulong rhs_offset = rhs_term * 2ul;
  const ulong anti0 =
      (lhs_x[lhs_offset] & rhs_z[rhs_offset]) ^
      (lhs_z[lhs_offset] & rhs_x[rhs_offset]);
  const ulong anti1 =
      (lhs_x[lhs_offset + 1ul] & rhs_z[rhs_offset + 1ul]) ^
      (lhs_z[lhs_offset + 1ul] & rhs_x[rhs_offset + 1ul]);
  const uint parity =
      (static_cast<uint>(popcount(anti0)) ^ static_cast<uint>(popcount(anti1))) & 1u;
  out[lhs_term * params.rhs_terms + rhs_term] = parity == 0u ? 1u : 0u;
}

kernel void fp_pairwise_commutation_generic(
    device const ulong* lhs_x [[buffer(0)]],
    device const ulong* lhs_z [[buffer(1)]],
    device const ulong* rhs_x [[buffer(2)]],
    device const ulong* rhs_z [[buffer(3)]],
    device uchar* out [[buffer(4)]],
    constant CommutationKernelParams& params [[buffer(5)]],
    uint2 pair [[thread_position_in_grid]]) {
  const ulong rhs_term = static_cast<ulong>(pair.x);
  const ulong lhs_term = static_cast<ulong>(pair.y);
  if (lhs_term >= params.lhs_terms || rhs_term >= params.rhs_terms) {
    return;
  }

  const ulong lhs_offset = lhs_term * params.words;
  const ulong rhs_offset = rhs_term * params.words;

  uint parity = 0;
  for (ulong word = 0; word < params.words; ++word) {
    const ulong anti =
        (lhs_x[lhs_offset + word] & rhs_z[rhs_offset + word]) ^
        (lhs_z[lhs_offset + word] & rhs_x[rhs_offset + word]);
    parity ^= static_cast<uint>(popcount(anti)) & 1u;
  }
  out[lhs_term * params.rhs_terms + rhs_term] = parity == 0u ? 1u : 0u;
}

kernel void fp_count_commuting_total_atomic(
    device const uchar* data [[buffer(0)]],
    device atomic_uint* total [[buffer(1)]],
    constant CountKernelParams& params [[buffer(2)]],
    uint count_index [[thread_position_in_grid]]) {
  if (static_cast<ulong>(count_index) >= params.entries) {
    return;
  }
  if (data[count_index] != 0u) {
    atomic_fetch_add_explicit(total, 1u, memory_order_relaxed);
  }
}

kernel void fp_count_commuting_total_block_sums(
    device const uchar* data [[buffer(0)]],
    device uint* partials [[buffer(1)]],
    constant CountKernelParams& params [[buffer(2)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint block_index [[threadgroup_position_in_grid]]) {
  threadgroup uint scratch[256];
  const ulong entry = static_cast<ulong>(block_index) * 256ul +
                      static_cast<ulong>(thread_index);
  scratch[thread_index] = entry < params.entries && data[entry] != 0u ? 1u : 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = 128u; stride > 0u; stride >>= 1u) {
    if (thread_index < stride) {
      scratch[thread_index] += scratch[thread_index + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (thread_index == 0u) {
    partials[block_index] = scratch[0];
  }
}

kernel void fp_count_commuting_rows(
    device const uchar* data [[buffer(0)]],
    device ulong* counts [[buffer(1)]],
    constant CountKernelParams& params [[buffer(2)]],
    uint row [[thread_position_in_grid]]) {
  if (static_cast<ulong>(row) >= params.rows) {
    return;
  }

  const ulong offset = static_cast<ulong>(row) * params.cols;
  ulong count = 0;
  for (ulong col = 0; col < params.cols; ++col) {
    count += data[offset + col] != 0u ? 1ul : 0ul;
  }
  counts[row] = count;
}

kernel void fp_count_commuting_cols(
    device const uchar* data [[buffer(0)]],
    device ulong* counts [[buffer(1)]],
    constant CountKernelParams& params [[buffer(2)]],
    uint col [[thread_position_in_grid]]) {
  if (static_cast<ulong>(col) >= params.cols) {
    return;
  }

  ulong count = 0;
  for (ulong row = 0; row < params.rows; ++row) {
    count += data[row * params.cols + static_cast<ulong>(col)] != 0u ? 1ul : 0ul;
  }
  counts[col] = count;
}

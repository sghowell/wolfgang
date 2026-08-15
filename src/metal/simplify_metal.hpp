#pragma once

#include "fastpauli/device_pauli_sum.hpp"

#include <cstddef>

namespace wolfgang::metal_detail {

struct MetalSimplifyTimingDecomposition {
  double host_preflight_seconds = 0.0;
  double scratch_and_output_allocation_seconds = 0.0;
  double command_encoding_seconds = 0.0;
  double command_execution_seconds = 0.0;
  double output_accounting_seconds = 0.0;
  double total_observed_seconds = 0.0;
};

struct MetalSimplifyDispatchCounts {
  std::size_t clear_invalid = 0;
  std::size_t init_keys = 0;
  std::size_t bitonic_sort = 0;
  std::size_t mark_heads = 0;
  std::size_t prefix_sum = 0;
  std::size_t clear_survivors = 0;
  std::size_t reduce_by_key = 0;
  std::size_t compact_survivors = 0;
  std::size_t total_kernel_dispatches = 0;
};

struct MetalSimplifyPipelineCache {
  const char* boundary = "prewarmed_static_pipeline_cache";
  const char* library_source = "runtime_source";
  const char* scope = "process_static_private_benchmark_hook";
};

struct MetalSimplifyPerformanceDecision {
  const char* candidate_status = "experimental";
  const char* reason =
      "Campaign 8 timing decomposition keeps this candidate experimental until "
      "a lower-pass sort or reusable output boundary beats same-host CPU "
      "simplify and the transfer-reference bridge.";
};

struct MetalSimplifyPrimitiveStack {
  const char* sort = "bitonic_sort_words1";
  const char* prefix_sum = "hillis_steele_inclusive_scan_uint32";
  const char* reduce_by_key = "head_parallel_duplicate_sum_words1";
  const char* compaction = "prefix_compacted_survivors_words1";
};

struct MetalSimplifyCandidateResult {
  DevicePauliSum output;
  MetalSimplifyPrimitiveStack primitive_stack;
  std::size_t input_terms = 0;
  std::size_t output_terms = 0;
  std::size_t padded_terms = 0;
  std::size_t bitonic_passes = 0;
  std::size_t prefix_sum_passes = 0;
  std::size_t workspace_reserved_bytes = 0;
  MetalSimplifyTimingDecomposition timing;
  MetalSimplifyDispatchCounts dispatch_counts;
  MetalSimplifyPipelineCache pipeline_cache;
  MetalSimplifyPerformanceDecision performance_decision;
};

[[nodiscard]] MetalSimplifyCandidateResult simplify_words1_device_candidate_for_testing(
    const DevicePauliSum& input,
    double atol,
    double rtol);

}  // namespace wolfgang::metal_detail

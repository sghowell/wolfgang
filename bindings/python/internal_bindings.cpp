#include "bindings.hpp"

#include "wolfgang/cpu_backend.hpp"
#include "wolfgang/device_commutation_matrix.hpp"
#include "wolfgang/device_pauli_sum.hpp"
#include "wolfgang/pauli_sum.hpp"
#if WOLFGANG_BUILD_CUDA_ENABLED
#include "cuda/device_commutation_matrix.cuh"
#include "cuda/workspace.cuh"
#endif
#if WOLFGANG_BUILD_METAL_ENABLED
#include "metal/simplify_metal.hpp"
#endif

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

#if WOLFGANG_BUILD_CUDA_ENABLED
namespace {

nb::dict workspace_snapshot_to_dict(
    const wolfgang::cuda::detail::WorkspaceSnapshot& snapshot,
    const char* label) {
  nb::dict item;
  item["label"] = label;
  item["device_ordinal"] = snapshot.device_ordinal;
  item["reserved_bytes"] = snapshot.reserved_bytes;
  item["high_watermark_bytes"] = snapshot.high_watermark_bytes;
  item["allocation_count"] = snapshot.allocation_count;
  item["growth_count"] = snapshot.growth_count;
  return item;
}

std::vector<std::size_t> parse_workspace_reserve_bytes(nb::iterable values) {
  std::vector<std::size_t> parsed;
  for (nb::handle value : values) {
    if (!PyLong_Check(value.ptr())) {
      throw nb::value_error("reserve_bytes entries must be non-negative integers");
    }
    const unsigned long long raw_value = PyLong_AsUnsignedLongLong(value.ptr());
    if (PyErr_Occurred()) {
      PyErr_Clear();
      throw nb::value_error("reserve_bytes entries must be non-negative integers");
    }
    parsed.push_back(static_cast<std::size_t>(raw_value));
  }
  return parsed;
}

}  // namespace
#endif

#if WOLFGANG_BUILD_METAL_ENABLED
namespace {

nb::dict metal_simplify_primitive_stack_to_dict(
    const wolfgang::metal_detail::MetalSimplifyPrimitiveStack& stack) {
  nb::dict item;
  item["sort"] = stack.sort;
  item["prefix_sum"] = stack.prefix_sum;
  item["reduce_by_key"] = stack.reduce_by_key;
  item["compaction"] = stack.compaction;
  return item;
}

nb::dict metal_simplify_timing_to_dict(
    const wolfgang::metal_detail::MetalSimplifyTimingDecomposition& timing) {
  nb::dict item;
  item["host_preflight"] = timing.host_preflight_seconds;
  item["scratch_and_output_allocation"] =
      timing.scratch_and_output_allocation_seconds;
  item["command_encoding"] = timing.command_encoding_seconds;
  item["command_execution"] = timing.command_execution_seconds;
  item["output_accounting"] = timing.output_accounting_seconds;
  item["total_observed"] = timing.total_observed_seconds;
  return item;
}

nb::dict metal_simplify_dispatch_counts_to_dict(
    const wolfgang::metal_detail::MetalSimplifyDispatchCounts& counts) {
  nb::dict item;
  item["clear_invalid"] = counts.clear_invalid;
  item["init_keys"] = counts.init_keys;
  item["bitonic_sort"] = counts.bitonic_sort;
  item["mark_heads"] = counts.mark_heads;
  item["prefix_sum"] = counts.prefix_sum;
  item["clear_survivors"] = counts.clear_survivors;
  item["reduce_by_key"] = counts.reduce_by_key;
  item["compact_survivors"] = counts.compact_survivors;
  item["total_kernel_dispatches"] = counts.total_kernel_dispatches;
  return item;
}

nb::dict metal_simplify_pipeline_cache_to_dict(
    const wolfgang::metal_detail::MetalSimplifyPipelineCache& cache) {
  nb::dict item;
  item["boundary"] = cache.boundary;
  item["library_source"] = cache.library_source;
  item["scope"] = cache.scope;
  return item;
}

nb::dict metal_simplify_performance_decision_to_dict(
    const wolfgang::metal_detail::MetalSimplifyPerformanceDecision& decision) {
  nb::dict item;
  item["candidate_status"] = decision.candidate_status;
  item["reason"] = decision.reason;
  return item;
}

}  // namespace
#endif

namespace wolfgang::python {

void register_pauli_diagnostics(nb::module_& module);

void register_internal_bindings(nb::module_& module) {
#if WOLFGANG_ENABLE_INTERNAL_BINDINGS
#if WOLFGANG_BUILD_CUDA_ENABLED
  module.def(
      "_cuda_workspace_probe_for_testing",
      [](int device, nb::iterable reserve_bytes, bool reset, bool release) {
        nb::dict report;
        const wolfgang::CudaStatus status = wolfgang::DevicePauliSum::cuda_status();
        report["cuda_enabled"] = true;
        report["runtime_available"] = status.runtime_available;
        report["device_ordinal"] = device;
        report["workspace_mode"] = wolfgang::cuda::detail::workspace_timing_mode_name(
            wolfgang::cuda::detail::workspace_timing_mode_from_env());
        if (!status.runtime_available) {
          report["status"] = "skipped";
          report["skip_reason"] = status.skip_reason;
          report["snapshots"] = nb::list();
          report["allocation_count"] = 0;
          report["growth_count"] = 0;
          report["high_watermark_bytes"] = 0;
          return report;
        }
        if (device < 0 || device >= status.device_count) {
          throw nb::value_error("CUDA workspace probe device ordinal is out of range");
        }

        wolfgang::cuda::detail::CudaWorkspace workspace(device);
        const std::vector<std::size_t> byte_requests =
            parse_workspace_reserve_bytes(reserve_bytes);
        nb::list snapshots;
        snapshots.append(workspace_snapshot_to_dict(workspace.snapshot(), "before_reserve"));
        for (std::size_t index = 0; index < byte_requests.size(); ++index) {
          workspace.reserve_bytes(byte_requests[index], 256);
          const std::string label = "after_reserve_" + std::to_string(index);
          snapshots.append(workspace_snapshot_to_dict(workspace.snapshot(), label.c_str()));
        }
        if (reset) {
          workspace.reset();
          snapshots.append(workspace_snapshot_to_dict(workspace.snapshot(), "after_reset"));
        }
        if (release) {
          workspace.release();
          snapshots.append(workspace_snapshot_to_dict(workspace.snapshot(), "after_release"));
        }

        const wolfgang::cuda::detail::WorkspaceSnapshot final_snapshot = workspace.snapshot();
        report["status"] = "ok";
        report["snapshots"] = snapshots;
        report["allocation_count"] = final_snapshot.allocation_count;
        report["growth_count"] = final_snapshot.growth_count;
        report["high_watermark_bytes"] = final_snapshot.high_watermark_bytes;
        return report;
      },
      nb::arg("device") = 0,
      nb::arg("reserve_bytes") = nb::make_tuple(4096, 8192),
      nb::arg("reset") = true,
      nb::arg("release") = true,
      "Private CUDA workspace lifetime probe for tests and benchmark validation.");
#endif

#if WOLFGANG_BUILD_METAL_ENABLED
  module.def(
      "_metal_simplify_words1_candidate_for_testing",
      [](const wolfgang::DevicePauliSum& input,
         double atol,
         double rtol,
         bool include_output) {
        nb::dict report;
        report["object_backend"] = "metal";
        report["operation"] = "simplify";
        report["variant"] = "metal_simplify_device_candidate";
        report["metal_simplify_strategy"] = "device_candidate";
        if (input.words() != 1) {
          report["status"] = "unavailable";
          report["skip_reason"] =
              "Metal words1 simplify candidate requires exactly one packed word";
          report["transfer_boundary"] = "not_applicable";
          report["metal_simplify_strategy_status"] = "unavailable";
          return report;
        }

        wolfgang::metal_detail::MetalSimplifyCandidateResult result;
        try {
          result = wolfgang::metal_detail::simplify_words1_device_candidate_for_testing(
              input,
              atol,
              rtol);
        } catch (const std::invalid_argument& error) {
          report["status"] = "rejected_with_evidence";
          report["skip_reason"] = error.what();
          report["transfer_boundary"] = "status_only";
          report["metal_simplify_strategy_status"] = "rejected_with_evidence";
          return report;
        }
        report["status"] = "ok";
        report["transfer_boundary"] = "device_resident";
        report["metal_simplify_strategy_status"] = "benchmark_only";
        report["input_terms"] = result.input_terms;
        report["output_terms"] = result.output_terms;
        report["padded_terms"] = result.padded_terms;
        report["bitonic_passes"] = result.bitonic_passes;
        report["prefix_sum_passes"] = result.prefix_sum_passes;
        report["workspace_reserved_bytes"] = result.workspace_reserved_bytes;
        report["campaign8_timing_schema"] = "checked_device_resident_simplify_v1";
        report["timing_decomposition_seconds"] =
            metal_simplify_timing_to_dict(result.timing);
        report["dispatch_counts"] =
            metal_simplify_dispatch_counts_to_dict(result.dispatch_counts);
        report["pipeline_cache"] =
            metal_simplify_pipeline_cache_to_dict(result.pipeline_cache);
        report["performance_decision"] =
            metal_simplify_performance_decision_to_dict(result.performance_decision);
        report["primitive_stack"] =
            metal_simplify_primitive_stack_to_dict(result.primitive_stack);
        if (include_output) {
          report["device_output"] = nb::cast(std::move(result.output));
        }
        return report;
      },
      nb::arg("op"),
      nb::arg("atol") = 1.0e-12,
      nb::arg("rtol") = 0.0,
      nb::arg("include_output") = false,
      "Private benchmark-only Apple Metal words=1 simplify primitive stack.");
#endif

#endif

  module.def(
      "_build_info",
      []() {
        const wolfgang::CpuBackendReport cpu_backend =
            wolfgang::cpu_backend_report_from_environment();
        nb::dict info;
        info["cpu_backend"] = cpu_backend.active_backend;
        info["active_cpu_backend"] = cpu_backend.active_backend;
        info["requested_cpu_backend"] = cpu_backend.requested_backend;
        info["cpu_backend_env_var"] = "WOLFGANG_CPU_BACKEND";
        info["cuda_enabled"] = static_cast<bool>(WOLFGANG_BUILD_CUDA_ENABLED);
        info["cuda_architectures"] = WOLFGANG_BUILD_CUDA_ARCHITECTURES;
        info["cuda_toolkit_version"] = WOLFGANG_CUDA_TOOLKIT_VERSION;
        info["hip_enabled"] = static_cast<bool>(WOLFGANG_BUILD_HIP_ENABLED);
        info["hip_architectures"] = WOLFGANG_BUILD_HIP_ARCHITECTURES;
        info["rocm_toolkit_version"] = WOLFGANG_ROCM_TOOLKIT_VERSION;
        info["metal_enabled"] = static_cast<bool>(WOLFGANG_BUILD_METAL_ENABLED);
        info["native_enabled"] = static_cast<bool>(WOLFGANG_BUILD_NATIVE_ENABLED);

        nb::dict cpu_cmake_options;
        cpu_cmake_options["WOLFGANG_ENABLE_CUDA"] = WOLFGANG_REQUESTED_ENABLE_CUDA;
        cpu_cmake_options["WOLFGANG_CUDA_ARCHITECTURES"] =
            WOLFGANG_REQUESTED_CUDA_ARCHITECTURES;
        cpu_cmake_options["WOLFGANG_CUDA_USE_CUB"] = WOLFGANG_REQUESTED_CUDA_USE_CUB;
        cpu_cmake_options["WOLFGANG_CUDA_USE_THRUST"] =
            WOLFGANG_REQUESTED_CUDA_USE_THRUST;
        cpu_cmake_options["WOLFGANG_ENABLE_HIP"] = WOLFGANG_REQUESTED_ENABLE_HIP;
        cpu_cmake_options["WOLFGANG_HIP_ARCHITECTURES"] =
            WOLFGANG_REQUESTED_HIP_ARCHITECTURES;
        cpu_cmake_options["WOLFGANG_ENABLE_METAL"] = WOLFGANG_REQUESTED_ENABLE_METAL;
        cpu_cmake_options["WOLFGANG_ENABLE_NATIVE"] = WOLFGANG_REQUESTED_ENABLE_NATIVE;
        cpu_cmake_options["WOLFGANG_ENABLE_OPENMP"] = WOLFGANG_REQUESTED_ENABLE_OPENMP;
        cpu_cmake_options["WOLFGANG_ENABLE_TBB"] = WOLFGANG_REQUESTED_ENABLE_TBB;
        cpu_cmake_options["WOLFGANG_ENABLE_AVX2"] = WOLFGANG_REQUESTED_ENABLE_AVX2;
        cpu_cmake_options["WOLFGANG_ENABLE_AVX512"] = WOLFGANG_REQUESTED_ENABLE_AVX512;
        cpu_cmake_options["WOLFGANG_ENABLE_ARM_NEON"] =
            WOLFGANG_REQUESTED_ENABLE_ARM_NEON;
        cpu_cmake_options["WOLFGANG_ENABLE_ARM_SVE"] =
            WOLFGANG_REQUESTED_ENABLE_ARM_SVE;
        cpu_cmake_options["WOLFGANG_BUILD_CPU_BACKEND"] =
            std::string(WOLFGANG_BUILD_CPU_BACKEND);
        info["cpu_cmake_options"] = cpu_cmake_options;

        nb::dict cpu_backend_build_flags;
        cpu_backend_build_flags["scalar"] = true;
        cpu_backend_build_flags["tbb"] = static_cast<bool>(WOLFGANG_BUILD_TBB_ENABLED);
        cpu_backend_build_flags["avx2"] = static_cast<bool>(WOLFGANG_BUILD_AVX2_ENABLED);
        cpu_backend_build_flags["avx512"] =
            static_cast<bool>(WOLFGANG_BUILD_AVX512_ENABLED);
        cpu_backend_build_flags["neon"] =
            static_cast<bool>(WOLFGANG_BUILD_ARM_NEON_ENABLED);
        cpu_backend_build_flags["sve"] =
            static_cast<bool>(WOLFGANG_BUILD_ARM_SVE_ENABLED);
        info["cpu_backend_build_flags"] = cpu_backend_build_flags;
        info["oneTBB_version"] = WOLFGANG_BUILD_TBB_VERSION;

        nb::dict cpu_auto_dispatch_thresholds;
        cpu_auto_dispatch_thresholds["tbb_pairwise_entries"] =
            wolfgang::kAutoTbbPairwiseEntryThreshold;
        info["cpu_auto_dispatch_thresholds"] = cpu_auto_dispatch_thresholds;

        nb::dict optimized_cpu_kernels;
        nb::list tbb_kernels;
        if (WOLFGANG_BUILD_TBB_ENABLED) {
          tbb_kernels.append("commutes_with");
          tbb_kernels.append("full_group_commutation_graph");
        }
        optimized_cpu_kernels["tbb"] = tbb_kernels;

        nb::list avx2_kernels;
        if (WOLFGANG_BUILD_AVX2_ENABLED) {
          avx2_kernels.append("commutes_with_words_1_2");
          avx2_kernels.append("full_group_commutation_graph_words_1_2");
        }
        optimized_cpu_kernels["avx2"] = avx2_kernels;

        nb::list avx512_kernels;
        if (WOLFGANG_BUILD_AVX512_ENABLED) {
          avx512_kernels.append("commutes_with_words_1_2");
          avx512_kernels.append("full_group_commutation_graph_words_1_2");
        }
        optimized_cpu_kernels["avx512"] = avx512_kernels;

        nb::list neon_kernels;
        if (WOLFGANG_BUILD_ARM_NEON_ENABLED) {
          neon_kernels.append("commutes_with_words_1_2");
          neon_kernels.append("full_group_commutation_graph_words_1_2");
        }
        optimized_cpu_kernels["neon"] = neon_kernels;
        optimized_cpu_kernels["sve"] = nb::list();
        info["optimized_cpu_kernels"] = optimized_cpu_kernels;

        nb::list cuda_kernels;
        if (WOLFGANG_BUILD_CUDA_ENABLED) {
          cuda_kernels.append("simplify");
          cuda_kernels.append("expectation_statevector");
          cuda_kernels.append("commutes_with");
          cuda_kernels.append("commutes_with_device");
          cuda_kernels.append("matmul");
        }
        info["cuda_kernels"] = cuda_kernels;

        nb::list hip_kernels;
        if (WOLFGANG_BUILD_HIP_ENABLED) {
          hip_kernels.append("simplify");
          hip_kernels.append("expectation_statevector");
          hip_kernels.append("commutes_with");
          hip_kernels.append("commutes_with_device");
          hip_kernels.append("commutation_count_consumers");
          hip_kernels.append("matmul");
        }
        info["hip_kernels"] = hip_kernels;

        nb::list metal_kernels;
        if (WOLFGANG_BUILD_METAL_ENABLED) {
          metal_kernels.append("commutes_with");
          metal_kernels.append("commutes_with_device");
          metal_kernels.append("commutation_count_consumers");
          metal_kernels.append("benchmark_simplify_primitives");
        }
        info["metal_kernels"] = metal_kernels;

        nb::dict compiler_build_config;
        compiler_build_config["CMAKE_CXX_COMPILER_ID"] =
            WOLFGANG_CMAKE_CXX_COMPILER_ID;
        compiler_build_config["CMAKE_CXX_COMPILER_VERSION"] =
            WOLFGANG_CMAKE_CXX_COMPILER_VERSION;
        compiler_build_config["CMAKE_BUILD_TYPE"] = WOLFGANG_CMAKE_BUILD_TYPE;
        compiler_build_config["CMAKE_CXX_FLAGS"] = WOLFGANG_CMAKE_CXX_FLAGS;
        compiler_build_config["CMAKE_CUDA_COMPILER_ID"] =
            WOLFGANG_CMAKE_CUDA_COMPILER_ID;
        compiler_build_config["CMAKE_CUDA_COMPILER_VERSION"] =
            WOLFGANG_CMAKE_CUDA_COMPILER_VERSION;
        compiler_build_config["CMAKE_CUDA_HOST_COMPILER"] =
            WOLFGANG_CMAKE_CUDA_HOST_COMPILER;
        compiler_build_config["WOLFGANG_CUDA_HOST_COMPILER"] =
            WOLFGANG_CUDA_HOST_COMPILER_METADATA;
        compiler_build_config["WOLFGANG_CUDA_HOST_COMPILER_SOURCE"] =
            WOLFGANG_CUDA_HOST_COMPILER_METADATA_SOURCE;
        compiler_build_config["CMAKE_HIP_COMPILER"] = WOLFGANG_CMAKE_HIP_COMPILER;
        compiler_build_config["CMAKE_HIP_COMPILER_ID"] =
            WOLFGANG_CMAKE_HIP_COMPILER_ID;
        compiler_build_config["CMAKE_HIP_COMPILER_VERSION"] =
            WOLFGANG_CMAKE_HIP_COMPILER_VERSION;
        compiler_build_config["CMAKE_OBJCXX_COMPILER_ID"] =
            WOLFGANG_CMAKE_OBJCXX_COMPILER_ID;
        compiler_build_config["CMAKE_OBJCXX_COMPILER_VERSION"] =
            WOLFGANG_CMAKE_OBJCXX_COMPILER_VERSION;
        compiler_build_config["CMAKE_OBJCXX_FLAGS"] = WOLFGANG_CMAKE_OBJCXX_FLAGS;
        info["compiler_build_config"] = compiler_build_config;

        const wolfgang::CudaStatus cuda_status = wolfgang::DevicePauliSum::cuda_status();
        info["cuda_runtime_available"] = cuda_status.runtime_available;
        info["cuda_runtime_version"] = cuda_status.runtime_version;
        info["cuda_driver_version"] = cuda_status.driver_version;
        const wolfgang::HipStatus hip_status = wolfgang::DevicePauliSum::hip_status();
        info["hip_runtime_available"] = hip_status.runtime_available;
        info["hip_runtime_version"] = hip_status.runtime_version;
        info["hip_driver_version"] = hip_status.driver_version;
        const wolfgang::MetalStatus metal_status = wolfgang::DevicePauliSum::metal_status();
        info["metal_runtime_available"] = metal_status.runtime_available;
        info["metal_device_name"] = metal_status.metal_device_name;
        info["metal_macos_version"] = metal_status.macos_version;
        info["metal_xcode_or_clt_version"] = metal_status.xcode_or_clt_version;
        info["metal_storage_mode"] = metal_status.storage_mode;
        info["metal_capability_summary"] = metal_status.capability_summary;

        const bool cuda_enabled = static_cast<bool>(WOLFGANG_BUILD_CUDA_ENABLED);
        const bool hip_enabled = static_cast<bool>(WOLFGANG_BUILD_HIP_ENABLED);
        const bool metal_enabled = static_cast<bool>(WOLFGANG_BUILD_METAL_ENABLED);
        const int accelerator_enabled_count =
            (cuda_enabled ? 1 : 0) + (hip_enabled ? 1 : 0) + (metal_enabled ? 1 : 0);
        if (accelerator_enabled_count > 1) {
          info["accelerator_build_mode"] = "unsupported_mixed_accelerator_request";
        } else if (cuda_enabled) {
          info["accelerator_build_mode"] = "cuda_only";
        } else if (hip_enabled) {
          info["accelerator_build_mode"] = "hip_only";
        } else if (metal_enabled) {
          info["accelerator_build_mode"] = "metal_only";
        } else {
          info["accelerator_build_mode"] = "cpu_only";
        }

        nb::list compiled_accelerator_backends;
        nb::list runtime_visible_accelerator_backends;
        nb::list compiled_all_backends;
        nb::list runtime_visible_all_backends;
        compiled_all_backends.append("cpu");
        runtime_visible_all_backends.append("cpu");
        if (cuda_status.built) {
          compiled_accelerator_backends.append("cuda");
          compiled_all_backends.append("cuda");
        }
        if (hip_status.built) {
          compiled_accelerator_backends.append("hip");
          compiled_all_backends.append("hip");
        }
        if (metal_status.built) {
          compiled_accelerator_backends.append("metal");
          compiled_all_backends.append("metal");
        }
        if (cuda_status.runtime_available) {
          runtime_visible_accelerator_backends.append("cuda");
          runtime_visible_all_backends.append("cuda");
        }
        if (hip_status.runtime_available) {
          runtime_visible_accelerator_backends.append("hip");
          runtime_visible_all_backends.append("hip");
        }
        if (metal_status.runtime_available) {
          runtime_visible_accelerator_backends.append("metal");
          runtime_visible_all_backends.append("metal");
        }
        info["compiled_accelerator_backends"] = compiled_accelerator_backends;
        info["runtime_visible_accelerator_backends"] = runtime_visible_accelerator_backends;
        info["compiled_backends"] = compiled_all_backends;
        info["runtime_visible_backends"] = runtime_visible_all_backends;

        nb::list compiled_backends;
        for (const std::string& backend : cpu_backend.compiled_backends) {
          compiled_backends.append(backend);
        }
        info["compiled_cpu_backends"] = compiled_backends;

        nb::list available_backends;
        for (const std::string& backend : cpu_backend.available_backends) {
          available_backends.append(backend);
        }
        info["available_cpu_backends"] = available_backends;

        nb::dict unavailable_backends;
        nb::list candidates;
        for (const wolfgang::CpuBackendCandidate& candidate : cpu_backend.candidates) {
          nb::dict item;
          item["name"] = candidate.name;
          item["status"] = candidate.status;
          candidates.append(item);
          if (candidate.status != "available") {
            unavailable_backends[candidate.name.c_str()] = candidate.status;
          }
        }
        info["cpu_backend_candidates"] = candidates;
        info["unavailable_cpu_backends"] = unavailable_backends;
        return info;
      },
      "Return build settings used by the validation harness.");

#if WOLFGANG_ENABLE_INTERNAL_BINDINGS
  module.def(
      "_cpu_backend_report_for_testing",
      [](std::string selector) {
        const wolfgang::CpuBackendReport report =
            wolfgang::cpu_backend_report_for_selector(selector);
        nb::dict info;
        info["requested_cpu_backend"] = report.requested_backend;
        info["active_cpu_backend"] = report.active_backend;
        nb::list candidates;
        for (const wolfgang::CpuBackendCandidate& candidate : report.candidates) {
          nb::dict item;
          item["name"] = candidate.name;
          item["status"] = candidate.status;
          candidates.append(item);
        }
        info["cpu_backend_candidates"] = candidates;
        return info;
      },
      "Private test hook for CPU backend dispatch resolution.");
#endif

  register_pauli_diagnostics(module);
}

#if WOLFGANG_ENABLE_INTERNAL_BINDINGS
namespace {
void append_uint64_vector(nb::dict& item, const char* key, const std::vector<std::uint64_t>& values) {
  nb::list list;
  for (std::uint64_t value : values) {
    list.append(nb::cast(value));
  }
  item[key] = std::move(list);
}

std::uint64_t checksum_uint64_vector(const std::vector<std::uint64_t>& values) {
  std::uint64_t checksum = 1469598103934665603ULL;
  for (std::uint64_t value : values) {
    checksum ^= value;
    checksum *= 1099511628211ULL;
  }
  return checksum;
}

bool campaign8_validation_csr_allowed(std::size_t rows, std::size_t cols) {
  constexpr std::size_t kMaxValidationEntries = 1'000'000;
  return cols == 0U || rows <= kMaxValidationEntries / cols;
}

[[noreturn]] void throw_cuda_rebuild_guidance() {
  throw std::runtime_error(
      "Wolfgang was built without CUDA support; rebuild from source with "
      "WOLFGANG_ENABLE_CUDA=ON to use PauliSum.to_device().");
}

nb::dict unavailable_fused_consumer_report(
    const std::string& mode,
    const std::string& reason) {
  nb::dict report;
  report["status"] = "unavailable";
  report["mode"] = mode;
  report["rows"] = 0;
  report["cols"] = 0;
  report["timings"] = nb::dict();
  report["output_sizes"] = nb::dict();
  report["correctness_digest"] = nb::none();
  report["unavailable_reason"] = reason;
  return report;
}

void append_campaign8_status_fields(
    nb::dict& report,
    const std::string& mode,
    bool cuda_unavailable) {
  report["campaign"] = "h100_campaign8";
  report["device_resident_graph_status"] = "not_applicable";
  report["public_grouping_api_status"] = "not_applicable";
  report["dlpack_interop_status"] = "not_applicable";
  report["non_h100_portability_status"] = cuda_unavailable ? "not_run" : "not_run";
  report["stream_graph_status"] = "not_applicable";
  report["scatter_tuning_status"] = "not_applicable";

  if (cuda_unavailable) {
    report["device_resident_graph_status"] = "unavailable";
    report["public_grouping_api_status"] =
        mode == "device_grouping_consumer" ? "deferred" : "not_applicable";
    report["dlpack_interop_status"] = mode == "dlpack_consumer" ? "unavailable" : "not_applicable";
    report["stream_graph_status"] = mode == "stream_graph_probe" ? "deferred" : "not_applicable";
    report["scatter_tuning_status"] =
        mode == "csr_scatter_ab" ? "rejected_no_consumer" : "not_applicable";
    return;
  }

  if (mode == "device_resident_graph") {
    report["device_resident_graph_status"] = "retained";
    report["scatter_tuning_status"] = "rejected_no_consumer";
  } else if (mode == "device_grouping_consumer") {
    report["device_resident_graph_status"] = "retained";
    report["public_grouping_api_status"] = "deferred";
  } else if (mode == "dlpack_consumer") {
    report["device_resident_graph_status"] = "retained";
    report["dlpack_interop_status"] = "deferred";
  } else if (mode == "stream_graph_probe") {
    report["stream_graph_status"] = "deferred";
  } else if (mode == "csr_scatter_ab") {
    report["device_resident_graph_status"] = "retained";
    report["scatter_tuning_status"] = "rejected_no_consumer";
  }
}

nb::dict unavailable_campaign8_consumer_report(
    const std::string& mode,
    const std::string& reason) {
  nb::dict report;
  report["status"] = "unavailable";
  report["mode"] = mode;
  report["rows"] = 0;
  report["cols"] = 0;
  report["boundary"] = "private_benchmark_only";
  report["timing_boundary"] = "device_resident_consumer";
  report["timings"] = nb::dict();
  report["output_sizes"] = nb::dict();
  report["correctness_digest"] = nb::dict();
  report["unavailable_reason"] = reason;
  append_campaign8_status_fields(report, mode, true);
  return report;
}

bool is_valid_campaign8_consumer_mode(const std::string& mode) {
  return mode == "device_resident_graph" || mode == "device_grouping_consumer" ||
         mode == "dlpack_consumer" || mode == "stream_graph_probe" ||
         mode == "csr_scatter_ab" || mode == "portability_check";
}

[[noreturn]] void throw_invalid_campaign8_consumer_mode() {
  throw nb::value_error(
      "mode must be device_resident_graph, device_grouping_consumer, dlpack_consumer, stream_graph_probe, csr_scatter_ab, or portability_check");
}

bool is_valid_fused_consumer_mode(const std::string& mode) {
  return mode == "csr_anticommutation_graph" || mode == "conflict_degrees" ||
         mode == "grouping_summary" || mode == "bitpacked_ab";
}

[[noreturn]] void throw_invalid_fused_consumer_mode() {
  throw nb::value_error(
      "mode must be csr_anticommutation_graph, conflict_degrees, grouping_summary, or bitpacked_ab");
}

nb::dict benchmark_cuda_fused_commutation_consumer(
    const std::string& mode,
    nb::object matrix_obj,
    bool include_outputs,
    std::size_t top_k,
    bool require_cuda) {
  if (!is_valid_fused_consumer_mode(mode)) {
    throw_invalid_fused_consumer_mode();
  }

#if WOLFGANG_BUILD_CUDA_ENABLED
  if (matrix_obj.is_none()) {
    throw nb::value_error("matrix must be a DeviceCommutationMatrix when CUDA is built");
  }
  DeviceCommutationMatrix& matrix = nb::cast<DeviceCommutationMatrix&>(matrix_obj);

  nb::dict report;
  report["status"] = "ok";
  report["mode"] = mode;
  report["rows"] = matrix.rows();
  report["cols"] = matrix.cols();
  report["timings"] = nb::dict();
  report["unavailable_reason"] = nb::none();

  nb::dict output_sizes;
  nb::dict digest;

  if (mode == "csr_anticommutation_graph") {
    auto result = wolfgang::cuda::benchmark::fused_anticommutation_csr(
        matrix,
        include_outputs);
    output_sizes["row_offsets_uint64"] = static_cast<unsigned long long>(matrix.rows() + 1U);
    output_sizes["col_indices_uint64"] = static_cast<unsigned long long>(result.edge_count);
    output_sizes["host_bytes"] = static_cast<unsigned long long>(
        (matrix.rows() + 1U + static_cast<std::size_t>(result.edge_count)) *
        sizeof(std::uint64_t));
    digest["edge_count"] = static_cast<unsigned long long>(result.edge_count);
    digest["row_offset_checksum"] = static_cast<unsigned long long>(result.row_offset_checksum);
    digest["col_index_checksum"] = static_cast<unsigned long long>(result.col_index_checksum);
    if (include_outputs) {
      append_uint64_vector(report, "row_offsets", result.row_offsets);
      append_uint64_vector(report, "col_indices", result.col_indices);
    }
  } else if (mode == "conflict_degrees") {
    auto result = wolfgang::cuda::benchmark::fused_conflict_degrees(
        matrix,
        include_outputs);
    output_sizes["row_conflicts_uint64"] = static_cast<unsigned long long>(matrix.rows());
    output_sizes["col_conflicts_uint64"] = static_cast<unsigned long long>(matrix.cols());
    output_sizes["host_bytes"] = static_cast<unsigned long long>(
        (matrix.rows() + matrix.cols()) * sizeof(std::uint64_t));
    digest["row_conflict_sum"] = static_cast<unsigned long long>(result.row_conflict_sum);
    digest["col_conflict_sum"] = static_cast<unsigned long long>(result.col_conflict_sum);
    if (include_outputs) {
      append_uint64_vector(report, "row_conflicts", result.row_conflicts);
      append_uint64_vector(report, "col_conflicts", result.col_conflicts);
    }
  } else if (mode == "grouping_summary") {
    auto result = wolfgang::cuda::benchmark::fused_grouping_summary(
        matrix,
        top_k,
        include_outputs);
    output_sizes["top_rows"] = static_cast<unsigned long long>(result.top_row_indices.size());
    output_sizes["top_cols"] = static_cast<unsigned long long>(result.top_col_indices.size());
    output_sizes["host_bytes"] = static_cast<unsigned long long>(
        (result.top_row_indices.size() + result.top_row_conflicts.size() +
         result.top_col_indices.size() + result.top_col_conflicts.size()) *
        sizeof(std::uint64_t));
    digest["row_conflict_sum"] = static_cast<unsigned long long>(result.row_conflict_sum);
    digest["col_conflict_sum"] = static_cast<unsigned long long>(result.col_conflict_sum);
    append_uint64_vector(report, "top_row_indices", result.top_row_indices);
    append_uint64_vector(report, "top_row_conflicts", result.top_row_conflicts);
    append_uint64_vector(report, "top_col_indices", result.top_col_indices);
    append_uint64_vector(report, "top_col_conflicts", result.top_col_conflicts);
    if (include_outputs) {
      append_uint64_vector(report, "row_conflicts", result.row_conflicts);
      append_uint64_vector(report, "col_conflicts", result.col_conflicts);
    }
  } else if (mode == "bitpacked_ab") {
    report["status"] = "unavailable";
    report["unavailable_reason"] =
        "bit-packed fused consumer is deferred until dense fused evidence proves a capacity or bandwidth limit";
    output_sizes["host_bytes"] = 0;
    digest["decision_status"] = "deferred_no_dense_capacity_or_bandwidth_trigger";
  } else {
    throw_invalid_fused_consumer_mode();
  }

  report["output_sizes"] = std::move(output_sizes);
  report["correctness_digest"] = std::move(digest);
  return report;
#else
  (void)matrix_obj;
  (void)include_outputs;
  (void)top_k;
  if (require_cuda) {
    throw_cuda_rebuild_guidance();
  }
  return unavailable_fused_consumer_report(
      mode,
      "Wolfgang was built without CUDA support; rebuild from source with WOLFGANG_ENABLE_CUDA=ON to use PauliSum.to_device().");
#endif
}

nb::dict benchmark_cuda_device_resident_consumer(
    const std::string& mode,
    nb::object matrix_obj,
    bool include_outputs,
    std::size_t top_k,
    bool require_cuda) {
  if (!is_valid_campaign8_consumer_mode(mode)) {
    throw_invalid_campaign8_consumer_mode();
  }

#if WOLFGANG_BUILD_CUDA_ENABLED
  if ((mode == "device_resident_graph" || mode == "device_grouping_consumer" ||
       mode == "dlpack_consumer" || mode == "portability_check") &&
      matrix_obj.is_none()) {
    throw nb::value_error("matrix must be a DeviceCommutationMatrix for Campaign 8 retained modes");
  }

  nb::dict report;
  report["status"] = "ok";
  report["mode"] = mode;
  report["timings"] = nb::dict();
  report["unavailable_reason"] = "";
  append_campaign8_status_fields(report, mode, false);

  if (mode == "dlpack_consumer") {
    DeviceCommutationMatrix& matrix = nb::cast<DeviceCommutationMatrix&>(matrix_obj);
    const auto pointer = matrix.data_pointer_for_cuda_array_interface();
    report["status"] = "ok";
    report["boundary"] = "framework_consumer";
    report["timing_boundary"] = "device_resident_consumer";
    report["dlpack_interop_status"] = "implemented";
    report["rows"] = matrix.rows();
    report["cols"] = matrix.cols();
    nb::dict output_sizes;
    output_sizes["dense_uint8_device_bytes"] =
        static_cast<unsigned long long>(matrix.num_entries());
    output_sizes["host_metadata_bytes"] =
        static_cast<unsigned long long>(2 * sizeof(std::int64_t));
    report["output_sizes"] = std::move(output_sizes);
    nb::dict digest;
    digest["device_pointer_nonzero"] = pointer != 0 || matrix.num_entries() == 0;
    digest["commuting_count"] =
        static_cast<unsigned long long>(matrix.count_commuting());
    report["correctness_digest"] = std::move(digest);
    report["unavailable_reason"] = "";
    return report;
  }
  if (mode == "stream_graph_probe") {
    report["status"] = "unavailable";
    report["boundary"] = "private_benchmark_only";
    report["timing_boundary"] = "kernel_only";
    report["rows"] = 0;
    report["cols"] = 0;
    report["output_sizes"] = nb::dict();
    report["correctness_digest"] = nb::dict();
    report["unavailable_reason"] =
        "CUDA Graph and stream-aware execution are deferred until capture safety, error propagation, synchronization, lifetime, and Python ownership contracts are accepted.";
    return report;
  }
  if (mode == "csr_scatter_ab") {
    report["status"] = "unavailable";
    report["boundary"] = "private_benchmark_only";
    report["timing_boundary"] = "kernel_only";
    report["rows"] = 0;
    report["cols"] = 0;
    report["output_sizes"] = nb::dict();
    report["correctness_digest"] = nb::dict();
    report["unavailable_reason"] =
        "CSR scatter tuning is rejected for Campaign 8 because the retained graph and grouping consumers avoid full CSR edge-list materialization.";
    return report;
  }

  DeviceCommutationMatrix& matrix = nb::cast<DeviceCommutationMatrix&>(matrix_obj);
  report["rows"] = matrix.rows();
  report["cols"] = matrix.cols();
  nb::dict output_sizes;
  nb::dict digest;

  if (mode == "device_resident_graph" || mode == "portability_check") {
    auto result = wolfgang::cuda::benchmark::fused_conflict_degrees(matrix, include_outputs);
    output_sizes["compact_metadata_uint64"] =
        static_cast<unsigned long long>(matrix.rows() + matrix.cols());
    output_sizes["compact_host_bytes"] = static_cast<unsigned long long>(
        (matrix.rows() + matrix.cols()) * sizeof(std::uint64_t));
    output_sizes["full_csr_host_bytes"] = 0;
    digest["edge_count"] = static_cast<unsigned long long>(result.row_conflict_sum);
    digest["row_conflict_sum"] = static_cast<unsigned long long>(result.row_conflict_sum);
    digest["col_conflict_sum"] = static_cast<unsigned long long>(result.col_conflict_sum);
    if (include_outputs) {
      digest["row_conflict_checksum"] =
          static_cast<unsigned long long>(checksum_uint64_vector(result.row_conflicts));
      digest["col_conflict_checksum"] =
          static_cast<unsigned long long>(checksum_uint64_vector(result.col_conflicts));
      append_uint64_vector(report, "row_conflicts", result.row_conflicts);
      append_uint64_vector(report, "col_conflicts", result.col_conflicts);
      if (campaign8_validation_csr_allowed(matrix.rows(), matrix.cols())) {
        auto csr = wolfgang::cuda::benchmark::fused_anticommutation_csr(matrix, true);
        output_sizes["full_csr_host_bytes"] = static_cast<unsigned long long>(
            (matrix.rows() + 1U + static_cast<std::size_t>(csr.edge_count)) *
            sizeof(std::uint64_t));
        digest["validation_csr_edge_count"] = static_cast<unsigned long long>(csr.edge_count);
        digest["validation_csr_row_offset_checksum"] =
            static_cast<unsigned long long>(csr.row_offset_checksum);
        digest["validation_csr_col_index_checksum"] =
            static_cast<unsigned long long>(csr.col_index_checksum);
        append_uint64_vector(report, "validation_row_offsets", csr.row_offsets);
        append_uint64_vector(report, "validation_col_indices", csr.col_indices);
        report["validation_csr_status"] = "available";
      } else {
        report["validation_csr_status"] = "not_run_large_output_guard";
      }
    } else {
      report["validation_csr_status"] = "not_requested";
    }
    report["boundary"] = "compact_host_copy";
    report["timing_boundary"] = "device_resident_consumer";
  } else if (mode == "device_grouping_consumer") {
    auto result = wolfgang::cuda::benchmark::fused_grouping_summary(
        matrix,
        top_k,
        include_outputs);
    output_sizes["top_rows_uint64"] =
        static_cast<unsigned long long>(result.top_row_indices.size() * 2U);
    output_sizes["top_cols_uint64"] =
        static_cast<unsigned long long>(result.top_col_indices.size() * 2U);
    output_sizes["compact_host_bytes"] = static_cast<unsigned long long>(
        (result.top_row_indices.size() + result.top_row_conflicts.size() +
         result.top_col_indices.size() + result.top_col_conflicts.size()) *
        sizeof(std::uint64_t));
    output_sizes["full_csr_host_bytes"] = 0;
    digest["row_conflict_sum"] = static_cast<unsigned long long>(result.row_conflict_sum);
    digest["col_conflict_sum"] = static_cast<unsigned long long>(result.col_conflict_sum);
    digest["top_row_checksum"] =
        static_cast<unsigned long long>(checksum_uint64_vector(result.top_row_indices));
    digest["top_col_checksum"] =
        static_cast<unsigned long long>(checksum_uint64_vector(result.top_col_indices));
    append_uint64_vector(report, "top_row_indices", result.top_row_indices);
    append_uint64_vector(report, "top_row_conflicts", result.top_row_conflicts);
    append_uint64_vector(report, "top_col_indices", result.top_col_indices);
    append_uint64_vector(report, "top_col_conflicts", result.top_col_conflicts);
    if (include_outputs) {
      append_uint64_vector(report, "row_conflicts", result.row_conflicts);
      append_uint64_vector(report, "col_conflicts", result.col_conflicts);
    }
    report["boundary"] = "private_benchmark_only";
    report["timing_boundary"] = "compact_materialization";
  } else {
    throw_invalid_campaign8_consumer_mode();
  }

  report["output_sizes"] = std::move(output_sizes);
  report["correctness_digest"] = std::move(digest);
  return report;
#else
  (void)matrix_obj;
  (void)include_outputs;
  (void)top_k;
  if (require_cuda) {
    throw_cuda_rebuild_guidance();
  }
  return unavailable_campaign8_consumer_report(
      mode,
      "Wolfgang was built without CUDA support; rebuild from source with WOLFGANG_ENABLE_CUDA=ON to use PauliSum.to_device().");
#endif
}

}  // namespace
#endif


namespace {

class InternalPythonBufferView {
public:
  explicit InternalPythonBufferView(nb::handle object) {
    if (PyObject_GetBuffer(object.ptr(), &view_, PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) != 0) {
      throw nb::python_error();
    }
  }

  InternalPythonBufferView(const InternalPythonBufferView&) = delete;
  InternalPythonBufferView& operator=(const InternalPythonBufferView&) = delete;
  ~InternalPythonBufferView() { PyBuffer_Release(&view_); }

  [[nodiscard]] const Py_buffer& get() const noexcept { return view_; }

private:
  Py_buffer view_{};
};

void copy_device_commutation_matrix_from_python_for_testing(
    wolfgang::DeviceCommutationMatrix& matrix,
    nb::handle values_obj) {
  InternalPythonBufferView buffer(values_obj);
  const Py_buffer& view = buffer.get();
  if (view.ndim != 2 || view.itemsize != 1 || view.format == nullptr ||
      (std::strcmp(view.format, "?") != 0 && std::strcmp(view.format, "B") != 0)) {
    throw nb::type_error(
        "DeviceCommutationMatrix testing copy expects a 2-dimensional bool or uint8 array");
  }
  if (PyBuffer_IsContiguous(&view, 'C') == 0) {
    throw nb::type_error("DeviceCommutationMatrix testing copy expects C-contiguous values");
  }
  if (view.shape[0] < 0 || view.shape[1] < 0 ||
      static_cast<std::size_t>(view.shape[0]) != matrix.rows() ||
      static_cast<std::size_t>(view.shape[1]) != matrix.cols() ||
      view.len < 0 || static_cast<std::size_t>(view.len) != matrix.num_entries()) {
    throw nb::value_error("DeviceCommutationMatrix testing copy shape does not match matrix");
  }
  wolfgang::copy_device_commutation_matrix_from_host_for_testing(
      matrix,
      std::span<const std::uint8_t>(
          static_cast<const std::uint8_t*>(view.buf),
          matrix.num_entries()));
}

std::size_t checked_internal_size(nb::handle value, const char* name) {
  if (!PyLong_Check(value.ptr())) {
    throw nb::value_error((std::string(name) + " must be a non-negative integer").c_str());
  }
  const unsigned long long raw = PyLong_AsUnsignedLongLong(value.ptr());
  if (PyErr_Occurred() ||
      raw > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
    PyErr_Clear();
    throw nb::value_error((std::string(name) + " must fit size_t").c_str());
  }
  return static_cast<std::size_t>(raw);
}

wolfgang::AcceleratorBackend parse_internal_backend_selector(nb::object backend_obj) {
  if (backend_obj.is_none()) {
    return wolfgang::AcceleratorBackend::None;
  }
  try {
    return wolfgang::accelerator_backend_from_name(nb::cast<std::string>(backend_obj));
  } catch (const nb::cast_error&) {
    throw nb::value_error("backend must be None, 'auto', 'cuda', 'hip', or 'metal'");
  } catch (const std::invalid_argument& error) {
    throw nb::value_error(error.what());
  }
}

nb::dict cuda_status_to_internal_dict(const wolfgang::CudaStatus& status) {
  nb::dict info;
  info["built"] = status.built;
  info["runtime_available"] = status.runtime_available;
  info["device_count"] = status.device_count;
  info["skip_reason"] = status.skip_reason;
  info["runtime_version"] = status.runtime_version;
  info["driver_version"] = status.driver_version;
  nb::list devices;
  for (const auto& device : status.devices) {
    nb::dict item;
    item["ordinal"] = device.ordinal;
    item["name"] = device.name;
    item["compute_capability"] = nb::make_tuple(
        device.compute_capability_major, device.compute_capability_minor);
    item["total_memory_bytes"] = device.total_memory_bytes;
    devices.append(item);
  }
  info["devices"] = devices;
  return info;
}

nb::dict hip_status_to_internal_dict(const wolfgang::HipStatus& status) {
  nb::dict info;
  info["built"] = status.built;
  info["runtime_available"] = status.runtime_available;
  info["device_count"] = status.device_count;
  info["skip_reason"] = status.skip_reason;
  info["runtime_version"] = status.runtime_version;
  info["driver_version"] = status.driver_version;
  info["toolkit_version"] = status.toolkit_version;
  nb::list devices;
  for (const auto& device : status.devices) {
    nb::dict item;
    item["ordinal"] = device.ordinal;
    item["name"] = device.name;
    item["gfx_target"] = device.gfx_target;
    item["total_memory_bytes"] = device.total_memory_bytes;
    devices.append(item);
  }
  info["devices"] = devices;
  return info;
}

nb::dict metal_status_to_internal_dict(const wolfgang::MetalStatus& status) {
  nb::dict info;
  info["built"] = status.built;
  info["runtime_available"] = status.runtime_available;
  info["device_count"] = status.device_count;
  info["skip_reason"] = status.skip_reason;
  info["macos_version"] = status.macos_version;
  info["xcode_or_clt_version"] = status.xcode_or_clt_version;
  info["metal_device_name"] = status.metal_device_name;
  info["storage_mode"] = status.storage_mode;
  info["capability_summary"] = status.capability_summary;
  nb::list devices;
  for (const auto& device : status.devices) {
    nb::dict item;
    item["ordinal"] = device.ordinal;
    item["name"] = device.name;
    item["registry_id"] = device.registry_id;
    item["recommended_max_working_set_size"] = device.recommended_max_working_set_size;
    item["capability_summary"] = device.capability_summary;
    item["low_power"] = device.low_power;
    item["headless"] = device.headless;
    item["removable"] = device.removable;
    item["unified_memory"] = device.unified_memory;
    devices.append(item);
  }
  info["devices"] = devices;
  return info;
}

nb::dict accelerator_status_to_internal_dict() {
  const auto cuda = wolfgang::DevicePauliSum::cuda_status();
  const auto hip = wolfgang::DevicePauliSum::hip_status();
  const auto metal = wolfgang::DevicePauliSum::metal_status();
  nb::dict info;
  info["cuda"] = cuda_status_to_internal_dict(cuda);
  info["hip"] = hip_status_to_internal_dict(hip);
  info["metal"] = metal_status_to_internal_dict(metal);
  nb::list compiled;
  nb::list available;
  compiled.append("cpu");
  available.append("cpu");
  nb::list compiled_accelerators;
  nb::list available_accelerators;
  for (const auto& entry : {std::pair{"cuda", std::pair{cuda.built, cuda.runtime_available}},
                            std::pair{"hip", std::pair{hip.built, hip.runtime_available}},
                            std::pair{"metal", std::pair{metal.built, metal.runtime_available}}}) {
    if (entry.second.first) {
      compiled.append(entry.first);
      compiled_accelerators.append(entry.first);
    }
    if (entry.second.second) {
      available.append(entry.first);
      available_accelerators.append(entry.first);
    }
  }
  info["compiled_backends"] = compiled;
  info["available_backends"] = available;
  info["compiled_accelerator_backends"] = compiled_accelerators;
  info["available_accelerator_backends"] = available_accelerators;
  const int visible = static_cast<int>(cuda.runtime_available) +
      static_cast<int>(hip.runtime_available) + static_cast<int>(metal.runtime_available);
  info["active_backend"] = visible == 1
      ? (cuda.runtime_available ? "cuda" : (hip.runtime_available ? "hip" : "metal"))
      : "none";
  return info;
}

}  // namespace

void register_pauli_diagnostics(nb::module_& module) {
  module.def("_cuda_status", []() {
    return cuda_status_to_internal_dict(wolfgang::DevicePauliSum::cuda_status());
  });
  module.def("_hip_status", []() {
    return hip_status_to_internal_dict(wolfgang::DevicePauliSum::hip_status());
  });
  module.def("_metal_status", []() {
    return metal_status_to_internal_dict(wolfgang::DevicePauliSum::metal_status());
  });
  module.def("_accelerator_status", []() { return accelerator_status_to_internal_dict(); });
#if WOLFGANG_ENABLE_INTERNAL_BINDINGS
  nb::object pauli_type = module.attr("PauliSum");
  nb::object checked_matmul = nb::cpp_function(
      [](nb::handle lhs_terms, nb::handle rhs_terms, nb::handle max_terms) {
        try {
          return wolfgang::PauliSum::checked_matmul_intermediate_terms_for_testing(
              checked_internal_size(lhs_terms, "lhs_terms"),
              checked_internal_size(rhs_terms, "rhs_terms"),
              checked_internal_size(max_terms, "max_intermediate_terms"));
        } catch (const std::invalid_argument& error) {
          throw nb::value_error(error.what());
        }
      },
      nb::arg("lhs_terms"),
      nb::arg("rhs_terms"),
      nb::arg("max_intermediate_terms"));
  nb::object staticmethod_type = nb::module_::import_("builtins").attr("staticmethod");
  pauli_type.attr("_checked_matmul_size_for_testing") =
      staticmethod_type(checked_matmul);

  nb::object checked_commutation = nb::cpp_function(
      [](nb::handle lhs_terms, nb::handle rhs_terms, nb::handle max_entries) {
        try {
          return wolfgang::PauliSum::checked_commutation_matrix_entries_for_testing(
              checked_internal_size(lhs_terms, "lhs_terms"),
              checked_internal_size(rhs_terms, "rhs_terms"),
              checked_internal_size(max_entries, "max_commutation_matrix_entries"));
        } catch (const std::invalid_argument& error) {
          throw nb::value_error(error.what());
        }
      },
      nb::arg("lhs_terms"),
      nb::arg("rhs_terms"),
      nb::arg("max_commutation_matrix_entries"));
  pauli_type.attr("_checked_commutation_size_for_testing") =
      staticmethod_type(checked_commutation);

  pauli_type.attr("_packed_words_for_testing") = nb::cpp_function(
      [](const wolfgang::PauliSum& op) {
        nb::list x_words;
        nb::list z_words;
        for (std::uint64_t word : op.x_words()) {
          x_words.append(nb::cast(word));
        }
        for (std::uint64_t word : op.z_words()) {
          z_words.append(nb::cast(word));
        }
        return nb::make_tuple(x_words, z_words);
      },
      nb::is_method());

  module.def(
      "_accelerator_backend_selection_for_testing",
      [](nb::object requested_backend,
         bool cuda_built,
         bool cuda_runtime_available,
         bool hip_built,
         bool hip_runtime_available,
         bool metal_built,
         bool metal_runtime_available) {
        const AcceleratorBackend requested =
            parse_internal_backend_selector(std::move(requested_backend));
        const AcceleratorBackend selected = select_accelerator_backend(
            requested,
            cuda_built,
            cuda_runtime_available,
            hip_built,
            hip_runtime_available,
            metal_built,
            metal_runtime_available);
        return std::string(accelerator_backend_name(selected));
      },
      nb::arg("requested_backend").none(),
      nb::arg("cuda_built"),
      nb::arg("cuda_runtime_available"),
      nb::arg("hip_built"),
      nb::arg("hip_runtime_available"),
      nb::arg("metal_built"),
      nb::arg("metal_runtime_available"),
      "Private test hook for backend-neutral accelerator selection policy.");

  module.def(
      "_accelerator_context_validation_for_testing",
      [](std::string operation,
         std::string left_backend,
         int left_device,
         std::string right_backend,
         int right_device) {
        validate_same_accelerator_context(
            operation,
            {accelerator_backend_from_name(left_backend), left_device},
            {accelerator_backend_from_name(right_backend), right_device});
        return std::string("ok");
      },
      nb::arg("operation"),
      nb::arg("left_backend"),
      nb::arg("left_device"),
      nb::arg("right_backend"),
      nb::arg("right_device"),
      "Private test hook for backend-neutral accelerator device/context validation.");

  module.def(
      "_copy_device_commutation_matrix_from_host_for_testing",
      &copy_device_commutation_matrix_from_python_for_testing,
      nb::arg("matrix"),
      nb::arg("values"),
      "Private test hook for copying a host bool/uint8 matrix into a DeviceCommutationMatrix.");

  module.def(
      "_benchmark_cuda_fused_commutation_consumer",
      &benchmark_cuda_fused_commutation_consumer,
      nb::arg("mode"),
      nb::arg("matrix") = nb::none(),
      nb::arg("include_outputs") = false,
      nb::arg("top_k") = 8,
      nb::arg("require_cuda") = false,
      "Private benchmark-only fused DeviceCommutationMatrix consumer hook.");

  module.def(
      "_benchmark_cuda_device_resident_consumer",
      &benchmark_cuda_device_resident_consumer,
      nb::arg("mode"),
      nb::arg("matrix") = nb::none(),
      nb::arg("include_outputs") = false,
      nb::arg("top_k") = 8,
      nb::arg("require_cuda") = false,
      "Private Campaign 8 benchmark-only device-resident consumer hook.");
#endif
}

}  // namespace wolfgang::python

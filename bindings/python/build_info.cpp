#include "bindings.hpp"

#include "wolfgang/cpu_backend.hpp"
#include "wolfgang/device_pauli_sum.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

namespace wolfgang::python {

void bind_build_info(nb::module_& module) {
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
        cpu_auto_dispatch_thresholds["neon_full_grouping_scalar_min_entries"] =
            wolfgang::kAutoNeonFullGroupingScalarMinEntries;
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

        nb::list compiled_accelerator_backends;
        if (WOLFGANG_BUILD_CUDA_ENABLED) {
          compiled_accelerator_backends.append("cuda");
        }
        if (WOLFGANG_BUILD_HIP_ENABLED) {
          compiled_accelerator_backends.append("hip");
        }
        if (WOLFGANG_BUILD_METAL_ENABLED) {
          compiled_accelerator_backends.append("metal");
        }
        info["compiled_accelerator_backends"] = compiled_accelerator_backends;

        nb::list runtime_visible_accelerator_backends;
        if (cuda_status.runtime_available) {
          runtime_visible_accelerator_backends.append("cuda");
        }
        if (hip_status.runtime_available) {
          runtime_visible_accelerator_backends.append("hip");
        }
        if (metal_status.runtime_available) {
          runtime_visible_accelerator_backends.append("metal");
        }
        info["runtime_visible_accelerator_backends"] = runtime_visible_accelerator_backends;

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

        nb::list compiled_backends;
        compiled_backends.append("cpu");
        for (nb::handle backend : compiled_accelerator_backends) {
          compiled_backends.append(backend);
        }
        info["compiled_backends"] = compiled_backends;

        nb::list runtime_visible_backends;
        runtime_visible_backends.append("cpu");
        for (nb::handle backend : runtime_visible_accelerator_backends) {
          runtime_visible_backends.append(backend);
        }
        info["runtime_visible_backends"] = runtime_visible_backends;

        nb::list compiled_cpu_backends;
        for (const std::string& backend : cpu_backend.compiled_backends) {
          compiled_cpu_backends.append(backend);
        }
        info["compiled_cpu_backends"] = compiled_cpu_backends;

        nb::list available_cpu_backends;
        for (const std::string& backend : cpu_backend.available_backends) {
          available_cpu_backends.append(backend);
        }
        info["available_cpu_backends"] = available_cpu_backends;

        nb::list cpu_backend_candidates;
        nb::dict unavailable_cpu_backends;
        for (const auto& candidate_report : cpu_backend.candidates) {
          nb::dict candidate;
          candidate["name"] = candidate_report.name;
          candidate["status"] = candidate_report.status;
          if (candidate_report.status != "available") {
            unavailable_cpu_backends[candidate_report.name.c_str()] =
                candidate_report.status;
          }
          cpu_backend_candidates.append(candidate);
        }
        info["cpu_backend_candidates"] = cpu_backend_candidates;
        info["unavailable_cpu_backends"] = unavailable_cpu_backends;

        return info;
      },
      "Return build settings used by the validation harness.");
}

}  // namespace wolfgang::python

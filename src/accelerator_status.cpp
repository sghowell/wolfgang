#include "fastpauli/accelerator_status.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>

namespace wolfgang {

std::string_view accelerator_backend_name(AcceleratorBackend backend) noexcept {
  switch (backend) {
    case AcceleratorBackend::None:
      return "none";
    case AcceleratorBackend::Cuda:
      return "cuda";
    case AcceleratorBackend::Hip:
      return "hip";
    case AcceleratorBackend::Metal:
      return "metal";
  }
  return "unknown";
}

AcceleratorBackend accelerator_backend_from_name(std::string_view backend) {
  std::string normalized(backend);
  std::transform(
      normalized.begin(),
      normalized.end(),
      normalized.begin(),
      [](unsigned char value) { return static_cast<char>(std::tolower(value)); });

  if (normalized == "auto") {
    return AcceleratorBackend::None;
  }
  if (normalized == "cuda") {
    return AcceleratorBackend::Cuda;
  }
  if (normalized == "hip") {
    return AcceleratorBackend::Hip;
  }
  if (normalized == "metal") {
    return AcceleratorBackend::Metal;
  }
  throw std::invalid_argument("backend must be None, 'auto', 'cuda', 'hip', or 'metal'");
}

std::string accelerator_not_built_message(AcceleratorBackend backend) {
  if (backend == AcceleratorBackend::Cuda) {
    return "FastPauli was built without CUDA support; rebuild from source with "
           "FASTPAULI_ENABLE_CUDA=ON to use PauliSum.to_device().";
  }
  if (backend == AcceleratorBackend::Hip) {
    return "FastPauli was built without HIP support; rebuild from source with "
           "FASTPAULI_ENABLE_HIP=ON to use PauliSum.to_device().";
  }
  if (backend == AcceleratorBackend::Metal) {
    return "FastPauli was built without Metal support; rebuild from source on Apple Silicon with "
           "FASTPAULI_ENABLE_METAL=ON to use PauliSum.to_device().";
  }
  return "FastPauli was built without CUDA, HIP, or Metal accelerator support; rebuild from "
         "source with FASTPAULI_ENABLE_CUDA=ON, FASTPAULI_ENABLE_HIP=ON, or "
         "FASTPAULI_ENABLE_METAL=ON to use PauliSum.to_device().";
}

AcceleratorBackend select_accelerator_backend(
    AcceleratorBackend requested,
    bool cuda_built,
    bool cuda_runtime_available,
    bool hip_built,
    bool hip_runtime_available,
    bool metal_built,
    bool metal_runtime_available) {
  if (requested == AcceleratorBackend::Cuda) {
    if (!cuda_built) {
      throw std::runtime_error(accelerator_not_built_message(AcceleratorBackend::Cuda));
    }
    return AcceleratorBackend::Cuda;
  }
  if (requested == AcceleratorBackend::Hip) {
    if (!hip_built) {
      throw std::runtime_error(accelerator_not_built_message(AcceleratorBackend::Hip));
    }
    return AcceleratorBackend::Hip;
  }
  if (requested == AcceleratorBackend::Metal) {
    if (!metal_built) {
      throw std::runtime_error(accelerator_not_built_message(AcceleratorBackend::Metal));
    }
    return AcceleratorBackend::Metal;
  }

  const int built_count = (cuda_built ? 1 : 0) + (hip_built ? 1 : 0) + (metal_built ? 1 : 0);
  if (built_count == 0) {
    throw std::runtime_error(accelerator_not_built_message(AcceleratorBackend::None));
  }
  if (built_count == 1 && cuda_built) {
    return AcceleratorBackend::Cuda;
  }
  if (built_count == 1 && hip_built) {
    return AcceleratorBackend::Hip;
  }
  if (built_count == 1 && metal_built) {
    return AcceleratorBackend::Metal;
  }

  const int runtime_count = (cuda_runtime_available ? 1 : 0) +
                            (hip_runtime_available ? 1 : 0) +
                            (metal_runtime_available ? 1 : 0);
  if (runtime_count == 1 && cuda_runtime_available) {
    return AcceleratorBackend::Cuda;
  }
  if (runtime_count == 1 && hip_runtime_available) {
    return AcceleratorBackend::Hip;
  }
  if (runtime_count == 1 && metal_runtime_available) {
    return AcceleratorBackend::Metal;
  }

  throw std::runtime_error(
      "ambiguous accelerator backend: mixed accelerator source builds require "
      "backend=\"cuda\", backend=\"hip\", or backend=\"metal\" when the selector is "
      "omitted or 'auto'");
}

void validate_same_accelerator_context(
    std::string_view operation,
    AcceleratorObjectIdentity left,
    AcceleratorObjectIdentity right) {
  if (left.backend == right.backend && left.device_ordinal == right.device_ordinal) {
    return;
  }

  throw std::invalid_argument(
      std::string(operation) +
      " requires operands on the same accelerator backend and same device "
      "(operation=" + std::string(operation) +
      ", left_backend=" + std::string(accelerator_backend_name(left.backend)) +
      ", left_device=" + std::to_string(left.device_ordinal) +
      ", right_backend=" + std::string(accelerator_backend_name(right.backend)) +
      ", right_device=" + std::to_string(right.device_ordinal) +
      ", failure_stage=pre_allocation)");
}

}  // namespace wolfgang

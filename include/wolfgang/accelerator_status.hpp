#pragma once

#include <string>
#include <string_view>

namespace wolfgang {

enum class AcceleratorBackend {
  None,
  Cuda,
  Hip,
  Metal,
};

struct AcceleratorObjectIdentity {
  AcceleratorBackend backend = AcceleratorBackend::None;
  int device_ordinal = -1;
};

[[nodiscard]] std::string_view accelerator_backend_name(AcceleratorBackend backend) noexcept;
[[nodiscard]] AcceleratorBackend accelerator_backend_from_name(std::string_view backend);
[[nodiscard]] std::string accelerator_not_built_message(AcceleratorBackend backend);

[[nodiscard]] AcceleratorBackend select_accelerator_backend(
    AcceleratorBackend requested,
    bool cuda_built,
    bool cuda_available,
    bool hip_built,
    bool hip_available,
    bool metal_built,
    bool metal_available);

void validate_same_accelerator_context(
    std::string_view operation,
    AcceleratorObjectIdentity left,
    AcceleratorObjectIdentity right);

}  // namespace wolfgang



#pragma once

#include "device_pauli_sum.hip.hpp"

namespace wolfgang::hip_detail {

constexpr const char* kHipSimplifyDefaultStrategy = "rocthrust_default";
constexpr const char* kHipSimplifyCustomPackedKeyStrategy = "custom_packed_key";
constexpr const char* kHipSimplifyGenericParallelStrategy =
    "rocthrust_generic_parallel_reduce_by_key";
constexpr const char* kHipSimplifyGenericSerialStrategy = "serial_kernel";

}  // namespace wolfgang::hip_detail

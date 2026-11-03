#pragma once

#include <cstddef>

// Plain C++ declarations for the Wolfgang CUDA memory pool, usable from host
// translation units that cannot include Thrust (the Python bindings).
namespace wolfgang::cuda_detail {

// Bytes currently handed out by the device's pool (0 when the device does not
// use the pool), and whether the device uses the pool at all. Test hooks only.
[[nodiscard]] std::size_t pool_used_bytes_for_testing(int device);
[[nodiscard]] bool pool_enabled_for_testing(int device);

}  // namespace wolfgang::cuda_detail

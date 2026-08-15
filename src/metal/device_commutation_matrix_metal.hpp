#pragma once

#include "wolfgang/device_commutation_matrix.hpp"

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>

namespace wolfgang {

struct DeviceCommutationMatrix::Impl {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t entries = 0;
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> command_queue = nil;
  id<MTLBuffer> data = nil;
  int device_ordinal = 0;

  ~Impl();
};

}  // namespace wolfgang

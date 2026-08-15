#pragma once

#include "fastpauli/device_commutation_matrix.hpp"

#include "device_pauli_sum.hip.hpp"

#include <cstddef>
#include <cstdint>

namespace wolfgang {

struct DeviceCommutationMatrix::Impl {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t entries = 0;
  int device_ordinal = 0;
  std::uint8_t* data = nullptr;

  ~Impl();
};

}  // namespace wolfgang

#include "wolfgang/pauli_sum.hpp"

#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}

int main() {
  try {
    using wolfgang::PauliSum;
    // Independent Pauli algebra oracle: X Y = i Z, including multiword packing.
    for (const std::size_t width : {1U, 65U, 130U}) {
      const std::string identities(width - 1, 'I');
      const auto x = PauliSum::from_labels({identities + "X"}, {1.0});
      const auto y = PauliSum::from_labels({identities + "Y"}, {1.0});
      const auto xy = x.matmul(y);
      require(xy.to_labels() == std::vector<std::string>{identities + "Z"}, "XY label");
      require(xy.coeffs()[0] == std::complex<double>(0, 1), "XY phase");
      require(x.commutes_with(y) == std::vector<std::uint8_t>{0}, "X/Y anticommute");
      require(x.add(x.scalar_multiply(-1)).simplify().num_terms() == 0, "cancellation");
      const std::complex<double> large(1.7e308, 1.7e308);
      require(PauliSum::from_labels({identities + "X"}, {large}).simplify(0).num_terms() == 1,
              "large finite magnitude");
    }
    const auto z = PauliSum::from_labels({"Z"}, {1.0});
    require(std::abs(z.expectation_z_counts({"0", "1"}, {1.7e308, 0.85e308}).real() - 1.0 / 3) < 1e-14,
            "count normalization");
    bool overflow = false;
    try {
      (void)PauliSum::checked_matmul_intermediate_terms_for_testing(
          std::numeric_limits<std::size_t>::max(), 2, std::numeric_limits<std::size_t>::max());
    } catch (const std::invalid_argument&) { overflow = true; }
    require(overflow, "multiplication size overflow guard");
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

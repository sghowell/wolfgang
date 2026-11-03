#include "wolfgang/pauli_sum.hpp"

#include <cmath>
#include <cstdlib>
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

    // Packed-word construction rejects bits above num_qubits and round-trips.
    const std::vector<std::uint64_t> x_words{0b101}, z_words{0b001};
    const std::vector<std::complex<double>> one{1.0};
    require(PauliSum::from_words(3, x_words, z_words, one).to_labels() ==
                std::vector<std::string>{"XIY"},
            "from_words label packing");
    bool tail_rejected = false;
    try {
      (void)PauliSum::from_words(2, x_words, z_words, one);
    } catch (const std::invalid_argument&) { tail_rejected = true; }
    require(tail_rejected, "from_words tail-bit guard");

    // Commutators keep only anticommuting pairs, each doubled: [X, Y] = 2iZ.
    const auto xs = PauliSum::from_labels({"X", "Z"}, {1.0, 1.0});
    const auto ys = PauliSum::from_labels({"Y"}, {1.0});
    const auto commutator = xs.commutator(ys, false);
    require(commutator.to_labels() == std::vector<std::string>{"Z", "X"}, "commutator terms");
    require(commutator.coeffs()[0] == std::complex<double>(0, 2), "commutator phase");
    require(xs.anticommutator(xs).simplify().to_labels() == std::vector<std::string>{"I"},
            "anticommutator identity");
    const std::complex<double> huge(1.0e308, 0.0);
    const auto big = PauliSum::from_labels({"X"}, {huge});
    const auto tiny = PauliSum::from_labels({"Z"}, {1.0e-10});
    require(big.commutator(tiny).num_terms() == 1, "commutator scales after multiplying");
    const std::vector<std::size_t> picks{1, 1, 0};
    const auto picked = xs.select_terms(picks);
    require(picked.to_labels() == std::vector<std::string>{"Z", "Z", "X"}, "select_terms order");
    require(picked.structurally_equal(picked) && !picked.structurally_equal(xs), "structural equality");
    require(xs.negate().adjoint().coeffs()[0] == std::complex<double>(-1.0, -0.0), "negate/adjoint");

    // Statevector tools: Y|0> = i|1>, and dense/CSR matrices agree.
    const auto y_op = PauliSum::from_labels({"Y"}, {1.0});
    const std::vector<std::complex<double>> zero_state{1.0, 0.0};
    const auto applied = y_op.apply_statevector_complex128(zero_state);
    require(applied[0] == std::complex<double>(0, 0) && applied[1] == std::complex<double>(0, 1),
            "apply Y|0>");
    require(std::abs(y_op.variance_statevector_complex128(zero_state) - 1.0) < 1e-15, "variance");
    const auto dense = y_op.to_dense_matrix(4);
    require(dense[1] == std::complex<double>(0, -1) && dense[2] == std::complex<double>(0, 1),
            "dense Y matrix");
    const auto csr = PauliSum::from_labels({"XX", "YY"}, {1.0, 1.0}).to_csr_matrix(64);
    require(csr.data.size() == 2 && csr.indptr.size() == 5, "csr cancellation");
    bool sparse_guard = false;
    try {
      (void)PauliSum::empty(20).to_csr_matrix(1024);
    } catch (const std::invalid_argument&) { sparse_guard = true; }
    require(sparse_guard, "csr row-pointer guard");

    // Counts in the X basis: outcome 0 is the +1 eigenstate.
    const auto x_op = PauliSum::from_labels({"X"}, {2.0});
    const auto estimate = x_op.estimate_counts({"0", "1"}, {70.0, 30.0}, "X");
    require(std::abs(estimate.mean.real() - 0.8) < 1e-12, "estimate_counts mean");
    require(std::abs(estimate.standard_error - std::sqrt(3.36 / 99.0)) < 1e-12,
            "estimate_counts standard error");
#if WOLFGANG_BUILD_TBB_ENABLED && !defined(_WIN32)
    // oneTBB simplify equals scalar bit for bit, including duplicate runs that
    // cross the kernel's 4096-entry chunk boundaries and total cancellation.
    for (const std::size_t width : {3U, 64U, 100U, 130U}) {
      std::vector<std::string> labels;
      std::vector<std::complex<double>> coeffs;
      const std::string pattern = "XYZI";
      for (std::size_t term = 0; term < 20000; ++term) {
        std::string label(width, 'I');
        // Long runs of identical labels interleaved with distinct ones.
        const std::size_t group = term % 7 == 0 ? term : term / 3000;
        for (std::size_t qubit = 0; qubit < width; ++qubit) {
          label[qubit] = pattern[(group * 2654435761U + qubit * 40503U) % 4];
        }
        labels.push_back(label);
        coeffs.emplace_back(static_cast<double>(term % 13) - 6.0, 0.125 * static_cast<double>(term % 5));
      }
      const auto op = PauliSum::from_labels(labels, coeffs);
      setenv("WOLFGANG_CPU_BACKEND", "scalar", 1);
      const auto scalar = op.simplify();
      const auto cancelled_scalar = op.add(op.negate()).simplify();
      setenv("WOLFGANG_CPU_BACKEND", "tbb", 1);
      const auto parallel = op.simplify();
      const auto cancelled_parallel = op.add(op.negate()).simplify();
      unsetenv("WOLFGANG_CPU_BACKEND");
      require(parallel.structurally_equal(scalar), "tbb simplify equals scalar");
      require(cancelled_parallel.num_terms() == 0 && cancelled_scalar.num_terms() == 0,
              "tbb simplify cancellation");
    }
#endif
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

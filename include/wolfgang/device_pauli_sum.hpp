#pragma once

#include "wolfgang/accelerator_status.hpp"
#include "wolfgang/device_commutation_matrix.hpp"
#include "wolfgang/pauli_sum.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace wolfgang {

class DevicePauliSum;

namespace metal_detail {
struct MetalSimplifyCandidateResult;
[[nodiscard]] MetalSimplifyCandidateResult simplify_words1_device_candidate_for_testing(
    const DevicePauliSum& input,
    double atol,
    double rtol);
}  // namespace metal_detail

struct CudaDeviceInfo {
  int ordinal = -1;
  std::string name;
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
  std::size_t total_memory_bytes = 0;
};

struct CudaStatus {
  bool built = false;
  bool runtime_available = false;
  int device_count = 0;
  std::string skip_reason;
  std::string runtime_version;
  std::string driver_version;
  std::vector<CudaDeviceInfo> devices;
};

struct HipDeviceInfo {
  int ordinal = -1;
  std::string name;
  std::string gfx_target;
  std::size_t total_memory_bytes = 0;
};

struct HipStatus {
  bool built = false;
  bool runtime_available = false;
  int device_count = 0;
  std::string skip_reason;
  std::string runtime_version;
  std::string driver_version;
  std::string toolkit_version;
  std::vector<HipDeviceInfo> devices;
};

struct MetalDeviceInfo {
  int ordinal = -1;
  std::string name;
  std::uint64_t registry_id = 0;
  std::size_t recommended_max_working_set_size = 0;
  std::string capability_summary;
  bool low_power = false;
  bool headless = false;
  bool removable = false;
  bool unified_memory = false;
};

struct MetalStatus {
  bool built = false;
  bool runtime_available = false;
  int device_count = 0;
  std::string skip_reason;
  std::string macos_version;
  std::string xcode_or_clt_version;
  std::string metal_device_name;
  std::string storage_mode;
  std::string capability_summary;
  std::vector<MetalDeviceInfo> devices;
};

enum class DeviceStatevectorDtype {
  Complex64,
  Complex128,
};

class DevicePauliSum {
public:
  DevicePauliSum() noexcept;
  ~DevicePauliSum();

  DevicePauliSum(DevicePauliSum&& other) noexcept;
  DevicePauliSum& operator=(DevicePauliSum&& other) noexcept;

  DevicePauliSum(const DevicePauliSum&) = delete;
  DevicePauliSum& operator=(const DevicePauliSum&) = delete;

  [[nodiscard]] static DevicePauliSum from_host(const PauliSum& host, int device = 0);
  [[nodiscard]] static DevicePauliSum from_host(
      const PauliSum& host,
      AcceleratorBackend backend,
      int device = 0);
  [[nodiscard]] static CudaStatus cuda_status();
  [[nodiscard]] static bool cuda_available();
  [[nodiscard]] static HipStatus hip_status();
  [[nodiscard]] static bool hip_available();
  [[nodiscard]] static MetalStatus metal_status();
  [[nodiscard]] static bool metal_available();

  [[nodiscard]] PauliSum to_host() const;
  [[nodiscard]] DevicePauliSum simplify(double atol = 1.0e-12, double rtol = 0.0) const;
  [[nodiscard]] std::complex<double> expectation_statevector_complex128(
      std::span<const std::complex<double>> psi) const;
  [[nodiscard]] std::complex<double> expectation_statevector_complex64(
      std::span<const std::complex<float>> psi) const;
  [[nodiscard]] std::complex<double> expectation_statevector_device_pointer(
      std::uintptr_t device_pointer,
      DeviceStatevectorDtype dtype,
      std::size_t length) const;
  [[nodiscard]] std::vector<std::uint8_t> commutes_with(
      const DevicePauliSum& rhs,
      std::size_t max_commutation_matrix_entries = 100000000) const;
  // Fill a caller-owned host buffer with dense commutation flags in row-major
  // order over lhs terms, then rhs terms. The output span must contain exactly
  // `num_terms() * rhs.num_terms()` writable host bytes after max-entry
  // validation. Each byte is 1 when the pair commutes and 0 otherwise; callers
  // may reinterpret this as a dense bool array only when their bool
  // representation is one byte. This is a supported C++ API for high-throughput
  // integrations that already own host output storage. It has the same device,
  // qubit-count, moved-from, and max-entry exceptions as commutes_with().
  void commutes_with_into(
      const DevicePauliSum& rhs,
      std::span<std::uint8_t> output,
      std::size_t max_commutation_matrix_entries = 100000000) const;
  [[nodiscard]] DeviceCommutationMatrix commutes_with_device(
      const DevicePauliSum& rhs,
      std::size_t max_commutation_matrix_entries = 100000000) const;
  void commutes_with_device_into(
      const DevicePauliSum& rhs,
      DeviceCommutationMatrix& output,
      std::size_t max_commutation_matrix_entries = 100000000) const;
  [[nodiscard]] DevicePauliSum matmul(
      const DevicePauliSum& rhs,
      bool simplify_output = true,
      std::size_t max_intermediate_terms = 50000000) const;
  [[nodiscard]] std::size_t num_qubits() const noexcept;
  [[nodiscard]] std::size_t num_terms() const noexcept;
  [[nodiscard]] std::size_t words() const noexcept;
  [[nodiscard]] int device() const noexcept;
  [[nodiscard]] std::string backend() const;

private:
  friend metal_detail::MetalSimplifyCandidateResult
  metal_detail::simplify_words1_device_candidate_for_testing(
      const DevicePauliSum& input,
      double atol,
      double rtol);

  struct Impl;

  explicit DevicePauliSum(std::unique_ptr<Impl> impl) noexcept;
  [[nodiscard]] std::complex<double> expectation_statevector_device_pointer_impl(
      std::uintptr_t device_pointer,
      DeviceStatevectorDtype dtype,
      std::size_t length,
      bool synchronize_input) const;

  std::unique_ptr<Impl> impl_;
};

}  // namespace wolfgang



# Python API

The public package exports three native types, immutable capability records,
backend discovery helpers, and optional ecosystem adapters.

## Capability discovery

Call `wolfgang_quantum.capabilities()` to obtain a frozen `WolfgangCapabilities`
snapshot describing the compiled CPU and accelerator backends and their current
runtime visibility. `CpuCapabilities` and `BackendCapabilities` provide typed
per-backend records. Convenience helpers such as `cuda_available()` and
`cuda_devices()` are also public, with corresponding HIP and Metal helpers.

Capability discovery reports what this build and runtime can use; it does not
upgrade an experimental accelerator backend into a supported wheel target.

## `PauliSum`

Host-owned sparse Pauli operator.

Primary constructors:

- `PauliSum(num_qubits, num_terms)`
- `PauliSum.from_labels(labels, coefficients=None)`
- `PauliSum.from_sparse_list(terms, num_qubits)`
- optional `PauliSum.from_qiskit(...)`
- optional `PauliSum.from_openfermion(...)`

Core operations:

- `to_labels()` / `to_sparse_list()`
- `simplify(atol=1e-12, rtol=0.0)`
- addition and scalar multiplication
- `lhs @ rhs` Pauli multiplication
- `commutes_with(rhs)`
- `group_commuting()`
- statevector and counts expectation values
- `to_device(device=0, backend=None)` in a matching accelerator source build

## `DevicePauliSum`

Owning accelerator mirror. Availability and native/fallback behavior depend on the compiled backend and current support contract. Device operations are synchronous unless documentation for a specific API states otherwise.

## `DeviceCommutationMatrix`

Owning dense device output with compact reductions. Dense outputs are guarded because `rows * cols` may be large. Interop protocols have explicit backend and read-only limitations.

## Exceptions and invalid inputs

- `ValueError`: semantically invalid values, dimensions, indices, or selectors.
- `TypeError`: unsupported Python dtype, layout, or protocol field type.
- `RuntimeError`: requested backend unavailable, moved-from object, runtime/toolkit failure, or unsupported compiled capability.
- `OverflowError`/guard errors: requested output exceeds checked public limits.

## Optional dependencies

The base package imports without Qiskit or OpenFermion. Adapter calls provide install guidance when an optional library is absent.

## Stability

Wolfgang is pre-1.0. Public behavior is documented and changes are recorded, but deliberate migration may occur. Private names beginning with `_`, campaign hooks, benchmark schemas, and internal native helpers are not stable unless explicitly promoted. See the [API stability policy](../architecture/api_stability.md).

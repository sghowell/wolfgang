# Python API

The public package exports four native types (`PauliSum`, `DevicePauliSum`,
`DeviceCommutationMatrix` and `DeviceArrayView`), immutable capability
records, backend discovery and CPU thread helpers, the `measurement` module,
and optional ecosystem adapters.

## Capability discovery

Call `wolfgang_quantum.capabilities()` to obtain a frozen `WolfgangCapabilities`
snapshot describing the compiled CPU and accelerator backends and their current
runtime visibility. `CpuCapabilities` and `BackendCapabilities` provide typed
per-backend records. The helpers `cuda_available()` and `cuda_devices()` and
their HIP and Metal counterparts are also public.

Capability discovery reports what this build and runtime can use. It does not
turn an experimental accelerator backend into a supported wheel target.

## CPU threads

`set_num_threads(n)` caps the threads Wolfgang's parallel CPU kernels (oneTBB)
may use, for the whole process. `get_num_threads()` returns the current count:
oneTBB's default (hardware threads, respecting the process affinity mask), the
cap if it is lower, or 1 in builds without oneTBB. The `WOLFGANG_NUM_THREADS`
environment variable sets the same cap without code changes; it is read the
first time Wolfgang needs it, a later `set_num_threads()` replaces it, and a
value that is not a positive integer raises `ValueError`. Results do not
depend on the cap.

```python
import wolfgang_quantum

wolfgang_quantum.set_num_threads(4)  # e.g. one of four worker processes on 16 cores
assert wolfgang_quantum.get_num_threads() <= 4
```

Below four threads, auto dispatch uses the scalar or SIMD kernels instead of
oneTBB, so `set_num_threads(1)` gives single-threaded execution, and a process
pool with fewer than four threads per process runs those kernels. A cap at or
above oneTBB's default concurrency leaves oneTBB uncapped.

## `PauliSum`

`PauliSum` is a host-owned sparse Pauli operator.

Primary constructors:

- `PauliSum(num_qubits, num_terms)`
- `PauliSum.from_labels(labels, coefficients=None)`
- `PauliSum.from_sparse_list(terms, num_qubits)`
- `PauliSum.from_arrays(x, z, coeffs, num_qubits)` from packed `uint64` masks
- optional `PauliSum.from_qiskit(...)`
- optional `PauliSum.from_openfermion(...)`
- optional `PauliSum.from_cudaq(...)`

Core operations:

- `to_labels()` / `to_sparse_list()`
- `to_arrays()`: read-only zero-copy `(x, z, coeffs)` views of packed storage
- `simplify(atol=1e-12, rtol=0.0)`
- addition, subtraction, negation and scalar multiplication
- `lhs @ rhs` Pauli multiplication
- `commutator(rhs)` / `anticommutator(rhs)`: fused kernels that skip cancelling pairs
- `adjoint()`
- `len(op)`, `op[i]`, `op[a:b]` and iteration over single-term operators
- `op == other` (structural, unhashable) and `op.equiv(other, atol)` (mathematical)
- `commutes_with(rhs)`
- `group_commuting()`
- statevector tools: `expectation_statevector`, `variance_statevector` and
  `apply_statevector` (single states or 2-D batches), and `to_matrix(sparse=False)`
- counts estimation: `expectation_z_counts(counts)` and
  `estimate_counts(counts, basis=None)` returning `(mean, standard_error)`
- measurement planning: `qwc_basis()` and `diagonalizing_clifford()`
- `to_device(device=0, backend=None)` in a matching accelerator source build

Packed arrays avoid string conversion at the Python boundary:

```python
import numpy as np
from wolfgang_quantum import PauliSum

h = PauliSum.from_labels(["XXI", "IZZ", "YIY"], [0.5, -1.0, 0.25])
x, z, coeffs = h.to_arrays()          # uint64 (3, 1), uint64 (3, 1), complex128 (3,)
same = PauliSum.from_arrays(x, z, coeffs, h.num_qubits)
assert same == h

zz = PauliSum.from_labels(["ZZI"], [1.0])
generator = h.commutator(zz)          # [h, ZZ] without forming h @ ZZ and ZZ @ h
assert generator.equiv((h @ zz - zz @ h).simplify())
```

The precise rules are in the [semantic contracts](../architecture/semantic_contracts.md).

## Measuring grouped operators

`wolfgang_quantum.measurement` turns commuting groups into circuits and turns
counts back into estimates:

```python
from wolfgang_quantum import PauliSum
from wolfgang_quantum.measurement import allocate_shots

h = PauliSum.from_labels(["XX", "YY", "ZZ", "ZI"], [0.5, 0.5, 1.0, -0.3])
groups = h.group_commuting(mode="full")
shots = allocate_shots(groups, total_shots=10_000)

for group, budget in zip(groups, shots):
    gates, diagonal = group.diagonalizing_clifford()
    print(budget, gates, diagonal.to_labels()[0])
    # Run `gates`, measure every qubit in Z for `budget` shots, then:
    #     mean, error = diagonal.estimate_counts(counts)
```

Qubit-wise commuting groups need only single-qubit rotations; pass
`group.qwc_basis()` as `basis` to `group.estimate_counts` to skip building the
diagonal operator.

`measurement.to_qiskit_circuit(gates, num_qubits)` converts a gate list to a
Qiskit circuit when Qiskit is installed.

## `DevicePauliSum`

`DevicePauliSum` is an owning accelerator mirror of an operator. Availability and native/fallback behavior depend on the compiled backend and the current support contract. Device operations are synchronous unless the documentation for a specific API states otherwise.

On CUDA, `device_arrays()` hands the packed buffers to CuPy or PyTorch without
a copy, and `from_device_arrays()` builds an operator from device arrays
without a host round trip:

```python
import cupy
import wolfgang_quantum as wq

op = wq.PauliSum.from_labels(["XZI", "IYY"], [0.5, -1.0j]).to_device(backend="cuda")
x, z, coeffs = (cupy.from_dlpack(view) for view in op.device_arrays())  # read-only, zero copy
again = wq.DevicePauliSum.from_device_arrays(x, z, coeffs, op.num_qubits)  # device-to-device copy
```

The views keep `op` alive. They are flagged read-only, but CuPy and PyTorch do
not enforce the flag, so treat them as read-only: writing through them is
undefined behavior. PyTorch reads them with `torch.from_dlpack(view)`.
`from_device_arrays` waits for producers on any stream and returns after its
copy completes. HIP and Metal raise for both.

## `DeviceCommutationMatrix`

`DeviceCommutationMatrix` is an owning dense device output with compact reductions. Dense outputs are guarded because `rows * cols` may be large. Its interop protocols have explicit backend and read-only limitations.

## Exceptions and invalid inputs

- `ValueError`: semantically invalid values, dimensions, indices, or selectors.
- `TypeError`: unsupported Python dtype, layout, or protocol field type.
- `RuntimeError`: requested backend unavailable, moved-from object, runtime/toolkit failure, or unsupported compiled capability.
- `OverflowError`/guard errors: requested output exceeds checked public limits.

## Optional dependencies

The base package imports without Qiskit, OpenFermion or CUDA-Q. When an optional library is missing, the adapter call reports how to install it.

`PauliSum.from_cudaq(op, num_qubits=None)` and `PauliSum.to_cudaq()` convert to
and from CUDA-Q spin operators (the `cudaq` extra, CUDA-Q 0.11 or later):

```python
import cudaq
from cudaq import spin
from wolfgang_quantum import PauliSum

hamiltonian = PauliSum.from_cudaq(0.5 * spin.x(0) * spin.z(2) - 0.25 * spin.y(1))
energy = cudaq.observe(kernel, hamiltonian.simplify().to_cudaq()).expectation()  # any @cudaq.kernel
```

CUDA-Q operators carry no width, so pass `num_qubits` to keep identity
qubits above the largest index. Qubit `q` is the same qubit in both libraries.

## Stability

Wolfgang is pre-1.0. Public behavior is documented and changes are recorded, but a release may still make deliberate changes that require migration. Private names beginning with `_`, campaign hooks, benchmark schemas, and internal native helpers are not stable unless explicitly promoted. See the [API stability policy](../architecture/api_stability.md).

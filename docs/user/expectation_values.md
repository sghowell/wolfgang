# Expectation Values

Wolfgang provides scalar CPU expectation kernels for statevectors and
computational-basis counts. CUDA source builds also provide a
`DevicePauliSum.expectation_statevector()` kernel for host NumPy statevectors
and CUDA-array-interface statevectors on the same CUDA device.

```python
import numpy as np

from wolfgang_quantum import PauliSum

h = PauliSum.from_labels(["ZI", "IZ", "XX"], [1.0, -0.5, 0.25])
psi = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

assert h.expectation_statevector(psi) == 0.5 + 0.0j
```

## Statevectors

`PauliSum.expectation_statevector(psi)` computes `<psi|H|psi>` and returns a
Python `complex`.

Requirements:

```text
psi is one-dimensional
len(psi) == 2 ** h.num_qubits
psi dtype is complex64 or complex128
psi is C-contiguous
h.num_qubits <= 63 for the initial CPU implementation
```

For CUDA source builds:

```python
device_h = h.to_device()
energy = device_h.expectation_statevector(psi)
```

CUDA inputs follow the same length, dtype, and contiguity rules. Host NumPy
statevectors are copied to device memory inside the call. Device-resident arrays
are accepted through `__cuda_array_interface__` when they are one-dimensional,
contiguous, `complex64` or `complex128`, and allocated on the same CUDA device
as `device_h`.

The statevector basis order follows the dense-label convention used everywhere
in Wolfgang. In a label such as `"ZI"`, the right-most character is qubit 0,
so the vector index bits are interpreted as `|q_{n-1} ... q_0>`.

```python
z0 = PauliSum.from_labels(["IZ"], [1.0])
z1 = PauliSum.from_labels(["ZI"], [1.0])
state_01 = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.complex128)

assert z0.expectation_statevector(state_01) == -1.0 + 0.0j
assert z1.expectation_statevector(state_01) == 1.0 + 0.0j
```

## Z Counts

`PauliSum.expectation_z_counts(counts)` computes an expectation value from a
mapping of dense bitstrings to counts.

```python
z0 = PauliSum.from_labels(["IZ"], [1.0])

assert z0.expectation_z_counts({"00": 3, "01": 1}) == 0.5 + 0.0j
```

Only diagonal operators are accepted by the initial Z-count path. Any `X` or
`Y` term raises `ValueError`.

Counts must be finite, non-negative numeric values, and the total count must be
positive. Bitstrings must have length `num_qubits` and may contain only `0` and
`1`. Bitstrings use the same display convention as dense labels: the right-most
bit is qubit 0.

```python
counts = {"01": 10}

assert PauliSum.from_labels(["IZ"], [1.0]).expectation_z_counts(counts) == -1.0 + 0.0j
assert PauliSum.from_labels(["ZI"], [1.0]).expectation_z_counts(counts) == 1.0 + 0.0j
```

## Benchmark

Use the expectation benchmark smoke for local validation:

```bash
python benchmarks/bench_expectation.py --smoke --repeat 1
python benchmarks/bench_cuda_kernels.py --smoke --repeat 1
```

The benchmark reports three cases: few terms over a larger statevector, many
terms over a smaller statevector, and diagonal Z-count expectation. Reports
include Wolfgang build metadata and direct Python reference timings.
The CUDA benchmark separately reports transfer-inclusive and device-resident
timings for the CUDA statevector expectation kernel when CUDA is built and a
runtime device is visible.

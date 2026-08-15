# Wolfgang

**Packed Pauli algebra with a portable CPU oracle and evidence-gated accelerator kernels.**

Wolfgang stores sparse sums of Pauli strings as compact symplectic bit masks and complex coefficients. The same representation drives deterministic CPU semantics, runtime-dispatched SIMD and oneTBB kernels, and source-build CUDA, ROCm/HIP, and Apple Metal experiments.

```python
import numpy as np
from wolfgang_quantum import PauliSum

hamiltonian = PauliSum.from_labels(
    ["XX", "YY", "ZI", "IZ"],
    np.array([0.5, 0.5, -1.0, -1.0], dtype=np.complex128),
)

reduced = (hamiltonian @ hamiltonian).simplify()
print(reduced.to_labels())
```

## Why the project is interesting

Wolfgang is built around a deliberate engineering loop:

1. lock exact algebraic semantics in a scalar C++ implementation;
2. expose those semantics through a small Python API;
3. add property and cross-library oracles;
4. optimize one operation and one timing boundary at a time;
5. retain named hardware/toolchain evidence;
6. reject optimizations or public APIs when evidence does not justify them.

This makes the repository both a usable library and an inspectable record of low-level, agent-assisted systems work. Agent output is never treated as evidence by itself: code, tests, builds, profilers, and independent review determine acceptance.

## Explore

- [Install Wolfgang](getting-started/installation.md)
- [Run the quickstart](getting-started/quickstart.md)
- [Understand the packed representation](guide/architecture.md)
- [Read the accelerator support boundary](accelerators/overview.md)
- [Inspect the benchmark protocol](benchmarks/protocol.md)
- [Check the current release matrix](release/support_matrix.md)

!!! warning "Accelerator packaging"
    Default wheels are portable CPU builds. CUDA, ROCm/HIP, and Metal capabilities are source-build and evidence-gated; backend implementation does not imply a public accelerator wheel.

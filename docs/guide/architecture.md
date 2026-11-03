# Architecture

Wolfgang is organized around one compact representation, the packed symplectic `PauliSum`, rather than around any one accelerator API.

```mermaid
flowchart TB
    IO[Labels, sparse terms, adapters] --> P[Packed PauliSum]
    P --> S[Scalar C++ kernels]
    P --> D[CPU dispatch:<br/>AVX2, AVX-512, NEON, oneTBB]
    P --> X[DevicePauliSum:<br/>CUDA, ROCm/HIP, Metal]
    S --> O[Canonical outputs]
    D --> O
    X --> O
```

## Packed storage

For `n` qubits, each term uses `ceil(n / 64)` words for its `x` mask and the same number for `z`. Commutation then reduces to a parity calculation over bitwise intersections, and multiplication mostly to XOR plus phase accounting.

The representation provides:

- exact structure and compact memory;
- deterministic canonical keys;
- vectorized popcount kernels;
- coalesced device storage;
- one semantic model shared across backends.

## Correctness before dispatch

Every build retains a scalar path. Runtime dispatch selects only kernels whose instruction set is available and whose covered shape is validated. A forced unsupported selector fails rather than silently executing a differently labeled path.

## Device ownership

`DevicePauliSum` owns device buffers and records backend/device identity. Device outputs retain explicit ownership and same-device rules. External-pointer and DLPack paths are separate trust boundaries with protocol, lifetime, read-only, and synchronization contracts.

## Evidence-gated optimization

Optimization work is promoted only after:

1. scalar or independent-library correctness comparison;
2. retained regression tests;
3. named hardware/toolchain evidence;
4. explicit transfer/allocation/synchronization boundary;
5. repeated measurements;
6. a demonstrated win in a relevant regime.

Rejected ideas and unavailable tooling are recorded, not turned into optimistic claims. Agent-driven low-level work depends on this rule: agents generate hypotheses and implementations, and executable evidence decides which of them survive. [Agent-driven kernel engineering](agent-driven-engineering.md) describes the process.

## Source map

- `include/wolfgang/`: documented native API declarations.
- `src/`: scalar CPU semantics, dispatch, and backend implementations.
- `bindings/python/`: nanobind ownership and protocol boundary.
- `python/wolfgang_quantum/`: canonical Python package and optional adapters.
- `tests/`: behavioral, property, packaging, protocol, and hardware-gated tests.
- `benchmarks/`: deterministic benchmark producers.
- `docs/benchmarks/reports/`: reviewed, sanitized conclusions.

# Apple Metal Accelerator Bring-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a source-build-only Apple Silicon Metal accelerator lane that preserves CPU wheel reliability and the backend-neutral object model.

**Architecture:** The backend identity is `"metal"` and the initial build flag is `WOLFGANG_ENABLE_METAL=ON` with `FASTPAULI_ENABLE_METAL` accepted only as a deprecated fallback alias. The first implementation uses private Metal source files, Wolfgang-owned `MTLBuffer` storage, synchronous command-buffer completion, and target-specific Metal-only source builds rather than mixed CUDA/HIP/Metal binaries.

**Tech Stack:** C++20, nanobind, CMake, Apple Metal, metal-cpp or Objective-C++ private translation units, Metal Shading Language, pytest, benchmark smoke scripts, and Xcode Instruments or `xctrace` profiler evidence.

---

## Scope

This plan is executable only after `docs/architecture/apple_accelerator.md` is
accepted and source-of-truth registered. It must not add CUDA, ROCm/HIP,
combined accelerator, or Metal wheel support.

In scope:

```text
Metal source-build flag and CPU-only stubs
private src/metal source layout
runtime status and build metadata
backend selector extension for backend="metal"
Metal transfer round trips
Metal pairwise commutation
compact count and conflict consumers
benchmark smoke and profiler evidence
Apple Silicon validation report
```

Out of scope:

```text
Metal wheels
PyTorch mps device object implementation
MPSGraph-first sparse kernels
public command queue or async APIs
raw Metal buffer export
DLPack export
CUDA Array Interface export
multi-GPU Metal
mixed CUDA/HIP/Metal source builds
```

## File Structure

Create these files during implementation:

```text
src/metal/accelerator_metal.mm
src/metal/device_pauli_sum_metal.mm
src/metal/device_commutation_matrix_metal.mm
src/metal/commutation_metal.mm
src/metal/workspace_metal.mm
src/metal/kernels/commutation.metal
src/metal/device_pauli_sum_metal.hpp
src/metal/device_commutation_matrix_metal.hpp
src/metal/workspace_metal.hpp
tests/test_apple_metal_foundation.py
benchmarks/bench_metal_kernels.py
docs/benchmarks/reports/apple_metal_bringup_<date>.md
```

Modify these files during implementation:

```text
CMakeLists.txt
bindings/python/module.cpp
bindings/python/pauli_sum_py.cpp
src/accelerator_status.cpp
src/device_pauli_sum_stub.cpp
src/device_commutation_matrix_stub.cpp
include/wolfgang/device_pauli_sum.hpp
include/wolfgang/device_commutation_matrix.hpp
benchmarks/_benchmark_metadata.py
scripts/validate.py
README.md
docs/roadmap.md
docs/architecture/semantic_contracts.md
docs/architecture/backend_neutral_accelerators.md
docs/architecture/hardware_targets_and_testing.md
docs/benchmarks/protocol.md
```

## Task 1: Build Flag, Source Layout, And CPU Safety

**Files:**

```text
Modify: CMakeLists.txt
Modify: scripts/validate.py
Create: src/metal/accelerator_metal.mm
Create: src/metal/device_pauli_sum_metal.hpp
Create: src/metal/device_commutation_matrix_metal.hpp
Test: tests/test_apple_metal_foundation.py
```

- [ ] **Step 1: Add failing layout tests**

Add tests that assert:

```text
WOLFGANG_ENABLE_METAL is declared OFF by default
WOLFGANG_ENABLE_METAL is rejected when CUDA or HIP is also ON
src/metal files exist
CPU-only public headers do not include Metal, Foundation, MPS, or MPSGraph headers
```

Run:

```bash
python -m pytest tests/test_apple_metal_foundation.py -q
```

Expected: fails before CMake and source layout changes.

- [ ] **Step 2: Add CMake target-specific Metal switch**

Add:

```text
_wolfgang_bool_option(WOLFGANG_ENABLE_METAL FASTPAULI_ENABLE_METAL "Build Apple Metal backend support" OFF)
```

Add configure-time rejection when `WOLFGANG_ENABLE_METAL=ON` is combined with
`WOLFGANG_ENABLE_CUDA=ON` or `WOLFGANG_ENABLE_HIP=ON`.

- [ ] **Step 3: Add private empty Metal source files**

Create private `src/metal/` files that compile only when
`WOLFGANG_ENABLE_METAL=ON`. Public headers must remain framework-free.

- [ ] **Step 4: Validate CPU-only safety**

Run:

```bash
python scripts/validate.py
```

Expected: CPU-only validation passes without Metal-specific imports or link
requirements.

## Task 2: Status And Build Metadata

**Files:**

```text
Modify: src/accelerator_status.cpp
Modify: bindings/python/module.cpp
Modify: benchmarks/_benchmark_metadata.py
Modify: scripts/validate.py
Test: tests/test_apple_metal_foundation.py
```

- [ ] **Step 1: Add status tests**

Add tests that verify a Metal build reports:

```text
accelerator_build_mode == "metal_only"
compiled_accelerator_backends includes "metal"
compiled_backends includes "cpu" and "metal"
_metal_status()["built"] is true
_accelerator_status() lists metal as compiled
```

CPU-only builds must report Metal as not compiled.

- [ ] **Step 2: Implement `_metal_status()`**

Expose a private status binding that reports:

```text
built
runtime_available
device_count
devices
macos_version
xcode_or_clt_version
metal_device_name
skip_reason
```

- [ ] **Step 3: Extend benchmark metadata**

Add `metal_only` as an accelerator build mode and include Apple Metal device
metadata when available.

- [ ] **Step 4: Validate on Apple Silicon**

Run:

```bash
WOLFGANG_VALIDATE_METAL=1 WOLFGANG_ENABLE_METAL=ON python scripts/validate.py
```

Expected: status checks pass on a named Apple Silicon machine.

## Task 3: Transfers And Object Identity

**Files:**

```text
Modify: include/fastpauli/device_pauli_sum.hpp
Modify: include/fastpauli/device_commutation_matrix.hpp
Modify: src/metal/device_pauli_sum_metal.mm
Modify: src/metal/device_commutation_matrix_metal.mm
Modify: bindings/python/pauli_sum_py.cpp
Test: tests/test_apple_metal_foundation.py
```

- [ ] **Step 1: Add transfer tests**

Cover:

```text
PauliSum.to_device(backend="metal")
DevicePauliSum.backend == "metal"
DevicePauliSum.to_host() round trip
empty operators
multi-word operators
duplicate-heavy operators
unsupported backend errors when Metal is not compiled
```

- [ ] **Step 2: Implement `MTLBuffer` ownership**

Use Wolfgang-owned Metal buffers with shared storage for initial bring-up.
Record device ordinal and backend identity on every object.

- [ ] **Step 3: Keep command semantics synchronous**

Public methods must wait for command-buffer completion before returning until a
separate async API plan exists.

## Task 4: Pairwise Commutation And Compact Consumers

**Files:**

```text
Create: src/metal/kernels/commutation.metal
Modify: src/metal/commutation_metal.mm
Modify: src/metal/device_commutation_matrix_metal.mm
Test: tests/test_apple_metal_foundation.py
Benchmark: benchmarks/bench_metal_kernels.py
```

- [ ] **Step 1: Add CPU/Metal equivalence tests**

Cover one-word, two-word, generic multi-word, empty, duplicate-heavy, and
guarded allocation cases for pairwise commutation.

- [ ] **Step 2: Implement the first Metal compute kernel**

Implement pairwise commutation parity over packed `x` and `z` buffers. Keep the
output layout identical to CUDA/HIP `DeviceCommutationMatrix`.

- [ ] **Step 3: Implement compact consumers**

Add:

```text
DeviceCommutationMatrix.count_commuting(axis=None|0|1)
DeviceCommutationMatrix.conflict_degrees(axis=None|0|1)
```

for backend `"metal"`.

- [ ] **Step 4: Benchmark**

Run:

```bash
python benchmarks/bench_metal_kernels.py --smoke --repeat 1 --json
```

Expected: benchmark emits CPU scalar/default/NEON rows plus Metal
transfer-inclusive, device-resident, and compact-consumer rows.

## Task 5: Report, Review, And Closeout

**Files:**

```text
Create: docs/benchmarks/reports/apple_metal_bringup_<date>.md
Modify: README.md
Modify: docs/roadmap.md
Modify: docs/benchmarks/protocol.md
```

- [ ] **Step 1: Capture profiler evidence**

Use Xcode Instruments, Metal System Trace, `xctrace`, or a precise blocker
record. Include command-buffer, transfer, kernel, and host materialization
boundaries.

- [ ] **Step 2: Publish a checked report**

The report must include:

```text
git revision
Apple SoC and GPU core count when available
macOS version
Xcode or Command Line Tools version
Metal device name
storage mode
threadgroup size
validation commands
benchmark rows
profiler evidence or blocker
support limitations
```

- [ ] **Step 3: Review and close out**

Complete independent review, resolve blocking findings, rerun validation,
merge to `main`, push, confirm CI, and delete the feature branch.

## Acceptance

The Apple Metal bring-up is complete only when:

```text
CPU-only default builds remain clean without Metal
WOLFGANG_ENABLE_METAL=ON source build passes on Apple Silicon
WOLFGANG_ENABLE_METAL=ON is target-specific and mutually exclusive with CUDA/HIP
backend="metal" works for transfers and pairwise commutation
DevicePauliSum.backend and DeviceCommutationMatrix.backend return "metal"
Metal compact consumers match CPU results
benchmarks separate transfer-inclusive, device-resident, and host-materialized boundaries
the checked report records profiler evidence or exact tooling blockers
README wording does not claim Metal wheels or generic Apple GPU support
independent review is complete
```

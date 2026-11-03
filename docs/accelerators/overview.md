# Accelerator overview

Wolfgang uses one host representation and target-specific source builds. Accelerator implementation, runtime evidence, and distributable wheels are separate claims.

| Surface | Default CPU wheel | CUDA source build | ROCm/HIP source build | Metal source build |
|---|---:|---:|---:|---:|
| Import and host `PauliSum` | Yes | Yes | Yes | Yes |
| Owning `DevicePauliSum` | No | Hardware/toolkit gated | Hardware/toolkit gated | Hardware/runtime gated |
| Device array interop: zero-copy export (DLPack, CUDA Array Interface), import by device copy (CUDA Array Interface) | No | Yes (`device_arrays`, `from_device_arrays`) | No | No |
| oneTBB CPU kernels | Bundled in wheels built after `v0.2.3` | When CMake finds oneTBB | When CMake finds oneTBB | When CMake finds oneTBB |
| Accelerator wheel | No | Not currently public | Not currently public | Not currently public |
| Combined accelerator binary | N/A | No | No | No |

This table is a summary. The exact release boundary is maintained in [`docs/release/support_matrix.md`](../release/support_matrix.md).

## CUDA

The CUDA backend covers owning transfers, commutation, compact consumers, simplify (one fused kernel for small operators), multiplication, statevector expectation, and interoperability: zero-copy device arrays and commutation matrices through DLPack and the CUDA Array Interface. Each public operation defines whether timing includes allocation, transfer, host materialization, and synchronization.

## ROCm/HIP

ROCm/HIP follows the same semantic contracts but remains target-specific. Evidence from one MI300X configuration does not establish broad AMD portability. One extension cannot enable both CUDA and HIP; that exclusion is deliberate and stays until a mixed-runtime design is justified.

## Apple Metal

Metal is a separate Objective-C++/Metal implementation. Some operations may use an explicit correctness bridge instead of a retained native kernel. That fallback behavior is documented and must not be marketed as accelerator performance.

## Why source-build first

Accelerator wheels require decisions about runtime libraries, driver/toolkit compatibility, architecture targeting, artifact size, clean-machine installation, and hosted hardware CI. Those distribution contracts are unresolved, and Wolfgang does not cover them with a generic `pip install` claim.

## Evidence vocabulary

Accelerator claims use four evidence levels. The terms are not interchangeable.

| Term | Meaning |
|---|---|
| compile-tested | The toolchain produced the target. |
| runtime-tested | Correctness tests executed on named hardware. |
| performance-tested | Benchmarked with a declared timing boundary. |
| release-supported | Installation and runtime behavior are covered by release evidence and support policy. |

# Accelerator overview

Wolfgang uses one host representation and target-specific source builds. Accelerator implementation, runtime evidence, and distributable wheels are separate claims.

| Surface | Default CPU wheel | CUDA source build | ROCm/HIP source build | Metal source build |
|---|---:|---:|---:|---:|
| Import and host `PauliSum` | Yes | Yes | Yes | Yes |
| Owning `DevicePauliSum` | No | Hardware/toolkit gated | Hardware/toolkit gated | Hardware/runtime gated |
| Accelerator wheel | No | Not currently public | Not currently public | Not currently public |
| Combined accelerator binary | N/A | No | No | No |

The exact release boundary—not this summary—is maintained in [`docs/release/support_matrix.md`](../release/support_matrix.md).

## CUDA

CUDA work covers owning transfers, commutation, compact consumers, simplify, multiplication, statevector expectation, and carefully reviewed interoperability. Each public operation defines whether timing includes allocation, transfer, host materialization, and synchronization.

## ROCm/HIP

ROCm/HIP follows the same semantic contracts but remains target-specific. Evidence from one MI300X configuration does not establish broad AMD portability. CUDA and HIP are intentionally mutually exclusive in one extension until a mixed-runtime design is justified.

## Apple Metal

Metal is a separate Objective-C++/Metal implementation. Some operations may use an explicit correctness bridge rather than a retained native kernel. Such fallback behavior is documented and must not be marketed as accelerator performance.

## Why source-build first

Accelerator wheels require decisions about runtime libraries, driver/toolkit compatibility, architecture targeting, artifact size, clean-machine installation, and hosted hardware CI. Wolfgang does not hide those unresolved distribution contracts behind a generic `pip install` claim.

## Evidence vocabulary

- **compile-tested**: toolchain produced the target.
- **runtime-tested**: correctness tests executed on named hardware.
- **performance-tested**: benchmarked with a declared timing boundary.
- **release-supported**: installation and runtime behavior are covered by release evidence and support policy.

These terms are not interchangeable.

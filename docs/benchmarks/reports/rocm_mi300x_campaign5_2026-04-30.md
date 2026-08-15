# ROCm MI300X Campaign 5 Interop And Execution-Control Report

Date: 2026-04-30

## Scope

Campaign 5 evaluated the first public HIP Python interop boundary after
MI300X resident commutation outputs and HIP simplify were already retained. The
candidate public expansion was read-only HIP `DeviceCommutationMatrix` DLPack
export with DLPack `kDLROCM` device typing. The campaign also assigned terminal
statuses to public streams, graph replay, public workspaces, HIP expectation,
HIP matmul, ROCm portability, ROCm wheels, multi-GPU ROCm, and simultaneous
CUDA+HIP source builds.

## Host And Build

| Field | Value |
|---|---:|
| Host | AMD Instinct MI300X VF |
| GFX target | `gfx942:sramecc+:xnack-` |
| HIP runtime / driver | `7.2.26015` / `7.2.26015` |
| ROCm toolkit | `7.2.26015-fc0010cf6a` |
| HIP compiler | `/opt/rocm/bin/amdclang++`, Clang `22.0.0` |
| CPU | Intel Xeon Platinum 8568Y+ |
| Commit recorded in retained-build raw evidence | `daf0b22d377457fd1526fbdef6253a4498e59231` |
| Temporary candidate probe base commit | `efebbb7d968630dfce517d31842b246dc21caa1c` |

Evidence:

```text
docs/benchmarks/data/rocm_mi300x_campaign5_2026-04-30/
docs/benchmarks/data/rocm_mi300x_campaign5_2026-04-30/summary.json
docs/benchmarks/plots/rocm_mi300x_campaign5_interop.svg
docs/benchmarks/plots/accelerator_landscape_with_rocm.svg
```

## Implementation Outcome

HIP DLPack is not retained. A candidate implementation exported a versioned
read-only `kDLROCM` capsule and PyTorch ROCm consumed it on MI300X, but PyTorch
accepted mutation of the imported view. That violates FastPauli's read-only
export contract, so `DeviceCommutationMatrix.__dlpack__` and
`__dlpack_device__` remain unavailable for HIP-backed matrices.

The accepted-mutation result is recorded in the separate temporary-candidate
artifact
`docs/benchmarks/data/rocm_mi300x_campaign5_2026-04-30/raw/rocm_campaign5_candidate_dlpack_probe_mi300x.json`.
The retained-build benchmark rows do not claim that candidate result as a live
public-build measurement; they record `not_run_in_retained_build`.

CUDA behavior is preserved through backend-neutral DLPack accessors. HIP
continues to reject `__cuda_array_interface__`, because HIP pointers must not be
presented as CUDA memory.

## Consumer Availability

| Consumer | Version | Backend | Available | Read-only enforced | Status |
|---|---:|---|---:|---:|---|
| PyTorch | `2.13.0.dev20260428+rocm7.2` | ROCm | yes | no | rejected_with_evidence |
| CuPy | not installed | unavailable | no | no | rejected_with_evidence |

PyTorch ROCm was installed from the ROCm 7.2 nightly PyTorch wheel index and
reported HIP `7.2.53211`. Optional consumer probes now run in subprocesses so
consumer discovery does not perturb FastPauli's HIP runtime state.

## DLPack Contract Table

| Item | Result |
|---|---|
| HIP `__dlpack_device__` | unavailable |
| HIP `__dlpack__(max_version=(1, 0))` | unavailable |
| HIP `copy=True` export | unavailable |
| HIP `stream=0` export | unavailable |
| HIP legacy unversioned export | unavailable |
| HIP `__cuda_array_interface__` | rejected_with_evidence |
| Candidate PyTorch ROCm consume | correctness passed before mutation |
| Candidate PyTorch ROCm mutation guard | failed, mutation accepted |

## Timing Evidence

The retained public HIP paths in this campaign remain dense `to_host()` and
compact `count_commuting(axis=None)`. DLPack producer and framework-consumer
timings are `null` because the public DLPack path is rejected.

| Case | Entries | HIP `to_host()` median | HIP compact count median | Status |
|---|---:|---:|---:|---|
| campaign5_dlpack_consumer_mid | 262,144 | 0.019798665991 s | 0.000531313999 s | rejected_with_evidence |
| campaign5_profiler_dlpack_boundary | 1,048,576 | 0.021190694999 s | 0.000497806002 s | rejected_with_evidence |

## Execution-Control Decisions

| Surface | Final status | Reason |
|---|---|---|
| Public streams | rejected_with_evidence | No accepted Python handle, ownership, synchronization, error propagation, shape-change, or device-mismatch contract. |
| Public graphs | rejected_with_evidence | No accepted replay lifetime or shape-stability contract. |
| Public workspaces | rejected_with_evidence | No ownership-safe public API or retained-operation speedup evidence. |
| HIP expectation | out_of_scope_with_next_trigger | Needs CPU/CUDA parity fixtures promoted to HIP. |
| HIP matmul | out_of_scope_with_next_trigger | Needs CPU/CUDA parity fixtures promoted to HIP. |
| Portability beyond MI300X `gfx942` | out_of_scope_with_next_trigger | Campaign evidence remains single-host MI300X. |
| ROCm wheels | out_of_scope_with_next_trigger | Needs separate packaging and CI support work. |
| Multi-GPU ROCm | out_of_scope_with_next_trigger | Needs backend-neutral device ownership design. |
| Simultaneous CUDA+HIP | unavailable | Configure-time rejection remains active. |

## Profiler Evidence

`rocprof --hip-trace --stats` completed for
`interop-campaign5-profiler`. The trace captures FastPauli HIP work around
device-resident commutation output, dense host materialization, and compact
counting. It does not contain retained DLPack framework kernels because HIP
DLPack was rejected before public retention.

Profiler artifacts are under:

```text
docs/benchmarks/data/rocm_mi300x_campaign5_2026-04-30/profiler/
```

## README Landscape

The README landscape remains the broad CPU/CUDA/ROCm/external view. Campaign 5
did not add a retained comparable DLPack performance row, so the renderer
preserved the existing broad landscape and emitted a report-local interop plot.

## Remaining Headroom And Next Campaign

The recommended next ROCm campaign is Campaign 6 HIP expectation and HIP matmul
parity. Public HIP DLPack should stay blocked until a real ROCm consumer both
consumes a versioned read-only capsule and rejects mutation of the imported
view. ROCm portability, CI, packaging, multi-GPU, and backend-neutral
multi-accelerator design remain separate waves.

# Benchmark evidence

Every performance statement in Wolfgang's documentation points to a report in
`docs/benchmarks/reports/`. Each report names its hardware, toolchain, commit,
command, timing boundary and limitations, and follows the
[benchmark protocol](protocol.md). A speedup applies to the configuration a
report names, not to every platform or shape.

![Speedups measured in the portability and interop campaign](plots/portability_campaign_speedups.svg)

## Portability and interop campaign (November 2026)

Plan: `docs/plans/portability_and_interop_campaign_plan.md`. Review closeout:
`docs/reviews/portability_interop_campaign_2026-11-03.md`.

| Report | What it measured |
| --- | --- |
| [oneTBB in release wheels](reports/onetbb_wheels_2026-11-03.md) | Parallel simplify from pip wheels on an x86 VM and an M4 Pro |
| [Zero-copy device arrays](reports/device_arrays_h100_2026-11-03.md) | DLPack and CUDA Array Interface export and device-to-device import on an H100 |
| [Fused small simplify](reports/small_simplify_2026-11-03.md) | One-kernel simplify for small operators on H100 and MI300X |
| [Pooled commutation output](reports/cuda_commutation_pool_h100_2026-11-03.md) | `DeviceCommutationMatrix` from the memory pool on an H100 |
| [Small-operation profile](reports/cuda_small_op_profile_h100_2026-11-03.md) | Nsight Systems breakdown behind the streams-and-graphs decision |
| [HIP parity on MI300X](reports/hip_parity_mi300x_2026-11-03.md) | HIP simplify numeric contract and call overhead |
| [Metal follow-ups](reports/metal_followups_2026-11-03.md) | Parallel reduction, large inputs and precompiled kernels on an M4 Pro |

## Usability and kernel campaign (November 2026)

| Report | What it measured |
| --- | --- |
| [CPU parallel simplify](reports/cpu_parallel_simplify_2026-11-03.md) | oneTBB simplify against scalar on two x86 hosts |
| [CUDA allocation reuse](reports/cuda_allocation_reuse_h100_2026-11-03.md) | Pooled allocation and cheaper device selection on an H100 |
| [CUDA simplify paths](reports/cuda_simplify_paths_h100_2026-11-03.md) | Simplify path selection on an H100 |
| [Metal hybrid simplify](reports/metal_hybrid_simplify_2026-11-03.md) | GPU sort with CPU reduction on an M4 Pro |
| [Host API](reports/host_api_2026-11-03.md) | Host `PauliSum` boundary additions and commutator timings |

## Earlier campaigns

- **CUDA:** [Phase 11 H100](reports/cuda_phase11_h100_2026-04-27.md),
  [deep optimization](reports/cuda_deep_optimization_h100_2026-04-28.md)
  and campaigns 2-8, [cross-architecture](reports/cuda_cross_architecture_campaign10_2026-04-29.md),
  [residual risk](reports/cuda_residual_risk_campaign11_2026-04-29.md).
- **ROCm/HIP:** [MI300X bring-up](reports/rocm_mi300x_bringup_2026-04-29.md),
  campaigns 2-7, [architecture readiness](reports/rocm_campaign8_architecture_readiness_2026-05-01.md).
- **Apple Metal:** [bring-up](reports/apple_metal_bringup_2026-05-01.md),
  optimization campaigns 1-8, [Wave 1D commutation](reports/apple_metal_wave1d_2026-08-21.md).
- **CPU:** [Apple Silicon M4 Pro](reports/cpu_phase9_apple_silicon_m4_pro_2026-04-25.md),
  [AMD EPYC 9654](reports/cpu_phase9_x86_64_amd_epyc_9654_2026-04-25.md),
  [small-shape dispatch](reports/cpu_small_shape_dispatch_wave1b_2026-08-21.md).
- **Cross-backend:** [backend-neutral accelerators](reports/backend_neutral_accelerator_campaign9_2026-05-01.md),
  [Python boundary hardening](reports/python_boundary_hardening_2026-09-21.md).

The [research provenance ledger](../research/provenance.md) lists every
campaign with its plan, report and outcome, including rejected optimizations.

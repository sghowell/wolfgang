# CUDA Small-Operation Profile: Streams And Graphs

Date: 2026-11-03

This report closes slice 9 of the portability and interop campaign
(`docs/plans/portability_and_interop_campaign_plan.md`). The question was
whether CUDA graphs or a public stream argument would cut the cost of small
device operations now that slice 5 fused small simplify into one kernel. An
Nsight Systems breakdown says neither would today, and points at cheaper
fixes.

## Evidence

```text
data: docs/benchmarks/data/cuda_small_op_profile_h100_2026-11-03/api_breakdown.json
tool: Nsight Systems 2024.6.2, --trace=cuda; nsys stats cuda_api_sum, cuda_gpu_kern_sum, cuda_gpu_mem_time_sum
```

| Item | Value |
| --- | --- |
| Device | NVIDIA H100 80GB HBM3, sm_90, CUDA runtime 12.8, driver 13.0 |
| Build | `01997f7`; CUDA sources identical to main at `493eec3` |
| Method | Per operation, 50 warmup and 500 profiled calls in one process. Per-call API cost is the median duration times calls per operation, which excludes one-time pool growth. Wall time comes from a separate 2,000-call loop. |

## Breakdown

Per call, on device-resident operators (24 qubits unless noted):

| Operation | Wall | Launches | GPU kernel time | Largest host costs |
| --- | ---: | ---: | ---: | --- |
| `simplify`, 128 terms (fused) | 18.6 us | 1 | 7.8 us | sync 12.9 us (includes the kernel), launch 3.4 us |
| `simplify`, 4,096 terms | 188 us | 19 | 66.8 us | 19 launches 62.0 us, 8 stream syncs 44.4 us, 5 memcpys 41.7 us, 10 memsets 12.9 us |
| `matmul`, 128 x 128 | 202 us | 20 | 72.2 us | 20 launches 64.5 us, 8 stream syncs 46.5 us, 5 memcpys 41.7 us, 10 memsets 30.4 us |
| `commutes_with_device`, 128 x 128 | 64.1 us | 1 | 1.4 us | `cudaMalloc` 28.5 us, `cudaFree` 26.0 us |
| `expectation_statevector`, 64 terms, 12 qubits, host state | 30.1 us | 1 | 4.9 us | 2 `cudaMemcpy` 27.4 us |
| `to_device`, 128 terms | 17.7 us | 0 | 0 | 3 pageable `cudaMemcpy` 20.5 us |
| `to_host`, 128 terms | 26.3 us | 0 | 0 | 3 pageable `cudaMemcpy` 30.0 us |

## Findings

- **Single-launch operations gain nothing from graphs.** Small simplify,
  commutation and expectation each launch one kernel. Their remaining cost is
  the result synchronization, allocation, or transfers, which a graph does not
  remove.
- **Graph capture is not possible on the multi-launch paths as written.**
  Simplify above 1,024 terms and `matmul` synchronize eight times per call,
  because Thrust's `reduce`, `reduce_by_key` and `copy_if` return values (the
  maximum magnitude, run counts, survivor counts) to the host between stages.
  The next stage's sizes depend on those values, and a captured graph needs
  them fixed. Keeping the counts on the device would remove most of those
  synchronizations first. The 19-20 launches would then queue while earlier
  kernels run, with no graph needed.
- **`commutes_with_device` pays 55 us for an unpooled allocation.** Its
  output can be exported through DLPack and the CUDA Array Interface, so it
  uses `cudaMalloc`/`cudaFree` instead of the pool. Slice 3 gave
  `DevicePauliSum` a safe way to pool exportable buffers: synchronize the
  device before an exported buffer returns to the pool. The commutation
  output can use the same rule.
- **Transfers use pageable memory.** `to_device`, `to_host` and the host-state
  expectation issue one pageable `cudaMemcpy` per buffer, about 7-10 us each.

## Decision

- No CUDA graphs. They need fixed shapes per instantiation and help only
  launch-heavy sequences, and the only launch-heavy paths are blocked by
  data-dependent synchronizations.
- No public stream argument. Every operation returns host-visible results or
  sizes and synchronizes before returning. A stream argument would need
  asynchronous result objects, an API change with no measured benefit here.
  The slice 3 interop contract (synchronize before return; import synchronizes
  the device) already makes DLPack and CUDA Array Interface handoffs safe on
  any consumer stream.
- Follow-ups for slice 10, in order of measured payoff: pool the commutation
  output with sync-on-free after export; keep simplify's intermediate counts
  on the device; stage transfers through pinned memory.

HIP was not re-profiled. On the MI300X the slice 2 report
(`docs/benchmarks/reports/hip_parity_mi300x_2026-11-03.md`) puts most of the
remaining multi-launch simplify cost in 26 `hipMalloc`/`hipFree` calls
(60 us) and three rocThrust value returns (43 us), so the same reasoning
applies.

## Limitations

- One H100, one build, 24-qubit operators (`matmul` and multi-launch simplify
  take the 32-qubit packed-key path).
- Nsight Systems adds per-call overhead; the wall times come from runs
  without it.

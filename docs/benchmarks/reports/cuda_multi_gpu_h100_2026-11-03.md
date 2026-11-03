# CUDA On Two GPUs

Date: 2026-11-03

This report qualifies slice 6 of the accelerator follow-ups campaign
(`docs/plans/accelerator_followups_campaign_plan.md`): peer access for pooled
exports and multi-GPU tests. The new tests (`tests/test_cuda_multi_gpu.py`)
ran on two GPUs for the first time and found three problems, which the slice
fixes:

1. **Peer reads of exports faulted.** Pool memory was mapped only into its own
   device, so a kernel on device 0 that read a buffer exported from device 1
   (CuPy enables peer access for exactly this) hit an illegal address. The
   error is sticky: every later CUDA call in the process failed too.
2. **Two GPUs ran no faster than one.** Every device call held the GIL, so two
   Python threads, one per GPU, took turns.
3. **A process that used only GPU 1 held 516 MiB on GPU 0.** After each call,
   Wolfgang switched back to the thread's previous device with `cudaSetDevice`,
   which since CUDA 12 creates that device's primary context.

## What changed

- **Peer mapping.** The first export from a device (`device_arrays()`, or a
  `DeviceCommutationMatrix` through DLPack or the CUDA Array Interface) maps
  that device's pool into every peer GPU that can access it, with
  `cudaMemPoolSetAccess`. The mapping covers the pool's current and future
  blocks; processes that never export pay nothing.
- **GIL.** CUDA device calls release the GIL
  (`device_without_gil` in `bindings/python/pauli_sum_py.cpp`). Argument
  parsing, buffer pinning and NumPy results stay under it. Compute calls
  then take a per-device lock: without it, two threads on one GPU contended for
  its default stream and launch path and lost about a third of their
  throughput on small calls (0.63-0.65 times one thread for `matmul` of
  128 x 128 terms and simplify of 4,096 terms). Transfers skip the lock
  because their host-side copying overlaps. HIP and Metal builds keep the GIL.
- **Device switches.** One guard (`src/cuda/device_guard.cuh`) replaces the
  seven places that switched devices. It switches back to the caller's device
  only if that device already has a primary context; otherwise the thread
  stays on the operand's device, as with PyTorch's device guard.
- **Exit.** compute-sanitizer counted the idle pinned buffers of the
  process-wide `to_host` staging pool (slice 3) as leaks. An exit handler
  registered after the first pinned allocation, so it runs before the
  runtime's own, now frees them.

## Evidence

```text
benchmarks: benchmarks/bench_cuda_multi_gpu.py --suite threads --repeat 5   (3 runs per build)
            benchmarks/bench_cuda_call_overhead.py                          (3 runs per build)
            benchmarks/bench_cuda_multi_gpu.py --suite peer --repeat 5      (5 fresh processes per mode)
data:       docs/benchmarks/data/cuda_multi_gpu_h100_2026-11-03/multi_gpu_summary.json
            docs/benchmarks/data/cuda_multi_gpu_h100_2026-11-03/overhead_summary.json
```

| Item | Value |
| --- | --- |
| Devices | 2 x NVIDIA H100 80GB HBM3 (SXM), sm_90, connected by NVLink (`nvidia-smi topo -m`: NV18), driver 570.148.08 |
| Host | Intel Xeon Platinum 8480+, 52 vCPUs (KVM guest), Ubuntu 22.04.5, GCC 11.4, Python 3.12.16, CUDA 12.8.93, CuPy 14.2.0, NumPy 2.5.4 |
| Baseline (`base`) | main at `e9c8d0e`; its checkout also held the new test and benchmark files |
| Candidate (`cand`) | `f068943` |
| Build | `WOLFGANG_ENABLE_CUDA=ON`, `WOLFGANG_CUDA_ARCHITECTURES=90`, system oneTBB 2021.5 |

The threads and call-overhead suites alternated between the builds, three runs
each.

## Results

### Two GPUs from two threads

Each operation ran from one thread on device 0, then from two threads at once
(one per device, each making the same number of calls), then from two threads
on device 0. The table gives the two-thread throughput as a multiple of the
one-thread throughput, median over three runs:

| Operation (24 qubits) | One thread, per call | Two GPUs, base | Two GPUs, cand | One GPU, two threads, base | One GPU, two threads, cand |
| --- | ---: | ---: | ---: | ---: | ---: |
| simplify, 1M terms, device-resident | 344 us | 0.98 | 1.92 | 1.00 | 0.99 |
| simplify, 4,096 terms | 112 us | 0.99 | 1.88 | 0.99 | 0.96 |
| `matmul`, 128 x 128 terms | 125 us | 0.99 | 1.89 | 0.99 | 0.97 |
| `commutes_with_device` 2,048 x 2,048 and `count_commuting` | 98 us | 1.03 | 1.90 | 1.02 | 0.95 |
| `to_device`, 1M terms | 1.98 ms | 0.99 | 1.96 | 1.00 | 0.99 |
| `to_host`, 1M terms | 2.31 ms | 1.00 | 1.98 | 1.00 | 1.96 |

The per-run spread of the two-GPU ratio was within 0.04 for every candidate
row. `to_host` overlaps even on one GPU because most of its time is host-side
copying into new memory, which the lock does not cover.

### Cost of the peer mapping

Five fresh processes per mode on device 1, with device 0 as the peer; medians:

| | base, export | cand, export | cand, no export |
| --- | ---: | ---: | ---: |
| First `device_arrays()` | 19 us | 26.1 ms (13.4-29.5) | n/a |
| Second `device_arrays()` | 3 us | 4 us | n/a |
| Growing the pool by 1 GiB (eight 4M-term `to_device`) | 87.8 ms | 88.1 ms | 88.0 ms |
| simplify of 4,096 terms, per call, afterwards | 113 us | 111 us | 110 us |
| Memory this process holds on GPU 0 | 516 MiB | 0 MiB | 0 MiB |

The mapping is a one-time cost of about 26 ms per exporting device. It did not
slow later pool growth or calls, and it created no context on the peer: the
516 MiB in the baseline column is the context the old device switch created,
with or without an export.

### Single-GPU call overhead

`bench_cuda_call_overhead.py`, medians of per-run medians, three runs per
build: all 15 rows changed by 0.985-1.021x (baseline over candidate), within
run-to-run spread. Releasing and reacquiring the GIL and taking the device lock
added no measurable cost to calls of 10 us and up.

## Correctness

| Check | Result |
| --- | --- |
| `tests/test_cuda_multi_gpu.py` (new, 13 tests) | Before the fix the peer-read, GIL and second-device-only tests failed (illegal address, a 48 ms stall in a 54 ms `to_host`, 516 MiB on GPU 0); all pass after it. They cover every operation on device 1 against the CPU, the current device after calls from either device, mixed-device errors, a device-array round trip on device 1, peer reads of exported arrays and matrices from device 0, four threads on two devices, and the GIL during a long transfer |
| `WOLFGANG_VALIDATE_CUDA=1 python scripts/validate.py --profile core` | Passed on the two-GPU host at `82e0fac` (1,098 tests in the CUDA run, 747 in the CPU run) |
| `compute-sanitizer --tool memcheck --leak-check full` | 0 errors and 0 leaks over the in-process multi-GPU tests and `tests/test_device_transfers.py` at `3e5ba24` |

## Limitations

- One pair of NVLink-connected H100s. A pair without peer access gets no
  mapping (peer reads still fault, as before); the first-export cost on a
  PCIe-connected pair was not measured.
- The four-thread test is too slow under memcheck and was left out of the
  sanitizer run; the peer-read and second-device subprocess tests were also
  left out, because their children exit with live objects, whose pool blocks
  the sanitizer reports as unfreed.
- Peer reads were checked with a CuPy kernel. PyTorch refuses cross-device
  tensors, so it was not tested.
- HIP and Metal keep the GIL and were not tested on more than one device; no
  two-GPU MI300X host was available.
- Threads that share a GPU gain nothing from concurrency by design: they take
  turns on the device lock.
- When a thread's previous device has no context, the thread now stays on the
  operand's device after a call. Code that relied on Wolfgang making an unused
  device 0 current again would see device 1.

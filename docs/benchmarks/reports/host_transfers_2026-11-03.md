# Host Transfers Through Pinned Memory

Date: 2026-11-03

This report qualifies slice 3 of the accelerator follow-ups campaign
(`docs/plans/accelerator_followups_campaign_plan.md`). The previous campaign's
profile (`docs/benchmarks/reports/cuda_small_op_profile_h100_2026-11-03.md`)
found that transfers used one pageable `cudaMemcpy` per buffer. The plan
called for pinned staging "where that is measurably faster". Measured, that
turned out to be device-to-host copies only.

## What changed

`DevicePauliSum.to_host()` copies the x words, z words and coefficients:

- **CUDA:** asynchronously into a pinned buffer borrowed from a process-wide
  pool, in batches. An operator that fits in half the buffer (8 MiB) waits for
  the GPU once instead of three times; a larger one alternates halves,
  copying into one while the other is appended to the result. The result
  vectors are reserved rather than zero-filled.
- **HIP:** the same staged path for transfers up to 64 KiB. Larger transfers
  copy directly with ROCm's pageable `hipMemcpy`, which measured faster than
  staging from about 2 MB on the MI300X.
- **Both:** result vectors of 4 MiB or more are advised for transparent huge
  pages before their first touch (Linux only; NumPy does the same for its
  large arrays). The pool keeps at most four idle buffers (CUDA buffers grow
  to 16 MiB, HIP buffers to 128 KiB). If pinned memory cannot be had,
  `to_host` copies directly.

The shared logic is `src/detail/staged_transfer.hpp`. `tests/native/contracts.cpp`
tests it on the CPU with two fake streams, one that lands each copy at the
next wait and one that lands it when issued, and tests the pool with `malloc`
standing in for pinned allocation.

`to_device()` and the statevector upload in `expectation_statevector` are
unchanged. The driver's pageable host-to-device path beat staging up to 32 MB
on the H100 (3.3 against 7.6 us at 4 KiB, 2.24 against 2.65 ms at 32 MiB), and
the operator's host buffers are already resident, so there are no page faults
to avoid.

## Why device-to-host was slow

Copying into a new `std::vector` costs more than the copy. On these hosts
(KVM guests with transparent huge pages set to `madvise`), fresh host memory
faults in 4 KiB at a time at about 2 us per page. In the H100 microbenchmark,
where every repetition copies into one freshly allocated vector:

| Size | Pageable into `vector(n)` | Staged | Staged, huge pages | Direct, huge pages |
| ---: | ---: | ---: | ---: | ---: |
| 4 KiB | 9.5 us | 7.3 us | n/a | n/a |
| 64 KiB | 24.1 us | 17.0 us | n/a | n/a |
| 2 MiB | 421 us | 259 us | n/a | n/a |
| 8 MiB | 1.39 ms | 0.67 ms | 0.66 ms | 1.30 ms |
| 32 MiB | 16.2 ms | 17.9 ms | 9.4 ms | 8.4 ms |
| 128 MiB | 63.8 ms | 71.5 ms | 35.6 ms | 31.7 ms |

(`host_copy_microbench_h100.json`, from `benchmarks/native/host_copy_microbench.cu`.
"n/a": the huge-page columns advise only whole 2 MiB pages, so below 4 MiB they
repeat the work of the unadvised columns.) At 32 MiB the DMA itself takes about
1.5 ms; the rest of the pageable copy is page faults. Staging alone helps up to
8 MiB (by 23-52%), where glibc reuses freed heap memory, but not beyond, where
each vector faults in afresh; huge pages cut the faults.

The microbenchmark favored a direct copy into huge-page memory above 16 MiB,
and a build that did so for CUDA was measured end to end
(`cuda_h100_direct_above_16mib.json`). It lost to staging at every size above
the cutoff: 9.9 against 5.4 ms at 1M terms and 24 qubits, 8.1 against 3.2 ms at
1M terms and 100 qubits, 27.2 against 16.3 ms at 4M terms and 24 qubits, and a
tie at 4M terms and 100 qubits. `to_host` fills three vectors, not one, and the
allocator often reuses their memory, so the microbenchmark's every-page-faults
case is not representative. CUDA therefore stages at every size.

## Evidence

```text
benchmark: benchmarks/bench_host_transfers.py --backend <cuda|hip> --label <build>
probes:    benchmarks/probe_cold_to_host.py <cuda|hip>
data:      docs/benchmarks/data/host_transfers_2026-11-03/cuda_h100_summary.json
           docs/benchmarks/data/host_transfers_2026-11-03/hip_mi300x_summary.json
           docs/benchmarks/data/host_transfers_2026-11-03/cuda_h100_direct_above_16mib.json
           docs/benchmarks/data/host_transfers_2026-11-03/hip_mi300x_earlier_cutoffs.json
           docs/benchmarks/data/host_transfers_2026-11-03/cold_to_host.json
           docs/benchmarks/data/host_transfers_2026-11-03/host_copy_microbench_h100.json
```

| Item | H100 | MI300X |
| --- | --- | --- |
| Device | NVIDIA H100 80GB HBM3, driver 580.105.08, CUDA 12.8 | AMD Instinct MI300X VF, HIP 7.16.26385 |
| Host | Intel Xeon Platinum 8480+ (KVM), Ubuntu 24.04, THP `madvise` | Intel Xeon Platinum 8568Y+ (KVM), Ubuntu 24.04.5, THP `madvise` |
| Baseline | main at `4ca82d5` | main at `ad6f7a8` (HIP sources as in `4ca82d5`) |
| Candidate | `9e667c3` | `9e667c3` |

Both baseline checkouts held the candidate's benchmark script. Each build ran
the benchmark three times in randomized order; rows are medians of the three
per-run medians.

## Results

`to_host`, medians of per-run medians:

| Operator | H100 baseline | H100 new | Speedup | MI300X baseline | MI300X new | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 term, 24 qubits | 23.9 us | 9.7 us | 2.45x | 39.5 us | 21.7 us | 1.82x |
| 128 terms, 24 qubits | 26.3 us | 10.2 us | 2.58x | 31.6 us | 17.0 us | 1.86x |
| 4,096 terms, 24 qubits | 67.7 us | 20.9 us | 3.23x | 32.4 us | 33.0 us | 0.98x |
| 65,536 terms, 24 qubits | 532 us | 185 us | 2.88x | 3.00 ms | 3.00 ms | 1.00x |
| 1M terms, 24 qubits | 14.8 ms | 5.52 ms | 2.69x | 13.8 ms | 6.41 ms | 2.16x |
| 4M terms, 24 qubits | 38.0 ms | 15.5 ms | 2.46x | 32.3 ms | 10.4 ms | 3.12x |
| 4,096 terms, 100 qubits | 81.5 us | 25.4 us | 3.22x | 36.0 us | 36.8 us | 0.98x |
| 1M terms, 100 qubits | 8.74 ms | 3.21 ms | 2.72x | 2.44 ms | 2.45 ms | 1.00x |
| 4M terms, 100 qubits | 87.4 ms | 46.9 ms | 1.86x | 86.5 ms | 27.0 ms | 3.20x |

On the H100 every `to_host` row is 1.86-3.23x faster. On the MI300X the
staged small transfers are 1.82-1.88x faster, three of the four largest are
2.16-3.20x faster from huge pages, and the rest stay within 0.96-1.00x, which
is within their run-to-run spread. `to_device` stayed within 0.99-1.03x on the
H100 and 1.00-1.02x on the MI300X.

The HIP cutoff comes from two earlier MI300X builds
(`hip_mi300x_earlier_cutoffs.json`). Staging every transfer (`1277015`) was
0.67-0.82x at 65,536 terms and at 1M terms with 100 qubits. A 256 KiB cutoff
(`1f1d9c3`) removed those losses but was 0.98x at 4,096 terms (131 KB), so the
cutoff became 64 KiB.

### First call on a new thread

A call borrows its buffer from the process-wide pool, so only the first call
that needs a given size allocates pinned memory (`cold_to_host.json`; 15 fresh
threads each, medians):

| | H100 main | H100 new | MI300X main | MI300X new |
| --- | ---: | ---: | ---: | ---: |
| 1 term, first / second call | 59 / 26 us | 47 / 11 us | 50 / 42 us | 26 / 22 us |
| 128 terms, first / second call | 41 / 27 us | 27 / 11 us | 61 / 33 us | 45 / 18 us |
| 1M terms, first / second call | 6.3 / 6.0 ms | 9.8 / 2.9 ms | 1.67 / 1.64 ms | 1.70 / 1.69 ms |

An earlier per-thread buffer made each new thread pay for its own pinned
allocation; on HIP, 64 KiB took about 80 us, which a thread making a single
small call never recovered, hence the pool. The CUDA 1M-term first call on a
new thread is still slower than on main; see Limitations.

## Correctness

| Check | Result |
| --- | --- |
| `tests/native/contracts.cpp` | Staged copies through staging buffers of 64 bytes to 64 KiB, 0 to 1,000 terms, under both fake streams, round-trip exactly; a small operator waits once. A deliberately broken single-buffer variant fails both modes. The pool reuses a returned buffer (smallest fit first), never shares a busy one, and keeps at most four idle |
| `tests/test_device_transfers.py` (new) | Round trips from the empty operator to 1.2M terms (several double-buffered batches on CUDA, the direct path on HIP), from three threads (`to_host` holds the GIL, so the calls do not overlap), and after simplify, on both GPUs |
| `WOLFGANG_VALIDATE_CUDA=1 python scripts/validate.py --profile core` | Passed on the H100 at `1277015` (1,051 tests in the CUDA run); later commits passed the transfer, foundation, device-array and pipeline suites on both GPUs at `9e667c3` |
| `compute-sanitizer` on the transfer tests | memcheck (with leak check), racecheck and synccheck at `1277015`: 0 errors, 0 leaks |

## Limitations

- Two KVM guests. Page-fault cost depends heavily on virtualization and the
  host's transparent huge page setting: with THP disabled the advice does
  nothing, and on bare metal faults are cheaper, so the large-transfer gains
  may be smaller.
- On the H100, the first 1M-term `to_host` on a new thread took 9.8 ms against
  6.3 ms on main, while later calls on that thread took 2.9 ms against 6.0 ms.
  It looks like a new thread's results landing in fresh allocator memory,
  faulted in under the huge-page advice, but this was not isolated.
- The pool keeps up to four idle pinned buffers (up to 64 MiB on CUDA) for the
  life of the process; they are not revalidated after a device reset.
- Results between 4 and 32 MiB can come from the glibc heap once its dynamic
  mmap threshold has risen. The huge-page advice then stays on reused heap
  memory, and khugepaged may later grow the process's resident size, the same
  trade-off NumPy accepts.
- On the MI300X host, `to_host` of 65,536 terms at 24 qubits took a steady
  3.0 ms in both builds, against 0.19 ms for a larger 100-qubit operator; one
  earlier run of a candidate build took 0.14 ms. A plain `hipMemcpy` of each of
  its buffers takes 10-30 us, and the flat 3.0 ms looks like a fixed wait
  rather than copy work. It was not investigated further.
- The direct-copy fallback when pinned memory cannot be allocated has not run
  on hardware.

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

- **CUDA:** asynchronously into a per-thread pinned buffer (up to 16 MiB,
  freed when the thread exits), in batches that fill one half of the buffer
  while the other half is appended to the result vectors. An operator that
  fits in one half waits for the GPU once instead of three times. The result
  vectors are reserved rather than zero-filled.
- **HIP:** the same staged path for transfers up to 64 KiB. Larger transfers
  copy directly with ROCm's pageable `hipMemcpy`, which measured faster than
  staging from about 2 MB on the MI300X.
- **Both:** destinations of 4 MiB or more are advised for transparent huge
  pages before their first touch (Linux only; NumPy does the same for its
  large arrays). If pinned memory cannot be allocated, `to_host` falls back to
  the previous pageable copies.

The shared logic is `src/detail/staged_transfer.hpp`, tested on the CPU by
`tests/native/contracts.cpp` with a fake stream that runs each copy only at
the next wait.

`to_device()` and the statevector upload in `expectation_statevector` are
unchanged. The driver's pageable host-to-device path beat staging up to 32 MB
on the H100 (3.4 against 7.6 us at 4 KiB, 2.18 against 2.68 ms at 32 MiB), and
the operator's host buffers are already resident, so there are no page faults
to avoid.

## Why device-to-host was slow

Copying into a new `std::vector` costs more than the copy. On these hosts
(KVM guests with transparent huge pages set to `madvise`), each 4 KiB page of
fresh host memory faults in on first touch at about 2 us. At 32 MiB the H100
microbenchmark spent 16.2 ms on a pageable copy into a zero-filled vector, most
of it in page faults, although the DMA itself takes about 1.5 ms. Staging alone
did not help at that size (18.0 ms), because the result vector still faults in;
advising huge pages on the reserved storage cut it to 9.4 ms. At 8 MiB and
below, where glibc reuses freed heap memory, staging alone halved the time.

| Size | Pageable into `vector(n)` | Staged | Staged with huge pages |
| ---: | ---: | ---: | ---: |
| 4 KiB | 8.9 us | 7.3 us | 6.5 us |
| 64 KiB | 23.2 us | 16.9 us | 11.0 us |
| 2 MiB | 320 us | 257 us | 169 us |
| 8 MiB | 1.39 ms | 0.68 ms | 0.66 ms |
| 32 MiB | 16.2 ms | 18.0 ms | 9.4 ms |
| 128 MiB | 63.8 ms | 71.4 ms | 35.2 ms |

(`docs/benchmarks/data/host_transfers_2026-11-03/host_copy_microbench_h100.json`,
from `benchmarks/native/host_copy_microbench.cu`.)

## Evidence

```text
benchmark: benchmarks/bench_host_transfers.py --backend <cuda|hip> --label <build>
data:      docs/benchmarks/data/host_transfers_2026-11-03/cuda_h100_summary.json
           docs/benchmarks/data/host_transfers_2026-11-03/hip_mi300x_summary.json
           docs/benchmarks/data/host_transfers_2026-11-03/host_copy_microbench_h100.json
```

| Item | H100 | MI300X |
| --- | --- | --- |
| Device | NVIDIA H100 80GB HBM3, driver 580.105.08, CUDA 12.8 | AMD Instinct MI300X VF, HIP 7.16.26385 |
| Host | Intel Xeon Platinum 8480+ (KVM), Ubuntu 24.04, THP `madvise` | Intel Xeon Platinum 8568Y+ (KVM), Ubuntu 24.04.5, THP `madvise` |
| Baseline | main at `4ca82d5` | main at `ad6f7a8` (HIP sources as in `4ca82d5`) |
| Candidate | `1ea926e` | `1ea926e` (native build from `0dd5989`, identical native sources) |

Both baseline checkouts held the candidate's benchmark script. Each build ran
the benchmark three times in randomized order; rows are medians of the three
per-run medians.

## Results

`to_host`, medians of per-run medians:

| Operator | H100 baseline | H100 staged | Speedup | MI300X baseline | MI300X new | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 term, 24 qubits | 23.7 us | 9.7 us | 2.44x | 39.6 us | 21.8 us | 1.82x |
| 128 terms, 24 qubits | 26.2 us | 10.2 us | 2.58x | 31.8 us | 17.1 us | 1.86x |
| 4,096 terms, 24 qubits | 65.2 us | 20.9 us | 3.12x | 32.2 us | 33.7 us | 0.95x |
| 65,536 terms, 24 qubits | 516 us | 176 us | 2.94x | 3.00 ms | 3.00 ms | 1.00x |
| 1M terms, 24 qubits | 14.4 ms | 5.43 ms | 2.65x | 13.3 ms | 6.94 ms | 1.92x |
| 4M terms, 24 qubits | 37.0 ms | 16.3 ms | 2.27x | 33.0 ms | 12.3 ms | 2.69x |
| 4,096 terms, 100 qubits | 78.0 us | 25.2 us | 3.09x | 36.6 us | 36.4 us | 1.01x |
| 1M terms, 100 qubits | 7.73 ms | 3.22 ms | 2.40x | 2.91 ms | 2.49 ms | 1.17x |
| 4M terms, 100 qubits | 86.5 ms | 47.3 ms | 1.83x | 86.3 ms | 27.7 ms | 3.12x |

On the H100 every `to_host` row is 1.83-3.12x faster. On the MI300X the
staged small transfers are 1.82-1.87x faster, the largest are 1.92-3.12x
faster from huge pages, and the rows between stay within 0.95-1.17x. The
4,096-term row at 24 qubits takes the direct path, unchanged apart from the
huge-page advice that does not apply below 4 MiB; its 0.95x is within the
run-to-run spread of the earlier MI300X runs (0.97-0.98x). `to_device` rows
stayed within 0.97-1.03x on both devices.

## Correctness

| Check | Result |
| --- | --- |
| `tests/native/contracts.cpp` | Staged copies through 64-byte to 64 KiB staging buffers, 0 to 1,000 terms, with copies deferred to the next wait, round-trip exactly; a small operator waits once |
| `tests/test_device_transfers.py` (new) | Round trips from the empty operator to 1.2M terms (several double-buffered batches on CUDA, the direct path on HIP), on three threads at once, and after simplify, on both GPUs |
| `WOLFGANG_VALIDATE_CUDA=1 python scripts/validate.py --profile core` | Passed on the H100 at `1277015` (1,051 tests in the CUDA run); `1ea926e` changes only the HIP cutoff, the CUDA fallback path and benchmarks, and passed the transfer, foundation and device-array suites |
| `compute-sanitizer` on the transfer tests | memcheck (with leak check), racecheck and synccheck: 0 errors, 0 leaks |
| MI300X | Transfer, pipeline and pool suites pass at `0dd5989` |

## Limitations

- Two KVM guests. Page-fault cost depends heavily on virtualization and the
  host's transparent huge page setting: with THP disabled the advice does
  nothing, and on bare metal faults are cheaper, so the large-transfer gains
  may be smaller.
- Each thread that calls `to_host` can keep up to 16 MiB of pinned host memory
  until it exits.
- On the MI300X host, `to_host` of 65,536 terms at 24 qubits took 3.0 ms in
  both builds, against 0.19 ms for a larger 100-qubit operator. A plain
  `hipMemcpy` of each of its buffers takes 10-30 us, so the time goes to host
  memory management around the result. It was not investigated further.

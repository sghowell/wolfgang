# Host Commutation Into The Result

Date: 2026-11-03

This report qualifies the first follow-up after slice 8 of the accelerator
follow-ups campaign (`docs/plans/accelerator_followups_campaign_plan.md`).
`PauliSum.commutes_with` filled a zero-initialized `std::vector<uint8_t>`,
and the binding then copied it into a new NumPy array. Every kernel writes
every entry, so the zero fill was wasted, and the copy doubled the memory the
call touched.

Measuring the change showed that most of the cost was neither the fill nor the
copy but page faults. With two output-sized buffers per call, glibc often
returned freed memory to the system and the next call faulted it back in. How
often depended on where the allocator placed the buffers, so the same kernel
on the same operator could take 0.05 ms in one process and 0.32 ms in another.
Slice 8 had read this as "one-word AVX2 and AVX-512 commutation is bound by
writing its output" and kept those operators off oneTBB below 2048 x 2048
entries. That threshold is gone.

## What changed

- **Kernels write the result.** `PauliSum::commutes_with_into(rhs, out, max)`
  fills a caller-sized `std::span<uint8_t>`; `commutes_with` allocates once and
  calls it. The oneTBB kernel writes through the caller's pointer, so its
  threads also take the first-touch page faults in parallel. The whole-matrix
  SIMD and scalar wrappers are gone: every path runs a row kernel over
  `[0, rows)`.
- **Python.** `PauliSum.commutes_with` fills an uninitialized NumPy array
  directly. The new `PauliSum.commutes_with_into(other, output,
  max_commutation_matrix_entries=100000000)` fills a caller-owned
  one-dimensional C-contiguous NumPy bool array, with the same validation and
  messages as `DevicePauliSum.commutes_with_into`; the guardrail and every
  check run before anything is written. Both release the GIL.
- **Dispatch.** One-word operators on AVX2 or AVX-512 switch to oneTBB at
  `kAutoTbbPairwiseEntryThreshold` (331,776 entries), like wider ones.
  `kAutoTbbPairwiseOneWordX86SimdEntryThreshold` and its build-info key
  `tbb_pairwise_one_word_x86_simd_entries` are removed. Neither was released.
  The thread floors are unchanged: two threads, or three over AVX2 or AVX-512.
- `benchmarks/bench_cpu_commutation_kernels.py` gained `--method into` (a
  reused output array), the `auto` and `tbb` selectors, and `--layouts N`,
  which rebuilds the operators after N differently sized padding allocations
  and records each layout's median.

## Evidence

```text
benchmarks: benchmarks/bench_cpu_commutation_kernels.py --layouts 8 (EPYC, local M4 Pro) or 4 (M4 Pro wheels)
            three alternating runs per build; before = main, after = this branch, into = after with --method into
            benchmarks/bench_cpu_thread_floor.py, pinned shapes as in slice 8
data:       docs/benchmarks/data/host_commutation_into_2026-11-03/
```

| Host | Configurations | Builds |
| --- | --- | --- |
| AMD EPYC 9654, 16 vCPUs (KVM guest), Ubuntu 22.04, glibc 2.35, GCC 11.4, oneTBB 2021.5, Python 3.12.16, NumPy 2.5.4 | Single-threaded kernels pinned to vCPU 4; `auto` and `tbb` pinned to vCPUs `0,1,4,5`; thread floors pinned to `0,1`, `0,4`, `0,1,4` and `0,1,4,5` | source builds of main at `a410d48` and the branch at `b8b02cb` |
| Apple M4 Pro (8 performance and 4 efficiency cores), macOS 26.6, Apple clang 21, Python 3.12.12, NumPy 2.5.4 | Unpinned | source builds without oneTBB (NEON and scalar) of `a410d48` and `b8b02cb`; release wheels built by CI from the same commits (oneTBB 2022.3, 12 threads) |

Six long-running processes from another project share the EPYC host, unpinned
across its vCPUs, so every comparison alternates the builds and repeats. The
"before" runs used the branch's benchmark script against main's native build;
their provenance records `a410d48` with that one file modified.

Times below are the median of the per-run medians, and in brackets the fastest
and slowest heap layout across all runs of that build. Speedups divide the
before median by the after median.

## Results

### One thread, EPYC 9654

| Kernel, operator | Before | After | Speedup | Into a reused array |
| --- | --- | --- | --- | --- |
| AVX-512, 24 qubits, 576 x 576 | 0.054 ms [0.051-0.322] | 0.045 ms [0.042-0.047] | 1.21x | 0.043 ms |
| AVX-512, 24 qubits, 2048 x 2048 | 0.793 ms [0.639-3.982] | 0.644 ms [0.492-0.691] | 1.23x | 0.621 ms |
| AVX-512, 24 qubits, 4096 x 4096 | 14.6 ms [14.2-15.3] | 2.53 ms [2.43-2.74] | 5.78x | 2.60 ms |
| AVX-512, 100 qubits, 576 x 576 | 0.168 ms [0.120-0.207] | 0.157 ms [0.110-0.197] | 1.07x | 0.141 ms |
| AVX-512, 100 qubits, 2048 x 2048 | 1.96 ms [1.69-2.53] | 1.79 ms [1.53-2.45] | 1.09x | 2.25 ms |
| AVX-512, 100 qubits, 4096 x 4096 | 19.7 ms [19.0-21.7] | 7.23 ms [6.99-9.17] | 2.73x | 9.01 ms |
| AVX2, 24 qubits, 4096 x 4096 | 17.0 ms [16.5-17.6] | 4.84 ms [4.80-4.97] | 3.51x | 4.82 ms |
| AVX2, 100 qubits, 4096 x 4096 | 26.0 ms [25.4-26.5] | 13.7 ms [13.6-13.7] | 1.89x | 13.7 ms |
| scalar, 24 qubits, 4096 x 4096 | 81.4 ms | 68.8 ms | 1.18x | 68.7 ms |
| scalar, 300 qubits, 2048 x 2048 | 67.9 ms | 67.6 ms | 1.01x | 67.6 ms |

At 576 x 576 and 2048 x 2048 the old path's median was 0-23% slower, but its
slowest layouts were up to 6x slower than that median. At 4096 x 4096 it lost about 12 ms on every
kernel (AVX-512 12.1 ms, AVX2 12.2 ms, scalar 12.7 ms), the cost of faulting in
32 MB per call: the two freed 16 MB buffers crossed glibc's trim threshold, and
the single new one does not. The scalar kernel's own time hides most of the
saving below that size.

The two-word AVX-512 kernel's time depends on where the operators themselves
land, before and after: 1.5 or 2.4 ms at 2048 x 2048, whichever the layout.
That is why its reused-array column is slower than its allocating column; the
two ran in different processes with different operator placements. A one-off
probe, not kept as data, that placed only the output at each 64-byte offset
within a 4 KiB page changed no kernel's time by more than 5%.

### Four vCPUs, EPYC 9654

`auto` picks oneTBB from 331,776 entries with three or more threads. Before,
one-word operators below 2048 x 2048 ran one AVX-512 thread instead.

| Operator | Before | After | Speedup |
| --- | --- | --- | --- |
| 24 qubits, 576 x 576 | 0.055 ms [0.051-0.401] | 0.037 ms [0.034-0.079] | 1.50x |
| 24 qubits, 2048 x 2048 | 0.723 ms [0.675-3.765] | 0.345 ms [0.336-0.964] | 2.09x |
| 24 qubits, 4096 x 4096 | 13.7 ms [13.4-15.4] | 1.36 ms [1.32-1.64] | 10.06x |
| 100 qubits, 576 x 576 | 0.125 ms [0.114-0.168] | 0.088 ms [0.080-0.132] | 1.42x |
| 100 qubits, 2048 x 2048 | 1.29 ms [1.14-2.05] | 0.916 ms [0.768-1.157] | 1.41x |
| 100 qubits, 4096 x 4096 | 16.0 ms [15.7-16.4] | 3.74 ms [3.48-3.97] | 4.29x |
| 300 qubits, 576 x 576 | 2.32 ms | 2.29 ms | 1.02x |
| 300 qubits, 2048 x 2048 | 29.3 ms | 28.8 ms | 1.02x |

Forced oneTBB gives the same after times; before, it was 1.85x slower than
after at 576 x 576 even where auto did not pick it.

### M4 Pro

With the release wheels' oneTBB on 12 threads, `auto` is 2.44x faster at
24 qubits and 576 x 576 (0.075 to 0.031 ms), 1.42-1.44x at 2048 x 2048 and
4096 x 4096, 1.22-2.20x at 100 qubits, and 1.05-1.21x at 300 qubits. Once the
kernel runs in parallel, the serial fill and copy were a large share of the
call.

Single-threaded, NEON is 1.03-1.10x faster and scalar 1.00-1.04x, with the
layouts within a few percent of each other: macOS's allocator did not show the
glibc effect. A reused array is no faster than `commutes_with` there.

### One-word oneTBB against one AVX-512 thread

`benchmarks/bench_cpu_thread_floor.py` forces oneTBB at each thread cap and
one AVX-512 thread on the same one-word operators (24 qubits, 576 x 576 to
2896 x 2896) and reports oneTBB's speedup. Ranges cover the five sizes.

| Pinned shape | oneTBB threads | Before | After |
| --- | --- | --- | --- |
| One core's two hyperthreads (`0,1`) | 2 | 0.82-1.00 | 0.93-0.99 |
| Two cores (`0,4`) | 2 | 0.97-1.05 | 1.23-1.61 |
| Three vCPUs (`0,1,4`) | 3 | 0.80-0.98 | 1.38-1.92 |
| Four vCPUs (`0,1,4,5`) | 3 | 0.97-1.05 | 1.31-1.88 |
| Four vCPUs (`0,1,4,5`) | 4 | 0.95-1.04 | 1.25-1.86 |

Before, oneTBB tied or lost at every size and thread count, because both
paths paid the same serial fill, copy and page faults; slice 8 saw the same on
the EPYC and the Xeon and raised the one-word switch point to 2048 x 2048.
After, from the three-thread floor up, oneTBB wins from 576 x 576, the base
switch point, so the exception is removed. One core's two hyperthreads still
lose to one AVX-512 thread, which is why the floor over AVX2 and AVX-512 stays
at three. At two words oneTBB with three or more threads was 1.67-2.18x faster
after.

## Correctness

`tests/test_host_commutation_into.py` checks `commutes_with_into` against
`commutes_with` for every compiled selector at 24, 100 and 300 qubits and
shapes 577 x 613, 1 x 33, 33 x 1 and 1 x 1, into arrays prefilled with False
and with True, plus the dtype, shape, contiguity, length, read-only, qubit and
guardrail checks; the guardrail test confirms nothing is written. The
thread-floor tests now expect one-word operators on AVX2 or AVX-512 to use
oneTBB at 600 x 600 with three threads and not at 500 x 500. On the EPYC host
these suites passed pinned to one, two (one core's hyperthreads, and two
cores) and four vCPUs, and the full validation passed on the M4 Pro.

## Limitations

- The Xeon 8480+ that supported the 2048 x 2048 threshold in slice 8 was not
  available. The threshold is removed on the EPYC evidence, plus the finding
  that its measurements on both hosts had compared paths dominated by the same
  serial page faults and copy.
- No AVX2-only x86 host was measured. On AVX2 builds oneTBB runs the AVX2 row
  kernel, whose single-threaded gain here (1.10-3.51x at one word) was similar
  to AVX-512's.
- Page-fault costs depend on the allocator and its tuning (`M_TRIM_THRESHOLD`,
  `M_MMAP_THRESHOLD`, or another malloc), so the before-and-after ranges are
  specific to glibc 2.35's defaults.
- The two-word AVX-512 kernel's sensitivity to operator placement is
  unexplained and unchanged by this slice.

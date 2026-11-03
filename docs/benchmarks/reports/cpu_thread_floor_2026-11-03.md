# oneTBB Thread Floors

Date: 2026-11-03

This report qualifies slice 8 of the accelerator follow-ups campaign
(`docs/plans/accelerator_followups_campaign_plan.md`). Auto dispatch ran
simplify and large pairwise commutation on oneTBB only with at least four
threads, a floor nobody had measured. Two- and three-CPU environments are
common (Colab, small containers and CI runners, `set_num_threads(2)`), and all
of them got the single-threaded kernels. The approved rule: lower each kernel's
floor to the smallest thread count at which oneTBB is at least as fast as the
alternative from the switch point up on every host measured, and where oneTBB
loses only near the switch point, raise the switch point instead.

Measuring commutation exposed a second problem. The oneTBB commutation kernel
ran a scalar loop in each thread, so on a machine with AVX-512 it needed far
more threads than the floor to match one SIMD thread. With four threads on a
four-vCPU EPYC 9654 it ran at 0.23-0.26 times one AVX-512 thread, and auto
dispatch picked it for every matrix of 331,776 entries or more.

## What changed

- **oneTBB commutation runs SIMD.** Each SIMD commutation kernel became a
  row-range kernel with the whole-matrix function on top, and every oneTBB
  chunk runs the widest kernel the build and CPU support (AVX-512, AVX2 or
  NEON for one- and two-word operators, scalar otherwise). Keeping the operand
  pointers and the column count in locals also made the single-threaded
  kernels faster (see below).
- **Floors.** `kAutoTbbSimplifyMinConcurrency` is 2 (was 4). Pairwise
  commutation needs `kAutoTbbPairwiseMinConcurrency` = 2 threads when the
  alternative is scalar and `kAutoTbbPairwiseMinConcurrencyOverSimd` = 3 when a
  SIMD kernel covers the operator.
- **One-word threshold.** One-word commutation on AVX2 or AVX-512 uses oneTBB
  only from `kAutoTbbPairwiseOneWordX86SimdEntryThreshold` = 2048 x 2048
  entries. Its single-threaded kernel is now bound by writing the output (one
  byte per pair, zero-filled and then copied into NumPy), so threads pay off
  only for large matrices.
- `_build_info()["cpu_auto_dispatch_thresholds"]` reports the new values.

## Evidence

```text
benchmark: benchmarks/bench_cpu_thread_floor.py (new), pinned with taskset
data:      docs/benchmarks/data/cpu_thread_floor_2026-11-03/
```

| Host | Configurations | Builds |
| --- | --- | --- |
| AMD EPYC 9654, 16 vCPUs (KVM guest), Ubuntu 22.04, GCC 11.4, oneTBB 2021.5 | Pinned to one core's two hyperthreads (`0,1`), two cores (`0,4`), three vCPUs (`0,1,4`) and four vCPUs (`0,1,4,5`) | main at `e9c8d0e` (`epyc_main_*`), the final kernel at `9b5f421` (`epyc_*`) |
| Intel Xeon Platinum 8480+, 52 vCPUs (KVM guest, the two-H100 host), Ubuntu 22.04, GCC 11.4, oneTBB 2021.5 | The same four pinned shapes (`0,1`, `0,2`, `0,1,2`, `0,1,2,3`), eight vCPUs (`0-7`), and unpinned | the first kernel at `f83d837` (`xeon_first_kernel_*`), the final kernel at `a08b16d` (`xeon_*`) |
| Apple M4 Pro (8 performance and 4 efficiency cores), macOS 26 | Unpinned; macOS has no affinity control | release wheels built by CI from `f83d837` and `a08b16d` (oneTBB 2022.3) |

The EPYC host shares six of its vCPUs with long-running jobs from another
project; the benchmark used the four vCPUs on the two idle cores. Each case runs
the alternative and oneTBB at each cap in a shuffled order, five times
(nine for the two-core simplify repeat), and reports the per-call median.
`a08b16d` and `9b5f421` have the same kernels; `9b5f421` only adds the
one-word threshold.

## Results

### Simplify

oneTBB speed over scalar simplify from 8,192 to 1M terms (24 and 130 qubits,
no duplicates and half duplicates), as the minimum and median over sizes:

| Host and shape | One oneTBB thread | Two threads | Three | Four |
| --- | --- | --- | --- | --- |
| EPYC, one core's hyperthreads | 0.83, 0.90 | 1.09, 1.28 | | |
| EPYC, two cores (repeat, nine runs) | 0.83, 0.91 | 1.14, 1.28 | | |
| EPYC, four vCPUs | 0.83, 0.91 | 1.30, 1.45 | 1.19, 1.56 | 1.29, 1.59 |
| Xeon, one core's hyperthreads | 0.84, 0.91 | 1.11, 1.22 | | |
| Xeon, two cores | 0.84, 0.91 | 1.45, 1.54 | | |
| Xeon, four vCPUs | 0.85, 0.90 | 1.22, 1.49 | 1.40, 1.71 | 1.49, 1.89 |
| M4 Pro | 0.90, 1.10 | 1.70, 2.04 | 2.10, 2.63 | 2.26, 3.09 |

Two threads beat scalar at every size on every host. The first two-core EPYC
run had a bimodal 8,192- and 16,384-term case (0.57: some samples ran at 1.2 ms,
others at 0.49 ms) while the shared host was busy; the nine-run repeat measured
1.41 and 1.29. One oneTBB thread loses on the x86 hosts, so the floor is two.

### oneTBB commutation before and after

Four-vCPU EPYC, four oneTBB threads, as a multiple of one AVX-512 thread:

| Operator | main's oneTBB (scalar loop) | final oneTBB (AVX-512 per chunk) |
| --- | ---: | ---: |
| One word, 576 x 576 to 4096 x 4096 | 0.23-0.26 | 0.93-1.07 |
| Two words, 576 x 576 to 4096 x 4096 | 0.23-0.26 | 1.19-1.73 |

Single-threaded kernels, Xeon, one pinned core, medians of five alternating
fresh-process runs (`xeon_single_thread_kernels.json`, main `e9c8d0e` against
`9b5f421`):

| Kernel | One word, 576 x 576 | One word, 4096 x 4096 | Two words, 576 x 576 | Two words, 4096 x 4096 |
| --- | ---: | ---: | ---: | ---: |
| AVX-512 | 0.350 to 0.283 ms (1.24x) | 17.09 to 13.87 ms (1.23x) | 0.252 to 0.203 ms (1.24x) | 22.45 to 20.37 ms (1.10x) |
| AVX2 | 0.479 to 0.343 ms (1.40x) | 22.02 to 16.38 ms (1.34x) | 0.491 to 0.360 ms (1.36x) | 32.08 to 26.44 ms (1.21x) |
| Scalar | 2.15 to 1.33 ms (1.62x) | 105.3 to 66.3 ms (1.59x) | 3.60 to 3.54 ms (1.02x) | 188.4 to 183.5 ms (1.03x) |

The first version of the row-range refactor wrote through a byte pointer that
may alias the operand vectors, and the two-word AVX-512 kernel ran 6-20% slower
than main until the pointers and column count moved into locals.

### Commutation floors

Final kernel, oneTBB as a multiple of the alternative auto would otherwise use,
minimum to maximum over 576 x 576 to 4096 x 4096 (one-word AVX-512 split at
2048 x 2048):

| Host and shape | One word, AVX-512, below 2048 x 2048 | One word, AVX-512, from 2048 x 2048 | Two words, AVX-512 | Five words, scalar |
| --- | --- | --- | --- | --- |
| EPYC, hyperthreads, 2 threads | 0.82-0.96 | 0.99-1.00 | 0.94-1.12 | 1.26-1.27 |
| EPYC, two cores, 2 threads | 0.98-1.02 | 1.02-1.06 | 1.25-1.56 | 1.62-1.66 |
| EPYC, three vCPUs, 3 threads | 0.80-0.92 | 0.97-1.06 | 1.21-1.72 | 2.14-2.24 |
| EPYC, four vCPUs, 3 threads | 0.98-1.01 | 1.03-1.07 | 1.19-1.71 | 2.15-2.25 |
| Xeon, hyperthreads, 2 threads | 0.83-0.94 | 0.98-1.09 | 0.96-1.10 | 1.23-1.32 |
| Xeon, two cores, 2 threads | 0.95-1.01 | 1.01-1.18 | 1.29-1.58 | 1.96-2.07 |
| Xeon, four vCPUs, 3 threads | 0.86-0.93 | 1.01-1.21 | 1.31-1.57 | 2.13-2.35 |
| Xeon, eight vCPUs, 3-8 threads | 0.86-0.99 | 0.98-1.25 | 1.42-2.25 | |
| Xeon, unpinned, 2-16 threads | 0.93-1.00 | 0.99-1.25 | 1.29-3.36 | |
| M4 Pro (NEON), 2 threads | 1.54-1.76 (NEON) | 1.87-1.89 (NEON) | 1.70-1.92 | 1.94-1.98 |

- With no SIMD kernel (five words), two threads beat scalar everywhere: floor 2.
- Over a SIMD kernel, two threads on one core's hyperthreads tie or lose (the
  two share the core's vector units), and three threads win for two-word
  operators everywhere: floor 3.
- One-word AVX-512 commutation loses below 2048 x 2048 at every thread count
  on both x86 hosts. From 2048 x 2048 it is 0.97-1.25 times one AVX-512 thread:
  about even at 2048 x 2048 and ahead at 4096 x 4096. Its switch point rises to
  2048 x 2048. NEON on the M4 Pro wins from the
  base switch point with two threads, so NEON keeps it. AVX2 shares the
  AVX-512 rule without its own measurement: neither x86 host lacks AVX-512.

### What auto now picks on small machines

| Machine | simplify, 8,192 terms and up | Two-word commutation, 576 x 576 and up | One-word commutation |
| --- | --- | --- | --- |
| Two vCPUs (one core's hyperthreads) with AVX-512 | oneTBB, 1.09-1.47x over scalar on the EPYC and 1.11-1.33x on the Xeon (was scalar) | AVX-512, unchanged | AVX-512, unchanged |
| Four vCPUs (two cores) with AVX-512 | oneTBB, as before | oneTBB with AVX-512 per chunk, 1.19-1.73x over one AVX-512 thread (was main's oneTBB at 0.23-0.26x) | AVX-512 below 2048 x 2048 (was main's oneTBB at 0.23-0.26x), oneTBB above |
| M4 Pro, any cap of 2 or more | oneTBB | oneTBB from 3 threads | oneTBB from 3 threads |

## Correctness

| Check | Result |
| --- | --- |
| `tests/test_cpu_tbb_commutation.py` (new) | Forced oneTBB commutation equals forced scalar at 24, 64, 100 and 300 qubits, with term counts that leave partial SIMD lanes and uneven chunks, under caps of 1, 2 and 3 (x86 with AVX-512 and the M4 Pro wheel's NEON path) |
| `tests/test_cpu_thread_floor.py` (new) | The reported floors, and which kernel auto runs at each cap (seen through the fork guard: a child forked after oneTBB ran reports one thread). Written before the floors changed, and failed against the four-thread floor; the one-word threshold test failed before its change |
| Existing CPU suites | `tests/test_cpu_threads.py` and `tests/test_cpu_parallel_simplify.py` updated for the new floors; passed pinned to two, three and four vCPUs on both x86 hosts |
| `python scripts/validate.py --profile core` | Passed on the Xeon host at `292a124` (766 passed); the final commit's dispatch, equivalence and threads suites passed in every pinned shape |

## Limitations

- Two x86 hosts in KVM guests and one M4 Pro. The EPYC host was shared with
  another project's jobs, and one two-core run was disturbed (repeated). The M4
  Pro numbers come from CI-built wheels with oneTBB 2022.3; the x86 hosts used
  the system oneTBB 2021.5.
- AVX2 commutation shares the AVX-512 rule but was not measured on an AVX2-only
  CPU.
- The pinned shapes stand in for small VMs; a real two-vCPU VM may schedule its
  hyperthreads differently.
- Full grouping graph construction still runs its scalar oneTBB kernel; auto
  prefers SIMD for it, as before.
- One-word commutation is dominated by zero-filling the output and copying it
  into NumPy; skipping both is a separate follow-up.

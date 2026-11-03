# CUDA Simplify Paths On H100

Date: 2026-11-03

This report qualifies the CUDA simplify changes from the usability and kernel
campaign (`docs/plans/usability_and_kernel_campaign_plan.md`, slice 1):

- the CUDA build, broken on `main` since `ee4f95f`, compiles again;
- operators wider than 128 qubits reduce duplicates with a parallel segmented
  `reduce_by_key` instead of a one-thread kernel;
- one- and two-word paths use `stable_sort_by_key`;
- single-term inputs skip sort and reduction.

Every timed output was checked against CPU canonical labels and coefficients
before its timings were kept.

## Evidence

```text
summary: docs/benchmarks/data/cuda_simplify_paths_h100_2026-11-03/summary.json
raw data: docs/benchmarks/data/cuda_simplify_paths_h100_2026-11-03/raw/
benchmark: benchmarks/bench_cuda_simplify_paths.py
tests: tests/test_cuda_generic_simplify.py
```

Environment:

```text
GPU: NVIDIA H100 80GB HBM3, compute capability 9.0, driver 580.105.08
CUDA toolkit: 12.8.93; compiled CUDA architectures: 90
host: x86_64, 26 logical CPUs, GCC 13.3.0, Python 3.12.3, scalar CPU backend
loaded native build matches checkout native sources: true for every run
```

Commands:

```text
python benchmarks/bench_cuda_simplify_paths.py --suite generic --output generic.json
python benchmarks/bench_cuda_simplify_paths.py --suite specialized --label <base|cand> --seed <20261102+run> --output ...
```

Defaults: 2 warmups and 9 timed repetitions per variant. Seed 20261102.
Operators are random dense labels with complex normal coefficients. A
duplicate fraction `d` builds `round(n * (1 - d))` unique strings and fills
the rest by resampling them.

Boundaries: device-resident rows time `DevicePauliSum.simplify()` on an
operator already on the GPU. The call synchronizes and includes output
allocation. Transfer-inclusive rows add host-to-device copy and host
materialization.

## Generic path (more than 128 qubits)

Revision `2a3053b` (contains `4829fd3`). Within each repetition the parallel
path, the retained serial kernel (`WOLFGANG_CUDA_BENCH_GENERIC_MULTIWORD_REDUCTION=serial_kernel`)
and CPU simplify run in a randomized order. CPU simplify is the unchanged control.
All times are device-resident medians.

| qubits | terms | dup | parallel | serial kernel | CPU | vs serial | vs CPU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 130 | 4096 | 0.0 | 0.242 ms | 3.53 ms | 0.444 ms | 14.6x | 1.8x |
| 130 | 65536 | 0.0 | 0.436 ms | 60.4 ms | 10.0 ms | 138x | 23.0x |
| 130 | 65536 | 0.9 | 0.499 ms | 56.8 ms | 10.1 ms | 114x | 20.2x |
| 256 | 65536 | 0.0 | 0.566 ms | 58.6 ms | 10.7 ms | 104x | 18.9x |
| 1024 | 4096 | 0.9 | 0.658 ms | 11.3 ms | 0.798 ms | 17.1x | 1.2x |
| 1024 | 65536 | 0.0 | 0.650 ms | 184 ms | 20.2 ms | 283x | 31.0x |
| 1024 | 65536 | 0.9 | 1.10 ms | 196 ms | 19.4 ms | 178x | 17.6x |

At 128 terms the parallel path is 1.1x-2.1x faster than the serial kernel, but
both are about 30x slower than CPU simplify (0.2 ms vs 0.007-0.016 ms): the
per-call GPU floor dominates. The full 24-row matrix is in `summary.json`.

## Single-term fast path

The first generic run (`4232c1f`) showed the parallel path paying 0.21-0.26 ms
at one term, about 0.05 ms more than the serial kernel. After the fast path,
one-term simplify takes 0.029-0.030 ms at 130, 256 and 1024 qubits. The CPU
control stayed at 0.001 ms in both runs. The fast path applies to every
operator width.

## Specialized paths and stable sort

Seven paired reruns compare baseline `c65553c` (the build fix only) with
candidate `4232c1f`. Baseline/candidate order is randomized per rerun, and
each rerun uses a fresh seed shared by both builds. Values are
medians of the per-rerun medians; duplicate fraction 0.5.

| path | qubits | terms | device-resident cand/base | transfer-inclusive cand/base | CPU control cand/base |
| --- | ---: | ---: | ---: | ---: | ---: |
| one-word packed 32-bit key | 24 | 50,000 | 0.997 | 0.999 | 1.006 |
| one-word packed 32-bit key | 24 | 1,000,000 | 0.998 | 0.988 | 1.004 |
| one-word two-mask key | 60 | 50,000 | 0.999 | 0.986 | 1.000 |
| one-word two-mask key | 60 | 1,000,000 | 1.009 | 0.973 | 1.007 |
| two-word key | 100 | 50,000 | 0.993 | 0.985 | 0.995 |
| two-word key | 100 | 1,000,000 | 1.001 | 1.000 | 0.994 |

Every ratio is within the movement of the unchanged CPU control, so the
stable sort has no measurable cost. Absolute device-resident medians at one
million terms are 1.53 ms (packed key), 1.83 ms (two-mask key) and 2.46 ms (two
words); transfer-inclusive medians are 11.7-16.6 ms.

## Correctness and sanitizers

```text
tests/test_cuda_generic_simplify.py: 130, 256, 1024 qubits; 0, 1, 128, 4096 terms;
  duplicate fractions 0.0 and 0.9 with exact cancellation; relative threshold;
  single-term drop rules at 0, 3, 60, 100, 130 qubits; both reduction strategies
compute-sanitizer memcheck --leak-check full: 0 errors, 0 bytes leaked
compute-sanitizer racecheck: 0 hazards
tests/test_phase10_cuda_foundation.py and tests/test_phase11_cuda_kernels.py: pass
```

Compute Sanitizer runs print nanobind "leaked instances" notices at interpreter
shutdown. The same tests under plain pytest print none, so the notices come
from shutdown ordering under the sanitizer, not from leaked references.

## Interpretation

The generic path was the only CUDA simplify path whose cost grew linearly on
one GPU thread. It now scales like the specialized paths and beats CPU
simplify from about 4,096 terms. Below that, the fixed per-call cost of
allocation, launches and synchronization (about 0.2 ms) dominates every
CUDA simplify path. That cost is what slice 7 of the campaign targets.

## Limitations

- One H100 SXM (80 GB HBM3) host and one toolkit (12.8). No A100, Blackwell or HIP
  rows; HIP already used the parallel design and was not re-measured.
- Dense random labels only. Structured chemistry operators with 130+ qubits
  were not measured.
- Device-resident rows include output allocation and a device synchronize per
  call, so they are not kernel-only times.
- The single-term comparison uses two separate runs, not paired reruns. The
  CPU control is unchanged between them, and the effect (about 7x) is far
  larger than run-to-run variation.

# Metal Hybrid Simplify

Date: 2026-11-03

This report qualifies slice 8 of the usability and kernel campaign
(`docs/plans/usability_and_kernel_campaign_plan.md`). Metal kernels cannot do
`double` arithmetic, so the public `DevicePauliSum.simplify` on Metal used to
download the operator, simplify it on the CPU and upload the result (the host
bridge). For one-word operators (up to 64 qubits) of at least 12,288 terms it
now runs a hybrid:

1. The GPU sorts (x, z, input index) with an all-ascending bitonic network.
   Positions past the term count act as fixed +inf sentinels, so buffers are
   not padded to a power of two. Steps whose partner distance fits a
   1,024-entry tile run in threadgroup memory: 66 dispatches at 2^20 positions
   instead of 210.
2. The GPU gathers each sorted entry's coefficient bits into sorted order (a
   raw 16-byte copy, no floating-point arithmetic).
3. The CPU sums each duplicate run in float64 over the shared buffers, seeded
   with the run's first coefficient and in input order. It applies the CPU drop
   rule and compacts survivors in place. The result keeps the input's device
   and command queue.

Results equal CPU simplify bit for bit, signed zeros included. Unlike the
Campaign 7 fixed32 candidate, the hybrid places no restriction on coefficient
values. The kernel source is embedded in the extension at build time and
compiled once per process. If its pipelines cannot be built, simplify keeps the
host bridge.

The slice also caches the process's default `MTLDevice`, which uploads
previously created three times per `to_device(backend="metal")` call, and
uploads check availability without building the full status report. These
upload changes are not separately benchmarked here.

## Evidence

```text
benchmark: benchmarks/bench_metal_hybrid_simplify.py
summaries: docs/benchmarks/data/metal_hybrid_simplify_2026-11-03/summary_apple_m4_pro.json
           docs/benchmarks/data/metal_hybrid_simplify_2026-11-03/summary_apple_m4_pro_previous_68e1340.json
first call: docs/benchmarks/data/metal_hybrid_simplify_2026-11-03/first_call_apple_m4_pro.json
per-run reports: private archive (docs/benchmarks/data/**/raw/ is not published)
```

| Item | Value |
| --- | --- |
| Machine | Mac mini, Apple M4 Pro (8 performance + 4 efficiency cores) |
| OS | macOS 26.6.2 |
| Toolchain | Apple clang 21.0.0, Release build, no native CPU flags; Metal kernels compiled at run time from the embedded source |
| Metal device | Apple M4 Pro, unified memory |
| Storage mode | `MTLResourceStorageModeShared` for inputs, scratch and outputs |
| Command-buffer boundary | one command buffer per simplify (tile sort, global steps, tile merges, coefficient gather), committed and waited before the CPU reduction |
| CPU backend | scalar; oneTBB not installed; thread counts not controlled |
| Python | 3.12.12 |

Commands (3 runs, seed `20261105 + run`, 2 warmups and 9 timed repetitions,
variants in randomized order each repetition; default tolerances
`atol=1e-12`, `rtol=0`):

```text
candidate (cc7a9c7):        python benchmarks/bench_metal_hybrid_simplify.py --seed <s> --output run_<n>.json
previous design (68e1340):  same command; that revision's sizes were 196,608 to 1,048,576
first call (cc7a9c7):       python benchmarks/bench_metal_hybrid_simplify.py --first-call-processes 10 --output <file>
summary:                    python benchmarks/bench_metal_hybrid_simplify.py --summarize <dir> --output <summary>
```

Every run's loaded extension matched its checkout's native sources. The
candidate trees were clean. The previous-design trees were dirty only because
of uncommitted documentation edits. Commits after `cc7a9c7` change only
comments and documentation.

Variants, on random dense operators at 24 and 60 qubits with duplicate
fractions 0.0, 0.5 and 0.9:

- `metal_hybrid`: `DevicePauliSum.simplify` with
  `WOLFGANG_METAL_BENCH_SIMPLIFY=hybrid`, so sizes below the threshold measure
  the hybrid too;
- `metal_host_bridge`: the same call with `=host_bridge`, the previous public
  path and the public path below the threshold;
- `cpu_simplify`: `PauliSum.simplify` on the host copy.

Both Metal variants go from an operator already in shared buffers to a new
`DevicePauliSum`. There is no discrete transfer on unified memory: the bridge's
transfer cost is the copy into host vectors and the copy into new shared
buffers. Device-resident and transfer-inclusive timings therefore coincide for
the hybrid. Hybrid and bridge outputs were checked against CPU simplify bit for
bit before timing. The summary records per-row output sizes, per-run medians,
raw interquartile ranges and min/max.

## Results

Medians of per-run medians, min-max over the 6 width and duplicate
combinations at each size:

| Terms | Hybrid | Host bridge | CPU simplify | Hybrid vs bridge | Hybrid vs CPU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0.24-0.38 ms | 0.16-0.36 ms | 0.06-0.17 ms | 0.54-1.13x | 0.17-0.63x |
| 8,192 | 0.30-0.45 ms | 0.45-0.49 ms | 0.33-0.37 ms | 1.00-1.60x | 0.76-1.24x |
| 12,288 | 0.38-0.54 ms | 0.68-0.77 ms | 0.52-0.60 ms | 1.34-1.85x | 1.05-1.38x |
| 16,384 | 0.55-0.59 ms | 0.95-1.15 ms | 0.76-0.85 ms | 1.63-2.03x | 1.28-1.49x |
| 16,385 | 0.66-0.70 ms | 0.95-1.11 ms | 0.74-0.83 ms | 1.44-1.58x | 1.08-1.22x |
| 65,536 | 1.66-1.86 ms | 4.17-4.41 ms | 3.54-3.80 ms | 2.37-2.57x | 2.03-2.22x |
| 65,537 | 2.00-2.28 ms | 4.14-4.20 ms | 3.53-3.78 ms | 1.82-2.09x | 1.56-1.90x |
| 262,144 | 5.67-6.46 ms | 17.96-18.33 ms | 16.41-17.37 ms | 2.83-3.20x | 2.59-3.07x |
| 262,145 | 8.00-8.74 ms | 17.48-18.25 ms | 16.24-17.34 ms | 2.01-2.24x | 1.88-2.15x |
| 1,048,576 | 19.2-22.6 ms | 78.6-82.8 ms | 76.6-78.3 ms | 3.52-4.13x | 3.44-4.03x |
| 1,048,577 | 23.3-26.4 ms | 78.2-82.5 ms | 76.8-78.8 ms | 3.04-3.36x | 2.92-3.29x |

- **Threshold.** `kHybridSimplifyMinTerms = 12288` is the smallest measured
  size at which the hybrid beats both the bridge and same-host CPU simplify in
  every combination. Beating CPU simplify at the measured sizes is the plan's
  promotion gate, and it holds at every size from 12,288 up. At 8,192 terms the
  hybrid matches or beats the bridge but loses to CPU in some combinations; at
  4,096 its fixed cost (one command-buffer round trip, about 0.25 ms)
  dominates.
- **Just past a power of two.** Sizes of 2^k + 1 cost 1.2-1.4x the 2^k time,
  because the network gains a stage. The design keeps the cost proportional to
  the term count, not to the padded size. The hybrid still wins every
  combination at every such size from 16,385 up.
- **Duplicates.** From 262,144 terms the hybrid is fastest at a 0.9 duplicate
  fraction, because the CPU reduction writes fewer survivors. Below that the
  order varies.

**First call.** The first hybrid call in a process also compiles the
simplify kernels. At 12,288 terms and 40 qubits, over 10 fresh processes, it
took 4.0-4.3 ms (median 4.2 ms), against 1.1-1.2 ms for the bridge's first
call. Second calls took a median 0.53 ms and 0.76 ms respectively. The system
Metal shader cache was warm; an empty cache would make the first call slower.

**Previous design.** The previous design (`68e1340`) used a padded,
global-only bitonic sort, gathered coefficients through the index on the CPU,
and rebuilt the result through host vectors. Against the candidate, with the
same seeds, sizes and protocol in separate sessions:

| Terms | Previous hybrid | Candidate hybrid | Speedup |
| ---: | ---: | ---: | ---: |
| 262,144 | 9.91-13.15 ms | 5.67-6.46 ms | 1.75-2.10x |
| 1,048,576 | 39.4-49.9 ms | 19.2-22.6 ms | 1.95-2.34x |

The unchanged controls (CPU simplify and the bridge) moved by at most 2.8%
between the two sessions at the shared sizes. The previous design was not
measured at 2^k + 1 sizes; its network and buffers doubled there by
construction.

A stage trace of an intermediate build (tiled sort, but still gathering
coefficients through the sort index on the CPU and rebuilding the result
through `from_host`) at 1,048,577 terms attributed about 9.5 ms to GPU
execution, 4.5-14 ms to the CPU reduction and up to 6 ms to `from_host`. Those
measurements motivated the gather kernel and the in-place output.

## Correctness

| Check | Result |
| --- | --- |
| `tests/test_metal_hybrid_simplify.py` (26 tests) | Bit-identical to CPU simplify (raw coefficient bits) at 5, 33 and 64 qubits and all tolerance modes, including negated operators with `-0.0` components. Explicit `-0.0` in single-term and duplicate runs. 1-70,001 terms through the forced selector, covering tile edges (1,023, 1,024, 1,025, 2,047, 2,049). An all-Y 64-qubit key equal to the sentinel key; 262,145 and 300,007 terms; non-finite input and accumulation overflow errors; full cancellation. A run counter shows the public path takes the hybrid at 12,288 terms and the bridge at 12,287, for two-word operators and under `host_bridge`. A contract test pins that the kernels are embedded. |
| Missing source tree | With `src/metal/kernels/simplify.metal` moved away, a fresh process ran the hybrid at 20,000 terms and matched CPU simplify. |
| Metal suites | `tests/test_*metal*.py` and `tests/test_apple*.py`: 118 passed, 21 skipped (raw archive and wheel-only checks). |
| Independent review | A 72-case stress run (sizes 1-131,073, reverse-sorted keys, all keys equal to the sentinel key, `-0.0`, cancellation with atol/rtol, chained re-simplify and `commutes_with` on outputs) was bit-identical to CPU. |

## Limitations

- One Apple GPU (M4 Pro). Threadgroup memory use is 20 KiB per tile. Devices
  whose pipelines allow fewer than 512 threads per threadgroup keep the bridge.
- Measured against scalar CPU simplify only. A Metal source build on a Mac
  with oneTBB (`WOLFGANG_ENABLE_TBB=auto` finds Homebrew's) runs parallel CPU
  simplify from 8,192 terms in both the bridge and CPU paths, while the
  hybrid's CPU reduction stayed single-threaded in this report (it is parallel
  from 2^20 terms since `metal_followups_2026-11-03.md`). The threshold and
  speedups are unmeasured there. This shares the Apple Silicon oneTBB deferral in the plan.
- One-word operators only. Wider operators use the host bridge.
- The reduction runs on the CPU, so the public capability record for Metal
  `simplify` stays `host_bridge`.
- One command buffer covers the whole sort. Nothing above 1,048,577 terms was
  measured here; very large inputs could approach the GPU command-buffer
  timeout on a Mac driving a display. Follow-up: `metal_followups_2026-11-03.md`
  ran 4M, 16M and 64M terms without a timeout and splits large sorts across
  command buffers.
- GPU clocks ramp with load. Medians are from interleaved variants, so the GPU
  idles while the CPU variants run, which is representative of mixed workloads
  but slower than back-to-back GPU calls.
- Random dense operators only.

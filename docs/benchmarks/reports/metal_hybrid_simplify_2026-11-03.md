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
values.

The slice also caches the process's default `MTLDevice`
(`MTLCreateSystemDefaultDevice` cost about 45 us per call and was called three
times per upload). `to_device(backend="metal")` checks availability without
building the full status report. A one-term upload fell from about 228 us to
107 us (local spot check; median of 300 calls).

## Evidence

```text
benchmark: benchmarks/bench_metal_hybrid_simplify.py
summaries: docs/benchmarks/data/metal_hybrid_simplify_2026-11-03/summary_apple_m4_pro.json
           docs/benchmarks/data/metal_hybrid_simplify_2026-11-03/summary_apple_m4_pro_previous_68e1340.json
per-run reports: private archive (docs/benchmarks/data/**/raw/ is not published)
```

| Item | Value |
| --- | --- |
| Machine | Mac mini, Apple M4 Pro (8 performance + 4 efficiency cores) |
| OS | macOS 26.6.2 |
| Toolchain | Apple clang 21.0.0; Metal kernels compiled at runtime from source |
| Metal device | Apple M4 Pro, unified memory |
| Storage mode | `MTLResourceStorageModeShared` for inputs, scratch and outputs |
| Command-buffer boundary | one command buffer per simplify (tile sort, global steps, tile merges, coefficient gather), committed and waited before the CPU reduction |
| CPU backend | scalar (no oneTBB on this machine) |
| Python | 3.12.12 |

Commands (3 runs, seed `20261105 + run`, 2 warmups and 9 timed repetitions,
variants in randomized order each repetition):

```text
candidate (88a8242 + docs):  python benchmarks/bench_metal_hybrid_simplify.py --seed <s> --output run_<n>.json
previous design (68e1340):   same command, power-of-two sizes only
summary:                     python benchmarks/bench_metal_hybrid_simplify.py --summarize <dir> --output <summary>
```

Every run's loaded extension matched its checkout's native sources. The
candidate trees were dirty only because of uncommitted documentation.

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
bit before timing.

## Results

Medians of per-run medians, min-max over the 6 width and duplicate
combinations at each size:

| Terms | Hybrid | Host bridge | CPU simplify | Hybrid vs bridge | Hybrid vs CPU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 0.25-0.36 ms | 0.15-0.32 ms | 0.06-0.15 ms | 0.50-1.12x | 0.18-0.56x |
| 8,192 | 0.30-0.44 ms | 0.44-0.51 ms | 0.32-0.36 ms | 1.08-1.56x | 0.76-1.16x |
| 12,288 | 0.41-0.53 ms | 0.71-0.75 ms | 0.54-0.60 ms | 1.38-1.74x | 1.08-1.33x |
| 16,384 | 0.54-0.57 ms | 0.96-1.02 ms | 0.75-0.82 ms | 1.70-1.86x | 1.32-1.48x |
| 16,385 | 0.64-0.70 ms | 0.93-1.05 ms | 0.73-0.82 ms | 1.43-1.57x | 1.11-1.25x |
| 65,536 | 1.52-1.85 ms | 4.15-4.42 ms | 3.57-3.87 ms | 2.39-2.84x | 2.03-2.51x |
| 65,537 | 1.95-2.25 ms | 4.20-4.42 ms | 3.54-3.80 ms | 1.97-2.23x | 1.64-1.95x |
| 262,144 | 5.42-6.32 ms | 17.9-18.4 ms | 16.5-17.5 ms | 2.88-3.34x | 2.66-3.19x |
| 262,145 | 7.93-8.79 ms | 17.8-18.4 ms | 16.4-17.5 ms | 2.08-2.30x | 1.91-2.20x |
| 1,048,576 | 18.9-22.9 ms | 78.9-83.0 ms | 77.0-79.6 ms | 3.55-4.21x | 3.43-4.14x |
| 1,048,577 | 21.2-26.4 ms | 79.4-83.2 ms | 78.1-79.6 ms | 3.08-3.74x | 2.97-3.68x |

- **Threshold.** `kHybridSimplifyMinTerms = 12288` is the smallest measured
  size at which the hybrid beats the bridge in every combination. At 4,096
  terms the hybrid's fixed cost (one command-buffer round trip, about 0.25 ms)
  dominates.
- **Just past a power of two.** Sizes of 2^k + 1 cost 1.1-1.5x the 2^k time,
  because the network gains a stage. The design keeps the cost proportional to
  the term count, not to the padded size. The hybrid still wins every
  combination at every such size from 16,385 up.
- **Duplicates.** At a 0.9 duplicate fraction the hybrid is fastest, because
  the CPU reduction writes fewer survivors.

Previous design (`68e1340`, padded global-only bitonic sort, CPU gather through
the index, result rebuilt through host vectors) against the candidate, same
seeds, sizes and protocol in separate sessions:

| Terms | Previous hybrid | Candidate hybrid | Speedup |
| ---: | ---: | ---: | ---: |
| 262,144 | 9.91-13.15 ms | 5.42-6.32 ms | 1.76-2.12x |
| 1,048,576 | 39.4-49.9 ms | 18.9-22.9 ms | 1.88-2.35x |

The unchanged controls moved by at most 3% between the two sessions: CPU
simplify at 1,048,576 terms ran 76.1-78.9 ms before and 77.0-79.6 ms after. The
previous design was not measured at 2^k + 1 sizes. Its network and buffers
doubled there by construction.

A stage trace of an intermediate build (tiled sort, but still gathering
coefficients through the sort index on the CPU and rebuilding the result
through `from_host`) at 1,048,577 terms attributed about 9.5 ms to GPU
execution, 4.5-14 ms to the CPU reduction and up to 6 ms to `from_host`. Those
measurements motivated the gather kernel and the in-place output.

## Correctness

| Check | Result |
| --- | --- |
| `tests/test_metal_hybrid_simplify.py` (22 tests) | Bit-identical to CPU simplify (raw coefficient bits) at 5, 33, 64 qubits and all tolerance modes; `-op` signed zeros; 1-70,001 terms through the forced selector, covering tile edges (1,023, 1,024, 1,025, 2,047, 2,049); an all-Y 64-qubit key equal to the sentinel key; 262,145 and 300,007 terms; non-finite input and accumulation overflow errors; full cancellation. |
| Metal suites | `tests/test_*metal*.py` and `tests/test_apple*.py`: 114 passed, 21 skipped (raw archive and wheel-only checks). |

## Limitations

- One Apple GPU (M4 Pro). Threadgroup memory use is 20 KiB per tile. Devices
  whose pipelines allow fewer than 512 threads per threadgroup keep the bridge.
- One-word operators only. Wider operators use the host bridge.
- The reduction runs on the CPU, so the public capability record for Metal
  `simplify` stays `host_bridge`.
- GPU clocks ramp with load. Medians are from interleaved variants, so the GPU
  idles while the CPU variants run, which is representative of mixed workloads
  but slower than back-to-back GPU calls.
- Random dense operators only.

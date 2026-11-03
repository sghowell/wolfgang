# Metal Hybrid Simplify Follow-Ups

Date: 2026-11-03

This report qualifies slice 6 of the portability and interop campaign
(`docs/plans/portability_and_interop_campaign_plan.md`). It covers four
follow-ups to the Metal hybrid simplify
(`docs/benchmarks/reports/metal_hybrid_simplify_2026-11-03.md`):

1. **Parallel CPU reduction from 2^20 terms.** GCD chunks start on run
   boundaries, so each run keeps its sequential input-order sum and results
   stay bit-identical. When at least half the terms survive, chunks compact in
   place and shift into the scratch buffers, which become the output, so peak
   memory does not grow. Otherwise survivors are copied into exact-size
   outputs. The sort index is released right after the GPU wait.
2. **Large inputs.** At 4M, 16M and 64M terms the hybrid stayed bit-identical
   to CPU with no GPU timeout. As insurance, each merge stage commits its own
   command buffer above 2^22 positions, and each global step does above 2^26.
   One buffer covering a 2^26-position sort ran about 2 s.
3. **Precompiled kernels.** When Xcode's Metal compiler works at configure
   time (a trial compile of the kernels), they are embedded as a `.metallib`
   built for the deployment target. Otherwise the embedded source compiles on
   the first call. A private hook reports which library actually loaded.
4. **One source of truth for the sort tile.** A single CMake variable sets the
   tile size for the host code and both kernel paths.

## Evidence

```text
benchmark: benchmarks/bench_metal_hybrid_simplify.py
summaries: docs/benchmarks/data/metal_followups_2026-11-03/summary_large_main.json
           docs/benchmarks/data/metal_followups_2026-11-03/summary_large_candidate.json
first call: docs/benchmarks/data/metal_followups_2026-11-03/first_call_precompiled.json
            docs/benchmarks/data/metal_followups_2026-11-03/first_call_runtime_source.json
per-run reports: private archive (docs/benchmarks/data/**/raw/ is not published)
```

| Item | Value |
| --- | --- |
| Machine | Mac mini, Apple M4 Pro (8 performance + 4 efficiency cores) |
| OS | macOS 26.6.2 |
| Toolchain | Apple clang 21.0.0; Metal toolchain 32023.883 (precompiled build) |
| Storage mode | `MTLResourceStorageModeShared` |
| Command-buffer boundary | one buffer up to 2^22 sort positions; one per merge stage above; one per global step above 2^26 |
| CPU backend | scalar |

Commands: 3 runs, main (`1e669b9`) and the candidate (`ec7d9b0`) in randomized
order per run, 2 warmups and 5 timed repetitions:

```text
python benchmarks/bench_metal_hybrid_simplify.py --sizes 1048576 1048577 4194304 --repeat 5 --seed <20261105+run> --output <run>
python benchmarks/bench_metal_hybrid_simplify.py --first-call-processes 10 --output <file>
WOLFGANG_METAL_BENCH_RUNTIME_KERNELS=1 python benchmarks/bench_metal_hybrid_simplify.py --first-call-processes 10 --output <file>
```

Every run's loaded extension matched its checkout's native sources. The
candidate tree was dirty only because of uncommitted documentation.

## Results

Hybrid simplify, medians of per-run medians, min-max over the 6 width and
duplicate combinations:

| Terms | Main | Candidate | Speedup | CPU control, candidate / main |
| ---: | ---: | ---: | ---: | ---: |
| 1,048,576 | 18.5-23.0 ms | 17.5-20.2 ms | 1.01-1.17x | 1.00-1.03 |
| 1,048,577 | 19.2-26.1 ms | 19.9-23.4 ms | 0.91-1.13x | 1.00-1.02 |
| 4,194,304 | 83.2-98.4 ms | 80.2-85.9 ms | 0.99-1.15x | 0.99-1.03 |

Spot checks with a scratch script, two alternating runs per build, 40 qubits,
from packed arrays:

- **16,777,216 terms, half duplicates:** 437-495 ms on main, 387-436 ms on
  the candidate; CPU simplify took 1.59-1.63 s.
- **All-unique terms (the in-place path):** 13.7 against 11.8 ms at
  1,048,576 terms, and 85.2-86.0 against 76.6-76.8 ms at 4,194,304.

Below 2^20 terms the code path is unchanged. During development, a 65,536-term
threshold was neutral against main from 65,536 to 262,145 terms (0.95-1.09x),
because the GPU sort dominates there. That is why the parallel reduction
starts at 2^20 terms.

**First call** at 12,288 terms, 10 fresh processes each (warm system shader
cache):

| Kernels | First call, min / median / max |
| --- | --- |
| precompiled `.metallib` | 2.84 / 3.08 / 4.79 ms |
| runtime source (forced) | 3.34 / 3.89 / 5.17 ms |

The host bridge's first call took 1.1 ms. In a standalone test with a cold
shader cache, creating the library and pipelines took 154 ms from source
against 49 ms from the `.metallib`.

## Correctness

| Check | Result |
| --- | --- |
| `tests/test_metal_hybrid_simplify.py` (30 tests) | New cases are bit-identical to CPU simplify above 2^20 terms: runs spanning chunks, all-identical input, cancellation, an overflow in the last chunk, a 4,194,305-term input on the split-command-buffer path, and a 1.2M-term all-unique input on the in-place path. Hybrid run counters confirm the path. Precompiled and runtime-source kernels give identical results in fresh processes, each reporting the library it loaded. |
| Independent stress script | n = 2^20-1, 2^20, 2^20+1 and 3·2^20+7, with run boundaries at, before and after every naive chunk split, ±0.0 coefficients and all tolerance modes: bit-identical to CPU. |
| Large inputs | 4M, 16M and 64M terms bit-identical to CPU. |
| Metal suites and accelerator parity | Pass on the M4 Pro. |

## Limitations

- One Apple GPU and one macOS version.
- The parallel reduction's gain is modest (about 1.0-1.2x end to end), because
  the GPU sort dominates. Some 1,048,577-term combinations were slower
  (0.91x), within the run-to-run spread.
- The precompiled library needs Xcode's Metal toolchain at configure time. If
  the toolchain is removed afterwards, an incremental rebuild in the same build
  directory fails at the precompile step; reconfigure to fall back.
- Inputs near the 2^31-term limit were not run. The per-global-step split bounds
  each command buffer to one dispatch.

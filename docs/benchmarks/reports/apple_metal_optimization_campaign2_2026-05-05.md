# Apple Metal Campaign 2 Optimization And Evidence Report

Date: 2026-05-05

## Summary

Apple Metal Campaign 2 keeps Metal source-build support behind the existing
`backend="metal"` API and changes only the internal pairwise-commutation
kernel lane. The retained path uses a two-dimensional dispatch grid for
commutation. It selects `fp_pairwise_commutation_words1` for one packed word
and `fp_pairwise_commutation_generic` for two or more packed words.

The campaign also keeps `fp_pairwise_commutation_flat_generic` and forced
generic/specialized selectors, including `fp_pairwise_commutation_words2`, as
benchmark-only A/B candidates. These are not public APIs. All benchmark rows
preserve CPU correctness checks.

## Commands

```bash
env FASTPAULI_VALIDATE_METAL=1 uv pip install -e '.[test]' \
  --config-settings=cmake.define.FASTPAULI_ENABLE_CUDA=OFF \
  --config-settings=cmake.define.FASTPAULI_ENABLE_HIP=OFF \
  --config-settings=cmake.define.FASTPAULI_ENABLE_METAL=ON

env FASTPAULI_VALIDATE_METAL=1 .venv/bin/python benchmarks/bench_metal_kernels.py \
  --profile specialization \
  --repeat 20 \
  --json \
  --output docs/benchmarks/data/apple_metal_optimization_campaign2_2026-05-05/raw/metal_benchmark_specialization.json

.venv/bin/python scripts/render_apple_metal_assets.py \
  --data-dir docs/benchmarks/data/apple_metal_optimization_campaign2_2026-05-05 \
  --plot-dir docs/benchmarks/plots
```

## Environment

```text
Host: Mac mini, Apple M4 Pro
CPU cores: 12 total, 8 performance and 4 efficiency
CPU architecture: arm64
Metal device: Apple M4 Pro
Metal storage mode: MTLResourceStorageModeShared
macOS: Version 26.2 (Build 25C56)
Xcode/CLT compiler: AppleClang 21.0.0.21000099
Python: 3.12.12
NumPy: 2.4.4
FastPauli build mode: metal_only
Compiled backends: cpu, metal
Runtime-visible backends during benchmark: cpu, metal
Compiled CPU backends: scalar, neon
oneTBB, AVX2, AVX-512, and SVE: not compiled on this host
```

## Results

Timings are median seconds over 20 timed repetitions. The A/B rows use an
interleaved, rotating order inside each case. Each A/B correctness check
poisons the reused device-output matrix with the inverse expected matrix before
running the selected kernel, so stale expected contents cannot certify a
partial or missing write. CPU rows are same-host Apple Silicon CPU baselines.
Transfer-inclusive rows include operand transfer and host output.
Device-resident rows keep operands on Metal. Reused-output rows time
`commutes_with_device(..., output=existing_matrix)`.

| Case | Qubits | Terms | Words | CPU default | CPU scalar | CPU NEON | Metal transfer | Metal resident host output | Metal matrix reuse | Auto A/B reuse | Generic 2D A/B | Flat generic A/B | Specialized A/B |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| words1_512x512 | 64 | 512x512 | 1 | 1.64354e-04 | 5.16875e-04 | 1.46687e-04 | 8.72959e-04 | 2.34417e-04 | 1.11479e-04 | 1.12229e-04 | 1.17855e-04 | 1.43708e-04 | 1.08666e-04 |
| words2_384x384 | 96 | 384x384 | 2 | 1.34000e-04 | 4.71770e-04 | 1.27625e-04 | 9.22667e-04 | 2.25666e-04 | 1.30437e-04 | 1.86105e-04 | 1.28812e-04 | 1.46083e-04 | 1.21605e-04 |
| generic_words3_192x192 | 130 | 192x192 | 3 | 1.86750e-04 | 1.79000e-04 | n/a | 8.27708e-04 | 1.19354e-04 | 1.06646e-04 | 1.12437e-04 | 1.12354e-04 | 1.16792e-04 | n/a |

![FastPauli accelerator performance landscape](../plots/accelerator_landscape_with_rocm.svg)

## Interpretation

The retained kernel shape is correct across one-word, two-word, and generic
multi-word inputs. The 2D dispatch path records the intended
`[rhs_terms, lhs_terms, 1]` grid and `[16, 16, 1]` threadgroup shape, while the
legacy flat baseline records `[matrix_entries, 1, 1]` and `[256, 1, 1]`.

The one-word specialization is the clearest retained win in the interleaved
A/B lane on this host. The two-word result remains close enough to treat as
unsettled: the final Campaign 2 repeat shows the candidate slightly ahead of
generic 2D, while earlier checked evidence in this slice had the ordering
reversed. Campaign 2 therefore keeps `fp_pairwise_commutation_words2` as a
benchmark-only candidate and leaves the retained words=2 default on the generic
2D kernel. Shader-counter profiling and repeated A/B evidence should decide
whether a retuned two-word selector becomes default in a later Metal campaign.

The larger public boundary has not changed: host-output Metal paths are still
launch/allocation/materialization bound on these sizes, and same-host NEON CPU
remains very competitive. The device-output reuse boundary remains the right
place to study Metal kernel changes because it reduces output allocation and
host materialization noise.

## Checked Evidence

```text
Raw benchmark JSON: docs/benchmarks/data/apple_metal_optimization_campaign2_2026-05-05/raw/metal_benchmark_specialization.json
Summary JSON: docs/benchmarks/data/apple_metal_optimization_campaign2_2026-05-05/summary.json
README landscape plot: docs/benchmarks/plots/accelerator_landscape_with_rocm.svg
Campaign plan: docs/plans/apple_metal_optimization_campaign2_plan.md
Renderer: scripts/render_apple_metal_assets.py
Profiler evidence retained from: docs/benchmarks/data/apple_metal_bringup_2026-05-01/profiler/metal_system_trace_summary.json
```

## Remaining Headroom

1. Capture shader-counter and shader-timeline profiler evidence with an
   Instruments template that exposes Apple GPU counter data.
2. Use profiler evidence to decide whether the two-word specialized selector
   should remain benchmark-only, become default, or be retuned.
3. A/B test offline `.metallib` compilation against runtime NSString source
   compilation.
4. A/B test private storage plus blit staging for larger dense matrices.
5. Add true Metal reduction kernels for compact count and conflict consumers
   only if profiler and A/B evidence beat the current shared-memory CPU scan.
6. Add MPSGraph and PyTorch MPS external baselines only where the semantic
   mapping and timing boundary are exact.

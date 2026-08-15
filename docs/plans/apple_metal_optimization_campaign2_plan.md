# Apple Metal Optimization Campaign 2 Plan

Date: 2026-05-05

This plan defines the next Apple Metal optimization campaign after
`docs/benchmarks/reports/apple_metal_optimization_campaign1_2026-05-05.md`.
Campaign 1 showed that the current Metal path is correct but still dominated
by launch, allocation, and small-kernel overhead for the public host-output
boundary. Campaign 2 targets low-risk commutation-kernel overhead that can be
improved without changing the Python API or support claims.

## Scope

Campaign 2 is an internal Metal commutation kernel and benchmark-evidence
slice. It does not expand public API support beyond the existing
source-build-only `backend="metal"` lane.

The campaign covers:

```text
one-word specialized commutation kernel retained after A/B evidence
two-word specialized commutation kernel retained as a benchmark-only candidate
generic 2D fallback retained as the default commutation kernel for words >= 2
two-dimensional dispatch grid over rhs term by lhs term for the retained default
benchmark metadata that records selected kernel, 2D grid shape, and threadgroup shape
benchmark-only A/B rows for generic 2D and legacy flat generic dispatch
poisoned reused-output correctness checks for every benchmark-only A/B selector
specialization benchmark profile with one-word, two-word, and generic multi-word cases
README broad performance landscape refresh with the latest Apple Metal rows
checked report and summary artifacts for the retained implementation
```

Out of scope unless later A/B evidence justifies them:

```text
private storage, offline `.metallib`, Metal reductions, MPSGraph, and PyTorch MPS
Metal wheels
raw Metal buffer exports
public command queue, command buffer, event, heap, stream, graph, or workspace APIs
mixed CUDA/HIP/Metal source builds
new Metal operations such as simplify, expectation, or matmul
```

## Rationale

The previous generic kernel mapped a flat matrix-entry index to
`lhs_term, rhs_term` with division by `rhs_terms` and then looped over
`params.words`. Campaign 2 tests two possible ways to remove those costs:

```text
generic_2d: keep the generic packed-word loop but use a 2D grid to remove flat-index division
words1 candidate: one anti-commutation popcount with no loop
words2 candidate: two anti-commutation popcounts with no loop
flat_generic baseline: legacy flat grid with entry-to-row division
```

The 2D grid uses `thread_position_in_grid.x` for the rhs term and
`thread_position_in_grid.y` for the lhs term. This keeps output layout unchanged
as row-major `lhs_terms x rhs_terms`, but avoids the per-thread flat-index
division that was unnecessary on Metal. The specialized kernels are retained
as defaults only when A/B evidence beats the generic 2D default for the target
case. Campaign 2 retains words=1 specialization on the checked Apple M4 Pro
evidence and keeps words=2 specialization as a benchmark-only candidate because
the checked words=2 A/B rows have not consistently beaten generic 2D.

## Execution Ladder

1. Add failing tests for the Campaign 2 source-of-truth plan, benchmark
   profile, kernel names, 2D dispatch metadata, generated summary, and README
   landscape.
2. Implement specialized Metal commutation kernels and a private internal
   selector keyed by packed-word count.
3. Update the checked `.metal` source mirror and benchmark metadata so
   implementation and evidence use the same kernel names, grid shape, and
   threadgroup shape.
4. Add an internal benchmark-only override,
   `FASTPAULI_EXPERIMENTAL_METAL_COMMUTATION_KERNEL`, so the specialization
   profile can compare the retained `auto` selector against `generic_2d` and
   `flat_generic` baselines without adding a public Python API.
5. Extend `benchmarks/bench_metal_kernels.py` with the `specialization`
   profile. The profile must include one-word, two-word, and generic multi-word
   cases with CPU scalar, CPU default, and NEON baselines where available.
6. Generate Campaign 2 raw JSON, summary JSON, and the broad README landscape
   plot from correctness-checked benchmark data on the local Apple M4 Pro.
7. Update the architecture, roadmap, README, benchmark protocol, and report
   surfaces with retained evidence and remaining headroom.
8. Revalidate CPU-only, validate Metal when the runtime is visible, run the
   review stage, merge locally, validate the merged result, push, and confirm
   CI.

## Benchmark Profile

The `specialization` profile must include:

```text
metal_specialization_words1_512x512: 64 qubits, 512x512 terms, 1 packed word
metal_specialization_words2_384x384: 96 qubits, 384x384 terms, 2 packed words
metal_specialization_generic_words3_192x192: 130 qubits, 192x192 terms, 3 packed words
```

Each Metal commutation row must record:

```text
kernel: fp_pairwise_commutation_words1, fp_pairwise_commutation_words2, or fp_pairwise_commutation_generic
baseline kernel: fp_pairwise_commutation_flat_generic when the benchmark-only flat baseline is forced
dispatch_api: dispatchThreads_2d
grid_shape: [rhs_terms, lhs_terms, 1]
threadgroup_size: [16, 16, 1]
flat baseline dispatch_api: dispatchThreads_1d
flat baseline grid_shape: [matrix_entries, 1, 1]
flat baseline threadgroup_size: [256, 1, 1]
storage mode
command-buffer synchronization boundary
buffer allocation or reuse boundary
transfer boundary
correctness status
```

## Acceptance Criteria

Campaign 2 is complete only when:

```text
docs/plans/apple_metal_optimization_campaign2_plan.md is registered as a source-of-truth path
src/metal/commutation_metal.mm selects the specialized kernel for one-word inputs and generic 2D for two-word and larger inputs
src/metal/kernels/commutation.metal mirrors the retained kernel names and 2D dispatch shape
benchmarks/bench_metal_kernels.py exposes a specialization profile through --list-cases
specialization raw JSON contains interleaved auto, generic_2d baseline, flat_generic baseline, and one-word or two-word specialized-candidate rows for retained-output timing
docs/benchmarks/data/apple_metal_optimization_campaign2_2026-05-05/summary.json is generated from raw benchmark JSON
docs/benchmarks/reports/apple_metal_optimization_campaign2_2026-05-05.md records commands, environment, results, limitations, and remaining headroom
docs/benchmarks/plots/accelerator_landscape_with_rocm.svg remains a broad CPU/CUDA/ROCm/Apple Metal/external landscape rather than a narrow Metal-only chart
the smoke Metal benchmark remains suitable for scripts/validate.py
all retained benchmark rows preserve correctness checks and timing-boundary labels
benchmark-only A/B rows poison reused output before selector correctness checks
CPU-only validation and Metal validation pass on the appropriate local build modes
```

## Validation Commands

CPU-only default:

```bash
python scripts/validate.py
```

Metal source build and runtime lane when Metal is visible to the process:

```bash
FASTPAULI_VALIDATE_METAL=1 python scripts/validate.py
```

Specialization benchmark evidence:

```bash
FASTPAULI_VALIDATE_METAL=1 python benchmarks/bench_metal_kernels.py \
  --profile specialization \
  --repeat 5 \
  --json \
  --output docs/benchmarks/data/apple_metal_optimization_campaign2_2026-05-05/raw/metal_benchmark_specialization.json

python scripts/render_apple_metal_assets.py \
  --data-dir docs/benchmarks/data/apple_metal_optimization_campaign2_2026-05-05 \
  --plot-dir docs/benchmarks/plots
```

## Remaining Headroom After Campaign 2

The next Apple GPU campaign should use Campaign 2 data and profiler evidence to
choose from:

```text
shader-counter and shader-timeline profiling through Instruments
offline `.metallib` compilation versus runtime NSString source compilation
private storage plus blit staging for larger matrices
workspace or heap reuse for repeated commutation calls
true Metal reduction kernels for compact consumers
MPSGraph and PyTorch MPS external baselines when semantic mappings are exact
```

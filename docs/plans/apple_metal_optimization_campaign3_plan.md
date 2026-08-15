# Apple Metal Optimization Campaign 3 Plan

Date: 2026-05-06

This plan follows
`docs/benchmarks/reports/apple_metal_optimization_campaign2_2026-05-05.md`.
Campaign 2 improved the retained pairwise-commutation launch shape and left a
small set of focused Apple GPU questions. Campaign 3 answers those questions
with benchmark-only experimental selectors, profiler evidence, and checked
report artifacts while preserving the existing source-build-only
`backend="metal"` public boundary.

## Scope

Campaign 3 is an Apple Metal performance-hardening and evidence slice. It does
not expand public API support beyond the current source-build-only Metal lane.

The campaign covers:

```text
shader-counter and shader-timeline profiler evidence for the current local toolchain
two-word specialized selector decision using Campaign 3 A/B rows
offline `.metallib` compilation versus runtime NSString source compilation
private storage plus blit staging for larger host-output commutation calls
workspace or heap reuse decision based on the measured private/blit and retained-output boundaries
true Metal reduction kernels for compact consumers under a benchmark-only selector
MPSGraph and PyTorch MPS external baselines only where semantic mapping and timing boundaries are exact
README broad performance landscape refresh with CPU, CUDA, ROCm/HIP, Apple Metal, and external rows preserved
checked raw JSON, summary JSON, plot, and report artifacts
```

Out of scope:

```text
no public Metal API expansion
Metal wheels
raw Metal buffer exports
public Metal queues, command buffers, events, heaps, streams, graphs, or workspaces
changing CUDA, HIP, or CPU public behavior
claiming generic Apple GPU support from one local Apple M4 Pro evidence run
shipping MPSGraph or PyTorch MPS as FastPauli backend identities
```

## Experimental Selectors

Campaign 3 may add private environment selectors for benchmark evidence:

```text
FASTPAULI_EXPERIMENTAL_METAL_LIBRARY_PATH: load compute pipelines from an offline `.metallib` path
FASTPAULI_EXPERIMENTAL_METAL_OUTPUT_STORAGE=private: use private output storage plus shared blit staging for host output
FASTPAULI_EXPERIMENTAL_METAL_COMPACT_CONSUMER=gpu: run compact count consumers through Metal reduction kernels
```

These selectors are not public API. They exist so the benchmark can compare
runtime-source compilation, offline library loading, shared host-visible
storage, private storage with explicit blit staging, CPU compact scans, and GPU
compact reductions without changing Python method signatures.

## Benchmark Profile

The `campaign3` profile must include:

```text
metal_campaign3_words2_decision_384x384: 96 qubits, 384x384 terms, 2 packed words
metal_campaign3_large_private_storage_1024x1024: 64 qubits, 1024x1024 terms, 1 packed word
metal_campaign3_compact_reduction_512x512: 64 qubits, 512x512 terms, 1 packed word
```

Each retained or experimental Metal row must record:

```text
kernel name
dispatch API and grid shape
threadgroup shape
storage mode
library source: runtime_source or offline_metallib when applicable
metallib path when an offline library is used
command-buffer synchronization boundary
buffer allocation or reuse boundary
transfer boundary
correctness status
```

Private-storage rows must distinguish device-private output, shared staging,
and blit synchronization from shared output rows. Compact-consumer rows must
distinguish CPU scans over shared storage from GPU reduction kernels over the
same checked matrix. External baseline rows must record a semantic mapping and
must be skipped instead of approximated when no exact timing boundary exists.

## Acceptance Criteria

Campaign 3 is complete only when:

```text
docs/plans/apple_metal_optimization_campaign3_plan.md is registered as a source-of-truth path
benchmarks/bench_metal_kernels.py exposes the campaign3 profile through --list-cases
offline `.metallib` pipeline loading is implemented behind a benchmark-only selector
private storage plus blit staging is implemented behind a benchmark-only selector
true Metal reduction kernels are implemented behind a benchmark-only compact-consumer selector
Campaign 3 raw JSON contains two-word selector, private-blit, GPU compact-consumer, and metallib rows
MPSGraph and PyTorch MPS baseline statuses record exact semantic mapping or explicit skip reasons
docs/benchmarks/data/apple_metal_optimization_campaign3_2026-05-06/summary.json is generated from raw benchmark JSON
docs/benchmarks/reports/apple_metal_optimization_campaign3_2026-05-06.md records commands, environment, results, limitations, and remaining headroom
docs/benchmarks/plots/accelerator_landscape_with_rocm.svg remains an across-the-board CPU/CUDA/ROCm/HIP/Apple Metal/external landscape
README.md and docs/roadmap.md point to Campaign 3 as the latest Apple Metal evidence
all experimental rows preserve correctness checks and do not change public API behavior
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

Campaign 3 benchmark evidence:

```bash
FASTPAULI_VALIDATE_METAL=1 python benchmarks/bench_metal_kernels.py \
  --profile campaign3 \
  --repeat 10 \
  --json \
  --output docs/benchmarks/data/apple_metal_optimization_campaign3_2026-05-06/raw/metal_benchmark_campaign3.json

python scripts/render_apple_metal_assets.py \
  --data-dir docs/benchmarks/data/apple_metal_optimization_campaign3_2026-05-06 \
  --plot-dir docs/benchmarks/plots
```

## Remaining Headroom After Campaign 3

Campaign 3 should leave only evidence-backed next work. Possible follow-up
items include:

```text
promote private storage, GPU compact reductions, or two-word specialization only if measured wins are stable across larger shapes
replace benchmark-only selectors with an internal policy if multiple Campaign 3 rows justify the change
repeat Apple Metal profiling on additional Apple Silicon generations before broadening support claims
consider MPSGraph or PyTorch MPS only if exact sparse Pauli mappings become available without host materialization
```

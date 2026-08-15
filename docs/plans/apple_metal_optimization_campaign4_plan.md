# Apple Metal Optimization Campaign 4 Plan

Date: 2026-05-06

This plan follows
`docs/benchmarks/reports/apple_metal_optimization_campaign3_2026-05-06.md`.
Campaign 3 left a narrow set of evidence-backed Apple GPU questions. Campaign
4 addresses those questions without changing the source-build-only
`backend="metal"` public boundary and without reopening release publication or
platform-support work.

## Scope

Campaign 4 is an Apple Metal performance-hardening slice for the current local
Apple M4 Pro source-build lane. It does not expand public API support.

The campaign covers:

```text
two-word selector remains benchmark-only while larger 768x768 two-word evidence is captured
larger compact-consumer matrices that stress CPU shared-memory scans and GPU reductions
parallel block-reduction compact total count under a benchmark-only selector
private storage for device-only intermediate workflows, not current host-output default policy
sanitized derived shader-counter exports or an explicit blocker when Instruments cannot emit narrow value CSVs
MPSGraph and PyTorch MPS remain skipped unless an exact sparse Pauli mapping exists
README broad performance landscape refresh with CPU, CUDA, ROCm/HIP, Apple Metal, and external rows preserved
checked raw JSON, summary JSON, plot, profiler evidence, and report artifacts
```

Out of scope:

```text
PyPI publication, Windows support, and older macOS compatibility are out of scope
no public Metal API expansion
Metal wheels
raw Metal buffer exports
public Metal queues, command buffers, events, heaps, streams, graphs, or workspaces
changing CUDA, HIP, or CPU public behavior
claiming generic Apple GPU support from one local Apple M4 Pro evidence run
shipping MPSGraph or PyTorch MPS as FastPauli backend identities
```

## Experimental Selectors

Campaign 4 may use all Campaign 3 benchmark-only selectors and adds one new
compact-consumer selector:

```text
FASTPAULI_EXPERIMENTAL_METAL_LIBRARY_PATH: load compute pipelines from an offline `.metallib` path
FASTPAULI_EXPERIMENTAL_METAL_OUTPUT_STORAGE=private: use private output storage plus shared blit staging for host output
FASTPAULI_EXPERIMENTAL_METAL_COMPACT_CONSUMER=gpu: run current GPU compact reductions
FASTPAULI_EXPERIMENTAL_METAL_COMPACT_CONSUMER=gpu_parallel_total: run the total count through a per-threadgroup partial-sum reduction
```

These selectors are not public API. `gpu_parallel_total` is intentionally a
total-count experiment; row and column counts stay on existing CPU or GPU
paths until a separate axis reduction design is justified by evidence.

## Benchmark Profile

The `campaign4` profile must include:

```text
metal_campaign4_words2_large_768x768: 96 qubits, 768x768 terms, 2 packed words
metal_campaign4_compact_large_2048x2048: 64 qubits, 2048x2048 terms, 1 packed word
metal_campaign4_private_device_boundary_2048x2048: 64 qubits, 2048x2048 terms, 1 packed word
```

Each retained or experimental Metal row must record:

```text
kernel name
dispatch API and grid shape
threadgroup shape
storage mode
library source: runtime_source or offline_metallib when applicable
command-buffer synchronization boundary
buffer allocation or reuse boundary
transfer boundary
correctness status
```

Parallel compact rows must record the partial-output count, input-entry count,
threadgroup width, and the transfer boundary
`compact_consumer_gpu_parallel_block_reduction`. The benchmark must compare the
parallel reduction against the same checked matrix's CPU shared scan and
current GPU atomic total reduction.

## Acceptance Criteria

Campaign 4 is complete only when:

```text
docs/plans/apple_metal_optimization_campaign4_plan.md is registered as a source-of-truth path
benchmarks/bench_metal_kernels.py exposes the campaign4 profile through --list-cases
fp_count_commuting_total_block_sums is implemented behind the gpu_parallel_total benchmark selector
Campaign 4 raw JSON contains larger two-word selector, larger compact-consumer, private-blit, GPU atomic, GPU parallel compact, and metallib rows
MPSGraph and PyTorch MPS baseline statuses record exact semantic mapping or explicit skip reasons
profiler evidence records sanitized derived shader-counter export status without retaining raw trace bundles
docs/benchmarks/data/apple_metal_optimization_campaign4_2026-05-06/summary.json is generated from raw benchmark JSON
docs/benchmarks/reports/apple_metal_optimization_campaign4_2026-05-06.md records commands, environment, results, limitations, and remaining headroom
docs/benchmarks/plots/accelerator_landscape_with_rocm.svg remains an across-the-board CPU/CUDA/ROCm/HIP/Apple Metal/external landscape
README.md and docs/roadmap.md point to Campaign 4 as the latest Apple Metal evidence
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

Campaign 4 benchmark evidence:

```bash
FASTPAULI_VALIDATE_METAL=1 python benchmarks/bench_metal_kernels.py \
  --profile campaign4 \
  --repeat 10 \
  --json \
  --output docs/benchmarks/data/apple_metal_optimization_campaign4_2026-05-06/raw/metal_benchmark_campaign4.json

python scripts/render_apple_metal_assets.py \
  --data-dir docs/benchmarks/data/apple_metal_optimization_campaign4_2026-05-06 \
  --plot-dir docs/benchmarks/plots
```

## Remaining Headroom After Campaign 4

Campaign 4 should leave only evidence-backed next work. Possible follow-up
items include:

```text
promote two-word specialization only after stable wins across more shapes and at least one additional Apple Silicon generation
replace compact GPU selectors with an internal policy only if parallel reductions beat CPU shared scans on realistic downstream workloads
design axis-wise parallel reductions only if row or column compact consumers become a demonstrated bottleneck
use private storage only inside future fused device-only workflows that avoid host-output blit costs
repeat profiler counter export attempts when Instruments can produce sanitized value CSVs without raw trace retention
```

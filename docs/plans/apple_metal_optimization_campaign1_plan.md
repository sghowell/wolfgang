# Apple Metal Optimization Campaign 1 Plan

Date: 2026-05-05

This plan defines the first Apple Metal optimization and evidence campaign after
the source-build bring-up in
`docs/benchmarks/reports/apple_metal_bringup_2026-05-01.md`.

## Scope

Campaign 1 is a measured hardening pass for the existing Metal commutation
surface. It does not expand public API support beyond the source-build-only
`backend="metal"` lane.

The campaign covers:

```text
architecture and README evidence drift after the successful Metal System Trace
Metal benchmark scaling beyond the original 128x128 smoke case
same-host CPU scalar, CPU default, and NEON baselines for every Metal case
transfer-inclusive, device-resident, retained device matrix, reused-output, host-materialization, and compact-consumer timing boundaries
broad README performance landscape rows that include Apple Metal next to CPU, CUDA, ROCm/HIP, and external baselines
profiling status that distinguishes available Metal System Trace evidence from remaining shader-counter and shader-timeline work
```

Out of scope:

```text
Metal wheels
raw Metal buffer exports
public command queue, command buffer, event, heap, stream, graph, or workspace APIs
mixed CUDA/HIP/Metal source builds
MPSGraph-first kernel rewrites
PyTorch MPS implementation paths
new Metal operations such as simplify, expectation, or matmul
```

## Execution Ladder

1. Add tests that fail if the Metal benchmark remains smoke-only, the README
   broad landscape omits Apple Metal rows, or the architecture doc still says
   profiler tooling is blocked.
2. Extend `benchmarks/bench_metal_kernels.py` with smoke and scaling profiles.
   The smoke profile remains validation-friendly. The scaling profile records
   one-word, multi-word, rectangular, and larger square commutation regimes.
3. Add reusable evidence rendering for Apple Metal benchmark JSON. The renderer
   must produce a checked summary and refresh the broad README landscape while
   preserving CPU, CUDA, ROCm/HIP, and external rows.
4. Run a Metal source-build benchmark profile under the same elevated Apple
   Silicon execution requirement recorded by the bring-up report.
5. Update the architecture, roadmap, README, benchmark protocol, and checked
   report surfaces with the new evidence and remaining headroom.
6. Revalidate CPU-only, validate Metal when the runtime is visible, run the
   review stage, merge locally, validate the merged result, push, and confirm
   CI.

## Benchmark Profiles

The benchmark script owns these profiles:

```text
smoke: one deterministic 16-qubit 128x128 case for validation
scaling: multiple deterministic cases that cover one-word, multi-word, rectangular, and larger square matrices
```

Every case must record:

```text
case name
profile
num_qubits
lhs_terms
rhs_terms
term_weight
random_seed
packed word count
matrix entry count
repeat count
build metadata
metal runtime status
Apple SoC and Metal device metadata when available
storage mode
command-buffer synchronization boundary
buffer allocation or reuse boundary
```

Every timed row must keep correctness checks enabled and label one of these
boundaries:

```text
host_materialized
transfer_inclusive
device_resident
device_output_allocating
device_output_reused
device_output_to_host
compact_consumer
```

## Accepted Initial Optimization Targets

Campaign 1 may retain only low-risk optimizations that preserve public behavior
and benchmark boundaries:

```text
measure reused-output commutes_with_device_into separately from allocation-inclusive commutes_with_device
measure DeviceCommutationMatrix.to_host separately from device-output creation
measure axis-specific compact consumers separately from total count
record whether compact consumers are CPU scans over shared Metal storage or true Metal reductions
```

Private-storage buffers, reusable workspaces, command queue injection, compiled
`.metallib` packaging, specialized one-word kernels, two-dimensional dispatch,
and Metal reduction kernels require A/B data before they are retained in a
public implementation slice.

## Acceptance Criteria

Campaign 1 is complete only when:

```text
docs/architecture/apple_accelerator.md records that Metal System Trace evidence exists
README.md no longer says Apple Metal rows still need to be added to the landscape plot
docs/benchmarks/plots/accelerator_landscape_with_rocm.svg includes CPU, CUDA, ROCm/HIP, and Apple Metal rows
docs/benchmarks/data/apple_metal_optimization_campaign1_2026-05-05/summary.json is generated from checked raw benchmark JSON
docs/benchmarks/reports/apple_metal_optimization_campaign1_2026-05-05.md records commands, environment, results, limitations, and next headroom
the smoke Metal benchmark remains suitable for scripts/validate.py
the scaling Metal benchmark can run explicitly without changing default validation cost
pytest coverage fails on stale README, architecture, benchmark profile, or plot-evidence drift
all retained rows preserve correctness checks and timing-boundary labels
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

Scaling benchmark evidence:

```bash
FASTPAULI_VALIDATE_METAL=1 python benchmarks/bench_metal_kernels.py \
  --profile scaling \
  --repeat 5 \
  --json \
  --output docs/benchmarks/data/apple_metal_optimization_campaign1_2026-05-05/raw/metal_benchmark_scaling.json

python scripts/render_apple_metal_assets.py \
  --data-dir docs/benchmarks/data/apple_metal_optimization_campaign1_2026-05-05 \
  --plot-dir docs/benchmarks/plots
```

## Remaining Headroom After Campaign 1

The next Apple GPU campaign should be opened only after Campaign 1 evidence
identifies a specific bottleneck. Candidate paths are:

```text
offline .metallib compilation versus runtime NSString source compilation
one-word and two-word specialized commutation kernels
two-dimensional grid dispatch to avoid entry-to-row division in large square matrices
private storage plus blit staging for larger matrices
workspace or heap reuse for repeated commutation calls
true Metal reduction kernels for compact consumers
shader-counter and shader-timeline profiling through Instruments when the template exposes them
MPSGraph and PyTorch MPS external baselines when semantic mappings are exact
```

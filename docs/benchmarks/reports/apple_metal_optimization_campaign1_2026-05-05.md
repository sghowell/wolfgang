# Apple Metal Campaign 1 Optimization And Evidence Report

Date: 2026-05-05

## Summary

Apple Metal Campaign 1 closes the immediate evidence drift after Metal bring-up:
the architecture now records that Metal System Trace evidence exists, the Metal
benchmark has a scaling profile beyond the original smoke case, and the broad
README performance landscape includes Apple Metal rows alongside CPU, CUDA,
ROCm/HIP, and external baseline evidence.

This campaign does not add Metal wheels, raw Metal buffer exports, public async
or command queue APIs, MPSGraph kernels, PyTorch MPS implementation paths, or
new Metal operations. The public behavior remains the source-build-only
`backend="metal"` pairwise commutation lane.

## Commands

```bash
uv pip install -e '.[test]' \
  --config-settings=cmake.define.FASTPAULI_ENABLE_CUDA=OFF \
  --config-settings=cmake.define.FASTPAULI_ENABLE_HIP=OFF \
  --config-settings=cmake.define.FASTPAULI_ENABLE_METAL=ON

env FASTPAULI_VALIDATE_METAL=1 .venv/bin/python benchmarks/bench_metal_kernels.py \
  --profile scaling \
  --repeat 3 \
  --json \
  --output docs/benchmarks/data/apple_metal_optimization_campaign1_2026-05-05/raw/metal_benchmark_scaling.json

.venv/bin/python scripts/render_apple_metal_assets.py \
  --data-dir docs/benchmarks/data/apple_metal_optimization_campaign1_2026-05-05 \
  --plot-dir docs/benchmarks/plots

.venv/bin/python scripts/render_apple_metal_assets.py \
  --data-dir docs/benchmarks/data/apple_metal_optimization_campaign1_2026-05-05 \
  --plot-dir docs/benchmarks/plots \
  --check-only
```

The scaling benchmark was run outside the non-elevated command sandbox so the
Apple GPU was visible to `MTLCreateSystemDefaultDevice()`, matching the
environment requirement recorded in the bring-up report.

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

All rows keep correctness checks enabled. Timings are median seconds over three
timed repetitions. CPU default and NEON rows are same-host Apple Silicon CPU
baselines, not cross-machine claims. The three-word case records NEON as
unavailable because the current NEON commutation kernel supports one- and
two-word packed Pauli inputs. Metal kernel rows record the
`fp_pairwise_commutation` dispatch metadata: `dispatchThreads`, grid shape
`[matrix_entries, 1, 1]`, threadgroup size `[256, 1, 1]`, synchronous
`commit_and_waitUntilCompleted_per_operation`, shared storage, and the explicit
allocation or reuse boundary. Non-kernel Metal rows record why threadgroup and
grid metadata are not applicable.

| Case | Qubits | Terms | Words | Entries | CPU default | CPU scalar | CPU NEON | Metal transfer | Metal resident host output | Metal matrix allocate | Metal matrix reuse | Metal to_host | Metal compact count |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| one_word_256x256 | 32 | 256x256 | 1 | 65,536 | 5.77091e-05 | 1.35458e-04 | 4.22089e-05 | 1.01879e-03 | 3.49042e-04 | 6.45000e-04 | 1.72417e-04 | 9.33406e-06 | 8.25000e-06 |
| rectangular_128x1024 | 48 | 128x1024 | 1 | 131,072 | 9.40830e-05 | 2.45625e-04 | 7.85840e-05 | 8.50041e-04 | 1.71084e-04 | 7.12625e-04 | 1.30583e-04 | 1.26670e-05 | 1.64580e-05 |
| multiword_192x192 | 130 | 192x192 | 3 | 36,864 | 1.83666e-04 | 2.06459e-04 | n/a | 6.95166e-04 | 1.99167e-04 | 3.59667e-04 | 1.25542e-04 | 9.99996e-06 | 4.70900e-06 |
| square_512x512 | 64 | 512x512 | 1 | 262,144 | 1.82625e-04 | 4.94416e-04 | 1.91458e-04 | 7.24042e-04 | 3.63833e-04 | 7.27208e-04 | 1.43250e-04 | 1.60000e-05 | 3.27501e-05 |

![FastPauli accelerator performance landscape](../plots/accelerator_landscape_with_rocm.svg)

## Interpretation

The current Metal implementation is correct but still launch/allocation-bound
for host-output pairwise commutation. Transfer-inclusive Metal rows remain
slower than same-host CPU rows on these sizes because each call copies operands
to Metal buffers and synchronously materializes host output.

The retained device-output path is the more relevant accelerator boundary. The
`commutes_with_device(..., output=existing_matrix)` row removes repeated output
allocation and becomes competitive with the Apple CPU NEON path on the largest
one-word case while preserving the public API and correctness checks. This is
evidence for a later workspace or reusable-output optimization campaign, not a
new public workspace contract.

Compact consumer timings are fast because `DeviceCommutationMatrix` uses shared
Apple Silicon memory and the current compact consumers scan that storage on the
CPU. These rows are useful as public boundary evidence, but they are not proof
that Metal reduction kernels would be faster. A true Metal reduction requires a
separate A/B implementation and profiler evidence.

## Profiler Status

The Metal profiler blocker from the first bring-up is closed: full Xcode,
`xctrace`, and the downloadable Metal Toolchain component are installed, and
the checked bring-up evidence includes a sanitized Metal System Trace summary
with FastPauli python process presence and Metal/GPU schemas.

Campaign 1 reuses that profiler evidence and focuses on benchmark scaling and
plot integration. Deeper Apple GPU profiling remains open because the default
Metal System Trace capture has GPU counter profile and shader timeline disabled.
The next profiling campaign should use Instruments settings that expose shader
counter and shader-timeline data or record the exact template limitation.

## Checked Evidence

```text
Raw benchmark JSON: docs/benchmarks/data/apple_metal_optimization_campaign1_2026-05-05/raw/metal_benchmark_scaling.json
Summary JSON: docs/benchmarks/data/apple_metal_optimization_campaign1_2026-05-05/summary.json
README landscape plot: docs/benchmarks/plots/accelerator_landscape_with_rocm.svg
Campaign plan: docs/plans/apple_metal_optimization_campaign1_plan.md
Renderer: scripts/render_apple_metal_assets.py
Profiler evidence retained from: docs/benchmarks/data/apple_metal_bringup_2026-05-01/profiler/metal_system_trace_summary.json
```

## Remaining Headroom

1. Capture shader-counter and shader-timeline profiler evidence with a tuned
   Instruments template.
2. A/B test offline `.metallib` compilation against runtime NSString source
   compilation.
3. A/B test one-word and two-word specialized kernels against the generic
   loop-over-packed-words kernel.
4. A/B test two-dimensional dispatch that avoids per-entry division by
   `rhs_terms` in large square matrices.
5. A/B test private storage plus blit staging for larger dense matrices.
6. Add true Metal reduction kernels for compact count and conflict consumers
   only if profiler and A/B evidence beat the current shared-memory CPU scan.
7. Add MPSGraph and PyTorch MPS external baselines only where the semantic
   mapping and timing boundary are exact.

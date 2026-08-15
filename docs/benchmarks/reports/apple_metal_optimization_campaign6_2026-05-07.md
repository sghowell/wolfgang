# Apple Metal Optimization Campaign 6 Report

Date: 2026-05-07

Apple Metal Campaign 6 status: retained private workspace groundwork for
future device-resident simplify, with the public Metal simplify behavior still
on the Campaign 5 transfer-reference correctness bridge.

Campaign 6 adds a private `MetalWorkspace` model, `WorkspaceTimingMode`, and
Campaign 6 simplify benchmark rows. It does not retain a device-resident
simplify implementation. The device-resident simplify candidate remains blocked
until checked Metal sort/prefix/reduce primitives exist.

## Commands

```bash
uv pip install -e '.[test]' \
  --config-settings=cmake.define.FASTPAULI_ENABLE_CUDA=OFF \
  --config-settings=cmake.define.FASTPAULI_ENABLE_HIP=OFF \
  --config-settings=cmake.define.FASTPAULI_ENABLE_METAL=ON
env FASTPAULI_VALIDATE_METAL=1 .venv/bin/python benchmarks/bench_metal_kernels.py \
  --profile campaign6 --repeat 1 --json
env FASTPAULI_VALIDATE_METAL=1 .venv/bin/python benchmarks/bench_metal_kernels.py \
  --profile campaign6 --repeat 10 --json \
  --output docs/benchmarks/data/apple_metal_optimization_campaign6_2026-05-07/raw/metal_benchmark_campaign6.json
.venv/bin/python scripts/render_apple_metal_assets.py \
  --data-dir docs/benchmarks/data/apple_metal_optimization_campaign6_2026-05-07 \
  --plot-dir docs/benchmarks/plots
```

## Environment

```text
Host: Apple M4 Pro, Mac mini Mac16,11
OS: macOS-26.2-arm64-arm-64bit
Compiler: Apple clang version 21.0.0 (clang-2100.0.123.102)
Objective-C++ compiler: AppleClang 21.0.0.21000099
Python: 3.12.12
NumPy: 2.4.4
FastPauli build: metal_only
Metal device: Apple M4 Pro
Metal storage: MTLResourceStorageModeShared
Compiled CPU backends: scalar, neon
Active CPU backend: scalar
```

The benchmark JSON records git provenance as `ef26d41+dirty` because evidence
was captured before committing the Campaign 6 implementation and documentation
slice.

## Workspace Groundwork

Campaign 6 replaces the previous inert workspace reservation with a private
`MetalWorkspace` class in `src/metal/workspace_metal.hpp` and
`src/metal/workspace_metal.mm`. The model records:

```text
reserved bytes
high-watermark bytes
allocation count
growth count
workspace timing mode
device ordinal
```

`FASTPAULI_METAL_BENCH_WORKSPACE_TIMING` accepts `absent`,
`grow_inside_timing`, and `pre_reserved_outside_timing` for benchmark
vocabulary. `FASTPAULI_EXPERIMENTAL_METAL_SIMPLIFY_STRATEGY` is reserved as
the benchmark selector vocabulary for future simplify candidates. No Campaign 6
selector retains a device-resident simplify implementation.

The `metal_simplify_workspace_probe` rows use `transfer_boundary: status_only`,
`metal_simplify_strategy: device_candidate`, and
`metal_simplify_strategy_status: rejected_with_evidence`. They record the
private MetalWorkspace scratch model and explicitly mark the device-resident
simplify candidate as not executed.

## Results

Median seconds from
`docs/benchmarks/data/apple_metal_optimization_campaign6_2026-05-07/raw/metal_benchmark_campaign6.json`:

| Case | Output terms | CPU default median | CPU scalar median | CPU NEON | Metal transfer reference median | Workspace probe |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| words1 duplicate-heavy, 8192 terms | 1229 | 0.000401 | 0.000372 | unavailable for scalar-only simplify | 0.000674 | retained_private_model, 655360 bytes |
| words1 duplicate-light, 8192 terms | 7782 | 0.000374 | 0.000416 | unavailable for scalar-only simplify | 0.000759 | retained_private_model, 655360 bytes |
| words2 duplicate-heavy, 4096 terms | 1229 | 0.000171 | 0.000194 | unavailable for scalar-only simplify | 0.000491 | retained_private_model, 458752 bytes |
| generic multi-word, 2048 terms | 1024 | 0.0000605 | 0.0000551 | unavailable for scalar-only simplify | 0.000367 | retained_private_model, 294912 bytes |
| cancellation, 4096 terms | 0 | 0.000161 | 0.000157 | unavailable for scalar-only simplify | 0.000444 | retained_private_model, 327680 bytes |

The transfer-reference path remains slower than same-host CPU on every
Campaign 6 case. The workspace probe is not timed and is not plotted as a
speedup; it exists to lock down the private scratch shape and metadata contract
for a future device-resident simplify prototype.

## README Landscape

The broad README landscape was regenerated from Campaign 6 summary data:

```text
docs/benchmarks/data/apple_metal_optimization_campaign6_2026-05-07/summary.json
docs/benchmarks/plots/accelerator_landscape_with_rocm.svg
```

The plot remains an across-the-board view that includes CPU, CUDA, ROCm/HIP,
CuPy, and Apple Metal source-build rows. Campaign 6 adds no new timed
device-resident simplify series because no retained Metal duplicate-reduction
kernel exists yet.

## Decision

Retain the private MetalWorkspace groundwork and Campaign 6 benchmark status
rows. Keep public Metal `DevicePauliSum.simplify()` on the Campaign 5
transfer-reference behavior. Do not promote `metal_simplify_workspace_probe` as
a device-resident GPU result and do not expose public Metal workspaces or raw
buffers in this slice.

## Remaining Headroom

Evidence-backed next work:

```text
prototype a benchmark-only Metal key-sort primitive for packed one-word simplify inputs
prototype a benchmark-only prefix-sum primitive for survivor compaction
prototype a benchmark-only reduce-by-key primitive for duplicate coefficient summation
measure any candidate against CPU default, CPU scalar, and retained transfer reference on Campaign 5 and Campaign 6 cases
only retain a device_resident simplify row when it produces correct canonical output and beats the transfer-reference boundary
keep forced NEON simplify rows skipped until simplify has a real optimized CPU path
capture shader-counter evidence only after a retained simplify kernel exists
validate on additional Apple Silicon generations before changing support or selector policy
```

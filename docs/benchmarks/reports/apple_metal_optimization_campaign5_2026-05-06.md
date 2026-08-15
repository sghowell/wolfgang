# Apple Metal Optimization Campaign 5 Report

Date: 2026-05-06

Apple Metal Campaign 5 status: retained source-build simplify correctness
bridge with explicit transfer-reference timing.

Campaign 5 adds the first retained Metal `DevicePauliSum.simplify(atol, rtol)`
lane for source builds. The implementation preserves the Metal object identity
on return, but it is intentionally transfer-assisted:

```text
Metal DevicePauliSum -> host PauliSum -> CPU PauliSum.simplify() -> Metal DevicePauliSum
```

The retained Metal simplify implementation is a correctness bridge, not a device-resident GPU duplicate-reduction path.

## Commands

```bash
uv pip install -e '.[test]' \
  --config-settings=cmake.define.FASTPAULI_ENABLE_CUDA=OFF \
  --config-settings=cmake.define.FASTPAULI_ENABLE_HIP=OFF \
  --config-settings=cmake.define.FASTPAULI_ENABLE_METAL=ON
FASTPAULI_VALIDATE_METAL=1 .venv/bin/python -m pytest \
  tests/test_apple_metal_campaign5.py tests/test_apple_metal_foundation.py -q
FASTPAULI_VALIDATE_METAL=1 .venv/bin/python benchmarks/bench_metal_kernels.py \
  --profile campaign5 --repeat 1 --json
FASTPAULI_VALIDATE_METAL=1 .venv/bin/python benchmarks/bench_metal_kernels.py \
  --profile campaign5 --repeat 10 --json \
  --output docs/benchmarks/data/apple_metal_optimization_campaign5_2026-05-06/raw/metal_benchmark_campaign5.json
.venv/bin/python scripts/render_apple_metal_assets.py \
  --data-dir docs/benchmarks/data/apple_metal_optimization_campaign5_2026-05-06 \
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

The benchmark JSON records git provenance as `c23f5f1+dirty` because evidence
was captured before committing the implementation and documentation slice.

## Semantic Coverage

`tests/test_apple_metal_campaign5.py` checks:

```text
empty operators preserve num_qubits
single-term simplify parity
duplicate-heavy simplify parity
cancellation to empty output
two-word and generic multi-word packed keys
inclusive atol/rtol threshold behavior
returned DevicePauliSum backend="metal"
negative, NaN, and infinite tolerance rejection
```

The targeted Metal validation after implementation was:

```text
24 passed, 1 skipped
```

The skip is the CPU-only Metal-absence test in `tests/test_apple_metal_foundation.py`,
which is expected while the Metal source build is active.

## Timing Boundary

Campaign 5 introduces `metal_simplify_transfer_reference` with transfer
boundary `device_to_host_cpu_simplify_host_to_device`. The timed operation is
`device_op.simplify()` only; its implementation performs host materialization,
CPU simplify, and construction of a new Metal `DevicePauliSum`. The benchmark
does not include the post-timing correctness `to_host()` check.

Forced `FASTPAULI_CPU_BACKEND=neon` rows are present but skipped because
FastPauli simplify is a scalar-only CPU operation today. Labeling them as NEON
timings would overstate the CPU path, so the benchmark records the skip reason
instead.

## Results

Median/min/max seconds from
`docs/benchmarks/data/apple_metal_optimization_campaign5_2026-05-06/raw/metal_benchmark_campaign5.json`:

| Case | Output terms | CPU default median | CPU scalar median | CPU NEON | Metal transfer reference median |
| --- | ---: | ---: | ---: | --- | ---: |
| words1 duplicate-heavy, 8192 terms | 1229 | 0.000346 | 0.000391 | unavailable for scalar-only simplify | 0.000883 |
| words1 duplicate-light, 8192 terms | 7782 | 0.000437 | 0.000381 | unavailable for scalar-only simplify | 0.000925 |
| words2 duplicate-heavy, 4096 terms | 1229 | 0.000162 | 0.000147 | unavailable for scalar-only simplify | 0.000621 |
| generic multi-word, 2048 terms | 1024 | 0.0000578 | 0.0000503 | unavailable for scalar-only simplify | 0.000402 |
| cancellation, 4096 terms | 0 | 0.000165 | 0.000162 | unavailable for scalar-only simplify | 0.000452 |

The transfer-reference path is slower than same-host CPU on every retained
Campaign 5 case. That is expected for a correctness bridge because it performs
two host/device object materialization steps around CPU simplify.

## README Landscape

The README broad performance landscape was regenerated from Campaign 5 summary
data:

```text
docs/benchmarks/data/apple_metal_optimization_campaign5_2026-05-06/summary.json
docs/benchmarks/plots/accelerator_landscape_with_rocm.svg
```

The plot now includes the `Apple Metal simplify transfer reference` series
alongside existing CPU, CUDA, ROCm/HIP, CuPy, and Apple Metal commutation and
compact-consumer rows. This keeps the README view across the full evidence
landscape rather than narrowing it to the latest Metal-only row set.

## Decision

Retain the transfer-reference implementation as the correct Metal simplify API
behavior for source builds. Do not promote it as a performance path. Do not add
public Metal queues, buffers, streams, workspaces, DLPack, MPSGraph, PyTorch MPS,
Metal statevector expectation, Metal matmul, Metal wheels, PyPI publication,
Windows support, or older macOS support in this slice.

## Remaining Headroom

Evidence-backed next work:

```text
design a private Metal duplicate-reduction scratch/workspace model before any device-resident simplify candidate
prototype device-resident Metal sort or reduce-by-key only behind benchmark-only selectors
measure candidate kernels against CPU default, CPU scalar, and the retained transfer reference on the Campaign 5 cases
keep forced NEON simplify rows skipped until simplify has a real optimized CPU path
add Metal statevector expectation only after simplify remains stable under source-build validation
add Metal matmul only after simplify and expectation have retained correctness and evidence boundaries
capture sanitized shader-counter exports for simplify candidates only when Instruments can emit narrow value CSVs without raw trace retention
validate on additional Apple Silicon generations before changing selector or support policies
```

# Apple Metal Optimization Campaign 7 Report

Date: 2026-05-07

Apple Metal Campaign 7 status: retained a checked device-resident simplify primitive stack
for one-word packed Pauli inputs with
signed fixed32 dyadic coefficients whose accumulated sums and tolerance
threshold fit exact uint64 squared-magnitude comparison. Public Metal
`DevicePauliSum.simplify()` remains on the Campaign 5 transfer-reference
correctness bridge.

Campaign 7 adds a private `metal_simplify_device_candidate` benchmark row
backed by Metal sort, prefix-sum, reduce-by-key, and survivor-compaction
primitives. The candidate is intentionally not promoted to public simplify:
Apple Metal rejected native `double` arithmetic in this kernel direction, so
the retained candidate is limited to exactly representable signed fixed32
dyadic coefficients inside that checked integer comparison domain.

## Commands

```bash
env FASTPAULI_VALIDATE_METAL=1 uv pip install -e '.[test]' \
  --config-settings=cmake.define.FASTPAULI_ENABLE_CUDA=OFF \
  --config-settings=cmake.define.FASTPAULI_ENABLE_HIP=OFF \
  --config-settings=cmake.define.FASTPAULI_ENABLE_METAL=ON
xcrun -sdk macosx metal -c src/metal/kernels/simplify.metal \
  -o /tmp/fastpauli_simplify.air
env FASTPAULI_VALIDATE_METAL=1 .venv/bin/python -m pytest \
  tests/test_apple_metal_campaign7.py -q
env FASTPAULI_VALIDATE_METAL=1 .venv/bin/python benchmarks/bench_metal_kernels.py \
  --profile campaign7 --repeat 1 --json
env FASTPAULI_VALIDATE_METAL=1 .venv/bin/python benchmarks/bench_metal_kernels.py \
  --profile campaign7 --repeat 10 --json \
  --output docs/benchmarks/data/apple_metal_optimization_campaign7_2026-05-07/raw/metal_benchmark_campaign7.json
.venv/bin/python scripts/render_apple_metal_assets.py \
  --data-dir docs/benchmarks/data/apple_metal_optimization_campaign7_2026-05-07 \
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

The benchmark JSON records git provenance as `b554056+dirty` because evidence
was captured before committing the Campaign 7 implementation and documentation
slice.

## Primitive Stack

The private candidate uses one command buffer per simplify call with the
following kernel stack:

```text
fp_simplify_words1_init_keys
fp_simplify_words1_bitonic_sort_step
fp_simplify_words1_mark_heads
fp_simplify_prefix_sum_step
fp_simplify_words1_reduce_by_key
fp_simplify_words1_compact_survivors
```

The stack sorts one-word `(x, z)` packed keys, marks duplicate-run heads,
prefix-scans the head and survivor flags, sums duplicate coefficients in the
head lane, and compacts nonzero survivors into a new Metal `DevicePauliSum`.
Candidate rows use:

```text
variant: metal_simplify_device_candidate
operation: simplify
transfer_boundary: device_resident
metal_simplify_strategy: device_candidate
metal_simplify_strategy_status: benchmark_only
metal_simplify_coefficient_domain: signed_fixed32_dyadic_coefficients_only
```

The coefficient restriction is not cosmetic. Metal Shading Language does not
provide the FP64 arithmetic needed for the public complex-double coefficient
domain on this host. Campaign 7 therefore converts only exactly representable
fixed-dyadic coefficient bits into a signed fixed32 pair for device reduction,
then writes exact FP64 bits back for checked benchmark materialization. The
private hook reports non-dyadic coefficients as `rejected_with_evidence`, and
the host side rejects cases whose worst-case duplicate sum could overflow the
signed fixed32 accumulator or whose nonzero tolerance comparison cannot be kept
inside exact uint64 squared-magnitude arithmetic. Survivor filtering compares
the complex magnitude contract, not independent real and imaginary components.

## Results

Median seconds from
`docs/benchmarks/data/apple_metal_optimization_campaign7_2026-05-07/raw/metal_benchmark_campaign7.json`:

| Case | Output terms | CPU default median | CPU scalar median | CPU NEON | Metal transfer reference median | Device candidate median | Candidate status |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| words1 duplicate-heavy, 8192 terms | 1222 | 0.000424 | 0.000435 | unavailable for scalar-only simplify | 0.000996 | 0.000492 | ok, fixed-dyadic, 91 bitonic passes, 26 prefix passes |
| words1 duplicate-light, 8192 terms | 7740 | 0.000449 | 0.000408 | unavailable for scalar-only simplify | 0.000663 | 0.000488 | ok, fixed-dyadic, 91 bitonic passes, 26 prefix passes |
| words1 cancellation, 4096 terms | 0 | 0.000154 | 0.000169 | unavailable for scalar-only simplify | 0.000418 | 0.000428 | ok, fixed-dyadic, 78 bitonic passes, 24 prefix passes |
| words2 duplicate-heavy, 4096 terms | 1222 | 0.000151 | 0.000183 | unavailable for scalar-only simplify | 0.000391 | unavailable | one-word candidate rejects multi-word input |

The device candidate beats the retained transfer-reference path on the two
duplicate rows and is effectively tied with the transfer-reference path on the
cancellation row after the additional checked-domain scan and exact
complex-magnitude tolerance fix. It still does not beat the same-host CPU
simplify rows. The large bitonic pass count and the host-side fixed32 safety
scan are the main known cost centers. The result is useful as checked
primitive-stack evidence, not as a public performance win.

## README Landscape

The broad README landscape was regenerated from Campaign 7 summary data:

```text
docs/benchmarks/data/apple_metal_optimization_campaign7_2026-05-07/summary.json
docs/benchmarks/plots/accelerator_landscape_with_rocm.svg
```

The plot remains an across-the-board view that includes CPU, CUDA, ROCm/HIP,
CuPy, and Apple Metal source-build rows. Campaign 7 adds the
`Apple Metal simplify device candidate` series so the checked primitive stack is
visible next to CPU and transfer-reference simplify evidence.

## Decision

Retain the private checked primitive stack and benchmark rows. Keep public
Metal `DevicePauliSum.simplify()` on the Campaign 5 transfer-reference bridge.
Do not expose the private hook as public API and do not describe Campaign 7 as
a general FP64 Metal simplify implementation.

## Remaining Headroom

Evidence-backed next work:

```text
replace the O(log^2 n) bitonic stack with a lower-pass radix or bucketed key sort if a Metal-friendly design can be kept deterministic
test whether fixed-dyadic coefficient handling can be widened without pretending to support arbitrary FP64 arithmetic in Metal kernels
separate command-buffer and scratch-allocation overhead from kernel time in the candidate row
explore persistent pipeline/library caching so runtime source compilation is not on the hot path for repeated simplify calls
add shader-counter or xctrace evidence once a lower-pass candidate exists
validate on additional Apple Silicon generations before changing support or selector policy
keep CPU simplify optimization on the roadmap because every Campaign 7 checked device row remains slower than same-host CPU simplify
```

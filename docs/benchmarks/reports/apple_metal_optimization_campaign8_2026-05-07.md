# Apple Metal Optimization Campaign 8 Report

Apple Metal Campaign 8 status: the checked one-word device-resident simplify
candidate remains private and experimental.

Campaign 8 makes the Campaign 7 candidate measurable enough to answer the next
question: can it become performance-relevant, or should it stay experimental?
The answer from the local Apple M4 Pro run is that the candidate is not ready
for public promotion. It is correct on the accepted one-word fixed-dyadic
domain and the benchmark now records useful timing decomposition, but the
candidate does not beat same-host CPU simplify on the checked workloads.

Public `DevicePauliSum.simplify(atol, rtol)` remains the Campaign 5
transfer-reference bridge:

```text
Metal DevicePauliSum -> host PauliSum -> CPU simplify -> Metal DevicePauliSum
```

## Commands

```bash
env FASTPAULI_VALIDATE_METAL=1 uv pip install -e '.[test]' \
  --config-settings=cmake.define.FASTPAULI_ENABLE_CUDA=OFF \
  --config-settings=cmake.define.FASTPAULI_ENABLE_HIP=OFF \
  --config-settings=cmake.define.FASTPAULI_ENABLE_METAL=ON

env FASTPAULI_VALIDATE_METAL=1 .venv/bin/python benchmarks/bench_metal_kernels.py \
  --profile campaign8 \
  --repeat 10 \
  --json \
  --output docs/benchmarks/data/apple_metal_optimization_campaign8_2026-05-07/raw/metal_benchmark_campaign8.json

.venv/bin/python scripts/render_apple_metal_assets.py \
  --data-dir docs/benchmarks/data/apple_metal_optimization_campaign8_2026-05-07 \
  --plot-dir docs/benchmarks/plots
```

## Environment

```text
Host: Apple M4 Pro, Mac mini Mac16,11
OS: macOS 26.2 build 25C56
Compiler: AppleClang 21.0.0.21000099
Python: 3.12.12
NumPy: 2.4.4
FastPauli build mode: metal_only
Compiled backends: cpu, metal
Metal storage mode: MTLResourceStorageModeShared
Metal device: Apple M4 Pro
Metal capability summary: unified_memory=true; low_power=false; headless=false; removable=false; recommended_max_working_set_size_bytes=40200896512
```

The checked benchmark JSON records `git_commit: 4c6e991+dirty` because the
evidence was generated from this in-progress Campaign 8 branch before commit.

## What Changed

Campaign 8 adds benchmark-only metadata to the private
`metal_simplify_device_candidate` path:

```text
timing_decomposition_seconds.host_preflight
timing_decomposition_seconds.scratch_and_output_allocation
timing_decomposition_seconds.command_encoding
timing_decomposition_seconds.command_execution
timing_decomposition_seconds.output_accounting
timing_decomposition_seconds.total_observed
dispatch_counts.total_kernel_dispatches
pipeline_cache.boundary
pipeline_cache.library_source
performance_decision.candidate_status
```

The pipeline boundary is now explicit: Campaign 8 prewarms the static private
pipeline cache before the measured operation. This keeps runtime source
compilation out of the diagnostic kernel path and makes the remaining overhead
visible: scratch/output allocation, command encoding, command execution, and
output accounting.

## Benchmark Results

Median seconds, repeat 10:

| Case | CPU default | Metal transfer-reference | Metal device candidate | Candidate decision |
| --- | ---: | ---: | ---: | --- |
| words1 cancellation, 4096 terms | 0.000135 | 0.000390 | 0.000533 | experimental |
| words1 duplicate-heavy, 8192 terms | 0.000313 | 0.000672 | 0.000773 | experimental |
| words1 duplicate-light, 8192 terms | 0.000366 | 0.000679 | 0.000908 | experimental |
| words1 duplicate-heavy, 16384 terms | 0.000852 | 0.001193 | 0.001113 | experimental |
| words2 status-only, 4096 terms | 0.000148 | 0.000553 | unavailable | unavailable |

The checked device candidate is slower than same-host CPU default on every
timed one-word case. It is also slower than the transfer-reference bridge on
the 4096-term cancellation and 8192-term cases. On the 16384-term
duplicate-heavy case, it is slightly faster than transfer-reference but still
slower than CPU simplify, so it does not meet the Campaign 8 promotion rule.

## Timing Decomposition

Representative ok device-candidate rows:

| Case | Dispatches | Command execution | Scratch/output allocation | Internal observed total |
| --- | ---: | ---: | ---: | ---: |
| cancellation, 4096 terms | 108 | 0.000519 | 0.0000568 | 0.000606 |
| duplicate-heavy, 8192 terms | 123 | 0.000714 | 0.0000654 | 0.000822 |
| duplicate-light, 8192 terms | 123 | 0.000784 | 0.0000560 | 0.000882 |
| duplicate-heavy, 16384 terms | 139 | 0.000794 | 0.0000628 | 0.000913 |

The dominant cost is command execution, not runtime source compilation. The
large dispatch count comes from the current bitonic sort plus two Hillis-Steele
prefix sums:

```text
4096 padded terms: 78 bitonic passes + 24 prefix passes + fixed kernels = 108 dispatches
8192 padded terms: 91 bitonic passes + 26 prefix passes + fixed kernels = 123 dispatches
16384 padded terms: 105 bitonic passes + 28 prefix passes + fixed kernels = 139 dispatches
```

This is the clearest remaining bottleneck. A lower-pass deterministic sort, or
a different duplicate-grouping design, is required before this path can be a
credible public simplify implementation.

## Correctness Boundary

The accepted domain is unchanged from Campaign 7:

```text
exactly one packed word
coefficients exactly representable as signed fixed32 dyadic values
fixed32 accumulated sums fit signed int64
nonzero tolerance checks fit exact uint64 squared-magnitude comparison
candidate output materializes back to CPU and matches PauliSum.simplify()
```

Multi-word input remains status-only unavailable. Non-fixed-dyadic,
accumulator-overflow, and unsupported tolerance domains remain rejected with
evidence. Campaign 8 does not broaden those semantics.

## Decision

Campaign 8 keeps the device-resident simplify candidate experimental.

The candidate is correct and now has useful diagnostics, but it does not beat
CPU simplify on the checked local Apple M4 Pro workloads. Public promotion
would require at least one of:

```text
a lower-pass deterministic Metal radix or bucketed key grouping design
a reusable scratch/output lifetime boundary that avoids per-call allocation
additional CPU simplify optimization to keep the baseline honest
additional Apple Silicon generation evidence before any selector policy change
```

The next Apple Metal-specific optimization should start with a lower-pass
deterministic sort or grouping prototype. If that does not beat CPU simplify,
the next practical performance work should pivot back to CPU simplify.

## Artifacts

```text
Raw JSON: docs/benchmarks/data/apple_metal_optimization_campaign8_2026-05-07/raw/metal_benchmark_campaign8.json
Summary JSON: docs/benchmarks/data/apple_metal_optimization_campaign8_2026-05-07/summary.json
Broad landscape plot: docs/benchmarks/plots/accelerator_landscape_with_rocm.svg
Plan: docs/plans/apple_metal_optimization_campaign8_plan.md
```

# Apple Metal Wave 1D Report

Wave 1D is the local Apple-host prestage for retained commutation reused-output
on the Metal path. It does not broaden the public API. It adds a benchmark
profile and evidence gate that compares retained reused-output against both the
allocating device-output boundary and the transfer-inclusive boundary, using the
required mean-of-medians methodology across 3 independent reruns.

## Commands

```bash
env FASTPAULI_VALIDATE_METAL=1 uv pip install -e '.[test]' \
  --config-settings=cmake.define.WOLFGANG_ENABLE_INTERNAL_BINDINGS=ON \
  --config-settings=cmake.define.WOLFGANG_ENABLE_METAL=ON \
  --config-settings=cmake.define.WOLFGANG_ENABLE_CUDA=OFF \
  --config-settings=cmake.define.WOLFGANG_ENABLE_HIP=OFF \
  --config-settings=cmake.define.WOLFGANG_ENABLE_NATIVE=OFF

env FASTPAULI_VALIDATE_METAL=1 .venv/bin/python benchmarks/bench_metal_kernels.py \
  --profile wave1d \
  --repeat 3 \
  --reruns 3 \
  --json \
  --output docs/benchmarks/data/apple_metal_wave1d_2026-08-21/raw/metal_benchmark_wave1d.json
```

## Environment

```text
Host: Apple M4 Pro, Mac mini Mac16,11
OS: macOS 26.2 build 25C56
Compiler: AppleClang 21.0.0.21000099
Python: 3.13.11
NumPy: 2.5.2
Wolfgang build mode: metal_only
Compiled backends: cpu, metal
Metal device: Apple M4 Pro
Metal storage mode: MTLResourceStorageModeShared
Runtime-visible backends: cpu, metal
```

The raw benchmark JSON records `git_commit: ee4f95f+dirty` because the evidence
was captured from the in-progress Wave 1D branch before commit.

## Methodology

Wave 1D uses these exact timing boundaries:

```text
device_output_reused
device_output_allocating
transfer_inclusive
```

Promotion metric:

```text
3 independent reruns per case
3 timed repetitions per rerun
mean-of-medians across reruns
small-row guard rejects any undocumented >5% reused-output regression versus device_output_allocating
```

CPU-vs-Metal correctness checks remained enabled throughout the benchmark run.
No equality checks failed, and the checked benchmark JSON reports `wave1d_evidence.status: go` with an empty `small_row_regressions` list.

## Mean-of-medians results

| Case | CPU default | CPU NEON | Metal transfer-inclusive | Metal allocating device output | Metal retained reused-output | Reused / allocating | Reused / transfer-inclusive | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| small rows 128x128 | 0.0000321 s | 0.0000249 s | 0.0035291 s | 0.0005911 s | 0.0002614 s | 0.442x | 0.074x | go |
| medium rows 512x512 | 0.0002492 s | 0.0002077 s | 0.0010073 s | 0.0005849 s | 0.0001719 s | 0.294x | 0.171x | go |
| large rows 2048x2048 | 0.0024873 s | 0.0021972 s | 0.0013492 s | 0.0008060 s | 0.0004013 s | 0.498x | 0.297x | go |

Interpretation:

```text
small-row regression guard passed: reused-output was faster than allocating, not slower
medium retained row improved by about 70.6% versus allocating device output
large retained row improved by about 50.2% versus allocating device output
transfer-inclusive timing remains dominated by host/device bridge cost on the small row
retained reused-output remains meaningfully faster than transfer-inclusive on every checked row
```

## Decision

Wave 1D is GO on the local Apple host.

What this proves:

```text
retained reused-output evidence is now labeled explicitly against both allocating and transfer-inclusive boundaries
three-rerun mean-of-medians methodology is implemented and recorded in the raw JSON
small-row regression guard did not trigger
CPU-vs-Metal equality checks were preserved for the measured commutation rows
no public API or semantic expansion was required
```

What this does not claim:

```text
no public Metal workspace, stream, heap, DLPack, or raw-buffer API
no wheel support claim
no non-Apple portability claim
no claim that transfer-inclusive host-facing small rows are now competitive with CPU
```

## Artifacts

```text
Raw JSON: docs/benchmarks/data/apple_metal_wave1d_2026-08-21/raw/metal_benchmark_wave1d.json
Report: docs/benchmarks/reports/apple_metal_wave1d_2026-08-21.md
Plan: docs/plans/wolfgang-kernel-performance-campaign.md
Architecture contract: docs/architecture/apple_accelerator.md
Benchmark protocol: docs/benchmarks/protocol.md
```

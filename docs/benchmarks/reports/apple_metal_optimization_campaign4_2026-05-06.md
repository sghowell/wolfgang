# Apple Metal Campaign 4 Optimization And Evidence Report

Date: 2026-05-06

## Summary

Apple Metal Campaign 4 keeps the existing source-build-only `backend="metal"`
API boundary and focuses on the remaining Campaign 3 headroom that can be
tested on the local Apple M4 Pro:

```text
larger two-word selector evidence
larger compact-consumer matrices
parallel block-reduction compact total count through FASTPAULI_EXPERIMENTAL_METAL_COMPACT_CONSUMER=gpu_parallel_total
private device-boundary evidence for future fused device-only workflows
profiler export status without retaining raw trace bundles
```

PyPI publication, Windows support, and older macOS compatibility are out of scope
for this Apple Metal optimization slice.

## Commands

```bash
env FASTPAULI_VALIDATE_METAL=1 uv pip install -e '.[test]' \
  --config-settings=cmake.define.FASTPAULI_ENABLE_CUDA=OFF \
  --config-settings=cmake.define.FASTPAULI_ENABLE_HIP=OFF \
  --config-settings=cmake.define.FASTPAULI_ENABLE_METAL=ON

xcrun xctrace list templates

env FASTPAULI_VALIDATE_METAL=1 .venv/bin/python benchmarks/bench_metal_kernels.py \
  --profile campaign4 \
  --repeat 10 \
  --json \
  --output docs/benchmarks/data/apple_metal_optimization_campaign4_2026-05-06/raw/metal_benchmark_campaign4.json

.venv/bin/python scripts/render_apple_metal_assets.py \
  --data-dir docs/benchmarks/data/apple_metal_optimization_campaign4_2026-05-06 \
  --plot-dir docs/benchmarks/plots
```

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
oneTBB, AVX2, AVX-512, SVE, CUDA, and HIP: not compiled on this host
```

## Provenance

The checked benchmark JSON records `git_commit` with a `+dirty` suffix and a
`git_provenance` block because Campaign 4 evidence was generated from the
Campaign 4 working tree before the closeout commit existed. The dirty working
tree status is retained in the raw JSON so the evidence is not represented as
coming from the pre-Campaign base revision.

## Results

Timings are median seconds over 10 timed repetitions. CPU rows are same-host
Apple Silicon baselines. Device-resident rows keep operands on Metal.
Reused-output rows time `commutes_with_device(..., output=existing_matrix)`.
The `.metallib` rows load kernels from an offline library compiled by
`xcrun metal` and `xcrun metallib`; the binary artifact is not retained.

| Case | Terms | Words | CPU default | CPU NEON | Metal resident host output | Matrix reuse | Best forced selector | Offline `.metallib` reuse |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| words2 large | 768x768 | 2 | 4.75896e-04 | 4.60354e-04 | 1.98291e-04 | 1.34146e-04 | words2 1.28834e-04 | 1.69417e-04 |
| compact large | 2048x2048 | 1 | 2.39406e-03 | 2.42321e-03 | 5.11417e-04 | 2.39042e-04 | auto 1.67271e-04 | 1.83687e-04 |
| private boundary | 2048x2048 | 1 | 2.27160e-03 | 2.25800e-03 | 4.91459e-04 | 1.72167e-04 | `.metallib` auto 1.63396e-04 | 1.63396e-04 |

| Compact consumer | CPU shared scan | GPU atomic total | GPU parallel total | Decision |
| --- | ---: | ---: | ---: | --- |
| total count, 2048x2048 | 5.38521e-04 | 3.25521e-04 | 5.19749e-04 | keep GPU paths benchmark-only; atomic wins this case but is not enough for a public policy |
| column counts, 2048x2048 | 5.46251e-04 | 7.14813e-04 | not implemented | keep CPU scan default |
| row counts, 2048x2048 | 5.44583e-04 | 3.25521e-04 | not implemented | keep CPU scan default until axis workload demand is clearer |

| Storage experiment | Median seconds | Decision |
| --- | ---: | --- |
| shared device-resident host output, 2048x2048 | 4.91459e-04 | retained default |
| private output plus shared blit staging, 2048x2048 | 7.19417e-04 | do not promote for host-output commutation |

![FastPauli accelerator performance landscape](../plots/accelerator_landscape_with_rocm.svg)

## Interpretation

The two-word selector remains benchmark-only. Campaign 4 finally produced a
larger 768x768 shape where the forced `words2` kernel edged the generic 2D
baseline, but Campaign 3 showed the opposite on 384x384. One local Apple M4 Pro
run with mixed prior evidence is not enough to change the retained default for
all words >= 2.

The new parallel block-reduction compact total count is correct, but it does
not beat the current atomic-total GPU reduction on the 2048x2048 compact case.
The atomic total did beat the CPU shared scan for this total-count workload,
but row and column compact consumers remain mixed, and the hidden selector is
still benchmark-only. No public API behavior changes in this campaign.

Private storage plus shared blit staging stays a future device-only workflow
tool. It is still slower than shared output for current host-output
commutation, so the retained default remains shared storage.

Offline `.metallib` loading remains functional. It wins one 2048x2048
private-boundary row and loses or ties other rows, so it remains an evidence
tool rather than a default policy.

Profiler work remains bounded by what can be retained safely. `xcrun xctrace
list templates` confirms `Metal System Trace` is available, but this campaign
does not retain raw trace bundles or raw trace exports. The checked profiler
artifact records the derived blocker until the local toolchain can emit narrow
shader-counter value CSVs without process inventories or broad timeline data.

MPSGraph and PyTorch MPS external baselines remain skipped. Neither surface
currently provides an exact sparse Pauli packed-word commutation mapping with a
timing boundary comparable to FastPauli's device-resident rows.

## Checked Evidence

```text
Raw benchmark JSON: docs/benchmarks/data/apple_metal_optimization_campaign4_2026-05-06/raw/metal_benchmark_campaign4.json
Summary JSON: docs/benchmarks/data/apple_metal_optimization_campaign4_2026-05-06/summary.json
Profiler evidence: docs/benchmarks/data/apple_metal_optimization_campaign4_2026-05-06/profiler/metal_campaign4_profiler_evidence.json
README landscape plot: docs/benchmarks/plots/accelerator_landscape_with_rocm.svg
Campaign plan: docs/plans/apple_metal_optimization_campaign4_plan.md
Renderer: scripts/render_apple_metal_assets.py
```

## Remaining Headroom

1. Keep `words2` benchmark-only until it wins across additional shapes and at
   least one additional Apple Silicon generation.
2. Keep compact GPU selectors benchmark-only until total, row, and column
   consumers have a consistent policy story or a fused downstream consumer
   eliminates current command-buffer and host-readback costs.
3. Design axis-wise parallel reductions only if row or column compact consumers
   become a demonstrated workload bottleneck.
4. Use private storage only inside future fused device-only workflows that avoid
   host-output blit costs.
5. Repeat derived shader-counter export attempts when Instruments can produce
   sanitized value CSVs without raw trace retention.

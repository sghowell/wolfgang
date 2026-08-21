# Wave 1B CPU Small-Shape Dispatch Report

Wave 1B status: accepted on local Apple M4 Pro evidence.

The tested hypothesis was narrowly scoped: keep pairwise commutation auto-dispatch unchanged, but stop the NEON full-grouping auto path from mis-selecting the SIMD graph builder once the self-graph is already large enough that scalar wins on this host. The implementation adds one new CPU dispatch threshold, `neon_full_grouping_scalar_min_entries = 1024`, and uses it only in the `auto` full-grouping selector on the NEON lane.

## Commands

```bash
uv run --extra test python -m pytest tests/test_phase9_cpu_backend.py tests/test_cpu_thresholds_benchmark.py -q
uv run python benchmarks/bench_cpu_dispatch.py --repeat 5 --warmup 1 --json
uv run python benchmarks/bench_cpu_thresholds.py --repeat 5 --warmup 1 --json
uv run python scripts/validate.py
```

Each benchmark command was rerun three independent times. Promotion uses mean-of-medians.

## Environment

```text
Host: Apple M4 Pro, macOS-26.2-arm64-arm-64bit-Mach-O, arm64
Compiled CPU backends: scalar, neon
Available CPU backends: scalar, neon
Accepted baseline commit: ce6938f00c9a237cf83a2c291ce04dbf3e41e6b4
Benchmarked candidate revision: f38d8992cac8bd4c47780a0243daf084e876b756
```

The candidate benchmark JSON was regenerated after the artifact-hygiene fix so the tracked evidence is stamped to the exact benchmarked commit instead of a dirty pre-closeout label.

## What Changed

```text
include/wolfgang/cpu_backend.hpp
bindings/python/build_info.cpp
src/grouping.cpp
benchmarks/bench_cpu_dispatch.py
benchmarks/bench_cpu_thresholds.py
tests/test_phase9_cpu_backend.py
tests/test_cpu_thresholds_benchmark.py
benchmarks/_benchmark_metadata.py
tests/test_benchmark_metadata.py
```

Behavioral change:

```text
pairwise commutation auto selector: unchanged
full grouping auto selector on AVX2/AVX-512: unchanged
full grouping auto selector on NEON-only hosts:
- if num_terms^2 < 1024, keep NEON graph build
- if num_terms^2 >= 1024, use scalar graph build
forced neon selector: unchanged
forced scalar equivalence checks: unchanged
```

## Benchmark Results

Mean-of-medians across three reruns, lower is better:

| Case | Baseline mean | Candidate mean | Delta |
| --- | ---: | ---: | ---: |
| auto full grouping | 0.000041666 s | 0.000032528 s | -21.93% |
| forced scalar full grouping | 0.000054361 s | 0.000035166 s | -35.31% |
| forced neon full grouping | 0.000039722 s | 0.000042958 s | 8.15% |
| auto pairwise commutation | 0.000032625 s | 0.000025514 s | -21.80% |
| below-threshold pairwise sweep | 0.000072750 s | 0.000070806 s | -2.67% |
| at-threshold pairwise sweep | 0.000270347 s | 0.000271222 s | 0.32% |
| above-threshold pairwise sweep | 0.001043194 s | 0.001036097 s | -0.68% |

Selector outcome change recorded by the dispatch benchmark:

```text
auto_pairwise_commutation effective_backend_hint: neon -> neon
auto_full_grouping effective_backend_hint: neon -> scalar
```

## Correctness And Schema Checks

```text
Focused pytest: 14 passed in 2.43s
scripts/validate.py: passed
Dispatch correctness: every auto/forced grouping and pairwise row matched forced scalar
Threshold correctness: every threshold row matched forced scalar
Dispatch top-level JSON keys: unchanged
Threshold benchmark top-level JSON keys: unchanged
Threshold metadata addition: cpu_auto_dispatch_thresholds.neon_full_grouping_scalar_min_entries = 1024
```

The small-workload regression budget was preserved on the documented pairwise threshold benchmark: the candidate changed below-threshold by -2.67%, at-threshold by 0.32%, and above-threshold by -0.68%.

## Decision

Accept the change.

Reason:

```text
auto full grouping improved by 21.93% on the checked 128-term NEON dispatch case
pairwise commutation auto-dispatch stayed on neon and did not regress beyond the 5% rejection budget on the documented threshold sweep
forced neon and forced scalar behavior stayed correct and unchanged in semantics
```

## Artifacts

```text
Tracked benchmark JSON: docs/benchmarks/data/cpu_small_shape_dispatch_wave1b_2026-08-21/
Summary JSON: docs/benchmarks/data/cpu_small_shape_dispatch_wave1b_2026-08-21/summary.json
Report: docs/benchmarks/reports/cpu_small_shape_dispatch_wave1b_2026-08-21.md
Working raw copies: artifacts/wave1b_cpu_small_shape_dispatch/
```

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
Host: Apple M4 Pro, macOS 26.2, arm64
Compiled CPU backends: scalar, neon
Available CPU backends: scalar, neon
Accepted baseline commit: ce6938f00c9a237cf83a2c291ce04dbf3e41e6b4
Benchmarked candidate revision: ce6938f+dirty
```

The candidate benchmark JSON is dirty because the timings were captured from the in-progress Wave 1B branch before the closeout commit.

## What Changed

```text
include/wolfgang/cpu_backend.hpp
bindings/python/build_info.cpp
src/grouping.cpp
benchmarks/bench_cpu_dispatch.py
benchmarks/bench_cpu_thresholds.py
tests/test_phase9_cpu_backend.py
tests/test_cpu_thresholds_benchmark.py
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
| auto full grouping | 0.000025208 s | 0.000018903 s | -25.01% |
| forced scalar full grouping | 0.000020500 s | 0.000020361 s | -0.68% |
| forced neon full grouping | 0.000025875 s | 0.000026042 s | 0.64% |
| auto pairwise commutation | 0.000014167 s | 0.000013514 s | -4.61% |
| below-threshold pairwise sweep | 0.000070181 s | 0.000070139 s | -0.06% |
| at-threshold pairwise sweep | 0.000262306 s | 0.000268569 s | 2.39% |
| above-threshold pairwise sweep | 0.001003028 s | 0.001015625 s | 1.26% |

Selector outcome change recorded by the dispatch benchmark:

```text
auto_pairwise_commutation effective_backend_hint: neon -> neon
auto_full_grouping effective_backend_hint: neon -> scalar
```

## Correctness And Schema Checks

```text
Focused pytest: 14 passed
scripts/validate.py: passed
Dispatch correctness: every auto/forced grouping and pairwise row matched forced scalar
Threshold correctness: every threshold row matched forced scalar
Dispatch top-level JSON keys: unchanged
Threshold benchmark top-level JSON keys: unchanged
Threshold metadata addition: cpu_auto_dispatch_thresholds.neon_full_grouping_scalar_min_entries = 1024
```

The small-workload regression budget was preserved on the documented pairwise threshold benchmark: the candidate changed below-threshold by -0.06%, at-threshold by 2.39%, and above-threshold by 1.26%.

## Decision

Accept the change.

Reason:

```text
auto full grouping improved by 25.01% on the checked 128-term NEON dispatch case
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

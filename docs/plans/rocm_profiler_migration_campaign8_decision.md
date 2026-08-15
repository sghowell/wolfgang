# ROCm Campaign 8 Profiler Migration Decision

Campaign 8 defines when FastPauli should migrate ROCm evidence collection from
legacy `rocprof` commands to `rocprofv3`. This document does not change any HIP
kernel, benchmark workload, or support claim.

## Decision

Legacy `rocprof` remains accepted when it produces HIP trace and stats
artifacts for retained FastPauli HIP operations.

`rocprofv3` is the preferred ROCm 7.x-and-later profiler lane when it is
installed on the target host and can produce equivalent HIP API and kernel
evidence. Migration is accepted only after side-by-side evidence exists for at
least one retained operation profile.

## Required Side-By-Side Evidence

A future report may replace legacy `rocprof` only when it records:

```text
ROCm version
rocprof binary path and version or help output
rocprofv3 binary path and version or help output
same benchmark command for both profiler lanes
legacy rocprof output directory
rocprofv3 output directory
HIP API trace availability
kernel timing or stats availability
copy timing or stats availability when the operation transfers data
tool warnings
permission or provider restrictions
```

If `rocprofv3` is unavailable, the report must record the exact missing binary,
permission failure, incompatible option, provider image limitation, or runtime
error. A generic "tool unavailable" note is not sufficient.

## Command Shape

The retained legacy lane is:

```bash
rocprof -d <evidence>/profiler/legacy --hip-trace --stats \
  .venv/bin/python benchmarks/bench_rocm_kernels.py \
  --profile campaign7-profiler --repeat 1 --warmup 0 --json \
  --output <evidence>/raw/rocm_profiler_legacy.json
```

The Campaign 8 candidate `rocprofv3` lane is:

```bash
rocprofv3 --hip-trace --stats -d <evidence>/profiler/rocprofv3 -- \
  .venv/bin/python benchmarks/bench_rocm_kernels.py \
  --profile campaign7-profiler --repeat 1 --warmup 0 --json \
  --output <evidence>/raw/rocm_profiler_rocprofv3.json
```

The exact `rocprofv3` flags may be adjusted to the installed ROCm version, but
the report must preserve the semantic boundary: same benchmark profile, same
repeat and warmup policy, profiler output separated from benchmark timing, and
tooling overhead not mixed into release-smoke timing rows.

## Acceptance Criteria

`rocprofv3_migration` may be marked `accepted_for_future_implementation` only
when the migration plan includes side-by-side command shape and exact
unavailability evidence rules.

`legacy_rocprof_retention` remains `retained` until a future campaign checks in
equivalent `rocprofv3` artifacts and updates all benchmark/report instructions.

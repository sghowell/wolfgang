# ROCm MI300X Campaign 3 Report

Date: 2026-04-30

Git revision benchmarked: `99dd8e7`

Evidence root:
`docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30`

Plots:

```text
docs/benchmarks/plots/rocm_mi300x_campaign3_simplify.svg
docs/benchmarks/plots/accelerator_landscape_with_rocm.svg
```

## Scope

Campaign 3 adds the first sparse-output ROCm/HIP Pauli operation:
`DevicePauliSum.simplify()` for HIP-only source builds on MI300X `gfx942`.
The retained public behavior is intentionally narrow:

```text
HIP DevicePauliSum.simplify(atol=1e-12, rtol=0.0)
device-resident HIP DevicePauliSum output
CPU-equivalent canonical ordering and tolerance filtering
rocThrust duplicate reduction as the retained implementation path
checked MI300X benchmark and rocprof evidence
```

The campaign does not add HIP DLPack, public streams, public workspaces,
expectation, matmul, multi-GPU execution, ROCm wheels, additional AMD GPU
support claims, or simultaneous CUDA+HIP builds.

## Host And Build Inventory

The MI300X lane used `/root/FastPauli` on a ROCm-enabled Ubuntu host.
`results.sysinfo.txt` records:

```text
GPU: AMD Instinct MI300X VF
LLVM target: gfx942:sramecc+:xnack-
Compute units: 304
L2 cache: 4096 KB
L3 cache: 262144 KB
Max GPU clock: 2100 MHz
CPU: Intel Xeon Platinum 8568Y+
```

Benchmark JSON records:

```text
FASTPAULI_ENABLE_HIP=ON
FASTPAULI_HIP_ARCHITECTURES=gfx942
FASTPAULI_ENABLE_CUDA=OFF
FASTPAULI_ENABLE_NATIVE=OFF
ROCm runtime: 7.2.26015
ROCm toolkit: 7.2.26015-fc0010cf6a
HIP compiler: /opt/rocm/bin/amdclang++ 22.0.0
C++ compiler: GNU 13.3.0
Compiled CPU selectors: scalar, tbb, avx2, avx512
Active CPU backend during simplify benchmarks: scalar
```

## Implementation Outcome

`DevicePauliSum.simplify()` is implemented for HIP builds and returns a
HIP-backed `DevicePauliSum`. It preserves explicit device residency until the
caller invokes `to_host()`.

The implementation uses rocThrust/rocPRIM primitives for the retained path.
It includes:

```text
empty input handling
all-zero output with num_qubits preserved
one-word packed-key path
two-word path
generic multi-word fallback path
inclusive CPU-equivalent tolerance filtering
negative, NaN, and infinite tolerance rejection
HIP build metadata listing simplify under hip_kernels
```

The generic multi-word fallback is correct but not retained as a performance
claim. It is deliberately tracked as remaining headroom because it sorts and
reduces a wider key representation less efficiently than the one-word and
two-word cases.

## Validation

Local CPU-only validation run in the normal macOS checkout:

```bash
<private-path> -m pytest tests/test_rocm_campaign3_assets.py -q
<private-path> -m pytest tests/test_phase12_rocm_foundation.py -q
```

Result:

```text
tests/test_rocm_campaign3_assets.py: 2 passed
tests/test_phase12_rocm_foundation.py: 6 passed, 18 skipped
```

MI300X HIP validation run during implementation:

```bash
PATH=/opt/rocm/bin:$PATH .venv/bin/python -m pytest \
  tests/test_phase12_rocm_foundation.py::test_hip_simplify_matches_cpu_for_edge_cases_when_available \
  tests/test_phase12_rocm_foundation.py::test_hip_simplify_tolerance_matches_cpu_when_available \
  tests/test_phase12_rocm_foundation.py::test_hip_simplify_randomized_matches_cpu_when_available \
  tests/test_phase12_rocm_foundation.py::test_hip_simplify_rejects_invalid_tolerances_when_available \
  tests/test_phase12_rocm_foundation.py::test_hip_deferred_surfaces_remain_unavailable_after_simplify_when_available \
  -q

PATH=/opt/rocm/bin:$PATH .venv/bin/python -m pytest tests/test_phase12_rocm_foundation.py -q
```

Result:

```text
targeted HIP simplify tests: 7 passed
full ROCm foundation tests: 22 passed, 1 skipped
```

Closeout validation after the report and README updates:

```bash
<private-path> scripts/validate.py

ssh root@<private-address> \
  'cd /root/FastPauli && PATH=/opt/rocm/bin:$PATH .venv/bin/python -m pytest -q'

ssh root@<private-address> \
  'cd /root/FastPauli && PATH=/opt/rocm/bin:$PATH .venv/bin/python benchmarks/bench_rocm_kernels.py --profile simplify-smoke --repeat 3 --warmup 1 --json'
```

Result:

```text
local validate.py: passed, including pytest 202 passed / 78 skipped
MI300X full pytest: 219 passed / 61 skipped
MI300X simplify smoke on synced commit 9715756: status ok, correctness_passed true
```

## Benchmark Commands

The retained benchmark rows were generated on MI300X with:

```bash
PATH=/opt/rocm/bin:$PATH FASTPAULI_BENCHMARK_GIT_COMMIT=99dd8e7 \
  .venv/bin/python benchmarks/bench_rocm_kernels.py \
  --profile simplify-duplicate-pressure --repeat 5 --warmup 2 --json \
  --output docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30/raw/rocm_simplify_duplicate_pressure_mi300x.json

PATH=/opt/rocm/bin:$PATH FASTPAULI_BENCHMARK_GIT_COMMIT=99dd8e7 \
  .venv/bin/python benchmarks/bench_rocm_kernels.py \
  --profile simplify-wide-qubit --repeat 5 --warmup 2 --json \
  --output docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30/raw/rocm_simplify_wide_qubit_mi300x.json

PATH=/opt/rocm/bin:$PATH FASTPAULI_BENCHMARK_GIT_COMMIT=99dd8e7 \
  .venv/bin/python benchmarks/bench_rocm_kernels.py \
  --profile simplify-smoke --repeat 3 --warmup 1 --json \
  --output docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30/raw/rocm_simplify_smoke_mi300x.json
```

The rejected hipCUB strategy probe was generated with:

```bash
PATH=/opt/rocm/bin:$PATH \
FASTPAULI_HIP_BENCH_DUPLICATE_REDUCTION=hipcub_radix_sort_reduce \
FASTPAULI_BENCHMARK_GIT_COMMIT=99dd8e7 \
  .venv/bin/python benchmarks/bench_rocm_kernels.py \
  --profile simplify-strategy-ab --repeat 5 --warmup 2 --json \
  --output docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30/raw/rocm_simplify_strategy_hipcub_mi300x.json
```

Profiler and requested-counter passes were generated with:

```bash
PATH=/opt/rocm/bin:$PATH FASTPAULI_BENCHMARK_GIT_COMMIT=99dd8e7 \
  rocprof -d docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30/profiler \
  --hip-trace --stats \
  .venv/bin/python benchmarks/bench_rocm_kernels.py \
  --profile simplify-campaign3-profiler --repeat 1 --warmup 0 --json \
  --output docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30/raw/rocm_simplify_profiler_mi300x.json

PATH=/opt/rocm/bin:$PATH FASTPAULI_BENCHMARK_GIT_COMMIT=99dd8e7 \
  rocprof \
  -i docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30/profiler/rocprof_campaign3_requested_counters.txt \
  -o docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30/profiler/rocm_simplify_campaign3_requested_counters.csv \
  .venv/bin/python benchmarks/bench_rocm_kernels.py \
  --profile simplify-campaign3-profiler --repeat 1 --warmup 0 --json \
  --output docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30/raw/rocm_simplify_requested_counters_mi300x.json
```

## Benchmark Results

Median seconds, lower is better:

| Case | Qubits | Terms | CPU scalar | HIP transfer | HIP resident | HIP to_host | Resident speedup vs CPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| duplicate heavy | 24 | 32768 | 0.004234 | 0.000800 | 0.000335 | 0.000036 | 12.64x |
| duplicate light | 24 | 32768 | 0.004805 | 0.001106 | 0.000347 | 0.000146 | 13.83x |
| all zero | 24 | 4096 | 0.000040 | 0.000285 | 0.000228 | 0.0000004 | 0.18x |
| smoke one word | 8 | 128 | 0.000010 | 0.000346 | 0.000263 | 0.000033 | 0.04x |
| wide two word | 70 | 8192 | 0.002268 | 0.000508 | 0.000335 | 0.000056 | 6.77x |
| generic multiword | 130 | 4096 | 0.000826 | 0.005787 | 0.005534 | 0.000050 | 0.15x |

Interpretation:

```text
duplicate-heavy and duplicate-light one-word workloads benefit strongly from the HIP resident path
the two-word path is also faster than scalar CPU on the measured MI300X row
small and all-zero cases are dominated by fixed accelerator overhead
the current generic multi-word fallback is correct but slower than CPU scalar on this dataset
transfer-inclusive timing stays faster than CPU for the large one-word and two-word rows, but not for tiny or generic multi-word rows
```

## Strategy Decisions

| Strategy | Status | Decision |
|---|---|---|
| rocThrust default | retained | Correctness passed across edge, tolerance, randomized, duplicate-heavy, duplicate-light, one-word, two-word, and generic rows. It is the only retained production path in Campaign 3. |
| hipCUB radix sort/reduce | rejected_with_evidence | The benchmark selector records a rejected row with a named reason. No public or private production path was retained without separate correctness and allocation evidence. |
| custom packed key | unavailable | Not implemented in Campaign 3. It remains a future experiment only if rocThrust allocation, key packing, or generic multi-word headroom becomes dominant. |

## Profiler Evidence

Profiler artifacts are checked in under:

```text
docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30/profiler/
```

Important files:

```text
results.stats.csv
results.hip_stats.csv
results.copy_stats.csv
results.json
results.sysinfo.txt
rocm_simplify_campaign3_requested_counters.csv
rocprof_campaign3_requested_counters.txt
rocprof_requested_counters.stdout
rocprof_requested_counters.stderr
```

The requested counter CSV includes:

```text
SQ_WAVES
GRBM_GUI_ACTIVE
FETCH_SIZE
WRITE_SIZE
VALUUtilization
VALUBusy
```

`results.hip_stats.csv` attributes the profiler-instrumented run mostly to
`hipMemcpy` and `hipLaunchKernel`:

| HIP API | Calls | Total ns | Avg ns | Share |
|---|---:|---:|---:|---:|
| hipMemcpy | 15 | 78011018 | 5200734 | 50.60% |
| hipLaunchKernel | 46 | 74771624 | 1625470 | 48.50% |
| hipFree | 32 | 431098 | 13471 | 0.28% |
| hipMemcpyWithStream | 6 | 352747 | 58791 | 0.23% |
| hipMalloc | 32 | 265694 | 8302 | 0.17% |
| hipStreamSynchronize | 24 | 251937 | 10497 | 0.16% |

`results.stats.csv` confirms rocPRIM/rocThrust kernels dominate the retained
device-resident simplify path: reduction, radix sort, merge sort, reduce-by-key,
partition, and transform kernels appear as the primary kernel families.

## Independent Review

An independent review was run before merge over the Campaign 3 branch range.
It found no P0 correctness issues in the HIP `DevicePauliSum.simplify()` path
and confirmed the implementation remains scoped to HIP-only builds, keeps
public headers free of ROCm/HIP includes, preserves device-resident output, and
covers empty, one-word, two-word, generic multi-word, tolerance, and deferred
surface tests.

Review findings and resolutions:

| Finding | Severity | Resolution |
|---|---|---|
| Branch-level `git diff --check` failed on checked-in profiler text artifacts with trailing whitespace. | P1 | Stripped trailing whitespace from the raw text profiler artifacts and re-ran `git diff --check 874f420ac686b0053329bec5e76aa5045087f7e6..HEAD`, which passed. |
| Campaign 3 renderer test verified filenames but did not compare regenerated plots or path-normalized summary output against checked-in assets. | P2 | Added path-normalized summary comparison and exact SVG comparison to `tests/test_rocm_campaign3_assets.py`; targeted test passes. |

Residual non-blocking risk:

```text
raw benchmark rows are recorded at 99dd8e7, before later report/test-only closeout commits
post-report MI300X smoke was repeated on later synced heads with correctness enabled
final merge closeout must record the exact final-head validation command and result
```

## Terminal Status For Campaign 2 Headroom

| Item | Status | Reason | Next trigger |
|---|---|---|---|
| HIP simplify | accepted | Implemented, tested, benchmarked, profiled, and documented in Campaign 3. | Continue optimizing generic multi-word and allocation-heavy cases only with new measured headroom. |
| README broad landscape | accepted | `accelerator_landscape_with_rocm.svg` preserves CPU, CUDA, external, and ROCm rows. | Keep this as the README default plot for future accelerator reports. |
| HIP DLPack | out_of_scope_with_next_trigger | No ROCm consumer ownership and stream contract was accepted in this campaign. | Start a HIP interop plan with PyTorch ROCm or CuPy ROCm consumer evidence. |
| Public streams | out_of_scope_with_next_trigger | Campaign 3 simplify remains synchronous, matching earlier ROCm boundaries. | Reopen only if profiler evidence shows launch or synchronization control dominates retained workloads. |
| Public workspaces | out_of_scope_with_next_trigger | rocThrust allocation overhead is visible but not yet enough to justify a public workspace API. | Reopen with a private reusable workspace prototype and allocation-attribution A/B data. |
| Packed summaries | out_of_scope_with_next_trigger | Campaign 2 compact commutation consumers remain the retained summary boundary. | Reopen when a named public consumer needs a packed HIP summary rather than counts or conflict degrees. |
| HIP expectation | out_of_scope_with_next_trigger | No HIP statevector or counts semantic parity slice was in scope. | Promote CPU/CUDA expectation parity fixtures to HIP in a separate plan. |
| HIP matmul | out_of_scope_with_next_trigger | Sparse product kernels require separate phase/ordering coverage. | Start after HIP simplify and commutation are stable on broader workloads. |
| ROCm portability | out_of_scope_with_next_trigger | Evidence is MI300X `gfx942` only. | Run a portability campaign on another AMD GPU when release wording needs it. |
| ROCm wheels | out_of_scope_with_next_trigger | Source-build evidence does not imply binary wheel support. | Start packaging policy work after supported ROCm toolkit and GPU matrix decisions. |
| Multi-GPU MI300X | out_of_scope_with_next_trigger | No multi-device ownership or scheduling contract exists. | Reopen with a distributed/partitioned Pauli workload design. |
| Simultaneous CUDA+HIP | out_of_scope_with_next_trigger | Current object model intentionally permits only one accelerator backend per build. | Reopen with a backend-neutral device object architecture decision. |

## Remaining Headroom

The next ROCm campaign should focus on one of these measured gaps rather than
starting a broad rewrite:

```text
private HIP workspace/reusable scratch A/B test for rocThrust-heavy simplify
custom packed-key duplicate reduction for one-word and two-word simplify rows
generic multi-word simplify redesign, because the Campaign 3 fallback is slower than CPU scalar
HIP DLPack/consumer interop if a named PyTorch ROCm or CuPy ROCm consumer is available
HIP expectation or matmul only after parity fixtures are promoted from CPU/CUDA
```

The strongest immediate optimization target is generic multi-word simplify.
The strongest API-design target is a private workspace experiment, not a public
workspace surface.

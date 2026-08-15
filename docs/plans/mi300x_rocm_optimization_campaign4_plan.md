# MI300X ROCm Optimization Campaign 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exhaust the measured HIP simplify headroom from Campaign 3 on MI300X by testing private workspace reuse, custom packed-key duplicate reduction, and a parallel generic multi-word simplify redesign without expanding the public ROCm API.

**Architecture:** Campaign 4 keeps ROCm/HIP source-build-only, MI300X `gfx942` evidenced, and mutually exclusive with CUDA. `DevicePauliSum.simplify()` remains the only public HIP surface touched by this campaign; all workspace, allocator, custom-key, and generic-reduction variants are private implementation or benchmark-only experiments until correctness, profiling, and benchmark evidence justify retaining them. HIP DLPack, public streams, public workspaces, HIP expectation, HIP matmul, ROCm wheels, multi-GPU execution, broader AMD portability claims, and simultaneous CUDA+HIP builds remain out of implementation scope and must receive explicit report statuses.

**Tech Stack:** C++20, nanobind, CMake HIP language support, ROCm/HIP runtime, rocThrust, rocPRIM/hipCUB where available, AMD Instinct MI300X `gfx942`, rocprof, pytest, NumPy, existing FastPauli CPU/CUDA/ROCm benchmark-report infrastructure.

---

## Status

```text
complete
```

## Baseline

Campaign 3 is complete:

```text
plan: docs/plans/mi300x_rocm_optimization_campaign3_plan.md
report: docs/benchmarks/reports/rocm_mi300x_campaign3_2026-04-30.md
evidence: docs/benchmarks/data/rocm_mi300x_campaign3_2026-04-30/
retained public surface: HIP DevicePauliSum.simplify() with device-resident output
retained implementation path: rocThrust duplicate reduction
```

Campaign 3 benchmarked these median rows:

```text
duplicate heavy one-word: HIP resident 0.000335 s vs CPU scalar 0.004234 s
duplicate light one-word: HIP resident 0.000347 s vs CPU scalar 0.004805 s
wide two-word: HIP resident 0.000335 s vs CPU scalar 0.002268 s
generic multi-word: HIP resident 0.005534 s vs CPU scalar 0.000826 s
small/all-zero rows: dominated by fixed accelerator overhead
```

Campaign 3 explicitly identified the next measured headroom:

```text
private HIP workspace/reusable scratch A/B test for rocThrust-heavy simplify
custom packed-key duplicate reduction for one-word and two-word simplify rows
generic multi-word simplify redesign because Campaign 3 is slower than CPU scalar there
HIP DLPack/consumer interop only with a named PyTorch ROCm or CuPy ROCm consumer
HIP expectation or matmul only after CPU/CUDA parity fixtures are promoted to HIP
```

Campaign 4 implements only the first three performance items. Interop,
expectation, and matmul remain follow-on campaigns unless this campaign records
a concrete blocker that must be resolved first.

## Campaign 4 Scope

In scope:

```text
private HIP simplify strategy instrumentation
private workspace/reusable scratch A/B tests for simplify-related temporary storage
rocPRIM or hipCUB scratch-buffer probes when available
custom packed-key duplicate-reduction probe for one-word <=32-qubit rows
custom packed-key duplicate-reduction probe for one-word >32-qubit rows
custom packed-key duplicate-reduction probe for two-word rows
parallel generic multi-word simplify redesign using sorted indices plus parallel reduce-by-key or equivalent segment reduction
MI300X benchmark rows for Campaign 3 baseline, custom key variants, private workspace variants, and generic multi-word variants
rocprof trace/stats/counter evidence for retained and rejected simplify variants
README broad performance landscape refresh only if measured rows change the latest landscape
Campaign 4 report with terminal statuses for every in-scope experiment and every still-deferred Campaign 3 headroom item
```

Hard out of implementation scope:

```text
public HIP workspace handles
public HIP stream or graph parameters
HIP DLPack or __dlpack_device__
CUDA Array Interface exposure from HIP objects
HIP expectation kernels
HIP matmul kernels
multi-GPU MI300X execution
ROCm binary wheels
additional AMD GPU support claims beyond MI300X gfx942 evidence
simultaneous CUDA+HIP source builds
```

Campaign 4 may collect availability diagnostics for hard out-of-scope surfaces,
but it must not retain public APIs, packaging claims, portability claims, or
backend-object-model changes for them.

## Evidence Layout

Use this evidence root:

```text
docs/benchmarks/data/rocm_mi300x_campaign4_2026-04-30/
docs/benchmarks/data/rocm_mi300x_campaign4_2026-04-30/logs/
docs/benchmarks/data/rocm_mi300x_campaign4_2026-04-30/raw/
docs/benchmarks/data/rocm_mi300x_campaign4_2026-04-30/profiler/
docs/benchmarks/data/rocm_mi300x_campaign4_2026-04-30/summary.json
docs/benchmarks/reports/rocm_mi300x_campaign4_2026-04-30.md
docs/benchmarks/plots/rocm_mi300x_campaign4_simplify_hardening.svg
docs/benchmarks/plots/accelerator_landscape_with_rocm.svg
```

If execution crosses UTC midnight, keep this evidence root only if the report
states the exact execution dates and why the root remains named
`rocm_mi300x_campaign4_2026-04-30`.

## Retention Gates

A private optimization may be retained only when all gates pass:

```text
CPU/HIP equivalence passes for deterministic edge cases and fixed-seed randomized cases
invalid public inputs keep the same Python exception class and message intent
public API shape, public headers, CPU-only builds, CUDA builds, and CUDA+HIP mutual exclusion remain unchanged
median device-resident timing improves by at least 1.15x on the targeted row or eliminates a correctness-safe bottleneck documented by profiler evidence
median timing regresses by no more than 5 percent on non-target simplify rows unless the report rejects the variant or gates it away from those rows
transfer-inclusive and to_host materialization boundaries are reported separately
rocprof trace/stats/counter evidence or a precise tooling blocker is checked in
benchmark JSON names the active strategy and workspace mode for every row
```

For generic multi-word simplify, a retained redesign must also improve the
Campaign 3 generic row by at least 2x or make the row faster than scalar CPU.
If it improves less than 2x while preserving correctness, record it as
`rejected_with_evidence` or `benchmark_only` rather than relabeling it as a
production performance win.

## Acceptance Criteria

Campaign 4 is complete only when every applicable item has a terminal status in
the Campaign 4 report:

```text
local CPU-only validation passes with FASTPAULI_ENABLE_HIP=OFF
HIP source build succeeds on MI300X with FASTPAULI_ENABLE_HIP=ON and FASTPAULI_HIP_ARCHITECTURES=gfx942
CUDA+HIP configure-time rejection still passes
public headers still contain no HIP or ROCm runtime headers
HIP DevicePauliSum.simplify() still matches CPU simplify for Campaign 3 edge, tolerance, randomized, one-word, two-word, and generic multi-word cases
generic multi-word simplify has a parallel A/B implementation or a recorded blocked reason tied to ROCm library limitations
custom packed-key one-word and two-word strategies have correctness, benchmark, and profiler statuses
private workspace or reusable scratch experiments have allocation-attribution, correctness, benchmark, and profiler statuses
benchmark JSON separates transfer-inclusive, device-resident, to_host, strategy, and workspace boundaries
README broad performance landscape remains CPU/CUDA/ROCm/external rather than a narrow ROCm-only plot
HIP DLPack, public streams, public workspaces, HIP expectation, HIP matmul, multi-GPU, portability, ROCm wheels, and simultaneous CUDA+HIP claims remain unavailable unless separate plans accept them
independent review is recorded before merge
```

## Task 1: Contracts, Benchmark Schema, And Red Tests

**Files:**
- Modify: `docs/architecture/rocm_backend.md`
- Modify: `docs/benchmarks/protocol.md`
- Modify: `tests/test_phase12_rocm_foundation.py`
- Test: `python -m pytest tests/test_phase12_rocm_foundation.py -q`

- [ ] **Step 1: Add the Campaign 4 private-boundary contract**

Add a `Planned Campaign 4 Simplify Hardening Boundary` section to
`docs/architecture/rocm_backend.md` with these decisions:

```text
Campaign 4 may change private HIP simplify implementation details but may not add public HIP APIs.
DevicePauliSum.simplify() remains synchronous and returns a HIP-backed DevicePauliSum.
Private workspace and scratch-buffer experiments must not expose device pointers, streams, allocators, or lifetime handles to Python.
Custom packed-key and generic multi-word strategies must preserve PauliSum.simplify() canonical ordering, coefficient summation, and tolerance filtering.
Any retained private strategy must report its strategy name in benchmark JSON and HIP build metadata only if that metadata remains accurate for CPU-only builds.
Public HIP DLPack, streams, workspaces, expectation, matmul, multi-GPU, ROCm wheels, portability claims, and simultaneous CUDA+HIP builds stay unavailable in Campaign 4.
```

- [ ] **Step 2: Add Campaign 4 benchmark fields**

Add a `ROCm Campaign 4 simplify-hardening fields` subsection to
`docs/benchmarks/protocol.md` requiring these JSON fields when a Campaign 4 row
has `status: ok`:

```text
campaign: rocm_mi300x_campaign4
operation: simplify
backend: hip
hip_simplify_strategy: rocthrust_default, rocthrust_generic_parallel_reduce_by_key, custom_packed_key, rocprim_scratch_probe, or hipcub_scratch_probe
hip_simplify_strategy_status: retained, rejected_with_evidence, benchmark_only, unavailable, or blocked_external
hip_simplify_strategy_reason
hip_simplify_key_shape: empty, identity, packed32, key1, key2, or generic_multiword
hip_workspace_mode: absent, grow_inside_timing, pre_reserved_outside_timing, benchmark_only, or unavailable
hip_workspace_reserved_bytes
hip_workspace_high_watermark_bytes
hip_workspace_allocation_count
hip_workspace_growth_count
hip_simplify_transfer_seconds and p10/p90/min/max variants
hip_simplify_device_resident_seconds and p10/p90/min/max variants
hip_simplify_to_host_seconds and p10/p90/min/max variants
hip_simplify_output_terms
hip_simplify_output_words
generic_multiword_parallelism: serial_kernel, reduce_by_key, segmented_reduce, or not_applicable
timing_boundary: transfer_inclusive, device_resident, device_output_to_host, preallocated, or benchmark_only
correctness_digest with input_terms, output_terms, coefficient_l1, and canonical_label_hash
campaign4_terminal_statuses for workspace, custom packed key, generic multi-word, DLPack, streams, expectation, matmul, portability, ROCm wheels, multi-GPU, and simultaneous CUDA+HIP
```

- [ ] **Step 3: Add generic multi-word and regression guard tests**

Add tests to `tests/test_phase12_rocm_foundation.py` that reuse the existing
HIP runtime guard and compare to CPU simplify:

```python
def test_hip_simplify_campaign4_generic_multiword_pressure_when_available() -> None:
    _require_hip_runtime()

    rng = np.random.default_rng(40404)
    labels = []
    coeffs = []
    alphabet = np.asarray(["I", "X", "Y", "Z"])
    for _ in range(96):
        labels.append("".join(rng.choice(alphabet, size=193).tolist()))
        coeffs.append(complex(float(rng.normal()), float(rng.normal())))
    labels.extend(labels[:32])
    coeffs.extend([-value for value in coeffs[:16]])
    coeffs.extend(coeffs[16:32])

    op = fastpauli.PauliSum.from_labels(labels, coeffs)
    _assert_hip_simplify_matches_cpu(op, atol=1.0e-11, rtol=1.0e-12)


def test_hip_simplify_campaign4_one_and_two_word_regression_when_available() -> None:
    _require_hip_runtime()

    cases = [
        fastpauli.PauliSum.from_labels(["X" * 24, "X" * 24, "Z" * 24], [1.0, 2.0, -3.0]),
        fastpauli.PauliSum.from_sparse_list(
            [("XY", [0, 70], 1.0), ("XY", [0, 70], -0.25), ("Z", [69], 2.0)],
            num_qubits=72,
        ),
    ]
    for op in cases:
        _assert_hip_simplify_matches_cpu(op, atol=0.0, rtol=0.0)
```

The tests should pass on the current Campaign 3 implementation before any
performance rewrite. If they fail, fix correctness before starting
optimization.

- [ ] **Step 4: Run the red/guard step**

Run locally:

```bash
python -m pytest tests/test_phase12_rocm_foundation.py -q
```

Expected on macOS CPU-only builds: HIP-specific tests skip and existing local
tests pass.

Run on MI300X before implementation:

```bash
PATH=/opt/rocm/bin:$PATH .venv/bin/python -m pytest \
  tests/test_phase12_rocm_foundation.py::test_hip_simplify_campaign4_generic_multiword_pressure_when_available \
  tests/test_phase12_rocm_foundation.py::test_hip_simplify_campaign4_one_and_two_word_regression_when_available \
  -q
```

Expected: passes on the Campaign 3 implementation. These tests are guardrails,
not intentionally failing tests, because Campaign 4 is a correctness-preserving
performance campaign.

- [ ] **Step 5: Commit**

```bash
git add docs/architecture/rocm_backend.md docs/benchmarks/protocol.md tests/test_phase12_rocm_foundation.py
git commit -m "test: guard ROCm simplify hardening"
```

## Task 2: Parallel Generic Multi-Word Simplify Redesign

**Files:**
- Modify: `src/hip/simplify_hip.hip.cpp`
- Modify: `src/hip/simplify_hip.hip.hpp`
- Test: `python -m pytest tests/test_phase12_rocm_foundation.py -q`

- [ ] **Step 1: Replace the serial generic reducer with a parallel A/B path**

Keep the existing generic path available behind a private fallback while adding
a parallel reduce-by-key strategy over sorted term indices. The core shape
should be:

```cpp
struct GenericTermIndexEqual {
  const std::uint64_t* x;
  const std::uint64_t* z;
  std::size_t words;

  __host__ __device__ bool operator()(std::size_t lhs, std::size_t rhs) const noexcept {
    const std::size_t lhs_offset = lhs * words;
    const std::size_t rhs_offset = rhs * words;
    for (std::size_t word = 0; word < words; ++word) {
      if (x[lhs_offset + word] != x[rhs_offset + word] ||
          z[lhs_offset + word] != z[rhs_offset + word]) {
        return false;
      }
    }
    return true;
  }
};
```

Then use permutation iterators so reduction reads coefficients in sorted-key
order without copying coefficients through a serial kernel:

```cpp
auto sorted_coeffs = thrust::make_permutation_iterator(coeff_ptr, sorted_indices.begin());
thrust::device_vector<std::size_t> reduced_indices(impl_->num_terms);
thrust::device_vector<thrust::complex<double>> reduced_values(impl_->num_terms);
auto reduced_end = thrust::reduce_by_key(
    sorted_indices.begin(),
    sorted_indices.end(),
    sorted_coeffs,
    reduced_indices.begin(),
    reduced_values.begin(),
    GenericTermIndexEqual{impl_->x, impl_->z, impl_->words},
    thrust::plus<thrust::complex<double>>{});
```

Copy surviving representative indices and coefficients with `thrust::copy_if`,
then scatter packed `x` and `z` words from the representative input terms with a
parallel transform. Preserve the Campaign 3 serial path behind a private
fallback selector until benchmark evidence decides retention.

- [ ] **Step 2: Preserve canonical ordering**

Use the same lexicographic order as CPU `PauliSum.simplify()` and Campaign 3
generic sorting. Do not introduce hash-order, unstable bucket-order, or
non-deterministic output ordering.

- [ ] **Step 3: Validate generic correctness on MI300X**

Run:

```bash
PATH=/opt/rocm/bin:$PATH .venv/bin/python -m pytest \
  tests/test_phase12_rocm_foundation.py::test_hip_simplify_matches_cpu_for_edge_cases_when_available \
  tests/test_phase12_rocm_foundation.py::test_hip_simplify_randomized_matches_cpu_when_available \
  tests/test_phase12_rocm_foundation.py::test_hip_simplify_campaign4_generic_multiword_pressure_when_available \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/hip/simplify_hip.hip.cpp src/hip/simplify_hip.hip.hpp tests/test_phase12_rocm_foundation.py
git commit -m "perf: parallelize HIP generic simplify"
```

## Task 3: Custom Packed-Key One-Word And Two-Word A/B

**Files:**
- Modify: `src/hip/simplify_hip.hip.cpp`
- Modify: `benchmarks/bench_rocm_kernels.py`
- Test: `python -m pytest tests/test_phase12_rocm_foundation.py -q`

- [ ] **Step 1: Add private strategy selection**

Extend the private strategy enum without exposing a Python API:

```cpp
enum class DuplicateReductionStrategy {
  kRocThrustDefault,
  kCustomPackedKey,
  kGenericParallelReduceByKey,
};
```

Use `FASTPAULI_HIP_BENCH_DUPLICATE_REDUCTION=custom_packed_key` only as a
benchmark/test selector. Production auto-dispatch may retain the custom path
only for the exact key shapes that pass the retention gates.

- [ ] **Step 2: Implement custom one-word and two-word probes**

For one-word `num_qubits <= 32`, keep the packed 64-bit `(x << 32) | z` key.
For one-word `num_qubits > 32`, keep a two-field `HipKey1`. For two-word
operators, keep `HipKey2`. The custom path may replace rocThrust high-level
temporary allocations only if it uses bounded rocPRIM or hipCUB primitives with
explicit temp-storage reporting.

- [ ] **Step 3: Preserve rejection behavior for unavailable variants**

If rocPRIM or hipCUB primitives needed for a candidate path are unavailable on
the host or incompatible with the data layout, the benchmark row must report:

```text
hip_simplify_strategy_status: unavailable
strategy_unavailable_reason: exact compiler, library, or runtime reason
```

Do not silently fall back to rocThrust while labeling the row as custom.

- [ ] **Step 4: Validate one-word and two-word correctness on MI300X**

Run:

```bash
PATH=/opt/rocm/bin:$PATH FASTPAULI_HIP_BENCH_DUPLICATE_REDUCTION=custom_packed_key \
  .venv/bin/python -m pytest \
  tests/test_phase12_rocm_foundation.py::test_hip_simplify_matches_cpu_for_edge_cases_when_available \
  tests/test_phase12_rocm_foundation.py::test_hip_simplify_campaign4_one_and_two_word_regression_when_available \
  -q
```

Expected: all selected tests pass when the custom strategy is available, or the
test lane records an exact unavailable reason without changing production
behavior.

- [ ] **Step 5: Commit**

```bash
git add src/hip/simplify_hip.hip.cpp benchmarks/bench_rocm_kernels.py
git commit -m "perf: probe HIP packed-key simplify"
```

## Task 4: Private Workspace And Scratch Attribution

**Files:**
- Create: `src/hip/workspace_hip.hip.hpp`
- Create: `src/hip/workspace_hip.hip.cpp`
- Modify: `CMakeLists.txt`
- Modify: `src/hip/simplify_hip.hip.cpp`
- Modify: `benchmarks/bench_rocm_kernels.py`
- Test: `python -m pytest tests/test_phase12_rocm_foundation.py -q`

- [ ] **Step 1: Add a private RAII workspace**

Implement a private HIP-only workspace that owns a byte buffer and reports
allocation accounting. It must stay in `src/hip/` and must not appear in public
headers:

```cpp
class HipTemporaryWorkspace {
 public:
  explicit HipTemporaryWorkspace(int device_ordinal);
  ~HipTemporaryWorkspace();
  HipTemporaryWorkspace(HipTemporaryWorkspace&&) noexcept;
  HipTemporaryWorkspace& operator=(HipTemporaryWorkspace&&) noexcept;
  HipTemporaryWorkspace(const HipTemporaryWorkspace&) = delete;
  HipTemporaryWorkspace& operator=(const HipTemporaryWorkspace&) = delete;

  void* reserve(std::size_t bytes, const char* label);
  void release() noexcept;
  std::size_t capacity_bytes() const noexcept;
  std::size_t high_watermark_bytes() const noexcept;
  std::size_t allocation_count() const noexcept;
  std::size_t growth_count() const noexcept;
};
```

- [ ] **Step 2: Wire workspace only into benchmarkable private paths**

Use the workspace for rocPRIM or hipCUB temp storage when those primitives
support explicit scratch buffers. If rocThrust high-level algorithms cannot
reliably use the workspace without unstable allocator hooks, record that as a
`rejected_with_evidence` result rather than adding brittle allocator code.

- [ ] **Step 3: Add benchmark JSON accounting**

Every Campaign 4 row must report:

```text
hip_workspace_mode
hip_workspace_reserved_bytes
hip_workspace_high_watermark_bytes
hip_workspace_allocation_count
hip_workspace_growth_count
```

Rows that do not use a workspace must report `hip_workspace_mode: absent` and
zero counts.

- [ ] **Step 4: Validate no public API leak**

Run:

```bash
rg -n '#[[:space:]]*include[[:space:]]*[<"].*(hip|rocm|rocprim|hipcub|thrust)' include/fastpauli
python -m pytest tests/test_phase12_rocm_foundation.py::test_public_headers_do_not_include_rocm_or_hip_headers -q
```

Expected: no public include leak; the public-header test passes. HIP strings in
`bindings/python` are allowed for private status and build-info reporting and
should be reviewed manually rather than treated as public-header failures.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt src/hip/workspace_hip.hip.hpp src/hip/workspace_hip.hip.cpp src/hip/simplify_hip.hip.cpp benchmarks/bench_rocm_kernels.py
git commit -m "perf: add private HIP simplify workspace"
```

## Task 5: Benchmarks, Renderer, Report, And README Landscape

**Files:**
- Modify: `benchmarks/bench_rocm_kernels.py`
- Create: `scripts/render_rocm_campaign4_assets.py`
- Create: `tests/test_rocm_campaign4_assets.py`
- Create: `docs/benchmarks/reports/rocm_mi300x_campaign4_2026-04-30.md`
- Create: `docs/benchmarks/data/rocm_mi300x_campaign4_2026-04-30/summary.json`
- Create: `docs/benchmarks/plots/rocm_mi300x_campaign4_simplify_hardening.svg`
- Modify: `docs/benchmarks/plots/accelerator_landscape_with_rocm.svg`
- Modify: `README.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/plans/rocm_next_waves_plan.md`
- Test: `python -m pytest tests/test_rocm_campaign4_assets.py -q`

- [ ] **Step 1: Add Campaign 4 profiles**

Add these profiles to `benchmarks/bench_rocm_kernels.py`:

```text
simplify-campaign4-baseline
simplify-campaign4-custom-key-ab
simplify-campaign4-generic-multiword
simplify-campaign4-workspace-ab
simplify-campaign4-profiler
```

The generic multi-word profile must include at least:

```text
130 qubits, 4096 terms, duplicate_rate 0.25
193 qubits, 8192 terms, duplicate_rate 0.25
193 qubits, 32768 terms, duplicate_rate 0.875
257 qubits, 4096 terms, duplicate_rate 0.0625
```

- [ ] **Step 2: Capture MI300X benchmark evidence**

Run on MI300X:

```bash
PATH=/opt/rocm/bin:$PATH FASTPAULI_BENCHMARK_GIT_COMMIT=$(git rev-parse --short HEAD) \
  .venv/bin/python benchmarks/bench_rocm_kernels.py \
  --profile simplify-campaign4-baseline --repeat 5 --warmup 2 --json \
  --output docs/benchmarks/data/rocm_mi300x_campaign4_2026-04-30/raw/rocm_simplify_campaign4_baseline_mi300x.json

PATH=/opt/rocm/bin:$PATH FASTPAULI_BENCHMARK_GIT_COMMIT=$(git rev-parse --short HEAD) \
  .venv/bin/python benchmarks/bench_rocm_kernels.py \
  --profile simplify-campaign4-custom-key-ab --repeat 5 --warmup 2 --json \
  --output docs/benchmarks/data/rocm_mi300x_campaign4_2026-04-30/raw/rocm_simplify_campaign4_custom_key_mi300x.json

PATH=/opt/rocm/bin:$PATH FASTPAULI_BENCHMARK_GIT_COMMIT=$(git rev-parse --short HEAD) \
  .venv/bin/python benchmarks/bench_rocm_kernels.py \
  --profile simplify-campaign4-generic-multiword --repeat 5 --warmup 2 --json \
  --output docs/benchmarks/data/rocm_mi300x_campaign4_2026-04-30/raw/rocm_simplify_campaign4_generic_mi300x.json

PATH=/opt/rocm/bin:$PATH FASTPAULI_BENCHMARK_GIT_COMMIT=$(git rev-parse --short HEAD) \
  .venv/bin/python benchmarks/bench_rocm_kernels.py \
  --profile simplify-campaign4-workspace-ab --repeat 5 --warmup 2 --json \
  --output docs/benchmarks/data/rocm_mi300x_campaign4_2026-04-30/raw/rocm_simplify_campaign4_workspace_mi300x.json
```

- [ ] **Step 3: Capture profiler evidence**

Run:

```bash
PATH=/opt/rocm/bin:$PATH FASTPAULI_BENCHMARK_GIT_COMMIT=$(git rev-parse --short HEAD) \
  rocprof -d docs/benchmarks/data/rocm_mi300x_campaign4_2026-04-30/profiler \
  --hip-trace --stats \
  .venv/bin/python benchmarks/bench_rocm_kernels.py \
  --profile simplify-campaign4-profiler --repeat 1 --warmup 0 --json \
  --output docs/benchmarks/data/rocm_mi300x_campaign4_2026-04-30/raw/rocm_simplify_campaign4_profiler_mi300x.json
```

If requested counters fail, check in stdout/stderr and record the exact
provider/tool limitation in the report.

- [ ] **Step 4: Generate checked assets**

Run:

```bash
python scripts/render_rocm_campaign4_assets.py \
  --data-dir docs/benchmarks/data/rocm_mi300x_campaign4_2026-04-30 \
  --plot-dir docs/benchmarks/plots
```

The renderer test must compare regenerated summary JSON and SVGs against the
checked files after normalizing absolute temp paths, matching the Campaign 3
asset test quality bar.

- [ ] **Step 5: Write the Campaign 4 report**

The report must include:

```text
host and build inventory
exact git revisions for implementation, benchmark capture, report closeout, and final validation
benchmark commands and raw JSON paths
profiler commands and artifact paths
before/after Campaign 3 versus Campaign 4 tables
retained and rejected strategy table
workspace allocation-attribution table
generic multi-word bottleneck explanation
README landscape refresh status
review findings and resolutions
terminal-status table for workspace, custom key, generic multi-word, DLPack, streams, expectation, matmul, portability, ROCm wheels, multi-GPU, and simultaneous CUDA+HIP
remaining headroom with concrete next triggers
```

- [ ] **Step 6: Commit**

```bash
git add benchmarks/bench_rocm_kernels.py scripts/render_rocm_campaign4_assets.py tests/test_rocm_campaign4_assets.py README.md docs/roadmap.md docs/plans/rocm_next_waves_plan.md docs/benchmarks/reports/rocm_mi300x_campaign4_2026-04-30.md docs/benchmarks/data/rocm_mi300x_campaign4_2026-04-30 docs/benchmarks/plots
git commit -m "bench: record ROCm simplify hardening campaign"
```

## Task 6: Validation, Review, Merge, Push, And Cleanup

**Files:**
- Modify as needed based on review findings.
- Test: `python scripts/validate.py`

- [ ] **Step 1: Run local validation**

Run in the normal macOS checkout:

```bash
python scripts/validate.py
git diff --check origin/main..HEAD
```

Expected: validation passes and `git diff --check` prints no output.

- [ ] **Step 2: Run MI300X validation**

Run on MI300X:

```bash
PATH=/opt/rocm/bin:$PATH .venv/bin/python -m pytest tests/test_phase12_rocm_foundation.py -q
PATH=/opt/rocm/bin:$PATH .venv/bin/python -m pytest tests/test_rocm_campaign4_assets.py -q
PATH=/opt/rocm/bin:$PATH .venv/bin/python benchmarks/bench_rocm_kernels.py --profile simplify-campaign4-baseline --repeat 3 --warmup 1 --json
```

Expected: tests pass, benchmark rows report `status: ok`, and correctness
checks remain enabled.

- [ ] **Step 3: Request independent review**

Give the reviewer:

```text
branch name
commit list
Campaign 4 scope and retention gates from this plan
diff against origin/main
local validation summary
MI300X validation, benchmark, and profiler summary
known rejected or unavailable strategy reasons
```

Resolve every P0/P1 finding before merge. Fix or explicitly defer P2 findings
with named follow-up scope.

- [ ] **Step 4: Merge and validate main**

Run:

```bash
git switch main
git merge --ff-only codex/rocm-campaign4
python scripts/validate.py
```

- [ ] **Step 5: Push, confirm CI, and clean up**

Run:

```bash
git push origin main
gh run list --branch main --limit 5
gh run watch <run-id>
git branch -d codex/rocm-campaign4
```

If a stale remote campaign branch exists, delete it only after `main` is pushed
and CI is green.

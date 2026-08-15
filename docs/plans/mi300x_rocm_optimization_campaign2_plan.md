# MI300X ROCm Optimization Campaign 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and benchmark the first HIP device-resident commutation output path on MI300X so FastPauli can avoid dense host materialization when callers need compact commutation summaries.

**Architecture:** This campaign mirrors the existing CUDA `DeviceCommutationMatrix` lifetime model for HIP-only source builds while keeping CUDA and HIP mutually exclusive. HIP `DeviceCommutationMatrix` owns a dense `uint8` row-major device buffer through RAII, supports host materialization only through `to_host()`, and supports compact count/conflict reductions that copy only `uint64` summaries to the host. HIP DLPack, public streams, public workspaces, HIP simplify, HIP expectation, HIP matmul, and ROCm wheels stay out of scope.

**Tech Stack:** C++20, nanobind, CMake HIP language support, ROCm/HIP runtime, AMD Instinct MI300X `gfx942`, rocprof, pytest, NumPy, `benchmarks/bench_rocm_kernels.py`, existing CUDA device-output API as the semantic reference.

---

## Current Baseline

The completed bring-up report is:

```text
docs/benchmarks/reports/rocm_mi300x_bringup_2026-04-29.md
```

The current HIP path supports:

```text
PauliSum.to_device(device=0)
DevicePauliSum.to_host()
DevicePauliSum.backend == "hip"
DevicePauliSum.commutes_with(other) returning host-visible dense bool results
DevicePauliSum.commutes_with_into(other, host_output)
```

The current HIP path deliberately rejects:

```text
DevicePauliSum.commutes_with_device(other)
DevicePauliSum.commutes_with_device(other, output=existing_matrix)
HIP DLPack or ROCm Python array interop
```

## Scope

In scope:

```text
HIP-backed DeviceCommutationMatrix allocation and RAII cleanup
HIP DeviceCommutationMatrix.empty(shape, device=0)
HIP DeviceCommutationMatrix.to_host()
HIP DeviceCommutationMatrix.count_commuting(axis=None|0|1)
HIP DeviceCommutationMatrix.conflict_degrees(axis=None|0|1)
HIP DevicePauliSum.commutes_with_device(other)
HIP DevicePauliSum.commutes_with_device(other, output=existing_matrix)
MI300X benchmark rows for host-output, device-output allocating, device-output reused, compact counts, compact conflicts, and dense to_host boundaries
rocprof trace/stats/counter evidence for the retained HIP kernels
report, roadmap, README, and benchmark protocol updates tied to measured evidence
```

Out of scope:

```text
HIP DLPack, __dlpack_device__, or ROCm array-interface exposure
public HIP stream parameters
public HIP workspace handles
HIP simplify, expectation, or matmul kernels
multi-GPU MI300X execution
ROCm binary wheels
simultaneous CUDA+HIP source builds
```

## Evidence Layout

Use this date-stamped evidence root for Campaign 2:

```text
docs/benchmarks/data/rocm_mi300x_campaign2_2026-04-30/
docs/benchmarks/data/rocm_mi300x_campaign2_2026-04-30/logs/
docs/benchmarks/data/rocm_mi300x_campaign2_2026-04-30/raw/
docs/benchmarks/data/rocm_mi300x_campaign2_2026-04-30/profiler/
docs/benchmarks/data/rocm_mi300x_campaign2_2026-04-30/summary.json
docs/benchmarks/reports/rocm_mi300x_campaign2_2026-04-30.md
```

If execution continues on a later UTC date, record that date explicitly in the
report and explain why the Campaign 2 evidence root remains named
`rocm_mi300x_campaign2_2026-04-30`. The report must state the exact execution date, git revisions compared,
commands, environment, benchmark JSON paths, profiler paths, review findings,
and retained/rejected decisions.

## Acceptance Criteria

The campaign is complete only when every applicable item has a terminal status
in the report:

```text
CPU-only local validation passes with FASTPAULI_ENABLE_HIP=OFF
HIP source build succeeds on MI300X with FASTPAULI_ENABLE_HIP=ON and FASTPAULI_HIP_ARCHITECTURES=gfx942
CUDA+HIP configure-time rejection still passes
public headers still contain no HIP or ROCm runtime headers
HIP DeviceCommutationMatrix.empty() allocates the requested shape on the requested device
HIP DevicePauliSum.commutes_with_device() matches CPU commutation for empty, scalar, vector, matrix, one-word, multi-word, and randomized cases
HIP reused-output commutes_with_device(..., output=matrix) preserves object identity and rejects wrong shape, wrong device when at least two HIP devices are visible, and entry-limit violations
HIP count_commuting(axis=None|0|1) matches NumPy sums over to_host()
HIP conflict_degrees(axis=None|0|1) matches NumPy sums over logical-not to_host()
HIP DLPack and CUDA Array Interface claims remain unavailable or rejected with explicit HIP-specific errors
benchmark JSON separates host-output, device-output allocating, device-output reused, compact count, compact conflict, and dense to_host timings
rocprof trace/stats/counter evidence exists for the retained HIP fill and reduction kernels, or an exact tool/provider diagnosis is checked in
README and roadmap state only evidence-backed ROCm claims
independent review is recorded before merge
```

## Task 1: Contract And Failing Tests

**Files:**
- Modify: `docs/architecture/rocm_backend.md`
- Modify: `docs/benchmarks/protocol.md`
- Modify: `tests/test_phase12_rocm_foundation.py`
- Test: `python -m pytest tests/test_phase12_rocm_foundation.py -q`

- [x] **Step 1: Update the ROCm backend contract**

Add a section to `docs/architecture/rocm_backend.md` named
`Post-Bring-Up Public Boundaries` with this contract:

```text
The first accepted post-bring-up public HIP expansion is device-resident dense commutation output.
HIP DeviceCommutationMatrix must match the CUDA shape, dtype, device, to_host, count_commuting, and conflict_degrees semantics where the backend can support them.
HIP DeviceCommutationMatrix must not expose CUDA Array Interface semantics.
HIP DLPack remains unavailable until a separate HIP DLPack contract accepts kDLROCM device typing, ownership, stream, read-only, and consumer compatibility rules.
```

- [x] **Step 2: Add HIP timing fields to the benchmark protocol**

Add a ROCm Campaign 2 subsection to `docs/benchmarks/protocol.md` requiring
these JSON fields when available:

```text
hip_device_output_allocate_seconds and p10/p90/min/max variants
hip_device_output_reuse_seconds and p10/p90/min/max variants
hip_device_output_to_host_seconds and p10/p90/min/max variants
hip_count_commuting_axis_none_seconds and p10/p90/min/max variants
hip_count_commuting_axis_0_seconds and p10/p90/min/max variants
hip_count_commuting_axis_1_seconds and p10/p90/min/max variants
hip_conflict_degrees_axis_none_seconds and p10/p90/min/max variants
hip_conflict_degrees_axis_0_seconds and p10/p90/min/max variants
hip_conflict_degrees_axis_1_seconds and p10/p90/min/max variants
result_materialization_target: host_bool_matrix, device_uint8_matrix, compact_uint64_counts, compact_uint64_conflicts, or compact_uint64_counts_and_conflicts
result_materialization_targets: list containing every retained materialization target when one row reports multiple compact consumers
timing_boundary: transfer_inclusive, device_operand_host_output, device_output_allocating, device_output_reused, device_output_to_host, or compact_consumer
```

- [x] **Step 3: Add failing HIP device-output tests**

Append these tests to `tests/test_phase12_rocm_foundation.py`:

```python
def test_hip_device_commutation_matrix_matches_cpu_when_available() -> None:
    _require_hip_runtime()

    lhs = fastpauli.PauliSum.from_labels(["XX", "ZI", "IZ"], [1.0, 2.0, -1.0])
    rhs = fastpauli.PauliSum.from_labels(["YY", "XI"], [1.0, 1.0j])
    lhs_device = lhs.to_device()
    rhs_device = rhs.to_device()

    matrix = lhs_device.commutes_with_device(rhs_device)

    assert isinstance(matrix, fastpauli.DeviceCommutationMatrix)
    assert matrix.shape == (lhs.num_terms, rhs.num_terms)
    assert matrix.device == lhs_device.device
    assert matrix.dtype == "uint8"
    assert matrix.num_entries == lhs.num_terms * rhs.num_terms
    np.testing.assert_array_equal(matrix.to_host(), lhs.commutes_with(rhs))


def test_hip_device_commutation_matrix_counts_match_numpy_when_available() -> None:
    _require_hip_runtime()

    lhs = fastpauli.PauliSum.from_labels(["XI", "ZI"], [1.0, 1.0]).to_device()
    rhs = fastpauli.PauliSum.from_labels(["IX", "ZZ", "XX"], [1.0, 1.0, 1.0]).to_device()
    matrix = lhs.commutes_with_device(rhs)
    host = matrix.to_host().astype(np.uint64)

    assert matrix.count_commuting() == int(host.sum())
    np.testing.assert_array_equal(matrix.count_commuting(axis=1), host.sum(axis=1, dtype=np.uint64))
    np.testing.assert_array_equal(matrix.count_commuting(axis=0), host.sum(axis=0, dtype=np.uint64))

    host_conflicts = np.logical_not(host.astype(np.bool_)).astype(np.uint64)
    assert matrix.conflict_degrees() == int(host_conflicts.sum())
    np.testing.assert_array_equal(matrix.conflict_degrees(axis=1), host_conflicts.sum(axis=1, dtype=np.uint64))
    np.testing.assert_array_equal(matrix.conflict_degrees(axis=0), host_conflicts.sum(axis=0, dtype=np.uint64))


def test_hip_device_commutation_matrix_reuse_and_guardrails_when_available() -> None:
    status = _require_hip_runtime()

    lhs = fastpauli.PauliSum.from_labels(["XI", "ZI"], [1.0, 1.0]).to_device()
    rhs = fastpauli.PauliSum.from_labels(["IX", "ZZ", "XX"], [1.0, 1.0, 1.0]).to_device()
    output = fastpauli.DeviceCommutationMatrix.empty((lhs.num_terms, rhs.num_terms), device=lhs.device)

    same = lhs.commutes_with_device(rhs, output=output)

    assert same is output
    np.testing.assert_array_equal(output.to_host(), lhs.to_host().commutes_with(rhs.to_host()))

    wrong_shape = fastpauli.DeviceCommutationMatrix.empty((1, rhs.num_terms), device=lhs.device)
    with pytest.raises(ValueError, match="output shape"):
        lhs.commutes_with_device(rhs, output=wrong_shape)

    with pytest.raises(ValueError, match="commutation matrix entry count exceeds"):
        lhs.commutes_with_device(rhs, max_commutation_matrix_entries=3)

    if status["device_count"] >= 2:
        wrong_device = fastpauli.DeviceCommutationMatrix.empty(
            (lhs.num_terms, rhs.num_terms),
            device=1 if lhs.device == 0 else 0,
        )
        with pytest.raises(ValueError, match="same device"):
            lhs.commutes_with_device(rhs, output=wrong_device)


def test_hip_interop_surfaces_remain_explicitly_unavailable_when_available() -> None:
    _require_hip_runtime()

    matrix = fastpauli.PauliSum.from_labels(["XI"], [1.0]).to_device().commutes_with_device(
        fastpauli.PauliSum.from_labels(["IX"], [1.0]).to_device()
    )

    with pytest.raises((BufferError, RuntimeError, ValueError), match="HIP|ROCm|CUDA"):
        matrix.__cuda_array_interface__
    with pytest.raises((BufferError, RuntimeError, ValueError), match="HIP|ROCm|DLPack"):
        matrix.__dlpack__(max_version=(1, 0))
    with pytest.raises((BufferError, RuntimeError, ValueError), match="HIP|ROCm|DLPack"):
        matrix.__dlpack_device__()
```

- [x] **Step 4: Run the tests and confirm the expected failure**

Run on the MI300X HIP build:

```bash
PATH=/opt/rocm/bin:$PATH python -m pytest tests/test_phase12_rocm_foundation.py -q
```

Expected before implementation: the new device-output tests fail with
`HIP commutes_with_device is not implemented yet`, while existing HIP
foundation tests still pass.

- [x] **Step 5: Commit**

```bash
git add docs/architecture/rocm_backend.md docs/benchmarks/protocol.md tests/test_phase12_rocm_foundation.py
git commit -m "test: specify HIP device commutation outputs"
```

## Task 2: HIP DeviceCommutationMatrix RAII Implementation

**Files:**
- Create: `src/hip/device_commutation_matrix.hip.hpp`
- Create: `src/hip/device_commutation_matrix.hip.cpp`
- Modify: `CMakeLists.txt`
- Modify: `scripts/validate.py`
- Test: `python -m pytest tests/test_phase12_rocm_foundation.py::test_public_headers_do_not_include_rocm_or_hip_headers -q`

- [x] **Step 1: Add HIP source paths to the build**

Add the HIP matrix files to the `FASTPAULI_ENABLE_HIP` source list in
`CMakeLists.txt`, and add these paths to `HIP_FOUNDATION_SOURCES` in
`scripts/validate.py`:

```text
src/hip/device_commutation_matrix.hip.cpp
src/hip/device_commutation_matrix.hip.hpp
```

- [x] **Step 2: Implement the HIP PImpl**

Create `src/hip/device_commutation_matrix.hip.hpp` with a `DeviceCommutationMatrix::Impl`
that owns:

```text
std::size_t rows
std::size_t cols
int device_ordinal
std::uint8_t* data
```

The destructor must set the HIP device before `hipFree(data)` and must ignore
destructor-time HIP errors. Allocation failures must be checked at creation
time through the existing HIP error helper pattern in `src/hip/device_pauli_sum.hip.cpp`.

- [x] **Step 3: Implement allocation, metadata, and host copy**

In `src/hip/device_commutation_matrix.hip.cpp`, implement:

```text
DeviceCommutationMatrix::empty(rows, cols, device)
DeviceCommutationMatrix::to_host()
DeviceCommutationMatrix::rows()
DeviceCommutationMatrix::cols()
DeviceCommutationMatrix::num_entries()
DeviceCommutationMatrix::device()
DeviceCommutationMatrix::mutable_data_for_device_write()
DeviceCommutationMatrix::data_pointer_for_cuda_array_interface()
```

`data_pointer_for_cuda_array_interface()` must throw an error explaining that
CUDA Array Interface is not available for HIP-backed matrices. It must not
return a HIP pointer under a CUDA interface name.

- [x] **Step 4: Validate the public header rule locally**

Run:

```bash
python -m pytest tests/test_phase12_rocm_foundation.py::test_public_headers_do_not_include_rocm_or_hip_headers -q
```

Expected: pass.

- [x] **Step 5: Commit**

```bash
git add CMakeLists.txt scripts/validate.py src/hip/device_commutation_matrix.hip.cpp src/hip/device_commutation_matrix.hip.hpp
git commit -m "feat: add HIP device commutation matrix storage"
```

## Task 3: HIP Device-Output Commutation Fill

**Files:**
- Modify: `src/hip/commutation_hip.hip.cpp`
- Test: `python -m pytest tests/test_phase12_rocm_foundation.py -q`

- [x] **Step 1: Replace the HIP commutes_with_device stubs**

Implement `DevicePauliSum::commutes_with_device()` by:

```text
checking moved-from state for lhs and rhs
checking both operands use the same HIP device
checking num_qubits equality
checking max_commutation_matrix_entries before allocation
allocating DeviceCommutationMatrix::empty(lhs.num_terms(), rhs.num_terms(), lhs.device())
calling commutes_with_device_into()
returning the matrix
```

Implement `DevicePauliSum::commutes_with_device_into()` by:

```text
checking moved-from state for lhs, rhs, and output
checking lhs, rhs, and output are on the same HIP device
checking output shape equals lhs.num_terms() by rhs.num_terms()
checking max_commutation_matrix_entries before launch
launching the existing HIP commutation kernel into output.mutable_data_for_device_write()
synchronizing and translating HIP errors with messages naming HIP commutes_with_device
```

- [x] **Step 2: Run targeted HIP tests**

Run on the MI300X HIP build:

```bash
PATH=/opt/rocm/bin:$PATH python -m pytest tests/test_phase12_rocm_foundation.py -q
```

Expected: all HIP foundation and device-output tests pass or skip only for
documented unavailable multi-device cases.

- [x] **Step 3: Commit**

```bash
git add src/hip/commutation_hip.hip.cpp tests/test_phase12_rocm_foundation.py
git commit -m "feat: add HIP device commutation outputs"
```

## Task 4: HIP Compact Consumers

**Files:**
- Modify: `src/hip/device_commutation_matrix.hip.cpp`
- Test: `python -m pytest tests/test_phase12_rocm_foundation.py -q`

- [x] **Step 1: Implement total count reductions**

Add HIP kernels equivalent to the CUDA count kernels:

```text
count_total_kernel(data, entries, partials)
count_rows_kernel(data, rows, cols, output)
count_cols_kernel(data, rows, cols, output)
```

Use `__shared__ std::uint64_t scratch[256]` with block-local reduction, one
block per row or column for axis reductions, and a second host-side sum over
partials for the total until profiler evidence justifies a fully device-side
second-stage reduction.

- [x] **Step 2: Wire public methods**

Implement:

```text
DeviceCommutationMatrix::count_commuting()
DeviceCommutationMatrix::count_commuting_rows()
DeviceCommutationMatrix::count_commuting_cols()
```

`conflict_degrees()` uses `count_commuting()` plus matrix shape in the Python
binding, so Campaign 2 does not need separate conflict kernels. Keep that
contract unless profiler evidence shows the complement arithmetic or host
summary copy is material in the measured compact-consumer rows.

- [x] **Step 3: Run targeted tests**

Run on the MI300X HIP build:

```bash
PATH=/opt/rocm/bin:$PATH python -m pytest tests/test_phase12_rocm_foundation.py -q
```

Expected: HIP compact count and conflict tests pass.

- [x] **Step 4: Commit**

```bash
git add src/hip/device_commutation_matrix.hip.cpp bindings/python/pauli_sum_py.cpp tests/test_phase12_rocm_foundation.py
git commit -m "feat: add HIP compact commutation consumers"
```

## Task 5: Benchmark And Profiler Expansion

**Files:**
- Modify: `benchmarks/bench_rocm_kernels.py`
- Create: `scripts/render_rocm_campaign2_assets.py`
- Test: `python benchmarks/bench_rocm_kernels.py --smoke --repeat 1 --warmup 0 --json`

- [x] **Step 1: Add Campaign 2 benchmark modes**

Extend `benchmarks/bench_rocm_kernels.py` with profiles:

```text
commutation-device-output-smoke
commutation-device-output-scaling
commutation-compact-consumers
commutation-campaign2-profiler
```

Each row must record:

```text
backend: hip
device_name
gfx_target
num_qubits
lhs_terms
rhs_terms
entries
result_materialization_target
timing_boundary
median seconds and repeat distribution
correctness_digest with CPU-equivalent count
```

- [x] **Step 2: Add timing boundaries**

Add timing for:

```text
DevicePauliSum.commutes_with() host-output device operands
DevicePauliSum.commutes_with_device() allocating
DevicePauliSum.commutes_with_device(..., output=matrix) reused
DeviceCommutationMatrix.to_host()
DeviceCommutationMatrix.count_commuting(axis=None|0|1)
DeviceCommutationMatrix.conflict_degrees(axis=None|0|1)
```

- [x] **Step 3: Run benchmark smoke locally**

Run:

```bash
python benchmarks/bench_rocm_kernels.py --smoke --repeat 1 --warmup 0 --json
```

Expected on CPU-only local builds: the report emits HIP unavailable status and
does not fail validation.

- [x] **Step 4: Run benchmark profiles on MI300X**

Run on the MI300X HIP build:

```bash
PATH=/opt/rocm/bin:$PATH python benchmarks/bench_rocm_kernels.py --profile commutation-device-output-scaling --repeat 5 --warmup 2 --json --output docs/benchmarks/data/rocm_mi300x_campaign2_2026-04-30/raw/rocm_commutation_device_output_scaling_mi300x.json
PATH=/opt/rocm/bin:$PATH python benchmarks/bench_rocm_kernels.py --profile commutation-compact-consumers --repeat 5 --warmup 2 --json --output docs/benchmarks/data/rocm_mi300x_campaign2_2026-04-30/raw/rocm_commutation_compact_consumers_mi300x.json
```

Expected: both JSON files contain correctness-checked HIP rows and CPU
comparison rows for the same deterministic datasets.

- [x] **Step 5: Capture rocprof evidence**

Run:

```bash
PATH=/opt/rocm/bin:$PATH rocprof -d docs/benchmarks/data/rocm_mi300x_campaign2_2026-04-30/profiler --hip-trace --stats python benchmarks/bench_rocm_kernels.py --profile commutation-campaign2-profiler --repeat 1 --warmup 0 --json --output docs/benchmarks/data/rocm_mi300x_campaign2_2026-04-30/raw/rocm_commutation_campaign2_profiler_mi300x.json
```

Then capture counters with a checked `rocprof` input file under the profiler
directory. The report must include the exact input file contents and the
counter CSV path.

- [x] **Step 6: Commit**

```bash
git add benchmarks/bench_rocm_kernels.py scripts/render_rocm_campaign2_assets.py
git commit -m "bench: add ROCm device-output benchmarks"
```

## Task 6: Report, Roadmap, Review, And Closeout

**Files:**
- Create: `docs/benchmarks/reports/rocm_mi300x_campaign2_2026-04-30.md`
- Create: `docs/benchmarks/data/rocm_mi300x_campaign2_2026-04-30/summary.json`
- Modify: `README.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/quality/phase_quality_gates.md` if the wave adds a recurring gate
- Test: `python scripts/validate.py`

- [x] **Step 1: Write the report**

The report must include:

```text
scope and non-goals
host and build inventory
implementation outcome table
validation commands and outcomes
benchmark command table
benchmark result table with host-output, allocating device-output, reused device-output, compact count, compact conflict, and to_host rows
rocprof trace and counter interpretation
retained decisions
rejected or deferred decisions
review findings and resolutions
remaining ROCm headroom
release claim and rejected claims
```

- [x] **Step 2: Update roadmap and README**

Only state completed evidence after the report artifacts exist. The README
performance section must continue to show the broad landscape view and must not
replace it with a narrow ROCm-only plot.

- [x] **Step 3: Run local validation on merged candidate**

Run locally with HIP disabled:

```bash
python scripts/validate.py
```

Expected: pass.

- [x] **Step 4: Run MI300X validation**

Run on the MI300X HIP build:

```bash
PATH=/opt/rocm/bin:$PATH python -m pytest
PATH=/opt/rocm/bin:$PATH python -m pytest tests/test_phase12_rocm_foundation.py tests/test_phase6_commutation_grouping.py -q
PATH=/opt/rocm/bin:$PATH python benchmarks/bench_rocm_kernels.py --smoke --repeat 3 --warmup 1 --json
```

Expected: pytest passes, with only explicitly documented unavailable-hardware
skips, and benchmark smoke passes with correctness checks enabled.

- [x] **Step 5: Complete independent review**

Request independent review covering:

```text
HIP correctness and lifetime
CUDA regression risk
public API wording and docstrings
benchmark timing-boundary honesty
profiler evidence freshness
release-claim wording
```

Resolve blocking findings, rerun affected validation, and record residual risk
in the report.

- [ ] **Step 6: Commit**

This operational closeout step remains unchecked in the committed plan snapshot
until the commit actually exists. The final closeout response records the
commit evidence.

```bash
git add README.md docs/roadmap.md docs/quality/phase_quality_gates.md docs/benchmarks/reports/rocm_mi300x_campaign2_2026-04-30.md docs/benchmarks/data/rocm_mi300x_campaign2_2026-04-30
git commit -m "docs: report ROCm MI300X campaign 2"
```

- [ ] **Step 7: Merge, validate, push, and confirm CI**

This operational closeout step remains unchecked in the committed plan snapshot
until the merge, merged-main validation, push, CI confirmation, and branch
cleanup are actually complete. The final closeout response records that
evidence.

Follow the repository closeout flow:

```bash
git switch main
git merge --ff-only codex/rocm-mi300x-campaign2
python scripts/validate.py
git push origin main
gh run watch <run-id> --exit-status
git branch -d codex/rocm-mi300x-campaign2
```

Expected: merged `main` validates locally, pushed CI is green, and the feature
branch is deleted after merge.

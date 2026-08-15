# H100 CUDA Stream And Consumer Campaign 6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the five remaining-headroom items from Campaign 5 into a measured H100 campaign that designs stream/async semantics, adds real GPU-resident consumers for `DeviceCommutationMatrix`, evaluates bit-packed output only against those consumers, adds CuPy interop benchmarks, and keeps README performance visuals broad.

**Architecture:** Campaign 6 is a boundary-and-consumer campaign, not a raw kernel rewrite. Existing synchronous CUDA APIs remain the compatibility baseline while a separate async/stream review documents candidate semantics before any public stream surface is retained. Device-output work must move beyond isolated fill timing by adding benchmarked GPU consumers that operate on `DeviceCommutationMatrix` without materializing the full dense matrix on the host.

**Tech Stack:** C++20, CUDA C++ 12.x, nanobind, CUDA Array Interface v3, CuPy, NumPy, pytest, `bench_cuda_scaling.py`, `bench_cuda_kernels.py`, Nsight Systems, Nsight Compute, Compute Sanitizer, H100 source builds with `FASTPAULI_CUDA_ARCHITECTURES=90`.

---

## Status

Status: completed on H100. Campaign 6 retained compact
`DeviceCommutationMatrix.count_commuting(axis=None|0|1)` consumers, added CuPy
CUDA-array-interface consumer benchmarks, documented stream/async and
bit-packed deferrals, refreshed the broad README landscape, and published the
checked report at
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign6_2026-04-29.md`.

Campaign 5 source-of-truth evidence:

```text
plan: docs/plans/h100_deep_optimization_campaign5_plan.md
report: docs/benchmarks/reports/cuda_deep_optimization_h100_campaign5_2026-04-29.md
data: docs/benchmarks/data/cuda_deep_optimization_h100_campaign5_2026-04-29/
retained public CUDA API changes: DeviceCommutationMatrix and DevicePauliSum.commutes_with_device()
```

Campaign 5 remaining-headroom items that Campaign 6 must cover:

```text
1. Design an async/stream API plan with explicit event ownership, stream capture behavior, host synchronization semantics, and Python lifetime rules.
2. Add downstream GPU consumers for DeviceCommutationMatrix so benchmarks can measure end-to-end GPU-resident workflows instead of isolated output fill.
3. Revisit bit-packed output only if a real downstream consumer needs reduced memory bandwidth or capacity and can accept a documented packed layout.
4. Add CuPy consumer benchmarks through the CUDA Array Interface once the interop test dependency is available in CI or an H100 validation tier.
5. Continue keeping README plots broad; specialized device-output plots belong in reports unless they are integrated into the full CPU/CUDA/external view.
```

## Source Inputs

Read these files before implementation:

```text
README.md
docs/roadmap.md
docs/plans/cuda_deep_optimization_plan.md
docs/plans/h100_deep_optimization_campaign5_plan.md
docs/plans/cuda_commutation_device_output_api_review.md
docs/benchmarks/reports/cuda_deep_optimization_h100_campaign5_2026-04-29.md
docs/architecture/cuda_backend.md
docs/architecture/api_stability.md
docs/architecture/hardware_targets_and_testing.md
docs/architecture/testing_and_ci.md
docs/benchmarks/protocol.md
docs/quality/code_review.md
docs/quality/code_standards.md
docs/quality/documentation_standards.md
docs/user/performance.md
include/fastpauli/device_commutation_matrix.hpp
include/fastpauli/device_pauli_sum.hpp
src/cuda/commutation_cuda.cu
src/cuda/device_commutation_matrix.cu
src/cuda/device_commutation_matrix.cuh
bindings/python/pauli_sum_py.cpp
benchmarks/bench_cuda_scaling.py
benchmarks/bench_cuda_kernels.py
benchmarks/bench_competitive_baselines.py
tests/test_phase11_cuda_kernels.py
tests/test_cuda_scaling_benchmark.py
tests/test_cuda_deep_report_assets.py
```

## Scope

In scope:

```text
async/stream API review document with retained, rejected, and deferred semantics
private stream/event benchmark helpers only after the review document accepts their lifetime model
GPU-resident DeviceCommutationMatrix consumers that copy compact summaries to host instead of the full dense matrix
dense uint8 consumer pipeline benchmarks across default, stress, and extreme H100 scales
bit-packed commutation consumer prototype only as a benchmark-only comparison against the dense consumer pipeline
CuPy consumer benchmarks that read DeviceCommutationMatrix through __cuda_array_interface__
H100 correctness tests, Compute Sanitizer, Nsight Systems, Nsight Compute, and same-boundary A/B evidence
Campaign 6 checked report, raw data, generated plots, and README broad landscape refresh when evidence changes
```

Out of scope unless the Campaign 6 API review explicitly retains the public surface:

```text
public async methods
public stream-handle arguments
public event classes
public bit-packed commutation output
raw device pointer APIs
CUDA wheel release claims
non-H100 NVIDIA performance claims
HIP/AMD, Metal/MPS, or Apple GPU implementation
raw PTX or inline PTX without Nsight and SASS evidence for a specific compiler-codegen limit
```

## File Structure

Planned files for the implementation slice:

```text
docs/plans/cuda_async_stream_api_review.md
  Stream/async candidate contract, accepted private prototype rules, public deferrals, and Python lifetime rules.

docs/plans/cuda_commutation_consumer_api_review.md
  Compact-summary consumer API decision, benchmark-only fallback, bit-packed layout gate, and public deferrals.

docs/architecture/cuda_backend.md
  Updated optimization-boundary status for stream/async prototypes, GPU-resident consumers, and bit-packed output policy.

include/fastpauli/device_commutation_matrix.hpp
  Public declarations only if a compact summary consumer is retained as public API. Otherwise unchanged.

src/cuda/device_commutation_matrix.cu
src/cuda/device_commutation_matrix.cuh
  Dense matrix consumer kernels and optional compact summary helpers.

src/cuda/commutation_cuda.cu
  Benchmark-only bit-packed consumer path and stream/event private probes if accepted by the review.

bindings/python/pauli_sum_py.cpp
  Python bindings for retained public compact-summary methods only; benchmark-only helpers must remain private or benchmark-script-local.

benchmarks/bench_cuda_scaling.py
  Add Campaign 6 profiles for dense consumer, bit-packed consumer prototype, stream/event prototype, and CuPy interop consumer timings.

benchmarks/bench_cuda_kernels.py
  Add reusable timing helpers and instrumentation schema fields for Campaign 6 boundaries.

benchmarks/bench_competitive_baselines.py
  Add CuPy consumer rows when CuPy is importable and the CUDA Array Interface consumer benchmark is semantically comparable.

scripts/render_cuda_campaign6_assets.py
  Generate Campaign 6 summary JSON and report plots from checked raw data.

tests/test_phase11_cuda_kernels.py
  CUDA correctness tests for retained matrix-consumer behavior and CPU-only error behavior.

tests/test_cuda_scaling_benchmark.py
  Non-CUDA tests proving Campaign 6 profiles and schema are present without requiring a GPU.

tests/test_cuda_deep_report_assets.py
  Renderer freshness tests for Campaign 6 checked summary and plots.

docs/benchmarks/data/cuda_deep_optimization_h100_campaign6_YYYY-MM-DD/
docs/benchmarks/reports/cuda_deep_optimization_h100_campaign6_YYYY-MM-DD.md
docs/benchmarks/plots/cuda_h100_campaign6_*.svg
  Final evidence bundle after H100 execution.

README.md
docs/roadmap.md
docs/plans/cuda_deep_optimization_plan.md
docs/user/performance.md
  Source-of-truth links, next-slice status, and broad performance visual policy.
```

## Public API Decision Gate

Campaign 6 starts with two written API reviews. The default decision is conservative:

```text
existing public CUDA APIs remain default-stream and synchronize-before-return
private stream/event probes may be benchmarked only with explicit labels
public stream handles remain deferred unless the API review accepts exact semantics
public async return objects remain deferred unless the API review accepts exact semantics
compact-summary consumer methods remain benchmark-only unless the consumer review accepts exact method signatures, return types, synchronization, and CPU-only behavior
bit-packed output remains benchmark-only unless the consumer benchmarks prove dense uint8 is the limiting capacity or bandwidth cost
```

Required decision artifacts:

```text
docs/plans/cuda_async_stream_api_review.md
docs/plans/cuda_commutation_consumer_api_review.md
```

If a public compact-summary consumer is retained, prefer this narrow API over raw pointers:

```python
matrix = lhs_device.commutes_with_device(rhs_device)
row_counts = matrix.count_commuting(axis=1)
col_counts = matrix.count_commuting(axis=0)
total = matrix.count_commuting()
```

Required semantics for a retained compact-summary API:

```text
axis=None returns a Python int count of entries with value 1
axis=0 returns a NumPy uint64 vector of length matrix.cols
axis=1 returns a NumPy uint64 vector of length matrix.rows
all reductions execute on the matrix CUDA device
only the compact count result is copied to host
the method synchronizes before returning, matching existing public CUDA semantics
moved-from matrices raise RuntimeError
unsupported axis values raise ValueError
CPU-only builds raise the existing CUDA rebuild-guidance RuntimeError
```

If `docs/plans/cuda_commutation_consumer_api_review.md` rejects public compact
summaries, implement the same reductions as benchmark-only helpers and keep
them out of `python/fastpauli/__init__.py`. If that review rejects public
bit-packed output, bit-packed rows must remain benchmark-only and report-only
even when they outperform dense `uint8` in a consumer benchmark.

## Task 0: Branch, H100 Setup, And Baseline Capture

**Files:**
- Read: `README.md`
- Read: `docs/roadmap.md`
- Read: `docs/plans/h100_deep_optimization_campaign6_plan.md`
- Create during execution: `docs/benchmarks/data/cuda_deep_optimization_h100_campaign6_YYYY-MM-DD/metadata/`

- [x] **Step 0.1: Create the implementation branch**

Run:

```bash
git switch main
git pull --ff-only origin main
git switch -c codex/h100-campaign6
```

Expected: the branch starts from the current pushed `main`.

- [x] **Step 0.2: Prepare the H100 source-build checkout**

Run on the H100 host:

```bash
python -m venv .venv
.venv/bin/python -m pip install -U pip wheel build
FASTPAULI_CUDA_ARCHITECTURES=90 FASTPAULI_ENABLE_CUDA=ON .venv/bin/python -m pip install -e ".[test,qiskit,openfermion]"
.venv/bin/python -m pip install cupy-cuda12x cuquantum-python-cu12 qiskit-aer
```

Expected: `import fastpauli`, `import cupy`, and `fastpauli._fastpauli_core._build_info()["cuda_enabled"]` all succeed.

- [x] **Step 0.3: Capture baseline metadata**

Run on the H100 host:

```bash
mkdir -p docs/benchmarks/data/cuda_deep_optimization_h100_campaign6_YYYY-MM-DD/metadata
git rev-parse HEAD > docs/benchmarks/data/cuda_deep_optimization_h100_campaign6_YYYY-MM-DD/metadata/experiment-revision.txt
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv > docs/benchmarks/data/cuda_deep_optimization_h100_campaign6_YYYY-MM-DD/metadata/gpu.csv
/usr/local/cuda/bin/nvcc --version > docs/benchmarks/data/cuda_deep_optimization_h100_campaign6_YYYY-MM-DD/metadata/nvcc-version.txt
```

Expected: metadata records the H100 device, CUDA toolkit, and experiment revision.

## Task 1: Async, Stream, And Consumer API Reviews

**Files:**
- Create: `docs/plans/cuda_async_stream_api_review.md`
- Create: `docs/plans/cuda_commutation_consumer_api_review.md`
- Modify: `docs/architecture/cuda_backend.md`
- Modify: `docs/architecture/api_stability.md`
- Modify: `docs/user/performance.md`

- [x] **Step 1.1: Write the async/stream review**

Create `docs/plans/cuda_async_stream_api_review.md` with these required sections:

```text
# CUDA Async And Stream API Review
Status: review required before public API
Existing invariant: public CUDA methods synchronize before returning
Candidate stream handle forms
Candidate event ownership model
Stream capture behavior
Host synchronization semantics
Python object lifetime rules
Exception and error propagation
Benchmark-only private prototype rules
Retained public surfaces
Rejected public surfaces
Deferred public surfaces
```

Expected: the document explicitly says whether Campaign 6 retains no public stream API, a narrow public stream API, or benchmark-only private probes.

- [x] **Step 1.2: Write the commutation consumer API review**

Create `docs/plans/cuda_commutation_consumer_api_review.md` with these required sections:

```text
# CUDA Commutation Consumer API Review
Status: review required before public compact-summary or bit-packed API
Existing invariant: DeviceCommutationMatrix owns dense row-major uint8 flags
Compact-summary candidate methods
Compact-summary return types
Compact-summary synchronization semantics
CPU-only behavior
Benchmark-only fallback rules
Bit-packed layout candidate
Bit-packed public API rejection or retention criteria
CuPy consumer boundary
Retained public surfaces
Rejected public surfaces
Deferred public surfaces
```

Expected: the document explicitly says whether Campaign 6 retains public
`DeviceCommutationMatrix.count_commuting(...)`, keeps compact summaries
benchmark-only, or defers all consumer APIs. It also explicitly says whether
bit-packed output remains private, is rejected, or requires a later API plan.

- [x] **Step 1.3: Record the default public compatibility rule**

Update `docs/architecture/cuda_backend.md` so the post-Phase 11 boundary says:

```text
stream semantics: existing public CUDA APIs remain default-stream and synchronize-before-return; Campaign 6 may benchmark private stream/event probes only when docs/plans/cuda_async_stream_api_review.md accepts their lifetime and synchronization model
consumer semantics: DeviceCommutationMatrix compact-summary or bit-packed public APIs may be retained only when docs/plans/cuda_commutation_consumer_api_review.md accepts exact signatures, return types, synchronization, CPU-only behavior, and layout rules
```

Expected: existing `commutes_with_device`, `simplify`, `matmul`, and
`expectation_statevector` semantics remain unchanged unless the review
explicitly retains a public change.

- [x] **Step 1.4: Validate docs**

Run:

```bash
git diff --check
uv run python scripts/validate.py
```

Expected: link checks and stale-marker scans pass.

- [x] **Step 1.5: Commit the API review**

Run:

```bash
git add docs/plans/cuda_async_stream_api_review.md docs/plans/cuda_commutation_consumer_api_review.md docs/architecture/cuda_backend.md docs/architecture/api_stability.md docs/user/performance.md
git commit -m "Document CUDA stream and consumer API reviews"
```

Expected: the first Campaign 6 commit is a reviewable design boundary before code.

## Task 2: DeviceCommutationMatrix GPU Consumer Pipeline

**Files:**
- Modify: `include/fastpauli/device_commutation_matrix.hpp`
- Modify: `src/cuda/device_commutation_matrix.cuh`
- Modify: `src/cuda/device_commutation_matrix.cu`
- Modify: `src/device_commutation_matrix_stub.cpp`
- Modify: `bindings/python/pauli_sum_py.cpp`
- Modify: `python/fastpauli/__init__.py`
- Modify: `tests/test_phase11_cuda_kernels.py`
- Modify: `docs/user/performance.md`

- [x] **Step 2.1: Add failing retained-API tests if the consumer review accepts public compact summaries**

Add CUDA tests equivalent to:

```python
def test_cuda_device_commutation_matrix_count_commuting_matches_numpy() -> None:
    lhs = fastpauli.PauliSum.from_labels(["XI", "ZI"], [1.0, 1.0]).to_device()
    rhs = fastpauli.PauliSum.from_labels(["IX", "ZZ", "XX"], [1.0, 1.0, 1.0]).to_device()
    matrix = lhs.commutes_with_device(rhs)
    host = matrix.to_host()
    assert matrix.count_commuting() == int(host.sum())
    np.testing.assert_array_equal(matrix.count_commuting(axis=1), host.sum(axis=1, dtype=np.uint64))
    np.testing.assert_array_equal(matrix.count_commuting(axis=0), host.sum(axis=0, dtype=np.uint64))


def test_cuda_device_commutation_matrix_count_commuting_rejects_bad_axis() -> None:
    matrix = fastpauli.DeviceCommutationMatrix.empty((2, 3), device=0)
    with pytest.raises(ValueError, match="axis"):
        matrix.count_commuting(axis=2)
```

Expected before implementation: the tests fail because `count_commuting` is absent.

- [x] **Step 2.2: Implement compact GPU reductions**

Implement reductions with these C++ entry points if
`docs/plans/cuda_commutation_consumer_api_review.md` accepts public compact
summaries:

```cpp
[[nodiscard]] std::uint64_t count_commuting() const;
[[nodiscard]] std::vector<std::uint64_t> count_commuting_rows() const;
[[nodiscard]] std::vector<std::uint64_t> count_commuting_cols() const;
```

Implementation requirements:

```text
use CUDA kernels or CUB reductions on the matrix device
copy only the compact count result to host
avoid copying the dense uint8 matrix to host
preserve synchronize-before-return public semantics
throw RuntimeError for moved-from matrices
validate overflow before allocating count vectors
```

Expected: tests compare the compact results against `matrix.to_host()` for correctness, while timing instrumentation separates dense `to_host()` from compact-summary copies.

- [x] **Step 2.3: Keep CPU-only stubs explicit**

Update `src/device_commutation_matrix_stub.cpp` so CPU-only builds raise the existing CUDA rebuild guidance for retained compact-summary methods.

Expected: CPU-only imports still succeed and use failures remain actionable.

- [x] **Step 2.4: Bind and document retained public methods**

If `docs/plans/cuda_commutation_consumer_api_review.md` accepts public compact
summaries, bind one Python method:

```python
matrix.count_commuting(axis=None)
```

Expected:

```text
axis=None -> int
axis=0 -> numpy.ndarray dtype uint64 shape (cols,)
axis=1 -> numpy.ndarray dtype uint64 shape (rows,)
```

- [x] **Step 2.5: Validate and commit**

Run locally:

```bash
uv run python -m pytest tests/test_phase11_cuda_kernels.py tests/test_native_layout.py -q
git diff --check
```

Run on H100:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 uv run python scripts/validate.py
.venv/bin/python -m pytest tests/test_phase11_cuda_kernels.py -q
```

Commit:

```bash
git add include/fastpauli/device_commutation_matrix.hpp src/cuda/device_commutation_matrix.cuh src/cuda/device_commutation_matrix.cu src/device_commutation_matrix_stub.cpp bindings/python/pauli_sum_py.cpp python/fastpauli/__init__.py tests/test_phase11_cuda_kernels.py docs/user/performance.md
git commit -m "Add CUDA commutation matrix consumer reductions"
```

Expected: retained consumer behavior has CPU-only, CUDA correctness, and documentation coverage.

## Task 3: Campaign 6 Benchmarks For Dense Consumers, Bit-Packed Consumers, And Streams

**Files:**
- Modify: `benchmarks/bench_cuda_kernels.py`
- Modify: `benchmarks/bench_cuda_scaling.py`
- Modify: `tests/test_cuda_scaling_benchmark.py`
- Modify: `docs/benchmarks/protocol.md`

- [x] **Step 3.1: Add Campaign 6 scaling profile**

Add `campaign6_consumers` to `SCALE_PROFILES` with these operations:

```python
"campaign6_consumers": {
    "pairwise_commutation": [
        {"scale": "dense_consumer_terms_2048x2048", "num_qubits": 16, "terms": 2048, "term_weight": 3, "output_target": "device_uint8_matrix_consumer"},
        {"scale": "dense_consumer_terms_8192x8192", "num_qubits": 16, "terms": 8192, "term_weight": 3, "output_target": "device_uint8_matrix_consumer"},
        {"scale": "dense_consumer_terms_16384x16384", "num_qubits": 16, "terms": 16384, "term_weight": 3, "output_target": "device_uint8_matrix_consumer"},
    ],
}
```

Expected: non-CUDA tests can assert the profile shape without needing a GPU.

- [x] **Step 3.2: Add dense consumer timing fields**

Add result fields:

```text
cuda_device_output_consumer_total_seconds
cuda_device_output_consumer_axis0_seconds
cuda_device_output_consumer_axis1_seconds
cuda_device_output_consumer_to_host_bytes
cuda_device_output_dense_to_host_seconds
```

Expected: benchmark output distinguishes full dense host materialization from compact consumer summaries.

- [x] **Step 3.3: Add bit-packed prototype timing only behind an explicit label**

If Task 2 shows dense consumer capacity or memory bandwidth is limiting, add benchmark-only fields:

```text
cuda_bitpacked_consumer_total_seconds
cuda_bitpacked_consumer_axis0_seconds
cuda_bitpacked_consumer_axis1_seconds
cuda_bitpacked_output_bytes
bitpacked_layout: row-major uint64 words over rhs terms, one bit per rhs entry
```

Expected: bit-packed output is reported as `private_prototype` unless a separate public API review accepts its layout.

- [x] **Step 3.4: Add private stream/event timing fields only if accepted by Task 1**

If `docs/plans/cuda_async_stream_api_review.md` accepts private probes, add benchmark-only fields:

```text
cuda_private_stream_enqueue_seconds
cuda_private_event_elapsed_seconds
cuda_private_stream_synchronize_seconds
stream_boundary: private_benchmark_only
```

Expected: reports cannot confuse private stream timing with public async semantics.

- [x] **Step 3.5: Validate benchmark schema and commit**

Run:

```bash
uv run python -m pytest tests/test_cuda_scaling_benchmark.py tests/test_phase11_cuda_kernels.py::test_cuda_benchmark_preallocated_timing_schema_helper -q
git diff --check
```

Commit:

```bash
git add benchmarks/bench_cuda_kernels.py benchmarks/bench_cuda_scaling.py tests/test_cuda_scaling_benchmark.py docs/benchmarks/protocol.md
git commit -m "Add Campaign 6 CUDA consumer benchmark schema"
```

Expected: benchmark schema is test-covered before H100 execution.

## Task 4: CuPy CUDA-Array-Interface Consumer Benchmarks

**Files:**
- Modify: `benchmarks/bench_cuda_kernels.py`
- Modify: `benchmarks/bench_cuda_scaling.py`
- Modify: `benchmarks/bench_competitive_baselines.py`
- Modify: `tests/test_phase11_cuda_kernels.py`
- Modify: `tests/test_competitive_baselines_benchmark.py`
- Modify: `docs/architecture/testing_and_ci.md`

- [x] **Step 4.1: Add CuPy interop correctness test**

Add a CUDA test equivalent to:

```python
def test_cuda_device_commutation_matrix_cupy_consumer_matches_numpy() -> None:
    cupy = pytest.importorskip("cupy", reason="CuPy is required for CUDA array interface tests")
    lhs = fastpauli.PauliSum.from_labels(["XI", "ZI"], [1.0, 1.0]).to_device()
    rhs = fastpauli.PauliSum.from_labels(["IX", "ZZ", "XX"], [1.0, 1.0, 1.0]).to_device()
    matrix = lhs.commutes_with_device(rhs)
    cupy_view = cupy.asarray(matrix)
    assert cupy_view.shape == matrix.shape
    assert cupy_view.dtype == cupy.uint8
    np.testing.assert_array_equal(cupy.asnumpy(cupy_view), matrix.to_host().astype(np.uint8))
```

Expected: the test runs in the H100 validation tier when CuPy is installed and skips locally without CuPy.

- [x] **Step 4.2: Add CuPy consumer timing**

Add benchmark rows that time:

```text
cupy_asarray_export_seconds
cupy_sum_total_seconds
cupy_sum_axis0_seconds
cupy_sum_axis1_seconds
cupy_dense_to_host_seconds
```

Expected: CuPy consumer timings are labeled as CUDA Array Interface consumer timings, not as FastPauli kernel timings.

- [x] **Step 4.3: Add CI or H100-tier dependency policy**

Update `docs/architecture/testing_and_ci.md` with:

```text
CuPy CUDA Array Interface tests are mandatory in H100 validation when cupy-cuda12x is installed.
Public CI may skip them on CPU-only runners, but skipped status must be explicit.
```

Expected: the interop test dependency is documented before benchmark claims rely on it.

- [x] **Step 4.4: Validate and commit**

Run locally:

```bash
uv run python -m pytest tests/test_phase11_cuda_kernels.py tests/test_competitive_baselines_benchmark.py -q
```

Run on H100:

```bash
.venv/bin/python -m pytest tests/test_phase11_cuda_kernels.py -q
.venv/bin/python benchmarks/bench_cuda_scaling.py --profile campaign6_consumers --repeat 3 --warmup 1 --json
```

Commit:

```bash
git add benchmarks/bench_cuda_kernels.py benchmarks/bench_cuda_scaling.py benchmarks/bench_competitive_baselines.py tests/test_phase11_cuda_kernels.py tests/test_competitive_baselines_benchmark.py docs/architecture/testing_and_ci.md
git commit -m "Add CuPy CUDA array consumer benchmarks"
```

Expected: CuPy benchmark availability and skip behavior are explicit.

## Task 5: H100 Profiling, Sanitizers, And A/B Execution

**Files:**
- Create: `docs/benchmarks/data/cuda_deep_optimization_h100_campaign6_YYYY-MM-DD/raw/`
- Create: `docs/benchmarks/data/cuda_deep_optimization_h100_campaign6_YYYY-MM-DD/profiler/`
- Create: `docs/benchmarks/data/cuda_deep_optimization_h100_campaign6_YYYY-MM-DD/metadata/`

- [x] **Step 5.1: Run full H100 validation**

Run:

```bash
mkdir -p docs/benchmarks/data/cuda_deep_optimization_h100_campaign6_YYYY-MM-DD/metadata
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 uv run python scripts/validate.py | tee docs/benchmarks/data/cuda_deep_optimization_h100_campaign6_YYYY-MM-DD/metadata/experiment-validate-final.log
.venv/bin/python -m pytest tests/test_phase11_cuda_kernels.py -q | tee docs/benchmarks/data/cuda_deep_optimization_h100_campaign6_YYYY-MM-DD/metadata/experiment-phase11-cuda.log
```

Expected: CUDA tests pass on the H100 source build; skipped CuPy tests are recorded only when CuPy is absent.

- [x] **Step 5.2: Run Compute Sanitizer ladder**

Run:

```bash
compute-sanitizer --tool memcheck .venv/bin/python -m pytest tests/test_phase11_cuda_kernels.py -q
compute-sanitizer --tool racecheck .venv/bin/python -m pytest tests/test_phase11_cuda_kernels.py -q
compute-sanitizer --tool initcheck .venv/bin/python -m pytest tests/test_phase11_cuda_kernels.py -q
compute-sanitizer --tool synccheck .venv/bin/python -m pytest tests/test_phase11_cuda_kernels.py -q
```

Expected: zero CUDA memory errors, hazards, uninitialized accesses, and synchronization errors for retained code.

- [x] **Step 5.3: Run benchmark ladder**

Run:

```bash
mkdir -p docs/benchmarks/data/cuda_deep_optimization_h100_campaign6_YYYY-MM-DD/raw
.venv/bin/python benchmarks/bench_cuda_scaling.py --profile campaign6_consumers --repeat 7 --warmup 2 --json --output docs/benchmarks/data/cuda_deep_optimization_h100_campaign6_YYYY-MM-DD/raw/experiment_campaign6_consumers.json
.venv/bin/python benchmarks/bench_cuda_scaling.py --profile campaign5_device_output --repeat 7 --warmup 2 --json --output docs/benchmarks/data/cuda_deep_optimization_h100_campaign6_YYYY-MM-DD/raw/experiment_campaign5_device_output_reference.json
.venv/bin/python benchmarks/bench_cuda_kernels.py --profile default --repeat 7 --warmup 2 --json --output docs/benchmarks/data/cuda_deep_optimization_h100_campaign6_YYYY-MM-DD/raw/experiment_cuda_kernels_default.json
.venv/bin/python benchmarks/bench_competitive_baselines.py --repeat 7 --warmup 2 --json --output docs/benchmarks/data/cuda_deep_optimization_h100_campaign6_YYYY-MM-DD/raw/competitive_baselines_final.json
```

Expected: raw JSON includes dense consumer, optional bit-packed prototype, optional private stream/event probes, CuPy consumer timings, CPU selectors, CUDA boundaries, and external baselines where available.

- [x] **Step 5.4: Run Nsight Systems and Nsight Compute**

Run:

```bash
mkdir -p docs/benchmarks/data/cuda_deep_optimization_h100_campaign6_YYYY-MM-DD/profiler
nsys profile --force-overwrite true --stats true --trace=cuda,nvtx,osrt --output docs/benchmarks/data/cuda_deep_optimization_h100_campaign6_YYYY-MM-DD/profiler/nsys_campaign6_consumers .venv/bin/python benchmarks/bench_cuda_scaling.py --profile campaign6_consumers --repeat 3 --warmup 1 --json
ncu --set full --target-processes all --kernel-name regex:commutation --export docs/benchmarks/data/cuda_deep_optimization_h100_campaign6_YYYY-MM-DD/profiler/ncu_campaign6_consumers .venv/bin/python benchmarks/bench_cuda_scaling.py --profile campaign6_consumers --repeat 1 --warmup 1 --json
```

Expected: profiler exports identify whether dense output fill, consumer reductions, bit-packing, copies, allocation, or synchronization dominate.

## Task 6: Report, Plots, README Landscape, And Closeout

**Files:**
- Create: `scripts/render_cuda_campaign6_assets.py`
- Modify: `tests/test_cuda_deep_report_assets.py`
- Create: `docs/benchmarks/reports/cuda_deep_optimization_h100_campaign6_YYYY-MM-DD.md`
- Create: `docs/benchmarks/plots/cuda_h100_campaign6_*.svg`
- Modify: `README.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/plans/cuda_deep_optimization_plan.md`
- Modify: `docs/user/performance.md`

- [x] **Step 6.1: Add Campaign 6 renderer**

Create a renderer that produces:

```text
consumer pipeline comparison
dense vs bit-packed prototype comparison when bit-packed was measured
stream/event private-probe breakdown when private probes were measured
CuPy CUDA-array-interface consumer comparison
broad CPU/CUDA/external performance landscape
evidence status matrix
```

Expected: every plot is generated from checked raw JSON, summary JSON, and metadata.

- [x] **Step 6.2: Write the Campaign 6 report**

The report must include:

```text
async/stream API decision and public/deferred surfaces
DeviceCommutationMatrix consumer decision and retained/rejected methods
dense consumer same-boundary benchmark table
bit-packed consumer table only when the consumer need is proven
CuPy consumer benchmark table and dependency status
Nsight Systems and Nsight Compute summary tables
Compute Sanitizer status
CPU/CUDA/external broad comparison rows
README plot decision and limitations
remaining headroom after Campaign 6
```

Expected: the report distinguishes public APIs from benchmark-only prototypes.

- [x] **Step 6.3: Keep README broad**

Update README only if Campaign 6 supersedes Campaign 5 for the broad landscape. The README plot must include:

```text
CPU scalar
all captured optimized CPU selectors
CUDA transfer-inclusive rows
CUDA device-resident rows
boundary-specific rows such as device-output or compact-consumer summaries where relevant
CuPy consumer rows where semantically useful
external package baselines where available
unavailable baseline reasons
```

Expected: specialized stream, bit-packed, or CuPy-only plots remain in the Campaign 6 report unless integrated into the full landscape.

- [x] **Step 6.4: Review, validate, merge, push, and CI**

Run locally:

```bash
uv run python scripts/validate.py
git diff --check
```

Complete the repo workflow:

```text
commit sensible chunks
request independent review because this changes CUDA API/benchmark claims
resolve blocking findings
merge locally to main with fast-forward merge when possible
rerun uv run python scripts/validate.py on main
push main
confirm CI is green
delete the merged local and remote feature branches
```

Expected: no Campaign 6 completion claim is made without fresh H100 evidence, local validation, review closeout, and CI status.

## Exhaustion Criteria For Campaign 6

Campaign 6 is complete only when the final report shows:

```text
all five Campaign 5 remaining-headroom items were addressed
async/stream public API is explicitly retained, rejected, or deferred with event ownership and Python lifetime reasoning
DeviceCommutationMatrix has at least one benchmarked downstream GPU consumer or a documented rejection reason
dense consumer and full to_host materialization boundaries are timed separately
bit-packed output is benchmarked only when consumer evidence justifies it, and public bit-packed API remains deferred unless separately accepted
CuPy CUDA-array-interface consumer benchmarks run on H100 or the report records an installation/runtime blocker
Compute Sanitizer ladder is clean for retained CUDA code
Nsight Systems and Nsight Compute evidence is captured for retained performance claims
README performance landscape remains broad and generated from checked evidence
CPU-only import and rebuild-guidance behavior remains intact
public API docs and user-facing performance docs match retained behavior
```

## Remaining Headroom After Campaign 6

The Campaign 6 report must end with a new remaining-headroom section. Acceptable categories are:

```text
public async/stream implementation only if Campaign 6 leaves it deferred with a narrower accepted design
larger downstream GPU algorithms that consume DeviceCommutationMatrix beyond compact summaries
bit-packed public API only if Campaign 6 proves capacity or bandwidth pressure and documents layout semantics
DLPack interop only after CUDA Array Interface consumer evidence is stable
non-H100 portability runs after H100 evidence is complete
```

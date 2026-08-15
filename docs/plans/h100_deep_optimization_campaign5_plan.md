# H100 CUDA Device-Output Boundary Campaign 5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert Campaign 4's remaining H100 headroom into a supported device-resident commutation output boundary, then benchmark that public boundary against host-output, private reused-output, CPU, CUDA, and comparable external baselines without changing existing synchronous CUDA semantics.

**Architecture:** Campaign 5 starts with API review, because the remaining measured headroom is a result-materialization and ownership boundary rather than a raw kernel mechanics issue. The primary candidate is an experimental `DeviceCommutationMatrix` that owns dense row-major `uint8` flags on one CUDA device, exposes explicit host-copy and CUDA-array-interface access, and can be reused by `DevicePauliSum.commutes_with_device(..., output=...)`. Public stream or async behavior remains out of scope for retained API in this campaign; stream/event prototypes may be measured privately only after the device-output boundary is implemented and labeled.

**Tech Stack:** C++20, CUDA C++ 12.x, nanobind, NumPy buffer protocol, CUDA Array Interface v3, pytest, `bench_cuda_kernels.py`, `bench_cuda_scaling.py`, Nsight Systems, Nsight Compute, Compute Sanitizer, H100 source builds with `FASTPAULI_CUDA_ARCHITECTURES=90`.

---

## Status

Status: completed on H100 with checked report
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign5_2026-04-29.md`.

Campaign 4 source-of-truth evidence:

```text
report: docs/benchmarks/reports/cuda_deep_optimization_h100_campaign4_2026-04-29.md
data: docs/benchmarks/data/cuda_deep_optimization_h100_campaign4_2026-04-29/
retained public CUDA API changes: none
remaining headroom: supported device-output object or async/stream API, with device-output first
```

Campaign 5 is not a broad kernel hillclimb. It should not reopen CUB duplicate-reduction, packed simplify keys, raw PTX, or small commutation instruction edits unless the new device-output boundary changes the profiler bottleneck model and Nsight evidence identifies a specific new kernel limit.

Completion evidence:

```text
report: docs/benchmarks/reports/cuda_deep_optimization_h100_campaign5_2026-04-29.md
data: docs/benchmarks/data/cuda_deep_optimization_h100_campaign5_2026-04-29/
retained public CUDA API changes: DeviceCommutationMatrix and DevicePauliSum.commutes_with_device()
remaining headroom: async/stream API design, downstream GPU consumers, or bit-packed output only after a consumer contract exists
```

## Source Inputs

Read these files before implementation:

```text
README.md
docs/roadmap.md
docs/plans/cuda_deep_optimization_plan.md
docs/plans/h100_deep_optimization_campaign4_plan.md
docs/plans/cuda_commutation_device_output_api_review.md
docs/benchmarks/reports/cuda_deep_optimization_h100_campaign4_2026-04-29.md
docs/architecture/cuda_backend.md
docs/architecture/api_stability.md
docs/architecture/hardware_targets_and_testing.md
docs/benchmarks/protocol.md
docs/quality/code_review.md
docs/quality/code_standards.md
docs/quality/documentation_standards.md
docs/user/performance.md
```

## Scope

In scope:

```text
API review and documentation for an experimental dense device-output commutation result
move-only C++ ownership for device-resident row-major uint8 commutation flags
CPU-only stubs that preserve import and error behavior without CUDA headers in CPU-only public paths
Python bindings for allocation, shape/device metadata, host copy, and CUDA-array-interface exposure
DevicePauliSum.commutes_with_device(other, output=None, max_commutation_matrix_entries=...)
DevicePauliSum.commutes_with_device_into(other, output, max_commutation_matrix_entries=...)
same-device, shape, dtype, moved-from, and max-entry error coverage
H100 benchmarks that compare public host vector, public host fill, public device output allocation, public device output reuse, and Campaign 4 private reused-output labels
private stream/event timing probes only as benchmark labels, with public methods still synchronizing before returning
Nsight Systems, Nsight Compute, Compute Sanitizer, and same-boundary A/B evidence for retained changes
README performance landscape refresh only if Campaign 5 materially changes checked benchmark evidence
final checked-in Campaign 5 report with retained, rejected, deferred, and exhausted paths
```

Out of scope unless a separate accepted API plan is added first:

```text
public stream handles
public async methods
public bit-packed commutation output
public CUDA workspace object
public device-statevector wrapper
CUDA wheel release claims
non-H100 NVIDIA claims
HIP/AMD or Metal/MPS implementation
raw PTX or inline PTX without profiler and SASS evidence
```

## Public API Candidate

Campaign 5 should implement this candidate only after the API review task accepts it in the branch:

```python
matrix = lhs_device.commutes_with_device(
    rhs_device,
    max_commutation_matrix_entries=100_000_000,
)

reused = fastpauli.DeviceCommutationMatrix.empty(
    shape=(lhs_device.num_terms, rhs_device.num_terms),
    device=lhs_device.device,
)
same = lhs_device.commutes_with_device(
    rhs_device,
    output=reused,
    max_commutation_matrix_entries=100_000_000,
)
assert same is reused

host_bool = matrix.to_host()
cuda_view = matrix.__cuda_array_interface__
```

Required `DeviceCommutationMatrix` behavior:

```text
owns one contiguous CUDA allocation of uint8 flags
shape is exactly (lhs_terms, rhs_terms)
entries are row-major over lhs term first, then rhs term
flags are 1 for commuting pairs and 0 for anti-commuting pairs
device is the CUDA ordinal that owns the allocation
to_host() returns a NumPy bool array with shape
__cuda_array_interface__ exposes the uint8 device buffer with typestr "|u1"
copying is disabled; moving transfers ownership
empty(shape, device=0) validates non-negative dimensions and allocation overflow
wrong-device reuse raises ValueError
wrong-shape reuse raises ValueError
moved-from use raises RuntimeError
CPU-only builds raise RuntimeError with existing CUDA rebuild guidance
```

The candidate intentionally does not expose bit-packed output. Bit packing should remain benchmark-only until dense `uint8` output is proven insufficient and a consumer API for packed layout is designed.

## File Structure

Planned implementation files:

```text
include/fastpauli/device_commutation_matrix.hpp
  Public move-only C++ source-compatibility API for dense device-resident
  commutation flags, CPU-only stubs, metadata accessors, and to_host().

include/fastpauli/device_pauli_sum.hpp
  Add commutes_with_device() and commutes_with_device_into() declarations after
  DeviceCommutationMatrix is declared.

src/device_commutation_matrix_stub.cpp
  CPU-only definitions that preserve import and RuntimeError behavior when
  FASTPAULI_ENABLE_CUDA=OFF.

src/cuda/device_commutation_matrix.cuh
  Private CUDA implementation details for allocation, pointer access, move
  ownership, and CUDA-array-interface pointer metadata.

src/cuda/device_commutation_matrix.cu
  CUDA definitions for construction, destruction, host copy, validation, and
  allocation overflow checks.

src/cuda/commutation_cuda.cu
  Reuse the existing commutation kernel to fill DeviceCommutationMatrix storage,
  keep host-output paths unchanged, and remove private reused-output benchmark
  dependence from public timings.

bindings/python/pauli_sum_py.cpp
  Bind DeviceCommutationMatrix, expose shape/device/num_entries/to_host,
  implement __cuda_array_interface__, and bind commutes_with_device methods.

CMakeLists.txt
  Add the new CPU-only stub and CUDA sources to the same ON/OFF build structure
  used by DevicePauliSum.
```

Planned benchmark, report, and test files:

```text
tests/test_phase11_cuda_kernels.py
  CUDA correctness, error, CPU-only, CUDA-array-interface, and reuse tests for
  DeviceCommutationMatrix.

tests/test_phase10_cuda_foundation.py
  CPU-only availability and rebuild-guidance checks for the new public class.

benchmarks/bench_cuda_kernels.py
  Public device-output allocation and public device-output reuse timing fields.

benchmarks/bench_cuda_scaling.py
  campaign5_device_output profile for default, stress, and extreme dense
  commutation shapes.

tests/test_cuda_scaling_benchmark.py
  Schema coverage for campaign5_device_output and new timing-boundary labels.

scripts/render_cuda_campaign5_assets.py
  Renderer for Campaign 5 summary JSON and SVG plots.

tests/test_cuda_deep_report_assets.py
  Renderer freshness and README landscape coverage when Campaign 5 evidence is
  checked in.

docs/benchmarks/data/cuda_deep_optimization_h100_campaign5_YYYY-MM-DD/
docs/benchmarks/reports/cuda_deep_optimization_h100_campaign5_YYYY-MM-DD.md
docs/benchmarks/plots/cuda_h100_campaign5_*.svg
```

Planned documentation files:

```text
docs/plans/cuda_commutation_device_output_api_review.md
docs/architecture/cuda_backend.md
docs/architecture/api_stability.md
docs/benchmarks/protocol.md
docs/user/performance.md
README.md
docs/roadmap.md
docs/plans/cuda_deep_optimization_plan.md
```

## H100 Execution Contract

Use environment variables for the current H100 host and artifact roots:

```bash
export FASTPAULI_H100_SSH_TARGET="${FASTPAULI_H100_SSH_TARGET:?set to the current H100 SSH target}"
export FASTPAULI_H100_BASELINE_DIR=<private-path>
export FASTPAULI_H100_EXPERIMENT_DIR=<private-path>
export FASTPAULI_H100_ARTIFACT_ROOT=<private-path>
export FASTPAULI_H100_BASELINE_REVISION=72b46e86ad4d2564805b93eb4727ab0d9a8dde9b
export FASTPAULI_BRANCH=codex/h100-deep-optimization-campaign5
```

Baseline and experiment checkouts must stay independent:

```text
baseline checkout: exact post-Campaign-4 main revision that produced the README landscape plot
experiment checkout: Campaign 5 branch or exact experiment commit
never reset a dirty experiment checkout
record both exact revisions in the final report
```

Minimum remote preflight command:

```bash
ssh "$FASTPAULI_H100_SSH_TARGET" \
  'set -eu; \
   ART=<private-path> \
   mkdir -p "$ART" && \
   nvidia-smi --query-gpu=name,uuid,driver_version,cuda_version,compute_cap --format=csv \
     > "$ART/gpu.csv" && \
   nvidia-smi -q > "$ART/nvidia-smi-q.txt" && \
   lscpu > "$ART/lscpu.txt" && \
   /usr/local/cuda/bin/nvcc --version > "$ART/nvcc-version.txt" && \
   nsys --version > "$ART/nsys-version.txt" && \
   ncu --version > "$ART/ncu-version.txt" && \
   compute-sanitizer --version > "$ART/compute-sanitizer-version.txt"'
```

## Decision Gates

Record each decision in the Campaign 5 report:

```text
device-output API status: rejected, experimental public API, or design-deferred
output storage format: dense uint8 retained, bit-packed deferred, or both rejected
reuse policy: allocation-only, caller-owned reuse, or owned reusable object
CUDA-array-interface policy: uint8 exposed, omitted, or deferred with reason
synchronization policy: default-stream synchronize-before-return retained
stream prototype policy: not measured, private benchmark-only, or separate API plan
timing-boundary policy: host vector, host fill, device allocation, device reuse, private event-only
retention policy: production, experimental public, benchmark-only, report-only rejected, or design-deferred
README plot policy: unchanged, refreshed broad landscape, or report-only supporting plot
```

Promotion gate:

```text
Do not expose DeviceCommutationMatrix as public API unless docs, tests, CPU-only
stub behavior, CUDA correctness, same-device validation, and benchmark labels
land in the same branch. The retained API must be useful even if host-output
timings remain competitive for small matrices; speedup claims must compare
matching boundaries and must not present device-resident output as a direct
replacement for host materialization unless downstream consumption stays on GPU.
```

## Task 0: Baseline Reproduction And API Review Preflight

**Files:**

```text
docs/plans/cuda_commutation_device_output_api_review.md
docs/architecture/cuda_backend.md
docs/architecture/api_stability.md
```

- [ ] **Step 0.1: Create the Campaign 5 branch**

Run:

```bash
git switch main
git pull --ff-only origin main
git switch -c codex/h100-deep-optimization-campaign5
```

Expected: the new branch starts at or after `72b46e86ad4d2564805b93eb4727ab0d9a8dde9b`.

- [ ] **Step 0.2: Accept or reject the public API candidate before code**

Modify `docs/plans/cuda_commutation_device_output_api_review.md` so it records one of:

```text
accepted for Campaign 5 experimental public API
rejected with reason and replacement plan
design-deferred with required missing evidence
```

Expected: the file names the chosen method names, object name, ownership rules,
dtype, shape, device ordinal, synchronization rule, CUDA-array-interface rule,
and error behavior.

- [ ] **Step 0.3: Update architecture contracts for the accepted choice**

If accepted, update `docs/architecture/cuda_backend.md` and
`docs/architecture/api_stability.md` with the `DeviceCommutationMatrix`
contract from this plan.

Expected: public CUDA behavior docs identify device-output commutation as
experimental, dense `uint8`, synchronous, and same-device only.

- [ ] **Step 0.4: Validate docs before implementation**

Run:

```bash
git diff --check
uv run python scripts/validate.py
```

Expected: both commands pass before native code changes begin.

## Task 1: DeviceCommutationMatrix Ownership And CPU-Only Stub

**Files:**

```text
include/fastpauli/device_commutation_matrix.hpp
include/fastpauli/device_pauli_sum.hpp
src/device_commutation_matrix_stub.cpp
CMakeLists.txt
tests/test_phase10_cuda_foundation.py
tests/test_native_layout.py
scripts/validate.py
```

- [ ] **Step 1.1: Write CPU-only public-surface tests**

Add tests that assert:

```python
assert hasattr(fastpauli, "DeviceCommutationMatrix")
with pytest.raises(RuntimeError, match="built without CUDA.*FASTPAULI_ENABLE_CUDA=ON"):
    fastpauli.DeviceCommutationMatrix.empty((2, 3), device=0)
```

Expected: the test fails before the stub and binding exist.

- [ ] **Step 1.2: Add the public C++ header and CPU-only definitions**

Implement a move-only `DeviceCommutationMatrix` with:

```text
empty(shape0, shape1, device)
to_host()
rows()
cols()
num_entries()
device()
data_pointer_for_cuda_array_interface()
```

The CPU-only definitions must raise the same rebuild-guidance RuntimeError used
by `DevicePauliSum` CUDA stubs.

Expected: CPU-only import remains successful without CUDA headers.

- [ ] **Step 1.3: Update build and layout validation**

Add the new header/source to `CMakeLists.txt`, `scripts/validate.py`, and
`tests/test_native_layout.py`.

Expected: `uv run python scripts/validate.py` sees the new source layout.

- [ ] **Step 1.4: Run focused validation**

Run:

```bash
uv run python -m pytest tests/test_phase10_cuda_foundation.py tests/test_native_layout.py -q
git diff --check
```

Expected: focused tests and whitespace checks pass.

## Task 2: CUDA Device Output Implementation

**Files:**

```text
src/cuda/device_commutation_matrix.cuh
src/cuda/device_commutation_matrix.cu
src/cuda/commutation_cuda.cu
include/fastpauli/device_commutation_matrix.hpp
include/fastpauli/device_pauli_sum.hpp
tests/test_phase11_cuda_kernels.py
```

- [ ] **Step 2.1: Write CUDA behavior tests**

Add CUDA-gated tests for:

```text
empty allocation shape/device metadata
to_host returns bool array with expected shape
commutes_with_device matches CPU commutes_with
commutes_with_device output reuse returns the same output object
wrong output shape raises ValueError
wrong output device raises ValueError
max_commutation_matrix_entries guardrail fires before allocation
moved-from objects raise RuntimeError in C++ tests or Python-accessible probes
```

Expected: tests skip cleanly without CUDA and fail on a CUDA build before the implementation exists.

- [ ] **Step 2.2: Implement CUDA ownership**

Implement the CUDA allocation owner with:

```text
overflow-checked byte counts
ScopedCudaDevice around allocation, free, copy, and kernel fill
move constructor and move assignment that leave the source empty
destructor that frees on the owning device
to_host() that copies device uint8 flags and returns bool semantics through bindings
```

Expected: no public CUDA headers enter CPU-only public headers beyond the existing `FASTPAULI_ENABLE_CUDA` guarded build structure.

- [ ] **Step 2.3: Implement commutes_with_device paths**

Add:

```text
DevicePauliSum::commutes_with_device(rhs, max_entries)
DevicePauliSum::commutes_with_device_into(rhs, output, max_entries)
```

They must use the existing `commutation_kernel` and write directly into
`DeviceCommutationMatrix` storage. Existing `commutes_with()` and
`commutes_with_into()` host-output behavior must remain unchanged.

Expected: the public device-output path avoids device-to-host copy until
`DeviceCommutationMatrix.to_host()` is called.

- [ ] **Step 2.4: Run CUDA focused validation on H100**

Run on the H100 experiment checkout:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 uv run python scripts/validate.py
.venv/bin/python -m pytest tests/test_phase11_cuda_kernels.py -q
```

Expected: CUDA validation passes, with Phase 11 CUDA tests passing on the H100 host.
The focused test command intentionally uses the installed venv Python after the
CUDA source build. Plain `uv run` can re-sync the editable checkout with default
CPU-only CMake settings and turn CUDA tests into skipped tests.

## Task 3: Python Binding And CUDA Array Interface

**Files:**

```text
bindings/python/pauli_sum_py.cpp
python/fastpauli/__init__.py
tests/test_phase10_cuda_foundation.py
tests/test_phase11_cuda_kernels.py
docs/user/performance.md
```

- [ ] **Step 3.1: Add Python binding tests**

Assert the Python surface:

```python
matrix = lhs_device.commutes_with_device(rhs_device)
assert matrix.shape == (lhs_device.num_terms, rhs_device.num_terms)
assert matrix.device == lhs_device.device
assert matrix.dtype == "uint8"
cuda_interface = matrix.__cuda_array_interface__
assert cuda_interface["shape"] == matrix.shape
assert cuda_interface["typestr"] == "|u1"
assert cuda_interface["version"] >= 3
assert cuda_interface["strides"] in (None, (matrix.shape[1], 1))
assert isinstance(cuda_interface["data"][0], int)
assert cuda_interface["data"][0] != 0 or matrix.num_entries == 0
assert cuda_interface.get("stream") in (None, 1)
np.testing.assert_array_equal(matrix.to_host(), lhs.commutes_with(rhs))
```

Expected: test fails before bindings are complete.

The tests must also verify that the CUDA-array-interface dictionary remains
valid while the owning `DeviceCommutationMatrix` is alive and is not documented
as safe after that object is destroyed or moved. Public methods remain
default-stream synchronized, so a non-default stream value is not valid in
Campaign 5.

- [ ] **Step 3.2: Bind the public class**

Expose:

```text
DeviceCommutationMatrix.empty(shape, device=0)
DeviceCommutationMatrix.shape
DeviceCommutationMatrix.device
DeviceCommutationMatrix.dtype
DeviceCommutationMatrix.num_entries
DeviceCommutationMatrix.to_host()
DeviceCommutationMatrix.__cuda_array_interface__
```

Expected: Python users can allocate reusable output and pass it back to
`DevicePauliSum.commutes_with_device(..., output=matrix)`.

- [ ] **Step 3.3: Document the user-facing pattern**

Update `docs/user/performance.md` with a compact example showing:

```python
out = fastpauli.DeviceCommutationMatrix.empty((lhs.num_terms, rhs.num_terms), device=0)
lhs_d.commutes_with_device(rhs_d, output=out)
flags = out.to_host()
```

Expected: docs make clear this is for GPU-resident workflows and that
`to_host()` is the materialization boundary.

## Task 4: Benchmark Schema And Scaling Profiles

**Files:**

```text
benchmarks/bench_cuda_kernels.py
benchmarks/bench_cuda_scaling.py
tests/test_cuda_scaling_benchmark.py
tests/test_phase11_cuda_kernels.py
docs/benchmarks/protocol.md
```

- [ ] **Step 4.1: Add benchmark timing fields**

Record these fields for commutation cases:

```text
cuda_device_output_allocate_seconds
cuda_device_output_reuse_seconds
cuda_device_output_to_host_seconds
cuda_device_output_cuda_array_interface_export_seconds
result_materialization_target: device_uint8_matrix
timing_boundary: device_output_allocating, device_output_reused, device_output_to_host
```

Expected: JSON clearly separates kernel fill, device-output allocation, and host materialization.

- [ ] **Step 4.2: Add Campaign 5 scale profile**

Define `campaign5_device_output` with:

```text
default: 1024x1024, 2048x2048, 4096x4096
stress: 8192x8192, 16384x16384
extreme: one opt-in size selected by free H100 memory after preflight
```

Expected: default/stress run without the opt-in extreme flag; extreme is explicit.

- [ ] **Step 4.3: Add benchmark schema tests**

Tests must assert the new fields, labels, and profile names exist without
running CUDA.

Expected: `uv run python -m pytest tests/test_cuda_scaling_benchmark.py tests/test_phase11_cuda_kernels.py -q` passes locally.

## Task 5: H100 A/B Benchmark And Profiler Execution

**Files:**

```text
remote artifacts under <private-path>
docs/benchmarks/data/cuda_deep_optimization_h100_campaign5_YYYY-MM-DD/
```

- [ ] **Step 5.1: Prepare baseline and experiment checkouts**

Run:

```bash
ssh "$FASTPAULI_H100_SSH_TARGET" \
  "set -eu; \
   BASE=<private-path> \
   EXP=<private-path> \
   BASELINE_REV=72b46e86ad4d2564805b93eb4727ab0d9a8dde9b; \
   BRANCH=${FASTPAULI_BRANCH:?set FASTPAULI_BRANCH locally before running}; \
   if [ ! -d \"\$BASE/.git\" ]; then git clone https://github.com/sghowell/FastPauli.git \"\$BASE\"; fi; \
   if [ ! -d \"\$EXP/.git\" ]; then git clone https://github.com/sghowell/FastPauli.git \"\$EXP\"; fi; \
   git -C \"\$BASE\" fetch origin && git -C \"\$BASE\" checkout \"\$BASELINE_REV\"; \
   git -C \"\$EXP\" fetch origin && git -C \"\$EXP\" checkout \"\$BRANCH\""
```

Expected: baseline and experiment revisions are independent and recorded.

- [ ] **Step 5.2: Run correctness and sanitizer gates**

Run on the H100 experiment checkout:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 uv run python scripts/validate.py
.venv/bin/python -m pytest tests/test_phase11_cuda_kernels.py -q
compute-sanitizer --tool memcheck .venv/bin/python -m pytest tests/test_phase11_cuda_kernels.py -q
compute-sanitizer --tool racecheck .venv/bin/python -m pytest tests/test_phase11_cuda_kernels.py -q
compute-sanitizer --tool initcheck .venv/bin/python -m pytest tests/test_phase11_cuda_kernels.py -q
compute-sanitizer --tool synccheck .venv/bin/python -m pytest tests/test_phase11_cuda_kernels.py -q
```

Expected: no sanitizer errors for retained device-output code.

- [ ] **Step 5.3: Run same-boundary benchmarks**

Run:

```bash
.venv/bin/python benchmarks/bench_cuda_scaling.py --profile campaign5_device_output --repeat 7 --warmup 2 --json
.venv/bin/python benchmarks/bench_cuda_kernels.py --profile default --repeat 7 --warmup 2 --json
.venv/bin/python benchmarks/bench_competitive_baselines.py --repeat 7 --warmup 2 --json
```

Expected: results include CPU scalar, every available optimized CPU selector,
public host vector, public host fill, public device-output allocation, public
device-output reuse, CUDA transfer-inclusive, CUDA device-resident, and
available external baselines.

- [ ] **Step 5.4: Capture profiler evidence**

Run Nsight Systems on the Campaign 5 commutation profile:

```bash
nsys profile --force-overwrite true --stats true \
  --trace=cuda,nvtx,osrt \
  --output "$FASTPAULI_H100_ARTIFACT_ROOT/nsys_campaign5_device_output" \
  .venv/bin/python benchmarks/bench_cuda_scaling.py --profile campaign5_device_output --repeat 3 --warmup 1 --json
```

Run Nsight Compute for the commutation kernel:

```bash
ncu --set full --target-processes all --kernel-name regex:commutation_kernel \
  --export "$FASTPAULI_H100_ARTIFACT_ROOT/ncu_campaign5_commutation" \
  .venv/bin/python benchmarks/bench_cuda_scaling.py --profile campaign5_device_output --repeat 1 --warmup 1 --json
```

Expected: profiler exports separate allocation, kernel, copy, and synchronization costs.

## Task 6: Report, Plots, README Landscape, And Closeout

**Files:**

```text
scripts/render_cuda_campaign5_assets.py
tests/test_cuda_deep_report_assets.py
docs/benchmarks/data/cuda_deep_optimization_h100_campaign5_YYYY-MM-DD/
docs/benchmarks/reports/cuda_deep_optimization_h100_campaign5_YYYY-MM-DD.md
docs/benchmarks/plots/cuda_h100_campaign5_*.svg
README.md
docs/roadmap.md
docs/plans/cuda_deep_optimization_plan.md
```

- [ ] **Step 6.1: Add checked-in report assets**

Create a Campaign 5 renderer that produces:

```text
device-output boundary comparison
host-materialization decomposition
same-boundary A/B comparison
CPU/CUDA/external performance landscape when evidence materially changes
evidence status matrix
```

Expected: plot data comes only from checked raw JSON and metadata.

- [ ] **Step 6.2: Write the Campaign 5 report**

The report must include:

```text
accepted API contract and public semantics
retained and rejected experiments
same-boundary benchmark tables
Nsight Systems and Nsight Compute summary tables
Compute Sanitizer status
CPU/CUDA/external comparison rows
small, default, stress, and extreme regimes
remaining headroom after device-output boundary
```

Expected: the report distinguishes device-resident output from host-output replacement claims.

- [ ] **Step 6.3: Refresh README only with broad evidence**

If Campaign 5 supersedes Campaign 4 for the broad landscape, update README to
point at the Campaign 5 landscape plot. If Campaign 5 only adds a specialized
device-output report, keep the Campaign 4 README landscape and link Campaign 5
as supporting evidence.

Expected: README remains an across-the-board view, not a single-path snapshot.

- [ ] **Step 6.4: Complete review, validation, merge, push, and CI**

Run:

```bash
uv run python scripts/validate.py
git diff --check
```

Then complete the repo workflow:

```text
commit sensible chunks
request independent review because this changes CUDA public API and benchmark claims
resolve blocking findings
merge locally to main
rerun uv run python scripts/validate.py
push main
confirm CI green
delete the merged local feature branch
```

Expected: no phase or campaign completion claim is made without fresh validation and review evidence.

## Exhaustion Criteria For Campaign 5

Campaign 5 is complete only when the final report shows:

```text
public device-output API either retained with full evidence or explicitly rejected
CPU-only import and rebuild-guidance behavior remains intact
CUDA correctness and sanitizer coverage pass on H100
host vector, host fill, device output allocation, and device output reuse are timed separately
private event-only or stream measurements are labeled as private when present
Nsight Systems separates allocation, kernel, copy, and synchronization costs
Nsight Compute covers the commutation kernel under the new boundary
README landscape remains broad and current
remaining headroom is narrowed to a specific next public API or hardware target
```

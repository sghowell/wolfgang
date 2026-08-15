# MI300X ROCm/HIP Bring-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the first AMD ROCm/HIP backend evidence path for FastPauli on a 1x AMD Instinct MI300X, starting with source-build foundation, host/device transfer correctness, the first meaningful HIP kernel, and reproducible MI300X benchmark/profiling evidence.

**Architecture:** CUDA remains the first supported GPU backend and must not be disturbed by ROCm/HIP work. HIP is introduced as a separate optional source-build backend behind `FASTPAULI_ENABLE_HIP=ON`, with independent build metadata, runtime status, tests, benchmark evidence, and reports. The initial HIP implementation may share the existing `DevicePauliSum` public surface only in mutually exclusive CUDA-or-HIP builds; simultaneous CUDA+HIP builds are explicitly rejected until a backend-neutral multi-device container design is accepted.

**Tech Stack:** C++20, nanobind, scikit-build-core, CMake HIP language support, ROCm/HIP, AMD Instinct MI300X (`gfx942`), rocprof, pytest, existing FastPauli benchmark/report infrastructure.

---

## Current Baseline

FastPauli completed the CUDA Phase 11 residual-risk campaign in
`docs/benchmarks/reports/cuda_residual_risk_campaign11_2026-04-29.md`.
That report leaves no in-scope CUDA residual-risk item without a terminal
status. Remaining CUDA work is release packaging, optional additional NVIDIA
portability lanes, or a specific retained consumer with an accepted API and
memory-ownership contract.

The ROCm/HIP policy already exists in
`docs/architecture/hardware_targets_and_testing.md`: ROCm/HIP is the planned
second GPU backend, and HIP support claims require independent evidence for
toolkit version, GPU architecture, driver/runtime, source build, runtime
transfer tests, CPU/GPU equivalence, and transfer-inclusive/device-resident
benchmarks.

Vendor facts verified on 2026-04-29:

```text
AMD Developer Cloud offers 1x AMD Instinct MI300X instances with 192 GB GPU memory.
AMD Instinct MI300X is CDNA3 with 304 compute units, 192 GB HBM3, and 5.3 TB/s peak memory bandwidth.
Current ROCm Linux system requirements list AMD Instinct MI300X as supported with LLVM target gfx942.
HIP is the ROCm C++ runtime API and kernel language aligned with CUDA-style programming.
rocprof provides HIP profiling, performance counters, hardware traces, and runtime API/activity traces.
```

Sources:

```text
https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html
https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html
https://rocm.docs.amd.com/projects/HIP/en/develop/index.html
https://rocm.docs.amd.com/projects/rocprofiler/en/latest/how-to/using-rocprof.html
https://rocm.docs.amd.com/en/latest/how-to/gpu-performance/mi300x.html
```

## Scope

This plan covers one bounded MI300X campaign:

```text
single-node 1x MI300X bring-up
ROCm/HIP source-build foundation
runtime status and metadata reporting
host-to-device and device-to-host PauliSum transfers
first HIP correctness kernel, selected for semantic value and implementation risk
MI300X benchmark and rocprof evidence
checked-in report and README/roadmap update when evidence exists
```

This plan does not cover:

```text
8x MI300X multi-GPU work
distributed ROCm communication
ROCm wheel release claims
Metal/MPS implementation
additional NVIDIA portability lanes
CUDA kernel retuning
simultaneous CUDA+HIP runtime objects
public async stream, graph, or external workspace APIs
```

## Hard Decisions

Use these decisions unless a follow-up architecture review changes them before
implementation starts:

```text
Build flag: FASTPAULI_ENABLE_HIP=ON
Default HIP architecture list on MI300X: FASTPAULI_HIP_ARCHITECTURES=gfx942
Source layout: src/hip/ for HIP runtime, kernels, and workspace helpers
Public headers: no HIP or ROCm headers in include/fastpauli/*.hpp
CPU-only default: FASTPAULI_ENABLE_HIP=OFF
Release wheels: HIP support is source-build-only until separate packaging evidence exists
Backend coexistence: reject FASTPAULI_ENABLE_CUDA=ON together with FASTPAULI_ENABLE_HIP=ON in this campaign
Python import: importing fastpauli must not import ROCm or require a GPU
API compatibility: existing CUDA builds keep PauliSum.to_device(device=0) behavior
HIP-only API: PauliSum.to_device(device=0) may return DevicePauliSum backed by HIP only when FastPauli is built with HIP and without CUDA
Status API: add _hip_status() and _accelerator_status() rather than overloading _cuda_status()
Device metadata: DevicePauliSum gains a backend property or method returning "cuda" or "hip"
```

The mutually exclusive CUDA-or-HIP rule is intentional. It keeps the first
MI300X campaign focused on correctness and performance evidence instead of
forcing a larger backend-neutral multi-device object model before we know the
HIP kernel and packaging constraints.

## Evidence Layout

All MI300X campaign artifacts use one date-stamped evidence root:

```text
docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/
docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/logs/
docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/raw/
docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/profiler/
docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/summary.json
docs/benchmarks/reports/rocm_mi300x_bringup_2026-04-29.md
```

If the execution date changes, use that date consistently across the evidence
root, report, and README links.

## Acceptance Criteria

The campaign is complete only when every applicable item below has a terminal
status in the report:

```text
MI300X host inventory is captured, including OS, kernel, CPU, memory, ROCm version, hipcc version, rocminfo, rocm-smi, visible GPU model, LLVM target, VRAM, power limit, clocks, topology, and relevant environment variables
CPU-only local validation still passes with FASTPAULI_ENABLE_HIP=OFF
HIP source build succeeds on MI300X with FASTPAULI_ENABLE_HIP=ON and FASTPAULI_HIP_ARCHITECTURES=gfx942
FASTPAULI_ENABLE_CUDA=ON and FASTPAULI_ENABLE_HIP=ON together fail at configure time with a clear error
_build_info() reports HIP build flags, HIP architecture list, ROCm/HIP compiler metadata, and runtime availability without changing existing CUDA metadata semantics
_hip_status() reports build status, runtime availability, device count, driver/runtime/toolkit information where ROCm exposes it, device ordinal, device name, gfx target, and total memory
_accelerator_status() reports CPU-only, CUDA-only, or HIP-only state without requiring ROCm at import time
PauliSum.to_device().to_host() round-trips non-empty and empty operators on a HIP-only build
Moved-from and invalid-device errors are deterministic and mention HIP or accelerator backend accurately
At least one HIP kernel passes CPU/GPU equivalence tests on deterministic and randomized datasets
Benchmarks report CPU scalar, available optimized CPU selectors, HIP transfer-inclusive timing, HIP device-resident timing when applicable, and unavailable external baseline reasons
rocprof trace or counter evidence is captured for the retained HIP kernel unless the provider blocks profiling; blocked profiling must include command output and provider/permission diagnosis
README and roadmap are updated only with evidence-backed MI300X claims
Independent review is recorded before merge
```

## Task 1: Remote Instance Inventory And Evidence Harness

**Files:**
- Create: `tools/remote/collect_rocm_inventory.sh`
- Create during execution: `docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/logs/host_inventory_mi300x.log`
- Modify later if needed: `scripts/validate.py`

- [ ] **Step 1: Create the inventory script**

Create `tools/remote/collect_rocm_inventory.sh` with commands that never
modify the host:

```bash
#!/usr/bin/env bash
set -uo pipefail

echo "## date"
date -u +"%Y-%m-%dT%H:%M:%SZ"

echo "## uname"
uname -a

echo "## os-release"
cat /etc/os-release

echo "## lscpu"
lscpu

echo "## memory"
free -h

echo "## disks"
df -h

echo "## rocm paths"
command -v hipcc || true
command -v rocminfo || true
command -v rocm-smi || true
ls -ld /opt/rocm /opt/rocm/bin 2>/dev/null || true

echo "## hipcc"
hipcc --version 2>&1 || /opt/rocm/bin/hipcc --version 2>&1 || true

echo "## rocminfo"
rocminfo 2>&1 || /opt/rocm/bin/rocminfo 2>&1 || true

echo "## rocm-smi"
rocm-smi 2>&1 || /opt/rocm/bin/rocm-smi 2>&1 || true

echo "## rocm-smi detailed"
rocm-smi --showproductname --showdriverversion --showvbios --showbus --showmeminfo vram --showtopo --showpower --showclocks 2>&1 || \
  /opt/rocm/bin/rocm-smi --showproductname --showdriverversion --showvbios --showbus --showmeminfo vram --showtopo --showpower --showclocks 2>&1 || true

echo "## environment"
env | sort | grep -E '^(HIP|HSA|ROCM|LD_LIBRARY_PATH|PATH|PYTHON|VIRTUAL_ENV|CMAKE|CC|CXX)=' || true
```

- [ ] **Step 2: Validate the shell syntax locally**

Run:

```bash
bash -n tools/remote/collect_rocm_inventory.sh
```

Expected: command exits 0.

- [ ] **Step 3: Capture MI300X inventory after instance access is available**

Run on the MI300X host from a fresh checkout:

```bash
mkdir -p docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/logs
bash tools/remote/collect_rocm_inventory.sh \
  > docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/logs/host_inventory_mi300x.log \
  2>&1
```

Expected: the log contains MI300X or `gfx942` in `rocminfo` or `rocm-smi`
output, plus a visible ROCm/HIP compiler path.

- [ ] **Step 4: Commit**

```bash
git add tools/remote/collect_rocm_inventory.sh
git commit -m "chore: add ROCm inventory probe"
```

## Task 2: ROCm Backend Architecture Contract

**Files:**
- Create: `docs/architecture/rocm_backend.md`
- Modify: `docs/architecture/hardware_targets_and_testing.md`
- Modify: `docs/quality/phase_quality_gates.md`

- [ ] **Step 1: Document the HIP backend contract**

Create `docs/architecture/rocm_backend.md` with these required sections:

```text
Scope and non-goals
Build flags and source-build policy
Public API compatibility
Mutually exclusive CUDA/HIP build rule for the first campaign
Runtime status schema
Memory ownership and lifetime
Transfer semantics
Kernel implementation rules
Error handling and synchronization
Python interop and DLPack policy
Testing ladder
Benchmark and profiling evidence
Release and packaging boundaries
```

The document must explicitly state:

```text
No ROCm headers in public FastPauli headers.
No import-time ROCm runtime dependency.
No ROCm wheel claim without a separate release-packaging campaign.
No simultaneous CUDA+HIP runtime object support in the first MI300X campaign.
No performance claim without transfer-inclusive and device-resident evidence.
```

- [ ] **Step 2: Link the contract from hardware targets**

Add `docs/architecture/rocm_backend.md` to the ROCm/HIP paragraph in
`docs/architecture/hardware_targets_and_testing.md`.

- [ ] **Step 3: Add post-CUDA gates**

Add gates to `docs/quality/phase_quality_gates.md` requiring the ROCm backend
contract, MI300X inventory evidence, HIP source build, CPU/HIP equivalence
tests, rocprof evidence or blocked-profiler diagnosis, and benchmark report
before any ROCm support claim.

- [ ] **Step 4: Validate docs**

Run:

```bash
git diff --check
```

Expected: command exits 0.

- [ ] **Step 5: Commit**

```bash
git add docs/architecture/rocm_backend.md docs/architecture/hardware_targets_and_testing.md docs/quality/phase_quality_gates.md
git commit -m "docs: define ROCm backend contract"
```

## Task 3: Build-System HIP Foundation

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `bindings/python/module.cpp`
- Modify: `src/device_pauli_sum_stub.cpp`
- Modify: `src/device_commutation_matrix_stub.cpp`
- Modify or create: `src/hip/device_pauli_sum.hip.cpp`
- Modify or create: `src/hip/device_pauli_sum.hip.hpp`
- Create: `tests/test_phase12_rocm_foundation.py`

- [ ] **Step 1: Write CPU-only and configure-rule tests**

Create `tests/test_phase12_rocm_foundation.py` with tests covering:

```python
def test_cpu_only_build_reports_hip_absence() -> None:
    import fastpauli._fastpauli_core as core

    info = core._build_info()
    assert info["hip_enabled"] is False
    assert info["hip_architectures"] == "not_available"
    assert info["rocm_toolkit_version"] == "not_available"
    assert info["hip_runtime_available"] is False
    assert core._hip_status()["built"] is False
    assert core._accelerator_status()["active_backend"] == "none"
```

Also add a subprocess CMake configure test that passes both
`-DFASTPAULI_ENABLE_CUDA=ON` and `-DFASTPAULI_ENABLE_HIP=ON` and expects a
clear configure-time failure containing:

```text
FASTPAULI_ENABLE_CUDA and FASTPAULI_ENABLE_HIP cannot both be ON
```

- [ ] **Step 2: Run the new CPU-only tests and verify they fail**

Run:

```bash
python -m pytest tests/test_phase12_rocm_foundation.py -q
```

Expected: failure because `_hip_status`, `_accelerator_status`, and HIP build
metadata do not exist yet.

- [ ] **Step 3: Add CMake options and metadata**

In `CMakeLists.txt`, add:

```cmake
option(FASTPAULI_ENABLE_HIP "Build ROCm/HIP backend support" OFF)
set(FASTPAULI_HIP_ARCHITECTURES "gfx942" CACHE STRING "HIP architectures for FASTPAULI_ENABLE_HIP=ON source builds")

if(FASTPAULI_ENABLE_CUDA AND FASTPAULI_ENABLE_HIP)
  message(FATAL_ERROR "FASTPAULI_ENABLE_CUDA and FASTPAULI_ENABLE_HIP cannot both be ON in the first ROCm campaign.")
endif()
```

When HIP is enabled, use CMake HIP language support, record compiler and
architecture metadata, and append only `src/hip/*.hip.cpp` implementation files
to the extension. CPU-only and CUDA-only source lists must remain unchanged.

- [ ] **Step 4: Add Python metadata functions**

In `bindings/python/module.cpp`, add `_hip_status()` and
`_accelerator_status()` bindings. Extend `_build_info()` with:

```text
hip_enabled
hip_architectures
rocm_toolkit_version
hip_runtime_available
hip_kernels
compiler_build_config.CMAKE_HIP_COMPILER_ID
compiler_build_config.CMAKE_HIP_COMPILER_VERSION
```

CPU-only builds must return explicit `"not_available"` strings rather than
omitting keys.

- [ ] **Step 5: Run CPU-only validation locally**

Run:

```bash
python scripts/validate.py
```

Expected: validation passes on macOS CPU with HIP disabled.

- [ ] **Step 6: Run HIP configure on MI300X**

Run on MI300X:

```bash
python -m pip install -e .[test] \
  --config-settings=cmake.define.FASTPAULI_ENABLE_HIP=ON \
  --config-settings=cmake.define.FASTPAULI_HIP_ARCHITECTURES=gfx942
```

Expected: extension builds with HIP enabled or fails with a concrete ROCm/CMake
diagnosis recorded in the campaign report.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt bindings/python/module.cpp src/device_pauli_sum_stub.cpp src/device_commutation_matrix_stub.cpp src/hip tests/test_phase12_rocm_foundation.py
git commit -m "build: add HIP backend foundation"
```

## Task 4: HIP Runtime Status And Transfer Round-Trip

**Files:**
- Modify: `include/fastpauli/device_pauli_sum.hpp`
- Modify: `bindings/python/pauli_sum_py.cpp`
- Modify: `src/hip/device_pauli_sum.hip.cpp`
- Modify: `tests/test_phase12_rocm_foundation.py`

- [ ] **Step 1: Add transfer tests**

Extend `tests/test_phase12_rocm_foundation.py` with HIP-gated tests:

```python
def test_hip_round_trip_when_available() -> None:
    import pytest
    import fastpauli
    import fastpauli._fastpauli_core as core

    status = core._hip_status()
    if not status["runtime_available"]:
        pytest.skip(status["skip_reason"])

    op = fastpauli.PauliSum.from_labels(["XIZ", "YYI"], [1.0, -2.0j])
    device_op = op.to_device(device=0)
    assert device_op.backend == "hip"
    assert device_op.device == 0
    assert device_op.to_host().to_labels() == op.to_labels()


def test_hip_empty_round_trip_when_available() -> None:
    import pytest
    import fastpauli
    import fastpauli._fastpauli_core as core

    status = core._hip_status()
    if not status["runtime_available"]:
        pytest.skip(status["skip_reason"])

    host = fastpauli.PauliSum.empty(num_qubits=5)
    actual = host.to_device(device=0).to_host()
    assert actual.num_terms == 0
    assert actual.num_qubits == 5
```

- [ ] **Step 2: Run tests on MI300X and verify transfer failures before implementation**

Run:

```bash
python -m pytest tests/test_phase12_rocm_foundation.py -q
```

Expected: HIP transfer tests fail because the HIP implementation is not present
or returns a not-implemented error.

- [ ] **Step 3: Implement HIP status**

Implement HIP runtime status with `hipGetDeviceCount`, `hipGetDeviceProperties`,
and runtime/driver version calls available in the installed ROCm version. The
status must never throw during `_hip_status()` unless an internal invariant is
violated; missing devices become `runtime_available=False` with a skip reason.

- [ ] **Step 4: Implement HIP memory ownership**

Implement a HIP-only `DevicePauliSum::Impl` that owns:

```text
device ordinal
num_qubits
num_terms
words
uint64_t* x
uint64_t* z
complex coefficient storage compatible with HIP kernels
```

Use RAII cleanup with `hipFree` and deterministic moved-from checks matching
the CUDA behavior.

- [ ] **Step 5: Implement host/device copies**

Use `hipSetDevice`, `hipMalloc`, and `hipMemcpy` for host-to-device and
device-to-host copies. Empty operators must not require non-null allocations.
All HIP errors must be translated into `std::runtime_error` messages that name
the failed operation.

- [ ] **Step 6: Run HIP transfer tests**

Run on MI300X:

```bash
python -m pytest tests/test_phase12_rocm_foundation.py -q
```

Expected: all CPU-only tests pass and HIP transfer tests pass when a visible
MI300X is available.

- [ ] **Step 7: Commit**

```bash
git add include/fastpauli/device_pauli_sum.hpp bindings/python/pauli_sum_py.cpp src/hip tests/test_phase12_rocm_foundation.py
git commit -m "feat: add HIP PauliSum transfers"
```

## Task 5: First HIP Kernel

**Files:**
- Create or modify: `src/hip/commutation_hip.hip.cpp`
- Create or modify: `src/hip/commutation_hip.hip.hpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/test_phase12_rocm_foundation.py`
- Create: `benchmarks/bench_rocm_kernels.py`

- [ ] **Step 1: Select the first retained HIP kernel**

Implement pairwise commutation first. It has the best risk/reward balance for
MI300X bring-up because it is bitwise, maps directly to the sparse Pauli packed
layout, avoids rocThrust dependency during the first kernel, and already has
strong CPU/CUDA correctness coverage.

- [ ] **Step 2: Add HIP commutation tests**

Add deterministic and randomized tests comparing:

```text
PauliSum.commutes_with(rhs)
PauliSum.to_device().commutes_with(rhs.to_device())
```

Datasets must include:

```text
empty lhs
empty rhs
single-word operators
multi-word operators
same-device success
different-device rejection when multiple devices exist
max_commutation_matrix_entries guard
random labels with deterministic seeds
```

- [ ] **Step 3: Verify tests fail before kernel implementation**

Run on MI300X:

```bash
python -m pytest tests/test_phase12_rocm_foundation.py -q
```

Expected: commutation tests fail because the HIP kernel is not implemented.

- [ ] **Step 4: Implement the kernel**

Add a HIP kernel equivalent to the CUDA pairwise commutation kernel:

```text
one output byte per lhs/rhs pair
row-major output over lhs terms then rhs terms
commutation parity computed by popcount(x_lhs & z_rhs) xor popcount(z_lhs & x_rhs)
output byte is 1 when parity is even, 0 otherwise
grid-stride loop over total pair count
host wrapper validates qubit count, device, moved-from state, output size, and max-entry guard before launch
host wrapper synchronizes before returning host output
```

- [ ] **Step 5: Run correctness tests**

Run on MI300X:

```bash
python -m pytest tests/test_phase12_rocm_foundation.py tests/test_phase6_commutation_grouping.py -q
```

Expected: HIP-gated commutation tests pass and CPU commutation/grouping tests
remain unchanged.

- [ ] **Step 6: Add smoke benchmark**

Create `benchmarks/bench_rocm_kernels.py` with a smoke profile that reports:

```text
operation
dataset
num_qubits
lhs_terms
rhs_terms
words
backend
hip_architectures
device_name
transfer_inclusive_seconds
device_resident_seconds
cpu_scalar_seconds
available_cpu_selector_seconds
correctness_passed
```

The smoke profile must complete quickly on MI300X and still run in CPU-only
mode by reporting HIP as unavailable with a reason.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt src/hip tests/test_phase12_rocm_foundation.py benchmarks/bench_rocm_kernels.py
git commit -m "feat: add HIP commutation kernel"
```

## Task 6: MI300X Benchmarking And Profiling

**Files:**
- Modify: `benchmarks/_benchmark_metadata.py`
- Modify: `benchmarks/bench_rocm_kernels.py`
- Create during execution: `docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/raw/rocm_commutation_smoke_mi300x.json`
- Create during execution: `docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/raw/rocm_commutation_scaling_mi300x.json`
- Create during execution: `docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/raw/rocm_commutation_profiler_mi300x.json`
- Create during execution: `docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/profiler/`
- Create during execution: `docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/summary.json`

- [ ] **Step 1: Run smoke benchmark**

Run on MI300X:

```bash
python benchmarks/bench_rocm_kernels.py --smoke --repeat 3 --json \
  --output docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/raw/rocm_commutation_smoke_mi300x.json
```

Expected: JSON records correctness, CPU baseline timing, HIP timing, active
backend metadata, and unavailable baseline reasons.

- [ ] **Step 2: Run scaling benchmark**

Run on MI300X:

```bash
python benchmarks/bench_rocm_kernels.py --profile commutation-scaling --repeat 5 --warmup 2 --json \
  --output docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/raw/rocm_commutation_scaling_mi300x.json
```

Expected: scaling rows cover small transfer-bound datasets, mid-sized datasets,
and large dense-pair datasets that stress MI300X memory bandwidth.

- [ ] **Step 3: Capture rocprof trace**

Run on MI300X:

```bash
mkdir -p docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/profiler
rocprof -d docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/profiler --hip-trace --stats \
  python benchmarks/bench_rocm_kernels.py --profile commutation-profiler --repeat 1 --warmup 0 --json \
  --output docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/raw/rocm_commutation_profiler_mi300x.json
```

Expected: profiler output includes HIP API timing, copy timing, and kernel
timing. If rocprof is unavailable or blocked, capture the exact command output
and classify the profiler status as `blocked_tooling` or `blocked_permissions`.

- [ ] **Step 4: Capture performance counters when supported**

Run:

```bash
rocprof --list-derived > docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/logs/rocprof_list_derived_mi300x.log 2>&1
rocprof --list-basic > docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/logs/rocprof_list_basic_mi300x.log 2>&1
```

Select MI300X-supported memory, occupancy, and wavefront counters from those
logs. Record the exact selected counter set in the report before running the
counter capture.

- [ ] **Step 5: Build summary JSON**

Generate `summary.json` with:

```text
campaign name
git revision
host inventory path
raw benchmark paths
profiler artifact paths
validation log paths
status for each acceptance criterion
limitations
```

- [ ] **Step 6: Commit evidence**

```bash
git add benchmarks/_benchmark_metadata.py benchmarks/bench_rocm_kernels.py docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29
git commit -m "bench: capture MI300X HIP evidence"
```

## Task 7: Report, README, Review, And Closeout

**Files:**
- Create: `docs/benchmarks/reports/rocm_mi300x_bringup_2026-04-29.md`
- Modify: `README.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/benchmarks/protocol.md` if a new ROCm-specific evidence field is added

- [ ] **Step 1: Write the report**

The report must include:

```text
scope and non-goals
host and ROCm inventory
build commands and compiler metadata
correctness validation commands and outcomes
benchmark commands and raw artifact links
transfer-inclusive and device-resident timing tables
CPU scalar and optimized CPU comparison rows
external baseline availability or unavailable reasons
rocprof trace and counter evidence
correctness risks
performance interpretation
remaining headroom
release/support claims that remain out of scope
```

- [ ] **Step 2: Update README only with evidence-backed claims**

Add a short ROCm/MI300X status paragraph to README after the CUDA performance
section only if the HIP build and retained kernel pass. The paragraph must link
to the report and must state that ROCm/HIP is source-build evidence, not a
wheel claim.

- [ ] **Step 3: Update roadmap**

Mark the MI300X bring-up campaign as complete or blocked with terminal evidence.
If blocked, record the next exact unblock condition instead of leaving the
status open.

- [ ] **Step 4: Run full local validation**

Run locally after the MI300X branch is back on the workstation:

```bash
python scripts/validate.py
```

Expected: local CPU validation passes with HIP disabled.

- [ ] **Step 5: Run MI300X validation**

Run on MI300X:

```bash
python -m pytest
python -m pytest tests/test_phase12_rocm_foundation.py -q
python benchmarks/bench_rocm_kernels.py --smoke --repeat 1 --json
python -m build --sdist --outdir _skbuild/validate-dist
```

Expected: all non-HIP tests pass, HIP-gated tests pass when runtime is
available, benchmark smoke passes, and sdist builds.

- [ ] **Step 6: Complete independent review**

Follow `docs/quality/code_review.md`. The review must cover:

```text
CMake HIP/CUDA exclusivity
public API compatibility
runtime status behavior without a GPU
HIP memory ownership and synchronization
HIP error messages
correctness tests
benchmark metadata
README/report claim wording
```

- [ ] **Step 7: Merge, push, and clean up**

Run:

```bash
git switch main
git merge --ff-only codex/mi300x-rocm-foundation
python scripts/validate.py
git push origin main
git branch -d codex/mi300x-rocm-foundation
```

Expected: merged `main` validates locally, pushes successfully, and CI is
confirmed green when available.

## Go/No-Go Gates

Proceed from one gate to the next only when the current gate has evidence:

```text
Gate A: MI300X host inventory confirms ROCm/HIP and gfx942 visibility.
Gate B: HIP source build configures and imports without breaking CPU-only validation.
Gate C: HIP status and transfer round-trip pass.
Gate D: HIP pairwise commutation passes CPU/GPU equivalence.
Gate E: HIP benchmark and rocprof evidence are captured or blocked with a concrete provider/tooling diagnosis.
Gate F: report, README, roadmap, review, merge, push, and CI closeout are complete.
```

If Gate A or Gate B fails because of provider image or ROCm installation
problems, stop implementation changes after recording the blocker and fix the
instance/toolchain first. If Gate D shows the HIP kernel is slower than CPU on
small transfer-inclusive workloads, keep the result; the report must identify
CPU-faster, HIP-faster, and transfer-bound regimes instead of tuning away
correctness or changing the public interface.

## Future Work After This Campaign

Only plan the next HIP campaign after this one lands. Likely next options:

```text
HIP expectation_statevector kernel when statevector workloads justify ROCm support
HIP device-resident compact commutation summaries if MI300X profiling shows host materialization dominates
HIP simplify if rocThrust or custom radix/sort strategy is justified by benchmark evidence
HIP matmul only after transfer, commutation, and expectation evidence are stable
backend-neutral multi-accelerator DevicePauliSum only when simultaneous CUDA+HIP builds are required
ROCm source-build packaging documentation after MI300X source validation is repeatable
Metal/MPS design decision after HIP evidence is in place
```

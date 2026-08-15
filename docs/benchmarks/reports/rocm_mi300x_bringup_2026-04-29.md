# ROCm MI300X Bring-Up Report

Date: 2026-04-29

This report records the first FastPauli ROCm/HIP bring-up campaign on a single
AMD Instinct MI300X. The result is source-build evidence for a HIP backend
foundation, host/device PauliSum transfers, and a retained HIP pairwise
commutation kernel. It is not a ROCm wheel support claim, not a multi-GPU claim,
and not a claim that CUDA and HIP can be enabled in the same build.

## Evidence Map

```text
plan: docs/plans/mi300x_rocm_bringup_plan.md
architecture: docs/architecture/rocm_backend.md
summary: docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/summary.json
raw data: docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/raw/
logs: docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/logs/
profiler artifacts: docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/profiler/
code revision under benchmark: fa8970e
```

The checked summary gives every in-scope acceptance criterion a terminal
`passed` status. The report uses committed artifacts only.

## Scope

In scope:

```text
single-node 1x MI300X bring-up
ROCm/HIP source build behind FASTPAULI_ENABLE_HIP=ON
FASTPAULI_HIP_ARCHITECTURES=gfx942 source-build path
runtime status and build metadata reporting
PauliSum.to_device().to_host() round trips on HIP-only builds
HIP pairwise commutation correctness and benchmark evidence
rocprof trace and counter evidence for the retained HIP kernel
README, roadmap, benchmark-protocol, and validation updates tied to evidence
```

Out of scope:

```text
ROCm wheels
8x MI300X or distributed ROCm work
Metal/MPS implementation
simultaneous CUDA+HIP runtime objects
public HIP stream, graph, DLPack, or external workspace APIs
additional AMD GPU architecture lanes
```

## Host And Build Inventory

| Field | Captured value |
| --- | --- |
| Host OS | Ubuntu 24.04.4 LTS, Linux 6.8.0-106-generic |
| CPU | Intel Xeon Platinum 8568Y+, 20 logical CPUs visible |
| GPU | AMD Instinct MI300X VF |
| GFX target | `gfx942:sramecc+:xnack-` |
| VRAM | 205,822,885,888 bytes |
| HIP runtime | 7.2.26015 |
| HIP driver | 7.2.26015 |
| HIP toolkit version | 7.2.26015-fc0010cf6a |
| HIP compiler | `/opt/rocm/bin/amdclang++`, Clang 22.0.0 |
| FastPauli HIP architecture | `gfx942` |
| CPU selectors available on host | scalar, oneTBB, AVX2, AVX-512 |

Inventory evidence:

```text
docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/logs/host_inventory_mi300x.log
docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/logs/build_status_mi300x.json
```

The HIP source build used the ROCm clang compiler path rather than hardcoding
the `hipcc` wrapper as `CMAKE_HIP_COMPILER`. CPU-only local builds keep
`FASTPAULI_ENABLE_HIP=OFF`.

## Implementation Outcome

| Area | Terminal status | Evidence |
| --- | --- | --- |
| Backend architecture | passed | `docs/architecture/rocm_backend.md` |
| Source layout | passed | `src/hip/device_pauli_sum.hip.*`, `src/hip/commutation_hip.hip.*` |
| CUDA/HIP coexistence guard | passed | CMake rejects `FASTPAULI_ENABLE_CUDA=ON` with `FASTPAULI_ENABLE_HIP=ON` |
| Public header isolation | passed | Public headers do not include ROCm or HIP runtime headers |
| Runtime status | passed | `_hip_status()` and `_accelerator_status()` report HIP-only state |
| Transfers | passed | Non-empty and empty `PauliSum.to_device().to_host()` round trips |
| HIP kernel | passed | `DevicePauliSum.commutes_with()` CPU/GPU equivalence |
| Unsupported HIP surfaces | passed | HIP `commutes_with_device()`, streams, DLPack, and workspace APIs remain out of scope |

The first retained HIP kernel is pairwise commutation. It uses the existing
packed `x`/`z` word representation, launches a grid-stride HIP kernel, computes
symplectic parity with popcount, synchronizes before returning, and materializes
the dense `uint8` result on the host.

## Validation

Final local CPU-only validation after the report and source-of-truth updates
passed:

```text
.venv/bin/python scripts/validate.py
199 passed, 67 skipped, benchmark smokes passed, sdist smoke passed
```

Evidence: `logs/final_local_validate_macos_m4pro.log`.

Final remote MI300X full pytest passed:

```text
python -m pytest
205 passed, 61 skipped
```

Evidence: `logs/final_remote_pytest_mi300x.log`.

Final remote MI300X targeted validation passed:

```text
python -m pytest tests/test_phase12_rocm_foundation.py tests/test_phase6_commutation_grouping.py -q
22 passed, 1 skipped
```

Evidence: `logs/final_remote_rocm_targeted_mi300x.log`.

The remote ROCm benchmark smoke and sdist smoke passed:

```text
python benchmarks/bench_rocm_kernels.py --smoke --repeat 3 --warmup 1 --json
python -m build --sdist --outdir _skbuild/validate-dist
Successfully built fastpauli-0.1.0.tar.gz
```

Evidence: `logs/final_remote_rocm_smoke_mi300x.log` and
`logs/final_remote_sdist_mi300x.log`.

The HIP source build evidence after the code-revision commit is recorded in
`logs/final_remote_hip_source_build_mi300x.log`.

The skipped test requires at least two visible HIP devices to verify
different-device rejection. One MI300X device was visible on this host, so the
skip is expected and recorded.

## Benchmarks

Benchmark commands:

```text
python benchmarks/bench_rocm_kernels.py --smoke --repeat 3 --warmup 1 --json --output docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/raw/rocm_commutation_smoke_mi300x.json
python benchmarks/bench_rocm_kernels.py --profile commutation-scaling --repeat 5 --warmup 2 --json --output docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/raw/rocm_commutation_scaling_mi300x.json
python benchmarks/bench_rocm_kernels.py --profile commutation-profiler --repeat 1 --warmup 0 --json --output docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/raw/rocm_commutation_profiler_mi300x.json
```

The HIP "device-resident" timing boundary reuses device operands but still
materializes the dense output matrix to host. A retained HIP device-output
matrix is out of scope for this campaign.

| Case | LHS x RHS | Qubits | CPU scalar | oneTBB | AVX2 | AVX-512 | HIP transfer-inclusive | HIP device-operand |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small_transfer_bound | 128 x 128 | 8 | 66.70 us | 74.41 us | 19.85 us | 11.75 us | 117.03 us | 34.31 us |
| mid_dense_pairs | 2048 x 2048 | 64 | 14.14 ms | 1.45 ms | 3.63 ms | 1.72 ms | 378.48 us | 244.70 us |
| large_dense_pairs | 4096 x 4096 | 128 | 198.09 ms | 13.17 ms | 29.11 ms | 20.92 ms | 853.11 us | 566.05 us |

Interpretation:

```text
small rows remain transfer-bound and can lose to AVX-512 when host/device copies dominate
mid and large dense pairwise rows strongly favor HIP even with host result materialization
optimized CPU selector rows are retained in the same dataset so the HIP result is not compared only against scalar CPU
```

No ROCm external sparse-Pauli primitive baseline was retained in this campaign.
The report therefore does not compare against a separate GPU package. Future
ROCm baseline work must record version, installation method, device enablement,
semantic mapping, timing boundary, correctness oracle, and unavailable reasons.

## rocprof Evidence

Trace and counter commands retained:

```text
PATH=/opt/rocm/bin:$PATH rocprof -d docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/profiler --hip-trace --stats python benchmarks/bench_rocm_kernels.py --profile commutation-profiler --repeat 1 --warmup 0 --json --output docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/raw/rocm_commutation_profiler_mi300x.json
PATH=/opt/rocm/bin:$PATH rocprof -i docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/profiler/rocprof_commutation_counters.txt -o docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/profiler/rocm_commutation_counters.csv -d docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/profiler python benchmarks/bench_rocm_kernels.py --profile commutation-profiler --repeat 1 --warmup 0 --json --output docs/benchmarks/data/rocm_mi300x_bringup_2026-04-29/raw/rocm_commutation_profiler_mi300x.json
```

The first legacy rocprof trace attempt exited nonzero because `rocminfo` was
not on `PATH` during postprocessing. The diagnostic log was refreshed after the
`fa8970e` code-revision fix and is retained as a failure-mode artifact. The
path-fixed run succeeded and produced trace, stats, copy stats, HIP API stats,
sysinfo, and counter artifacts. Older `rpl_data_260430_000941_*`,
`rpl_data_260430_001010_*`, and `rpl_data_260430_001104_*` rocprof spill
directories plus older top-level `results.*` exports remain as original
pre-fix diagnostic files only; the accepted profiler evidence for this
closeout is the refreshed top-level rocprof logs, refreshed
`rocm_commutation_counters.csv`, refreshed raw profiler JSON, and refreshed
`rpl_data_260430_004*` rocprof spill directories for `fa8970e`.

Representative counter rows for `commutation_kernel`:

| Metric | Value range |
| --- | ---: |
| Grid size | 16,777,216 |
| Workgroup size | 256 |
| Wave size | 64 |
| Arch VGPRs | 28 |
| SGPRs | 48 |
| MeanOccupancyPerCU | 25.47 to 25.58 |
| VALUUtilization | 100.00 |
| VALUBusy | 64.48 to 64.85 |
| MemUnitStalled | 26.69 to 27.01 |
| WriteSize | 16,384.00 |

HIP API stats show the profiler run is dominated by host-visible transfer
costs: `hipMemcpy` accounts for about 98.0 percent of HIP API duration under
rocprof instrumentation, while `hipLaunchKernel` accounts for about 0.94
percent. Copy stats show device-to-host output materialization dominates copy
time for the profiler case.

## Acceptance Criteria

| Acceptance item | Status |
| --- | --- |
| MI300X host inventory captured | passed |
| CPU-only local validation still passes with HIP disabled | passed |
| HIP source build succeeds on MI300X with `gfx942` | passed |
| CUDA+HIP configure-time rejection is clear | passed |
| `_build_info()` reports HIP metadata | passed |
| `_hip_status()` reports runtime and MI300X device metadata | passed |
| `_accelerator_status()` reports HIP-only state | passed |
| HIP non-empty and empty transfers round trip | passed |
| HIP invalid-device and moved-state errors are deterministic | passed |
| HIP kernel passes deterministic and randomized CPU/GPU equivalence | passed |
| Benchmarks report CPU scalar, optimized CPU selectors, and HIP timings | passed |
| rocprof trace or counter evidence is captured | passed |
| README and roadmap are updated only with evidence-backed claims | passed in this closeout |
| Independent review is recorded before merge | passed after review fixes |

## Review Closeout

Two independent reviewer agents inspected the MI300X ROCm/HIP closeout before
merge. The first review round found three blocking classes of issues:

```text
roadmap completion language was ahead of the review gate
final validation pass counts were not backed by checked log artifacts
ROCm toolkit metadata was mislabeled as the HIP compiler version
```

The second review path found one additional CUDA-regression risk:

```text
the ROCm foundation test assumed active_backend == "none" whenever HIP was disabled, which would fail CUDA-only validation
```

All blocking findings were resolved before merge:

```text
final local and MI300X validation logs are checked under the evidence root
summary.json links the final validation logs
CMake extracts HIP toolkit metadata from hipcc --version instead of CMAKE_HIP_COMPILER_VERSION
the CUDA/HIP active-backend test accepts CUDA-only active_backend == "cuda" when CUDA runtime is available
the MI300X raw benchmark JSON was refreshed after the committed code-revision fix
```

## Correctness Risks

The current HIP surface is intentionally small. The highest correctness risks
are future expansion risks rather than unresolved defects in the retained slice:

```text
simultaneous CUDA+HIP builds still require a backend-neutral device object design
HIP stream or async APIs require explicit lifetime and synchronization contracts
HIP DLPack or array interop requires ownership and stream semantics before exposure
device-output HIP commutation matrices require a public result lifetime boundary
multi-GPU MI300X behavior is untested because only one visible device was used
```

## Remaining Headroom

Further ROCm work should be planned as separate slices:

```text
add a retained HIP device-output commutation matrix only after accepting its public lifetime contract
profile HIP output materialization alternatives, including packed or summary result consumers, before exposing new APIs
evaluate rocThrust or custom duplicate-reduction paths for HIP simplify once transfer and commutation surfaces are stable
add a ROCm external baseline only where the compared package has a semantically matched sparse-Pauli primitive or a clearly labeled framework-level boundary
run a second MI300X lane or a different AMD GPU architecture only when portability evidence is needed for release claims
```

## Release Claim

The accepted claim after this campaign is:

```text
FastPauli has source-build ROCm/HIP evidence on one MI300X for backend metadata, host/device transfers, and pairwise commutation.
```

The rejected claims are:

```text
FastPauli ships ROCm wheels
FastPauli supports every AMD GPU
FastPauli supports simultaneous CUDA+HIP builds
FastPauli exposes HIP stream, graph, DLPack, workspace, or device-output matrix APIs
```

# Backend-Neutral Accelerator Campaign 9 Closeout Report

Date: 2026-05-01

## Scope

Campaign 9 implements and closes the backend-neutral accelerator object model
for target-specific builds. FastPauli now exposes one public construction
contract across CPU, CUDA, and ROCm/HIP while normal source builds compile at
most one accelerator runtime:

```text
CPU-only
CUDA-only source build
HIP-only source build
```

`FASTPAULI_ENABLE_CUDA=ON` with `FASTPAULI_ENABLE_HIP=ON` remains a deliberate
configure-time error under the target-specific accelerator policy. This report
does not claim ROCm wheels, CUDA wheels, combined accelerator wheels,
non-MI300X AMD portability, HIP DLPack, HIP CUDA Array Interface, multi-GPU
execution, Metal/MPS support, public streams, public graphs, or public
workspaces.

## Evidence

Checked evidence:

```text
docs/benchmarks/data/backend_neutral_accelerator_campaign9_2026-05-01/
docs/benchmarks/data/backend_neutral_accelerator_campaign9_2026-05-01/summary.json
docs/benchmarks/data/backend_neutral_accelerator_campaign9_2026-05-01/logs/local_validate_apple_m4pro.log
docs/benchmarks/data/backend_neutral_accelerator_campaign9_2026-05-01/logs/h100_cuda_validate.log
docs/benchmarks/data/backend_neutral_accelerator_campaign9_2026-05-01/logs/mi300x_hip_validate.log
docs/benchmarks/data/backend_neutral_accelerator_campaign9_2026-05-01/logs/local_cuda_hip_rejection.log
```

The validation lanes were captured at commit `ec8fd19`. Later closeout commits
add the checked summary, report, source-of-truth updates, and tests; they do
not change the CUDA or HIP kernel implementations that generated this evidence.

## Host And Build Matrix

| Lane | Host class | CPU | Accelerator | Build mode | Target |
|---|---|---|---|---|---|
| Local CPU control | Apple Silicon development host | Apple M4 Pro, 12 cores | none | `cpu_only` | `cpu` |
| CUDA target | NVIDIA validation host | AMD EPYC 9654, 32 vCPU | NVIDIA H100 80GB HBM3 | `cuda_only` | CUDA `sm_90` |
| HIP target | MI300X ROCm validation host | Intel Xeon Platinum 8568Y+, 20 vCPU | AMD Instinct MI300X VF | `hip_only` | `gfx942` |
| Dual-request rejection | Apple Silicon development host | Apple M4 Pro, 12 cores | none | rejected configure | CUDA+HIP request |

CUDA evidence records CUDA toolkit `12.9.86`, CUDA runtime `12.9`, CUDA driver
API `13.0`, NVIDIA driver `580.126.09`, and device compute capability `9.0`.
The validation ladder built both the default architecture set
`70,75,80,86,89,90` and the requested H100 override `90`.

ROCm/HIP evidence records ROCm toolkit `7.2.26015-fc0010cf6a`, HIP runtime and
driver `7.2.26015`, HIP compiler `/opt/rocm/bin/amdclang++` with Clang
`22.0.0`, MI300X target `gfx942:sramecc+:xnack-`, and requested architecture
`gfx942`.

## Validation

Local CPU-only control:

```bash
UV_CACHE_DIR=/private/tmp/fastpauli-uv-cache uv run python scripts/validate.py
```

Observed result:

```text
242 passed, 89 skipped in 18.06s
Successfully built fastpauli-0.1.0.tar.gz
compiled_backends: ["cpu"]
runtime_visible_backends: ["cpu"]
```

CUDA target lane:

```bash
PATH=/usr/local/cuda/bin:/root/FastPauli/.venv/bin:$PATH \
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 \
.venv/bin/python scripts/validate.py
```

Observed result:

```text
CUDA-enabled semantic pytest: 264 passed, 67 skipped in 11.64s
CUDA transfer pytest: 6 passed in 0.37s
CUDA kernel pytest: 30 passed, 8 skipped in 1.50s
CUDA kernel benchmark smoke: passed under scripts/validate.py
CUDA scaling benchmark smoke: passed under scripts/validate.py
Successfully built fastpauli-0.1.0.tar.gz
compiled_accelerator_backends: ["cuda"]
runtime_visible_accelerator_backends: ["cuda"]
```

HIP target lane:

```bash
PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:/root/FastPauli/.venv/bin:$PATH \
FASTPAULI_VALIDATE_HIP=1 FASTPAULI_HIP_ARCHITECTURES=gfx942 \
.venv/bin/python scripts/validate.py
```

Observed result:

```text
HIP-enabled semantic pytest: 270 passed, 61 skipped in 13.44s
HIP foundation pytest: 37 passed, 2 skipped in 3.78s
ROCm kernel benchmark smoke on HIP build: correctness_passed true
Successfully built fastpauli-0.1.0.tar.gz
compiled_accelerator_backends: ["hip"]
runtime_visible_accelerator_backends: ["hip"]
```

Dual-request rejection lane:

```bash
UV_CACHE_DIR=/private/tmp/fastpauli-uv-cache uv run python -m cmake \
  -S . -B /tmp/fastpauli-campaign9-cuda-hip-reject \
  -DFASTPAULI_ENABLE_CUDA=ON -DFASTPAULI_ENABLE_HIP=ON
```

Observed result:

```text
exit_code=1
FASTPAULI_ENABLE_CUDA and FASTPAULI_ENABLE_HIP cannot both be ON under the
target-specific accelerator build policy.
```

## Acceptance Outcome

| Item | Status | Evidence |
|---|---|---|
| `backend_neutral_status_schema` | passed | `_accelerator_status()` and `_build_info()` structured backend sets |
| `object_local_backend_identity` | passed | CUDA and HIP object tests under target builds |
| `backend_construction_selector_contract` | passed | `PauliSum.to_device(backend=None|"auto"|"cuda"|"hip")` |
| `device_commutation_matrix_backend_property` | passed | CUDA and HIP matrix objects report backend identity |
| `ambiguous_dual_runtime_policy` | passed_by_cpu_simulation | mixed-runtime ambiguity remains future-only |
| `target_specific_accelerator_builds` | passed | CPU, CUDA, and HIP lanes validated separately |
| `mixed_cuda_hip_build_rejection` | passed | configure-time rejection evidence captured |
| `future_multi_runtime_design_gate` | retained | requires a later accepted mixed-runtime plan |
| `same_backend_same_device_validation` | passed | existing CUDA and HIP tests retained |
| `cpu_only_header_safety` | passed | source-shape tests and CPU-only build validation retained |
| `cuda_target_regression_lane` | passed | H100 CUDA validation lane |
| `hip_target_regression_lane` | passed | MI300X HIP validation lane |
| `benchmark_boundary_reporting` | passed_status_and_smoke_evidence | status-only and benchmark-smoke metadata captured |
| `no_wheel_or_portability_claim` | retained | README and architecture docs keep support boundaries explicit |

## Benchmark Boundary Outcome

Campaign 9 rows are support-boundary evidence, not new optimization claims.
The checked metadata distinguishes:

```text
build_mode: cpu_only, cuda_only, hip_only
object_backend: cpu, cuda, hip
compiled_backends
runtime_visible_backends
transfer_boundary: status_only or benchmark-smoke operation boundary
```

The broad README accelerator landscape remains the user-facing performance
view. This campaign only records that target-specific CPU, CUDA, and HIP build
boundaries are observable and reproducible.

## Residual Risk And Next Triggers

No Campaign 9 acceptance item remains open for the target-specific backend
model. The remaining accelerator topics are separate future campaigns:

```text
combined CUDA+HIP runtime or wheel: requires a mixed-runtime architecture plan
non-MI300X AMD portability: requires a real non-MI300X AMD GPU lane
ROCm or CUDA wheels: require packaging, CI hardware, and clean-machine evidence
HIP DLPack or external HIP statevector interop: requires a real consumer contract
Metal/MPS: requires a separate Apple accelerator design and implementation plan
multi-GPU execution: requires separate device-placement and copy semantics
```

The next useful work is not another Campaign 9 validation rerun unless one of
those triggers appears or a release cut requires fresh source-build evidence.

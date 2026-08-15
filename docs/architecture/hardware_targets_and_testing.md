# Wolfgang Hardware Targets And Testing

This document defines Wolfgang CPU, CUDA, and post-CUDA accelerator target policy. It is the source of truth for build options, dispatch choices, hardware support claims, test ladders, and benchmark environment capture.

Wolfgang is CPU-first for correctness and CUDA-required for product scope. Apple Silicon CPU performance is a first-class CPU target. Hardware-specific paths must never make the portable CPU package unreliable, non-deterministic, or difficult to install.

## Support Vocabulary

Use these terms consistently:

```text
planned: designed into the architecture but not implemented
compile-tested: builds successfully for the target
runtime-tested: imports and executes correctness tests on the target
performance-tested: benchmarked with captured environment and command output
release-supported: documented for users because CI or release evidence covers it
```

Do not describe a platform, CPU feature, GPU backend, accelerator toolkit, GPU architecture, or wheel tag as supported unless it is release-supported. Planning targets are not user-facing support claims.

## CPU Baseline Policy

Every Wolfgang build must include a portable scalar CPU path. This path is the correctness oracle for optimized CPU paths and CUDA paths after the relevant semantics are locked.

CPU wheels must obey these rules:

```text
do not compile the whole extension with -march=native
do not require AVX, AVX2, AVX-512, BMI, or non-baseline x86 features at import time
include a scalar path that can be forced for tests and benchmarks
keep optional oneTBB, SIMD, and CUDA paths behind build and runtime gates
```

Source builds may enable local CPU features, but any locally tuned build must be clearly separated from portable wheel claims.

## Initial CPU Platform Targets

Initial CPU release targets are:

```text
Linux x86_64 CPU wheels
macOS arm64 CPU wheels
macOS x86_64 CPU wheels when CI and local validation are available
source distribution builds on the same platforms
```

Apple Silicon CPUs are first-class CPU correctness, packaging, and benchmark targets. Phase 4 and later benchmark reports should include at least one named Apple Silicon machine when local hardware is available, and Phase 9 CPU optimization claims must distinguish Apple Silicon results from x86_64 results.

Windows x86_64, Linux aarch64, musllinux, and universal2 macOS wheels are valuable follow-on targets, but they are not release-supported until they have build, import, and test evidence.

Linux wheels should use current PyPA manylinux policy. The selected manylinux tag must be recorded in release evidence and must not imply CPU instruction-set requirements beyond the wheel's platform tag.

## CPU Compiler Policy

Wolfgang uses C++20. Phase 1 must enforce the compiler matrix in CI and document any source-build lower bounds in `pyproject.toml`, CMake errors, or contributor docs.

Initial CPU compiler targets:

```text
Linux: GCC 11 or newer, or Clang 15 or newer
macOS: AppleClang from Xcode 15 or newer
Windows: MSVC deferred until Windows is validated
CUDA host compiler: whatever is supported by the selected CUDA toolkit
```

If Phase 1 discovers a narrower minimum through actual CI or package-build evidence, update this document and the release standards in the same slice.

## CPU Build Options

CMake options should start with conservative defaults:

```text
WOLFGANG_ENABLE_TBB=auto
WOLFGANG_ENABLE_OPENMP=OFF
WOLFGANG_ENABLE_AVX2=auto
WOLFGANG_ENABLE_AVX512=auto
WOLFGANG_ENABLE_ARM_NEON=auto
WOLFGANG_ENABLE_ARM_SVE=auto
WOLFGANG_ENABLE_NATIVE=OFF
WOLFGANG_ENABLE_CUDA=OFF
```

Rules:

```text
WOLFGANG_ENABLE_NATIVE=ON is forbidden for release wheels
WOLFGANG_ENABLE_TBB=auto may use oneTBB when found but must keep a scalar fallback
WOLFGANG_ENABLE_AVX2=auto and WOLFGANG_ENABLE_AVX512=auto may compile dispatched objects only when the compiler supports them
WOLFGANG_ENABLE_ARM_NEON=auto and WOLFGANG_ENABLE_ARM_SVE=auto may compile dispatched objects only when the compiler and target CPU support them
WOLFGANG_ENABLE_CUDA=OFF must build without CUDA headers, libraries, or toolkit discovery
```

The build must expose enough diagnostics to say which paths were compiled.

## CPU Runtime Dispatch

Runtime dispatch must be explicit and testable once optimized paths exist.

Required behavior:

```text
auto dispatch never executes instructions unavailable on the current CPU
scalar can be forced for correctness tests and benchmark baselines
each SIMD path can be forced when the hardware supports it
unsupported forced paths fail with a clear error instead of silently falling back
dispatch decisions are observable in benchmark environment capture
```

The initial runtime control should be an environment variable or test helper with this shape:

```text
WOLFGANG_CPU_BACKEND=auto
WOLFGANG_CPU_BACKEND=scalar
WOLFGANG_CPU_BACKEND=avx2
WOLFGANG_CPU_BACKEND=avx512
WOLFGANG_CPU_BACKEND=neon
```

Future SIMD extensions should use the same control surface. Planned selectors include:

```text
WOLFGANG_CPU_BACKEND=sve
```

If the final implementation chooses a Python API or C++ test hook instead, it must preserve the same semantics.

The current implementation keeps scalar as the universal baseline and enables
operation-level `auto` dispatch only for kernels with correctness and benchmark
coverage:

```text
pairwise commutation: oneTBB for large dense matrices when compiled, otherwise AVX-512, AVX2, NEON, or scalar
full grouping graph construction: AVX-512, AVX2, NEON, or scalar
other operations: scalar until a named optimized kernel lands with tests and benchmark evidence
```

Forced optimized selectors remain available for benchmarking and validation.
They must produce the same public results as forced scalar for covered kernels.
For scalar-only operations, forced optimized selectors must raise a clear error
instead of executing scalar code under an optimized backend label.

## CPU SIMD Policy

SIMD is introduced only after scalar correctness and benchmark baselines exist.

Initial x86 SIMD lanes:

```text
AVX2 path: requires AVX2 and any additional declared bit-operation features
AVX-512 path: requires the exact AVX-512 feature group used by the implementation
```

Do not treat all AVX-512 CPUs as equivalent. A path that relies on vector popcount, byte/word operations, or mask behavior must declare and check those feature bits.

Initial ARM SIMD policy:

```text
macOS arm64 uses the portable scalar path first and is benchmarked as a first-class CPU target
explicit NEON paths require benchmark evidence and forced-path tests
SVE paths are source-build only until Linux ARM hardware and CI exist
```

Apple Accelerate or vecLib may be evaluated only for kernels that map naturally to those libraries. It is not a required Wolfgang backend, and any Accelerate-backed path must preserve scalar semantics, pass forced-path tests, and include benchmark evidence before performance claims.

Compiler autovectorization is allowed for scalar code, but it is not a substitute for runtime-dispatched SIMD correctness tests once explicit SIMD paths are added.

Explicit SIMD is implemented for packed-word commutation kernels while keeping
the portable scalar path in every build. Current covered kernels are pairwise
commutation and full grouping graph construction for one- and two-word
operators. Wider operators use scalar logic unless a future SIMD kernel expands
coverage with forced-backend tests and benchmark evidence.

Current SIMD lanes:

```text
AVX-512: VPOPCNTDQ plus AVX-512F, AVX-512BW, and AVX-512VL
AVX2: nibble-lookup vector popcount for batched commutation parity
ARM NEON: byte popcount and pairwise widening sums for Apple Silicon commutation parity
```

## CPU Parallelism Policy

oneTBB is the first CPU parallelism backend. OpenMP remains optional and disabled by default until a specific benchmark or deployment need justifies it.

oneTBB paths must preserve externally visible ordering. Parallel simplify, multiplication, commutation, and expectation paths must match scalar output exactly except where a numeric tolerance is explicitly documented.

Benchmark reports must state:

```text
oneTBB enabled or disabled
oneTBB version when available
thread count
thread affinity or pinning when controlled
NUMA policy when controlled
```

oneTBB is implemented as an optional CMake-discovered backend for deterministic
commutation kernels. Current covered kernels are dense pairwise commutation and
full commutation graph construction. Pairwise commutation uses oneTBB in `auto`
only above the benchmarked scheduling threshold; full grouping graph
construction currently prefers SIMD in `auto` because oneTBB did not amortize
overhead in the measured regimes. Future oneTBB kernels must keep scalar import
and forced scalar execution available, record oneTBB version and thread count,
and show a win on large enough datasets before `auto` uses them.

The active pairwise oneTBB auto-dispatch threshold is reported by
`wolfgang_quantum._wolfgang_core._build_info()["cpu_auto_dispatch_thresholds"]` and
must be characterized by `benchmarks/bench_cpu_thresholds.py` before it changes.

## CPU Testing Ladder

CPU validation grows in this order:

```text
L0 docs and harness checks: source docs exist, links resolve, stale markers are absent
L1 scalar debug build: import and semantic tests pass
L2 scalar release build: import and semantic tests pass
L3 sanitizer build: AddressSanitizer and UndefinedBehaviorSanitizer pass for CPU tests where supported
L4 oneTBB build: oneTBB on/off equivalence and deterministic output tests pass
L5 SIMD dispatch: auto, scalar-forced, and feature-forced tests pass on capable hardware
L6 packaging: source distribution and CPU wheels build, install, and import
L7 CPU benchmark characterization: scalar, oneTBB, and SIMD paths are measured on named hardware
```

Phase 1 must establish L0, L1, and the release-wheel-safe build defaults. Later phases add stricter levels as implementation surface exists.

## CPU Benchmark Requirements

CPU benchmark reports must capture:

```text
CPU model
CPU architecture
core count and logical CPU count
available instruction sets as reported by runtime detection
active Wolfgang CPU backend
compiler and compiler flags relevant to CPU features
CMake CPU options
oneTBB status and version
thread count
thread affinity, CPU governor, clocks, power mode, and NUMA policy when controlled
memory size and memory speed when known
operating system and kernel or macOS version
```

Benchmark comparisons should include:

```text
scalar single-thread path
scalar path under default threading policy if different
oneTBB path where compiled and available
AVX2 path where compiled and available
AVX-512 path where compiled and available
NEON path where compiled and available
SVE path when implemented
CUDA paths when implemented
```

Benchmark reports must identify compute-bound, memory-bound, dispatch-overhead, and threading-overhead regimes when the data supports that interpretation.

## CUDA Toolkit Policy

Initial source-build CUDA support targets CUDA 12.9.x or the current CUDA 12.x line available in CI and developer environments.

Reasoning:

```text
CUDA 12.x preserves offline compilation support for Volta-era sm_70 targets
CUDA 13.x is a forward-compatibility lane for newer GPUs
moving the baseline to CUDA 13.x would drop sm_70 from the normal source-build target set
```

CUDA 13.x should be added as a compile-test lane after CUDA 12.x source builds are stable. CUDA 13.x release evidence must use a GPU architecture target set that starts at compute capability 7.5 or newer unless a separate CUDA 12.x lane is kept for Volta.

## CUDA Architecture Targets

Initial CUDA 12.x source-build architecture targets:

```text
sm_70: Volta compatibility lane
sm_75: Turing and T4 compatibility lane
sm_80: A100 and related Ampere data-center GPUs
sm_86: Ampere workstation and consumer GPUs
sm_89: Ada GPUs
sm_90: Hopper GPUs
```

Blackwell and newer targets are planned but not release-supported until toolkit support, compiler flags, hardware access, and runtime tests are confirmed:

```text
sm_100 and sm_103: Blackwell data-center class targets when supported by the selected toolkit
sm_110: Jetson Thor class targets when relevant
sm_120 and sm_121: Blackwell workstation, consumer, and GB10-class targets when supported by the selected toolkit
```

Do not hard-code a CUDA architecture in build scripts without documenting whether it is compile-tested, runtime-tested, performance-tested, or release-supported.

Campaign 8 source-build evidence is performance-tested on H100 `sm_90` only.
The required non-H100 NVIDIA retained-consumer portability run was blocked by
hardware availability and is recorded in
`docs/benchmarks/reports/cuda_portability_campaign8_non_h100_nvidia_2026-04-29.md`.
Do not broaden Campaign 8 GPU claims beyond H100 until an A100 `sm_80`, RTX
6000 Ada `sm_89`, L4/A10 `sm_89`/`sm_86`, or equivalent non-H100 NVIDIA run
replaces that blocker with validation and benchmark evidence. Campaign 7 has
the same historical H100-only boundary in
`docs/benchmarks/reports/cuda_portability_campaign7_non_h100_nvidia_2026-04-29.md`.
Campaign 10 replaced the Campaign 9 non-H100 blocker with real source-build
evidence. A100 `sm_80` and RTX PRO 6000 Blackwell `sm_120` both compiled and
ran with CUDA 12.8 source builds, CUDA validation, DLPack tests, Compute
Sanitizer, Nsight Systems, and benchmark rows captured in
`docs/benchmarks/reports/cuda_cross_architecture_campaign10_2026-04-29.md`.
This is source-build portability evidence, not a CUDA wheel release claim.
Future RTX 6000 Ada, L4, A10, or similar `sm_86`/`sm_89` lanes should use the
same Campaign 10-style schema before broadening release-support language.

## CUDA Build Options

CUDA source builds should expose:

```text
WOLFGANG_ENABLE_CUDA=ON/OFF
WOLFGANG_CUDA_ARCHITECTURES=<semicolon-separated CMake CUDA architectures>
WOLFGANG_CUDA_USE_CUB=ON
WOLFGANG_CUDA_USE_THRUST=ON
```

Default behavior:

```text
WOLFGANG_ENABLE_CUDA=OFF for all normal CPU wheels
WOLFGANG_ENABLE_CUDA=ON only when explicitly requested
WOLFGANG_CUDA_ARCHITECTURES uses the documented source-build target set unless overridden
native CUDA architecture detection is allowed only for local source builds, not release evidence
```

Phase 10 must print the detected CUDA toolkit, host compiler, requested architectures, and visible device summary in validation output.

## CUDA Testing Ladder

CUDA validation grows in this order:

```text
L0 CPU-only no-CUDA build: CUDA headers and toolkit are not required
L1 CUDA configure and compile: WOLFGANG_ENABLE_CUDA=ON builds with no runtime device required
L2 CUDA runtime smoke: device discovery, to_device, to_host, and skip messages work on any visible supported GPU
L3 CPU/GPU equivalence: implemented CUDA operations match scalar CPU on representative datasets
L4 sanitizer pass: compute-sanitizer or equivalent memory checks pass for CUDA kernels
L5 interop pass: __cuda_array_interface__ providers are accepted or rejected according to contract
L6 benchmark characterization: transfer-inclusive and device-resident timings are captured on named GPUs
L7 release matrix: CUDA source-build evidence covers the claimed toolkit, driver, OS, host compiler, and GPU architecture set
```

CUDA tests may skip when CUDA is unavailable, but required CUDA jobs must fail if CUDA was expected and unavailable.

## CUDA Benchmark Requirements

CUDA benchmark reports must capture:

```text
GPU model
compute capability
driver version
CUDA toolkit version
CUDA runtime version
CUDA architectures compiled into the extension
host compiler used by nvcc
GPU clocks, power limit, persistence mode, and MIG status when controlled
CPU model and memory configuration
PCIe or NVLink topology when transfer measurements are central to the claim
active Wolfgang CPU backend used for comparison
```

CUDA reports must include:

```text
CPU scalar timing
every available CPU optimized selector timing for covered operations
unavailable CPU optimized selector reasons
CUDA transfer-inclusive timing
CUDA device-resident timing
host-to-device and device-to-host transfer timing when measured separately
peak device memory when available
```

When GPU-library competitive baselines are included, reports must additionally
capture:

```text
library name and version
installation channel and GPU enablement status
backend, target, device, or simulator selector
whether timing is framework-level, transfer-inclusive primitive, or device-resident primitive
semantic mapping from the Wolfgang benchmark dataset to the competitor API
correctness oracle and tolerance
unavailable reason when the library, GPU backend, or exact workload mapping is absent
```

The report must identify CPU-faster, CUDA-faster, and transfer-bound regimes.

## Post-CUDA Accelerator Policy

CUDA is the first required GPU backend. Do not introduce another GPU backend before the CUDA foundation and at least one CUDA kernel are validated, unless a separate planning update explicitly changes the backend order.

Post-CUDA GPU targets are:

```text
ROCm/HIP: planned second GPU backend candidate after CUDA, with separate build flags, runtime availability checks, tests, and benchmark reports
Metal: planned Apple Silicon source-build backend candidate after CUDA and HIP source-build evidence, with separate build flags, command-buffer semantics, tests, and benchmark reports
```

ROCm/HIP work must not reuse CUDA support claims. It needs independent evidence for toolkit version, GPU architecture, driver/runtime, source build, runtime transfer tests, CPU/GPU equivalence, and transfer-inclusive and device-resident benchmarks.

The ROCm/HIP backend contract is `docs/architecture/rocm_backend.md`.
The first ROCm/HIP execution path is complete in
`docs/plans/mi300x_rocm_bringup_plan.md` with report evidence in
`docs/benchmarks/reports/rocm_mi300x_bringup_2026-04-29.md`. The first
post-bring-up MI300X execution campaign is complete in
`docs/plans/mi300x_rocm_optimization_campaign2_plan.md` with report evidence in
`docs/benchmarks/reports/rocm_mi300x_campaign2_2026-04-30.md`. HIP simplify is
complete in `docs/plans/mi300x_rocm_optimization_campaign3_plan.md` with report
evidence in `docs/benchmarks/reports/rocm_mi300x_campaign3_2026-04-30.md`. The
ROCm sequence is `docs/plans/rocm_next_waves_plan.md`, and the latest completed
ROCm architecture-readiness campaign is
`docs/plans/mi300x_rocm_optimization_campaign8_plan.md` with report evidence in
`docs/benchmarks/reports/rocm_campaign8_architecture_readiness_2026-05-01.md`.
The next executable ROCm campaign must satisfy one of the Campaign 8 trigger
gates before adding a new runtime, packaging, portability, or performance claim.
The accepted backend-neutral trigger plan is
`docs/plans/backend_neutral_accelerator_campaign9_plan.md`, with closeout
evidence in
`docs/benchmarks/reports/backend_neutral_accelerator_campaign9_2026-05-01.md`.
It uses the existing MI300X lane for HIP-target regression and the existing
NVIDIA lane for CUDA-target regression. It does not require a mixed NVIDIA+AMD
host because the normal support boundary is target-specific builds: CPU-only,
CUDA-target, HIP-target, and Apple Metal-target.
`WOLFGANG_ENABLE_CUDA=ON` with `WOLFGANG_ENABLE_HIP=ON` remains a documented
configure-time rejection unless a later accepted mixed-runtime plan reopens
that design.

A new AMD GPU architecture may be added to the support matrix only after the
same slice records:

```text
source build on that architecture
runtime status capture
retained HIP operation tests
benchmark smoke with correctness checks
profiler availability status
README support wording update
```

Without that evidence, non-MI300X AMD portability remains `blocked_external`
and must not be described as release-supported.

The accepted Apple accelerator design decision is
`docs/architecture/apple_accelerator.md`, with implementation handoff in
`docs/plans/apple_metal_mps_bringup_plan.md`. The reserved backend identity is
`metal`; MPS and MPSGraph are optional implementation adjuncts or external
baselines, not separate Wolfgang object backend identities. Metal remains
source-build-only and target-specific behind `WOLFGANG_ENABLE_METAL=ON`.
The initial source tree provides status metadata, host/device transfers, and
pairwise commutation with synchronous command-buffer completion. Metal is not
release-supported until CPU/Metal equivalence tests and benchmark evidence pass
on a named Apple Silicon host with a runtime-visible Metal device.

## Phase Responsibilities

Phase responsibilities:

```text
Phase 0: this target policy exists and is linked from source-of-truth docs
Phase 1: CPU build options, scalar baseline, validation entrypoint, and CPU CI surface follow this policy
Phase 4: first benchmark harness records CPU target metadata, including Apple Silicon when available
Phase 8: CPU expectation benchmarks record threading and dispatch metadata
Phase 9: CPU dispatch control, optimized-path availability reporting, and covered oneTBB/SIMD commutation kernels are implemented; future oneTBB/SIMD extensions continue to follow the CPU testing ladder, including Apple Silicon CPU evidence for Apple-specific claims
Phase 10: CUDA build, architecture list, and CUDA testing ladder are implemented
Phase 11: CUDA kernels meet CPU/GPU equivalence and benchmark evidence requirements
Post-CUDA: HIP is implemented for MI300X evidence; Metal has source-build bring-up code and still needs runtime-visible Apple Silicon validation before support claims
Release candidate: release-supported platform, CPU, CUDA, and wheel claims match evidence
```

## External References

Reference these upstream documents when changing target policy:

```text
NVIDIA CUDA Toolkit release notes: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
NVIDIA CUDA GPU compute capability table: https://developer.nvidia.com/cuda/gpus
NVIDIA CUDA architecture support guidance: https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/
NVIDIA cuQuantum documentation: https://docs.nvidia.com/cuda/cuquantum/
NVIDIA CUDA-Q documentation: https://nvidia.github.io/cuda-quantum/latest/
Qiskit Aer GPU simulator documentation: https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html
AMD ROCm documentation: https://rocm.docs.amd.com/
AMD HIP documentation: https://rocm.docs.amd.com/projects/HIP/
Apple Accelerate documentation: https://developer.apple.com/documentation/accelerate
Apple Metal MTLDevice documentation: https://developer.apple.com/documentation/metal/mtldevice
Apple Metal MTLBuffer documentation: https://developer.apple.com/documentation/metal/mtlbuffer
Apple Metal MTLCommandBuffer documentation: https://developer.apple.com/documentation/metal/mtlcommandbuffer
Apple metal-cpp documentation: https://developer.apple.com/metal/cpp/
Apple Metal Performance Shaders documentation: https://developer.apple.com/documentation/metalperformanceshaders
Apple Metal Performance Shaders Graph documentation: https://developer.apple.com/documentation/metalperformanceshadersgraph
PyTorch MPS backend documentation: https://docs.pytorch.org/docs/stable/notes/mps
Python packaging platform compatibility tags: https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/
PyPA manylinux project: https://github.com/pypa/manylinux
cibuildwheel platform documentation: https://cibuildwheel.pypa.io/en/latest/platforms/
```

Before changing CUDA toolkit baselines, compute capability targets, GPU-library
competitive baseline policy, manylinux tags, compiler minimums, Apple Silicon
CPU claims, Apple Metal backend policy, post-CUDA accelerator order, or wheel
platform claims, refresh these references and update this document with the
same commit.

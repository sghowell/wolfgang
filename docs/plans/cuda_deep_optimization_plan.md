# CUDA Deep Optimization Plan

This plan governs the post-Phase 11 CUDA performance push. It exists because
the first H100 hillclimb found real wins but did not prove that every relevant
CUDA path had reached its architectural limit.

## Goal

Push CUDA performance until retained gains flatten across all hot paths, while
preserving public semantics, correctness checks, benchmark honesty, and CPU-only
build safety.

## Current Bottleneck Model

The checked-in H100 Nsight report shows:

- Dense commutation is often limited by host result materialization, host buffer
  registration, device-to-host movement, and Python allocation, not only by the
  `commutation_kernel` instruction stream.
- Matrix-product generation is a smaller fraction of `matmul(...,
  simplify=True)` than duplicate reduction. Thrust/CUB sort/reduce stages are
  the important profiling target for that path.
- Statevector expectation is fast but still contains host API, temporary
  allocation, reduction, and synchronization costs that should be decomposed
  before declaring the path exhausted.
- Simplify performance is dominated by library sort/reduce pipelines and
  temporary storage behavior. Kernel instruction edits are unlikely to be the
  first lever unless profiler evidence shows source-level stalls.

## Exhaustion Criteria

CUDA optimization is not considered exhausted until a checked-in report shows:

- smoke, default, stress, and opt-in extreme scale profiles for every CUDA hot
  path;
- Nsight Systems traces that separate Python, CUDA API, allocation,
  synchronization, transfer, and kernel time;
- Nsight Compute reports for every custom kernel and every relevant
  Thrust/CUB-dominated path;
- Compute Sanitizer memcheck coverage for retained changes and at least one
  additional sanitizer pass when concurrency or uninitialized-memory behavior is
  touched;
- SASS/PTX inspection for any code-generation, inline PTX, launch-bound,
  register-pressure, or occupancy claim;
- A/B evidence for each retained and rejected experiment on the same hardware,
  dataset, command shape, and git revision family;
- all available FastPauli paths measured separately: CPU scalar, optimized CPU
  selectors, CUDA transfer-inclusive, CUDA device-resident, and CUDA
  preallocated/reused-output paths where available;
- open-source competitor baselines installed and benchmarked when the workload
  semantics are comparable, with explicit unavailable reasons otherwise.

## Profiling Ladder

Run the ladder with:

```bash
python scripts/cuda_deep_profile.py --dry-run --json --profile stress
```

Then execute the planned commands on the GPU host after confirming the output
root and package-install choices:

```bash
python scripts/cuda_deep_profile.py \
  --execute --json --profile stress --competitor-set all \
  --require-profiler-artifacts
```

The ladder must collect:

- repo validation with `FASTPAULI_VALIDATE_CUDA=1`;
- correctness-checked CUDA scaling benchmarks;
- Nsight Systems CUDA API timelines;
- Nsight Compute detailed reports for each operation-specific kernel set;
- Compute Sanitizer memcheck, plus racecheck/initcheck/synccheck when relevant;
- `cuobjdump` SASS/PTX output and `nvdisasm` output where supported by the
  binary format;
- optional competitor setup and benchmark commands.

## Experiment Queue

Run experiments in this order unless new profiling evidence reprioritizes them.

1. **Output-materialization decomposition**
   - Add and measure reusable caller-owned output buffers for dense CUDA
     commutation.
   - Compare normal `commutes_with()` with `commutes_with_into()` over default,
     stress, and extreme scales.
   - If Python allocation is material, keep the reusable-output API and document
     it as the high-throughput path.

2. **Reusable CUDA temporary storage**
   - Prototype explicit workspaces for simplify and matmul+simplify.
   - Measure allocation count, temporary bytes, and Thrust/CUB call time before
     and after.
   - Keep only if the API and lifetime rules are documented and CPU-only builds
     remain unaffected.

3. **CUB duplicate-reduction pipeline**
   - Replace ad hoc Thrust allocation patterns with explicit CUB/CCCL
     primitives where they reduce launches, temporary storage, or copies.
   - Maintain canonical ordering and tolerance semantics.
   - Compare against Thrust on the same keys, duplicate rates, and survivor
     counts.

4. **Statevector expectation reductions**
   - Decompose host-copy, device-pointer, term-value, and final-reduction costs.
   - Evaluate fused block reductions, CUB reductions, and optional deterministic
     modes.
   - Reject any path whose floating-point behavior exceeds documented
     tolerances or makes repeated-run evidence unstable.

5. **Commutation kernel specialization**
   - Revisit words==1 and words==2 only after preallocated-output and transfer
     costs are separated.
   - Use Nsight Compute occupancy, memory throughput, instruction mix, warp
     stall, and source-correlation data to decide whether tiling, vectorized
     loads, warp-level work partitioning, or bit-packed output is justified.

6. **Matmul product-generation specialization**
   - Profile product generation separately from simplify.
   - Evaluate words==1/2 specializations, launch bounds, and unrolling only if
     product generation is a meaningful portion of end-to-end time.

7. **Async and stream-aware APIs**
   - Do not remove documented synchronizations from current public methods.
   - Add explicit async/stream APIs only after ownership, lifetime, benchmark
     timing boundaries, and error-surfacing semantics are documented.

## DSL, Library, And PTX Decision Rules

Use the highest-level implementation that can hit the measured bottleneck:

- Prefer CUDA C++ plus CCCL/CUB/Thrust for sort, reduce, scan, compaction, and
  stable production integration.
- Consider CUTLASS/CuTe only for workloads that naturally map to tiled tensor,
  GEMM-like, or reusable layout abstractions. Current sparse-Pauli bit kernels
  do not justify a CUTLASS dependency by default.
- Consider Triton or CuTe DSL for scratch prototyping only when it shortens the
  experiment loop and can be matched against a production CUDA C++ path.
- Do not ship Numba kernels in the core package. They are useful for notebooks
  or exploratory baselines, not for the C++/CUDA extension ABI.
- Use inline PTX or raw PTX only after SASS/PTX inspection shows a specific
  compiler code-generation issue that CUDA C++ cannot express cleanly. Each PTX
  use must be architecture-gated, covered by CUDA tests, benchmarked against the
  CUDA C++ version, and documented with the exact instruction-level reason.

## Report Requirements

The final report from this optimization push must follow
`docs/benchmarks/protocol.md` and include:

- retained and rejected experiment tables;
- profiler screenshots or exported charts when useful;
- plots generated from checked-in benchmark JSON;
- a README-facing cross-comparison plot that includes CPU scalar, captured
  optimized CPU selectors, CUDA transfer-inclusive, CUDA device-resident, and
  semantically comparable external baselines where data exists;
- architecture and kernel diagrams;
- exact package-install and benchmark commands for competitors;
- a clear statement of what still has headroom and why.

## Completion Record

The 2026-04-28 H100 deep-optimization checkpoint is captured in
`docs/benchmarks/reports/cuda_deep_optimization_h100_2026-04-28.md`.
That report is the current source of truth for retained and rejected
experiments from this plan:

- retained: host-statevector byte-copy for CUDA statevector expectation;
- rejected: one/two-word 2D commutation grid specialization;
- captured: smoke/default/stress/extreme CUDA scaling, privileged Nsight
  Compute reports for custom and CCCL/Thrust-heavy paths, Nsight Systems
  timelines, Compute Sanitizer passes, cuobjdump PTX/SASS inventory, and
  comparable open-source baselines where semantics match.

The H100 campaign-2 follow-up is captured in
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign2_2026-04-28.md`.
It retained fused statevector expectation accumulation, captured fresh
same-boundary H100 evidence, and concluded that remaining headroom is still
allocation and materialization dominated:

```text
simplify/matmul+simplify temporary storage and sort/reduce allocation behavior
public workspace semantics before reusable temporary storage can ship
commutation host-output materialization
device-output and async lifetime/synchronization contracts
statevector reduction topology only if tolerance and repeatability evidence justify it
```

Future CUDA work should continue from the latest checked-in report's
remaining-headroom section. Treat reusable workspace, device-output,
device-statevector, and stream-aware work as design slices with explicit API and
benchmark boundaries; do not reopen small commutation instruction edits without
new profiling evidence or a changed materialization boundary.

The H100 campaign-3 follow-up is captured in
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign3_2026-04-28.md`.
It retained the packed-key simplify path for one-word operators with at most 32
qubits, quantified commutation output materialization with a private
benchmark-only reused device-output path, and replaced the README benchmark
snapshot with a broad CPU/CUDA/external cross-comparison from checked evidence.
Its remaining headroom is now a design problem before it is an implementation
problem: a real private CUDA workspace abstraction should precede any explicit
CUB scratch-buffer rewrite, and a public device-output commutation path needs a
separate API review.

The H100 campaign-4 follow-up is captured in
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign4_2026-04-29.md`.
It implemented the private CUDA workspace boundary, measured CUB/CCCL
scratch-buffer duplicate-reduction variants, refreshed commutation
materialization and external baselines, and retained no new public CUDA API.
The narrow CUB radix-sort duplicate-reduction prototype was rejected for
production on same-boundary H100 evidence; commutation device-output remains
deferred to API review.

The H100 campaign-5 follow-up is captured in
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign5_2026-04-29.md`.
It retained the experimental dense `DeviceCommutationMatrix` device-resident
commutation result API, measured host vector, caller-owned host fill, private
device-output reuse, public device-output allocation, public device-output
reuse, and explicit `to_host()` materialization as separate boundaries, and
refreshed the README broad CPU/CUDA/external performance landscape. Public
stream, async, and bit-packed CUDA APIs remain deferred to a separate accepted
API plan.

The H100 campaign-6 follow-up is captured in
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign6_2026-04-29.md`.
It retained compact `DeviceCommutationMatrix.count_commuting(axis=None|0|1)`
consumers, added CuPy CUDA-array-interface consumer baselines, documented the
stream/async and bit-packed deferrals, and refreshed the README broad
CPU/CUDA/external performance landscape from checked H100 evidence.

The H100 campaign-7 follow-up is captured in
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign7_2026-04-29.md`.
It retained private benchmark-only fused graph and grouping-oriented consumers
on `DeviceCommutationMatrix`, rejected count-reduction specialization for this
slice as not dominant, kept public async/stream and bit-packed APIs deferred,
recorded a non-H100 NVIDIA portability blocker, and refreshed the broad
CPU/CUDA/external README landscape from checked H100 evidence.

## Follow-Up CUDA Campaigns

Completed and planned CUDA campaign records:

```text
Campaign 2 plan: docs/plans/h100_deep_optimization_campaign2_plan.md
Campaign 2 report: docs/benchmarks/reports/cuda_deep_optimization_h100_campaign2_2026-04-28.md
Campaign 3 plan: docs/plans/h100_deep_optimization_campaign3_plan.md
Campaign 3 report: docs/benchmarks/reports/cuda_deep_optimization_h100_campaign3_2026-04-28.md
Campaign 4 plan: docs/plans/h100_deep_optimization_campaign4_plan.md
Campaign 4 report: docs/benchmarks/reports/cuda_deep_optimization_h100_campaign4_2026-04-29.md
Campaign 5 plan: docs/plans/h100_deep_optimization_campaign5_plan.md
Campaign 5 report: docs/benchmarks/reports/cuda_deep_optimization_h100_campaign5_2026-04-29.md
Campaign 6 plan: docs/plans/h100_deep_optimization_campaign6_plan.md
Campaign 6 report: docs/benchmarks/reports/cuda_deep_optimization_h100_campaign6_2026-04-29.md
Campaign 7 plan: docs/plans/h100_deep_optimization_campaign7_plan.md
Campaign 7 report: docs/benchmarks/reports/cuda_deep_optimization_h100_campaign7_2026-04-29.md
Campaign 8 plan: docs/plans/h100_deep_optimization_campaign8_plan.md
Campaign 8 report: docs/benchmarks/reports/cuda_deep_optimization_h100_campaign8_2026-04-29.md
Campaign 9 plan: docs/plans/h100_deep_optimization_campaign9_plan.md
Campaign 9 report: docs/benchmarks/reports/cuda_deferred_headroom_campaign9_2026-04-29.md
Campaign 10 plan: docs/plans/cuda_cross_architecture_campaign10_plan.md
Campaign 10 report: docs/benchmarks/reports/cuda_cross_architecture_campaign10_2026-04-29.md
Campaign 11 plan: docs/plans/cuda_residual_risk_campaign11_plan.md
```

Campaign 3 was H100-first: performance, profiling, sanitizer, and competitor
evidence was collected on the H100 host, while local Apple Silicon work was
limited to repo editing, documentation checks, and non-performance validation.
It started from the campaign-2 remaining-headroom section and prioritized:

```text
allocation and temporary-storage instrumentation
private reusable CUDA/CCCL/CUB workspace prototypes for simplify and matmul+simplify
CUB-backed or lower-allocation duplicate-reduction experiments
commutation output materialization alternatives behind private or benchmark-only surfaces
statevector expectation reduction topology only with accuracy and repeatability evidence
workload-specific external GPU baselines with explicit semantic and timing boundaries
```

Do not treat this follow-up as a raw PTX campaign. PTX or inline assembly is
allowed only after Nsight and SASS evidence identifies a specific compiler
code-generation problem that CUDA C++ plus CCCL/CUB cannot express.

Campaign 4 narrowed that follow-up into an execution path with these required
gates, all now evidenced by the checked Campaign 4 report:

```text
private workspace ownership, device binding, reset/release, and timing-boundary semantics first
CUB/CCCL scratch-buffer duplicate-reduction experiments second
commutation device-output and bit-packed output prototypes behind benchmark-only labels third
statevector reduction topology only when fresh profiler evidence identifies a reduction bottleneck
public workspace, stream, async, device-output, or bit-packed APIs only after separate API review
```

Campaign 5 narrowed the next follow-up to the supported device-output boundary,
with these gates now evidenced by the checked Campaign 5 report:

```text
review and accept or reject the dense DeviceCommutationMatrix API before code
keep current public CUDA methods synchronous and default-stream compatible
measure host vector, caller-owned host fill, device-output allocation, and device-output reuse as separate boundaries
use CUDA-array-interface exposure for dense uint8 output only after same-device and ownership tests land
defer public stream, async, and bit-packed APIs until device-output evidence shows a specific need
refresh README performance visuals only with broad CPU/CUDA/external checked evidence
```

Campaign 6 completed all five Campaign 5 remaining-headroom items:
async/stream API design with event and lifetime semantics, GPU-resident compact
consumers for `DeviceCommutationMatrix`, conditional bit-packed output
evaluation by documented deferral, CuPy CUDA-array-interface consumer
benchmarks, and broad README performance landscape upkeep. Campaign 7 completed
all five Campaign 6 remaining-headroom items by prioritizing fused downstream
algorithms over another isolated dense-output boundary campaign. Campaign 8
completed every Campaign 7 remaining-headroom item on H100: retained compact
device-resident graph and grouping consumers that avoid exporting full CSR edge
lists, deferred public fused grouping until exact return and ownership semantics
are accepted, deferred DLPack while retaining CUDA Array Interface interop,
recorded the non-H100 NVIDIA portability blocker, deferred CUDA Graphs and
stream-aware execution until a complete contract exists, and rejected CSR
scatter tuning because the retained consumer no longer needs full CSR scatter.

Campaign 9 completed every Campaign 8 remaining-headroom item with a final
non-deferred outcome: one named non-H100 NVIDIA portability access check
recorded `blocked_external`, privileged Nsight Compute counters passed, the
true public fused grouping API was rejected with evidence while
`DeviceCommutationMatrix.conflict_degrees(axis=None|0|1)` was implemented,
read-only `DeviceCommutationMatrix` DLPack export was implemented and validated
with CuPy, stream/CUDA Graph surfaces were rejected with evidence, and CSR
scatter reopening was rejected because retained compact consumers do not need
full CSR edge-list materialization.

Campaign 10 completed every Campaign 9 remaining-headroom item with a final
non-deferred outcome. A100 `sm_80` and RTX PRO 6000 Blackwell `sm_120`
source-build validation passed, PyTorch CUDA DLPack consumption passed on both
hosts, public grouping remained rejected while
`DeviceCommutationMatrix.conflict_degrees(axis=None|0|1)` remains the accepted
compact public summary API, stream/CUDA Graph work remained rejected because
launch overhead was not dominant in fresh Nsight Systems evidence, and CSR
scatter remained rejected because no retained consumer requires full CSR
edge-list materialization.

The next CUDA follow-up should not reopen Campaign 9 headroom by default. It
should start from Campaign 10 evidence and require a new concrete trigger:
release packaging, additional `sm_86` or `sm_89` portability lanes, non-H100
Nsight Compute counters when `ncu` is available, or a retained consumer with an
accepted API and memory-ownership contract.

Campaign 11 narrows the immediate follow-up to the two Campaign 10 residual
risks that are in scope now: install or enable Nsight Compute on the existing
A100 and RTX PRO 6000 Blackwell hosts, capture non-H100 counter evidence when
permissions allow, and investigate the nanobind reference-leak diagnostics from
Campaign 10 Compute Sanitizer logs. Campaign 11 explicitly excludes A10, L4,
RTX 6000 Ada, and other additional NVIDIA hosts.

Campaign 11 is complete and captured in
`docs/benchmarks/reports/cuda_residual_risk_campaign11_2026-04-29.md`.
RTX PRO 6000 Blackwell Nsight Compute counter capture passed and supports the
Campaign 9/10 compact-consumer bottleneck model. A100 counter capture reached a
terminal `blocked_permissions` state after non-root `ERR_NVGPUCTRPERM` and sudo
lock-failure evidence. The nanobind reference-leak messages are classified as
Compute Sanitizer/nanobind teardown diagnostics after fresh memcheck logs and
clean targeted lifecycle subprocesses on both hosts.

## References

- NVIDIA Nsight Compute CLI documentation:
  https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
- NVIDIA Nsight Compute roofline analysis:
  https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/
- NVIDIA CCCL/CUB device-wide primitives:
  https://nvidia.github.io/cccl/unstable/cub/api/device.html
- NVIDIA PTX ISA documentation:
  https://docs.nvidia.com/cuda/parallel-thread-execution/
- NVIDIA inline PTX assembly guide:
  https://docs.nvidia.com/cuda/inline-ptx-assembly/
- NVIDIA CUTLASS/CuTe documentation:
  https://docs.nvidia.com/cutlass/

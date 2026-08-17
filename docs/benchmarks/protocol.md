# Wolfgang Benchmark Protocol

This document defines how Wolfgang benchmarks are designed, run, and interpreted.

## Principles

```text
benchmarks are reproducible
benchmarks compare against meaningful baselines
speedup claims require captured command output
CUDA measurements report transfer-inclusive and device-resident timings separately
benchmark datasets are generated from documented parameters and deterministic seeds
```

Benchmarks do not replace correctness tests. A benchmark result is valid only when the corresponding correctness validation passes on the same code revision.

The current source-of-truth ranking handoff for shared follow-up CPU, CUDA,
ROCm/HIP, and Apple Metal optimization work is the Cross-backend kernel performance campaign in `docs/plans/wolfgang-kernel-performance-campaign.md`.

## Benchmark Entrypoints

Python benchmarks live under:

```text
benchmarks/
```

C++ benchmarks live under:

```text
cpp_benchmarks/
```

Current Python benchmark smoke commands:

```bash
python benchmarks/bench_cpu_dispatch.py --smoke --repeat 1
python benchmarks/bench_cpu_thresholds.py --smoke --repeat 1
python benchmarks/bench_simplify.py --smoke --repeat 1
python benchmarks/bench_multiply.py --smoke --repeat 1
python benchmarks/bench_grouping.py --smoke --repeat 1
python benchmarks/bench_expectation.py --smoke --repeat 1
python benchmarks/bench_cuda_kernels.py --smoke --repeat 1
python benchmarks/bench_cuda_scaling.py --profile smoke --repeat 1 --json
python benchmarks/bench_metal_kernels.py --smoke --repeat 1 --json
python benchmarks/bench_openfermion_conversion.py --smoke --repeat 1
python benchmarks/bench_competitive_baselines.py --smoke --repeat 1
python scripts/cuda_deep_profile.py --dry-run --json --profile smoke
```

Report rendering commands fall into two categories. Commands that consume only
checked-in summaries can run from a public clone. Commands below that pass a
`--raw-dir` reproduce historical reports only when the separately retained
private raw benchmark archive is available; those inputs are intentionally not
part of the public repository or source distribution.

```bash
python scripts/render_benchmark_plots.py \
  --cuda-report docs/benchmarks/reports/cuda_h100_nsight_hillclimb_2026-04-28.md \
  --output docs/benchmarks/plots/cuda_h100_nsight_hillclimb_default_backend_speedups.svg
python scripts/render_cuda_deep_report_assets.py \
  --raw-dir docs/benchmarks/data/cuda_deep_optimization_h100_2026-04-28/raw \
  --summary-output docs/benchmarks/data/cuda_deep_optimization_h100_2026-04-28/summary.json \
  --plot-dir docs/benchmarks/plots
python scripts/render_cuda_campaign2_assets.py \
  --summary docs/benchmarks/data/cuda_deep_optimization_h100_campaign2_2026-04-28/summary.json \
  --raw-dir docs/benchmarks/data/cuda_deep_optimization_h100_campaign2_2026-04-28/raw \
  --plot-dir docs/benchmarks/plots
python scripts/render_cuda_campaign3_assets.py \
  --raw-dir docs/benchmarks/data/cuda_deep_optimization_h100_campaign3_2026-04-28/raw \
  --summary-output docs/benchmarks/data/cuda_deep_optimization_h100_campaign3_2026-04-28/summary.json \
  --plot-dir docs/benchmarks/plots
python scripts/render_apple_metal_assets.py \
  --data-dir docs/benchmarks/data/apple_metal_optimization_campaign8_2026-05-07 \
  --plot-dir docs/benchmarks/plots
```

Full simplify runs can increase `--num-qubits`, `--num-terms`, `--term-weight`,
`--warmup`, and `--repeat`. The simplify command emits low-duplicate and
high-duplicate simplify cases and compares Wolfgang scalar CPU against a pure-Python
packed-key baseline.

Full multiply runs can increase `--num-qubits`, `--lhs-terms`, `--rhs-terms`,
`--term-weight`, `--max-intermediate-terms`, `--warmup`, and `--repeat`. The multiply
command emits single-term and small cross-product cases and compares Wolfgang scalar
CPU against a pure-Python dense-label reference.

Full grouping runs can increase `--num-qubits`, `--lhs-terms`, `--rhs-terms`,
`--group-terms`, `--term-weight`, `--max-commutation-matrix-entries`,
`--max-terms-for-graph`, `--warmup`, and `--repeat`. The grouping command emits
pairwise commutation, QWC grouping, full grouping, and guardrail-rejection cases and
compares successful cases against pure-Python dense-label references.

Full CPU dispatch runs can increase `--num-qubits`, `--num-terms`,
`--commutation-qubits`, `--lhs-terms`, `--rhs-terms`, `--group-terms`,
`--warmup`, and `--repeat`. The command compares `auto`, forced scalar, and each
compiled optimized selector for pairwise commutation and full grouping graph
construction. Dispatch output records whether each optimized result matches
forced scalar and includes an `effective_backend_hint` for `auto` cases.

CUDA scaling runs use `--profile smoke`, `default`, `stress`, or `extreme`.
The `extreme` profile is opt-in hardware evidence for scale-limit profiling and
is not part of default validation because it runs large correctness-checked CPU
references and dense CUDA outputs.

Deep CUDA profiling runs are orchestrated by `scripts/cuda_deep_profile.py`.
Run the script in `--dry-run` mode before long GPU execution so benchmark
reports can capture the exact Nsight, sanitizer, binary-inspection, benchmark,
and competitor-install commands that will execute.

Full OpenFermion conversion runs can increase `--num-qubits`,
`--round-trip-terms`, `--large-terms`, `--term-weight`, `--warmup`, and
`--repeat`. The command emits round-trip and larger sparse conversion cases and
compares Wolfgang conversion against OpenFermion `QubitOperator` rebuild paths.

Recommended Python harness:

```bash
python -m pytest benchmarks --benchmark-only
```

Recommended C++ harness after a checked-in benchmark target exists:

```text
No canonical C++ benchmark target is currently declared in CMake.
Use the Python harness above until a tracked benchmark binary lands.
```

## Dataset Generation

Every benchmark dataset must record:

```text
num_qubits
num_terms
term_weight distribution
duplicate_rate
coefficient dtype
random seed
operator construction method
```

Default benchmark dimensions:

```text
num_qubits: 32, 64, 128, 512, 2048
num_terms: 10_000, 100_000, 1_000_000
term_weight: 2, 4, 8, 16, 64
duplicate_rate: 0%, 10%, 50%, 90%
coefficient dtype: complex128
```

Generic tiny benchmark-smoke datasets should use:

```text
num_qubits: 8
num_terms: 32
term_weight: 2
duplicate_rate: 25%
```

Phase-specific benchmark smoke commands may override this when the phase gate
requires multiple regimes. The Phase 4 simplify smoke emits two duplicate
regimes on the same tiny dimensions:

```text
low_duplicate target duplicate_rate: 5%
high_duplicate target duplicate_rate: 90%
```

The Phase 5 multiply smoke emits:

```text
single_term: 8 qubits, 1 lhs term, 1 rhs term, explicit X @ Y phase fixture
small_cross_product: 8 qubits, 8 lhs terms, 6 rhs terms, term_weight 2
simplified_duplicate_cross_product: repeated labels with default simplify_output=true
```

The Phase 6 grouping smoke emits:

```text
pairwise_commutation: 8 qubits, 8 lhs terms, 6 rhs terms, term_weight 2
qwc_grouping: 8 qubits, 10 terms, term_weight 2
full_grouping: 8 qubits, 10 terms, term_weight 2
guardrail_rejection: 1 qubit, 3 lhs terms, 4 rhs terms, max entries 11
```

The Phase 7 OpenFermion conversion smoke emits:

```text
round_trip_conversion: 8 qubits, 16 requested terms, term_weight 2
large_sparse_conversion: 8 qubits, 64 requested terms, term_weight 2
```

The competitive baseline smoke emits:

```text
simplify: Wolfgang scalar compared with Qiskit SparsePauliOp.simplify when Qiskit is installed
multiply: Wolfgang scalar compared with OpenFermion QubitOperator multiplication when OpenFermion is installed
qiskit_grouping: Wolfgang full grouping compared with Qiskit SparsePauliOp.group_commuting when Qiskit is installed
```

The CPU dispatch smoke emits:

```text
auto_statevector_expectation and forced_scalar_statevector_expectation
forced_scalar_pairwise_commutation and auto_pairwise_commutation
forced_<backend>_pairwise_commutation for available optimized backends
forced_scalar_full_grouping and auto_full_grouping
forced_<backend>_full_grouping for available optimized backends
optimized_backend_availability
```

Benchmark data generators must be deterministic for a fixed seed.

## Environment Capture

Benchmark reports must capture:

```text
git commit
dirty working tree provenance when benchmark evidence is generated before the closeout commit exists
operating system
Python version
compiler and version
CMake version
CPU model
CPU architecture
CPU vendor or SoC family when relevant
core count and logical CPU count
available CPU instruction sets when reported by the runtime, or the feature-probe failure reason when unavailable
active Wolfgang CPU backend
CPU feature compile flags, CMake CPU options, and compiled backend feature flags when relevant
oneTBB enabled or disabled
oneTBB version when available
CUDA enabled or disabled
GPU model, compute capability, driver, CUDA toolkit, CUDA runtime, and compiled CUDA architectures when CUDA is used
non-CUDA accelerator backend, toolkit, runtime, device model, and architecture when HIP, Metal, or another backend is used
thread count or thread-affinity settings when controlled
power mode, clocks, NUMA policy, and memory configuration when controlled or central to the result
```

Apple Silicon CPU reports should include the named M-series SoC, performance and efficiency core counts when known, macOS version, compiler version, active CPU backend, and power mode. Do not merge Apple Silicon timings into x86_64 summaries without labeling the architecture.

## Baselines

Benchmarks compare against the baselines available for that phase:

```text
pure Python or NumPy reference where relevant
Qiskit SparsePauliOp when Qiskit is installed
OpenFermion QubitOperator when OpenFermion is installed
Wolfgang scalar CPU
Wolfgang oneTBB CPU
Wolfgang SIMD CPU where compiled and available
Wolfgang CUDA transfer-inclusive when implemented
Wolfgang CUDA device-resident when implemented
NVIDIA cuQuantum cuStateVec for semantically matched statevector Pauli-expectation cases
NVIDIA cuQuantum cuPauliProp for matching Pauli-expansion propagation, trace, sort, or deduplication cases
NVIDIA CUDA-Q for end-to-end spin-operator observe workflows
Qiskit Aer GPU or Aer cuStateVec for matching framework-level circuit/statevector workflows
cuDensityMat or other GPU quantum libraries only after a workload-specific semantic mapping is documented
Wolfgang HIP or Metal paths only after those backends are implemented and validated
```

Do not compare an optimized path only against itself.

Optional-library baselines must record both availability and package version.
If a library is not installed, the benchmark must emit an explicit unavailable
status instead of omitting the case.
Competitive baseline cases must also report whether competitor correctness was
actually checked for that case. If a competitor canonicalizes inputs before the
timed operation, the report must include the effective competitor term counts or
otherwise document the workload semantics.

GPU-library competitive baselines are optional dependencies, but they are part
of the post-Phase 11 benchmark roadmap where relevant. A GPU-library baseline
is relevant only when the operation being timed has the same mathematical
meaning, input normalization, coefficient dtype, precision, and host/device data
movement boundary as the Wolfgang case being compared. In particular:

```text
cuStateVec is a direct candidate for statevector Pauli expectation over the same resident or transfer-inclusive statevector
cuPauliProp is a candidate for Pauli-expansion sort, deduplication, trace, and propagation-style workloads
CUDA-Q is an end-to-end spin-operator observe baseline, not a device-resident sparse-Pauli primitive baseline
Qiskit Aer GPU is a framework-level simulator baseline for circuit or statevector workflows
```

cuPauliProp and Qiskit Aer GPU are not blanket replacements for Wolfgang
simplify, multiplication, or device-resident sparse-Pauli primitive benchmarks.

Reports that include GPU-library baselines must record:

```text
library name and version
installation channel and GPU enablement status
backend or target selector used by the library
semantic mapping from the Wolfgang dataset to the competitor API
whether timing is framework-level, transfer-inclusive primitive, or device-resident primitive
correctness oracle and tolerance
availability or unavailable reason
```

## Comprehensive Overnight Optimization Report

At the end of an extended CPU/CUDA optimization and hillclimb run, the agent
must create a comprehensive checked-in performance optimization and profiling
report under `docs/benchmarks/reports/`. This report is a source-of-truth
engineering artifact, not a chat summary. It must explain what changed, why it
changed, how it was validated, what was rejected, and where the next bottlenecks
remain.

The report must include:

```text
executive summary and scope boundaries
complete command log for retained benchmark, profiler, sanitizer, and validation runs
git revisions compared, including baseline and final revisions
hardware and software environment capture for every host used
correctness oracle, tolerance, and failure-mode summary for every benchmark family
performance tables for all retained Wolfgang execution paths
before/after performance deltas for every retained optimization
rejected experiment table with exact rationale and measured result where available
profiling interpretation that separates CPU, Python binding, CUDA API, transfer, kernel, allocator, and result-materialization costs
architecture and kernel explanations deep enough for a new kernel engineer to understand the implementation
residual risks, portability limits, and recommended next optimization experiments
```

The report must benchmark every relevant Wolfgang path available on the
hardware used:

```text
scalar CPU
auto CPU dispatch
forced oneTBB CPU when compiled
forced SIMD CPU selectors when compiled, including AVX2, AVX-512, NEON, or future SVE paths
CUDA transfer-inclusive paths
CUDA device-resident paths
CUDA preallocated or reused-output paths where the public API exposes them
future HIP, Metal, or other accelerator paths when implemented
```

The report must also install and benchmark comparable open-source packages
where the platform, Python version, CUDA/toolkit version, and license constraints
permit. Candidate packages include Qiskit, Qiskit Aer GPU or Aer cuStateVec,
OpenFermion, NVIDIA cuQuantum Python components such as cuStateVec and
cuPauliProp, CUDA-Q, CuPy-backed reference implementations, and any other
package with a documented semantic mapping to a Wolfgang workload. Installation
must happen in an isolated environment or clearly documented environment layer.
For each external package, record:

```text
package name and import name
version
installation command or channel
GPU enablement status and backend selector
semantic mapping to the Wolfgang dataset
timing boundary, such as framework-level, transfer-inclusive primitive, or device-resident primitive
whether the competitor canonicalizes, simplifies, densifies, or changes the effective term count before timing
correctness oracle and tolerance
measured timing, or explicit unavailable reason
```

Do not make an external-library speedup claim unless the report demonstrates
that the comparison is mathematically equivalent, uses compatible input
normalization and dtype, and has a clearly stated timing boundary. If a package
cannot be installed on the current machine, the report must include the exact
attempted install command and error or incompatibility reason.

### Visual Standards

The comprehensive report must include publication-quality visuals. Data-driven
plots must be generated from checked-in benchmark artifacts or directly emitted
benchmark JSON, not hallucinated or hand-drawn. Conceptual visuals, diagrams,
and illustrative architecture figures may use `gpt-image-2` when it can produce
a clearer result than code-generated SVG or Mermaid. Every generated visual must
store the prompt, source data or source description, generation date, and any
manual post-processing notes in the report or an adjacent appendix.

Required visual families:

```text
performance improvement charts for before/after retained optimizations
Wolfgang path comparison charts covering CPU scalar, CPU optimized, CUDA transfer-inclusive, CUDA resident, and comparable external packages
scaling plots across term count, qubit count, dense commutation matrix size, and statevector size where data exists
roofline-style or bottleneck-classification charts when profiler data supports them
overall software architecture diagram from Python API through C++ representation, CPU dispatch, CUDA device mirror, kernels, and result materialization
hardware architecture diagram for relevant CPU and GPU details, including CPU SIMD/threads, PCIe transfer boundary, H100 SMs, memory hierarchy, and host/device memory movement
kernel diagrams for simplify, expectation, commutation, and matmul+simplify showing data layout, parallel work partitioning, memory access patterns, synchronization, and reduction or sort stages
algorithm flow diagrams for canonicalization, duplicate reduction, commutation parity, Pauli multiplication phase handling, and statevector expectation
profiling timeline or stack/bottleneck diagrams that explain where wall time is spent
```

Visual quality requirements:

```text
clean technical aesthetic
publication-quality layout
legible at README and report-rendered sizes
consistent typography, color palette, labels, units, and captions
no decorative clutter or misleading visual embellishment
axis labels, legends, units, and uncertainty or repeat-count context where applicable
color choices that remain readable in grayscale and by colorblind readers where feasible
alt text or adjacent caption explaining what the visual proves
```

When using `gpt-image-2`, prompts must be detailed enough to produce optimal
technical visuals. Prompts should specify diagram type, layout, exact labels,
style, color palette, aspect ratio, typography preference, visual hierarchy,
what to exclude, and that the image must avoid clutter. Generated conceptual
visuals must not be used as evidence for numeric performance claims; numeric
performance charts must come from measured benchmark data.

## Timing Policy

Benchmark reports should include:

```text
warmup count
repeat count
median time
interquartile range or standard deviation
minimum and maximum time when available
peak memory or allocation count when available
```

For CUDA:

```text
transfer-inclusive timing includes host-to-device and device-to-host transfer
device-resident timing excludes initial host-to-device transfer and final host conversion
kernel timing must synchronize before stopping the timer
```

For Metal:

```text
transfer-inclusive timing includes host-to-shared-buffer construction and host materialization when the public operation returns host output
device-resident timing excludes initial host-to-shared-buffer construction
kernel timing must synchronize the command buffer before stopping the timer
commutation rows must record selected kernel name when specialized kernels are available
commutation rows must record dispatch API, grid shape, threadgroup shape, storage mode, command-buffer synchronization boundary, and allocation or reuse boundary
Campaign 2 rows use dispatchThreads_2d with grid_shape [rhs_terms, lhs_terms, 1] and threadgroup_size [16, 16, 1]
Campaign 2 A/B rows may use the internal WOLFGANG_EXPERIMENTAL_METAL_COMMUTATION_KERNEL selector to force generic_2d or flat_generic; reports must label those rows as benchmark-only baselines
Campaign 2 A/B correctness checks must poison reused output with the inverse expected matrix before each selector check
Campaign 3 rows may use benchmark-only WOLFGANG_EXPERIMENTAL_METAL_LIBRARY_PATH, WOLFGANG_EXPERIMENTAL_METAL_OUTPUT_STORAGE, and WOLFGANG_EXPERIMENTAL_METAL_COMPACT_CONSUMER selectors; reports must label offline `.metallib`, private-blit, and GPU compact-reduction rows as experimental evidence, not public API
Campaign 3 private-blit rows must record the device-private output, shared staging, blit synchronization, and device_resident_private_output_blit_to_shared_staging boundary separately from shared host-output rows
Campaign 3 GPU compact-consumer rows must record compact_consumer_gpu_reduction and compare against CPU scans over shared Metal storage before any default policy changes
Campaign 4 parallel compact-consumer rows must record compact_consumer_gpu_parallel_block_reduction, partial-output count, input-entry count, and threadgroup width before any default policy changes
Campaign 5 simplify rows must record operation: simplify; variant:
cpu_default, cpu_scalar, cpu_neon, metal_simplify_transfer_reference, or
metal_simplify_device_candidate; transfer_boundary: host_materialized,
device_to_host_cpu_simplify_host_to_device, or device_resident;
metal_simplify_strategy: cpu_reference, transfer_reference, or
device_candidate; metal_simplify_strategy_status: retained, benchmark_only,
rejected_with_evidence, or unavailable; num_terms; output_terms;
duplicate_rate; atol; rtol; correct; and median/min/max seconds when timed
```

`metal_simplify_transfer_reference` uses the
`device_to_host_cpu_simplify_host_to_device` boundary. It is a correctness
bridge that materializes the Metal object on the host, runs CPU
`PauliSum.simplify()`, and returns a Metal `DevicePauliSum`; it may not be
reported as a device-resident GPU duplicate-reduction speedup.

Campaign 6 extends the simplify evidence with private MetalWorkspace
groundwork.
Rows must keep the Campaign 5 CPU and transfer-reference fields and may add:
variant: `metal_simplify_workspace_probe`; transfer_boundary: `status_only`;
metal_simplify_strategy: `device_candidate`;
metal_simplify_strategy_status: `rejected_with_evidence`;
metal_simplify_workspace_model.status: `retained_private_model`; and
workspace_timing_mode: `absent`, `grow_inside_timing`, or
`pre_reserved_outside_timing`. The workspace probe records the private
MetalWorkspace scratch model for future duplicate reduction. It is not a timed
Metal kernel, may not be plotted as a speedup, and may not use the
`device_resident` boundary until checked Metal sort/prefix/reduce primitives
exist and the row reports correct canonical simplify output.

Apple Metal Campaign 7 extends the simplify evidence with a checked device-resident simplify primitive stack
for one-word Metal inputs. Rows may use variant
`metal_simplify_device_candidate` with transfer_boundary `device_resident` only
when checked Metal sort, prefix-sum, and reduce-by-key primitives are used by a
private Metal hook returning a Metal `DevicePauliSum`, the materialized
candidate output exactly matches CPU `PauliSum.simplify()` canonical ordering
within the documented tolerance, and the row records checked Metal sort,
prefix-sum, and reduce-by-key primitives. Campaign 7 candidate rows must record:

```text
metal_simplify_primitive_stack.sort: bitonic_sort_words1
metal_simplify_primitive_stack.prefix_sum: hillis_steele_inclusive_scan_uint32
metal_simplify_primitive_stack.reduce_by_key: head_parallel_duplicate_sum_words1
metal_simplify_coefficient_domain: signed_fixed32_dyadic_coefficients_only
metal_execution.kernel_stack
padded_terms
bitonic_passes
prefix_sum_passes
workspace_reserved_bytes
```

Apple Metal does not support `double` arithmetic in kernels on the local
Campaign 7 host. Therefore Campaign 7 device-candidate rows are benchmark-only
and limited to coefficients exactly representable as signed fixed32 dyadic
values whose accumulated fixed32 sums and tolerance threshold fit exact uint64
squared-magnitude comparison. Reports must not present those rows as a general
FP64 Metal simplify implementation or change public `DevicePauliSum.simplify()`
behavior. Multi-word inputs, non-fixed-dyadic coefficients, accumulator-overflow
risk, and nonzero tolerance cases outside the exact squared-magnitude comparison
domain must be recorded as unavailable or rejected with evidence.

Apple Metal Campaign 8 extends Campaign 7 with performance-relevance evidence
for the same private one-word fixed-dyadic simplify candidate. Rows with
variant `metal_simplify_device_candidate` and status `ok` must record:

```text
campaign8_timing_schema: checked_device_resident_simplify_v1
timing_decomposition_seconds.host_preflight
timing_decomposition_seconds.scratch_and_output_allocation
timing_decomposition_seconds.command_encoding
timing_decomposition_seconds.command_execution
timing_decomposition_seconds.output_accounting
timing_decomposition_seconds.total_observed
dispatch_counts.total_kernel_dispatches
pipeline_cache.boundary
pipeline_cache.library_source
performance_decision.candidate_status
```

The row may remain `benchmark_only`, `experimental`, or
`performance_relevant`, but public promotion requires a report showing that the
checked candidate beats same-host CPU default simplify and the Metal
transfer-reference path on at least one duplicate-heavy or cancellation
workload without broadening the accepted fixed32 correctness domain. Rejected
or unavailable Campaign 8 rows must keep `transfer_boundary: status_only` and
must not claim command buffers or kernel stacks.

Post-Phase 11 H100 CUDA optimization benchmarks, including campaign 2 and later
campaigns, must also record allocation and materialization boundaries when an
experiment changes temporary storage, workspace reuse, or output ownership:

```text
workspace enabled or disabled
workspace storage pre-reserved, allowed to grow inside timing, or absent
temporary storage bytes requested
allocation count when available, or unavailable reason
CUDA stream mode
result materialization target, such as vector-return host bytes, caller-owned host bytes, caller-owned device bytes, bit-packed prototype, or private benchmark-only output
duplicate rate and survivor count for simplify and matmul+simplify
correctness oracle and tolerance for every measured case
```

CUDA Campaign 4 benchmark JSON rows use these normalized fields where applicable:

```text
workspace_mode: absent, grow_inside_timing, or pre_reserved_outside_timing
workspace_reserved_bytes
workspace_high_watermark_bytes
workspace_allocation_count
workspace_growth_count
cub_strategy: none, device_radix_sort_reduce, device_run_length_encode, or device_reduce_by_key
scratch_bytes_requested
result_materialization_target: host_vector, caller_owned_host_bytes, caller_owned_device_bytes, bitpacked_device_words, or none
timing_boundary: transfer_inclusive, device_resident, preallocated, or prototype
```

Campaign 5 public device-output rows must additionally separate these fields
for pairwise commutation:

```text
cuda_device_output_allocate_seconds and p10/p90/min/max variants
cuda_device_output_reuse_seconds and p10/p90/min/max variants
cuda_device_output_to_host_seconds and p10/p90/min/max variants
cuda_device_output_cuda_array_interface_export_seconds and p10/p90/min/max variants
result_materialization_target: device_uint8_matrix
timing_boundary: device_output_allocating, device_output_reused, or device_output_to_host
```

`cuda_device_output_allocate_seconds` times the public
`DevicePauliSum.commutes_with_device()` allocation and kernel fill without host
materialization. `cuda_device_output_reuse_seconds` times the same kernel fill
into caller-owned `DeviceCommutationMatrix` storage. `cuda_device_output_to_host`
times `DeviceCommutationMatrix.to_host()` separately. Device-output plots must
not compare these rows against host-output rows without making the output
boundary explicit.

Campaign 6 commutation consumer rows must add compact-summary and CUDA Array
Interface consumer fields without relabeling them as raw Wolfgang kernel fill
timings:

```text
cuda_device_output_consumer_total_seconds and p10/p90/min/max variants
cuda_device_output_consumer_axis0_seconds and p10/p90/min/max variants
cuda_device_output_consumer_axis1_seconds and p10/p90/min/max variants
cuda_device_output_consumer_to_host_bytes
cuda_device_output_dense_to_host_seconds and p10/p90/min/max variants
cupy_asarray_export_seconds and p10/p90/min/max variants when CuPy is available
cupy_sum_total_seconds and p10/p90/min/max variants when CuPy is available
cupy_sum_axis0_seconds and p10/p90/min/max variants when CuPy is available
cupy_sum_axis1_seconds and p10/p90/min/max variants when CuPy is available
cupy_dense_to_host_seconds and p10/p90/min/max variants when CuPy is available
```

`cuda_device_output_consumer_*` fields time public
`DeviceCommutationMatrix.count_commuting(...)` reductions that execute on the
matrix CUDA device and copy compact `uint64` counts to host. `cupy_*` fields
time an external CuPy consumer over `DeviceCommutationMatrix.__cuda_array_interface__`;
they must be labeled as interop consumer timings and kept separate from
Wolfgang kernel timings.

ROCm Campaign 2 device-output rows must use the same boundary discipline for
HIP-backed `DeviceCommutationMatrix` without claiming CUDA Array Interface or
DLPack interop. When available, JSON rows must include:

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
result_materialization_targets: list containing every retained materialization target when a row reports multiple compact consumers
timing_boundary: transfer_inclusive, device_operand_host_output, device_output_allocating, device_output_reused, device_output_to_host, or compact_consumer
```

`hip_device_output_allocate_seconds` times
`DevicePauliSum.commutes_with_device()` allocation plus HIP kernel fill.
`hip_device_output_reuse_seconds` times the same kernel write targeting a
caller-owned HIP `DeviceCommutationMatrix`.
`hip_device_output_to_host_seconds` times explicit
dense materialization through `DeviceCommutationMatrix.to_host()`.
`hip_count_commuting_*` and `hip_conflict_degrees_*` rows time compact public
consumers that return scalar or `uint64` summaries to the host. Reports and
plots must not compare compact-consumer timings to dense host-output timings
without labeling the materialization target.

ROCm Campaign 3 simplify rows must keep transfer-inclusive, device-resident,
and explicit host-materialization boundaries separate. When a row reports
`status: ok`, JSON must include:

```text
operation: simplify
backend: hip
hip_simplify_transfer_seconds and p10/p90/min/max variants
hip_simplify_device_resident_seconds and p10/p90/min/max variants
hip_simplify_to_host_seconds and p10/p90/min/max variants
hip_simplify_strategy: rocthrust_default, hipcub_radix_sort_reduce, custom_packed_key, or unavailable
hip_simplify_strategy_status: retained, rejected_with_evidence, benchmark_only, or unavailable
hip_simplify_output_terms
hip_simplify_output_words
result_materialization_target: device_pauli_sum or host_pauli_sum
timing_boundary: transfer_inclusive, device_resident, or device_output_to_host
correctness_digest with input_terms, output_terms, coefficient_l1, and canonical_label_hash
campaign3_headroom_statuses with terminal status for DLPack, streams, workspaces, packed summaries, expectation, matmul, portability, ROCm wheels, multi-GPU, and simultaneous CUDA+HIP
```

`hip_simplify_transfer_seconds` times host `PauliSum.to_device().simplify()`
through HIP-backed output construction. `hip_simplify_device_resident_seconds`
times `DevicePauliSum.simplify()` on already-resident HIP operands.
`hip_simplify_to_host_seconds` times explicit `DevicePauliSum.to_host()`
materialization after simplify. A report may not combine these boundaries into
a single speedup claim.

Campaign 3 terminal statuses must use:

```text
accepted
rejected_with_evidence
deferred_with_blocker
out_of_scope_with_next_trigger
blocked_external
```

Hard out-of-implementation-scope items cannot use `accepted` as their
implementation status. Public HIP DLPack, streams, workspaces, expectation,
matmul, multi-GPU execution, ROCm wheels, additional AMD GPU support claims,
and simultaneous CUDA+HIP builds require a separate plan or architecture
decision before they may move from terminal-status reporting into retained
implementation scope.

ROCm Campaign 4 simplify-hardening rows must identify private strategy,
workspace, and generic multi-word boundaries. When a row reports `status: ok`,
JSON must include:

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

Campaign 4 rows must not describe a private workspace or custom strategy as a
public API. `hip_simplify_strategy` must name the attempted candidate even when
the candidate is rejected or unavailable; `hip_simplify_strategy_status` and
`hip_simplify_strategy_reason` carry the outcome. Public workspace, stream,
DLPack, expectation, matmul, multi-GPU, ROCm wheel, portability, and
simultaneous CUDA+HIP claims require separate plans even if Campaign 4 captures
availability diagnostics.

ROCm Campaign 5 interop and execution-control rows must close the Campaign 4
public-boundary headroom with real consumer evidence or explicit terminal
statuses. Rows must include:

```text
campaign: rocm_mi300x_campaign5
operation: commutation_interop, stream_graph_decision, workspace_decision, expectation_decision, matmul_decision, portability_decision, packaging_decision, multi_gpu_decision, or multi_backend_decision
backend: hip
mode: dlpack_pytorch, dlpack_cupy, dlpack_other_consumer, cuda_array_interface_guard, stream_graph_probe, workspace_probe, expectation_decision, matmul_decision, portability_decision, packaging_decision, multi_gpu_decision, or simultaneous_cuda_hip_decision
status: ok, retained, rejected_with_evidence, blocked_external, unavailable, or out_of_scope_with_next_trigger
final_status: retained, rejected_with_evidence, blocked_external, unavailable, or out_of_scope_with_next_trigger
hip_dlpack_device_type: 10 when HIP DLPack is retained, otherwise null
hip_dlpack_device_type_name: kDLROCM when HIP DLPack is retained, otherwise unavailable
consumer_library
consumer_version
consumer_backend
consumer_available
consumer_import_error
consumer_correctness_passed
consumer_read_only_enforced
consumer_mutation_error
hip_dlpack_export_seconds and p10/p90/min/max variants when retained
consumer_from_dlpack_seconds and p10/p90/min/max variants when a consumer runs
consumer_sum_seconds and p10/p90/min/max variants when a consumer runs
hip_device_output_to_host_seconds and p10/p90/min/max variants
hip_count_commuting_axis_none_seconds and p10/p90/min/max variants
timing_boundary: dlpack_producer, framework_consumer, compact_consumer, device_output_to_host, decision_only, or benchmark_only
correctness_digest with matrix_shape, host_sum, consumer_sum, and canonical_matrix_hash
campaign5_terminal_statuses for DLPack, CUDA Array Interface guard, streams, graphs, workspaces, expectation, matmul, portability, ROCm wheels, multi-GPU, and simultaneous CUDA+HIP
```

Campaign 5 rows may not retain HIP DLPack unless at least one real ROCm DLPack
consumer validates on MI300X and rejects mutation of the imported read-only
view. If no consumer validates, HIP DLPack rows must be `blocked_external` and
the public HIP DLPack surface must remain unavailable. HIP CUDA Array Interface
rows must remain `rejected_with_evidence`. Public stream, graph, and workspace
rows must be `rejected_with_evidence` unless they include a complete accepted
API contract and measured benefit. HIP expectation, HIP matmul, portability,
ROCm wheel, multi-GPU, and simultaneous CUDA+HIP rows must use terminal
statuses rather than open-ended follow-up text.

ROCm Campaign 6 expectation and matmul parity rows must prove that the existing
public `DevicePauliSum` methods are retained for HIP without adding new public
API shape. Rows must include:

```text
campaign: rocm_mi300x_campaign6
operation: expectation_statevector or matmul
backend: hip
mode: host_complex128, host_complex64, matmul_simplify_true, matmul_simplify_false, external_device_pointer_guard, profiler_expectation, profiler_matmul, portability_decision, packaging_decision, multi_gpu_decision, or simultaneous_cuda_hip_decision
status: ok, retained, rejected_with_evidence, blocked_external, unavailable, or out_of_scope_with_next_trigger
final_status: retained, rejected_with_evidence, unavailable, or out_of_scope_with_next_trigger
timing_boundary: transfer_inclusive, device_resident_kernel, device_output_to_host, decision_only, or benchmark_only
hip_expectation_input_dtype
hip_expectation_state_size
hip_expectation_num_terms
hip_expectation_words
hip_expectation_transfer_seconds and p10/p90/min/max variants
hip_expectation_device_resident_seconds and p10/p90/min/max variants
hip_expectation_result_copy_seconds and p10/p90/min/max variants
hip_matmul_lhs_terms
hip_matmul_rhs_terms
hip_matmul_output_terms
hip_matmul_words
hip_matmul_simplify_output
hip_matmul_transfer_seconds and p10/p90/min/max variants
hip_matmul_device_resident_seconds and p10/p90/min/max variants
hip_matmul_to_host_seconds and p10/p90/min/max variants
correctness_digest with operation-specific label hash, coefficient_l1, and result summary
campaign6_terminal_statuses for expectation, matmul, external device pointers, DLPack, CUDA Array Interface guard, streams, graphs, workspaces, portability, ROCm wheels, multi-GPU, and simultaneous CUDA+HIP
```

Campaign 6 rows may retain HIP expectation only for host NumPy `complex64` and
`complex128` statevectors. External HIP device-pointer statevectors, HIP
DLPack, HIP CUDA Array Interface, public streams, graph replay, and public
workspaces remain unavailable or rejected unless a separate plan accepts those
public contracts. Campaign 6 matmul rows must report whether
`simplify_output` is true or false and must not copy to host for
`simplify=True` before duplicate reduction.

ROCm Campaign 7 release-support rows must turn retained MI300X HIP operation
evidence into release-lane evidence without claiming ROCm wheels or broad AMD
GPU portability. Rows must include:

```text
campaign: rocm_mi300x_campaign7
operation: release_source_build, runtime_validation, retained_operation_smoke, profiler_smoke, duplicate_pressure_probe, portability_lane, packaging_decision, ci_runbook, or backend_neutral_decision
backend: hip, cpu, cuda, or none
mode: mi300x_repeatability, cpu_only_control, cuda_hip_rejection, retained_transfer, retained_commutation, retained_device_consumers, retained_simplify, retained_expectation, retained_matmul, simplify_duplicate_pressure, matmul_duplicate_pressure, rocprof_availability, alternate_amd_gpu_probe, rocm_wheel_policy, release_runbook, external_statevector_decision, multi_gpu_decision, or simultaneous_cuda_hip_decision
status: ok, passed, retained, rejected_with_evidence, blocked_external, unavailable, or out_of_scope_with_next_trigger
final_status: passed, retained, rejected_with_evidence, blocked_external, unavailable, or out_of_scope_with_next_trigger
host_role: local_cpu_control, primary_mi300x, alternate_amd_gpu, or decision_only
rocm_runtime_version
rocm_toolkit_version
hip_compiler_version
gpu_name
gfx_target
build_command
validation_command
profiler_command
timing_boundary: transfer_inclusive, device_resident, compact_consumer, device_output_to_host, source_build, validation_only, profiler_only, decision_only, or benchmark_only
correctness_digest
campaign7_terminal_statuses
```

Campaign 7 rows may use `passed` only for actual build, validation, profiler,
or benchmark evidence. Additional AMD GPU support requires source build,
runtime validation, retained-operation tests, benchmark smoke, and profiler
status from that GPU. If no alternate AMD GPU is available, the portability row
must be `blocked_external` and must record the access blocker. ROCm wheel,
multi-GPU ROCm, simultaneous CUDA+HIP, external HIP statevector interop, public
streams, public graphs, and public workspaces must stay unavailable or
out-of-scope unless a separate architecture plan accepts those public
boundaries.

`campaign7_terminal_statuses` must contain this exact residual-status key set:

```text
mi300x_repeatability
cpu_only_control
rocm_source_build_runbook
rocm_ci_or_release_lane
rocm_packaging_policy
rocm_wheel_support
alternate_amd_gpu_portability
profiler_availability
duplicate_pressure_simplify
duplicate_pressure_matmul
external_statevector_interop
hip_dlpack
hip_cuda_array_interface
public_streams
public_graphs
public_workspaces
multi_gpu_rocm
simultaneous_cuda_hip
backend_neutral_accelerator_design
```

ROCm Campaign 8 architecture-readiness rows must convert Campaign 7 residual
risks into explicit gates without adding HIP kernels, public APIs, wheels,
multi-GPU execution, or simultaneous CUDA+HIP builds. Rows must include:

```text
campaign: rocm_campaign8_architecture_readiness
operation: backend_neutral_decision, portability_gate, packaging_gate, profiler_migration, interop_reconsideration, performance_reopen_gate, or release_lane_retention
backend: hip, cuda, cpu, multi_backend, or none
mode: backend_neutral_object_model, simultaneous_cuda_hip_source_builds, multi_gpu_rocm_execution, non_mi300x_amd_portability, rocm_wheel_packaging_design, rocm_ci_hardware_policy, rocm_clean_machine_install_tests, rocprofv3_migration, legacy_rocprof_retention, external_hip_statevector_contract, hip_dlpack_reconsideration_contract, hip_cuda_array_interface_policy, public_streams_policy, public_graphs_policy, public_workspaces_policy, targeted_rocm_performance_reopen, or source_build_release_lane_retention
status: accepted_for_future_implementation, retained, rejected_with_evidence, blocked_external, unavailable, or out_of_scope_with_next_trigger
final_status: accepted_for_future_implementation, retained, rejected_with_evidence, blocked_external, unavailable, or out_of_scope_with_next_trigger
host_role: local_cpu_control, primary_mi300x, alternate_amd_gpu, packaging_runner, profiler_runner, or decision_only
decision_doc
evidence_command
validation_command
support_claim_change: none, source_build_only, packaging_gate_only, portability_gate_only, or architecture_gate_only
correctness_oracle
benchmark_boundary
campaign8_terminal_statuses
```

`campaign8_terminal_statuses` must contain this exact residual-status key set:

```text
backend_neutral_object_model
simultaneous_cuda_hip_source_builds
multi_gpu_rocm_execution
non_mi300x_amd_portability
rocm_wheel_packaging_design
rocm_ci_hardware_policy
rocm_clean_machine_install_tests
rocprofv3_migration
legacy_rocprof_retention
external_hip_statevector_contract
hip_dlpack_reconsideration_contract
hip_cuda_array_interface_policy
public_streams_policy
public_graphs_policy
public_workspaces_policy
targeted_rocm_performance_reopen
source_build_release_lane_retention
```

Campaign 8 rows must record a concrete decision document or evidence command
for every final status. A new ROCm performance campaign may be opened only when
the Campaign 8 performance-reopen row names:

```text
retained operation
profiler artifact
measured bottleneck
proposed implementation
correctness oracle
A/B timing boundary
rejection criteria
```

Same-host ROCm reruns without a profiler-backed retained-operation bottleneck
must be rejected as release-support repetition rather than performance work.

Backend-Neutral Accelerator Campaign 9 rows must treat accelerator support as
target-specific build evidence, not as a speedup claim by itself. The campaign
plan is `docs/plans/backend_neutral_accelerator_campaign9_plan.md`.

Required Campaign 9 metadata fields:

```text
campaign: backend_neutral_accelerator_campaign9
backend_neutral_status_schema
object_local_backend_identity
backend_construction_selector_contract
device_commutation_matrix_backend_property
ambiguous_dual_runtime_policy
target_specific_accelerator_builds
mixed_cuda_hip_build_rejection
future_multi_runtime_design_gate
same_backend_same_device_validation
cpu_only_header_safety
cuda_target_regression_lane
hip_target_regression_lane
benchmark_boundary_reporting
no_wheel_or_portability_claim
build_mode: cpu_only, cuda_only, hip_only, or metal_only
object_backend: cpu, cuda, hip, or metal
compiled_backends
runtime_visible_backends
Campaign 9 legacy transfer_boundary set: transfer_boundary: transfer_inclusive, device_resident, host_materialized, compact_consumer, or status_only
transfer_boundary: transfer_inclusive, device_resident, device_output_allocating, device_output_reused, device_output_to_host, device_resident_private_output_blit_to_shared_staging, host_materialized, compact_consumer, compact_consumer_gpu_reduction, compact_consumer_gpu_parallel_block_reduction, or status_only
```

The repo-local benchmark helper
`benchmarks._benchmark_metadata.benchmark_row_boundary()` is the preferred
normalization path for these fields. `_build_info()` must provide
`accelerator_build_mode`, `compiled_accelerator_backends`, and
`runtime_visible_accelerator_backends` so benchmark reports do not infer
Campaign 9 support boundaries from one global active accelerator. Synthetic or
defensive metadata with more than one accelerator build flag true must be
rejected by benchmark helpers rather than normalized into a supported
Campaign 9 build mode.

Campaign 9 rows must compare only like-for-like operation and timing boundaries
across CPU-only, CUDA-only target, HIP-only target, and Apple Metal target builds. A
configure, import, status, or dual-request rejection check is a `status_only`
row unless it also captures a measured operation under the same transfer
boundary as its baseline. Mixed CUDA+HIP source-build evidence is future-only
and must not appear as a supported Campaign 9 build mode unless a later
accepted mixed-runtime plan updates this protocol.
README wording must keep ROCm wheels, combined accelerator wheels, non-MI300X
AMD portability, HIP DLPack, multi-GPU ROCm, and Metal runtime or wheel support
out of support claims until separate evidence exists.

CUDA Campaign 9 deferred-headroom rows must close each H100 CUDA Campaign 8
remaining item with a final non-deferred status:

```text
campaign: cuda_deferred_headroom_campaign9
campaign8_headroom_item: integer in 1..6
final_status: implemented, passed, rejected_with_evidence, failed, or blocked_external
deferred_status_allowed: false
decision_doc: path to the final decision or report
```

Accepted Campaign 9 row-specific timing fields include:

```text
conflict_degrees_axis_none_seconds and p10/p90/min/max variants
conflict_degrees_axis_0_seconds and p10/p90/min/max variants
conflict_degrees_axis_1_seconds and p10/p90/min/max variants
dense_to_host_plus_numpy_conflicts_seconds and p10/p90/min/max variants
cupy_dlpack_from_dlpack_seconds and p10/p90/min/max variants
cupy_dlpack_sum_total_seconds and p10/p90/min/max variants
cupy_dlpack_dense_to_host_seconds and p10/p90/min/max variants
```

Large profiler binaries such as `.ncu-rep` may be kept out of git when they
are too large for normal repository storage, but the report must check in
human-readable CSV/stdout/stderr evidence, record the omitted binary path, and
add ignore rules so the binary is not accidentally staged.

Plots and A/B summaries must compare the same timing boundary. A report must
not present a workspace-preallocated or device-output prototype as a speedup
over a transfer-inclusive or allocation-inclusive public path unless the labels
make the boundary difference explicit.

The README performance plot must remain a broad landscape view, not a narrow
single-hot-path snapshot. When refreshing CUDA evidence, regenerate and check in
the README plot from tracked raw JSON so it includes the available operation
families, CPU scalar, captured optimized CPU selectors, CUDA transfer-inclusive
and device-resident timings, boundary-specific CUDA rows such as
operator-resident host-statevector timings where relevant, and semantically
comparable external baselines where available. Renderer freshness tests must
fail when this view is missing or stale.

For CPU:

```text
scalar baseline timing must be available before optimized CPU claims
forced scalar and forced optimized backends must report which path executed
thread count must be fixed or reported for oneTBB measurements
native-tuned source builds must not be compared against portable wheels without saying so
```

## Phase Benchmark Gates

Phase 4:

```text
bench_simplify.py exists
low-duplicate and high-duplicate simplify cases are measured
Wolfgang scalar CPU is compared with a Python, Qiskit, or OpenFermion baseline
Apple Silicon scalar CPU timing is captured when local hardware is available
```

Phase 5:

```text
bench_multiply.py exists
single-term multiplication benchmark reports Wolfgang scalar and Python baseline timings
small cross-product multiplication benchmark reports Wolfgang scalar and Python baseline timings
simplified duplicate cross-product benchmark reports default simplify_output=true timing
max_intermediate_terms prevents unsafe benchmark sizes
```

Phase 6:

```text
bench_grouping.py exists
pairwise commutation benchmark reports Wolfgang scalar and Python baseline timings
QWC grouping benchmark reports Wolfgang scalar and Python baseline timings
full grouping benchmark reports Wolfgang scalar and Python baseline timings
large-output guardrail cases are measured separately from successful dense matrix output
```

Phase 7:

```text
bench_openfermion_conversion.py exists
round-trip OpenFermion conversion reports Wolfgang scalar and OpenFermion baseline timings
large sparse QubitOperator-style workloads report Wolfgang scalar and OpenFermion baseline timings
```

Phase 8:

```text
bench_expectation.py exists
statevector_few_terms_large_state reports Wolfgang scalar and direct Python timings
statevector_many_terms_small_state reports Wolfgang scalar and direct Python timings
statevector_diagonal_many_terms reports Wolfgang scalar and direct Python timings
z_counts reports Wolfgang scalar and direct Python timings
few-terms-large-statevector and many-terms-small-statevector regimes are measured
```

Phase 9:

```text
bench_cpu_dispatch.py exists
bench_cpu_thresholds.py exists and characterizes the pairwise oneTBB auto-dispatch threshold
scalar, oneTBB, and SIMD paths are compared where available
unavailable optimized paths are reported with reasons instead of omitted
optimization PRs include before/after results
Apple Silicon and x86_64 optimization results are reported separately when both are available
checked-in CPU evidence reports live under docs/benchmarks/reports/ for hardware used to make performance or dispatch-threshold claims
reports identify compute-bound, memory-bound, dispatch-overhead, and threading-overhead regimes when supported by the data
CPU hardening before Phase 10 includes smoke, default, and stress-sized datasets for simplify, multiplication, commutation/grouping, statevector expectation, and Z-count expectation
CPU hardening stress datasets exercise both one-word and two-word packed Pauli paths when scalar optimizations touch those regimes
CPU hardening benchmark results must keep correctness checks enabled for the measured case; a benchmark may not skip semantic comparison just to time a larger workload
CPU hardening reports include direct Python references and optional Qiskit/OpenFermion baselines where the dependency operation is meaningfully comparable
```

Phase 10:

```text
transfer timing hooks exist
no CUDA speedup claims are made before kernels land
```

Phase 11:

```text
bench_cuda_kernels.py exists and can run smoke, default, and stress-sized CUDA datasets
bench_cuda_scaling.py exists and can run smoke, default, and stress-sized CUDA scale profiles
CPU and CUDA are compared on the same datasets
every available optimized CPU selector covered by a CUDA benchmark operation is timed separately
unavailable optimized CPU selectors are reported with reasons instead of omitted
transfer-inclusive and device-resident timings are reported
README plots include CPU scalar, every captured optimized CPU selector, CUDA transfer-inclusive, boundary-specific CUDA rows, CUDA device-resident, and semantically comparable external package baselines from checked-in evidence reports
README plots record unavailable external baseline status or reasons when a planned comparable baseline cannot run
GPU-library competitors are classified and included when a comparable accelerator-library workload mapping exists
reports identify CPU-faster, CUDA-faster, and transfer-bound regimes
CUDA performance hillclimb reports record Nsight Systems, Nsight Compute, Compute Sanitizer, CPU profiler evidence, and cuobjdump or nvdisasm evidence when binary/PTX/SASS changes are considered
profiling reports distinguish kernel execution limits from host allocation, synchronization, transfer, Python binding, and result materialization limits
overnight hillclimb reports satisfy the Comprehensive Overnight Optimization Report requirements, including open-source competitor installation/comparison and publication-quality visuals
```

Campaign 3 established the README-ready cross-comparison format. Future H100
or GPU reports that update README benchmark visuals must keep the same
discipline: broad CPU/CUDA/external comparisons from checked raw JSON first,
CUDA-only before/after plots only as supporting report evidence, and any
private prototype path visibly labeled as a prototype timing boundary.

Post-CUDA accelerator phases:

```text
HIP and Metal benchmark reports follow the same transfer-inclusive and device-resident split as CUDA
HIP and Metal reports are compared against the same scalar and optimized CPU datasets before speedup claims
backend-specific claims include toolkit, runtime, device model, architecture, and transfer topology where available
HIP reports label device-operand timings separately when dense results are still materialized to host
HIP reports retain rocprof trace, stats, counters, or an explicit provider/permission diagnosis
HIP external baselines are included only when the semantic mapping and timing boundary are documented
Metal reports record Apple SoC, GPU core count when available, macOS version,
Xcode or Command Line Tools version, Metal device name, storage mode,
threadgroup size, grid shape, command-buffer synchronization boundary, and
buffer allocation or reuse boundary.
Metal profiler evidence comes from Xcode Instruments, Metal System Trace,
`xctrace`, or a precise tooling blocker.
Metal rows in README landscape plots must include the same-host CPU scalar,
default, and NEON rows where available, plus transfer-inclusive,
device-resident host-output, retained device-output allocation, retained
device-output reuse, explicit `to_host()`, and compact-consumer rows when those
boundaries are measured.
MPS, MPSGraph, and PyTorch `mps` are external baselines only. Reports must
record semantic mapping, timing boundary, correctness oracle, version or system
library provenance, and unavailable reasons before those rows appear in README
plots.
```

ROCm Campaign 2 rows for HIP device-resident commutation outputs must separate
allocation, reuse, compact-consumer, and host-materialization boundaries:

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
result_materialization_targets: list containing every retained materialization target when a row reports multiple compact consumers
timing_boundary: transfer_inclusive, device_operand_host_output, device_output_allocating, device_output_reused, device_output_to_host, or compact_consumer
```

HIP device-output plots must not compare compact summary rows against dense
host-output rows without labeling the output boundary. README performance
landscape plots that include ROCm must keep CPU scalar, optimized CPU, CUDA
where captured, HIP, and external baselines visible together rather than
presenting a narrow ROCm-only view.

## Reporting Template

Use this template in PR descriptions, release notes, or checked-in benchmark reports:

```text
Benchmark revision: <git commit, or git commit plus explicit dirty-tree provenance>
Command: <exact command>
Environment: <CPU/GPU/compiler/Python summary>
Dataset: <num_qubits, num_terms, term_weight, duplicate_rate, dtype, seed>
Baselines: <baseline names>
Results: <median and variability>
Interpretation: <what changed and why it matters>
Limitations: <known caveats>
```

Do not describe a result as a speedup unless the baseline, dataset, command, and revision are present.

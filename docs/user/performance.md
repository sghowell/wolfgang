# Performance

Wolfgang always ships the portable scalar CPU backend and can additionally
compile optional oneTBB and SIMD backends for commutation-heavy workloads.
Current optimized coverage is intentionally narrow: pairwise commutation and
full grouping graph construction. CUDA source builds additionally expose
device-resident simplify, statevector expectation, pairwise commutation, and
matrix-product kernels on `DevicePauliSum`.

## Backend Selection

The runtime backend selector is controlled by `FASTPAULI_CPU_BACKEND`.

```bash
FASTPAULI_CPU_BACKEND=auto python benchmarks/bench_cpu_dispatch.py --smoke --repeat 1
FASTPAULI_CPU_BACKEND=scalar python benchmarks/bench_cpu_dispatch.py --smoke --repeat 1
```

Supported selectors:

```text
auto
scalar
tbb
avx2
avx512
neon
sve
```

`scalar` forces the portable baseline. The optimized selectors fail clearly
unless the corresponding backend was compiled and the current CPU supports it.
`auto` uses optimized kernels only where Wolfgang has coverage and benchmark
evidence:

```text
large pairwise commutation: oneTBB when compiled, otherwise AVX-512, AVX2, NEON, or scalar
full grouping graph construction: AVX-512, AVX2, NEON, or scalar
all other kernels: scalar
```

This keeps the default path fast for covered hot spots without implying that
every operation is SIMD- or oneTBB-accelerated.

## Build Metadata

`wolfgang_quantum._wolfgang_core._build_info()` reports backend state:

```python
import wolfgang_quantum._wolfgang_core as core

info = core._build_info()
assert info["active_cpu_backend"] == "scalar"
assert "scalar" in info["available_cpu_backends"]
print(info["optimized_cpu_kernels"])
```

Benchmark reports include:

```text
active CPU backend
requested CPU backend
compiled CPU backends
available CPU backends
unavailable backend reasons
CPU model and architecture
CPU instruction-set probe source or unavailable reason
compiler and CMake versions
CMake CPU options and compiled backend feature flags
compiler CPU flags reported through the build or environment
thread settings
oneTBB status
optimized CPU kernel coverage
CUDA status
```

## Benchmark Commands

Run the local smoke suite through the validation entrypoint:

```bash
python scripts/validate.py
```

Run the dispatch benchmark directly:

```bash
python benchmarks/bench_cpu_dispatch.py --smoke --repeat 1 --json
```

Run operation-specific benchmark smokes:

```bash
python benchmarks/bench_simplify.py --smoke --repeat 1 --json
python benchmarks/bench_multiply.py --smoke --repeat 1 --json
python benchmarks/bench_grouping.py --smoke --repeat 1 --json
python benchmarks/bench_expectation.py --smoke --repeat 1 --json
python benchmarks/bench_cuda_kernels.py --smoke --repeat 1 --json
python benchmarks/bench_competitive_baselines.py --smoke --repeat 1 --json
```

Do not treat benchmark smoke timings as performance claims. They prove that the
benchmark protocol runs and records metadata. Performance claims require a
dedicated run on named hardware, with scalar and optimized paths compared only
where the optimized paths are compiled and available.

## CPU Hardening Checkpoint

Before CUDA kernel implementation began, Wolfgang ran a dedicated CPU
hardening checkpoint. That work optimized scalar CPU hot paths first and records
before/after benchmark evidence for:

```text
simplify and canonical duplicate reduction across low- and high-duplicate regimes
Pauli-sum multiplication, including default simplified output under duplicate pressure
pairwise commutation and grouping
statevector expectation, including all-diagonal statevector workloads
diagonal Z-count expectation
optional Qiskit and OpenFermion competitive baselines when installed
```

Hardening benchmarks must keep correctness checks enabled for the measured
datasets. Results are not accepted if they remove reference comparisons, change
public APIs, alter coefficient semantics, weaken ordering guarantees, or imply
optimized or GPU coverage outside the kernels, build configurations, and
hardware targets that were compiled, validated, and benchmarked.

Run the combined CPU hardening suite with:

```bash
python benchmarks/bench_cpu_hardening.py --profile default --repeat 3 --warmup 1 --json
```

Use `--profile smoke` for quick validation and `--profile stress` for broader
CPU evidence.

Run competitive baselines with optional dependencies installed:

```bash
python benchmarks/bench_competitive_baselines.py --repeat 3 --warmup 1 --json
```

The competitive benchmark records unavailable libraries rather than inventing
comparisons. Current comparable baselines are Qiskit `SparsePauliOp` simplify
and grouping, and OpenFermion `QubitOperator` multiplication. Per-case output
records whether competitor correctness was actually checked and, for
OpenFermion, the effective canonical operand sizes used by the timed
multiplication.

## Benchmark Plots

Checked-in benchmark plots are generated from checked-in evidence reports, not
from ad hoc local timings. The current README plot is generated from the
Campaign 10 summary, which preserves broad Campaign 9 comparison rows and adds
cross-architecture A100 plus RTX PRO 6000 Blackwell source-build evidence,
PyTorch CUDA DLPack coverage, and final public-grouping, stream/graph, and CSR
scatter decisions:

```bash
python scripts/render_cuda_campaign10_assets.py \
  --data-dir docs/benchmarks/data/cuda_cross_architecture_campaign10_2026-04-29 \
  --plot-dir docs/benchmarks/plots
```

The prior Campaign 9 deferred-headroom plot is generated with:

```bash
python scripts/render_cuda_campaign9_assets.py \
  --data-dir docs/benchmarks/data/cuda_deferred_headroom_campaign9_2026-04-29 \
  --plot-dir docs/benchmarks/plots
```

The prior Campaign 8 compact-consumer plot is generated with:

```bash
python scripts/render_cuda_campaign8_assets.py \
  --data-dir docs/benchmarks/data/cuda_deep_optimization_h100_campaign8_2026-04-29 \
  --plot-dir docs/benchmarks/plots
```

The prior Campaign 7 fused-consumer plot is generated with:

```bash
python scripts/render_cuda_campaign7_assets.py \
  --data-dir docs/benchmarks/data/cuda_deep_optimization_h100_campaign7_2026-04-29 \
  --plot-dir docs/benchmarks/plots
```

The prior Campaign 6 consumer-count plot is generated with:

```bash
python scripts/render_cuda_campaign6_assets.py \
  --data-dir docs/benchmarks/data/cuda_deep_optimization_h100_campaign6_2026-04-29 \
  --plot-dir docs/benchmarks/plots
```

The plot includes CPU scalar, every optimized CPU selector captured by that
report for the benchmarked operation, CUDA transfer-inclusive, and CUDA
device-resident timings, plus operator-resident, device-output, compact
consumer, CuPy consumer, and external package points where the operation
supports those boundaries. Speedups are relative to CPU scalar on the same
deterministic dataset. Treat the figure as source-build evidence for the named
hardware and report revision only.

## SIMD And oneTBB

Explicit SIMD and oneTBB are optional compiled backends. They do not change the
release-wheel scalar baseline and they are reported as unavailable when the
compiler, package, or hardware support is absent.

Current covered optimized kernels:

```text
oneTBB: commutes_with, full_group_commutation_graph
AVX2: commutes_with_words_1_2, full_group_commutation_graph_words_1_2
AVX-512: commutes_with_words_1_2, full_group_commutation_graph_words_1_2
NEON: commutes_with_words_1_2, full_group_commutation_graph_words_1_2
```

Use `benchmarks/bench_cpu_dispatch.py` to compare `auto`, forced scalar, and
every compiled optimized selector on the same deterministic datasets. Use
`benchmarks/bench_cpu_thresholds.py` to characterize the large pairwise
commutation threshold where `auto` may switch to oneTBB.

Forced optimized selectors fail clearly for scalar-only operations instead of
relabeling scalar execution as optimized. Forced SIMD selectors also fail
clearly for commutation workloads outside the current one- or two-word
packed-width coverage. `auto` may still choose scalar for unsupported
operations or widths.

oneTBB is used for large dense pairwise commutation where thread overhead is
amortized. Full graph construction remains available as a forced oneTBB path,
but `auto` currently prefers SIMD or scalar for that operation because the
measured oneTBB full-graph overhead did not win on the Phase 9 hardware.

## Apple Silicon And x86_64

Apple Silicon and x86_64 results are reported separately. Local Apple Silicon
results are first-class CPU evidence for portability and baseline timing. x86_64
results are required before making x86-specific optimization claims.

## CUDA Source-Build Kernels

CUDA remains opt-in at source-build time:

```bash
FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 python scripts/validate.py
python benchmarks/bench_cuda_kernels.py --profile default --repeat 3 --warmup 1 --json
python benchmarks/bench_cuda_scaling.py --profile default --repeat 3 --warmup 1 --json
```

Validated CUDA kernel coverage:

```text
DevicePauliSum.simplify
DevicePauliSum.expectation_statevector
DevicePauliSum.commutes_with
DevicePauliSum.commutes_with_device
DeviceCommutationMatrix.count_commuting
DeviceCommutationMatrix.conflict_degrees
DeviceCommutationMatrix.__dlpack__
DevicePauliSum.matmul
```

For repeated dense commutation workflows that keep the result on GPU, allocate
and reuse the experimental device-output matrix:

```python
import wolfgang_quantum

lhs_d = lhs.to_device(device=0)
rhs_d = rhs.to_device(device=0)

out = wolfgang_quantum.DeviceCommutationMatrix.empty(
    (lhs_d.num_terms, rhs_d.num_terms),
    device=lhs_d.device,
)
same = lhs_d.commutes_with_device(rhs_d, output=out)
assert same is out

cuda_view = out.__cuda_array_interface__
total_commuting = out.count_commuting()
row_counts = out.count_commuting(axis=1)
col_counts = out.count_commuting(axis=0)
total_conflicts = out.conflict_degrees()
row_conflicts = out.conflict_degrees(axis=1)
col_conflicts = out.conflict_degrees(axis=0)
dlpack_device = out.__dlpack_device__()
flags = out.to_host()
```

`DeviceCommutationMatrix` owns a dense row-major `uint8` CUDA buffer with `1`
for commuting pairs and `0` for anti-commuting pairs. `to_host()` is the host
materialization boundary and returns a NumPy bool matrix.
`count_commuting(axis=None|0|1)` and
`conflict_degrees(axis=None|0|1)` are compact downstream consumer boundaries:
the reductions execute on the CUDA device and copy only `uint64` count results
to host. `__dlpack__` exports a read-only dense `uint8` view for real CUDA
DLPack consumers such as CuPy. Benchmark claims for this API must report
device-output allocation, device-output reuse, CUDA-array-interface export,
DLPack export/consumer timing, compact-summary reductions, and `to_host()`
copy time separately.

CUDA benchmark reports include CPU scalar timing, CUDA transfer-inclusive timing,
CUDA device-resident timing, toolkit/runtime metadata, driver/device metadata,
compiled CUDA architectures, and active CPU backend metadata. Tiny smoke cases
are expected to be CPU-faster and transfer-bound; default and stress profiles
are used to identify regimes where keeping operands and statevectors resident on
the GPU is faster. `benchmarks/bench_cuda_scaling.py` extends the fixed profiles
across multiple sizes when profiling scaling behavior or hillclimbing a CUDA
change. The `extreme` scaling profile is reserved for explicit hardware evidence
runs because it executes large correctness-checked CPU references and dense CUDA
outputs. These are source-build benchmark claims only. CUDA wheel distribution
remains deferred until the release and packaging policy is updated.

The current Campaign 10 report closes all Campaign 9 remaining-headroom items
with final non-deferred decisions. It records A100 `sm_80` and RTX PRO 6000
Blackwell `sm_120` source-build evidence, validates PyTorch CUDA DLPack
consumption of dense `DeviceCommutationMatrix` buffers, keeps
`DeviceCommutationMatrix.conflict_degrees(axis=None|0|1)` as the compact public
summary API, rejects the true public grouping API with evidence, rejects public
stream/CUDA Graph replay because launch overhead is not dominant, and rejects
CSR scatter reopening because retained compact consumers avoid full CSR scatter.
GPU performance claims remain source-build hardware evidence only. The H100
Campaign 9 report records privileged Nsight Compute counter evidence and CuPy
DLPack export coverage. The H100 Campaign 8 report records private
benchmark-only compact device-resident graph
and grouping consumers over `DeviceCommutationMatrix`. The H100 Campaign 7
report records private benchmark-only fused anti-commutation CSR graph export,
conflict-degree summaries, and grouping-oriented summaries over
`DeviceCommutationMatrix`. The H100 Campaign 6 report records compact
`DeviceCommutationMatrix.count_commuting(axis=None|0|1)` consumers, CuPy CUDA
Array Interface consumer comparisons, Nsight Systems, Nsight Compute, Compute
Sanitizer, and broad CPU/CUDA/external evidence. The earlier H100 Nsight
hillclimb report also records rejected CUDA experiments, CPU `perf`, and x86
SIMD A/B evidence. Its retained CPU-side change improves forced AVX2 and
AVX-512 commutation stores, while oneTBB remains the preferred large dense
pairwise CPU selector on the measured x86 host.

Deeper CUDA optimization work follows
`docs/plans/cuda_deep_optimization_plan.md`. Public CUDA APIs remain
default-stream and synchronize-before-return. Stream-aware and async CUDA
surfaces are deferred until a dedicated API review accepts ownership, event,
Python lifetime, and error-propagation semantics. Public bit-packed commutation
output is also deferred; Campaign 6 did not retain a packed prototype because
dense `uint8` output remains useful for CUDA Array Interface consumers and no
capacity or bandwidth need was proven for a public packed layout. For
repeated dense commutation workloads, `DevicePauliSum.commutes_with_into(other,
output)` fills a caller-owned one-dimensional NumPy bool array and can be
benchmarked separately from the allocation performed by
`DevicePauliSum.commutes_with(other)`.

The completed Campaign 9 path is captured in
`docs/benchmarks/reports/cuda_deferred_headroom_campaign9_2026-04-29.md`.
The completed Campaign 8 path is captured in
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign8_2026-04-29.md`.
The Campaign 9 closure plan is captured in
`docs/plans/h100_deep_optimization_campaign9_plan.md`.
The Campaign 8 decision artifacts are
`docs/plans/cuda_fused_grouping_public_api_campaign8_review.md`,
`docs/plans/cuda_dlpack_interop_campaign8_review.md`, and
`docs/plans/cuda_graphs_stream_campaign8_decision.md`; each keeps its public
surface deferred for this campaign.

The completed Campaign 7 path is captured in
`docs/benchmarks/reports/cuda_deep_optimization_h100_campaign7_2026-04-29.md`.
The fused-consumer API review remains
`docs/plans/cuda_fused_commutation_consumer_api_review.md`; CSR graph
construction, conflict-degree summaries, and grouping summaries are private
benchmark-only helpers rather than documented user-facing API.

The completed Campaign 10 path is captured in
`docs/benchmarks/reports/cuda_cross_architecture_campaign10_2026-04-29.md`.
It replaces the Campaign 9 non-H100 blocker with A100 `sm_80` and RTX PRO 6000
Blackwell `sm_120` source-build evidence, adds PyTorch CUDA DLPack coverage,
rejects a true public grouping API while retaining
`DeviceCommutationMatrix.conflict_degrees(axis=None|0|1)`, rejects stream/CUDA
Graph work because launch overhead is not dominant, and keeps CSR scatter
closed because no retained Campaign 10 consumer requires full CSR edge lists.

Future CUDA optimization paths should start from Campaign 10 evidence. New work
should target release packaging, additional portability lanes such as `sm_86`
or `sm_89`, non-H100 Nsight Compute counters when `ncu` is available, or a
specific retained consumer with a written API and memory-ownership contract.

## GPU Library Baselines

Wolfgang should benchmark against GPU quantum libraries where they expose the
same workload semantics:

```text
cuQuantum cuStateVec: statevector Pauli expectation with the same statevector, Pauli strings, dtype, and transfer boundary
cuQuantum cuPauliProp: Pauli-expansion sort, deduplication, trace, or propagation-style workloads with exact mapping
CUDA-Q: end-to-end spin-operator observe workflows, including framework and kernel execution costs by design
Qiskit Aer GPU or Aer cuStateVec: framework-level circuit/statevector workflows when installed with GPU support
cuDensityMat and other GPU libraries: future candidates only after a workload-specific mapping is documented
```

GPU-library baselines are not substitutes for Wolfgang CPU scalar, optimized
CPU, and native CUDA kernel timings. Reports must label whether a competitor
timing is framework-level, transfer-inclusive primitive, or device-resident
primitive, and must emit explicit unavailable statuses when an optional GPU
library is absent or not GPU-enabled.

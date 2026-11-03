# Wolfgang Changelog

All user-visible Wolfgang changes are recorded here before each release.

## Unreleased

Next version: TBD

- CPU, oneTBB on small machines:
  - Auto dispatch runs simplify on oneTBB from two threads instead of four.
    Two threads beat scalar from 8,192 terms on every host measured: 1.09-1.47x
    on one core's two hyperthreads of an EPYC 9654, 1.11-1.33x on a Xeon
    8480+'s, 1.45-1.67x on two Xeon cores, and 1.7-3.4x on an M4 Pro.
  - oneTBB pairwise commutation runs the best SIMD kernel in each thread, where
    it used to run a scalar loop. On a four-vCPU AVX-512 machine, auto picked
    oneTBB at 0.23-0.26 times the speed of one AVX-512 thread; with three
    threads the new kernel beats AVX-512 by 1.2-1.7x on two-word operators.
  - Pairwise commutation uses oneTBB from two threads when no SIMD kernel covers
    the operator and from three when one does, and one-word operators on AVX2
    or AVX-512 need at least 2048 x 2048 entries, below which writing the
    output, not the arithmetic, limits them.
  - Single-threaded commutation kernels are faster: AVX-512 by 1.10-1.24x, AVX2
    by 1.21-1.40x, scalar by up to 1.6x.
  (`docs/benchmarks/reports/cpu_thread_floor_2026-11-03.md`)
- CUDA, several GPUs:
  - Device calls release the GIL, so one Python thread per GPU keeps several
    GPUs busy. Two threads on two H100s reach 1.88-1.98 times one thread's
    throughput, against 1.0 before. Threads on the same GPU take turns
    through a per-device lock; transfers overlap.
  - A kernel on a peer GPU can read buffers exported with `device_arrays()`,
    DLPack or the CUDA Array Interface. Before, such a read faulted with an
    illegal address. The first export from a device maps its memory pool into
    every peer that supports peer access, which took 26 ms on the H100 pair,
    and freeing an exported buffer now waits for those peers as well as its
    own device.
  - A process that only uses GPU 1 no longer creates a 516 MiB context on
    GPU 0. Wolfgang switches back to the caller's device only if that device
    already has a context
    (`docs/benchmarks/reports/cuda_multi_gpu_h100_2026-11-03.md`).
- Documentation: the README, the site home page, the performance guide and the
  benchmark evidence index show three figures generated from committed
  benchmark data by `scripts/render_readme_figures.py`: simplify time against
  operator size on each accelerator host beside that host's CPU, Wolfgang against Qiskit, OpenFermion and
  cuStateVec, and per-call GPU times across the November campaigns
  (`tests/test_readme_figures.py` fails when a figure is stale). Mermaid
  diagrams render on the documentation site again, and the user and
  engineering docs were edited for plain, specific prose.
- The release software bill of materials lists the oneTBB runtime bundled in
  CPU wheels (version, source commit, `Apache-2.0 AND BSD-3-Clause`) and the
  vendored DLPack header (version and SHA-256) alongside the Python
  dependencies, and the release workflow validates it against the CycloneDX
  schema (`scripts/sbom_bundled_components.py`).
- `wolfgang_quantum.set_num_threads(n)` and the `WOLFGANG_NUM_THREADS`
  environment variable cap the threads Wolfgang's oneTBB CPU kernels use, for
  the whole process; `get_num_threads()` reports the current count. Auto
  dispatch's oneTBB thread floors count the capped threads. Inside a 4-CPU quota on a
  26-vCPU host, a cap of 4 made simplify 1.25x faster on average and its
  90th-percentile time 2.6x shorter, at the cost of a slower median call
  (`docs/benchmarks/reports/cpu_threads_2026-11-03.md`).
- CUDA and HIP: `DevicePauliSum.to_host()` stages its copies through pinned
  buffers borrowed from a small process-wide pool, waiting for the GPU once
  per batch instead of once per buffer (HIP stages only transfers up to
  64 KiB and copies larger ones directly). Staged results are no longer
  zero-filled before the copy, and on Linux results of 4 MiB or more ask for
  transparent huge pages before their first touch. On an H100, `to_host` is
  1.9-3.2x faster from 1 term to 4M terms (1M terms at 24 qubits: 5.5 ms
  instead of 14.8 ms); on an MI300X, 1.8-1.9x for small operators and
  2.2-3.2x at 1M terms (24 qubits) and 4M terms, and unchanged in between
  (`docs/benchmarks/reports/host_transfers_2026-11-03.md`). `to_device()` is
  unchanged.
- HIP: simplify for operators the fused kernel does not take uses the
  pipeline shared with CUDA, built with hipCUB, so it no longer returns counts through blocking
  copies, and its scratch (with rocThrust's algorithm temporaries) comes from a
  per-thread cache of up to two blocks of at most 256 MiB, freed when the
  thread exits. On an MI300X, `simplify` on 4,096 terms takes 98 us instead of
  182 us, `matmul` on 128 x 128 terms 124 us instead of 202 us, and
  device-resident simplify is 1.6-2.7x faster from 50,000 to 1M terms
  (`docs/benchmarks/reports/hip_device_counts_mi300x_2026-11-03.md`). Pinned
  result buffers are freed when the thread that used them exits, as on CUDA.
- CUDA: simplify for operators the fused kernel does not take (more than
  1,024 terms, more than 512 at three or four words, or more than four words),
  and `matmul` through it, keeps the input maximum, drop rule, run count and
  survivor count on the device. It waits for the GPU once before sizing its
  output instead of eight times. On an H100, `simplify` on 4,096 terms takes
  108 us instead of 188 us and `matmul` on 128 x 128 terms 123 us instead of
  202 us; device-resident simplify is 1.08-1.74x faster from 4,096 to 1M terms
  at up to 128 qubits and 1.11-1.85x faster on wider operators
  (`docs/benchmarks/reports/cuda_device_counts_h100_2026-11-03.md`). Pinned
  result buffers are now freed when the thread that used them exits.
- CUDA: `DeviceCommutationMatrix` output and its count scratch come from the
  Wolfgang memory pool; an exported matrix synchronizes the device before its
  block is reused. On an H100, `commutes_with_device` on 128 x 128 terms takes
  10.6 us instead of 64.1 us when its output was the process's only
  `cudaMalloc` buffer, and about 1.14x less time when another one kept the
  driver's allocation mapped; `count_commuting` and `conflict_degrees` are 1.1x
  faster (`docs/benchmarks/reports/cuda_commutation_pool_h100_2026-11-03.md`).
  `device_arrays()` now finishes pending host-to-device copies before handing
  out pointers.
- Release CPU wheels bundle oneTBB 2022.3.0, built from a pinned commit, so
  `pip install` users get parallel simplify. Installed wheels simplify 1.5-3.4x
  faster on a 16-vCPU x86 VM and 1.5-5.6x on an Apple M4 Pro from 10,000 terms
  (`docs/benchmarks/reports/onetbb_wheels_2026-11-03.md`). The license
  expression is now `MIT AND Apache-2.0 AND BSD-3-Clause`, with oneTBB's and
  DLPack's notices as license files, and the wheel smoke test requires
  parallel simplify to match scalar bit for bit.
- oneTBB does not support `fork()`. A child process forked after a oneTBB
  kernel ran now uses the scalar or SIMD kernels (bit-identical results), and
  a forced `WOLFGANG_CPU_BACKEND=tbb` raises there; previously such a child
  could crash inside oneTBB on macOS. Auto dispatch of pairwise commutation to
  oneTBB now also needs at least four oneTBB threads.
- `PauliSum.from_cudaq()` and `PauliSum.to_cudaq()` convert to and from CUDA-Q
  `SpinOperator` (the `cudaq` extra, CUDA-Q 0.11 or later). Whole operators
  move through CUDA-Q's flat serialization, checked once per process, with a
  fallback to CUDA-Q's public term API. Expectation values agree with
  `cudaq.observe`. A new CI job tests CUDA-Q 0.11.0 and the latest release.
- CUDA: `DevicePauliSum.device_arrays()` returns zero-copy views of the x
  words, z words and coefficients, flagged read-only, that CuPy reads through
  DLPack or the CUDA Array Interface and PyTorch through DLPack, and
  `DevicePauliSum.from_device_arrays()` builds an operator from CUDA device
  arrays with a device-to-device copy. On an H100 at 1M terms, handing the
  buffers to CuPy takes 36 us instead of 8.8 ms through the host, and building
  from CuPy arrays 84 us instead of 21 ms
  (`docs/benchmarks/reports/device_arrays_h100_2026-11-03.md`). HIP and Metal
  raise for both. `DeviceArrayView` is exported for type annotations.
- CUDA and HIP: operators of up to 256 qubits and 1,024 terms simplify in one
  fused kernel. On an H100 small simplify is 3.0-7.8x faster (128 terms:
  143.5 to 18.4 us); on an MI300X it is 3.5-5.7x faster. Their coefficients
  now equal CPU simplify bit for bit
  (`docs/benchmarks/reports/small_simplify_2026-11-03.md`).
- Every backend rounds `atol + rtol * max_abs` as a separate multiply and add.
  CPU simplify on Apple Silicon (and other arm64 hosts) previously computed it
  as a fused multiply-add and could drop a term at the threshold that x86 kept.
- CUDA and HIP simplify follow the CPU numeric contract: non-finite
  coefficients raise `ValueError` and a duplicate sum that overflows raises
  `OverflowError`. Previously an infinite input was kept, and a GPU tree sum
  that met as `inf + -inf = NaN` was dropped silently. HIP also uses the CPU
  drop rule, so a finite coefficient whose magnitude exceeds the largest double
  no longer makes the threshold NaN and drop every term. Out-of-memory on any
  CUDA or HIP allocation raises `MemoryError`.
- HIP: small device simplify and matmul are 1.5-1.6x faster and single-term
  simplify 8x faster on an MI300X. rocThrust no longer synchronizes after
  every algorithm, scratch is not zero-filled, and single-term inputs skip
  sorting. HIP keeps `hipMalloc`, because a stream-ordered pool measured slower
  on ROCm (`docs/benchmarks/reports/hip_parity_mi300x_2026-11-03.md`).
- `benchmarks/bench_cuda_call_overhead.py --backend hip` measures HIP call
  overhead.
- Metal source builds: `DevicePauliSum.simplify` on one-word operators (up to
  64 qubits) of at least 12,288 terms sorts keys on the GPU and sums duplicates
  in float64 on the CPU over shared memory, with results identical to CPU
  simplify bit for bit. On an M4 Pro it is 1.3-4.1x faster than the previous
  host bridge and 1.05-4.0x faster than CPU simplify from 12,288 terms
  (`docs/benchmarks/reports/metal_hybrid_simplify_2026-11-03.md`). Metal
  uploads no longer recreate the default device on every call.
- Metal hybrid simplify from 2^20 terms reduces duplicate runs on all CPU
  cores, with results still identical to CPU simplify bit for bit, and splits
  the GPU sort across command buffers above 2^22 positions. Inputs of 4M, 16M
  and 64M terms ran without a GPU timeout. When Xcode's Metal compiler works
  at configure time, the kernels are embedded precompiled, so the first call
  in a process takes about 3.1 ms instead of 3.9 ms (49 ms instead of 154 ms
  with a cold shader cache); otherwise the embedded source compiles on the
  first call. `build_info()["metal_simplify_kernels"]` reports which
  (`docs/benchmarks/reports/metal_followups_2026-11-03.md`).
- oneTBB source builds: `simplify` runs a parallel sort-and-reduce kernel from
  8,192 terms on hosts with at least four oneTBB threads
  (`WOLFGANG_CPU_BACKEND=tbb` forces it), with results identical to scalar bit
  for bit. It is 1.6-5.7x faster than scalar from 10,000 terms on two x86 hosts
  (`docs/benchmarks/reports/cpu_parallel_simplify_2026-11-03.md`).
- Every CPU simplify path seeds each duplicate run with its first coefficient,
  so a lone `-0.0` component survives; scalar simplify previously kept or
  dropped the sign depending on which internal path it chose.
- CUDA: skip full device discovery on transfers and output creation, pool
  `DevicePauliSum` and scratch allocations in a Wolfgang-private stream-ordered
  pool (up to 1 GiB cached per device, trimmed and retried on out-of-memory),
  and drop redundant Thrust synchronizations and scratch fills. On H100,
  `to_device` of a small operator is 57x faster, `commutes_with_device` 14x,
  small `simplify` and `matmul` 1.5-1.7x, and simplify of 50,000-1,000,000
  terms 2.1-2.9x (device-resident). CUDA allocation failures now raise
  `MemoryError`.
- Restore the CUDA source build, broken since the shared workspace helper
  refactor, and pin the fully qualified aliases in the header contract test.
- CUDA simplify for operators wider than 128 qubits now reduces duplicates in
  parallel: 91-273x faster than the previous one-thread kernel on H100 at
  65,536 terms. Single-term simplify skips sort and reduction; one- and
  two-word paths use stable sorts with no measured cost.
- CUDA simplify no longer drops every term when a finite coefficient's
  magnitude overflows `DBL_MAX`; it now applies the CPU drop rule.
- Add `apply_statevector`, `variance_statevector` (of the normalized state),
  batched 2-D input for `expectation_statevector`, and `to_matrix` (dense, or
  SciPy CSR with `sparse=True` and the new `scipy` extra) with explicit size
  guards. Behavior change: a 2-D `psi` used to raise `ValueError`; it is now a
  batch, and the stubs return `complex | ndarray`.
- Add `estimate_counts(counts, basis)` for mean and standard error in any local
  measurement basis, and `wolfgang_quantum.measurement` with `qwc_basis`,
  `diagonalizing_clifford`, variance-weighted `allocate_shots` and
  `to_qiskit_circuit`.
- Add `PauliSum.from_arrays` / `to_arrays` for packed `uint64` mask and
  `complex128` coefficient exchange; `to_arrays` returns read-only zero-copy
  views. `from_labels` converts NumPy coefficient arrays in bulk.
- Add subtraction, exact negation, `len`, indexing and slicing, structural
  `==` (unhashable), `equiv`, `adjoint`, and fused `commutator` /
  `anticommutator` that skip term pairs which cancel (7-9x faster than
  composing two products in `docs/benchmarks/reports/host_api_2026-11-03.md`).
- Behavior change: `PauliSum == other` now compares structure (same terms,
  order and coefficients) instead of object identity, `PauliSum` is no longer
  hashable, and `bool(op)` follows `len(op)`, so an empty operator is falsy.
  Migration: use `op is other` for identity, `op.equiv(other)` for mathematical
  equality, `id(op)` as a dictionary key, and `op.num_terms > 0` where
  truthiness meant "not None".
- Exclude local worktrees and release staging from source distributions.
- Separate the native CMake library from Python, add standalone sanitizer
  contracts, bulk coefficient export and scoped CPU GIL release.
- Record loaded-extension/source fingerprints in benchmarks and keep requested
  revision labels separate from verified git state.
- Share core, quality, docs and all validation profiles with CI; add Python 3.13
  core coverage and reject unsupported forced streaming-grouping selectors.
- Preserve arbitrarily small nonzero coefficients when exporting to OpenFermion;
  duplicate export terms are combined before exact-zero removal.
- Harden CPU simplify at finite complex128 limits and normalize count weights
  without overflowing their total. Document the input-relative tolerance rule.
- Bound Metal temporary-object lifetimes on Python and worker threads.
- Make the Wave 1D promotion gate reject incomplete or invalid evidence and
  preserve CPU-only skips.
- Repair shipped-build GPU device discovery and expose immutable per-operation
  execution locations. Correct constructor/count typing and installation docs.
- Validate the agent source map and require a nonempty public-header scan.

## 0.2.3

Status: published as GitHub release `v0.2.3` with one source distribution, six
CPU wheels, and no TestPyPI or PyPI publication.

### Added

- Wolfgang `0.2.3` successor release evidence in `docs/release/0.2.3.md`, documenting the published GitHub-only patch path from the repaired `wolfgang_quantum.capabilities()` lineage while preserving the immutable `v0.2.2` tag as historical provenance.
- Active release/support surfaces now point at the published `v0.2.3` GitHub-only successor ledger while preserving the quarantined `v0.2.0` and `v0.2.1` tags/draft releases, the immutable `v0.2.2` checkpoint, and the published `0.1.0` evidence unchanged.
- Release supply-chain hardening with exact project-version tag binding,
  immutable GitHub Action pins, least-privilege workflow permissions,
  dependency auditing, SBOM generation, artifact attestations, and automated
  dependency updates.

- Apple Metal Campaign 8 simplify performance-relevance evidence with private
  timing decomposition, pipeline-cache boundary, dispatch-count metadata,
  Campaign 8 benchmark rows, and refreshed benchmark/report/README evidence.
  Public Metal `DevicePauliSum.simplify()` remains on the transfer-reference
  bridge unless a later design proves a broader retained device-resident
  implementation.
- Apple Metal Campaign 7 checked device-resident simplify primitive stack with
  a private benchmark-only one-word Metal sort, prefix-sum, reduce-by-key, and
  survivor-compaction path; a private validation hook; Campaign 7 benchmark
  rows; and refreshed benchmark/report/README evidence. The candidate remains
  benchmark-only and fixed-dyadic-coefficient limited because this Apple Metal
  toolchain rejects `double` arithmetic in kernels; accepted rows also require
  exact fixed32 accumulation and uint64 squared-magnitude tolerance comparison.
  Public Metal `DevicePauliSum.simplify()` stays on the transfer-reference
  bridge.
- Apple Metal Campaign 7 planning in
  `docs/plans/apple_metal_optimization_campaign7_plan.md` for checked
  device-resident simplify primitives while keeping public workspaces, public
  raw Metal exports, Metal statevector expectation, Metal matmul, Metal wheels,
  PyPI publication, Windows support, and older macOS compatibility out of
  scope.
- Apple Metal Campaign 6 device-resident simplify groundwork with a private
  `MetalWorkspace` model, `WorkspaceTimingMode`, Campaign 6 simplify benchmark
  cases, `metal_simplify_workspace_probe` status rows, Campaign 6 benchmark
  JSON, report, validation smoke, and README landscape evidence. The
  device-resident simplify candidate remains blocked until checked Metal
  sort/prefix/reduce primitives exist; the public Metal simplify behavior
  remains the Campaign 5 transfer-reference correctness bridge.
- Apple Metal Campaign 6 planning in
  `docs/plans/apple_metal_optimization_campaign6_plan.md` for
  source-build-only device-resident simplify groundwork while keeping public
  Metal workspaces, Metal statevector expectation, Metal matmul, Metal wheels,
  PyPI publication, Windows support, and older macOS compatibility out of
  scope.
- Apple Metal Campaign 5 source-build simplify behavior with
  `DevicePauliSum.simplify()` parity for empty, duplicate-heavy, cancellation,
  and multi-word operators; finite non-negative tolerance validation; Campaign
  5 benchmark JSON, report, validation smoke, and README landscape evidence.
  The retained Metal simplify path is recorded as
  `metal_simplify_transfer_reference` with the
  `device_to_host_cpu_simplify_host_to_device` boundary and is not a
  device-resident GPU duplicate-reduction speedup.
- Apple Metal Campaign 5 planning in
  `docs/plans/apple_metal_optimization_campaign5_plan.md` for
  source-build-only Metal `DevicePauliSum.simplify()` bring-up, including
  correctness, benchmark, validation, and documentation acceptance criteria
  while keeping PyPI publication, Windows support, older macOS support, Metal
  wheels, Metal statevector expectation, and Metal matmul out of scope.
- Apple Metal Campaign 4 benchmark evidence and documentation, including a
  benchmark-only parallel compact total-count selector, larger local Apple M4
  Pro benchmark cases, and refreshed broad accelerator landscape assets while
  keeping public Metal API support unchanged.

## 0.1.0

Status: final PyPI release under validation. `v0.1.0` tag-ref
artifacts have been built and validated, TestPyPI trusted publishing and clean
install smoke have passed, and PyPI publication is blocked by PyPI
trusted-publisher configuration for the observed `pypi` environment claims.
PyPI publication is not claimed.

### Added

- Final `0.1.0` CPU wheelhouse readiness foundation with cibuildwheel
  configuration, a manual release-wheelhouse workflow, installed-wheel smoke,
  checksum manifest generation, complete wheelhouse checks, checksum-free
  publish upload filtering, and tag-ref-gated TestPyPI/PyPI trusted-publishing
  gates while keeping accelerator and Windows wheel claims unavailable.
- Hosted release-wheelhouse dry-run evidence for the complete current CPU
  artifact shape: one source distribution, six CPU wheels, checksum manifest
  validation, merged-artifact metadata checks, and checksum-free package-index
  upload preparation with publication disabled.
- Exact-tag `v0.1.0` wheelhouse evidence for one source distribution, six CPU
  wheels, checksum validation, and checksum-free publish upload preparation; the
  previous TestPyPI blockers are cleared, TestPyPI upload and clean install
  smoke passed, and PyPI upload is blocked by missing matching PyPI
  trusted-publisher configuration.

## 0.1.0rc2

Status: published as GitHub prerelease `v0.1.0rc2` with source distribution,
macOS arm64 CPU wheel, and external checksum manifest artifacts.

### Added

- Release-readiness hardening for the next public checkpoint with a checked
  support matrix, release-readiness checker, and source-of-truth routing for
  CPU wheels, CUDA source-build support, ROCm/HIP source-build support, Apple
  Metal source-build evidence, unavailable accelerator wheels, unsupported
  Windows wheels, and package-index publication status.

### Support Boundaries

- CPU wheels remain the release-candidate artifact target.
- CUDA support remains source-build-only unless a later CUDA packaging plan adds
  wheel evidence.
- ROCm/HIP support remains source-build-only unless a later ROCm packaging plan
  adds wheel evidence.
- Apple Metal support remains source-build-only unless a later Apple
  accelerator packaging plan adds macOS arm64 wheel evidence.
- CUDA wheels remain unavailable.
- ROCm/HIP wheels remain unavailable.
- Metal wheels remain unavailable.
- Combined accelerator wheels remain unavailable.
- Windows wheels remain unavailable.
- PyPI or another package-index publication is not claimed.

## 0.1.0rc1

Status: published as GitHub prerelease `v0.1.0rc1` with source distribution,
macOS arm64 CPU wheel, and external checksum manifest artifacts.

### Added

- CPU package scaffold with C++20/nanobind extension and `wolfgang_quantum.PauliSum`.
- Dense-label and sparse-list construction/export with documented endianness.
- Optional Qiskit and OpenFermion adapters.
- Scalar simplify, arithmetic, multiplication, commutation, grouping, and
  expectation kernels.
- Runtime CPU backend metadata with portable scalar fallback and optional
  oneTBB/SIMD dispatch where compiled and runtime-available.
- CUDA source-build backend with explicit transfers, simplify, expectation,
  commutation, compact consumers, DLPack export for device commutation matrices,
  and matmul kernels.
- ROCm/HIP source-build backend with MI300X-evidenced transfers, commutation,
  compact consumers, simplify, expectation, and matmul kernels.
- Backend-neutral accelerator selectors for CPU-only, CUDA-target, and
  HIP-target builds.
- Apple Metal source-build backend with transfer, pairwise commutation,
  retained device-matrix, compact-consumer evidence, generic two-dimensional
  commutation dispatch, retained one-word specialization, and benchmarked
  baseline rows for common packed inputs including a benchmark-only two-word
  specialization candidate, offline `.metallib` loading, private-blit
  host-output staging, and GPU compact-reduction evidence.
- Release-candidate foundation artifacts: release evidence ledger and CPU wheel
  build/install/import validation.

### Support Boundaries

- CPU wheels are the first release-candidate artifact target.
- CUDA support remains source-build-only unless a later CUDA packaging plan adds
  wheel evidence.
- ROCm/HIP support remains source-build-only unless a later ROCm packaging plan
  adds wheel evidence.
- CUDA wheels remain unavailable.
- ROCm/HIP wheels remain unavailable.
- Combined accelerator wheels remain unavailable.
- Apple Metal support remains source-build-only unless a later Apple
  accelerator packaging plan adds macOS arm64 wheel evidence.

# Wolfgang Phase Quality Gates

This document defines the quality bar for every Wolfgang phase. It complements the roadmap and implementation plan by defining what evidence must exist before a phase can be called complete.

## Global Gates

Every phase must satisfy these gates:

```text
1. Scope is narrow and matches docs/roadmap.md.
2. Public behavior matches docs/architecture/semantic_contracts.md.
3. CUDA-relevant layout choices remain compatible with docs/architecture/cuda_backend.md.
4. CPU, CUDA, compiler, dispatch, and hardware support choices follow docs/architecture/hardware_targets_and_testing.md.
5. New behavior has tests that fail before implementation when practical.
6. Validation is run on the merged result, not only on a feature branch.
7. Documentation is updated in the same slice as user-visible or contract changes.
8. Benchmarks are added or updated whenever performance claims, algorithm choices, or backend choices are introduced.
9. Commits are small, reviewable, and descriptive.
10. Agent-facing guidance in AGENTS.md stays short, current, and linked to deeper source-of-truth docs.
11. Review requirements in docs/quality/code_review.md are satisfied.
12. Repeatedly relevant rules are promoted into docs, tests, validation scripts, or CI.
13. Code follows docs/quality/code_standards.md.
14. User-facing and API documentation follows docs/quality/documentation_standards.md.
15. Public API changes follow docs/architecture/api_stability.md.
16. Native-code, dependency, and release-integrity choices follow docs/quality/security_and_supply_chain.md.
17. Packaging and release work follows docs/quality/release_and_packaging.md.
```

No phase is complete with known failing required tests, unresolved blocking review findings, untriaged benchmark regressions, undocumented public API changes, or unexplained skipped tests.

## Evidence Record

Each phase closeout must record:

```text
branch name
commit list
validation commands and outcomes
review scope, findings, resolutions, and deferrals
benchmark commands and outcomes when applicable
known limitations
follow-up items that are explicitly out of scope
```

For now this can be captured in the final agent report. Once CI and release docs exist, release-candidate evidence should also live in a checked-in artifact or GitHub release notes.

## Phase 0: Planning And Architecture Lock

Quality bar:

```text
source-of-truth docs exist
known semantic risks have explicit contracts
CUDA is planned as a required backend without blocking CPU-first correctness
```

Required evidence:

```text
docs/architecture/semantic_contracts.md exists
docs/architecture/cuda_backend.md exists
docs/architecture/hardware_targets_and_testing.md exists
docs/roadmap.md exists
docs/quality/phase_quality_gates.md exists
docs/architecture/testing_and_ci.md exists
docs/architecture/adapter_contracts.md exists
docs/benchmarks/protocol.md exists
docs/quality/agent_harness.md exists
docs/quality/code_review.md exists
docs/quality/code_standards.md exists
docs/quality/documentation_standards.md exists
docs/architecture/api_stability.md exists
docs/quality/security_and_supply_chain.md exists
docs/quality/release_and_packaging.md exists
CONTRIBUTING.md exists
AGENTS.md exists and links to source-of-truth docs
```

## Phase 1: CPU Package Scaffold

Quality bar:

```text
the package installs cleanly
the extension imports cleanly
CPU-only builds do not require CUDA
there is one repo-local validation command
```

Required evidence:

```text
pyproject.toml defines build, test, and optional dependency groups
CMake config has WOLFGANG_ENABLE_CUDA=OFF by default
CMake config keeps WOLFGANG_ENABLE_NATIVE=OFF for release-wheel defaults
portable scalar CPU path is present and can be identified in validation output
python -m pip install -e ".[test]" with
`--config-settings=cmake.define.WOLFGANG_ENABLE_INTERNAL_BINDINGS=ON` succeeds in a clean environment
python -c "import wolfgang_quantum; print(wolfgang_quantum.__version__)" succeeds
pytest succeeds
scripts/validate.py or equivalent repo-local validation command exists
CI runs the validation command for CPU-only builds
AGENTS.md links resolve to existing repo paths
docs/source-of-truth existence check is part of the validation command
initial repo contract checks are documented in docs/quality/agent_harness.md
initial review-policy checks are documented in docs/quality/agent_harness.md
initial code/documentation standard checks are documented in the validation command
initial API/security/release standard checks are documented in the validation command
initial hardware target source-doc check is part of the validation command
```

Performance gate:

```text
no performance claims are allowed in Phase 1
build time and import time should remain ordinary for a tiny extension
```

Documentation gate:

```text
public scaffold APIs have accurate docstrings or documented intentional deferral
README and AGENTS.md point to code and documentation standards
README, AGENTS.md, and CONTRIBUTING.md point to API, security, and release standards
no unsupported performance or maturity claims are added
```

## Phase 2: Packed Representation And I/O

Quality bar:

```text
endianness and memory invariants are locked before adapters or algorithms build on them
constructors preserve input order before simplify or sort
empty and identity operators are explicit
```

Required evidence:

```text
unit tests cover dense label parsing and export
unit tests cover sparse list parsing and export
tests cover PauliSum.empty(num_qubits)
tests cover invalid labels, duplicate sparse indices, out-of-range indices, coefficient length mismatch, and empty from_labels rejection
tests cover final-word high-bit zeroing for non-multiple-of-64 qubit counts
tests cover multi-word operators at 64, 65, 128, and 129 qubits
```

Performance gate:

```text
construction should avoid per-qubit Python callbacks after inputs reach C++
no benchmark speedup claims before benchmark protocol is in place
```

## Phase 3: Qiskit Adapter

Quality bar:

```text
Qiskit integration is optional, deterministic, phase-correct, and never dense-matrix based for conversion
```

Required evidence:

```text
tests skip cleanly when Qiskit is absent
from_qiskit and to_qiskit round-trip simple, empty, identity, duplicate, and phased operators
small random n <= 8 operators match dense-matrix semantics
adapter behavior follows docs/architecture/adapter_contracts.md
```

Performance gate:

```text
initial adapter may use labels for robustness
any optimized x/z extraction path must keep the label path as a tested semantic oracle during the same phase
```

## Phase 4: Simplify And Canonical Ordering

Quality bar:

```text
simplify is correct, deterministic, idempotent, and benchmarked before optimization claims
```

Required evidence:

```text
tests cover duplicate reduction
tests cover tolerance formula exactly
tests cover negative tolerance errors
tests cover all-zero output returning PauliSum.empty(self.num_qubits)
tests cover canonical order for one-word and multi-word operators
property tests cover simplify idempotence
Qiskit comparison tests pass for small random operators when Qiskit is installed
bench_simplify.py or equivalent benchmark exists
```

Performance gate:

```text
benchmark report compares Wolfgang scalar path against Qiskit or a Python baseline
benchmarks include low-duplicate and high-duplicate datasets
Apple Silicon scalar CPU timing is captured when local hardware is available
parallel or SIMD work is not accepted without scalar baseline numbers
```

## Phase 5: Arithmetic And Multiplication

Quality bar:

```text
arithmetic preserves documented order and multiplication phases are exact
```

Required evidence:

```text
tests cover addition without implicit simplify
tests cover scalar multiplication including zero scalar
tests cover all six single-qubit multiplication phase fixtures
tests cover max_intermediate_terms guardrail before allocation
tests cover nested-loop product order when simplify=False
small random associativity tests pass after simplify
small random dense-matrix comparisons pass
```

Performance gate:

```text
benchmarks cover single-term multiplication and small cross-product multiplication
guardrail tests include overflow-safe dimension checks
```

## Phase 6: Commutation And Grouping

Quality bar:

```text
commutation APIs are memory-safe and grouping is deterministic, valid, and explicitly heuristic
```

Required evidence:

```text
tests cover scalar, vector, and matrix commutes_with return shapes
tests cover max_commutation_matrix_entries before dense allocation
tests cover QWC and full commutation formulas
property tests cover commutation symmetry
tests prove every returned QWC group is internally QWC-compatible
tests prove every returned full group is internally commuting
tests prove grouping order is deterministic
```

Performance gate:

```text
bench_grouping.py exists
benchmarks cover pairwise commutation, QWC grouping, and full grouping
large-input tests prove guardrails fail before unsafe allocation
```

## Phase 7: OpenFermion Adapter

Quality bar:

```text
OpenFermion integration is optional, deterministic after simplify, and preserves identity and inferred-width semantics
```

Required evidence:

```text
tests skip cleanly when OpenFermion is absent
from_openfermion and to_openfermion round-trip simple, identity, duplicate, and sparse multi-qubit operators
num_qubits inference follows docs/architecture/adapter_contracts.md
coefficients match after simplify
bench_openfermion_conversion.py or equivalent benchmark exists
```

Performance gate:

```text
benchmarks compare conversion/manipulation against OpenFermion object-heavy paths for large sparse operators
```

## Phase 8: CPU Expectation Kernels

Quality bar:

```text
expectation results are phase-correct, dtype-aware, and validated against dense or direct oracles
```

Required evidence:

```text
statevector expectation matches dense matrix for n <= 8
tests cover complex64 and complex128 statevectors
tests cover invalid shape, dtype, and length errors
Z-count expectation matches direct Python computation
Z-count tests cover bitstring endianness
non-diagonal Z-count inputs raise ValueError
```

Performance gate:

```text
benchmarks cover few-terms-large-statevector and many-terms-small-statevector regimes
parallel strategy changes include benchmark evidence
```

## Phase 9: CPU Optimization

Quality bar:

```text
optimization is measurement-driven and cannot change public semantics
```

Required evidence:

```text
scalar baseline remains available
optimized paths have CPU feature detection tests where practical
AVX2 and AVX-512 paths are gated by runtime feature checks
explicit ARM SIMD paths, if introduced, are gated by runtime feature checks
forced scalar and forced optimized CPU paths are tested where compiled and supported
release-wheel defaults keep native CPU tuning disabled
oneTBB paths have deterministic-output tests
all optimized paths pass the same semantic test suite as scalar paths
auto dispatch is enabled only for named kernels with forced-backend correctness and benchmark evidence
```

Performance gate:

```text
benchmark report compares scalar, oneTBB, and SIMD paths where available
benchmark report records unavailable optimized paths with explicit reasons
benchmark report records active CPU backend, CPU feature set, compiler flags, thread settings, and oneTBB status
benchmark report records optimized kernel coverage and effective auto backend hints where auto dispatch is operation-specific
Apple Silicon and x86_64 optimization results are reported separately when both are available
regressions are either fixed or documented with an explicit tradeoff
```

## Phase 10: CUDA Backend Foundation

Quality bar:

```text
CUDA support is source-buildable, optional at build time, and impossible to accidentally require for CPU users
```

Required evidence:

```text
WOLFGANG_ENABLE_CUDA=OFF builds without CUDA installed
WOLFGANG_ENABLE_CUDA=ON builds on a CUDA-capable runner or machine
CUDA validation output records toolkit, host compiler, requested architectures, driver, device, and compute capability where available
PauliSum.to_device and DevicePauliSum.to_host preserve labels and coefficients
CUDA absence skip messages distinguish build-time absence from runtime device absence
public CPU-only headers do not include CUDA headers
```

Performance gate:

```text
no CUDA speedup claims are allowed in the foundation phase
transfer timing hooks exist before CUDA kernels claim performance
CUDA architecture target selection follows docs/architecture/hardware_targets_and_testing.md
```

## Phase 11: CUDA Kernels

Quality bar:

```text
CUDA kernels are CPU-equivalent, transfer-aware, and benchmark-justified
```

Required evidence:

```text
CUDA simplify matches CPU simplify canonical output
CUDA expectation matches CPU expectation within dtype-specific tolerance
CUDA commutation enforces dense-output guardrails
CUDA multiplication enforces max_intermediate_terms
CUDA unsupported dtype, layout, and device mismatch errors are tested
CPU/GPU equivalence tests cover all CUDA operations
compute-sanitizer or equivalent CUDA memory checking is run for release-relevant kernels when available
```

Performance gate:

```text
benchmark reports include CPU scalar, CPU optimized, CUDA transfer-inclusive, and CUDA device-resident timings
CUDA benchmark reports record toolkit, driver, compiled architectures, device model, compute capability, and transfer topology when relevant
CUDA benchmark reports time every available optimized CPU selector covered by the operation and report unavailable selectors with reasons
CUDA benchmark reports include preallocated or reused-output timings when public APIs expose those paths
CUDA benchmark reports classify and include GPU-library competitors when an accelerator library maps cleanly to the workload
README performance plots are generated from checked-in benchmark reports and show CPU scalar, every captured optimized CPU selector, CUDA transfer-inclusive, and CUDA device-resident paths
reports identify transfer-bound, CPU-faster, and CUDA-faster regimes
overnight CUDA/CPU hillclimb work produces a comprehensive checked-in optimization and profiling report with deep technical explanation, measured plots, architecture/kernel/hardware visuals, rejected-experiment evidence, and open-source competitor comparisons as defined in docs/benchmarks/protocol.md
post-Phase 11 deep CUDA hillclimbing follows docs/plans/cuda_deep_optimization_plan.md and records any deviation in the final report
the first CUDA kernel is selected by benchmark evidence from earlier phases
```

## Post-CUDA Accelerator Candidates

Quality bar:

```text
non-CUDA GPU work is sequenced after CUDA and receives its own architecture, testing, packaging, and benchmark gates
```

Required evidence before implementation:

```text
ROCm/HIP backend contract is documented before HIP implementation
Metal backend contract is documented before Metal implementation
MPS and MPSGraph roles are classified before any Apple accelerator implementation
promoted backend build flags, runtime availability checks, transfer semantics, and Python interop are specified
CPU/backend equivalence strategy is defined for each operation
CI or local hardware validation path is identified
```

Required evidence before a ROCm/HIP support claim:

```text
docs/architecture/rocm_backend.md exists and is linked from hardware-target docs
MI300X host inventory captures ROCm, HIP compiler, GPU model, gfx target, driver/runtime versions, CPU, memory, power, clocks, and topology where available
HIP source build succeeds with WOLFGANG_ENABLE_HIP=ON and WOLFGANG_HIP_ARCHITECTURES=gfx942
WOLFGANG_ENABLE_CUDA=ON together with WOLFGANG_ENABLE_HIP=ON fails at configure time with a clear error
CPU-only validation still passes with WOLFGANG_ENABLE_HIP=OFF
_hip_status(), _accelerator_status(), and _build_info() report HIP metadata without requiring ROCm at import time
host/device transfer tests pass for non-empty and empty operators
CPU/HIP equivalence tests pass for every implemented HIP operation
rocprof trace or counter evidence is captured for performance claims, or a blocked-profiler diagnosis records the exact command and provider/tooling limitation
benchmark reports include CPU scalar, available optimized CPU selectors, transfer-inclusive HIP timing, device-resident HIP timing where applicable, and unavailable external-baseline reasons
README, roadmap, and release/support wording identify ROCm/HIP evidence as source-build evidence, not wheel support
```

Required evidence before a post-bring-up ROCm/HIP optimization claim:

```text
docs/plans/rocm_next_waves_plan.md identifies the active wave and out-of-scope later waves
the active ROCm campaign plan identifies public API, lifetime, synchronization, and ownership boundaries before implementation
HIP source build validation passes on the named AMD GPU architecture used for the claim
CPU-only validation passes with HIP disabled after the optimization lands
CPU/HIP equivalence tests cover every retained HIP operation and compact consumer
benchmarks compare host-output, device-resident, compact-consumer, CPU scalar, optimized CPU, and latest relevant CUDA rows when available
rocprof evidence separates kernel, HIP API, transfer, allocation, compact-consumer, and host-materialization costs where tooling permits
README performance visuals remain broad cross-path views when they are updated
release wording continues to distinguish performance-tested source builds from release-supported wheels
```

Performance gate:

```text
benchmarks compare scalar CPU, optimized CPU, CUDA, and the candidate backend on the same datasets when all are available
transfer-inclusive and device-resident timings are reported separately
backend-specific performance claims record toolkit, runtime, device model, architecture, and transfer topology where available
```

## Release Candidate Gate

A release candidate requires:

```text
all phase-required tests pass
all public docs match current behavior
CI is green for required CPU jobs
CUDA source build is validated on at least one CUDA environment
benchmark report is refreshed for release-relevant workloads
known limitations are documented in README or release notes
```

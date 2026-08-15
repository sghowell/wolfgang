# Wolfgang Testing And CI Architecture

This document defines how Wolfgang maintains correctness while moving from scaffold to optimized CPU and CUDA kernels.

## Principles

```text
semantic contracts drive tests
small exact oracles precede large performance cases
optional integrations skip cleanly when absent
benchmarks support performance claims but do not replace correctness tests
one repo-local validation command is the developer and CI entrypoint
```

## Validation Entrypoint

Phase 1 should introduce a repo-local validation command, preferably:

```bash
python scripts/validate.py
```

The command should run the checks available at that phase. It should grow monotonically:

```text
Phase 1: import smoke test and pytest
Phase 2: semantic unit tests
Phase 3: optional Qiskit tests when installed
Phase 4: simplify tests and benchmark import smoke
Phase 5: arithmetic and multiplication tests
Phase 6: commutation and grouping tests
Phase 7: optional OpenFermion tests when installed
Phase 8: expectation tests and benchmark smoke
Phase 9: CPU dispatch tests and optimized-path availability reporting
Phase 10: CUDA availability, architecture target, and transfer tests when enabled
Phase 11: CUDA equivalence tests and benchmark smoke
```

If a check is skipped, the validation output must say why.

Phase 1 should also introduce the first agent-harness checks from `docs/quality/agent_harness.md`, including source-doc existence checks and `AGENTS.md` link checks.

Phase 1 should introduce the first code/documentation standard checks from `docs/quality/code_standards.md` and `docs/quality/documentation_standards.md`. These can start as lightweight source-doc existence, stale-marker, and link checks, then grow into formatter, lint, warning, docstring, and docs-build gates as tooling lands.

Phase 1 should also introduce the first API, security, and release standard checks from `docs/architecture/api_stability.md`, `docs/quality/security_and_supply_chain.md`, and `docs/quality/release_and_packaging.md`. These can start as source-doc existence and link checks, then grow into sanitizer, package-build, dependency, and release-evidence gates as implementation matures.

## Test Taxonomy

Wolfgang uses these test groups:

```text
smoke: installation, import, version, minimal PauliSum
semantic: endianness, representation, ordering, empty, identity, dtype, tolerance
oracle: dense-matrix comparisons for small n
property: generated random operators and invariants
adapter: Qiskit and OpenFermion optional integration tests
guardrail: allocation and blowup protection
performance-smoke: benchmarks import and execute tiny datasets
cuda: CUDA build, transfer, and CPU/GPU equivalence tests
```

Test files should be named by behavior, not by implementation detail.

## Reference Oracles

Dense-matrix oracle tests are allowed only for small qubit counts:

```text
n <= 8 for routine CI
n <= 10 only for explicitly marked slower local tests
```

Oracle sources:

```text
direct NumPy dense matrices for core semantics
Qiskit SparsePauliOp when Qiskit is installed
OpenFermion QubitOperator when OpenFermion is installed
direct Python computation for Z-count expectation
CPU scalar Wolfgang path as CUDA equivalence oracle after CPU correctness is locked
```

Random generators must use deterministic seeds. Property-test failures must print enough data to reproduce the failing operator.

## Required Phase Test Coverage

Phase coverage is defined in `docs/quality/phase_quality_gates.md`. The short version:

```text
Phase 1: install/import/pytest
Phase 2: representation, parsing, export, empty, ordering, invalid input
Phase 3: Qiskit round-trip, phase folding, dense oracle
Phase 4: simplify, canonical order, tolerance, idempotence, benchmark smoke
Phase 5: arithmetic, multiplication phases, guardrails, associativity
Phase 6: commutation formulas, grouping validity, deterministic order, matrix guardrails
Phase 7: OpenFermion round-trip, identity, num_qubits inference
Phase 8: statevector and Z-count expectation plus benchmark smoke
Phase 9: CPU dispatch surface, forced scalar checks, unavailable optimized-path checks
Phase 10: CUDA build and transfer equivalence
Phase 11: CUDA kernel equivalence and benchmark smoke
```

CPU and CUDA target-specific tests follow the ladders in `docs/architecture/hardware_targets_and_testing.md`.

Agent-harness checks are part of the required validation surface. When a harness rule becomes mechanically checkable, add it to `scripts/validate.py` or CI instead of leaving it as prose.

Code-standard and documentation-standard checks are also part of the required validation surface. Do not introduce formatters, linters, doc generators, or warning policies without adding them to the validation entrypoint.

API-stability, security, and packaging checks are part of the required validation surface when relevant. Do not introduce public API, dependency, native-code safety, or packaging changes without updating validation where the rule can be checked mechanically.

## Optional Dependency Policy

Optional integrations must not be imported at package import time.

When Qiskit or OpenFermion is missing:

```text
core tests still run
adapter tests skip with an explicit reason
from_qiskit, to_qiskit, from_openfermion, and to_openfermion raise ImportError with an installation hint
```

CI should include:

```text
core job without optional dependencies
adapter jobs with each landed optional integration installed
```

Once lower bounds are validated, optional dependency versions should be declared in `pyproject.toml`.

## CUDA Test Policy

CUDA tests are split by availability:

```text
CUDA build tests: require FASTPAULI_ENABLE_CUDA=ON
CUDA runtime tests: require a visible CUDA device
CUDA interop tests: require a supported __cuda_array_interface__ provider
```

Skip messages must distinguish:

```text
Wolfgang was built without CUDA
CUDA runtime library is unavailable
no CUDA device is available
required Python device-array package is unavailable
```

CPU-only CI must prove that CUDA headers and toolkit are not required.

CuPy CUDA Array Interface tests are mandatory in H100 validation when
`cupy-cuda12x` is installed. Public CI may skip them on CPU-only runners, but
the skipped status must explicitly say that the required Python device-array
package or CUDA runtime is unavailable.

## CPU Target Test Policy

CPU tests are split by path and availability:

```text
scalar tests: required on every CPU job
oneTBB tests: required when oneTBB is enabled or discovered
SIMD dispatch tests: required when explicit SIMD paths exist and hardware supports them
forced-path tests: required for scalar and each supported optimized path
sanitizer tests: required on at least one CPU job once native code has meaningful behavior
```

Skip messages must distinguish:

```text
optimized path was not compiled
hardware feature is unavailable
runtime backend was forced to a different path
optional oneTBB dependency is unavailable
sanitizer is unsupported on this platform
```

CPU jobs must prove that release-wheel defaults do not require native CPU flags or import-time CPU feature checks.
The release artifact lane must additionally build a source distribution and CPU
wheel, install the produced wheel into a clean virtual environment, and assert
CPU-only non-native build metadata with a scalar fallback.

## CI Jobs

Phase 1 should add CPU CI. Required CPU jobs:

```text
Linux CPU, Python minimum supported version
Linux CPU, Python latest supported version
macOS CPU, Python latest supported version
portable scalar build with FASTPAULI_ENABLE_NATIVE=OFF
source distribution and CPU wheel build/import smoke once release-candidate foundation lands
```

After optional adapters land, add:

```text
adapter test jobs with each landed optional integration installed
```

After CUDA foundation lands, add at least one CUDA validation path:

```text
CUDA source build
CUDA transfer tests
CUDA equivalence tests for implemented kernels
```

CI should run the same repo-local validation command used by developers, plus
job-specific dependency installation. Release-artifact CI should run
`scripts/validate_release_artifacts.py` on Linux x86_64 and macOS arm64 for the
first CPU release-candidate lane.

## Quality Controls

The validation command should fail on:

```text
test failures
unexpected skips in required jobs
import-time optional dependency failures
known stale generated files after a generation step exists
format or lint failures after those tools are introduced
```

Do not add a linter or formatter without also documenting its command in the validation entrypoint.

## Benchmark CI Policy

Full benchmarks are not required on every CI run. CI should run benchmark smoke tests that prove benchmark modules import and execute tiny datasets.

Performance claims require local or dedicated-run benchmark output following `docs/benchmarks/protocol.md`.

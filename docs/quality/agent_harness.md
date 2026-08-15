# Wolfgang Agent Harness

Wolfgang is intended to be built by Codex agents. The harness is the set of repository-local docs, checks, fixtures, tests, benchmarks, and workflows that make agent execution reliable.

This document adapts harness-engineering principles to a C++/CUDA numerical Python package.

## Principles

```text
humans steer, agents execute
the repository is the system of record
AGENTS.md is a map, not an encyclopedia
agent-legible checks beat prose-only expectations
mechanical boundaries beat repeated reminders
performance claims require reproducible evidence
cleanup and drift control are continuous work
```

## Repository Knowledge Model

Agents should be able to discover the project state without chat history.

Knowledge layers:

```text
AGENTS.md: short operating map
README.md: public project entrypoint
docs/roadmap.md: phase order and release gates
docs/plans/fastpauli_cpp_cuda_implementation_plan.md: implementation plan
docs/architecture/: semantic, CUDA, hardware target, testing, and adapter contracts
docs/quality/: phase gates, review policy, and harness rules
docs/benchmarks/: benchmark protocol and future reports
```

Do not move core operating rules into long chat prompts. If a rule matters for future work, put it in the repository.

## Agent-Legible Feedback Loops

Wolfgang needs domain-specific feedback loops instead of UI/browser harnesses.

Correctness feedback:

```text
deterministic unit tests
small dense-matrix reference oracles
Qiskit and OpenFermion optional adapter oracles
property tests with reproducible failing examples
CPU scalar path as the oracle for optimized CPU and CUDA paths after it is locked
```

Performance feedback:

```text
deterministic benchmark dataset generators
benchmark smoke tests in CI
saved benchmark commands and reports for performance claims
CPU backend, compiler, thread, and hardware metadata
transfer-inclusive and device-resident CUDA timings
CUDA toolkit, driver, compiled architecture, and device metadata
```

Architecture feedback:

```text
repo shape checks
docs link/source checks
CPU-only build checks that run without CUDA installed
portable scalar CPU build checks with native tuning disabled
CPU dispatch checks that can force scalar and optimized paths
checks that public CPU headers do not include CUDA headers
checks that optional integrations are not imported at package import time
```

Review feedback:

```text
independent agent review before implementation merge
focused reviewer roles for high-risk native, CUDA, performance, security, or release slices
findings ordered by severity with file and line references when available
P0/P1 findings resolved before merge unless the human owner explicitly accepts the risk
review evidence recorded in closeout
```

## Phase 1 Harness Requirements

Phase 1 must establish the first executable harness. It should add:

```text
scripts/validate.py
.github/workflows/ci.yml
tests/ for import and minimal PauliSum scaffold
repo contract checks for docs links and source layout
CPU-only build validation with WOLFGANG_ENABLE_CUDA=OFF
portable scalar build validation with WOLFGANG_ENABLE_NATIVE=OFF
review-policy existence and closeout checklist validation
```

The validation command should be:

```bash
python scripts/validate.py
```

The script must print each check it runs and fail on the first failing required check. It should be easy for agents and CI to use without extra context.

Initial validation checks:

```text
python -m pytest
python -c "import wolfgang_quantum"
docs/source-of-truth files exist
AGENTS.md links resolve to existing repo paths
CPU-only CMake configuration does not require CUDA headers or toolkit
hardware target source docs exist
review policy source docs exist
```

CI should run `python scripts/validate.py` on at least Linux and macOS CPU jobs once the package scaffold exists.

## Mechanical Checks To Add Over Time

Phase 2:

```text
representation and endianness fixtures
multi-word layout fixtures
final-word high-bit zeroing checks
review evidence template or checklist
```

Phase 3 and Phase 7:

```text
optional adapter import checks
missing dependency ImportError checks
adapter round-trip oracle tests
```

Phase 4:

```text
simplify benchmark smoke
canonical ordering fixtures
tolerance formula fixtures
```

Phase 5 and Phase 6:

```text
multiplication phase fixtures
allocation guardrail checks
grouping deterministic-order checks
```

Phase 8:

```text
statevector oracle checks
Z-count direct-computation checks
dtype and shape rejection checks
```

Phase 9:

```text
scalar versus optimized CPU equivalence checks
runtime CPU feature detection checks where practical
forced scalar and optimized-path dispatch checks
CPU benchmark metadata checks
```

Phase 10 and Phase 11:

```text
CPU-only build without CUDA installed
CUDA build when enabled
CUDA architecture target validation
CUDA transfer equivalence checks
CUDA kernel equivalence checks
CUDA benchmark smoke with explicit availability skips
CUDA benchmark metadata checks
```

## Drift Control

Agents should reduce future ambiguity whenever they touch the repo.

When implementation exposes a gap:

```text
1. add or update the relevant source-of-truth doc
2. add a test, benchmark, or validation check when practical
3. wire the check into scripts/validate.py or CI when it is required
4. keep the change scoped and commit it with the implementation slice
```

Examples of drift that should be fixed immediately:

```text
docs describe behavior not implemented by tests
benchmarks use undocumented datasets
CPU or CUDA support claims lack target evidence
CI and local validation run different required checks
CUDA and CPU semantics diverge
optional dependencies become import-time requirements
performance claims lack baseline, command, environment, or revision
```

## Merge Philosophy

Wolfgang should favor small, reviewable slices that can be validated end to end.

Required merge loop:

```text
branch
implement narrow slice
validate locally
commit in sensible chunks
complete independent agent review required by docs/quality/code_review.md
resolve P0/P1 findings
commit review fixes when needed
merge locally to main
validate merged main
push
confirm CI when CI exists
delete merged feature branch
```

Do not use large speculative branches to cover multiple phases at once. Phase gates exist to keep agent throughput from turning into unreviewable drift.

## What This Harness Does Not Require

Wolfgang does not need web-app harness machinery such as browser automation, DOM snapshots, or UI video capture unless a future UI is added.

Wolfgang does need equivalent observability for its domain:

```text
test output
benchmark output
compiler/build output
CUDA availability and device metadata
clear validation summaries
```

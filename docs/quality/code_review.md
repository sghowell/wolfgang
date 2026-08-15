# Wolfgang Agent-Driven Code Review

This document defines the review stage for Codex-driven Wolfgang development. Review is a required quality control, not a substitute for tests, benchmarks, sanitizers, static analysis, or CI.

The goal is adversarial engineering review: catch contract drift, unsafe native patterns, weak validation, unsupported performance claims, and documentation gaps before a slice merges to `main`.

## Review Requirement

Every implementation slice requires an independent agent-driven review before merge.

Docs-only slices require review when they change:

```text
architecture contracts
semantic behavior
phase gates
agent harness rules
code or documentation standards
API stability policy
CPU or CUDA target policy
benchmark protocol
security or supply-chain policy
release or packaging policy
user-facing behavior claims
```

Native C++, CUDA, public API, packaging, performance, security, and release changes always require review.

## Independence

The reviewer must be independent from the implementer.

For Codex-driven work, independence means:

```text
use a fresh reviewer agent or subagent when the environment supports it
give the reviewer the branch diff and relevant source-of-truth docs
do not ask the same agent instance to rubber-stamp its own work as the only review
do not let review replace the implementer's own self-checks
```

If a subagent cannot be spawned in the current environment, the implementer must still perform a structured self-review and record that independent agent review was unavailable. High-risk slices should not merge without independent review unless the human owner explicitly accepts that exception.

## Review Timing

The normal sequence is:

```text
1. Implement the narrow slice on a feature branch.
2. Run local validation required for the slice.
3. Commit the branch in sensible chunks.
4. Request independent agent review before merge.
5. Fix valid findings or record why they do not apply.
6. Rerun validation after fixes and commit review fixes when needed.
7. Merge locally to main.
8. Validate merged main.
9. Push and confirm CI when CI exists.
```

Review should happen after the implementer believes the slice is ready, but before local merge to `main`.

## Reviewer Inputs

The reviewer should receive:

```text
branch name
commit list
git diff or PR diff
phase or slice goal
validation commands and output summary
benchmark commands and output summary when relevant
known limitations and out-of-scope items
links to relevant source-of-truth docs
```

For native, CUDA, benchmark, packaging, or release slices, include hardware and environment details when they affect the claim under review.

## Required Review Scope

Every review checks:

```text
scope matches docs/roadmap.md and the requested slice
public behavior matches docs/architecture/semantic_contracts.md
phase gates in docs/quality/phase_quality_gates.md are satisfied
code follows docs/quality/code_standards.md
user-facing and API docs follow docs/quality/documentation_standards.md
API changes follow docs/architecture/api_stability.md
CPU and CUDA target claims follow docs/architecture/hardware_targets_and_testing.md
benchmark claims follow docs/benchmarks/protocol.md
security and dependency choices follow docs/quality/security_and_supply_chain.md
packaging and release changes follow docs/quality/release_and_packaging.md
agent harness expectations follow docs/quality/agent_harness.md
validation evidence is fresh and sufficient for the claim
```

The reviewer must check both implementation and documentation drift. A slice is not review-clean if code and docs disagree.

## Focused Reviewer Roles

Use one broad reviewer for low-risk slices. Use focused reviewers for high-risk slices or large changes.

Focused roles:

```text
Semantic/API reviewer: contracts, Python behavior, exceptions, adapters, public API shape
C++ safety reviewer: ownership, lifetimes, overflow checks, undefined behavior, RAII, error translation
CUDA reviewer: device ownership, synchronization, streams, transfers, architecture targets, kernel launch safety
Performance reviewer: benchmark design, baselines, dataset validity, timing methodology, unsupported speedup claims
Testing/harness reviewer: validation entrypoint, CI parity, skips, property tests, oracles, sanitizer coverage
Docs/release reviewer: user-facing docs, release evidence, platform claims, installation, changelog
Security reviewer: native memory safety, dependencies, build provenance, secrets, supply-chain risk
```

Multiple focused reviewers are required before merge when a slice combines CUDA kernels, public API changes, and performance claims.

## Finding Format

Review findings must lead the review report and be ordered by severity.

Use this severity scale:

```text
P0: must fix before merge because it can corrupt results, break builds, create unsafe native behavior, or invalidate release claims
P1: must fix before merge unless the human owner explicitly defers it with rationale
P2: should fix before merge or record a follow-up with clear scope
P3: optional polish, clarity, or maintainability improvement
```

Each finding should include:

```text
severity
file path and line number when available
specific observed problem
why it matters
suggested correction or decision needed
```

Avoid vague findings such as "needs more tests" without naming the missing behavior, oracle, or failure mode.

## Finding Resolution

The implementer must resolve every P0 and P1 finding before merge.

Resolution options:

```text
fix the issue and rerun validation
prove the finding is invalid with evidence
ask the human owner to explicitly accept the risk
split the risky part out of scope and update docs or roadmap
```

P2 findings may be deferred only with a named follow-up and an explanation of why the current slice remains safe. P3 findings can be accepted or ignored without blocking.

## Review Evidence

Each reviewed slice closeout must record:

```text
reviewer type: independent agent, focused subagent, human, or self-review exception
review scope
finding counts by severity
P0/P1 resolution summary
P2 deferrals with follow-up scope
validation rerun after review fixes
residual risk
```

Before Phase 1 creates a durable validation script, this evidence may live in the agent final report. After Phase 1, the repo should add a lightweight closeout template or script support so review evidence is captured consistently for implementation and review-required docs/process slices.

## Review Automation Roadmap

Phase 1 should add review-policy checks that are easy to automate:

```text
docs/quality/code_review.md exists
AGENTS.md links to the review policy
CONTRIBUTING.md mentions the review stage
phase closeout checklist includes review evidence
```

Later phases should add stronger checks where practical:

```text
review evidence template
PR template with review checklist
CI check for required source docs
benchmark report schema checks
release evidence schema checks
CUDA availability and hardware metadata checks
```

Do not block early implementation on heavyweight review tooling. The required standard is independent review and recorded evidence; automation should make that easier over time.

## Non-Goals

The review stage does not require:

```text
large speculative branches
multiple reviewers for trivial docs typo fixes
human review for every local commit before a branch is ready
review replacing tests, benchmarks, sanitizers, or CI
```

Review exists to raise execution quality while preserving small, fast, validated slices.

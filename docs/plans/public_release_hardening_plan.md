# Public Release Hardening Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Transform FastPauli into a privacy-safe, technically rigorous, externally legible, production-quality open-source artifact that demonstrates agent-driven CPU/GPU kernel engineering.

**Architecture:** Preserve the validated CPU and accelerator semantics while separating public product surfaces from internal research machinery. Fix interop correctness before refactoring boundaries, make release artifacts minimal and reproducible, publish an excellent user-facing documentation layer, and add automated gates that keep privacy, protocol, native safety, packaging, and support claims correct.

**Tech Stack:** C++20, nanobind, Python 3.10+, NumPy, CMake/scikit-build-core, CUDA, ROCm/HIP, Metal, pytest/Hypothesis, Ruff, Pyright, GitHub Actions, MkDocs Material.

---

## Non-negotiable gates

1. Use strict RED-GREEN-REFACTOR for every behavioral code change.
2. Never disclose secret values while sanitizing artifacts.
3. Do not claim runtime validation for unavailable accelerator hardware.
4. Keep the repository buildable and the CPU test suite green after every task.
5. Run a spec-compliance review and an independent quality review for each implementation slice.
6. Do not rewrite or force-push remote Git history until the local sanitized tree is complete, backed up, and explicitly verified.
7. Do not publish packages or change repository visibility in this plan; prepare and verify the artifact, then report remaining external/authentication gates honestly.

## Task 1: Establish release-safe privacy and artifact policy

**Objective:** Remove sensitive/raw infrastructure evidence from the public tree and prevent recurrence without losing scientifically useful benchmark conclusions.

**Files:**
- Create: `docs/benchmarks/data/README.md`
- Create: `docs/quality/public_artifact_policy.md`
- Create: `scripts/audit_public_artifacts.py`
- Modify: `tools/remote/collect_rocm_inventory.sh`
- Modify: benchmark render/metadata collectors that propagate SSH targets, hostnames, absolute paths, GPU UUIDs, or unrestricted environment data
- Modify: `.gitignore`
- Modify: `pyproject.toml` sdist inclusion/exclusion policy
- Test: `tests/test_public_artifact_policy.py`
- Remove: raw profiler databases/reports and sensitive benchmark captures under `docs/benchmarks/data/`, retaining only sanitized summaries needed by public reports

**Steps:**
1. Write failing tests that detect private/home paths, SSH targets, IP addresses, hostnames, environment dumps, profiler binaries, and forbidden raw-artifact extensions in tracked files and sdists.
2. Verify the tests fail against the current repository.
3. Replace raw evidence with sanitized summaries/manifests and an archive policy; remove sensitive tracked files.
4. Change collectors to explicit metadata allowlists and deterministic redaction.
5. Exclude internal research data, plans, tests, and remote tooling from the sdist where not required for source builds.
6. Build an sdist, scan every member, verify size/path policy, and run relevant report/schema tests.
7. Commit the privacy-safe slice.

## Task 2: Correct and harden DLPack interoperability

**Objective:** Implement standards-compliant DLPack version negotiation and eliminate hand-maintained ABI drift.

**Files:**
- Modify: `bindings/python/pauli_sum_py.cpp`
- Create or vendor with license: official `dlpack.h` under an intentional third-party include path
- Modify: `CMakeLists.txt`
- Modify: `docs/architecture/api_stability.md`
- Test: focused DLPack tests in the existing CUDA interoperability test module

**Steps:**
1. Add failing tests for lower, equal, future-minor, and future-major `max_version` values and read-only capsule semantics.
2. Verify the current producer incorrectly echoes future versions.
3. Replace local DLPack ABI declarations with a pinned official header.
4. Define the exact producer-supported version and negotiate it against the consumer maximum.
5. Reject maximum versions that cannot express FastPauli's read-only contract.
6. Run focused tests, CPU suite, and compile checks.
7. Commit the protocol fix.

## Task 3: Harden CUDA Array Interface trust boundaries

**Objective:** Prevent a forged array-interface shape from driving native reads beyond the backing CUDA allocation.

**Files:**
- Modify: `src/cuda/expectation_cuda.cu`
- Modify: `bindings/python/pauli_sum_py.cpp` if ownership/trust documentation requires it
- Modify: `docs/architecture/cuda_backend.md`
- Test: CUDA protocol-validation tests and compile-only CPU stubs

**Steps:**
1. Add a failing CUDA test using a valid allocation with an inflated advertised shape.
2. Add checked byte arithmetic and query the allocation address range before kernel launch.
3. Reject pointers whose required byte range is outside the allocation.
4. Document trusted-producer and lifetime semantics.
5. Run compile tests locally; retain the runtime test with an explicit hardware skip when CUDA is unavailable.
6. Commit the boundary hardening.

## Task 4: Repair version, tag, release, and supply-chain integrity

**Objective:** Make source identity, tag identity, and publish provenance unambiguous and immutable.

**Files:**
- Modify: `pyproject.toml`
- Modify: `CMakeLists.txt`
- Modify: `CHANGELOG.md`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/release-wheelhouse.yml`
- Create: `.github/dependabot.yml`
- Modify/Create: release check scripts and tests

**Steps:**
1. Add failing tests requiring a single intended next-release version and exact `GITHUB_REF_NAME == v${version}` publish binding.
2. Move current development to the next coherent version and update release-facing metadata.
3. Pin every GitHub Action to a reviewed full commit SHA with a human-readable version comment.
4. Add least-privilege permissions, workflow concurrency, dependency audit, SBOM, and artifact attestation where supported.
5. Add project URLs and accurate metadata.
6. Run release-readiness and wheelhouse tests.
7. Commit the release-integrity slice.

## Task 5: Build an exceptional public landing page and documentation site

**Objective:** Make FastPauli immediately understandable, installable, credible, and visually compelling to users, researchers, and hiring/technical reviewers.

**Files:**
- Rewrite: `README.md`
- Create: `mkdocs.yml`
- Create: `docs/index.md`
- Create user guides under `docs/getting-started/`, `docs/guide/`, `docs/accelerators/`, and `docs/api/`
- Create: `.github/workflows/docs.yml`
- Move or link detailed research ledgers away from the README
- Test: documentation structure, links, snippets, and public-claim checks

**Steps:**
1. Add failing documentation tests for required installation, quickstart, support matrix, architecture, benchmark methodology, and navigation surfaces.
2. Write a concise README with a runnable first-screen example, clear support table, architecture overview, scoped performance highlights, and links.
3. Build a coherent MkDocs site with user journeys and architecture/kernel explanations.
4. Include diagrams/tables that explain packed symplectic representation, dispatch, and accelerator boundaries without marketing exaggeration.
5. Test README examples and build the documentation with strict warnings.
6. Commit the documentation slice.

## Task 6: Add public governance, security, support, and citation surfaces

**Objective:** Make the project safe and welcoming for external use and contribution.

**Files:**
- Create: `SECURITY.md`
- Create: `CODE_OF_CONDUCT.md`
- Create: `SUPPORT.md`
- Create: `GOVERNANCE.md`
- Create: `CITATION.cff`
- Create: `.github/ISSUE_TEMPLATE/*`
- Create: `.github/pull_request_template.md`
- Modify: `CONTRIBUTING.md`
- Modify: `README.md`
- Test: metadata/community-file validation

**Steps:**
1. Add failing tests for required public surfaces and metadata consistency.
2. Add concise, project-specific policies rather than boilerplate-only files.
3. Replace agent-centric contributor onboarding with human-first setup while retaining `AGENTS.md` as an automation guide.
4. Validate citation and issue-form YAML.
5. Commit the community-readiness slice.

## Task 7: Make the Python API typed, discoverable, and polished

**Objective:** Provide first-class static typing and predictable capability discovery without changing validated numerical semantics.

**Files:**
- Create: `python/fastpauli/_fastpauli_core.pyi`
- Create: `python/fastpauli/py.typed`
- Modify: `python/fastpauli/__init__.py`
- Modify: adapter modules
- Modify: `pyproject.toml`
- Create/Modify: API and typing tests

**Steps:**
1. Add failing package-content and Pyright tests for shipped stubs and public signatures.
2. Add reviewed nanobind stubs and the PEP 561 marker.
3. Reduce dynamic monkeypatching where it prevents truthful typing, or model it explicitly in stubs.
4. Add a structured capability report for available backends and operations, tested in CPU-only builds.
5. Run Pyright and adapter tests.
6. Commit the typed API slice.

## Task 8: Separate product bindings from internal benchmark machinery

**Objective:** Make the shipping extension legible and minimal while preserving internal research instrumentation in opt-in builds.

**Files:**
- Split: `bindings/python/pauli_sum_py.cpp`
- Split: `bindings/python/module.cpp`
- Create focused binding translation units for host API, device API, DLPack/array interop, status, and internal research hooks
- Modify: `CMakeLists.txt`
- Add: `FASTPAULI_BUILD_INTERNAL_TOOLS` option
- Modify tests to verify release wheels omit internal hooks

**Steps:**
1. Add failing artifact/API tests requiring internal campaign hooks to be absent from release builds and present only in explicit research builds.
2. Extract helpers without changing behavior, keeping tests green after each move.
3. Build public `_fastpauli_core` and opt-in `_fastpauli_bench` boundaries, or an equivalent clean separation.
4. Update benchmark harness imports and internal documentation.
5. Run CPU, adapter, package, and compile tests.
6. Commit the binding-boundary refactor.

## Task 9: Reduce accelerator backend duplication safely

**Objective:** Share stable host-side semantics between CUDA and HIP without obscuring backend-specific kernel engineering.

**Files:**
- Create shared detail helpers under `src/accelerator/` or `src/detail/`
- Modify corresponding CUDA/HIP matmul, expectation, transfer, and simplify sources
- Add cross-backend contract tests

**Steps:**
1. Start with the nearly identical matmul host orchestration and add source/contract tests.
2. Extract checked shape/size semantics and result construction into backend-neutral helpers.
3. Extract allocation/copy/error traits only where the abstraction remains clearer than duplication.
4. Repeat narrowly for expectation and transfer code.
5. Do not merge genuinely different kernels or primitives.
6. Require CUDA/HIP compile evidence; do not claim runtime equivalence without hardware.
7. Commit each narrow refactor independently.

## Task 10: Improve native performance ergonomics and build structure

**Objective:** Avoid unnecessary Python serialization and make build logic maintainable.

**Files:**
- Modify binding call wrappers
- Split `CMakeLists.txt` into `cmake/*.cmake`
- Add `CMakePresets.json`
- Add native warning/sanitizer options and tests

**Steps:**
1. Add concurrency/lifetime tests for representative long-running native calls.
2. Release the GIL only around pure native computation and synchronization, reacquiring before Python object work.
3. Modularize CMake by CPU features, CUDA, HIP, Metal, compiler metadata, and release policy.
4. Add project-only warning flags and sanitizer presets while treating third-party headers separately.
5. Run build matrix and regression tests.
6. Commit the ergonomics/build slice.

## Task 11: Upgrade CI and quality gates

**Objective:** Make every public quality claim continuously enforced.

**Files:**
- Modify/Create GitHub workflows
- Configure Ruff, Pyright, codespell, documentation checks, package audits, ASan/UBSan, CodeQL, and artifact scans
- Add optional/scheduled accelerator workflow contracts

**Steps:**
1. Add local tests for workflow structure and support-claim consistency.
2. Enforce Ruff/formatting on maintained Python code with deliberate exclusions for archived evidence.
3. Add Pyright, codespell, package build/smoke, dependency audit, secret scan, and public-artifact audit.
4. Add ASan/UBSan CPU jobs and CodeQL.
5. Add scheduled/self-hosted accelerator lanes with explicit compile/runtime evidence vocabulary.
6. Run all locally reproducible checks.
7. Commit the CI hardening slice.

## Task 12: Final integration, external review, and publication handoff

**Objective:** Prove the complete repository is coherent, privacy-safe, reproducible, and ready for a reviewed public-release PR.

**Steps:**
1. Run the full local validation matrix, package builds, clean-wheel smoke tests, docs build, public-artifact scan, type/lint/security checks, and Git diff checks.
2. Build a fresh local clone/archive and repeat public-tree and sdist scans.
3. Dispatch independent spec, security, architecture, docs, and release reviewers.
4. Fix every critical or important issue and re-review.
5. Produce a history-sanitization runbook and verify a local rewritten clone before any remote force push.
6. Commit all verified changes in coherent conventional commits.
7. Push the feature branch and open a detailed PR when GitHub authentication is available.
8. Do not merge, publish, force-push rewritten history, or change visibility without passing hosted CI and the external account gates.

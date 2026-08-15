# Wolfgang Rename and Public Publication Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Rename the active public FastPauli tree to the Wolfgang brand, with collision-resistant machine identities, while keeping existing reachable Git history private and preserving historical FastPauli evidence as provenance rather than rewriting the past.

**Architecture:** Treat the rename as a current-tree publication boundary, not a history-rewrite exercise. Move the active product surface to `Wolfgang` / `wolfgang-quantum` / `wolfgang_quantum`, preserve historical FastPauli references only inside provenance and release/evidence ledgers, and add narrow compatibility shims for one transition release so the rename is mechanically auditable instead of being a blind search-and-replace.

**Tech Stack:** C++20, nanobind, Python 3.10+, NumPy, scikit-build-core, CMake, pytest, GitHub Actions, MkDocs Material.

---

## Decided public identity

Use this exact public identity set:

- Product / project name: `Wolfgang`
- GitHub org namespace: `wolfgang-quantum`
- Repository name: `wolfgang`
- Python distribution: `wolfgang-quantum`
- Python import package: `wolfgang_quantum`
- C++ namespace: `wolfgang`
- Primary domain candidate: `wolfgangquantum.com`

Why this is the correct choice:

1. Bare `wolfgang` is already occupied on PyPI and npm, so the machine-facing names must stay qualified even if the human-facing brand is simply `Wolfgang`.
2. `wolfgang-quantum` / `wolfgang_quantum` were the cleanest candidate distribution/import pair in the namespace validation handoff.
3. The repository path should be `wolfgang-quantum/wolfgang`: the org carries the disambiguation, while the repo path stays short and brand-forward.
4. The active public tree should use `wolfgang` as the C++ namespace for ergonomic APIs, while Python keeps the collision-resistant import `wolfgang_quantum`.

## Compatibility policy

1. Active public naming changes immediately to Wolfgang across package metadata, docs, workflows, install paths, headers, bindings, and tests.
2. Historical FastPauli references remain intact in provenance, release ledgers, benchmark reports, historical URLs, and archived plan/evidence documents. Do not rewrite these documents as though Wolfgang was always the historical name.
3. Provide a one-transition-release compatibility layer for source consumers only:
   - Python import shim: `python/fastpauli/` remains as a forwarding package with deprecation warnings.
   - C++ header shim: `include/fastpauli/*.hpp` forwards to `include/wolfgang/*.hpp` with deprecation comments.
   - Legacy environment-variable fallback: runtime and script entrypoints may accept `FASTPAULI_*` as aliases to `WOLFGANG_*` for one transition release where practical.
4. Do not dual-publish a PyPI distribution named `fastpauli`.
5. Remove the compatibility shims in the first post-transition minor release only after tests, docs, and release notes stop depending on them.

## Release and history boundary policy

1. Never publish from existing reachable private history unless `scripts/audit_public_artifacts.py --history` and `docs/release/history_sanitization.md` both pass on a reviewed rewritten mirror.
2. Default publication plan: create a clean public repository/root commit from the reviewed Wolfgang tree if the full-history audit remains non-zero.
3. Do not rename the private remote during implementation.
4. Do not change repository visibility, tags, package indexes, or trusted-publishing settings in the rename implementation slice.

## Mandatory gates

1. RED-GREEN-REFACTOR for every behavior or contract change.
2. Keep all historical/provenance documents factually correct about the FastPauli past.
3. Never print or commit secrets while auditing history or artifacts.
4. The rename is not done until focused identity tests and broad validation both pass.
5. The plan implementation must end with either:
   - a reviewed rename branch ready for clean-history publication, or
   - an honest stop because history/privacy gates still fail.

## Rollback points

- Rollback A: after identity-contract tests land but before package path renames.
- Rollback B: after package/header/binding rename but before docs/workflow rewrites.
- Rollback C: after docs/workflow rewrites but before release/history gate updates.
- Rollback D: before any future history rewrite or public repo creation; restore from a verified private bundle as described in `docs/release/history_sanitization.md`.

## Task 1: Freeze the rename contract in tests

**Objective:** Add failing tests that define the new active public identity and the required preserved historical exceptions.

**Files:**
- Modify: `tests/test_public_project_surface.py`
- Create: `tests/test_wolfgang_identity_contract.py`
- Modify: `tests/test_public_artifact_policy.py`

**Step 1: Write failing tests**

Add tests that require:
- `pyproject.toml` project name `wolfgang-quantum`
- `python/wolfgang_quantum/` to exist
- `mkdocs.yml` and `README.md` to say `Wolfgang`
- public URLs to target the future Wolfgang repo/domain surfaces
- historical documents under `docs/research/` and `docs/release/` to preserve explicit `FastPauli` provenance language

**Step 2: Run the focused tests and verify failure**

Run: `pytest tests/test_public_project_surface.py tests/test_wolfgang_identity_contract.py tests/test_public_artifact_policy.py -q`
Expected: FAIL with missing Wolfgang package paths and FastPauli-first public metadata.

**Step 3: Commit the red-phase tests only after reviewing the failure output in the task log**

Do not make code changes yet.

**Step 4: Rollback point**

If the tests cannot express the provenance exception cleanly, stop and simplify the contract before any rename work.

## Task 2: Rename packaging metadata and Python package roots

**Objective:** Move the canonical Python package identity to `wolfgang-quantum` / `wolfgang_quantum` without breaking import-time smoke coverage.

**Files:**
- Modify: `pyproject.toml`
- Create: `python/wolfgang_quantum/__init__.py`
- Create: `python/wolfgang_quantum/__init__.pyi`
- Create: `python/wolfgang_quantum/_capabilities.py`
- Create: `python/wolfgang_quantum/_fastpauli_core.pyi`
- Create: `python/wolfgang_quantum/_version.py`
- Create: `python/wolfgang_quantum/openfermion.py`
- Create: `python/wolfgang_quantum/qiskit.py`
- Create: `python/wolfgang_quantum/py.typed`
- Modify: `CMakeLists.txt`
- Modify: any import/package tests that hard-code `python/fastpauli`

**Step 1: Copy the package tree into the new canonical path**

Create `python/wolfgang_quantum/` as the canonical package and update `tool.scikit-build.wheel.packages` accordingly.

**Step 2: Run targeted tests to verify they still fail for import wiring**

Run: `pytest tests/test_wolfgang_identity_contract.py -q`
Expected: FAIL until package exports and install paths are updated consistently.

**Step 3: Update packaging metadata minimally**

Change:
- `[project].name` to `wolfgang-quantum`
- author display text from `FastPauli contributors` to `Wolfgang contributors`
- project URLs to the planned Wolfgang surfaces
- wheel package root from `python/fastpauli` to `python/wolfgang_quantum`
- install destination from `fastpauli` toward `wolfgang_quantum`

**Step 4: Run focused tests again**

Run: `pytest tests/test_wolfgang_identity_contract.py tests/test_basic.py -q`
Expected: package-root and metadata tests move to PASS or to the next narrower failure.

**Step 5: Rollback point**

If the canonical package path breaks extension packaging too widely, stop before touching bindings and revert to the previous passing state.

## Task 3: Add the Python compatibility import shim

**Objective:** Keep `import fastpauli` working for one transition release while making `wolfgang_quantum` canonical.

**Files:**
- Modify: `python/fastpauli/__init__.py`
- Modify: `python/fastpauli/__init__.pyi`
- Modify: `python/fastpauli/_capabilities.py`
- Modify: `python/fastpauli/_version.py`
- Modify: `python/fastpauli/openfermion.py`
- Modify: `python/fastpauli/qiskit.py`
- Create: `tests/test_fastpauli_compatibility.py`

**Step 1: Write failing compatibility tests**

Require:
- `import wolfgang_quantum` is canonical
- `import fastpauli` still succeeds
- both packages expose matching `PauliSum` / capability / version surfaces
- importing through `fastpauli` emits a deprecation warning or carries a documented deprecation marker

**Step 2: Run the compatibility test first**

Run: `pytest tests/test_fastpauli_compatibility.py -q`
Expected: FAIL because `fastpauli` is still the canonical package rather than a shim.

**Step 3: Implement the forwarding package**

Make `python/fastpauli/` a thin import-forwarding layer into `wolfgang_quantum`.

**Step 4: Re-run focused tests**

Run: `pytest tests/test_fastpauli_compatibility.py tests/test_basic.py tests/test_qiskit_adapter.py tests/test_openfermion_adapter.py -q`
Expected: PASS on CPU-only environments, with optional-dependency skips unchanged.

## Task 4: Rename the native extension, bindings, headers, and namespace surface

**Objective:** Make Wolfgang the active native identity while preserving temporary forwarders for source consumers.

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `bindings/python/module.cpp`
- Modify: `bindings/python/stable_bindings.cpp`
- Modify: `bindings/python/internal_bindings.cpp`
- Modify: `bindings/python/pauli_sum_py.cpp`
- Create: `include/wolfgang/accelerator_status.hpp`
- Create: `include/wolfgang/cpu_backend.hpp`
- Create: `include/wolfgang/device_commutation_matrix.hpp`
- Create: `include/wolfgang/device_pauli_sum.hpp`
- Create: `include/wolfgang/errors.hpp`
- Create: `include/wolfgang/pauli_sum.hpp`
- Modify: existing headers under `include/fastpauli/`
- Modify: C++ sources that reference `fastpauli` namespace or `_fastpauli_core`
- Create: `tests/test_binding_layer_policy.py` updates or split-out rename-specific native tests

**Step 1: Add/extend failing tests**

Require:
- built extension module name `_wolfgang_core`
- canonical package imports extension from Wolfgang paths
- public header includes compile under `include/wolfgang/*`
- compatibility headers under `include/fastpauli/*` forward cleanly

**Step 2: Run targeted tests or compile checks**

Run: `pytest tests/test_binding_layer_policy.py -q`
Expected: FAIL on `_fastpauli_core`, `include/fastpauli`, and namespace assumptions.

**Step 3: Implement the rename in the smallest safe order**

Change, in order:
1. extension target and install destination
2. canonical include directory
3. namespace tokens in public headers/bindings
4. compatibility forwarding headers

**Step 4: Run focused validation**

Run: `pytest tests/test_binding_layer_policy.py tests/test_public_typing_and_capabilities.py -q`
Expected: PASS.

**Step 5: Rollback point**

If native extension renaming causes widespread unresolved imports, revert to the last green state before proceeding to docs/workflows.

## Task 5: Rename environment variables, build flags, and runtime metadata with temporary aliases

**Objective:** Make `WOLFGANG_*` the canonical build/runtime identity without abruptly dropping old automation.

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `python/wolfgang_quantum/__init__.py`
- Modify: `src/accelerator_status.cpp`
- Modify: `src/cpu_backend.cpp`
- Modify: `src/grouping.cpp`
- Modify: `src/commute.cpp`
- Modify: `src/cuda/commutation_cuda.cu`
- Modify: `src/cuda/simplify_cuda.cu`
- Modify: `src/cuda/workspace.cu`
- Modify: `src/hip/device_pauli_sum.hip.cpp`
- Modify: `src/hip/simplify_hip.hip.cpp`
- Modify: tests that assert exact `FASTPAULI_*` identifiers
- Create: `tests/test_wolfgang_env_compat.py`

**Step 1: Write failing env-alias tests**

Require:
- canonical docs/metadata expose `WOLFGANG_*`
- legacy `FASTPAULI_*` values still map to the same runtime behavior where compatibility is promised
- ambiguous mixed use fails loudly instead of silently preferring one value

**Step 2: Run the focused env tests**

Run: `pytest tests/test_wolfgang_env_compat.py -q`
Expected: FAIL because only `FASTPAULI_*` exists.

**Step 3: Implement canonical names and alias logic**

Prefer `WOLFGANG_*` everywhere public-facing while accepting `FASTPAULI_*` as deprecated aliases for one transition release.

**Step 4: Re-run focused validation**

Run: `pytest tests/test_wolfgang_env_compat.py tests/test_public_typing_and_capabilities.py -q`
Expected: PASS.

## Task 6: Rewrite user-facing docs and metadata, but preserve historical ledgers

**Objective:** Make the README, docs site, governance files, and metadata say Wolfgang while clearly marking legacy FastPauli provenance where required.

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `CONTRIBUTING.md`
- Modify: `SECURITY.md`
- Modify: `SUPPORT.md`
- Modify: `GOVERNANCE.md`
- Modify: `CODE_OF_CONDUCT.md`
- Modify: `CITATION.cff`
- Modify: `mkdocs.yml`
- Modify: `docs/index.md`
- Modify: `docs/getting-started/installation.md`
- Modify: `docs/getting-started/quickstart.md`
- Modify: `docs/getting-started/conventions.md`
- Modify: `docs/guide/architecture.md`
- Modify: `docs/guide/agent-driven-engineering.md`
- Modify: `docs/guide/python-api.md`
- Modify: `docs/accelerators/overview.md`
- Modify: `docs/api/index.md`
- Modify: `docs/roadmap.md`
- Modify: `.github/ISSUE_TEMPLATE/bug_report.yml`
- Modify: `.github/ISSUE_TEMPLATE/performance_regression.yml`
- Modify: `.github/ISSUE_TEMPLATE/hardware_support.yml`
- Modify: `.github/pull_request_template.md`
- Modify: tests that assert old headings and names

**Step 1: Expand the failing docs tests first**

Require:
- README headings say `Wolfgang`
- install commands use `pip install wolfgang-quantum`
- examples import `wolfgang_quantum`
- provenance docs still explain that historical evidence was generated under the FastPauli name

**Step 2: Run focused docs tests**

Run: `pytest tests/test_public_project_surface.py -q`
Expected: FAIL on old FastPauli headings and URLs.

**Step 3: Rewrite public surfaces**

Rewrite user-facing docs and governance files to Wolfgang, but only add provenance disclaimers to historical ledgers instead of globally search-replacing them.

**Step 4: Re-run docs validation**

Run: `pytest tests/test_public_project_surface.py -q && python -m mkdocs build --strict`
Expected: PASS.

**Step 5: Rollback point**

If strict docs build fails because historical files cannot preserve old names and still satisfy link rules, fix the navigation/provenance policy before continuing.

## Task 7: Update workflows, release scripts, and artifact naming to Wolfgang

**Objective:** Ensure CI, packaging, docs, and release scripts operate on the Wolfgang identity set without publishing.

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/quality.yml`
- Modify: `.github/workflows/codeql.yml`
- Modify: `.github/workflows/docs.yml`
- Modify: `.github/workflows/release-wheelhouse.yml`
- Modify: `scripts/check_release_readiness.py`
- Modify: `scripts/check_release_tag.py`
- Modify: `scripts/check_release_wheelhouse.py`
- Modify: `scripts/prepare_publish_dist.py`
- Modify: `scripts/validate.py`
- Modify: `scripts/validate_release_artifacts.py`
- Modify: `scripts/wheel_smoke.py`
- Modify: `scripts/write_release_checksums.py`
- Modify: `tests/test_release_supply_chain.py`
- Modify: `tests/test_release_wheelhouse_foundation.py`
- Modify: `tests/test_release_candidate_foundation.py`
- Modify: `tests/test_release_candidate_rc2.py`
- Modify: `tests/test_release_final_010.py`
- Modify: `tests/test_release_next_checkpoint.py`
- Modify: `tests/test_release_artifact_validation.py`
- Modify: `tests/test_validate_entrypoint.py`

**Step 1: Add or tighten failing release tests**

Require:
- dist artifacts are named `wolfgang_quantum` / `wolfgang-quantum` consistently
- workflows reference Wolfgang package paths
- release-readiness checks reject stale `FastPauli` public artifact names except inside historical ledgers

**Step 2: Run release-focused tests before implementation**

Run: `pytest tests/test_release_supply_chain.py tests/test_release_artifact_validation.py tests/test_validate_entrypoint.py -q`
Expected: FAIL on FastPauli naming assumptions.

**Step 3: Implement workflow/script updates**

Update artifact names, wheel smoke imports, URL checks, and release-readiness text to Wolfgang-aware rules.

**Step 4: Re-run focused release validation**

Run: `pytest tests/test_release_supply_chain.py tests/test_release_artifact_validation.py tests/test_validate_entrypoint.py -q`
Expected: PASS.

## Task 8: Explicitly preserve historical FastPauli evidence and add provenance guardrails

**Objective:** Mark the historical ledger as legacy identity evidence so future maintainers do not rewrite it incorrectly.

**Files:**
- Modify: `docs/research/provenance.md`
- Modify: `docs/release/README.md`
- Modify: `docs/release/0.1.0-rc1.md`
- Modify: `docs/release/0.1.0-rc2.md`
- Modify: `docs/release/0.1.0-wheelhouse-dry-run.md`
- Modify: `docs/release/0.1.0.md`
- Modify: `docs/benchmarks/protocol.md`
- Modify: any renderer/test helpers that assume all retained prose must be Wolfgang-branded
- Create: `tests/test_historical_provenance_policy.py`

**Step 1: Write failing provenance-policy tests**

Require:
- historical ledgers say they describe FastPauli-era artifacts
- public-facing docs do not present those old names as the current brand
- renderers/tests allow this exact exception set

**Step 2: Run the provenance tests**

Run: `pytest tests/test_historical_provenance_policy.py -q`
Expected: FAIL until the exception policy is encoded.

**Step 3: Add explicit legacy disclaimers, not rewrites**

Add short preambles describing the documents as historical FastPauli evidence preserved during the Wolfgang rename.

**Step 4: Re-run the provenance tests**

Run: `pytest tests/test_historical_provenance_policy.py tests/test_public_artifact_policy.py -q`
Expected: PASS.

## Task 9: Run package, import, docs, and broad repository validation

**Objective:** Prove the renamed tree works locally before any clean-history/publication step is considered.

**Files:**
- No new source files required unless validation exposes gaps.

**Step 1: Run the focused matrix**

Run:
- `pytest tests/test_wolfgang_identity_contract.py tests/test_fastpauli_compatibility.py tests/test_wolfgang_env_compat.py tests/test_historical_provenance_policy.py -q`
- `pytest tests/test_basic.py tests/test_public_typing_and_capabilities.py tests/test_binding_layer_policy.py tests/test_public_project_surface.py -q`
- `pytest tests/test_release_supply_chain.py tests/test_release_artifact_validation.py tests/test_validate_entrypoint.py -q`

Expected: PASS.

**Step 2: Run broad validation**

Run:
- `python scripts/validate.py`
- `python -m build`
- `python scripts/validate_release_artifacts.py --output-dir /tmp/wolfgang-release-artifacts`
- `python scripts/audit_public_artifacts.py --tracked`

Expected: PASS.

**Step 3: Smoke-test the built package**

Run in a fresh venv:
- `python -m pip install dist/*.whl`
- `python -c "import wolfgang_quantum, fastpauli; print(wolfgang_quantum.__version__); print(fastpauli.__version__)"`

Expected: PASS, with the compatibility import still available.

## Task 10: Rehearse the privacy boundary for publication

**Objective:** Decide whether public release can proceed from rewritten history or must proceed from a clean root commit.

**Files:**
- Modify: `docs/release/history_sanitization.md` only if the rename changes command examples or artifact names
- Create if needed: `docs/release/wolfgang_publication_checklist.md`

**Step 1: Audit tracked files and history**

Run:
- `python scripts/audit_public_artifacts.py --tracked`
- `python scripts/audit_public_artifacts.py --history`

Expected:
- `--tracked` must PASS
- `--history` may FAIL because existing reachable history is already known to be unsafe

**Step 2: Branch on the result**

- If `--history` PASSes on a reviewed rewritten mirror, document the exact rewrite/publish gates.
- If `--history` still fails, stop pretending history can be published and document the clean-root public-repo path as mandatory.

**Step 3: Record the final gate honestly**

The final implementation handoff must state which of these is true:
1. `Reviewed Wolfgang tree ready for clean public root commit/repo creation`, or
2. `Reviewed Wolfgang tree ready for rewritten-history publication`, or
3. `Rename complete but publication blocked by unresolved privacy/history findings`.

## Task 11: Commit strategy

**Objective:** Land the rename in reviewable slices and keep the current plan task itself constrained.

**Files:**
- Modify: `docs/plans/2026-08-11-wolfgang-rename-and-publication-plan.md` only in this planning task

**Step 1: For this planning task only**

Run:
- `git add docs/plans/2026-08-11-wolfgang-rename-and-publication-plan.md`
- `git commit -m "docs: add Wolfgang rename and publication plan"`

Expected: exactly one new committed plan file.

**Step 2: For the future implementation task**

Use sensible commits by slice:
- `test: freeze Wolfgang identity contract`
- `feat: rename canonical Python package to wolfgang_quantum`
- `feat: rename native and docs surfaces to Wolfgang`
- `ci: rename release and workflow artifact identities`
- `docs: mark FastPauli evidence as historical provenance`

**Step 3: Final reminder**

Do not rename the GitHub remote, change visibility, tag a release, upload packages, or rewrite history as part of the implementation slice itself.

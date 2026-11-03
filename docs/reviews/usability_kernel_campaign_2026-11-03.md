# Usability And Kernel Campaign Review — 2026-11-03

Plan: [usability and kernel campaign](../plans/usability_and_kernel_campaign_plan.md).
Each slice had an independent agent review against
[the review policy](../quality/code_review.md) before merge.

## Slice 1: CUDA simplify hardening

Branch `codex/cuda-simplify-hardening`, commits `c65553c`..`3198e0e`.

- Reviewer: independent agent (CUDA and performance roles), read-only.
- Findings: P0 0, P1 0, P2 3, P3 9.
- P2 resolutions:
  - Generic rows gained transfer-inclusive timings, and the CPU crossover claim
    now names its boundary.
  - Every CUDA path uses the CPU drop rule for finite coefficients whose
    magnitude overflows, with an oracle test. Previously a 0 * inf threshold
    dropped every term.
  - CPU tests now syntax-check the CUDA and HIP workspace headers, so the
    `ee4f95f` build break is caught without a GPU toolkit.
- P3 resolutions:
  - comment wording, test construction and long-run tile coverage;
  - selector validation order;
  - report accuracy and the changelog range;
  - selector updates moved out of timed regions;
  - doc drift;
  - initcheck and synccheck runs added.
- P3 deferred with reason: squashing the test fix into `c65553c` for
  bisectability. Rewriting history would change the revisions recorded in the
  benchmark evidence. Commits `c65553c`..`2a3053b` fail
  `test_backend_workspace_headers_share_private_contract_helpers`, which
  `4724bba` repairs.
- Validation after fixes (H100, CUDA 12.8, `sm_90`):
  - `WOLFGANG_VALIDATE_CUDA=1 python scripts/validate.py --profile core` passed;
  - CUDA foundation, kernel and generic-simplify tests: 130 passed;
  - Compute Sanitizer memcheck, racecheck and synccheck clean;
  - initcheck reports only CUB `ReduceByKeyAgent` look-back reads, identical
    (1,356) on the pre-change baseline;
  - local quality and docs profiles passed.
- Residual risk:
  - H100 is the only GPU with evidence;
  - HIP lacks the single-term fast path and the overflow drop rule;
  - CPU CI still compiles no `.cu` sources.

Evidence: [CUDA simplify paths report](../benchmarks/reports/cuda_simplify_paths_h100_2026-11-03.md).

## Slices 2 and 3: packed-array I/O and operator algebra

Branch `codex/host-array-io-and-algebra`, commits `7c64436`..`de87752`. The
branch is merged with a merge commit so the benchmark revisions stay reachable.

- Reviewer: independent agent (semantic/API and C++ safety roles), read-only.
- Findings: P0 0, P1 0, P2 6, P3 8.
- P2 resolutions:
  - Commutator coefficients are computed as `2 * (a * b)`, so doubling cannot
    overflow on its own.
  - Non-finite coefficients raise up front, regardless of which pairs cancel.
  - The bulk `from_labels` path is limited to NumPy arrays, so duck-typed
    arrays keep working.
  - The changelog records the `==`, hashing and `bool()` behavior change with a
    migration note.
  - The speed wording is backed by the host API report.
  - Zero-copy CUDA `DevicePauliSum` views moved to slice 7 in the plan.
- P3 resolutions:
  - `from_words` validates its own copy;
  - the binding policy documents the zero-copy view exception;
  - typed `self`, `op[None]` and guardrail messages fixed;
  - commutator output is sized exactly with a counting pass;
  - the NumPy object-array quirk and the `equiv` `rtol` scale are documented;
  - bool coefficients are accepted, and the byte-order message is clearer;
  - native contract tests cover `from_words`, `commutator`, `select_terms` and
    equality.
- Validation after fixes (Apple M4 Pro): `python scripts/validate.py --profile core`
  passed (598 Python tests, native CTest); the new test file passes 37 tests.
  The refreshed timings showed no commutator regression.
- Residual risk:
  - random operands only; Hamiltonians where most terms commute change the
    fused-vs-composed ratio;
  - x86_64 commutator rows predate the counting pass.

Evidence: [host API report](../benchmarks/reports/host_api_2026-11-03.md).

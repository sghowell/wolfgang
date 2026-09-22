# Repository Review Hardening

Baseline: `4173a08`. The September 2026 repository review identified reproducible
correctness and lifetime failures and gaps in public API and benchmark contracts.
This plan records the accepted implementation sequence.

## Sequence and acceptance

1. Correctness: preserve small OpenFermion coefficients, make CPU simplify and
   count normalization robust at finite numeric limits, and preserve the existing
   input-relative tolerance contract while limiting the idempotence claim to
   absolute tolerance. Add failing regressions before fixes.
2. Metal lifetimes and evidence: contain autoreleased execution objects, verify
   main/worker-thread reuse, and reject incomplete Wave 1D evidence. Hardware
   absence must remain a skip rather than a promotion or an exception.
3. Public contracts: repair capability fallback and typing, describe operation
   execution locations, execute onboarding examples, remove absent sorting API
   promises, and fix checks that inspect empty or stale file sets.
4. Native boundary and validation: give the core an independent build target,
   reduce coefficient export allocation, release the GIL only around safe CPU
   computation, and unify explicit validation profiles. Record loaded-build
   provenance before attributing benchmark results to a checkout.
5. Performance experiments: measure local boundary changes with unchanged
   controls and prepare specific CUDA generic-reduction, count-reduction, and
   output-reuse experiments. Promotion requires correctness, runtime and sanitizer
   evidence on the affected hardware, including small-workload regressions.

No paid hardware is acquired by this plan. CUDA/HIP kernel candidates must not
be promoted on CPU/Metal validation alone. Historical benchmark and release
ledgers remain immutable; current guidance records qualifications separately.

## Validation and review

Each implementation slice uses the repository validation entrypoint, focused
regressions, appropriate adapter or Metal execution, and an independent agent
review before merge. Default public builds and internal validation builds must
both work. CPU-only public headers must compile without framework dependencies.
Closeout records exact commands, outcomes, review resolutions and remaining
hardware qualification boundaries. No timing improvement is claimed without
repeated same-boundary measurements.

Performance follow-ups and qualification inputs: [experiment queue](review_performance_experiments.md).

Status: implementation in progress; CPU/Metal hardening and boundary validation underway.

# ROCm Campaign 8 Architecture-Readiness Report

Date: 2026-05-01

## Scope

Campaign 8 converts the Campaign 7 residual ROCm items into explicit gates for
future implementation. It does not add runtime behavior.

No HIP kernel, public Python API, ROCm wheel, or multi-GPU runtime behavior changed
in this campaign. Simultaneous CUDA+HIP source builds remain
unavailable. Non-MI300X AMD portability remains blocked until a real
non-MI300X AMD GPU lane provides source-build, runtime, benchmark, and profiler
evidence.

## Evidence

Checked evidence:

```text
docs/benchmarks/data/rocm_campaign8_architecture_readiness_2026-05-01/
docs/benchmarks/data/rocm_campaign8_architecture_readiness_2026-05-01/summary.json
docs/benchmarks/data/rocm_campaign8_architecture_readiness_2026-05-01/raw/readiness_commands.json
docs/benchmarks/data/rocm_campaign8_architecture_readiness_2026-05-01/raw/render_manifest.json
docs/benchmarks/data/rocm_campaign8_architecture_readiness_2026-05-01/logs/readiness_lane.txt
```

The readiness command inventory was regenerated from commit
`5bca1095b06f97300b165492b08fed85cd91dd70` after review-response fixes clarified
that validation and configure checks are external closeout requirements rather
than script-executed results. Later Campaign 8 commits add the checked report and
documentation closeout; they do not change FastPauli runtime code, HIP kernels,
public Python APIs, wheels, or multi-GPU behavior.

The broad README performance landscape is preserved:

![FastPauli accelerator performance landscape](../plots/accelerator_landscape_with_rocm.svg)

## Decision Outcomes

| Item | Status | Evidence or decision |
|---|---|---|
| `backend_neutral_object_model` | `accepted_for_future_implementation` | `docs/architecture/backend_neutral_accelerators.md` |
| `simultaneous_cuda_hip_source_builds` | `unavailable` | Current configure-time rejection retained |
| `multi_gpu_rocm_execution` | `out_of_scope_with_next_trigger` | Requires later multi-GPU design and hardware evidence |
| `non_mi300x_amd_portability` | `blocked_external` | No non-MI300X AMD GPU host was provided for Campaign 8 |
| `rocm_wheel_packaging_design` | `accepted_for_future_implementation` | Packaging gate added to release policy |
| `rocm_ci_hardware_policy` | `accepted_for_future_implementation` | CI hardware gate added before any ROCm wheel/support claim |
| `rocm_clean_machine_install_tests` | `accepted_for_future_implementation` | Clean-machine gate added before any ROCm wheel claim |
| `rocprofv3_migration` | `accepted_for_future_implementation` | `docs/plans/rocm_profiler_migration_campaign8_decision.md` |
| `legacy_rocprof_retention` | `retained` | Legacy `rocprof` remains accepted while it produces HIP trace/stats |
| `external_hip_statevector_contract` | `accepted_for_future_implementation` | `docs/plans/rocm_hip_interop_reconsideration_campaign8_decision.md` |
| `hip_dlpack_reconsideration_contract` | `accepted_for_future_implementation` | Reconsideration gate accepted; implementation remains unavailable |
| `hip_cuda_array_interface_policy` | `rejected_with_evidence` | HIP pointers must not be exposed as CUDA memory |
| `public_streams_policy` | `rejected_with_evidence` | Campaign 5 rejection retained |
| `public_graphs_policy` | `rejected_with_evidence` | Campaign 5 rejection retained |
| `public_workspaces_policy` | `rejected_with_evidence` | Campaign 5 rejection retained |
| `targeted_rocm_performance_reopen` | `accepted_for_future_implementation` | Future performance work requires profiler-backed retained-operation bottleneck |
| `source_build_release_lane_retention` | `retained` | Campaign 7 MI300X source-build release lane retained |

## Architecture Outcome

Campaign 8 accepts `docs/architecture/backend_neutral_accelerators.md` as the
gate for any future multi-backend source build. Current builds remain CPU-only,
CUDA-only, or HIP-only. A later implementation campaign must prove object-local
backend identity, device ordinal semantics, same-device validation,
cross-backend errors or explicit copy behavior, structured accelerator status,
and package/support wording before simultaneous CUDA+HIP can move out of
unavailable status.

## Portability And Packaging Outcome

The non-MI300X AMD lane is executable only when real hardware exists. The lane
requires:

```text
source build on that architecture
runtime status capture
retained HIP operation tests
benchmark smoke with correctness checks
profiler availability status
README support wording update
```

ROCm wheels remain unavailable. A future ROCm wheel campaign must specify:

```text
supported package channel
ROCm runtime dependency policy
CI hardware that can build and import the wheel
clean-machine install test for the produced artifact
support-matrix wording separating source-build evidence from wheel support
manylinux and platform-tag policy
```

## Profiler Migration Outcome

Legacy `rocprof` remains retained. `rocprofv3` is accepted as the future ROCm
7.x migration lane only after side-by-side evidence exists for a retained
operation. Profiler unavailability must record the exact missing binary,
permission failure, incompatible option, provider image limitation, or runtime
error.

## Interop Reconsideration Outcome

External HIP statevector pointers and HIP DLPack remain unavailable. A future
interop campaign must define ownership, stream synchronization, read-only
behavior, consumer-library versions, mutation tests, and benchmark timing
boundaries. HIP CUDA Array Interface remains rejected because HIP memory must
not be exposed as CUDA memory.

## Targeted Performance Reopen Outcome

Campaign 8 rejects same-host ROCm reruns as performance work unless a future
campaign first records:

```text
retained operation
profiler artifact
measured bottleneck
proposed implementation
correctness oracle
A/B timing boundary
rejection criteria
```

This keeps ROCm performance work tied to a concrete bottleneck and prevents
release-support repetition from being reported as optimization.

## Validation

Campaign 8 local validation:

```bash
uv run python -m pytest tests/test_rocm_campaign8_plan.py -q
uv run python scripts/run_rocm_campaign8_readiness_lane.py --write-evidence docs/benchmarks/data/rocm_campaign8_architecture_readiness_2026-05-01
uv run python -m cmake -S . -B /tmp/fastpauli-campaign8-cuda-hip-reject -DFASTPAULI_ENABLE_CUDA=ON -DFASTPAULI_ENABLE_HIP=ON
uv run python scripts/render_rocm_campaign8_assets.py --data-dir docs/benchmarks/data/rocm_campaign8_architecture_readiness_2026-05-01 --plot-dir docs/benchmarks/plots
```

For archive-only reruns with no `.git`, the readiness lane requires
`--source-commit <40-hex>` or the substituted `scripts/archive_source_identity.json`.
The lane fails closed if trusted source identity is absent.

Observed results:

```text
Campaign 8 plan tests: 4 passed
readiness evidence generation: passed; validation/configure checks are recorded as external closeout requirements rather than script-executed results
CUDA+HIP configure-time rejection: passed, nonzero configure exit with cannot-both-be-ON diagnostic
asset renderer: passed and preserved docs/benchmarks/plots/accelerator_landscape_with_rocm.svg
```

Full validation is run during closeout with:

```bash
uv run python scripts/validate.py
```

## Residual Risk And Next Triggers

Next ROCm work should start only when one of these triggers exists:

```text
non-MI300X AMD GPU access for the portability lane
ROCm wheel packaging infrastructure and CI hardware for packaging work
ROCm 7.x host with rocprofv3 available for side-by-side profiler migration
accepted backend-neutral implementation scope for simultaneous CUDA+HIP source builds
real ROCm consumer that can enforce HIP DLPack read-only behavior
profiler artifact showing a retained-operation bottleneck worth optimizing
```

Absent one of those triggers, another same-host MI300X release-support rerun is
not the next useful ROCm campaign.

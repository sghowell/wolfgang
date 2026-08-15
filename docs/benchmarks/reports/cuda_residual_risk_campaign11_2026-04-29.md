# CUDA Residual-Risk Campaign 11 Report

Date: 2026-04-29

Campaign 11 closes the two Campaign 10 residual-risk items that were in scope:
non-H100 Nsight Compute counter evidence on the existing A100 and RTX PRO 6000
Blackwell lanes, and nanobind reference-leak diagnostics from Compute
Sanitizer. This report does not add CUDA wheel claims, stream/CUDA Graph
claims, CSR scatter claims, or additional NVIDIA host lanes.

## Evidence Map

```text
plan: docs/plans/cuda_residual_risk_campaign11_plan.md
summary: docs/benchmarks/data/cuda_residual_risk_campaign11_2026-04-29/summary.json
raw data: docs/benchmarks/data/cuda_residual_risk_campaign11_2026-04-29/raw/
logs: docs/benchmarks/data/cuda_residual_risk_campaign11_2026-04-29/logs/
profiler exports: docs/benchmarks/data/cuda_residual_risk_campaign11_2026-04-29/profiler/
```

The checked Campaign 11 summary rejects deferred terminal states and covers
both residual-risk items on both in-scope hosts.

## Final Outcomes

| Residual-risk item | Host | Terminal status | Evidence | Limitation | Decision |
| --- | --- | --- | --- | --- | --- |
| Non-H100 Nsight Compute counters | A100 | blocked_permissions | `ncu_permission_probe_a100.log`, `ncu_compact_consumers_a100.log`, `raw/ncu_compact_consumers_a100.json` | `ncu` is available at `/usr/local/cuda/bin/ncu`, but non-root counter access returns `ERR_NVGPUCTRPERM`; sudo retry reaches `InterprocessLockFailed`. | Counter capture is blocked by host permissions for this campaign; no substitute host is used. |
| Non-H100 Nsight Compute counters | RTX PRO 6000 Blackwell | passed | `ncu_campaign11_compact_consumers_rtxpro6000blackwell.ncu-rep`, `.txt`, and benchmark raw JSON | `ncu` is available through the CUDA toolkit path rather than default `PATH`. | Counter capture passed and supports the Campaign 9/10 compact-consumer bottleneck model on `sm_120`. |
| Nanobind reference-leak diagnostics | A100 | rejected_with_evidence | `compute_sanitizer_memcheck_a100.log`, `nanobind_lifecycle_subprocess_a100.log` | Compute Sanitizer still prints nanobind process-teardown diagnostics, but reports zero CUDA memory errors. | Classified as Compute Sanitizer/nanobind teardown diagnostics, not a reachable runtime binding leak. |
| Nanobind reference-leak diagnostics | RTX PRO 6000 Blackwell | rejected_with_evidence | `compute_sanitizer_memcheck_rtxpro6000blackwell.log`, `nanobind_lifecycle_subprocess_rtxpro6000blackwell.log` | Compute Sanitizer still prints nanobind process-teardown diagnostics, but reports zero CUDA memory errors. | Classified as Compute Sanitizer/nanobind teardown diagnostics, not a reachable runtime binding leak. |

Campaign 11 explicitly excludes A10, L4, RTX 6000 Ada, and all other NVIDIA
hosts from this closure. Those can be planned as separate portability lanes if
the project later needs them.

## Hardware

| Host lane | SSH target | GPU | Compute capability | Requested architecture | Toolkit | Driver/runtime | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A100 | `ubuntu@<private-address>` | NVIDIA A100-SXM4-80GB | 8.0 | `80` | CUDA 12.8.93 | NVIDIA driver 580.126.09, CUDA driver API 13.0 / runtime 12.8 | CUDA subset passed; NCU counters blocked by permissions |
| RTX PRO 6000 Blackwell | `root@<private-address> -p 22` | NVIDIA RTX PRO 6000 Blackwell Server Edition | 12.0 | `120` | CUDA 12.8.93 | NVIDIA driver 580.126.09, CUDA driver API 13.0 / runtime 12.8 | CUDA subset passed; NCU counters captured |

Both hosts had `ncu` and `nvcc` outside default `PATH` and available under
`/usr/local/cuda/bin`. Host inventory logs record `hostname`, `uname -a`,
`nvidia-smi`, `nvidia-smi --query-gpu`, CUDA toolkit discovery, package search,
and the git revision used for remote evidence capture.

## Nsight Compute

The RTX PRO 6000 Blackwell lane captured Nsight Compute report and text export
artifacts for the retained compact consumer workload:

```text
docs/benchmarks/data/cuda_residual_risk_campaign11_2026-04-29/profiler/ncu_campaign11_compact_consumers_rtxpro6000blackwell.ncu-rep
docs/benchmarks/data/cuda_residual_risk_campaign11_2026-04-29/profiler/ncu_campaign11_compact_consumers_rtxpro6000blackwell.txt
```

The text export includes repeated `commutation_kernel` captures on `CC 12.0`.
Representative rows show about 20 microsecond launches for the dense
commutation kernel with high achieved occupancy and modest memory throughput,
plus compact consumer kernels where device-resident summary work remains much
cheaper than full dense or CSR materialization. This is consistent with the
Campaign 9/10 conclusion that retained wins come primarily from avoiding host
materialization and full edge-list export rather than from another small kernel
instruction edit.

The A100 lane could not capture counters. The non-root probe failed with
`ERR_NVGPUCTRPERM`. The sudo retry did not unlock a report; it failed with
`InterprocessLockFailed` while opening the Nsight Compute lock. The benchmark
program itself still ran and emitted raw JSON, so the blocker is profiler
counter access, not CUDA runtime correctness.

## Nanobind Lifecycle

Fresh Compute Sanitizer memcheck runs on both hosts passed the CUDA test file:

```text
A100 memcheck: 29 passed, 8 skipped, ERROR SUMMARY: 0 errors
RTX PRO 6000 Blackwell memcheck: 29 passed, 8 skipped, ERROR SUMMARY: 0 errors
```

Both sanitizer logs still print nanobind leaked instance/type/function
diagnostics at interpreter teardown. A targeted lifecycle subprocess now creates
a `PauliSum`, transfers it to `DevicePauliSum`, builds a
`DeviceCommutationMatrix`, consumes it, deletes all three wrappers, and forces
garbage collection. On both hosts the subprocess prints local refcounts of
`2 2 2` before cleanup and exits with `normal lifecycle clean`, without
nanobind leak output from the normal process.

The evidence classifies the sanitizer messages as process-teardown diagnostics
rather than a reachable runtime ownership leak. No binding ownership patch is
retained in Campaign 11.

## Validation

The initial full remote `scripts/validate.py` attempts failed because this
Campaign 11 summary/report had not yet been checked in, which was expected at
that point in the slice. The CUDA subset validation on both hosts passed after
source builds with explicit architectures:

```text
A100: FASTPAULI_CUDA_ARCHITECTURES=80, 35 passed, 8 skipped
RTX PRO 6000 Blackwell: FASTPAULI_CUDA_ARCHITECTURES=120, 35 passed, 8 skipped
```

Final remote validation passed on committed revision `f9e9e46`, and the
review-fix branch passed local validation on `08f51c7` after replacing the
checked validation logs with the final remote outputs:

```text
local macOS CPU validation: 195 passed, 59 skipped, plus benchmark and sdist smokes
A100 full CUDA validation: CPU pytest 194 passed / 60 skipped, CUDA-enabled pytest 216 passed / 38 skipped, CUDA kernel pytest 30 passed / 8 skipped, CUDA benchmark smoke passed, sdist smoke passed
RTX PRO 6000 Blackwell full CUDA validation: CPU pytest 194 passed / 60 skipped, CUDA-enabled pytest 216 passed / 38 skipped, CUDA kernel pytest 30 passed / 8 skipped, CUDA benchmark smoke passed, sdist smoke passed
```

## Review Closeout

Independent agent review found two blocking evidence issues before merge: the
checked A100 and RTX PRO 6000 Blackwell validation logs still contained early
expected failures, and `git diff --check` failed across the branch range because
of whitespace in captured evidence artifacts. Both were fixed before merge by
checking in the final passing remote validation logs and normalizing text
artifact whitespace. The reviewer rechecked `08f51c7`; the validation-log grep
showed only passing markers and build success, and
`git diff --check f035a0b1ce99d18da6aa63eb9b52b9168b879870..HEAD` exited clean.

## Residual Risk And Next Work

Campaign 11 leaves no in-scope residual-risk item without a terminal status.
The remaining risks are deliberate scope boundaries:

```text
A100 NCU counters require an infrastructure-level change to performance-counter permissions or the Nsight Compute lock path before a .ncu-rep can be captured.
A10, L4, RTX 6000 Ada, and other NVIDIA hosts remain unclaimed until a separate portability campaign is approved.
Future nanobind or CUDA binding changes should continue to run the lifecycle subprocess test and treat any normal-process leak output as a new bug.
CUDA wheels and release packaging remain separate from source-build validation.
```

Future CUDA work should start from Campaign 10 and Campaign 11 evidence rather
than reopening rejected stream/CUDA Graph, CSR scatter, or public grouping
surfaces by default.

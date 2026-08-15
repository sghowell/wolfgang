# CUDA Stream And Graph Campaign 10 Decision

Status: rejected with evidence for Campaign 10.

Campaign 10 reprofiled retained compact commutation consumers on A100 `sm_80`
and RTX PRO 6000 Blackwell `sm_120` before deciding whether to introduce public
stream-aware execution or CUDA Graph replay.

## Acceptance Contract

Public stream or CUDA Graph support is acceptable only after a separate API
review settles:

```text
stream ownership and lifetime across Python and C++
default-stream compatibility for existing synchronous APIs
event and synchronization responsibilities
error propagation for asynchronous CUDA failures
interaction with DLPack and CUDA Array Interface consumers
capture-safe allocation and workspace behavior
documentation of timing boundaries and unsupported operations
```

Even with that contract, Campaign 10 required profiler evidence that launch or
graph replay overhead is a dominant cost for a retained public or private
consumer. The campaign threshold was not met.

## Campaign 10 Profiler Evidence

Nsight Systems was available on both non-H100 hosts. Nsight Compute was not
installed on either host, so Campaign 10 retained the Campaign 9 H100 privileged
Nsight Compute counter evidence as the latest counter-level source and recorded
non-H100 evidence through Nsight Systems plus Compute Sanitizer.

```text
A100 cudaLaunchKernel API time: 0.5% of CUDA API time, 2.326822 ms across 154 calls
A100 dominant CUDA API costs: cudaMemcpy 36.2%, cudaMalloc 30.5%, cudaHostRegister 24.1%
A100 GPU memory time: 99.7% Device-to-Host

RTX PRO 6000 Blackwell cudaLaunchKernel API time: 0.8% of CUDA API time, 2.039050 ms across 154 calls
RTX PRO 6000 Blackwell dominant CUDA API costs: cudaMalloc 35.4%, cudaMemcpy 31.1%, cudaHostRegister 21.7%
RTX PRO 6000 Blackwell GPU memory time: 99.8% Device-to-Host
```

The retained compact consumer path is dominated by allocation, registration,
copy, and materialization boundaries rather than launch overhead. CUDA Graph
replay would not address those dominant costs without a broader capture-safe
workspace and public lifetime contract.

## Decision

Do not add public stream-aware execution or CUDA Graph replay in Campaign 10.

Future work may reopen this only after a stream/lifetime API contract is
accepted and fresh profiler evidence shows launch or replay overhead dominates a
retained consumer after allocation and host materialization are already removed
from the measured boundary.

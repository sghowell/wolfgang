# CUDA Portability Campaign 9 Non-H100 NVIDIA Report

Date: 2026-04-29

Status: `blocked_external`

Campaign 9 required one named non-H100 NVIDIA source-build validation run from
the target set `a100_sm80`, `rtx6000ada_sm89`, `l4_sm89`, or `a10_sm86`. No
usable non-H100 NVIDIA host was available during this execution.

Provider and instance metadata for a non-H100 NVIDIA target was not available
to the agent. The only non-H100 candidate input was an SSH endpoint supplied by
the user; no provider control-plane access, non-H100 NVIDIA instance type, or
successful provisioning receipt was provided.

## Access Evidence

The H100 control host remained available:

```text
host: ubuntu@<private-address>
hostname: 0151-dsm-prxmx30065
gpu: NVIDIA H100 PCIe
compute capability: 9.0
driver: 580.126.09
```

The only additional reachable host was not a valid non-H100 NVIDIA target:

```text
host: ubuntu@<private-address>
hostname: glorious-tremendous-pheasant
provider: not available to agent; user supplied SSH endpoint only
instance type: not available to agent; user supplied SSH endpoint only
nvidia-smi path: /usr/bin/nvidia-smi
exit code: 9
result: NVIDIA-SMI could not communicate with the NVIDIA driver.
```

## Decision

Non-H100 NVIDIA portability is closed for Campaign 9 as
`blocked_external`, not deferred. No non-H100 performance or compatibility
claim is broadened from the H100 evidence.

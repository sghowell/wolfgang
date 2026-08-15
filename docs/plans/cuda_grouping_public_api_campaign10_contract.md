# CUDA Grouping Public API Campaign 10 Contract

Status: rejected with evidence for Campaign 10.

Campaign 10 rechecked whether FastPauli should expose a true public CUDA
grouping API on top of the device-resident commutation matrix work retained in
Campaigns 7 through 9. The accepted public surface remains
`DeviceCommutationMatrix.conflict_degrees(axis=None|0|1)`.

## Acceptance Contract

A true public CUDA grouping API is acceptable only when all of the following
contracts are settled before implementation:

```text
return type: exact Python object shape, ownership, and host/device residence
ordering: deterministic group order and deterministic term order inside groups
semantics: qwc and full commuting modes match PauliSum.group_commuting()
lifetime: outputs do not depend on temporary DeviceCommutationMatrix internals
materialization: large dense host transfers are avoided or explicitly requested
error model: allocation, device mismatch, and unsupported mode errors are clear
documentation: user-facing docs explain timing boundaries and memory behavior
tests: CPU/GPU equivalence, determinism, empty inputs, and memory guards pass
```

The implementation must also prove that it improves a retained user workflow
rather than only optimizing a private benchmark boundary.

## Campaign 10 Evidence

Campaign 10 measured the currently retained compact public summary surface on
A100 `sm_80` and RTX PRO 6000 Blackwell `sm_120` source builds.

```text
A100 conflict_degrees(axis=None): 0.000090925 s
A100 dense_to_host_plus_numpy_conflicts: 0.006950354 s
RTX PRO 6000 Blackwell conflict_degrees(axis=None): 0.000049529 s
RTX PRO 6000 Blackwell dense_to_host_plus_numpy_conflicts: 0.002591294 s
```

The compact summary path is already public, deterministic, device-resident until
the scalar/vector result boundary, and significantly cheaper than exporting the
dense matrix and regrouping on the host for the measured Campaign 10 workload.
No accepted ownership, ordering, or return-shape contract exists for a richer
public device grouping result.

## Decision

Do not add a true public CUDA grouping API in Campaign 10.

Future work may reopen this only with a concrete consumer and an accepted API
stability document that specifies result ownership, deterministic ordering,
materialization behavior, and compatibility with the CPU
`PauliSum.group_commuting()` contract.

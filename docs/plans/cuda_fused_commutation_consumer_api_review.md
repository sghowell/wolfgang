# CUDA Fused Commutation Consumer API Review

Status: benchmark-only fused consumers first; public fused API deferred unless
Campaign 7 evidence accepts exact return semantics.

## Existing Invariant

`DeviceCommutationMatrix` owns dense row-major `uint8` flags on one CUDA device.
The public meaning remains stable: `1` means commuting and `0` means
anti-commuting. Public methods continue to use the default stream and
synchronize before Python observes a result.

## Candidate Consumers

Primary fused consumer: anti-commutation CSR graph construction from
`DeviceCommutationMatrix`.

Secondary fused consumer: row and column conflict-degree summaries.

Grouping-oriented consumer: deterministic compact conflict summary for greedy
grouping preparation.

Benchmark-only private method:
`fastpauli._fastpauli_core._benchmark_cuda_fused_commutation_consumer`.

Allowed benchmark modes:

```text
csr_anticommutation_graph
conflict_degrees
grouping_summary
bitpacked_ab
```

The private hook is allowed only from benchmark scripts and CUDA-gated tests. It
must never be re-exported from `fastpauli.__init__` or described as public user
API.

## Public API Decision

Public fused graph or grouping APIs are rejected for this slice by default. A
public method can be reconsidered only after Campaign 7 evidence defines and
accepts every field below:

```text
exact Python method name and C++ method name
return type and shape
commuting or anti-commuting edge convention
stable ordering of returned edges, rows, groups, or summaries
device and stream synchronization semantics
host copy size and transfer boundary
CPU-only error behavior
moved-from object behavior
memory allocation limit and failure mode
correctness oracle against a CPU reference
benchmark labels for fill, fused consumer, compact host copy, and dense to_host()
```

Until that review is complete, benchmark helpers may use private C++ helpers in
`src/cuda/device_commutation_matrix.*`, but no declaration belongs in installed
public headers.

## Memory Ownership

FastPauli owns every device buffer allocated by the benchmark helpers. The
helpers may allocate temporary row counts, row offsets, column indices, or
compact summaries on the matrix device. They must free those allocations before
returning and must synchronize before any host-visible result, checksum, or
timing boundary is observed.

No raw device pointer API is accepted by this review. CUDA Array Interface
exposure remains limited to the existing dense `DeviceCommutationMatrix`
representation.

## Ordering

CSR graph output uses row-major ordering by left-hand-side row, then
right-hand-side column. Within each row, `col_indices` are sorted ascending.

Conflict-degree summaries preserve the original deterministic term ordering.
Grouping-oriented top-k summaries sort by descending conflict count and use the
original index as the deterministic tie-breaker.

## Correctness Oracle

Small CUDA tests validate private outputs against CPU extraction from
`matrix.to_host()`.

CSR convention:

```text
row_offsets length = rows + 1
row_offsets[0] = 0
row_offsets[i + 1] - row_offsets[i] = anti-commuting entries in row i
col_indices contain columns where DeviceCommutationMatrix[row, col] == 0
```

Conflict-degree convention:

```text
row_conflicts[i] = cols - matrix.count_commuting(axis=1)[i]
col_conflicts[j] = rows - matrix.count_commuting(axis=0)[j]
```

Grouping summaries must remain compatible with existing CPU grouping
correctness tests before they can be promoted beyond benchmark evidence.

## Failure Modes

Required failure behavior:

```text
CPU-only build, benchmark mode: return status="unavailable" with rebuild guidance
CPU-only build, CUDA-required test mode: raise the existing CUDA rebuild-guidance RuntimeError
CUDA build with missing matrix: raise ValueError
unsupported mode: raise ValueError
moved-from matrix: raise the existing DeviceCommutationMatrix moved-from RuntimeError
allocation failure: propagate CUDA allocation failure with context
unsupported public API: no public method is exposed
```

## Benchmark Labels

Campaign 7 benchmark rows must distinguish:

```text
fill allocation path
fill reuse path
fused CSR graph
fused conflict degrees
fused grouping summary
compact host copy
full dense to_host
CuPy dense consumer
CPU scalar extraction
best available optimized CPU extraction
```

Reports must label the private hook as `private_benchmark_only` and must carry
`count_specialization_status`, `bitpacked_decision_status`, and
`portability_gpu` fields so decisions remain visible even on non-CUDA machines.

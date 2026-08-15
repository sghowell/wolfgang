# CUDA Commutation Consumer API Review

Status: retain public compact-summary counts; keep bit-packed output private and deferred.

## Existing Invariant: DeviceCommutationMatrix Owns Dense Row-Major Uint8 Flags

`DeviceCommutationMatrix` owns one dense C-contiguous row-major `uint8` CUDA
buffer. Each entry is `1` when the corresponding `(lhs_term, rhs_term)` pair
commutes and `0` otherwise. The allocation is owned by FastPauli, exported
through the CUDA Array Interface, and materialized on the host only when
`to_host()` is called.

Campaign 6 keeps this dense matrix as the primary public device-output storage.

## Compact-Summary Candidate Methods

Campaign 6 retains one Python method and three C++ methods:

```python
matrix.count_commuting(axis=None)
matrix.count_commuting(axis=0)
matrix.count_commuting(axis=1)
```

```cpp
std::uint64_t DeviceCommutationMatrix::count_commuting() const;
std::vector<std::uint64_t> DeviceCommutationMatrix::count_commuting_rows() const;
std::vector<std::uint64_t> DeviceCommutationMatrix::count_commuting_cols() const;
```

These methods are narrow downstream GPU consumers: reductions execute on the
owning CUDA device and copy only compact `uint64` count results back to the
host. They do not expose raw pointers, layouts beyond the existing dense
matrix, or asynchronous behavior.

## Compact-Summary Return Types

Return types are:

```text
axis=None -> Python int, total count of entries equal to 1
axis=0 -> NumPy uint64 array with shape (matrix.cols,)
axis=1 -> NumPy uint64 array with shape (matrix.rows,)
```

C++ returns `std::uint64_t`, `std::vector<std::uint64_t>` over rows, and
`std::vector<std::uint64_t>` over columns.

## Compact-Summary Synchronization Semantics

The compact-summary API follows the existing public CUDA invariant:

```text
launch reduction kernels on the default stream
copy compact results to host
synchronize before returning through the blocking copy
raise CUDA errors before returning
```

## CPU-Only Behavior

CPU-only builds still import `fastpauli` successfully. Calling
`DeviceCommutationMatrix.empty(...)`, `to_host()`, CUDA-array-interface export,
or `count_commuting(...)` on a CPU-only build raises the existing CUDA
rebuild-guidance `RuntimeError`.

## Benchmark-Only Fallback Rules

The retained compact-summary API is public and must be benchmarked as a
supported synchronous method. Benchmark rows must still distinguish it from
full dense `to_host()` materialization by reporting compact-result copy sizes
and separate total, row, and column timing fields.

## Bit-Packed Layout Candidate

The only bit-packed layout considered for future work is:

```text
row-major uint64 words over rhs terms
one bit per rhs entry
word count per row = ceil(cols / 64)
unused high bits in the final word of each row are zero
bit value 1 means commuting, bit value 0 means anti-commuting
```

## Bit-Packed Public API Rejection Or Retention Criteria

Campaign 6 rejects public bit-packed output. It can be revisited only if
consumer benchmarks show dense `uint8` capacity or bandwidth is the limiting
cost and the consumer can directly operate on the packed layout without
immediately unpacking to dense flags.

Any public bit-packed API would require a separate API plan covering layout,
shape, dtype, CUDA Array Interface or DLPack representation, synchronization,
host materialization, and migration from dense `DeviceCommutationMatrix`.

## CuPy Consumer Boundary

CuPy benchmark rows consume `DeviceCommutationMatrix` through
`__cuda_array_interface__`. They are interop consumer timings, not FastPauli
kernel timings. CuPy availability, version, import/runtime errors, and
device-to-host materialization boundaries must be recorded in raw benchmark
JSON and reports.

## Retained Public Surfaces

```text
DeviceCommutationMatrix.count_commuting(axis=None)
DeviceCommutationMatrix.count_commuting(axis=0)
DeviceCommutationMatrix.count_commuting(axis=1)
```

## Rejected Public Surfaces

```text
raw device pointer accessors beyond CUDA Array Interface metadata
public bit-packed commutation output
public caller-owned raw device output buffers
public asynchronous compact-summary methods
```

## Deferred Public Surfaces

```text
bit-packed DeviceCommutationMatrix variant
DLPack export
larger downstream GPU algorithms that consume the matrix without host materialization
stream-aware compact reductions
```

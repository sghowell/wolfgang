# CUDA Commutation Device-Output API Review

Status: accepted for Campaign 5 experimental public API.

Campaign 4 benchmarked private commutation output materialization strategies
without promoting a device-output API. Public CUDA commutation behavior remains:

```text
DevicePauliSum.commutes_with() returns a bool or host NumPy bool array
DevicePauliSum.commutes_with_into() fills a caller-owned one-dimensional host bool array
outputs are row-major over lhs terms, then rhs terms
commutes_with enforces the dense-output max-entry guardrail before allocation
```

Campaign 5 promotes one device-output API:

```text
owned FastPauli DeviceCommutationMatrix object
dense uint8 row-major device buffer
CUDA-array-interface export for downstream GPU consumers
```

Caller-owned raw device pointers and public bit-packed output are rejected for
Campaign 5. Raw device pointers do not provide enough ownership and lifetime
structure for a first public surface, and bit-packed output still lacks a
documented consumer layout contract. Operands and device outputs must live on
the same CUDA device. The synchronization contract remains default-stream
synchronize-before-return unless a separate async API plan is accepted.

Required errors for a public design:

```text
FastPauli built without CUDA
CUDA runtime unavailable
moved-from operands
wrong operand or output device
wrong output dtype
wrong output shape
dense output exceeds max_commutation_matrix_entries
device allocation failure
```

Campaign 4 benchmark labels were:

```text
host_vector: supported public vector-return host path
caller_owned_host_bytes: supported public host output buffer path
caller_owned_device_bytes: private prototype label only
bitpacked_device_words: private prototype label only
```

Private prototype rows may appear in benchmark JSON and reports, but README and
user-facing docs must not present private labels as supported APIs.

## Campaign 5 Accepted API

Campaign 5 promotes the dense device-output path only with architecture docs,
API stability docs, tests, benchmarks, and user docs in the same slice.

The public Python surface is:

```python
matrix = lhs_device.commutes_with_device(
    rhs_device,
    max_commutation_matrix_entries=100_000_000,
)

output = fastpauli.DeviceCommutationMatrix.empty(
    shape=(lhs_device.num_terms, rhs_device.num_terms),
    device=lhs_device.device,
)
same = lhs_device.commutes_with_device(rhs_device, output=output)
assert same is output

host = matrix.to_host()
cuda_view = matrix.__cuda_array_interface__
```

The public C++ surface is:

```text
DeviceCommutationMatrix::empty(rows, cols, device)
DeviceCommutationMatrix::to_host()
DeviceCommutationMatrix::rows()
DeviceCommutationMatrix::cols()
DeviceCommutationMatrix::num_entries()
DeviceCommutationMatrix::device()
DevicePauliSum::commutes_with_device(rhs, max_entries)
DevicePauliSum::commutes_with_device_into(rhs, output, max_entries)
```

Required semantics:

```text
DeviceCommutationMatrix owns dense row-major uint8 flags on one CUDA device
1 means commuting and 0 means anti-commuting
shape is exactly (lhs_terms, rhs_terms)
to_host() returns a NumPy bool matrix with the same shape
__cuda_array_interface__ exposes the dense uint8 buffer with typestr "|u1"
all operands and output storage must live on the same CUDA device
commutes_with_device enforces the dense-output max-entry guardrail before
allocating its own output or filling caller-provided output
CPU-only builds keep import working and raise CUDA rebuild-guidance RuntimeError
methods synchronize before returning unless a later stream API review changes that contract
```

Campaign 5 must continue to reject public bit-packed output until a consumer
layout contract exists. Private bit-packed timing rows may remain in benchmark
reports only when visibly labeled as private benchmark evidence.

The retained API is explicitly experimental before `1.0.0`. Compatible
extensions may add stream-aware, bit-packed, or external-device-pointer
variants later, but those variants require their own API review before exposure.

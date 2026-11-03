# Zero-Copy Device Arrays On CUDA

Date: 2026-11-03

This report qualifies slice 3 of the portability and interop campaign
(`docs/plans/portability_and_interop_campaign_plan.md`):

- `DevicePauliSum.device_arrays()` returns views of a CUDA operator's x words,
  z words and coefficients, flagged read-only. CuPy consumes them through
  DLPack or the CUDA Array Interface and PyTorch through DLPack, without a
  copy, and each view keeps the operator alive.
- `DevicePauliSum.from_device_arrays(x, z, coeffs, num_qubits)` builds a CUDA
  operator from CUDA device arrays with a device-to-device copy, with no host
  round trip. It synchronizes the device before copying, so producers on any
  stream have finished, and returns after the copy completes.

HIP and Metal raise for `device_arrays()`, and builds without CUDA raise for
`from_device_arrays()`, until a consumer on those backends is validated.

## Evidence

```text
benchmark: benchmarks/bench_device_arrays.py
summary:   docs/benchmarks/data/device_arrays_h100_2026-11-03/summary.json
per-run reports: private archive (docs/benchmarks/data/**/raw/ is not published)
```

| Item | Value |
| --- | --- |
| Device | NVIDIA H100 80GB HBM3, compute capability 9.0 |
| CUDA | runtime 12.8 (nvcc 12.8.93), driver 13.0, compiled for sm_90 |
| Host | Intel Xeon Platinum 8480+, Ubuntu 24.04, GCC 13.3, Python 3.12.3 |
| CPU backend | scalar (oneTBB 2021.11 compiled in, not used here) |
| Consumers | CuPy 14.2.0; PyTorch 2.15.0 (tests only) |
| Commit | `01997f7`, clean tree, loaded extension matching the checkout's native sources |

Command, 3 runs with seeds 20261111-20261113, 3 warmups and 15 timed
repetitions, variants in a randomized order per repetition, each variant ending
with a device synchronization so a timing covers its own GPU work:

```text
python benchmarks/bench_device_arrays.py --seed <seed> --output <run>
```

Each run first checks that every variant returns the same arrays or operator.

## Results

Random dense operators, random coefficients. Medians of per-run medians, with
the min-max of per-run medians.

Export, `device_arrays()` + `cupy.from_dlpack` against
`to_host().to_arrays()` + `cupy.asarray`:

| Qubits | Terms | Zero-copy export | Through the host | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 1,000 | 15.7 us (15.6-16.5) | 100 us (100-101) | 6x |
| 64 | 100,000 | 31.0 us (30.5-31.7) | 1.08 ms (1.02-1.15) | 35x |
| 64 | 1,000,000 | 36.2 us (36.1-39.9) | 8.83 ms (8.40-9.11) | 244x |
| 100 | 1,000 | 15.9 us (15.8-16.5) | 109 us (109-110) | 7x |
| 100 | 100,000 | 31.7 us (31.6-32.5) | 1.52 ms (1.52-1.53) | 48x |
| 100 | 1,000,000 | 39.6 us (39.3-44.1) | 12.4 ms (12.0-13.4) | 314x |

Import, `from_device_arrays` against `cupy.asnumpy` + `PauliSum.from_arrays`
+ `to_device`:

| Qubits | Terms | Device copy | Through the host | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 1,000 | 33.8 us (32.7-35.0) | 80.4 us (80.0-81.2) | 2x |
| 64 | 100,000 | 54.3 us (53.9-55.7) | 927 us (913-943) | 17x |
| 64 | 1,000,000 | 83.7 us (83.5-89.2) | 20.7 ms (20.0-20.7) | 247x |
| 100 | 1,000 | 50.3 us (47.3-53.0) | 89.0 us (88.9-90.2) | 2x |
| 100 | 100,000 | 84.9 us (83.9-85.3) | 1.41 ms (1.39-1.41) | 17x |
| 100 | 1,000,000 | 131 us (127-132) | 27.6 ms (27.5-28.4) | 211x |

Export cost barely grows with size: three capsule handoffs and a device
synchronization. Import copies 32 bytes per term at 64 qubits (48 at 100) and
synchronizes twice. At 100 qubits it also counts terms with bits above
`num_qubits` on the device, which accounts for most of the gap to the 64-qubit
rows.

An earlier run of this benchmark timed `from_device_arrays` before the copy
had finished. With 64 qubits no tail check ran, nothing synchronized, and the
pending work was billed to whichever variant synchronized next. Review caught
this; the timings above are from the corrected benchmark and the
stream-safe import.

## Correctness

| Check | Result |
| --- | --- |
| `tests/test_device_arrays.py` | 16 passed on the H100 with CuPy 14.2.0 and PyTorch 2.15.0. Coverage: zero-copy views at 5, 64 and 130 qubits, read through DLPack and the CUDA Array Interface at the same device pointer; DLPack and CUDA Array Interface consumers keep the operator alive; zero-term and zero-width export and import; round trips at 3, 64, 100 and 200 qubits; an import waits for a producer on a non-blocking stream and finishes before returning; dtype, shape, dimension, contiguity, mask, null-pointer, pinned-host-pointer, device-ordinal, oversized-`num_qubits` and tail-bit errors; managed memory accepted; `__dlpack__` `max_version`, `copy` and `dl_device` handling; PyTorch `from_dlpack` reads, and PyTorch's rejection of read-only CUDA Array Interface input; producer attribute errors propagate. |
| Negative control | With the two synchronizations removed, both stream tests fail. |
| HIP (MI300X) | `device_arrays()` raises, and `from_device_arrays()` raises `RuntimeError` naming the CUDA build. |
| `WOLFGANG_VALIDATE_CUDA=1 python scripts/validate.py --profile core` | Passed on the H100 at `9e0194e` (944 tests in the CUDA run); `01997f7` changes only the tail-check reduction. |
| `compute-sanitizer` | memcheck, racecheck and synccheck at `9e0194e`, and initcheck and memcheck at `01997f7`: 0 errors and 0 hazards on the device-array tests, apart from the PyTorch test and the two timing-based stream tests. |

## Limitations

- One H100, CuPy as the timed consumer, and PyTorch in tests only.
- The read-only flag is advisory. CuPy and PyTorch import without enforcing
  it, and writing through a consumer array is undefined behavior.
- After an export, freeing the operator synchronizes the device, possibly
  during garbage collection; see `docs/architecture/cuda_backend.md` for the
  consequences. Operators that are never exported free without synchronizing.
- Import synchronizes the whole device before copying, which also waits for
  unrelated work on other streams.
- Import accepts the CUDA Array Interface only, not DLPack input.

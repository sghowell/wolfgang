# CUDA Simplify With Device-Side Counts

Date: 2026-11-03

This report qualifies slice 1 of the accelerator follow-ups campaign
(`docs/plans/accelerator_followups_campaign_plan.md`). The slice 9 profile of
the previous campaign (`docs/benchmarks/reports/cuda_small_op_profile_h100_2026-11-03.md`)
found that simplify above the fused kernel's limit, and `matmul` through it,
stopped eight times per call to read a value back to the host: the largest
coefficient magnitude, the duplicate-run count and the survivor count, each
needed to size the next Thrust stage.

Simplify for those operators now runs a pipeline in
`src/detail/device_simplify_pipeline.cuh`, a header that builds under nvcc and
hipcc so the HIP backend can share it (slice 2):

1. one CUB reduction computes the largest magnitude (NaN when a component is
   not finite) and the largest half magnitude;
2. a pack kernel writes the sort keys, and its first thread turns the
   reduction into the CPU drop rule on the device;
3. the keys are sorted. One-word operators of at most 32 qubits radix-sort a
   packed key of `2 * num_qubits` bits (48 bits at 24 qubits, where the old
   path sorted all 64), other one- and two-word operators keep Thrust's
   `stable_sort_by_key`, and wider operators sort term indices;
4. CUB reduce-by-key sums each duplicate run and leaves the run count in
   device memory;
5. an inclusive scan over "this run survives" flags assigns output positions.
   It runs over the full input length and flags runs past the device-side run
   count as 0, so it needs no count from the host;
6. a one-thread kernel copies the counts and status flags to pinned,
   device-mapped host memory.

The host waits once, raises `ValueError` or `OverflowError` if a flag is set,
allocates exactly the surviving terms, and one kernel scatters them into the
output. The four specialized Thrust paths it replaces are gone; the two
benchmark-only baselines (`cub_radix_sort_reduce` and the one-thread
`serial_kernel`) keep their host-side counts.

The slice also fixes a small leak the sanitizer run exposed. Every thread
that ran simplify kept its pinned result buffers until the process exited.
They are now freed when the thread exits.

## Evidence

```text
benchmarks: benchmarks/bench_cuda_call_overhead.py, benchmarks/bench_cuda_simplify_paths.py --suite all
profile:    Nsight Systems 2024.6.2 (--trace=cuda), 50 warmup + 500 profiled calls per operation
data:       docs/benchmarks/data/cuda_device_counts_h100_2026-11-03/overhead_summary.json
            docs/benchmarks/data/cuda_device_counts_h100_2026-11-03/paths_summary.json
            docs/benchmarks/data/cuda_device_counts_h100_2026-11-03/api_breakdown.json
```

| Item | Value |
| --- | --- |
| Device | NVIDIA H100 80GB HBM3, sm_90, driver 580.105.08 (CUDA 13.0), toolkit 12.8.93 (CCCL 2.7) |
| Host | Intel Xeon Platinum 8480+, Ubuntu 24.04, GCC 13.3, Python 3.12.3 |
| Baseline | main at `ad6f7a8`. Its checkout also held this slice's new test file, copied in for the sanitizer comparison; the recorded native source hash matches `ad6f7a8` |
| Candidate | `6ddf58b` |
| Build | `WOLFGANG_CUDA_ARCHITECTURES=90`, pool enabled, CPU backend auto (scalar for these rows) |

The call-overhead suite ran 7 times and the simplify-path suite 3 times
(seeds 20261201-20261203), with the baseline and candidate order randomized
per run.

## Results

Call overhead, device-resident 24-qubit operators. Medians of per-run medians,
with the min-max of per-run medians:

| Operation | Baseline | Candidate | Speedup |
| --- | ---: | ---: | ---: |
| `simplify`, 4,096 terms | 188.2 us (187.3-190.0) | 108.3 us (105.8-108.6) | 1.74x |
| `matmul`, 128 x 128 terms | 202.0 us (200.2-205.9) | 123.3 us (119.3-123.7) | 1.64x |

The other thirteen rows, which do not reach the pipeline (fused simplify at
128 and 1,024 terms, single-term and wide-operator simplify, transfers,
expectation, commutation and the CPU control), stayed within 0.99-1.00x.

Simplify paths, 50% duplicates for the specialized cases, median of three
paired runs:

| Path | Terms | Device-resident | Including transfers |
| --- | ---: | ---: | ---: |
| One word, packed key (24 qubits) | 50,000 | 0.262 -> 0.166 ms (1.58x) | 1.15x |
| One word, packed key (24 qubits) | 1,000,000 | 0.535 -> 0.388 ms (1.38x) | 1.01x |
| One word, two masks (60 qubits) | 50,000 | 0.242 -> 0.168 ms (1.44x) | 1.09x |
| One word, two masks (60 qubits) | 1,000,000 | 0.753 -> 0.667 ms (1.13x) | 0.99x |
| Two words (100 qubits) | 50,000 | 0.273 -> 0.192 ms (1.43x) | 1.06x |
| Two words (100 qubits) | 1,000,000 | 1.181 -> 1.090 ms (1.08x) | 1.01x |

Operators wider than two words (130, 256 and 1,024 qubits) gained 1.30-1.85x
at 128 to 65,536 distinct terms and 1.11-1.37x with 90% duplicates. Single-term
rows were unchanged. Transfer-inclusive timings at 1M terms are dominated by
pageable host-device copies (5-9 ms), which slice 3 addresses.

## Where the time goes

Per `simplify` call on 4,096 terms (24 qubits), from the Nsight Systems
breakdown:

| | Baseline | Candidate |
| --- | ---: | ---: |
| Stream synchronizations | 8 | 1 |
| Device-to-host copies | 5 (3 on the GPU timeline) | 0 |
| Pool allocations | 13 | 4 |
| Kernel launches | 19 | 17 |
| Memsets (radix sort) | 10 | 8 |
| GPU kernel time | 65.8 us | 60.1 us |
| Wall time | 189.0 us | 107.0 us |

`matmul` on 128 x 128 terms shows the same pattern: 8 synchronizations and 5
copies become 1 and 0, and 16 pool allocations become 7.

The remaining cost is now set by the GPU work and the launches that feed it,
not by round trips. The radix sort takes 32 us of the 60 us: six onesweep
passes of about 5.3 us each, which at 4,096 keys are bound by latency rather
than bandwidth. The 17 launches and 8 memsets cost about 65 us of host time,
and the GPU finishes about 29 us after the last one is queued. A CUB select in
place of the survivor scan measured the same in isolation (8.6 against 8.8 us
at 4,096 items, 11.8 against 12.3 us at 1M), and the scan has no 2^31-item
limit, so the scan stays.

## Correctness

| Check | Result |
| --- | --- |
| `tests/test_device_simplify_pipeline.py` (new) | 87 CUDA cases with the fused kernel switched off, so inputs from 2 terms up reach the pipeline. They cover every key kind at 1, 5, 31, 32, 33, 64, 65, 128, 129 and 300 qubits; full cancellation and a single survivor; duplicate-sum overflow; non-finite input taking precedence over an overflowing run at the first, middle and last position; the half-unit drop rule; the separately rounded threshold; error calls followed by valid ones; 32 calls across four threads; and `matmul`. All pass on the baseline too, as regression tests should |
| Existing simplify suites | `test_accelerator_simplify_parity.py`, `test_accelerator_small_simplify.py`, `test_cuda_generic_simplify.py`, `test_cuda_pool_allocator.py`: all pass |
| `WOLFGANG_VALIDATE_CUDA=1 python scripts/validate.py --profile core` | See the closeout |
| `compute-sanitizer` on the new tests | racecheck and synccheck: 0. memcheck: no errors besides leak reports, 17 leaked allocations in each build. Five were per-thread result buffers (4-byte flags in the baseline, 24-byte results in the candidate), now freed at thread exit: after the fix, the threads test reports none. The other 12 are the test's device operators, still alive when the sanitizer tears down the context, in both builds. initcheck: every report is inside a reduce-by-key kernel, 84,360 in CUB's (candidate) against 30,096 in Thrust's (baseline); none in Wolfgang's kernels or the scan |

The initcheck reports are the decoupled look-back pattern recorded in
`docs/benchmarks/reports/cuda_commutation_pool_h100_2026-11-03.md`: a tile
reads its predecessors' partial sums speculatively, including padding slots
that are never written, and discards them. CUB's reduce-by-key reads a
warp-wide window of predecessors, hence more reports than Thrust's.

## Numerics

Canonical order, the drop rule and the error contract are unchanged. CUB's
reduce-by-key associates long duplicate runs in a different tree than Thrust's,
which `docs/architecture/cuda_backend.md` allows (implementation-defined
parallel order within dtype tolerance); coefficients still match CPU simplify
to 1e-11 in the tests.

## Limitations

- One H100 and one toolkit (CCCL 2.7). Kernel choices inside CUB (onesweep
  versus single-tile radix sort, tile sizes) depend on the CCCL version.
- Operators between the fused limit (1,024 terms) and roughly 64K terms pay a
  near-constant 100-120 us, set by sort latency and launch count. A multi-item
  fused kernel for that range is a candidate follow-up; on HIP the MI300X's
  64 KiB of LDS per workgroup limits how far the fused kernel can grow.
- The pipeline holds keys, sorted keys or values, run keys, run sums and
  positions for the full input at once, about the same peak as the Thrust
  path it replaces.

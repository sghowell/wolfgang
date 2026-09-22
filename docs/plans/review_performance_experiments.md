# Review Performance Experiments

This is the next measured-work queue after repository hardening. Candidates must
pass correctness and small-workload guards before promotion. The hardening work
changes no CUDA/HIP kernels and makes no new GPU speedup claim.

| Candidate | Concrete implementation scope | Qualification dataset / control |
| --- | --- | --- |
| CUDA generic simplify | Replace the one-thread generic reduce stage with a parallel segmented reduction; use the HIP generic reduction and permutation iterators as a design reference, preserving canonical key and input-order sums | 130/256/1024 qubits; 0/1/128/4096/65536 terms; low/high duplicates; cancellation and finite-limit oracle; compare current CUDA generic and CPU |
| Compact total counts | Reduce device block sums to one scalar before host transfer; reuse private scratch and retain the existing path as a baseline | Empty, rectangular, all-zero/all-one and random matrices; 128 through 10000 rows; scalar count equality; measure transfer and allocation boundaries separately |
| Direct count and degree consumers | Fuse commutation and total/axis reductions to avoid the quadratic matrix when callers only need counts | Compare with existing matrix plus count pipeline; test both axes, symmetry, overflow bounds and multiple packed widths |
| CPU reusable outputs | Add span-based commutation kernels and one Python-owned writable output buffer; preserve dtype/shape/alias checks | Scalar/SIMD equivalence and reused versus allocating output; include small matrices and lifetime checks |
| CPU dispatch metadata | Cache immutable hardware detection while preserving environment-selector validation per call; separate hot selection from report construction | Small commutation/grouping rows, unchanged constructor control, forced backend errors; randomized paired runs |
| Dense diagonal expectations | Explore a Walsh-Hadamard transform only when density justifies it; preserve the sparse direct oracle and explicit memory bound | Vary qubits, term density and count sparsity; compare setup-inclusive and reused-transform cases |

For CUDA/HIP, record toolkit, driver, architecture, source/binary fingerprints,
workspace use, command and raw samples. Run the existing backend validation entrypoint
with `WOLFGANG_VALIDATE_CUDA=1` or `WOLFGANG_VALIDATE_HIP=1`, native sanitizer or
compute-sanitizer coverage, and tests around launch/count narrowing boundaries.
A CPU or Metal run does not qualify these candidates. No new cloud compute is
acquired by this plan.

For local Python-boundary comparisons, run `benchmarks/bench_python_boundary.py`
in baseline and candidate environments, alternating their process order with a
fixed random seed across at least seven paired reruns. Retain the individual
samples, medians, artifact hashes and unchanged construction control. A shared
movement in the control invalidates attribution and requires another run.

Historical CPU Wave 1B evidence remains archived as recorded. Its unchanged
controls moved substantially, so the earlier percentage changes are not sufficient
to isolate the dispatch implementation's effect. Requalifying dispatch is separate
from the measured coefficient-export change.

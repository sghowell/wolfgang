# Python Boundary Hardening — 2026-09-21

Coefficient export for 100,000 one-qubit terms decreased from 3.558 ms to
1.002 ms in this local comparison (71.8%). The unchanged construction control
moved by -1.1%. This supports the bundled boundary/build improvement on this
machine; it does not isolate the effect of memcpy from the CMake target change.

[Samples, rerun order and artifact fingerprints](../data/python_boundary_hardening_2026-09-21.json)
are retained for all 9 paired reruns, each with 21 samples. The deterministic
harness is `benchmarks/bench_python_boundary.py`. Run it in each environment with
`--repeat 21 --seed <29021 + rerun index>`, using a process-order RNG seeded with
29021 and `WOLFGANG_CPU_BACKEND=scalar`. The same paired procedure is available as:

```bash
python benchmarks/bench_python_boundary.py --baseline-python <baseline-venv>/bin/python --reruns 9 --repeat 21 --seed 29021
```

Both paths receive five warmups. Timings
include the Python method, allocation and result destruction, with prebuilt input.

| Case | Baseline mean of medians | Candidate mean of medians | Change |
| --- | ---: | ---: | ---: |
| Export 100,000 coefficients and labels | 3.558 ms | 1.002 ms | -71.8% |
| Construct 128 two-qubit terms (unchanged control) | 13.587 µs | 13.437 µs | -1.1% |
| Simplify three terms (small-workload guard) | 0.601 µs | 0.483 µs | -19.7% |

Baseline: fresh CPU build at `d0fde93`, copied into an isolated runtime before
binding/build changes. Candidate: review-hardening native source fingerprint
recorded in the sample file; the measured loaded extension has its own SHA-256.
Both use Python 3.13.11, NumPy 2.5.3, AppleClang 21.0.0.21000099 and macOS 26.6.2
on Apple M4 Pro. oneTBB is absent, scalar is forced, native CPU tuning is off.
The baseline puts CPU sources in the nanobind module; the candidate builds a
Release static core with CMake's `-O3` policy and retains nanobind defaults for
bindings. This build-policy difference is part of the measured change. The local extension
grew from 347,496 to 380,264 bytes (9.4%); this is not a wheel-size measurement.

Correctness uses an exact NumPy coefficient oracle before timing. Independent
ownership and concurrent CPU operations also have regression coverage. This is a
single-host source-build result, not wheel, GPU, wide-Pauli or general arithmetic
performance evidence. Historical benchmark ledgers are unchanged. Other proposed
optimizations remain in the [experiment queue](../../plans/review_performance_experiments.md).

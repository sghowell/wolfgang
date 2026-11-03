# oneTBB In Release Wheels

Date: 2026-11-03

This report qualifies slice 7 of the portability and interop campaign
(`docs/plans/portability_and_interop_campaign_plan.md`). Release wheels now
bundle the oneTBB runtime, so `pip install` users get the parallel CPU
simplify that source builds with oneTBB already had
(`docs/benchmarks/reports/cpu_parallel_simplify_2026-11-03.md`).

## What changed

- `scripts/build_onetbb.sh` clones oneTBB `v2022.3.0`, checks that it resolved
  to commit `f1862f38f83568d96e814e469ab61f88336cc595`, and builds only the
  core `tbb` library (no tests, examples, tbbmalloc or hwloc binding). The
  macOS build targets the wheel's minimum macOS, 11.0, which pyproject sets
  explicitly (the measured run reached 11.0 through clang's arm64 floor). The
  script also requires the clone's license notices to equal the committed
  copies in `third_party/onetbb/`.
- cibuildwheel runs it once per platform (`before-all`), builds with
  `WOLFGANG_ENABLE_TBB=ON` so a missing oneTBB fails the build instead of
  producing a scalar-only wheel, and records oneTBB's library directory in the
  extension's install rpath so the repair step can find it.
- auditwheel (Linux) vendors the library as
  `wolfgang_quantum.libs/libtbb-<hash>.so.12.17`, and delocate (macOS) as
  `wolfgang_quantum/.dylibs/libtbb.12.17.dylib`. The project license is now
  `MIT AND Apache-2.0 AND BSD-3-Clause`, and oneTBB's `LICENSE.txt`
  (Apache-2.0) and `third-party-programs.txt` (including the BSD-3-Clause ITT
  Notify notice) ship as PEP 639 license files, with DLPack's license.
- The wheel smoke test runs with `--require-tbb`. It requires oneTBB 2022.3.0
  to be compiled and available, and its notices to be installed. It simplifies
  operators at 20, 100 and 130 qubits, with duplicate runs of one to five terms
  of mixed magnitude, under `WOLFGANG_CPU_BACKEND=tbb` and `=scalar` in fresh
  processes, and requires identical result bytes. On POSIX it also forks after
  a oneTBB simplify and requires the child's simplify to finish and match.
- oneTBB does not support `fork()`. Review reproduced a child forked after a
  oneTBB simplify dying inside oneTBB on macOS (Linux children ran in five
  trials). Every oneTBB kernel now records its use, and a child forked after
  that reports oneTBB as `unavailable_after_fork`: `auto` runs the
  bit-identical scalar or SIMD kernels there, and a forced
  `WOLFGANG_CPU_BACKEND=tbb` raises.
- Auto dispatch of pairwise commutation to oneTBB now needs at least four
  oneTBB threads, as simplify already did, so low-core hosts keep the SIMD
  kernel.

The measured wheels grew from 221 to 341 KiB (macOS arm64, cp312) and from
259 to 440 KiB (manylinux x86_64, cp312).

## Evidence

```text
wheel workflow, candidate: release-wheelhouse.yml run 41605223043 (codex/onetbb-wheels at 233714e, publish-target=none)
wheel workflow, baseline:  release-wheelhouse.yml run 41605250834 (main at 8d68a66, publish-target=none;
                           it differs from the branch's merge-base only in CUDA and device-array files)
summaries: docs/benchmarks/data/onetbb_wheels_2026-11-03/summary_epyc.json
           docs/benchmarks/data/onetbb_wheels_2026-11-03/summary_m4pro.json
```

Both workflow runs passed every job: the dependency audit and SBOM, the
sdist, six wheels per run with `twine check` and the public artifact audit,
and the wheelhouse collection checks. All six candidate wheels (CPython
3.10-3.12, manylinux x86_64 and macOS arm64) passed the `--require-tbb`
smoke test, reporting oneTBB 2022.3.0 and parallel simplify identical to
scalar.

Timings: each wheel installed into its own virtual environment with NumPy;
3 runs, wheel order randomized per run; per case one warmup and 7 timed
`PauliSum.simplify()` calls with the default `auto` CPU backend. Operators
come from packed arrays with shuffled duplicates. Medians of per-run medians.

| Host | Wheel | CPU |
| --- | --- | --- |
| AMD EPYC 9654, 16-vCPU cloud VM, Ubuntu 22.04 | cp310 manylinux_2_28_x86_64 | 16 logical CPUs |
| Apple M4 Pro, macOS 26.6.2 | cp312 macosx_11_0_arm64 | 8 performance + 4 efficiency cores |

## Results

| Qubits | Terms | Duplicates | EPYC before | EPYC after | Speedup | M4 Pro before | M4 Pro after | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 10,000 | 0% | 1.07 ms | 0.63 ms | 1.70x | 0.43 ms | 0.25 ms | 1.68x |
| 24 | 10,000 | 50% | 1.00 ms | 0.53 ms | 1.88x | 0.45 ms | 0.19 ms | 2.34x |
| 24 | 100,000 | 0% | 10.1 ms | 6.55 ms | 1.55x | 5.62 ms | 1.43 ms | 3.92x |
| 24 | 100,000 | 50% | 9.48 ms | 3.99 ms | 2.38x | 5.63 ms | 1.45 ms | 3.89x |
| 24 | 1,000,000 | 0% | 141 ms | 65.4 ms | 2.15x | 70.7 ms | 12.7 ms | 5.57x |
| 24 | 1,000,000 | 50% | 116 ms | 44.8 ms | 2.59x | 69.4 ms | 12.8 ms | 5.44x |
| 130 | 10,000 | 0% | 1.05 ms | 0.71 ms | 1.49x | 0.50 ms | 0.34 ms | 1.49x |
| 130 | 10,000 | 50% | 1.07 ms | 0.63 ms | 1.71x | 0.53 ms | 0.27 ms | 1.92x |
| 130 | 100,000 | 0% | 14.0 ms | 5.21 ms | 2.69x | 7.58 ms | 2.12 ms | 3.57x |
| 130 | 100,000 | 50% | 14.3 ms | 4.99 ms | 2.87x | 8.06 ms | 2.07 ms | 3.89x |
| 130 | 1,000,000 | 0% | 239 ms | 69.9 ms | 3.42x | 121 ms | 25.7 ms | 4.69x |
| 130 | 1,000,000 | 50% | 245 ms | 71.3 ms | 3.44x | 134 ms | 26.5 ms | 5.04x |

"Before" is the baseline wheel, which has no oneTBB and runs scalar simplify;
"after" is the candidate wheel, whose `auto` backend picks oneTBB from 8,192
terms on hosts with at least four oneTBB threads. These are the first ARM
oneTBB timings; earlier evidence was x86 source builds only.

## Limitations

- One x86 cloud VM and one Mac. The EPYC VM ran the cp310 wheel (its system
  Python) and the Mac the cp312 wheel.
- The randomized order put the baseline first in all three Mac runs.
- Only `simplify` is timed here. `auto` also picks oneTBB for dense pairwise
  commutation above its threshold, as in source builds with oneTBB; that path
  was not re-timed from wheels.
- oneTBB keeps a worker thread pool per process. Applications that also load
  another oneTBB runtime (for example through other wheels) get a separate
  pool from Wolfgang's copy (renamed on Linux, its own install name on macOS),
  and the two pools can oversubscribe the cores.
- The timings above come from the first workflow run. The review fixes (fork
  guard, notices as license files, stronger smoke probe, pairwise thread
  floor) were validated by a second workflow run; they do not touch the timed
  kernels.
- The SBOM generated by the release workflow lists Python dependencies; the
  bundled oneTBB runtime is recorded here and in
  `docs/quality/security_and_supply_chain.md` by version, commit and license.

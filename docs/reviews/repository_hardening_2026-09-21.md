# Repository Hardening Review — 2026-09-21

Baseline: `4173a08`. Implementation commits: `d0fde93`, `491d13e`.
The [accepted plan](../plans/review_hardening_plan.md) covers correctness,
Metal lifetimes, public contracts, native/Python boundaries and validation.

## Changes and evidence

- OpenFermion export combines duplicate coefficients without tolerance pruning.
- CPU/Metal simplify checks finite input and accumulation and safely compares
  finite complex components whose magnitude overflows double. Fused CPU matmul
  uses the same numeric guards. Count normalization avoids overflowing totals.
- Input-relative tolerance semantics are retained; idempotence applies to `rtol=0`.
- Metal operations, constructors, status and destructors use scoped pools;
  C++ exception unwinding drains their temporaries.
- Wave 1D fails closed on incomplete/invalid evidence and preserves CPU-only skips.
- Public capability fallback discovers GPU devices, and immutable operation
  records distinguish device kernels, host bridges and unsupported operations.
- The native library builds without Python. CPU bindings release the GIL during
  pure computation and export coefficients with one owning NumPy allocation.
- Benchmarks identify the loaded artifact and native sources, preserving actual
  git state independently of requested labels. Validation profiles are shared by CI.
- Packaging excludes local worktrees/release staging, and the artifact validator
  explicitly rejects either directory in an archive.

Environment: Apple M4 Pro, macOS 26.6.2, AppleClang 21.0.0.21000099, Python
3.13.11, NumPy 2.5.3, OpenFermion 1.7.1 and Qiskit 1.4.5. CPU and Metal were
built separately, with internal hooks enabled for development checks; public
packaging kept internal hooks and all accelerators disabled.

| Validation | Result |
| --- | --- |
| `python scripts/validate.py --profile all` | Passed: 595 Python tests, 126 explicit skips; Ruff, Pyright (zero errors), spelling, strict MkDocs, native CTest, benchmark smokes and sdist build |
| Focused release-boundary tests after staging exclusions | 14 passed |
| Fresh public CPU wheel: numeric/binding/capability/onboarding tests | 48 passed |
| `python scripts/validate_release_artifacts.py --output-dir <temporary-artifact-directory>` | CPU sdist/wheel build, clean installation and metadata smoke passed |
| Rebuild wheel from extracted sdist; audit sdist | Passed |
| Standalone core CTest under AddressSanitizer/UndefinedBehaviorSanitizer | Passed |
| Metal native CTest with Python disabled | 2 passed, including 50 C++ exception unwinds with immediate object destruction |
| Metal numerical, foundation, campaign and lifetime slice tests | 74 passed, 7 expected CPU-only/private-archive skips |
| All Metal campaign and lifetime tests | 87 passed, 20 expected CPU-only/private-archive skips |
| Metal Wave 1D CLI, 3 reruns × 3 repetitions | Valid complete report, gate `go`; this smoke is not a new performance-promotion claim |
| CPU-only Wave 1D CLI | `skipped`, without exception or promotion |
| `git diff --check` and tracked-artifact audit | Passed |

Standalone sanitizer recipe:

```bash
cmake -S . -B <native-build> -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DWOLFGANG_BUILD_PYTHON=OFF -DWOLFGANG_BUILD_NATIVE_TESTS=ON \
  -DWOLFGANG_ENABLE_TBB=OFF \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build <native-build>
ctest --test-dir <native-build> --output-on-failure
```

The [paired boundary report](../benchmarks/reports/python_boundary_hardening_2026-09-21.md)
records all nine reruns and unchanged controls. Export of 100,000 coefficients
and labels improved by 71.8% locally, with the control moving -1.1%. The result
covers the bundled binding/build change and a 9.4% local extension-size increase;
it does not isolate memcpy or imply general CPU/GPU acceleration.

## Independent review

Two independent agents reviewed `d0fde93`. Counts: P0=0, P1=1, P2=3.

- P1: fused matmul bypassed finite reduction guards. Resolved with shared checks
  and regressions around the 127/128-term optimization boundary in both widths.
- P2: idempotence property still used relative tolerance. Resolved with `rtol=0`;
  explicit input-relative counterexamples remain covered.
- P2: aggregation could admit same-name rows from an unvalidated profile.
  Resolved by matching filters and a colliding-profile regression.
- P2: constructor/status `@autoreleasepool` blocks did not drain on C++ exceptions.
  Resolved with RAII pools and the native exception test.

The second independent review covered all fixes plus native/CMake, GIL, packaging,
benchmark and CI changes. Native/API reviewer: P0=0, P1=0, P2=0, P3=0;
58 independently executed focused tests passed. Accelerator/evidence reviewer:
P0=0, P1=0, P2=1; 30 focused tests and the exception CTest passed independently,
and all retained benchmark aggregates/fingerprints recomputed correctly.

The final P2 was the `conflict_degrees()` docstring's device-only claim. It now
explicitly describes CUDA/HIP reductions and Metal's CPU shared-memory scan;
fresh native builds verify the corrected docstring. No review finding is deferred.
The benchmark report names the measured candidate `491d13e`; subsequent native
changes alter docstrings only. Final merged-state validation and hosted CI attach
to the merged commit; the task result records their outcomes.

## Remaining qualification

CUDA/HIP kernels were unchanged. Their numerical-limit parity and the refactored
accelerator build targets need hardware/toolchain qualification. Local tests cover
CPU/Metal only. Device methods continue holding the GIL; native Metal pipeline
cache concurrency is a separate review before enabling concurrent device calls.
The existing deprecated offline Metal library loader remains outside this slice.

GPU algorithm candidates, direct consumers, reusable CPU outputs, dispatch
requalification and diagonal-transform experiments are specified in the
[performance queue](../plans/review_performance_experiments.md). No paid compute,
new release, package-index publication or broader hardware support is claimed.

# FastPauli CUDA Performance Hardening - NVIDIA H100 PCIe - 2026-04-28

## Summary

- Performance hardening revision: `f31cbdc`.
- Comparison baseline source revision: `56de0de` (`Refresh H100 CUDA benchmark plot`).
- GPU: NVIDIA H100 PCIe
- Compute capability: 9.0
- Driver from `nvidia-smi`: 580.126.09
- CUDA runtime reported by FastPauli: 12.9
- CUDA driver API version reported by FastPauli: 13.0
- CUDA toolkit: 12.9.86
- Compiled CUDA architecture for final benchmark: `90`
- Host compiler metadata: `/usr/bin/g++`
- CPU: Intel(R) Xeon(R) Platinum 8352Y CPU @ 2.20GHz
- Architecture: x86_64
- OS: Linux-6.8.0-90-generic-x86_64-with-glibc2.35
- Active CPU backend: scalar
- Available CPU backends: scalar, oneTBB, AVX2, AVX-512
- oneTBB version: 2021.5.0
- Thread settings: `MKL_NUM_THREADS=unset`, `OMP_NUM_THREADS=unset`,
  `controlled_thread_count=not_controlled`
- Python: 3.10.12
- NumPy: 2.2.6
- CUDA-array-interface provider for validation: CuPy 14.0.1 (`cupy-cuda12x`)

## Scope

This hardening slice focused on synchronization overhead in already-correct CUDA
paths. The retained code changes remove redundant synchronization around
synchronous host/device copies, avoid a conservative pre-sync for internally
copied host statevectors while keeping it for external CUDA-array-interface
inputs, and let `matmul(..., simplify=True)` pass directly into `simplify()`
without an intermediate device sync.

Rejected hillclimb candidates:

- Specialized one- and two-word two-dimensional commutation kernels regressed
  benchmark profiles and were reverted.
- Removing the commutation post-kernel synchronization did not produce a stable
  gain and was reverted.
- Replacing commutation result storage with a raw `cudaMalloc`/`cudaMemcpy`
  path regressed timings and was reverted.
- Batched `cudaMemcpyAsync` transfer helpers regressed the measured workloads
  and were removed.

## Commands

- CUDA validation:
  `PATH=/usr/local/cuda-12.9/bin:$PATH CUDACXX=/usr/local/cuda-12.9/bin/nvcc CUDAHOSTCXX=/usr/bin/g++ FASTPAULI_VALIDATE_CUDA=1 FASTPAULI_CUDA_ARCHITECTURES=90 python scripts/validate.py`
- CUDA sanitizer:
  `PATH=/usr/local/cuda-12.9/bin:$PATH compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest tests/test_phase11_cuda_kernels.py -q`
- CUDA benchmark default:
  `python benchmarks/bench_cuda_kernels.py --profile default --repeat 9 --warmup 3 --json`
- CUDA benchmark stress:
  `python benchmarks/bench_cuda_kernels.py --profile stress --repeat 7 --warmup 2 --json`
- CUDA benchmark plot refresh:
  `python scripts/render_benchmark_plots.py --cuda-report docs/benchmarks/reports/cuda_h100_performance_hardening_2026-04-28.md --output docs/benchmarks/plots/cuda_h100_performance_hardening_default_backend_speedups.svg`

## Validation Results

- Full CPU-first validation path on the H100 before CUDA rebuild: 133 passed,
  39 skipped.
- CUDA source build with default architectures
  (`70,75,80,86,89,90`): passed.
- CUDA source build with requested architecture override (`90`): passed.
- CUDA full pytest after `FASTPAULI_ENABLE_CUDA=ON`: 143 passed, 29 skipped.
- CUDA transfer pytest: 5 passed.
- CUDA kernel pytest: 13 passed.
- CUDA benchmark smoke: passed with CPU/GPU correctness checks enabled.
- Source distribution smoke: `fastpauli-0.1.0.tar.gz` built successfully.
- Compute sanitizer: 13 Phase 11 CUDA tests passed, `ERROR SUMMARY: 0 errors`.

Optional Qiskit and OpenFermion checks were skipped on this CUDA host because
those optional libraries were not installed. CUDA-array-interface validation did
run with CuPy installed.

## Benchmark Default

Default cases are deterministic medium-sized workloads selected to verify that
CUDA becomes useful when data stays resident or transfer cost is amortized. The
CPU optimized column is populated only for benchmark cases with named optimized
CPU kernels. In this report, that is pairwise commutation, where oneTBB, AVX2,
and AVX-512 were available and checked. oneTBB thread count was not controlled;
the benchmark environment recorded `MKL_NUM_THREADS=unset`,
`OMP_NUM_THREADS=unset`, and `controlled_thread_count=not_controlled`. NEON and
SVE were unavailable on this x86_64 host.

| Case | Dataset | CPU Scalar Seconds | CPU Optimized | CUDA Transfer-Inclusive Seconds | CUDA Device-Resident Seconds | Regime |
| --- | --- | ---: | --- | ---: | ---: | --- |
| simplify_duplicate_pressure | num_qubits=16; num_terms=50000; term_weight=3; duplicate_rate=0.9801; duplicate_pool_size=1024 | 0.006930211 | n/a | 0.001901285 | 0.000574626 | CUDA-faster |
| statevector_expectation | num_qubits=12; num_terms=2048; term_weight=3; duplicate_rate=0.15966796875; statevector_length=4096 | 0.036774736 | n/a | 0.001002643 | 6.674e-05 | CUDA-faster |
| pairwise_commutation | num_qubits=16; lhs_terms=2048; rhs_terms=2048; entries=4194304; term_weight=3 | 0.018419889 | tbb: 0.001983899; avx512: 0.005742285; avx2: 0.005227101 | 0.004051443 | 0.002237863 | CUDA-faster than scalar; oneTBB-faster |
| matmul_product_generation_simplify | num_qubits=12; lhs_terms=256; rhs_terms=256; intermediate_terms=65536; term_weight=3 | 0.011645763 | n/a | 0.003554215 | 0.000764082 | CUDA-faster |

## Benchmark Stress

Stress cases keep the same deterministic construction pattern but increase
terms, entries, or statevector size to better expose resident-data behavior and
large parallel kernels.

| Case | Dataset | CPU Scalar Seconds | CPU Optimized | CUDA Transfer-Inclusive Seconds | CUDA Device-Resident Seconds | Regime |
| --- | --- | ---: | --- | ---: | ---: | --- |
| simplify_duplicate_pressure | num_qubits=16; num_terms=100000; term_weight=3; duplicate_rate=0.99005; duplicate_pool_size=1024 | 0.016037463 | n/a | 0.002529179 | 0.000805897 | CUDA-faster |
| statevector_expectation | num_qubits=14; num_terms=4096; term_weight=3; duplicate_rate=0.1845703125; statevector_length=16384 | 0.303552863 | n/a | 0.001243973 | 0.000206784 | CUDA-faster |
| pairwise_commutation | num_qubits=16; lhs_terms=8192; rhs_terms=8192; entries=67108864; term_weight=3 | 0.360375774 | tbb: 0.073005563; avx512: 0.144784475; avx2: 0.14582058 | 0.068881765 | 0.066086858 | CUDA-faster |
| matmul_product_generation_simplify | num_qubits=12; lhs_terms=512; rhs_terms=512; intermediate_terms=262144; term_weight=3 | 0.054714954 | n/a | 0.006328309 | 0.00143085 | CUDA-faster |

## Baseline Comparison

The comparison baseline source tree was a detached local worktree at
`56de0de`. The early baseline benchmark JSON was captured before the Git
metadata sync was repaired, so this report records the known source revision
instead of the JSON field.

Default profile:

| Case | Baseline Transfer Seconds | Hardened Transfer Seconds | Transfer Ratio | Baseline Resident Seconds | Hardened Resident Seconds | Resident Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| simplify_duplicate_pressure | 0.001932027 | 0.001901285 | 1.016x | 0.000572287 | 0.000574626 | 0.996x |
| statevector_expectation | 0.00105622 | 0.001002643 | 1.053x | 8.013e-05 | 6.674e-05 | 1.201x |
| pairwise_commutation | 0.003458886 | 0.004051443 | 0.854x | 0.001526454 | 0.002237863 | 0.682x |
| matmul_product_generation_simplify | 0.003411569 | 0.003554215 | 0.960x | 0.000870446 | 0.000764082 | 1.139x |

Stress profile:

| Case | Baseline Transfer Seconds | Hardened Transfer Seconds | Transfer Ratio | Baseline Resident Seconds | Hardened Resident Seconds | Resident Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| simplify_duplicate_pressure | 0.0026867 | 0.002529179 | 1.062x | 0.00083639 | 0.000805897 | 1.038x |
| statevector_expectation | 0.001258131 | 0.001243973 | 1.011x | 0.000216414 | 0.000206784 | 1.047x |
| pairwise_commutation | 0.068354807 | 0.068881765 | 0.992x | 0.064228881 | 0.066086858 | 0.972x |
| matmul_product_generation_simplify | 0.006051966 | 0.006328309 | 0.956x | 0.001412815 | 0.00143085 | 0.987x |

## Interpretation

The retained synchronization changes provide stable benefit where they directly
remove overhead from the hot path: statevector expectation improves in both
profiles, and stress simplify improves modestly. The `matmul(...,
simplify=True)` sync move improves the default resident path but does not show a
stable stress-profile gain in this run.

Pairwise commutation source did not change in the retained patch. The stress
profile remains CUDA-faster than the available CPU paths, but the default
profile's oneTBB selector is faster than CUDA while CUDA remains faster than
scalar and the SIMD selectors. The full benchmark profile also shows noisy
default-profile commutation movement, and that movement should not be
interpreted as a real commutation optimization or regression without a dedicated
commutation-only benchmark run. The rejected commutation-kernel candidates above
were reverted because they did not produce a defensible improvement across
profiles.

These measurements are source-build evidence on one H100 host. They are not CUDA
wheel distribution claims, and they should not be generalized to other GPU
architectures without the hardware-target evidence required by
`docs/architecture/hardware_targets_and_testing.md`.

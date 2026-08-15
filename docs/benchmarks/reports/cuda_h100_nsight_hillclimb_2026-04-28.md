# FastPauli CUDA Nsight Hillclimb - NVIDIA H100 PCIe - 2026-04-28

## Summary

- Benchmark revision: `6196848` (`Add extreme CUDA scaling profile`).
- Retained performance revisions in this slice: `25a3890` (`Optimize CUDA profiling hot paths`) and `989f08f` (`Optimize SIMD commutation stores`).
- Comparison baseline report: `docs/benchmarks/reports/cuda_h100_performance_hardening_2026-04-28.md`.
- GPU: NVIDIA H100 PCIe, compute capability 9.0, 81079 MiB.
- Driver from `nvidia-smi`: 580.126.09.
- CUDA runtime reported by FastPauli: 12.9.
- CUDA driver API version reported by FastPauli: 13.0.
- CUDA toolkit: 12.9.86.
- Compiled CUDA architecture: `90`.
- Nsight Systems: 2025.1.3.140.
- Nsight Compute: 2025.2.1.0.
- Compute Sanitizer: 2025.2.1.0.
- Host compiler metadata: `/usr/bin/g++`.
- CPU: Intel(R) Xeon(R) Platinum 8352Y CPU @ 2.20GHz.
- Available CPU backends: scalar, tbb, avx2, avx512.
- oneTBB version: 2021.5.0.
- Thread settings: `MKL_NUM_THREADS=unset`, `OMP_NUM_THREADS=unset`, `controlled_thread_count=not_controlled` for CUDA runs; `OPENBLAS_NUM_THREADS=1` for the final CPU perf hot loop.
- Python: 3.10.12.
- NumPy: 2.2.6.

## Scope

This hillclimb used Nsight Systems, Nsight Compute, Compute Sanitizer, CUDA
benchmark scaling profiles, and CPU `perf` to separate host overhead,
result-copy cost, kernel execution cost, and CPU SIMD hot-loop cost. It also
used CUDA binary inspection during scratch experiments, but no SASS/PTX change
was retained. The measured retained changes:

- avoid full CUDA device-property discovery on every `PauliSum.to_device()`;
- avoid value-initializing CUDA commutation and expectation scratch buffers that
  kernels fully overwrite;
- remove redundant synchronization after CUDA commutation and expectation when a
  following host copy or Thrust reduction already provides ordering;
- let Python `DevicePauliSum.commutes_with()` fill the NumPy bool output buffer
  directly instead of copying through an intermediate `std::vector`;
- register large host output buffers before dense CUDA commutation result
  copies, with a safe fallback when registration is unavailable;
- replace AVX2/AVX-512 SIMD commutation stack stores with mask-to-byte stores;
- add `benchmarks/bench_cuda_scaling.py --profile extreme` for correctness-
  checked scale-limit runs.

Rejected experiments are recorded as implementation notes, not headline
benchmark claims:

- Staging large commutation results through a newly allocated pinned buffer was
  slower than direct NumPy fills because it added an extra host copy.
- Byte-packed and warp-ballot-packed commutation transfer paths were correct but
  did not improve transfer-inclusive timing consistently across 4096x4096,
  8192x8192, and 10000x10000 cases.
- `cudaMallocAsync` scratch allocation helped isolated small commutation cases
  but regressed or added noise for larger dense outputs and expectation, so it
  was not retained.
- Direct mapped-host commutation output, disabling large-output host
  registration, and a 512-thread expectation block size were correct but did not
  improve enough across the measured scale ladder to retain.
- Handwritten PTX/SASS was not retained. Nsight Compute reported no local-memory
  spill in the custom commutation or expectation samples; the remaining
  large-output limiter is host result movement and CUDA/Thrust API overhead
  rather than an obvious register-spill issue.
- Removing final operation synchronizations was rejected after follow-up A/B
  testing because it conflicts with the documented Phase 11 default-stream
  synchronization semantics and can make Python-call latency benchmarks
  misleading unless every timed callable explicitly synchronizes. The scratch
  stress A/B improved several medians, but the semantic risk is not acceptable.

## Commands

- CUDA source build:
  `PATH=/usr/local/cuda-12.9/bin:$PATH CUDACXX=/usr/local/cuda-12.9/bin/nvcc CUDAHOSTCXX=/usr/bin/g++ python -m pip install -e ".[test]" --config-settings=cmake.define.FASTPAULI_ENABLE_CUDA=ON --config-settings=cmake.define.FASTPAULI_CUDA_ARCHITECTURES=90 --config-settings=cmake.define.CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++`
- CUDA focused pytest:
  `PATH=/usr/local/cuda-12.9/bin:$PATH python -m pytest tests/test_phase11_cuda_kernels.py tests/test_cuda_scaling_benchmark.py`
- Compute Sanitizer:
  `PATH=/usr/local/cuda-12.9/bin:$PATH compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest tests/test_phase11_cuda_kernels.py -q`
- CUDA benchmark default:
  `PATH=/usr/local/cuda-12.9/bin:$PATH python benchmarks/bench_cuda_kernels.py --profile default --repeat 9 --warmup 3 --json`
- CUDA benchmark stress:
  `PATH=/usr/local/cuda-12.9/bin:$PATH python benchmarks/bench_cuda_kernels.py --profile stress --repeat 7 --warmup 2 --json`
- CUDA scaling default:
  `<private-path> benchmarks/bench_cuda_scaling.py --profile default --repeat 5 --warmup 2 --json`
- CUDA scaling stress:
  `<private-path> benchmarks/bench_cuda_scaling.py --profile stress --repeat 3 --warmup 1 --json`
- CUDA scaling extreme:
  `<private-path> benchmarks/bench_cuda_scaling.py --profile extreme --repeat 1 --warmup 0 --json`
- Nsight Systems:
  `PATH=/usr/local/cuda-12.9/bin:$PATH nsys profile --force-overwrite=true --stats=true --trace=cuda,osrt --output=<private-path> python benchmarks/bench_cuda_kernels.py --profile stress --repeat 2 --warmup 1 --json`
- Nsight Compute commutation:
  `sudo env TMPDIR=<private-path> PATH=/usr/local/cuda-12.9/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin ncu --target-processes all --set detailed --kernel-name-base function --kernel-name regex:commutation_kernel --launch-count 1 --force-overwrite --export <private-path> .venv/bin/python benchmarks/bench_cuda_scaling.py --profile default --operation pairwise_commutation --repeat 1 --warmup 0 --json`
- Nsight Compute expectation:
  `sudo env TMPDIR=<private-path> PATH=/usr/local/cuda-12.9/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin ncu --target-processes all --set detailed --kernel-name-base function --kernel-name regex:expectation_statevector_terms_kernel --launch-count 1 --force-overwrite --export <private-path> .venv/bin/python benchmarks/bench_cuda_scaling.py --profile default --operation statevector_expectation --repeat 1 --warmup 0 --json`
- Nsight Compute matmul:
  `sudo env TMPDIR=<private-path> PATH=/usr/local/cuda-12.9/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin ncu --target-processes all --set detailed --kernel-name-base function --kernel-name regex:matmul_product_kernel --launch-count 1 --force-overwrite --export <private-path> .venv/bin/python benchmarks/bench_cuda_scaling.py --profile default --operation matmul_product_generation_simplify --repeat 1 --warmup 0 --json`
- CPU perf:
  `sudo perf stat -e cycles,instructions,branches,branch-misses,cache-misses .venv/bin/python benchmarks/bench_cpu_dispatch.py --repeat 7 --warmup 2 --commutation-qubits 65 --lhs-terms 2048 --rhs-terms 2048 --group-terms 512 --json`
- Rejected synchronization-removal A/B:
  `python benchmarks/bench_cuda_kernels.py --profile default --warmup 3 --repeat 7 --json`
  and
  `python benchmarks/bench_cuda_kernels.py --profile stress --warmup 2 --repeat 5 --json`
  against a scratch patch that removed final `cudaDeviceSynchronize()` calls
  from CUDA simplify and unsimplified matmul.

## Validation Results

- CUDA focused pytest after retained CUDA changes: 14 passed.
- x86 CPU backend plus CUDA kernel pytest after retained SIMD changes: 24 passed.
- Local CUDA scaling benchmark tests after adding the extreme profile: 2 passed.
- Full local Apple Silicon validation: 164 passed, 10 skipped; benchmark smokes
  and source distribution smoke passed.
- Full H100 CUDA validation with `FASTPAULI_VALIDATE_CUDA=1`: CPU-first pytest
  135 passed, 39 skipped; CUDA-enabled pytest 145 passed, 29 skipped; CUDA
  transfer tests 5 passed; CUDA kernel tests 13 passed; source distribution
  smoke passed.
- Compute Sanitizer: 13 Phase 11 CUDA tests passed, `ERROR SUMMARY: 0 errors`.
- Review-fix validation after RAII cleanup and scalar benchmark-oracle fixes:
  focused H100 CUDA tests 20 passed; Compute Sanitizer 13 passed with
  `ERROR SUMMARY: 0 errors`; full H100 CUDA validation repeated with CPU-first
  pytest 135 passed, 39 skipped; CUDA-enabled pytest 145 passed, 29 skipped;
  CUDA transfer tests 5 passed; CUDA kernel tests 13 passed; source
  distribution smoke passed.
- CUDA scaling benchmark smoke is wired into `scripts/validate.py`; the extreme
  profile is opt-in and is not part of the default validation path.
- CPU `perf stat` on the final x86 dispatch profile recorded 12.876B cycles,
  18.236B instructions, 1.42 IPC, 2.872B branches, 0.17% branch misses, and
  4.757M cache misses over 1.198 s elapsed.
- CPU native hot-loop profiling with `OPENBLAS_NUM_THREADS=1` sampled the
  FastPauli extension rather than idle OpenBLAS worker threads. The normal
  optimized wheel is mostly stripped, so the profile is useful for shared-object
  concentration but not full symbolic source attribution.

## Rejected CUDA A/B Experiments

A scratch patch removed final synchronizations from CUDA simplify and
unsimplified matmul. This was not retained because
`docs/architecture/cuda_backend.md` requires the first CUDA backend to
synchronize before returning to Python, and timing calls that return before GPU
work completes would overstate device-resident speed. The measured stress
profile did improve, which makes this a candidate for a later explicit async
stream API rather than a Phase 11 change.

| Case | Profile | Baseline CUDA Transfer Seconds | Scratch CUDA Transfer Seconds | Baseline CUDA Resident Seconds | Scratch CUDA Resident Seconds | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| simplify_duplicate_pressure | default | 0.001003502999083139 | 0.001013652 | 0.0005667500008712523 | 0.000559439 | reject: flat/noisy |
| statevector_expectation | default | 0.00010884299990721047 | 0.000127462999 | 5.480799882207066e-05 | 5.61699999e-05 | reject: regression/noise |
| pairwise_commutation | default | 0.0014767209995625308 | 0.000652044 | 0.00046763099999225233 | 0.001323185 | reject: inconsistent |
| matmul_product_generation_simplify | default | 0.0016741420004109386 | 0.001735788 | 0.0009729140001581982 | 0.000780705001 | reject: semantic risk |
| simplify_duplicate_pressure | stress | 0.0017232500013051322 | 0.001644405 | 0.0009203980007441714 | 0.000845447001 | reject: semantic risk |
| statevector_expectation | stress | 0.00032775600084278267 | 0.000317303 | 0.00020130499979131855 | 0.000201018 | reject: semantic risk |
| pairwise_commutation | stress | 0.02054997899904265 | 0.016621191 | 0.01972011199904955 | 0.01673336 | reject: semantic risk |
| matmul_product_generation_simplify | stress | 0.004567592999592307 | 0.003653478 | 0.0017721420008456334 | 0.001667707 | reject: semantic risk |

Mapped host output was tested for large dense commutation by registering the
caller-owned host buffer with `cudaHostRegisterMapped` and launching the
commutation kernel directly against the mapped pointer. It was correct, but not
retained because the large-output transfer-inclusive cases were flat or worse.

| Scale | Baseline Transfer Seconds | Mapped Transfer Seconds | Baseline Resident Seconds | Mapped Resident Seconds | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| default terms_1024x1024 | 0.000241505 | 0.000262240999 | 0.000133214 | 0.000265782 | reject: regression |
| default terms_2048x2048 | 0.000595016001 | 0.001162048 | 0.000758733 | 0.000649023001 | reject: transfer regression |
| default terms_4096x4096 | 0.001630374 | 0.003890932 | 0.005106187 | 0.001928304 | reject: transfer regression/noise |
| stress terms_4096x4096 | 0.004756305 | 0.004029679 | 0.005222279 | 0.001868571 | reject: not stable at larger sizes |
| stress terms_8192x8192 | 0.020117103 | 0.020478371 | 0.021593838 | 0.020477017 | reject: flat |
| stress terms_10000x10000 | 0.029791451 | 0.030335762 | 0.029300479 | 0.033688448 | reject: regression |

Disabling large-output host registration was tested to make sure
`cudaHostRegister` overhead was justified. It was not retained because pageable
copies regressed the 8192x8192 and 10000x10000 commutation outputs, and a
6000x6000 threshold-edge case regressed from 0.006913963 s to 0.017988988 s.

| Scale | Registered Resident Seconds | Pageable Resident Seconds | Decision |
| --- | ---: | ---: | --- |
| stress terms_8192x8192 | 0.021593838 | 0.031199614 | keep registration |
| stress terms_10000x10000 | 0.029300479 | 0.048193466 | keep registration |
| threshold terms_6000x6000 | 0.006913963 | 0.017988988 | keep 32 MiB threshold |

Increasing the expectation kernel from 256 to 512 threads per block was tested
for larger statevectors. It was not retained because stress resident timings
regressed.

| Scale | Baseline Resident Seconds | 512-Thread Resident Seconds | Decision |
| --- | ---: | ---: | --- |
| default qubits_10_terms_1024 | 0.000042725 | 0.000043302 | reject: flat/regression |
| default qubits_14_terms_4096 | 0.000255134 | 0.000251737 | reject: too small to justify |
| stress qubits_14_terms_4096 | 0.000244231 | 0.000263888001 | reject: regression |
| stress qubits_16_terms_8192 | 0.001572618 | 0.001638692 | reject: regression |

## Nsight Findings

- Nsight Systems on the final stress all-ops profile still shows CUDA API and
  host-result costs dominating traced runtime: `cudaMalloc` 46.0% of CUDA API
  time, `cudaHostRegister` 37.1%, and `cudaMemcpy` 7.6%. GPU memops were 95.1%
  device-to-host by time.
- Nsight Systems GPU kernel time was led by the custom commutation kernel
  (2.720 ms total across 7 instances) and expectation kernel (1.310 ms total
  across 8 instances); Thrust/CUB merge-sort kernels dominate the remaining
  simplify/matmul-simplify kernel launches.
- Nsight Compute commutation sample: 12.74 us kernel duration, 32
  registers/thread, 82.66% achieved occupancy, 57.69% compute throughput, and
  20.40% memory throughput.
- Nsight Compute expectation sample: 12.10 us kernel duration, 27
  registers/thread, 72.92% achieved occupancy, 42.26% compute throughput, and
  41.25% memory throughput.
- Nsight Compute matmul product sample: 9.57 us kernel duration, 48
  registers/thread, 52.03% achieved occupancy, 34.77% compute throughput, and
  35.35% memory throughput. This custom product kernel is a small fraction of
  `matmul(..., simplify=True)`; Thrust/CUB simplify work dominates that path.

## Benchmark Default

Default cases are deterministic medium-sized workloads selected to verify that
CUDA becomes useful when data stays resident or transfer cost is amortized. The
CPU optimized column is populated only for benchmark cases with named optimized
CPU kernels.

| Case | Dataset | CPU Scalar Seconds | CPU Optimized | CUDA Transfer-Inclusive Seconds | CUDA Device-Resident Seconds | Regime |
| --- | --- | ---: | --- | ---: | ---: | --- |
| simplify_duplicate_pressure | num_qubits=16; num_terms=50000; duplicate_rate=0.9801; duplicate_pool_size=1024; term_weight=3 | 0.006761638998796116 | n/a | 0.001003502999083139 | 0.0005667500008712523 | CUDA-faster |
| statevector_expectation | num_qubits=12; num_terms=2048; duplicate_rate=0.15966796875; duplicate_pool_size=2048; statevector_length=4096; term_weight=3 | 0.03510923300018476 | n/a | 0.00010884299990721047 | 5.480799882207066e-05 | CUDA-faster |
| pairwise_commutation | num_qubits=16; lhs_terms=2048; rhs_terms=2048; entries=4194304; term_weight=3 | 0.01990857900091214 | tbb: 0.0018812220005202107; avx512: 0.002556884001023718; avx2: 0.004035961001136457 | 0.0014767209995625308 | 0.00046763099999225233 | CUDA-faster |
| matmul_product_generation_simplify | num_qubits=12; lhs_terms=256; rhs_terms=256; intermediate_terms=65536; term_weight=3 | 0.011370087999239331 | n/a | 0.0016741420004109386 | 0.0009729140001581982 | CUDA-faster |

## Benchmark Stress

Stress cases keep the same deterministic construction pattern but increase
terms, entries, or statevector size to expose resident-data behavior and larger
parallel kernels.

| Case | Dataset | CPU Scalar Seconds | CPU Optimized | CUDA Transfer-Inclusive Seconds | CUDA Device-Resident Seconds | Regime |
| --- | --- | ---: | --- | ---: | ---: | --- |
| simplify_duplicate_pressure | num_qubits=16; num_terms=100000; duplicate_rate=0.99005; duplicate_pool_size=1024; term_weight=3 | 0.017474189999120426 | n/a | 0.0017232500013051322 | 0.0009203980007441714 | CUDA-faster |
| statevector_expectation | num_qubits=14; num_terms=4096; duplicate_rate=0.1845703125; duplicate_pool_size=4096; statevector_length=16384; term_weight=3 | 0.29367914800059225 | n/a | 0.00032775600084278267 | 0.00020130499979131855 | CUDA-faster |
| pairwise_commutation | num_qubits=16; lhs_terms=8192; rhs_terms=8192; entries=67108864; term_weight=3 | 0.4666832960010652 | tbb: 0.0753151290009555; avx512: 0.10651851799957512; avx2: 0.1332066320010199 | 0.02054997899904265 | 0.01972011199904955 | CUDA-faster |
| matmul_product_generation_simplify | num_qubits=12; lhs_terms=512; rhs_terms=512; intermediate_terms=262144; term_weight=3 | 0.0686403239997162 | n/a | 0.004567592999592307 | 0.0017721420008456334 | CUDA-faster |

## Scaling Results

The scaling benchmark keeps correctness checks enabled and compares CPU scalar,
applicable optimized CPU selectors, CUDA transfer-inclusive, and CUDA
device-resident timings on each scale point.

| Profile | Case | Scale | CPU Scalar Seconds | CPU Optimized | CUDA Transfer Seconds | CUDA Resident Seconds |
| --- | --- | --- | ---: | --- | ---: | ---: |
| default | simplify_duplicate_pressure | terms_10000 | 0.001333098000031896 | n/a | 0.0005553949995373841 | 0.000306945999909658 |
| default | simplify_duplicate_pressure | terms_50000 | 0.007020471999567235 | n/a | 0.001025727000524057 | 0.0005894619989703642 |
| default | simplify_duplicate_pressure | terms_200000 | 0.02934815300068294 | n/a | 0.00202559599892993 | 0.0007875960000092164 |
| default | statevector_expectation | qubits_10_terms_1024 | 0.004571763000058127 | n/a | 7.486499998776708e-05 | 4.2724999730126e-05 |
| default | statevector_expectation | qubits_12_terms_2048 | 0.0358020250005211 | n/a | 0.00012909999895782676 | 7.583499973407015e-05 |
| default | statevector_expectation | qubits_14_terms_4096 | 0.30326305400012643 | n/a | 0.0003414229995541973 | 0.0002551340003265068 |
| default | pairwise_commutation | terms_1024x1024 | 0.005068108999694232 | tbb: 0.0005317209997883765; avx512: 0.000663590000840486; avx2: 0.0014188069999363506 | 0.00024150500030373223 | 0.0001332140000158688 |
| default | pairwise_commutation | terms_2048x2048 | 0.021438250001665438 | tbb: 0.0019734679990506265; avx512: 0.002692600000955281; avx2: 0.004275698998753796 | 0.0005950160011707339 | 0.0007587330001115333 |
| default | pairwise_commutation | terms_4096x4096 | 0.08694999500039557 | tbb: 0.009664444000009098; avx512: 0.01258056500046223; avx2: 0.01667583999915223 | 0.001630373999432777 | 0.005106186999910278 |
| default | matmul_product_generation_simplify | terms_128x128 | 0.002821191999828443 | n/a | 0.0008115250002447283 | 0.0003265510003984673 |
| default | matmul_product_generation_simplify | terms_256x256 | 0.011741398000594927 | n/a | 0.0017617360008443939 | 0.0007472520010196604 |
| default | matmul_product_generation_simplify | terms_512x512 | 0.05622607099940069 | n/a | 0.004210059998513316 | 0.0014450529997702688 |
| stress | simplify_duplicate_pressure | terms_100000 | 0.01661379299912369 | n/a | 0.001698971000223537 | 0.0007985159991221735 |
| stress | simplify_duplicate_pressure | terms_500000 | 0.082309041001281 | n/a | 0.0038427189992944477 | 0.0015867060010350542 |
| stress | simplify_duplicate_pressure | terms_1000000 | 0.17525251699953515 | n/a | 0.008932115000789054 | 0.0014458799996646121 |
| stress | statevector_expectation | qubits_14_terms_4096 | 0.28041803300038737 | n/a | 0.00032497999927727506 | 0.00024423099966952577 |
| stress | statevector_expectation | qubits_15_terms_4096 | 0.5752606570003991 | n/a | 0.000563177000003634 | 0.00047070700020412914 |
| stress | statevector_expectation | qubits_16_terms_8192 | 2.2684815550001076 | n/a | 0.0016872380001586862 | 0.0015726180008641677 |
| stress | pairwise_commutation | terms_4096x4096 | 0.08218452400069509 | tbb: 0.008858979999786243; avx512: 0.014770437999686692; avx2: 0.017364588999043917 | 0.004756305001137662 | 0.005222279000008712 |
| stress | pairwise_commutation | terms_8192x8192 | 0.3867647079987364 | tbb: 0.07704558399927919; avx512: 0.08876326199970208; avx2: 0.11943056999916735 | 0.02011710300030245 | 0.02159383800062642 |
| stress | pairwise_commutation | terms_10000x10000 | 0.5720454940001218 | tbb: 0.11302793100003328; avx512: 0.13631571900077688; avx2: 0.17683624500023143 | 0.029791451001074165 | 0.029300479000085033 |
| stress | matmul_product_generation_simplify | terms_512x512 | 0.051745244998528506 | n/a | 0.004252971000823891 | 0.0014445950000663288 |
| stress | matmul_product_generation_simplify | terms_1024x1024 | 0.2399669989990798 | n/a | 0.009119892998569412 | 0.002249282999400748 |
| stress | matmul_product_generation_simplify | terms_2048x2048 | 1.0407098470004712 | n/a | 0.018724241001109476 | 0.0052450669991230825 |

## Extreme Scaling Results

The extreme profile is opt-in and intended for GPU-host evidence runs, not for
routine validation. These runs used `--repeat 1 --warmup 0` with correctness
checks enabled.

| Case | Scale | CPU Scalar Seconds | CUDA Transfer Seconds | CUDA Resident Seconds |
| --- | --- | ---: | ---: | ---: |
| simplify_duplicate_pressure | terms_2000000 | 0.39179417900049884 | 0.026138247001654236 | 0.0026348480005253805 |
| simplify_duplicate_pressure | terms_5000000 | 1.0509690920007415 | 0.06888328800050658 | 0.005027092000091216 |
| statevector_expectation | qubits_17_terms_8192 | 4.617909521999536 | 0.004109212999537704 | 0.0035290719988552155 |
| statevector_expectation | qubits_18_terms_8192 | 9.33332920599969 | 0.008566586999222636 | 0.007856240999899455 |
| pairwise_commutation | terms_12000x12000 | 0.8307203550011764 | 0.04258426500018686 | 0.045970418001161306 |
| pairwise_commutation | terms_16384x16384 | 1.6096726499999932 | 0.0774501589985448 | 0.07805186399855302 |
| matmul_product_generation_simplify | terms_3072x3072 | 2.437561018999986 | 0.023108228999262792 | 0.010004159999880358 |
| matmul_product_generation_simplify | terms_4096x4096 | 4.37394347100053 | 0.03321117499945103 | 0.016808829999718 |

## CPU SIMD A/B Evidence

The retained AVX2/AVX-512 store change was A/B tested on this x86 H100 host
against the pre-change SIMD kernels. oneTBB remains the fastest CPU selector for
large pairwise commutation, but the SIMD selectors improved materially.

| Backend Case | Baseline 512x512 Seconds | Final 512x512 Seconds | Baseline 2048x2048 Seconds | Final 2048x2048 Seconds |
| --- | ---: | ---: | ---: | ---: |
| forced_avx2_pairwise_commutation | 0.0006742739997207536 | 0.0005680650001522736 | 0.010425256999951671 | 0.007845787000405835 |
| forced_avx512_pairwise_commutation | 0.0006459870000981027 | 0.00040270399949804414 | 0.01037464999990334 | 0.005573616999754449 |
| forced_tbb_pairwise_commutation | 0.00019482900006551063 | 0.00020219399993948173 | 0.0032628609997118474 | 0.0028371039998091874 |

Final CPU dispatch spot-check at 2048x2048 after all retained changes:

| Case | Seconds |
| --- | ---: |
| forced_scalar_pairwise_commutation | 0.051160040999093326 |
| auto_pairwise_commutation | 0.0031819849991734372 |
| forced_tbb_pairwise_commutation | 0.003183407001415617 |
| forced_avx2_pairwise_commutation | 0.009642367000196828 |
| forced_avx512_pairwise_commutation | 0.0067383999994490296 |

## Baseline Comparison

Compared with `cuda_h100_performance_hardening_2026-04-28.md`, the retained CUDA
changes improve the main transfer-inclusive hot paths. Some resident timings are
close enough to benchmark noise that they should be read as flat rather than as
meaningful regressions.

Default profile:

| Case | Previous Transfer Seconds | Final Transfer Seconds | Transfer Ratio | Previous Resident Seconds | Final Resident Seconds | Resident Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| simplify_duplicate_pressure | 0.001901285 | 0.001003502999083139 | 1.895x | 0.000574626 | 0.0005667500008712523 | 1.014x |
| statevector_expectation | 0.001002643 | 0.00010884299990721047 | 9.212x | 6.674e-05 | 5.480799882207066e-05 | 1.218x |
| pairwise_commutation | 0.004051443 | 0.0014767209995625308 | 2.744x | 0.002237863 | 0.00046763099999225233 | 4.786x |
| matmul_product_generation_simplify | 0.003554215 | 0.0016741420004109386 | 2.123x | 0.000764082 | 0.0009729140001581982 | 0.785x |

Stress profile:

| Case | Previous Transfer Seconds | Final Transfer Seconds | Transfer Ratio | Previous Resident Seconds | Final Resident Seconds | Resident Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| simplify_duplicate_pressure | 0.002529179 | 0.0017232500013051322 | 1.468x | 0.000805897 | 0.0009203980007441714 | 0.876x |
| statevector_expectation | 0.001243973 | 0.00032775600084278267 | 3.795x | 0.000206784 | 0.00020130499979131855 | 1.027x |
| pairwise_commutation | 0.068881765 | 0.02054997899904265 | 3.352x | 0.066086858 | 0.01972011199904955 | 3.351x |
| matmul_product_generation_simplify | 0.006328309 | 0.004567592999592307 | 1.385x | 0.00143085 | 0.0017721420008456334 | 0.807x |

## Interpretation

The largest retained CUDA wins came from removing host overhead around small
kernels, eliminating a duplicate Python host copy for commutation results, and
registering large host outputs before dense commutation result copies. The
largest retained CPU win came from avoiding stack spill/scalar lane stores in
AVX2 and AVX-512 commutation output materialization.

The follow-up experiments did not find a defensible retained synchronization,
mapped-output, no-registration, PTX/SASS, packed-output, block-size, or
async-allocation change beyond the retained commits. Large dense commutation is
now mainly bounded by the public API requirement to return a dense host bool
array and by host registration/copy cost. Matmul with simplification is mainly
bounded by Thrust/CUB sort/reduce work rather than the custom product kernel.
Future performance work should therefore prioritize a device-resident
commutation result API, reusable CUDA/Thrust temporary storage, or a custom
duplicate-reduce pipeline only if those interface and maintenance costs are
accepted.

The fixed, scaling, and extreme profiles show CUDA beating scalar CPU and all
captured optimized CPU selectors for the benchmarked CUDA workloads on this H100
host, except where oneTBB remains faster than CUDA for some small/default CPU
commutation cases. This is source-build evidence for this machine only. It is
not a portable CUDA wheel claim and should not be generalized to other GPU
architectures without the hardware-target evidence required by
`docs/architecture/hardware_targets_and_testing.md`.

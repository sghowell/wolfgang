# FastPauli CPU Evidence - Apple Silicon M4 Pro - 2026-04-25

## Summary

- Git commit: `a4eaa30`
- CPU: Apple M4 Pro
- Architecture: arm64
- OS: macOS-26.2-arm64-arm-64bit
- Compiler: Apple clang version 17.0.0 (clang-1700.6.3.2)
- Compiled CPU backends: scalar, neon
- Apple hardware: chip=Apple M4 Pro, core_summary=12 (8 performance and 4 efficiency), model_identifier=Mac16,11, model_name=Mac mini, total_core_count=12
- Available CPU backends: scalar, neon
- Unavailable CPU backends: avx2=not_compiled, avx512=not_compiled, sve=not_compiled, tbb=not_compiled
- oneTBB: enabled=False, version=not_available
- Thread settings: MKL_NUM_THREADS=unset, OMP_NUM_THREADS=unset, controlled_thread_count=not_controlled
- Auto-dispatch thresholds: tbb_pairwise_entries=331776

## Optimized Kernel Coverage

- avx2: none
- avx512: none
- neon: commutes_with_words_1_2, full_group_commutation_graph_words_1_2
- sve: none
- tbb: none

## Commands

- Dispatch: `<private-path> benchmarks/bench_cpu_dispatch.py --repeat 3 --warmup 1 --json`
- Thresholds: `<private-path> benchmarks/bench_cpu_thresholds.py --repeat 3 --warmup 1 --json`
- Hardening default: `<private-path> benchmarks/bench_cpu_hardening.py --profile default --repeat 3 --warmup 1 --json`
- Hardening stress: `<private-path> benchmarks/bench_cpu_hardening.py --profile stress --repeat 1 --warmup 1 --json`
- Competitive: `<private-path> benchmarks/bench_competitive_baselines.py --repeat 3 --warmup 1 --json`

## Dispatch Benchmark

| Case | Dataset | Matrix Entries | Backend Hint | Seconds | Correct |
| --- | --- | ---: | --- | ---: | --- |
| auto_statevector_expectation | num_qubits=10; num_terms=128; statevector_length=1024; random_seed=6211; coefficient_dtype=complex128 | n/a | scalar | 0.000142708 | n/a |
| forced_scalar_statevector_expectation | num_qubits=10; num_terms=128; statevector_length=1024; random_seed=6211; coefficient_dtype=complex128 | n/a | scalar | 0.000140875 | n/a |
| forced_scalar_pairwise_commutation | num_qubits=65; lhs_terms=128; rhs_terms=128; matrix_entries=16384; random_seed=7211; coefficient_dtype=complex128 | 16384 | scalar | 5.2291e-05 | True |
| auto_pairwise_commutation | num_qubits=65; lhs_terms=128; rhs_terms=128; matrix_entries=16384; random_seed=7211; coefficient_dtype=complex128 | 16384 | neon | 1.5416e-05 | True |
| forced_neon_pairwise_commutation | num_qubits=65; lhs_terms=128; rhs_terms=128; matrix_entries=16384; random_seed=7211; coefficient_dtype=complex128 | 16384 | neon | 1.5167e-05 | True |
| forced_scalar_full_grouping | num_qubits=65; num_terms=128; grouping_mode=full; strategy=largest_first; random_seed=8211; coefficient_dtype=complex128 | n/a | scalar | 2.5333e-05 | True |
| auto_full_grouping | num_qubits=65; num_terms=128; grouping_mode=full; strategy=largest_first; random_seed=8211; coefficient_dtype=complex128 | n/a | neon | 2.8083e-05 | True |
| forced_neon_full_grouping | num_qubits=65; num_terms=128; grouping_mode=full; strategy=largest_first; random_seed=8211; coefficient_dtype=complex128 | n/a | neon | 2.7458e-05 | True |
| optimized_backend_availability | not_recorded | n/a | scalar | not_recorded | n/a |

## Threshold Characterization

| Case | Dataset | Entries | Region | Auto Hint | Scalar Seconds | Auto Seconds | Optimized Seconds | Correct |
| --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |
| below_threshold | num_qubits=65; lhs_terms=288; rhs_terms=288; matrix_entries=82944; random_seed=90185; coefficient_dtype=complex128 | 82944 | below | neon | 0.000270833 | 7.0917e-05 | neon=7.4333e-05 | True |
| at_threshold | num_qubits=65; lhs_terms=576; rhs_terms=576; matrix_entries=331776; random_seed=339017; coefficient_dtype=complex128 | 331776 | at_or_above | neon | 0.00108292 | 0.000309709 | neon=0.000300417 | True |
| above_threshold | num_qubits=65; lhs_terms=1152; rhs_terms=1152; matrix_entries=1327104; random_seed=1334345; coefficient_dtype=complex128 | 1327104 | at_or_above | neon | 0.00431792 | 0.00122896 | neon=0.00109375 | True |

## CPU Hardening

| Profile | Operation | Case | Dataset | FastPauli Seconds | Baseline Seconds | Correctness Checked |
| --- | --- | --- | --- | ---: | ---: | --- |
| default | simplify | low_duplicate | num_qubits=64; num_terms=10000; term_weight=4; duplicate_rate=0.05; random_seed=1729; coefficient_dtype=complex128 | 0.000533709 | 0.0481537 | True |
| default | simplify | high_duplicate | num_qubits=64; num_terms=10000; term_weight=4; duplicate_rate=0.9; random_seed=1730; coefficient_dtype=complex128 | 0.000408459 | 0.0442754 | True |
| default | multiply | single_term | num_qubits=64; lhs_terms=1; rhs_terms=1; term_weight_distribution=fixed single active qubit; duplicate_rate=0.0; lhs_duplicate_rate=0.0; rhs_duplicate_rate=0.0; random_seed=deterministic_single_term; coefficient_dtype=complex128 | 1.12504e-06 | 5.62501e-06 | True |
| default | multiply | small_cross_product | num_qubits=64; lhs_terms=256; rhs_terms=256; term_weight_distribution=fixed term_weight=4; duplicate_rate=0.0; lhs_duplicate_rate=0.0; rhs_duplicate_rate=0.0; random_seed=2753; coefficient_dtype=complex128 | 0.000206125 | 0.239739 | True |
| default | multiply | simplified_duplicate_cross_product | num_qubits=64; lhs_terms=256; rhs_terms=256; term_weight_distribution=two repeated one-local Pauli pools; duplicate_rate=0.9921875; lhs_duplicate_rate=0.9921875; rhs_duplicate_rate=0.9921875; random_seed=2773; coefficient_dtype=complex128 | 0.000256625 | 0.534021 | True |
| default | grouping | pairwise_commutation | num_qubits=64; lhs_terms=256; rhs_terms=256; term_weight_distribution=fixed term_weight=4; lhs_duplicate_rate=0.0; rhs_duplicate_rate=0.0; matrix_entries=65536; random_seed=3181; coefficient_dtype=complex128 | 4.525e-05 | 0.188276 | True |
| default | grouping | qwc_grouping | num_qubits=64; num_terms=512; term_weight_distribution=fixed term_weight=4; duplicate_rate=0.0; grouping_mode=qwc; strategy=largest_first; random_seed=3191; coefficient_dtype=complex128 | 6.175e-05 | 0.0300682 | True |
| default | grouping | full_grouping | num_qubits=64; num_terms=512; term_weight_distribution=fixed term_weight=4; duplicate_rate=0.0; grouping_mode=full; strategy=largest_first; random_seed=3201; coefficient_dtype=complex128 | 0.000201958 | 0.100664 | True |
| default | grouping | guardrail_rejection | num_qubits=1; lhs_terms=3; rhs_terms=4; matrix_entries=12 | 1.3666e-05 | 8.39937e-08 | True |
| default | expectation | statevector_few_terms_large_state | num_qubits=12; num_terms=8; term_weight_distribution=fixed term_weight=3; duplicate_rate=0.0; statevector_length=4096; operator_random_seed=4211; statevector_random_seed=4221; coefficient_dtype=complex128 | 3.9334e-05 | 0.00763275 | True |
| default | expectation | statevector_many_terms_small_state | num_qubits=7; num_terms=512; term_weight_distribution=fixed term_weight=2; duplicate_rate=0.65625; statevector_length=128; operator_random_seed=4212; statevector_random_seed=4222; coefficient_dtype=complex128 | 7.7209e-05 | 0.0139986 | True |
| default | expectation | statevector_diagonal_many_terms | num_qubits=7; num_terms=512; term_weight_distribution=fixed diagonal term_weight=3; duplicate_rate=0.931640625; statevector_length=128; operator_random_seed=4214; statevector_random_seed=4224; coefficient_dtype=complex128 | 7.95798e-06 | 0.0143524 | True |
| default | expectation | z_counts | num_qubits=12; num_terms=128; term_weight_distribution=fixed diagonal term_weight=3; duplicate_rate=0.2421875; operator_random_seed=4213; counts_random_seed=4223; coefficient_dtype=complex128 | 8.8834e-05 | 0.0222561 | True |
| stress | simplify | low_duplicate | num_qubits=65; num_terms=20000; term_weight=4; duplicate_rate=0.05; random_seed=1729; coefficient_dtype=complex128 | 0.00126992 | 0.100826 | True |
| stress | simplify | high_duplicate | num_qubits=65; num_terms=20000; term_weight=4; duplicate_rate=0.9; random_seed=1730; coefficient_dtype=complex128 | 0.000599167 | 0.0918995 | True |
| stress | multiply | single_term | num_qubits=65; lhs_terms=1; rhs_terms=1; term_weight_distribution=fixed single active qubit; duplicate_rate=0.0; lhs_duplicate_rate=0.0; rhs_duplicate_rate=0.0; random_seed=deterministic_single_term; coefficient_dtype=complex128 | 1.95904e-06 | 7.54197e-06 | True |
| stress | multiply | small_cross_product | num_qubits=65; lhs_terms=256; rhs_terms=256; term_weight_distribution=fixed term_weight=4; duplicate_rate=0.0; lhs_duplicate_rate=0.0; rhs_duplicate_rate=0.0; random_seed=2753; coefficient_dtype=complex128 | 0.000566292 | 0.25048 | True |
| stress | multiply | simplified_duplicate_cross_product | num_qubits=65; lhs_terms=256; rhs_terms=256; term_weight_distribution=two repeated one-local Pauli pools; duplicate_rate=0.9921875; lhs_duplicate_rate=0.9921875; rhs_duplicate_rate=0.9921875; random_seed=2773; coefficient_dtype=complex128 | 0.000361042 | 0.541427 | True |
| stress | grouping | pairwise_commutation | num_qubits=65; lhs_terms=512; rhs_terms=512; term_weight_distribution=fixed term_weight=4; lhs_duplicate_rate=0.0; rhs_duplicate_rate=0.0; matrix_entries=262144; random_seed=3181; coefficient_dtype=complex128 | 0.000275291 | 0.74211 | True |
| stress | grouping | qwc_grouping | num_qubits=65; num_terms=1024; term_weight_distribution=fixed term_weight=4; duplicate_rate=0.0; grouping_mode=qwc; strategy=largest_first; random_seed=3191; coefficient_dtype=complex128 | 0.000176583 | 0.107177 | True |
| stress | grouping | full_grouping | num_qubits=65; num_terms=1024; term_weight_distribution=fixed term_weight=4; duplicate_rate=0.0; grouping_mode=full; strategy=largest_first; random_seed=3201; coefficient_dtype=complex128 | 0.00112313 | 0.352187 | True |
| stress | grouping | guardrail_rejection | num_qubits=1; lhs_terms=3; rhs_terms=4; matrix_entries=12 | 1.5417e-05 | 2.07976e-07 | True |
| stress | expectation | statevector_few_terms_large_state | num_qubits=14; num_terms=16; term_weight_distribution=fixed term_weight=3; duplicate_rate=0.0; statevector_length=16384; operator_random_seed=4211; statevector_random_seed=4221; coefficient_dtype=complex128 | 0.000302125 | 0.0647369 | True |
| stress | expectation | statevector_many_terms_small_state | num_qubits=8; num_terms=1024; term_weight_distribution=fixed term_weight=2; duplicate_rate=0.7587890625; statevector_length=256; operator_random_seed=4212; statevector_random_seed=4222; coefficient_dtype=complex128 | 0.000322 | 0.0622623 | True |
| stress | expectation | statevector_diagonal_many_terms | num_qubits=8; num_terms=1024; term_weight_distribution=fixed diagonal term_weight=3; duplicate_rate=0.9453125; statevector_length=256; operator_random_seed=4214; statevector_random_seed=4224; coefficient_dtype=complex128 | 1.575e-05 | 0.0638686 | True |
| stress | expectation | z_counts | num_qubits=16; num_terms=512; term_weight_distribution=fixed diagonal term_weight=3; duplicate_rate=0.345703125; operator_random_seed=4213; counts_random_seed=4223; coefficient_dtype=complex128 | 0.00116487 | 0.441624 | True |

## Competitive Baselines

| Case | Dataset | Competitor | Competitor Available | FastPauli Seconds | Competitor Seconds | Correctness Checked |
| --- | --- | --- | --- | ---: | ---: | --- |
| simplify | num_qubits=64; num_terms=10000; term_weight=4; duplicate_rate=0.875; random_seed=5153; coefficient_dtype=complex128; competitor=qiskit.SparsePauliOp.simplify | qiskit.SparsePauliOp.simplify | True | 0.000483125 | 0.000301208 | True |
| multiply | num_qubits=64; lhs_terms=256; rhs_terms=256; term_weight=4; lhs_duplicate_rate=0.0; rhs_duplicate_rate=0.0; random_seed=5163; coefficient_dtype=complex128; competitor=openfermion.QubitOperator.__mul__ | openfermion.QubitOperator.__mul__ | True | 0.00473642 | 0.0765724 | True |
| qiskit_grouping | num_qubits=64; num_terms=512; term_weight=4; duplicate_rate=0.0; grouping_mode=full; random_seed=5173; coefficient_dtype=complex128; competitor=qiskit.SparsePauliOp.group_commuting | qiskit.SparsePauliOp.group_commuting | True | 0.000218958 | 0.0407864 | True |

## Limitations

Apple Silicon report from local macOS arm64 hardware. oneTBB, AVX2, AVX-512, and SVE were not compiled on this target; NEON was compiled and available. Timings are local development measurements, not release claims.

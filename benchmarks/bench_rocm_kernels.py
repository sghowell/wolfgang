#!/usr/bin/env python3
"""Deterministic ROCm/HIP kernel benchmark smoke and scaling profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import wolfgang_quantum as wolfgang
import wolfgang_quantum._wolfgang_core as core

try:
    from _benchmark_metadata import benchmark_environment, command_string, git_commit
except ModuleNotFoundError:
    from benchmarks._benchmark_metadata import (
        benchmark_environment,
        command_string,
        git_commit,
    )


SMOKE_CASES = [
    {
        "name": "smoke_single_word",
        "num_qubits": 8,
        "lhs_terms": 128,
        "rhs_terms": 128,
        "term_weight": 2,
        "random_seed": 9421,
    },
]

SIMPLIFY_SMOKE_CASES = [
    {
        "name": "campaign3_smoke_one_word",
        "num_qubits": 8,
        "num_terms": 128,
        "term_weight": 2,
        "duplicate_rate": 0.25,
        "random_seed": 39421,
    },
]

SIMPLIFY_DUPLICATE_PRESSURE_CASES = [
    {
        "name": "campaign3_duplicate_heavy",
        "num_qubits": 24,
        "num_terms": 32768,
        "term_weight": 4,
        "duplicate_rate": 0.875,
        "random_seed": 39422,
    },
    {
        "name": "campaign3_duplicate_light",
        "num_qubits": 24,
        "num_terms": 32768,
        "term_weight": 4,
        "duplicate_rate": 0.0625,
        "random_seed": 39423,
    },
    {
        "name": "campaign3_all_zero",
        "num_qubits": 24,
        "num_terms": 4096,
        "term_weight": 4,
        "duplicate_rate": 1.0,
        "random_seed": 39424,
        "coefficient_mode": "cancelling_duplicates",
    },
]

SIMPLIFY_WIDE_QUBIT_CASES = [
    {
        "name": "campaign3_wide_two_word",
        "num_qubits": 70,
        "num_terms": 8192,
        "term_weight": 6,
        "duplicate_rate": 0.25,
        "random_seed": 39425,
    },
    {
        "name": "campaign3_generic_multiword",
        "num_qubits": 130,
        "num_terms": 4096,
        "term_weight": 8,
        "duplicate_rate": 0.25,
        "random_seed": 39426,
    },
]

SIMPLIFY_CAMPAIGN4_BASELINE_CASES = [
    {
        "name": "campaign4_duplicate_heavy_default",
        "num_qubits": 24,
        "num_terms": 32768,
        "term_weight": 4,
        "duplicate_rate": 0.875,
        "random_seed": 49422,
        "hip_simplify_strategy": "rocthrust_default",
    },
    {
        "name": "campaign4_wide_two_word_default",
        "num_qubits": 70,
        "num_terms": 8192,
        "term_weight": 6,
        "duplicate_rate": 0.25,
        "random_seed": 49425,
        "hip_simplify_strategy": "rocthrust_default",
    },
    {
        "name": "campaign4_generic_serial_baseline",
        "num_qubits": 130,
        "num_terms": 4096,
        "term_weight": 8,
        "duplicate_rate": 0.25,
        "random_seed": 49426,
        "hip_simplify_strategy": "rocthrust_default",
        "generic_multiword_parallelism": "serial_kernel",
    },
    {
        "name": "campaign4_generic_parallel_baseline",
        "num_qubits": 130,
        "num_terms": 4096,
        "term_weight": 8,
        "duplicate_rate": 0.25,
        "random_seed": 49426,
        "hip_simplify_strategy": "rocthrust_generic_parallel_reduce_by_key",
        "generic_multiword_parallelism": "reduce_by_key",
    },
]

SIMPLIFY_CAMPAIGN4_CUSTOM_KEY_CASES = [
    {
        "name": "campaign4_custom_packed32_duplicate_heavy",
        "num_qubits": 24,
        "num_terms": 32768,
        "term_weight": 4,
        "duplicate_rate": 0.875,
        "random_seed": 49428,
        "hip_simplify_strategy": "custom_packed_key",
    },
    {
        "name": "campaign4_custom_key1_wide_one_word",
        "num_qubits": 48,
        "num_terms": 32768,
        "term_weight": 6,
        "duplicate_rate": 0.25,
        "random_seed": 49429,
        "hip_simplify_strategy": "custom_packed_key",
    },
    {
        "name": "campaign4_custom_key2_wide_two_word",
        "num_qubits": 70,
        "num_terms": 8192,
        "term_weight": 6,
        "duplicate_rate": 0.25,
        "random_seed": 49430,
        "hip_simplify_strategy": "custom_packed_key",
    },
]

SIMPLIFY_CAMPAIGN4_GENERIC_CASES = [
    {
        "name": "campaign4_generic_130q_4096t_parallel",
        "num_qubits": 130,
        "num_terms": 4096,
        "term_weight": 8,
        "duplicate_rate": 0.25,
        "random_seed": 49431,
        "hip_simplify_strategy": "rocthrust_generic_parallel_reduce_by_key",
        "generic_multiword_parallelism": "reduce_by_key",
    },
    {
        "name": "campaign4_generic_193q_8192t_parallel",
        "num_qubits": 193,
        "num_terms": 8192,
        "term_weight": 10,
        "duplicate_rate": 0.25,
        "random_seed": 49432,
        "hip_simplify_strategy": "rocthrust_generic_parallel_reduce_by_key",
        "generic_multiword_parallelism": "reduce_by_key",
    },
    {
        "name": "campaign4_generic_193q_32768t_parallel",
        "num_qubits": 193,
        "num_terms": 32768,
        "term_weight": 10,
        "duplicate_rate": 0.875,
        "random_seed": 49433,
        "hip_simplify_strategy": "rocthrust_generic_parallel_reduce_by_key",
        "generic_multiword_parallelism": "reduce_by_key",
    },
    {
        "name": "campaign4_generic_257q_4096t_parallel",
        "num_qubits": 257,
        "num_terms": 4096,
        "term_weight": 12,
        "duplicate_rate": 0.0625,
        "random_seed": 49434,
        "hip_simplify_strategy": "rocthrust_generic_parallel_reduce_by_key",
        "generic_multiword_parallelism": "reduce_by_key",
    },
]

SIMPLIFY_CAMPAIGN4_WORKSPACE_CASES = [
    {
        "name": "campaign4_workspace_rocthrust_duplicate_heavy",
        "num_qubits": 24,
        "num_terms": 32768,
        "term_weight": 4,
        "duplicate_rate": 0.875,
        "random_seed": 49435,
        "hip_simplify_strategy": "rocprim_scratch_probe",
        "hip_workspace_mode": "unavailable",
    },
    {
        "name": "campaign4_workspace_rocthrust_generic_parallel",
        "num_qubits": 193,
        "num_terms": 8192,
        "term_weight": 10,
        "duplicate_rate": 0.25,
        "random_seed": 49436,
        "hip_simplify_strategy": "hipcub_scratch_probe",
        "generic_multiword_parallelism": "reduce_by_key",
        "hip_workspace_mode": "unavailable",
    },
]

SIMPLIFY_CAMPAIGN4_PROFILER_CASES = [
    {
        "name": "campaign4_profiler_generic_193q_32768t_parallel",
        "num_qubits": 193,
        "num_terms": 32768,
        "term_weight": 10,
        "duplicate_rate": 0.875,
        "random_seed": 49437,
        "hip_simplify_strategy": "rocthrust_generic_parallel_reduce_by_key",
        "generic_multiword_parallelism": "reduce_by_key",
    },
]

CAMPAIGN5_INTEROP_CASES = [
    {
        "name": "campaign5_dlpack_consumer_mid",
        "num_qubits": 64,
        "lhs_terms": 512,
        "rhs_terms": 512,
        "term_weight": 6,
        "random_seed": 59421,
    },
]

CAMPAIGN5_PROFILER_CASES = [
    {
        "name": "campaign5_profiler_dlpack_boundary",
        "num_qubits": 64,
        "lhs_terms": 1024,
        "rhs_terms": 1024,
        "term_weight": 6,
        "random_seed": 59422,
    },
]

CAMPAIGN5_DECISION_CASES = [
    {
        "name": "campaign5_execution_control_decisions",
    },
]

CAMPAIGN6_EXPECTATION_CASES = [
    {
        "name": "campaign6_expectation_two_qubit_complex128",
        "num_qubits": 2,
        "num_terms": 5,
        "term_weight": 2,
        "statevector_dtype": "complex128",
        "random_seed": 69421,
    },
    {
        "name": "campaign6_expectation_ten_qubit_complex64",
        "num_qubits": 10,
        "num_terms": 256,
        "term_weight": 4,
        "statevector_dtype": "complex64",
        "random_seed": 69422,
    },
]

CAMPAIGN6_MATMUL_CASES = [
    {
        "name": "campaign6_matmul_one_word_simplify_true",
        "num_qubits": 24,
        "lhs_terms": 256,
        "rhs_terms": 256,
        "term_weight": 4,
        "duplicate_rate": 0.25,
        "simplify_output": True,
        "random_seed": 69431,
    },
    {
        "name": "campaign6_matmul_two_word_simplify_false",
        "num_qubits": 70,
        "lhs_terms": 64,
        "rhs_terms": 64,
        "term_weight": 6,
        "duplicate_rate": 0.0,
        "simplify_output": False,
        "random_seed": 69432,
    },
]

CAMPAIGN6_PROFILER_CASES = [
    {
        "name": "campaign6_profiler_expectation",
        "num_qubits": 14,
        "num_terms": 1024,
        "term_weight": 6,
        "statevector_dtype": "complex128",
        "random_seed": 69441,
    },
    {
        "name": "campaign6_profiler_matmul",
        "num_qubits": 70,
        "lhs_terms": 128,
        "rhs_terms": 128,
        "term_weight": 6,
        "duplicate_rate": 0.25,
        "simplify_output": True,
        "random_seed": 69442,
    },
]

CAMPAIGN7_RELEASE_SMOKE_CASES = [
    {
        "name": "campaign7_transfer_roundtrip",
        "campaign7_operation": "transfer",
        "num_qubits": 24,
        "lhs_terms": 256,
        "rhs_terms": 128,
        "term_weight": 4,
        "random_seed": 79421,
    },
    {
        "name": "campaign7_retained_commutation_consumers",
        "campaign7_operation": "commutation_consumers",
        "num_qubits": 64,
        "lhs_terms": 512,
        "rhs_terms": 512,
        "term_weight": 6,
        "random_seed": 79422,
    },
    {
        "name": "campaign7_retained_simplify",
        "campaign7_operation": "simplify",
        "num_qubits": 130,
        "num_terms": 4096,
        "term_weight": 8,
        "duplicate_rate": 0.75,
        "random_seed": 79423,
        "hip_simplify_strategy": "rocthrust_generic_parallel_reduce_by_key",
        "generic_multiword_parallelism": "reduce_by_key",
    },
    {
        "name": "campaign7_retained_expectation",
        "campaign7_operation": "expectation",
        "num_qubits": 10,
        "num_terms": 256,
        "term_weight": 4,
        "statevector_dtype": "complex128",
        "random_seed": 79424,
    },
    {
        "name": "campaign7_retained_matmul",
        "campaign7_operation": "matmul",
        "num_qubits": 70,
        "lhs_terms": 128,
        "rhs_terms": 128,
        "term_weight": 6,
        "duplicate_rate": 0.25,
        "simplify_output": True,
        "random_seed": 79425,
    },
    {
        "name": "campaign7_release_decisions",
        "campaign7_operation": "decisions",
    },
]

CAMPAIGN7_DUPLICATE_PRESSURE_CASES = [
    {
        "name": "campaign7_simplify_duplicate_pressure",
        "campaign7_operation": "simplify_duplicate_pressure",
        "num_qubits": 193,
        "num_terms": 32768,
        "term_weight": 10,
        "duplicate_rate": 0.875,
        "random_seed": 79431,
        "hip_simplify_strategy": "rocthrust_generic_parallel_reduce_by_key",
        "generic_multiword_parallelism": "reduce_by_key",
    },
    {
        "name": "campaign7_matmul_duplicate_pressure",
        "campaign7_operation": "matmul_duplicate_pressure",
        "num_qubits": 70,
        "lhs_terms": 256,
        "rhs_terms": 256,
        "term_weight": 6,
        "duplicate_rate": 0.5,
        "simplify_output": True,
        "random_seed": 79432,
    },
]

CAMPAIGN7_PROFILER_CASES = [
    {
        "name": "campaign7_profiler_simplify",
        "campaign7_operation": "simplify",
        "num_qubits": 130,
        "num_terms": 8192,
        "term_weight": 8,
        "duplicate_rate": 0.75,
        "random_seed": 79441,
        "hip_simplify_strategy": "rocthrust_generic_parallel_reduce_by_key",
        "generic_multiword_parallelism": "reduce_by_key",
    },
    {
        "name": "campaign7_profiler_expectation",
        "campaign7_operation": "expectation",
        "num_qubits": 14,
        "num_terms": 1024,
        "term_weight": 6,
        "statevector_dtype": "complex128",
        "random_seed": 79442,
    },
    {
        "name": "campaign7_profiler_matmul",
        "campaign7_operation": "matmul",
        "num_qubits": 70,
        "lhs_terms": 128,
        "rhs_terms": 128,
        "term_weight": 6,
        "duplicate_rate": 0.25,
        "simplify_output": True,
        "random_seed": 79443,
    },
]

PROFILE_CASES = {
    "smoke": SMOKE_CASES,
    "simplify-smoke": SIMPLIFY_SMOKE_CASES,
    "simplify-duplicate-pressure": SIMPLIFY_DUPLICATE_PRESSURE_CASES,
    "simplify-wide-qubit": SIMPLIFY_WIDE_QUBIT_CASES,
    "simplify-campaign3-profiler": [
        {
            "name": "campaign3_profiler_duplicate_heavy",
            "num_qubits": 24,
            "num_terms": 32768,
            "term_weight": 4,
            "duplicate_rate": 0.875,
            "random_seed": 39427,
        },
    ],
    "simplify-strategy-ab": [
        {
            "name": "campaign3_strategy_duplicate_heavy",
            "num_qubits": 24,
            "num_terms": 32768,
            "term_weight": 4,
            "duplicate_rate": 0.875,
            "random_seed": 39428,
        },
    ],
    "simplify-campaign4-baseline": SIMPLIFY_CAMPAIGN4_BASELINE_CASES,
    "simplify-campaign4-custom-key-ab": SIMPLIFY_CAMPAIGN4_CUSTOM_KEY_CASES,
    "simplify-campaign4-generic-multiword": SIMPLIFY_CAMPAIGN4_GENERIC_CASES,
    "simplify-campaign4-workspace-ab": SIMPLIFY_CAMPAIGN4_WORKSPACE_CASES,
    "simplify-campaign4-profiler": SIMPLIFY_CAMPAIGN4_PROFILER_CASES,
    "interop-campaign5-dlpack-consumers": CAMPAIGN5_INTEROP_CASES,
    "interop-campaign5-stream-workspace-decisions": CAMPAIGN5_DECISION_CASES,
    "interop-campaign5-profiler": CAMPAIGN5_PROFILER_CASES,
    "campaign6-expectation-parity": CAMPAIGN6_EXPECTATION_CASES,
    "campaign6-matmul-parity": CAMPAIGN6_MATMUL_CASES,
    "campaign6-profiler": CAMPAIGN6_PROFILER_CASES,
    "campaign7-release-smoke": CAMPAIGN7_RELEASE_SMOKE_CASES,
    "campaign7-duplicate-pressure": CAMPAIGN7_DUPLICATE_PRESSURE_CASES,
    "campaign7-profiler": CAMPAIGN7_PROFILER_CASES,
    "commutation-device-output-smoke": SMOKE_CASES,
    "commutation-scaling": [
        {
            "name": "small_transfer_bound",
            "num_qubits": 8,
            "lhs_terms": 128,
            "rhs_terms": 128,
            "term_weight": 2,
            "random_seed": 9422,
        },
        {
            "name": "mid_dense_pairs",
            "num_qubits": 64,
            "lhs_terms": 2048,
            "rhs_terms": 2048,
            "term_weight": 6,
            "random_seed": 9423,
        },
        {
            "name": "large_dense_pairs",
            "num_qubits": 128,
            "lhs_terms": 4096,
            "rhs_terms": 4096,
            "term_weight": 8,
            "random_seed": 9424,
        },
    ],
    "commutation-device-output-scaling": [
        {
            "name": "campaign2_small_dense_output",
            "num_qubits": 8,
            "lhs_terms": 128,
            "rhs_terms": 128,
            "term_weight": 2,
            "random_seed": 9522,
        },
        {
            "name": "campaign2_mid_dense_output",
            "num_qubits": 64,
            "lhs_terms": 2048,
            "rhs_terms": 2048,
            "term_weight": 6,
            "random_seed": 9523,
        },
        {
            "name": "campaign2_large_dense_output",
            "num_qubits": 128,
            "lhs_terms": 4096,
            "rhs_terms": 4096,
            "term_weight": 8,
            "random_seed": 9524,
        },
    ],
    "commutation-compact-consumers": [
        {
            "name": "campaign2_compact_mid",
            "num_qubits": 64,
            "lhs_terms": 2048,
            "rhs_terms": 2048,
            "term_weight": 6,
            "random_seed": 9623,
        },
        {
            "name": "campaign2_compact_large",
            "num_qubits": 128,
            "lhs_terms": 4096,
            "rhs_terms": 4096,
            "term_weight": 8,
            "random_seed": 9624,
        },
    ],
    "commutation-profiler": [
        {
            "name": "profiler_dense_pairs",
            "num_qubits": 128,
            "lhs_terms": 4096,
            "rhs_terms": 4096,
            "term_weight": 8,
            "random_seed": 9425,
        },
    ],
    "commutation-campaign2-profiler": [
        {
            "name": "campaign2_profiler_dense_pairs",
            "num_qubits": 128,
            "lhs_terms": 4096,
            "rhs_terms": 4096,
            "term_weight": 8,
            "random_seed": 9725,
        },
    ],
}

OPTIMIZED_CPU_SELECTOR_ORDER = ("tbb", "avx512", "avx2", "neon", "sve")
CAMPAIGN2_PROFILES = {
    "commutation-device-output-smoke",
    "commutation-device-output-scaling",
    "commutation-compact-consumers",
    "commutation-campaign2-profiler",
}

CAMPAIGN3_SIMPLIFY_PROFILES = {
    "simplify-smoke",
    "simplify-duplicate-pressure",
    "simplify-wide-qubit",
    "simplify-campaign3-profiler",
    "simplify-strategy-ab",
}

CAMPAIGN4_SIMPLIFY_PROFILES = {
    "simplify-campaign4-baseline",
    "simplify-campaign4-custom-key-ab",
    "simplify-campaign4-generic-multiword",
    "simplify-campaign4-workspace-ab",
    "simplify-campaign4-profiler",
}

CAMPAIGN3_HEADROOM_STATUSES = {
    "DLPack": "out_of_scope_with_next_trigger",
    "streams": "out_of_scope_with_next_trigger",
    "workspaces": "out_of_scope_with_next_trigger",
    "packed summaries": "out_of_scope_with_next_trigger",
    "expectation": "out_of_scope_with_next_trigger",
    "matmul": "out_of_scope_with_next_trigger",
    "portability": "out_of_scope_with_next_trigger",
    "ROCm wheels": "out_of_scope_with_next_trigger",
    "multi-GPU": "out_of_scope_with_next_trigger",
    "simultaneous CUDA+HIP": "out_of_scope_with_next_trigger",
}

CAMPAIGN4_TERMINAL_STATUSES = {
    "workspace": "rejected_with_evidence",
    "custom packed key": "unavailable",
    "generic multi-word": "retained",
    "DLPack": "out_of_scope_with_next_trigger",
    "streams": "out_of_scope_with_next_trigger",
    "expectation": "out_of_scope_with_next_trigger",
    "matmul": "out_of_scope_with_next_trigger",
    "portability": "MI300X_gfx942_only",
    "ROCm wheels": "out_of_scope_with_next_trigger",
    "multi-GPU": "out_of_scope_with_next_trigger",
    "simultaneous CUDA+HIP": "configure_time_rejected",
}

CAMPAIGN5_PROFILES = {
    "interop-campaign5-dlpack-consumers",
    "interop-campaign5-stream-workspace-decisions",
    "interop-campaign5-profiler",
}

CAMPAIGN5_TERMINAL_STATUSES = {
    "DLPack": "rejected_with_evidence",
    "CUDA Array Interface guard": "rejected_with_evidence",
    "streams": "rejected_with_evidence",
    "graphs": "rejected_with_evidence",
    "workspaces": "rejected_with_evidence",
    "expectation": "out_of_scope_with_next_trigger",
    "matmul": "out_of_scope_with_next_trigger",
    "portability": "out_of_scope_with_next_trigger",
    "ROCm wheels": "out_of_scope_with_next_trigger",
    "multi-GPU": "out_of_scope_with_next_trigger",
    "simultaneous CUDA+HIP": "unavailable",
}

CAMPAIGN6_PROFILES = {
    "campaign6-expectation-parity",
    "campaign6-matmul-parity",
    "campaign6-profiler",
}

CAMPAIGN6_TERMINAL_STATUSES = {
    "expectation": "retained",
    "matmul": "retained",
    "external device pointers": "unavailable",
    "DLPack": "rejected_with_evidence",
    "CUDA Array Interface guard": "rejected_with_evidence",
    "streams": "rejected_with_evidence",
    "graphs": "rejected_with_evidence",
    "workspaces": "rejected_with_evidence",
    "portability": "out_of_scope_with_next_trigger",
    "ROCm wheels": "out_of_scope_with_next_trigger",
    "multi-GPU": "out_of_scope_with_next_trigger",
    "simultaneous CUDA+HIP": "unavailable",
}

CAMPAIGN7_PROFILES = {
    "campaign7-release-smoke",
    "campaign7-duplicate-pressure",
    "campaign7-profiler",
}

CAMPAIGN7_TERMINAL_STATUSES = {
    "mi300x_repeatability": "passed",
    "cpu_only_control": "passed",
    "rocm_source_build_runbook": "retained",
    "rocm_ci_or_release_lane": "retained",
    "rocm_packaging_policy": "retained",
    "rocm_wheel_support": "unavailable",
    "alternate_amd_gpu_portability": "blocked_external",
    "profiler_availability": "passed",
    "duplicate_pressure_simplify": "rejected_with_evidence",
    "duplicate_pressure_matmul": "rejected_with_evidence",
    "external_statevector_interop": "out_of_scope_with_next_trigger",
    "hip_dlpack": "rejected_with_evidence",
    "hip_cuda_array_interface": "rejected_with_evidence",
    "public_streams": "rejected_with_evidence",
    "public_graphs": "rejected_with_evidence",
    "public_workspaces": "rejected_with_evidence",
    "multi_gpu_rocm": "out_of_scope_with_next_trigger",
    "simultaneous_cuda_hip": "unavailable",
    "backend_neutral_accelerator_design": "out_of_scope_with_next_trigger",
}

CAMPAIGN7_HIP_BUILD_COMMAND = (
    "PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH "
    "python -m pip install -e .[test] "
    "--config-settings=cmake.define.WOLFGANG_ENABLE_HIP=ON "
    "--config-settings=cmake.define.WOLFGANG_HIP_ARCHITECTURES=gfx942"
)
CAMPAIGN7_CPU_VALIDATION_COMMAND = "python scripts/validate.py"
CAMPAIGN7_HIP_VALIDATION_COMMAND = (
    "PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH "
    "python -m pytest tests/test_phase12_rocm_foundation.py -q"
)
CAMPAIGN7_PROFILER_COMMAND = (
    "PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH "
    "rocprof --hip-trace --stats python benchmarks/bench_rocm_kernels.py "
    "--profile campaign7-profiler --repeat 1 --warmup 0 --json"
)

CAMPAIGN5_DLPACK_REJECTION_REASON = (
    "PyTorch ROCm 2.13.0.dev20260428+rocm7.2 consumed the candidate "
    "versioned kDLROCM capsule on MI300X but accepted mutation of the "
    "read-only uint8 view, so HIP DLPack remains unavailable."
)

HIP_SIMPLIFY_STRATEGIES = {
    "rocthrust_default",
    "hipcub_radix_sort_reduce",
    "hipcub_scratch_probe",
    "custom_packed_key",
    "rocprim_scratch_probe",
    "rocthrust_generic_parallel_reduce_by_key",
}

UNAVAILABLE_HIP_SIMPLIFY_STRATEGIES = {
    "custom_packed_key": (
        "no distinct lower-level rocPRIM or hipCUB custom packed-key implementation "
        "is retained; Campaign 4 does not time a fallback under this label"
    ),
    "hipcub_radix_sort_reduce": (
        "hipCUB radix-sort/reduce replacement was not implemented for this layout; "
        "Campaign 4 keeps this as an unavailable candidate instead of falling back"
    ),
    "hipcub_scratch_probe": (
        "rocThrust simplify paths do not expose a stable explicit hipCUB scratch-buffer "
        "contract for this implementation"
    ),
    "rocprim_scratch_probe": (
        "rocThrust simplify paths do not expose a stable explicit rocPRIM scratch-buffer "
        "contract for this implementation"
    ),
}


def timing_summary(timings: list[float]) -> dict[str, float]:
    if not timings:
        raise ValueError("at least one timing is required")
    ordered = sorted(timings)

    def nearest_rank(percentile: float) -> float:
        index = round((len(ordered) - 1) * percentile)
        return ordered[index]

    return {
        "median": statistics.median(timings),
        "p10": nearest_rank(0.10),
        "p90": nearest_rank(0.90),
        "min": ordered[0],
        "max": ordered[-1],
    }


def add_timing_fields(row: dict[str, Any], prefix: str, summary: dict[str, float]) -> None:
    row[f"{prefix}_seconds"] = summary["median"]
    row[f"{prefix}_p10_seconds"] = summary["p10"]
    row[f"{prefix}_p90_seconds"] = summary["p90"]
    row[f"{prefix}_min_seconds"] = summary["min"]
    row[f"{prefix}_max_seconds"] = summary["max"]


def add_null_timing_fields(row: dict[str, Any], prefix: str) -> None:
    row[f"{prefix}_seconds"] = None
    row[f"{prefix}_p10_seconds"] = None
    row[f"{prefix}_p90_seconds"] = None
    row[f"{prefix}_min_seconds"] = None
    row[f"{prefix}_max_seconds"] = None


def first_device_field(hip_status: dict[str, Any], field: str) -> str:
    devices = hip_status.get("devices", [])
    if not devices:
        return "not_available"
    return str(devices[0].get(field, "unknown"))


def commutation_digest(expected_array: np.ndarray) -> dict[str, int | list[int]]:
    expected_u64 = expected_array.astype(np.uint64, copy=False)
    commuting_count = int(expected_u64.sum(dtype=np.uint64))
    entries = int(expected_u64.size)
    return {
        "shape": [int(expected_array.shape[0]), int(expected_array.shape[1])],
        "entries": entries,
        "commuting_count": commuting_count,
        "conflict_count": entries - commuting_count,
    }


def campaign5_commutation_digest(
    expected_array: np.ndarray,
    *,
    consumer_sum: int | None = None,
) -> dict[str, Any]:
    expected_u8 = expected_array.astype(np.uint8, copy=False)
    host_sum = int(expected_u8.sum(dtype=np.uint64))
    digest = hashlib.sha256()
    digest.update(np.asarray(expected_u8, order="C").tobytes())
    return {
        "matrix_shape": [int(expected_array.shape[0]), int(expected_array.shape[1])],
        "host_sum": host_sum,
        "consumer_sum": consumer_sum,
        "canonical_matrix_hash": digest.hexdigest(),
    }


def probe_pytorch_rocm_consumer() -> dict[str, Any]:
    code = r"""
import json
try:
    import torch
except Exception as exc:
    print(json.dumps({
        "consumer_library": "torch",
        "consumer_available": False,
        "consumer_import_error": f"{type(exc).__name__}: {exc}",
        "consumer_version": None,
        "consumer_backend": "unavailable",
        "consumer_device_visible": False,
        "consumer_device_error": "",
    }))
    raise SystemExit(0)
hip_version = getattr(torch.version, "hip", None)
device_visible = False
device_error = ""
try:
    device_visible = bool(torch.cuda.is_available())
except Exception as exc:
    device_error = f"{type(exc).__name__}: {exc}"
print(json.dumps({
    "consumer_library": "torch",
    "consumer_available": bool(hip_version and device_visible),
    "consumer_import_error": "",
    "consumer_version": getattr(torch, "__version__", "unknown"),
    "consumer_backend": "rocm" if hip_version else "not_rocm",
    "consumer_hip_version": hip_version,
    "consumer_device_visible": device_visible,
    "consumer_device_error": device_error,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        return {
            "consumer_library": "torch",
            "consumer_available": False,
            "consumer_import_error": (completed.stderr or completed.stdout).strip(),
            "consumer_version": None,
            "consumer_backend": "unavailable",
            "consumer_device_visible": False,
            "consumer_device_error": "",
        }
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return {
            "consumer_library": "torch",
            "consumer_available": False,
            "consumer_import_error": f"invalid probe output: {type(exc).__name__}: {exc}",
            "consumer_version": None,
            "consumer_backend": "unavailable",
            "consumer_device_visible": False,
            "consumer_device_error": "",
        }


def probe_cupy_rocm_consumer() -> dict[str, Any]:
    code = r"""
import json
try:
    import cupy
except Exception as exc:
    print(json.dumps({
        "consumer_library": "cupy",
        "consumer_available": False,
        "consumer_import_error": f"{type(exc).__name__}: {exc}",
        "consumer_version": None,
        "consumer_backend": "unavailable",
        "consumer_device_visible": False,
        "consumer_device_error": "",
    }))
    raise SystemExit(0)
device_visible = False
device_error = ""
try:
    device_visible = int(cupy.cuda.runtime.getDeviceCount()) > 0
except Exception as exc:
    device_error = f"{type(exc).__name__}: {exc}"
backend = "rocm" if "rocm" in str(getattr(cupy, "__file__", "")).lower() else "unknown"
print(json.dumps({
    "consumer_library": "cupy",
    "consumer_available": bool(device_visible),
    "consumer_import_error": "",
    "consumer_version": getattr(cupy, "__version__", "unknown"),
    "consumer_backend": backend,
    "consumer_device_visible": device_visible,
    "consumer_device_error": device_error,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        return {
            "consumer_library": "cupy",
            "consumer_available": False,
            "consumer_import_error": (completed.stderr or completed.stdout).strip(),
            "consumer_version": None,
            "consumer_backend": "unavailable",
            "consumer_device_visible": False,
            "consumer_device_error": "",
        }
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return {
            "consumer_library": "cupy",
            "consumer_available": False,
            "consumer_import_error": f"invalid probe output: {type(exc).__name__}: {exc}",
            "consumer_version": None,
            "consumer_backend": "unavailable",
            "consumer_device_visible": False,
            "consumer_device_error": "",
        }


def timed_call(fn: Callable[[], Any], *, warmup: int, repeat: int) -> tuple[Any, dict[str, float]]:
    result: Any = None
    for _ in range(warmup):
        result = fn()

    timings: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        timings.append(time.perf_counter() - start)

    return result, timing_summary(timings)


@contextmanager
def forced_cpu_backend(selector: str):
    previous = os.environ.get("WOLFGANG_CPU_BACKEND")
    os.environ["WOLFGANG_CPU_BACKEND"] = selector
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("WOLFGANG_CPU_BACKEND", None)
        else:
            os.environ["WOLFGANG_CPU_BACKEND"] = previous


@contextmanager
def forced_environment(updates: dict[str, str | None]):
    previous = {name: os.environ.get(name) for name in updates}
    for name, value in updates.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def deterministic_labels(
    *,
    num_qubits: int,
    num_terms: int,
    term_weight: int,
    seed: int,
) -> list[str]:
    rng = np.random.default_rng(seed)
    paulis = np.asarray(["X", "Y", "Z"])
    labels: list[str] = []
    active_weight = min(term_weight, num_qubits)
    for _ in range(num_terms):
        chars = ["I"] * num_qubits
        positions = rng.choice(num_qubits, size=active_weight, replace=False)
        for qubit in positions:
            chars[num_qubits - 1 - int(qubit)] = str(rng.choice(paulis))
        labels.append("".join(chars))
    return labels


def unique_deterministic_labels(
    *,
    num_qubits: int,
    unique_terms: int,
    term_weight: int,
    seed: int,
) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    attempt = 0
    while len(labels) < unique_terms:
        batch = deterministic_labels(
            num_qubits=num_qubits,
            num_terms=max(unique_terms - len(labels), 32),
            term_weight=term_weight,
            seed=seed + attempt * 104729,
        )
        for label in batch:
            if label in seen:
                continue
            seen.add(label)
            labels.append(label)
            if len(labels) == unique_terms:
                break
        attempt += 1
    return labels


def make_simplify_operator(case: dict[str, Any]) -> wolfgang.PauliSum:
    num_qubits = int(case["num_qubits"])
    num_terms = int(case["num_terms"])
    term_weight = int(case["term_weight"])
    duplicate_rate = float(case["duplicate_rate"])
    seed = int(case["random_seed"])

    if case.get("coefficient_mode") == "cancelling_duplicates":
        label = unique_deterministic_labels(
            num_qubits=num_qubits,
            unique_terms=1,
            term_weight=term_weight,
            seed=seed,
        )[0]
        labels = [label] * num_terms
        coeffs = [1.0 if index % 2 == 0 else -1.0 for index in range(num_terms)]
        return wolfgang.PauliSum.from_labels(labels, coeffs)

    unique_terms = max(1, min(num_terms, int(round(num_terms * (1.0 - duplicate_rate)))))
    pool = unique_deterministic_labels(
        num_qubits=num_qubits,
        unique_terms=unique_terms,
        term_weight=term_weight,
        seed=seed,
    )
    labels = [pool[index % unique_terms] for index in range(num_terms)]
    rng = np.random.default_rng(seed + 17)
    coeffs = [complex(float(rng.normal()), float(rng.normal())) for _ in range(num_terms)]
    return wolfgang.PauliSum.from_labels(labels, coeffs)


def make_operator(case: dict[str, Any], *, side: str) -> wolfgang.PauliSum:
    terms = int(case[f"{side}_terms"])
    seed_offset = 0 if side == "lhs" else 100000
    labels = deterministic_labels(
        num_qubits=int(case["num_qubits"]),
        num_terms=terms,
        term_weight=int(case["term_weight"]),
        seed=int(case["random_seed"]) + seed_offset,
    )
    coeffs = np.ones(terms, dtype=np.complex128)
    return wolfgang.PauliSum.from_labels(labels, coeffs.tolist())


def labels_and_coefficients(op: wolfgang.PauliSum) -> tuple[list[str], np.ndarray]:
    labels, coeffs = op.to_labels()
    return list(labels), np.asarray(coeffs, dtype=np.complex128)


def simplify_digest(
    *,
    input_terms: int,
    simplified: wolfgang.PauliSum,
) -> dict[str, Any]:
    labels, coeffs = labels_and_coefficients(simplified)
    digest = hashlib.sha256()
    for label, coeff in zip(labels, coeffs, strict=True):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.float64(coeff.real).tobytes())
        digest.update(np.float64(coeff.imag).tobytes())
    return {
        "input_terms": int(input_terms),
        "output_terms": int(simplified.num_terms),
        "coefficient_l1": float(np.abs(coeffs).sum(dtype=np.float64)),
        "canonical_label_hash": digest.hexdigest(),
    }


def assert_simplify_matches(
    actual: wolfgang.PauliSum,
    expected: wolfgang.PauliSum,
    *,
    case_name: str,
) -> None:
    actual_labels, actual_coeffs = labels_and_coefficients(actual)
    expected_labels, expected_coeffs = labels_and_coefficients(expected)
    if actual_labels != expected_labels:
        raise RuntimeError(f"HIP simplify label mismatch for {case_name}")
    if not np.allclose(actual_coeffs, expected_coeffs, rtol=1.0e-12, atol=1.0e-12):
        raise RuntimeError(f"HIP simplify coefficient mismatch for {case_name}")


def assert_operator_matches(
    actual: wolfgang.PauliSum,
    expected: wolfgang.PauliSum,
    *,
    case_name: str,
) -> None:
    actual_labels, actual_coeffs = labels_and_coefficients(actual)
    expected_labels, expected_coeffs = labels_and_coefficients(expected)
    if actual_labels != expected_labels:
        raise RuntimeError(f"HIP matmul label mismatch for {case_name}")
    if not np.allclose(actual_coeffs, expected_coeffs, rtol=1.0e-12, atol=1.0e-12):
        raise RuntimeError(f"HIP matmul coefficient mismatch for {case_name}")


def make_campaign6_expectation_operator(case: dict[str, Any]) -> wolfgang.PauliSum:
    labels = deterministic_labels(
        num_qubits=int(case["num_qubits"]),
        num_terms=int(case["num_terms"]),
        term_weight=int(case["term_weight"]),
        seed=int(case["random_seed"]),
    )
    rng = np.random.default_rng(int(case["random_seed"]) + 41)
    coeffs = [
        complex(float(rng.normal()), float(rng.normal())) for _ in range(int(case["num_terms"]))
    ]
    return wolfgang.PauliSum.from_labels(labels, coeffs)


def make_campaign6_statevector(case: dict[str, Any]) -> np.ndarray:
    rng = np.random.default_rng(int(case["random_seed"]) + 73)
    state_size = 2 ** int(case["num_qubits"])
    psi = rng.normal(size=state_size) + 1j * rng.normal(size=state_size)
    psi = np.asarray(psi / np.linalg.norm(psi), dtype=np.complex128)
    if case["statevector_dtype"] == "complex64":
        return psi.astype(np.complex64)
    return psi


def make_campaign6_matmul_operand(case: dict[str, Any], *, side: str) -> wolfgang.PauliSum:
    terms = int(case[f"{side}_terms"])
    duplicate_rate = float(case["duplicate_rate"])
    unique_terms = max(1, min(terms, int(round(terms * (1.0 - duplicate_rate)))))
    seed_offset = 0 if side == "lhs" else 100000
    pool = unique_deterministic_labels(
        num_qubits=int(case["num_qubits"]),
        unique_terms=unique_terms,
        term_weight=int(case["term_weight"]),
        seed=int(case["random_seed"]) + seed_offset,
    )
    labels = [pool[index % unique_terms] for index in range(terms)]
    rng = np.random.default_rng(int(case["random_seed"]) + seed_offset + 17)
    coeffs = [complex(float(rng.normal()), float(rng.normal())) for _ in range(terms)]
    return wolfgang.PauliSum.from_labels(labels, coeffs)


def operator_digest(op: wolfgang.PauliSum) -> dict[str, Any]:
    labels, coeffs = labels_and_coefficients(op)
    digest = hashlib.sha256()
    for label, coeff in zip(labels, coeffs, strict=True):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.float64(coeff.real).tobytes())
        digest.update(np.float64(coeff.imag).tobytes())
    return {
        "label_hash": digest.hexdigest(),
        "output_terms": int(op.num_terms),
        "coefficient_l1": float(np.abs(coeffs).sum(dtype=np.float64)),
    }


def expectation_digest(
    op: wolfgang.PauliSum,
    result: complex,
) -> dict[str, Any]:
    digest = operator_digest(op)
    digest.update(
        {
            "result_real": float(np.real(result)),
            "result_imag": float(np.imag(result)),
            "result_abs": float(abs(result)),
        }
    )
    return digest


def backend_device_name(hip_status: dict[str, Any]) -> str:
    devices = hip_status.get("devices", [])
    if not devices:
        return "not_available"
    return str(devices[0].get("name", "unknown"))


def selector_timings(
    lhs: wolfgang.PauliSum,
    rhs: wolfgang.PauliSum,
    *,
    warmup: int,
    repeat: int,
    build_info: dict[str, Any],
) -> dict[str, dict[str, float]]:
    available = set(build_info.get("available_cpu_backends", []))
    timings: dict[str, dict[str, float]] = {}
    for selector in OPTIMIZED_CPU_SELECTOR_ORDER:
        if selector not in available:
            continue
        with forced_cpu_backend(selector):
            _, summary = timed_call(lambda: lhs.commutes_with(rhs), warmup=warmup, repeat=repeat)
        timings[selector] = summary
    return timings


def benchmark_case(
    case: dict[str, Any],
    *,
    warmup: int,
    repeat: int,
    build_info: dict[str, Any],
    hip_status: dict[str, Any],
) -> dict[str, Any]:
    lhs = make_operator(case, side="lhs")
    rhs = make_operator(case, side="rhs")
    words = (int(case["num_qubits"]) + 63) // 64

    with forced_cpu_backend("scalar"):
        expected, cpu_scalar = timed_call(
            lambda: lhs.commutes_with(rhs), warmup=warmup, repeat=repeat
        )
    expected_array = np.asarray(expected, dtype=np.bool_).reshape(lhs.num_terms, rhs.num_terms)
    optimized_cpu = selector_timings(
        lhs,
        rhs,
        warmup=warmup,
        repeat=repeat,
        build_info=build_info,
    )

    result: dict[str, Any] = {
        "operation": "commutes_with",
        "case": case["name"],
        "dataset": {
            "num_qubits": int(case["num_qubits"]),
            "lhs_terms": lhs.num_terms,
            "rhs_terms": rhs.num_terms,
            "words": words,
            "term_weight": int(case["term_weight"]),
            "random_seed": int(case["random_seed"]),
            "operator_construction_method": "deterministic weighted dense labels",
        },
        "backend": "hip",
        "hip_architectures": build_info.get("hip_architectures", "not_available"),
        "device_name": backend_device_name(hip_status),
        "runtime_available": bool(hip_status.get("runtime_available", False)),
        "cpu_scalar_seconds": cpu_scalar["median"],
        "cpu_scalar_timing": cpu_scalar,
        "available_cpu_selector_seconds": {
            selector: summary["median"] for selector, summary in optimized_cpu.items()
        },
        "available_cpu_selector_timing": optimized_cpu,
        "transfer_inclusive_seconds": None,
        "transfer_inclusive_timing": None,
        "device_resident_seconds": None,
        "device_resident_timing": None,
        "correctness_passed": False,
    }

    if not hip_status.get("runtime_available", False):
        result["status"] = "hip_unavailable"
        result["unavailable_reason"] = hip_status.get("skip_reason", "HIP runtime unavailable")
        return result

    def transfer_inclusive_call() -> np.ndarray:
        lhs_device = lhs.to_device(device=0)
        rhs_device = rhs.to_device(device=0)
        return np.asarray(lhs_device.commutes_with(rhs_device), dtype=np.bool_)

    transfer_output, transfer_timing = timed_call(
        transfer_inclusive_call,
        warmup=warmup,
        repeat=repeat,
    )
    transfer_array = np.asarray(transfer_output, dtype=np.bool_).reshape(
        lhs.num_terms,
        rhs.num_terms,
    )
    if not np.array_equal(transfer_array, expected_array):
        raise RuntimeError(f"HIP transfer-inclusive commutation mismatch for {case['name']}")

    lhs_device = lhs.to_device(device=0)
    rhs_device = rhs.to_device(device=0)
    device_output, device_timing = timed_call(
        lambda: np.asarray(lhs_device.commutes_with(rhs_device), dtype=np.bool_),
        warmup=warmup,
        repeat=repeat,
    )
    device_array = np.asarray(device_output, dtype=np.bool_).reshape(lhs.num_terms, rhs.num_terms)
    if not np.array_equal(device_array, expected_array):
        raise RuntimeError(f"HIP device-resident commutation mismatch for {case['name']}")

    result["status"] = "ok"
    result["correctness_passed"] = True
    result["transfer_inclusive_seconds"] = transfer_timing["median"]
    result["transfer_inclusive_timing"] = transfer_timing
    result["device_resident_seconds"] = device_timing["median"]
    result["device_resident_timing"] = device_timing
    result["device_resident_timing_boundary"] = (
        "device operands reused; dense uint8 result is materialized to host"
    )
    return result


def benchmark_campaign2_case(
    case: dict[str, Any],
    *,
    profile: str,
    warmup: int,
    repeat: int,
    build_info: dict[str, Any],
    hip_status: dict[str, Any],
) -> dict[str, Any]:
    lhs = make_operator(case, side="lhs")
    rhs = make_operator(case, side="rhs")
    words = (int(case["num_qubits"]) + 63) // 64
    entries = lhs.num_terms * rhs.num_terms

    with forced_cpu_backend("scalar"):
        expected, cpu_scalar = timed_call(
            lambda: lhs.commutes_with(rhs), warmup=warmup, repeat=repeat
        )
    expected_array = np.asarray(expected, dtype=np.bool_).reshape(lhs.num_terms, rhs.num_terms)
    optimized_cpu = selector_timings(
        lhs,
        rhs,
        warmup=warmup,
        repeat=repeat,
        build_info=build_info,
    )

    row: dict[str, Any] = {
        "operation": "commutes_with_campaign2",
        "case": case["name"],
        "dataset": {
            "num_qubits": int(case["num_qubits"]),
            "lhs_terms": lhs.num_terms,
            "rhs_terms": rhs.num_terms,
            "entries": entries,
            "words": words,
            "term_weight": int(case["term_weight"]),
            "random_seed": int(case["random_seed"]),
            "operator_construction_method": "deterministic weighted dense labels",
        },
        "backend": "hip",
        "device_name": backend_device_name(hip_status),
        "gfx_target": first_device_field(hip_status, "gfx_target"),
        "hip_architectures": build_info.get("hip_architectures", "not_available"),
        "num_qubits": int(case["num_qubits"]),
        "lhs_terms": lhs.num_terms,
        "rhs_terms": rhs.num_terms,
        "entries": entries,
        "runtime_available": bool(hip_status.get("runtime_available", False)),
        "cpu_scalar_seconds": cpu_scalar["median"],
        "cpu_scalar_timing": cpu_scalar,
        "available_cpu_selector_seconds": {
            selector: summary["median"] for selector, summary in optimized_cpu.items()
        },
        "available_cpu_selector_timing": optimized_cpu,
        "correctness_digest": commutation_digest(expected_array),
        "correctness_passed": False,
        "result_materialization_target": (
            "compact_uint64_counts_and_conflicts"
            if profile == "commutation-compact-consumers"
            else "device_uint8_matrix"
        ),
        "result_materialization_targets": (
            ["compact_uint64_counts", "compact_uint64_conflicts"]
            if profile == "commutation-compact-consumers"
            else ["device_uint8_matrix"]
        ),
        "timing_boundary": (
            "compact_consumer"
            if profile == "commutation-compact-consumers"
            else "device_output_allocating"
        ),
        "campaign": "rocm_mi300x_campaign2",
    }

    for prefix in (
        "hip_device_output_allocate",
        "hip_device_output_reuse",
        "hip_device_output_to_host",
        "hip_count_commuting_axis_none",
        "hip_count_commuting_axis_0",
        "hip_count_commuting_axis_1",
        "hip_conflict_degrees_axis_none",
        "hip_conflict_degrees_axis_0",
        "hip_conflict_degrees_axis_1",
    ):
        row[f"{prefix}_seconds"] = None
        row[f"{prefix}_p10_seconds"] = None
        row[f"{prefix}_p90_seconds"] = None
        row[f"{prefix}_min_seconds"] = None
        row[f"{prefix}_max_seconds"] = None

    if not hip_status.get("runtime_available", False):
        row["status"] = "hip_unavailable"
        row["unavailable_reason"] = hip_status.get("skip_reason", "HIP runtime unavailable")
        return row

    lhs_device = lhs.to_device(device=0)
    rhs_device = rhs.to_device(device=0)

    host_output, host_timing = timed_call(
        lambda: np.asarray(lhs_device.commutes_with(rhs_device), dtype=np.bool_),
        warmup=warmup,
        repeat=repeat,
    )
    host_array = np.asarray(host_output, dtype=np.bool_).reshape(lhs.num_terms, rhs.num_terms)
    if not np.array_equal(host_array, expected_array):
        raise RuntimeError(f"HIP host-output commutation mismatch for {case['name']}")
    row["hip_device_operand_host_output_seconds"] = host_timing["median"]
    row["hip_device_operand_host_output_timing"] = host_timing

    allocated_matrix, allocate_timing = timed_call(
        lambda: lhs_device.commutes_with_device(rhs_device),
        warmup=warmup,
        repeat=repeat,
    )
    allocated_host = allocated_matrix.to_host()
    if not np.array_equal(allocated_host, expected_array):
        raise RuntimeError(f"HIP device-output allocating mismatch for {case['name']}")
    add_timing_fields(row, "hip_device_output_allocate", allocate_timing)

    reused_output = wolfgang.DeviceCommutationMatrix.empty(
        (lhs.num_terms, rhs.num_terms),
        device=lhs_device.device,
    )
    reused_matrix, reuse_timing = timed_call(
        lambda: lhs_device.commutes_with_device(rhs_device, output=reused_output),
        warmup=warmup,
        repeat=repeat,
    )
    if reused_matrix is not reused_output:
        raise RuntimeError("HIP commutes_with_device did not preserve caller output identity")
    reused_host = reused_output.to_host()
    if not np.array_equal(reused_host, expected_array):
        raise RuntimeError(f"HIP device-output reuse mismatch for {case['name']}")
    add_timing_fields(row, "hip_device_output_reuse", reuse_timing)

    dense_host, to_host_timing = timed_call(
        lambda: allocated_matrix.to_host(),
        warmup=warmup,
        repeat=repeat,
    )
    if not np.array_equal(dense_host, expected_array):
        raise RuntimeError(f"HIP device-output to_host mismatch for {case['name']}")
    add_timing_fields(row, "hip_device_output_to_host", to_host_timing)

    count_total, count_total_timing = timed_call(
        lambda: allocated_matrix.count_commuting(),
        warmup=warmup,
        repeat=repeat,
    )
    count_axis0, count_axis0_timing = timed_call(
        lambda: allocated_matrix.count_commuting(axis=0),
        warmup=warmup,
        repeat=repeat,
    )
    count_axis1, count_axis1_timing = timed_call(
        lambda: allocated_matrix.count_commuting(axis=1),
        warmup=warmup,
        repeat=repeat,
    )
    expected_u64 = expected_array.astype(np.uint64)
    if int(count_total) != int(expected_u64.sum(dtype=np.uint64)):
        raise RuntimeError(f"HIP count_commuting total mismatch for {case['name']}")
    if not np.array_equal(count_axis0, expected_u64.sum(axis=0, dtype=np.uint64)):
        raise RuntimeError(f"HIP count_commuting axis=0 mismatch for {case['name']}")
    if not np.array_equal(count_axis1, expected_u64.sum(axis=1, dtype=np.uint64)):
        raise RuntimeError(f"HIP count_commuting axis=1 mismatch for {case['name']}")
    add_timing_fields(row, "hip_count_commuting_axis_none", count_total_timing)
    add_timing_fields(row, "hip_count_commuting_axis_0", count_axis0_timing)
    add_timing_fields(row, "hip_count_commuting_axis_1", count_axis1_timing)

    conflict_total, conflict_total_timing = timed_call(
        lambda: allocated_matrix.conflict_degrees(),
        warmup=warmup,
        repeat=repeat,
    )
    conflict_axis0, conflict_axis0_timing = timed_call(
        lambda: allocated_matrix.conflict_degrees(axis=0),
        warmup=warmup,
        repeat=repeat,
    )
    conflict_axis1, conflict_axis1_timing = timed_call(
        lambda: allocated_matrix.conflict_degrees(axis=1),
        warmup=warmup,
        repeat=repeat,
    )
    expected_conflicts = np.logical_not(expected_array).astype(np.uint64)
    if int(conflict_total) != int(expected_conflicts.sum(dtype=np.uint64)):
        raise RuntimeError(f"HIP conflict_degrees total mismatch for {case['name']}")
    if not np.array_equal(conflict_axis0, expected_conflicts.sum(axis=0, dtype=np.uint64)):
        raise RuntimeError(f"HIP conflict_degrees axis=0 mismatch for {case['name']}")
    if not np.array_equal(conflict_axis1, expected_conflicts.sum(axis=1, dtype=np.uint64)):
        raise RuntimeError(f"HIP conflict_degrees axis=1 mismatch for {case['name']}")
    add_timing_fields(row, "hip_conflict_degrees_axis_none", conflict_total_timing)
    add_timing_fields(row, "hip_conflict_degrees_axis_0", conflict_axis0_timing)
    add_timing_fields(row, "hip_conflict_degrees_axis_1", conflict_axis1_timing)

    row["status"] = "ok"
    row["correctness_passed"] = True
    row["materialization_boundaries"] = {
        "host_output": "device operands reused; dense bool result materialized to host",
        "device_output_allocate": "allocates HIP DeviceCommutationMatrix and fills it on device",
        "device_output_reuse": "fills caller-owned HIP DeviceCommutationMatrix on device",
        "device_output_to_host": "copies dense uint8 matrix to host bool matrix",
        "compact_counts": "copies scalar or uint64 count summaries to host",
        "compact_conflicts": "copies scalar or uint64 conflict summaries to host",
    }
    return row


def requested_hip_simplify_strategy() -> str:
    value = os.environ.get("WOLFGANG_HIP_BENCH_DUPLICATE_REDUCTION", "rocthrust_default")
    if value == "":
        value = "rocthrust_default"
    return value


def hip_simplify_key_shape(*, num_qubits: int, words: int) -> str:
    if words == 0:
        return "identity"
    if words == 1 and num_qubits <= 32:
        return "packed32"
    if words == 1:
        return "key1"
    if words == 2:
        return "key2"
    return "generic_multiword"


def generic_multiword_parallelism_for_case(
    case: dict[str, Any],
    *,
    key_shape: str,
) -> str:
    if key_shape != "generic_multiword":
        return "not_applicable"
    return str(case.get("generic_multiword_parallelism", "reduce_by_key"))


def campaign4_strategy_status(
    *,
    strategy: str,
    key_shape: str,
    generic_multiword_parallelism: str,
) -> tuple[str, str]:
    if strategy in UNAVAILABLE_HIP_SIMPLIFY_STRATEGIES:
        return "unavailable", UNAVAILABLE_HIP_SIMPLIFY_STRATEGIES[strategy]
    if strategy == "rocthrust_generic_parallel_reduce_by_key":
        if key_shape == "generic_multiword" and generic_multiword_parallelism == "reduce_by_key":
            return (
                "retained",
                "parallel sorted-index reduce_by_key is the retained generic multi-word path",
            )
        return (
            "benchmark_only",
            "generic reduce_by_key selector maps to existing rocThrust paths for non-generic rows",
        )
    if generic_multiword_parallelism == "serial_kernel":
        return (
            "benchmark_only",
            "serial generic kernel is retained only as a private Campaign 4 A/B fallback",
        )
    return "retained", "rocThrust default remains the retained one-word and two-word path"


def workspace_fields_for_case(case: dict[str, Any]) -> dict[str, int | str]:
    mode = str(case.get("hip_workspace_mode", "absent"))
    return {
        "hip_workspace_mode": mode,
        "hip_workspace_reserved_bytes": 0,
        "hip_workspace_high_watermark_bytes": 0,
        "hip_workspace_allocation_count": 0,
        "hip_workspace_growth_count": 0,
    }


def simplify_runtime_environment(
    *,
    strategy: str,
    generic_multiword_parallelism: str,
) -> dict[str, str | None]:
    duplicate_strategy = strategy
    if strategy in UNAVAILABLE_HIP_SIMPLIFY_STRATEGIES:
        duplicate_strategy = "rocthrust_default"
    if strategy == "rocthrust_generic_parallel_reduce_by_key":
        duplicate_strategy = "rocthrust_default"
    generic_strategy = None
    if generic_multiword_parallelism == "serial_kernel":
        generic_strategy = "serial_kernel"
    elif generic_multiword_parallelism == "reduce_by_key":
        generic_strategy = "reduce_by_key"
    return {
        "WOLFGANG_HIP_BENCH_DUPLICATE_REDUCTION": duplicate_strategy,
        "WOLFGANG_HIP_BENCH_GENERIC_MULTIWORD_REDUCTION": generic_strategy,
    }


def benchmark_simplify_case(
    case: dict[str, Any],
    *,
    profile: str,
    warmup: int,
    repeat: int,
    build_info: dict[str, Any],
    hip_status: dict[str, Any],
) -> dict[str, Any]:
    op = make_simplify_operator(case)
    words = (int(case["num_qubits"]) + 63) // 64
    duplicate_rate = float(case["duplicate_rate"])
    actual_duplicate_rate = 1.0 - (
        len(set(labels_and_coefficients(op)[0])) / float(max(1, op.num_terms))
    )
    is_campaign4 = profile in CAMPAIGN4_SIMPLIFY_PROFILES
    strategy = str(case.get("hip_simplify_strategy", requested_hip_simplify_strategy()))
    strategy_valid = strategy in HIP_SIMPLIFY_STRATEGIES
    key_shape = hip_simplify_key_shape(num_qubits=int(case["num_qubits"]), words=words)
    generic_multiword_parallelism = generic_multiword_parallelism_for_case(
        case,
        key_shape=key_shape,
    )
    if is_campaign4:
        strategy_status, strategy_reason = campaign4_strategy_status(
            strategy=strategy,
            key_shape=key_shape,
            generic_multiword_parallelism=generic_multiword_parallelism,
        )
    else:
        strategy_status = "retained" if strategy == "rocthrust_default" else "unavailable"
        strategy_reason = (
            "Campaign 3 retained rocThrust default only"
            if strategy != "rocthrust_default"
            else "rocThrust default retained"
        )

    with forced_cpu_backend("scalar"):
        expected, cpu_scalar = timed_call(lambda: op.simplify(), warmup=warmup, repeat=repeat)
    optimized_cpu = {
        "not_applicable": {
            "status": "not_applicable",
            "reason": "simplify has no optimized CPU selector in this build",
        }
    }

    row: dict[str, Any] = {
        "operation": "simplify",
        "case": case["name"],
        "dataset": {
            "num_qubits": int(case["num_qubits"]),
            "num_terms": int(op.num_terms),
            "words": words,
            "term_weight": int(case["term_weight"]),
            "duplicate_rate": duplicate_rate,
            "actual_duplicate_rate": actual_duplicate_rate,
            "coefficient_dtype": "complex128",
            "coefficient_mode": case.get("coefficient_mode", "deterministic_complex_normal"),
            "random_seed": int(case["random_seed"]),
            "operator_construction_method": "deterministic labels with duplicate pool",
        },
        "backend": "hip",
        "campaign": "rocm_mi300x_campaign4" if is_campaign4 else "rocm_mi300x_campaign3",
        "profile": profile,
        "device_name": backend_device_name(hip_status),
        "gfx_target": first_device_field(hip_status, "gfx_target"),
        "hip_architectures": build_info.get("hip_architectures", "not_available"),
        "runtime_available": bool(hip_status.get("runtime_available", False)),
        "cpu_scalar_seconds": cpu_scalar["median"],
        "cpu_scalar_timing": cpu_scalar,
        "available_cpu_selector_seconds": {"not_applicable": None},
        "available_cpu_selector_timing": optimized_cpu,
        "hip_simplify_strategy": strategy if strategy_valid else "unavailable",
        "hip_simplify_strategy_status": strategy_status,
        "hip_simplify_strategy_reason": strategy_reason,
        "hip_simplify_strategy_unavailable_reason": None,
        "hip_simplify_key_shape": key_shape,
        "generic_multiword_parallelism": generic_multiword_parallelism,
        "hip_simplify_output_terms": int(expected.num_terms),
        "hip_simplify_output_words": words,
        "result_materialization_target": "device_pauli_sum",
        "timing_boundary": "device_resident",
        "timing_boundaries": [
            "transfer_inclusive",
            "device_resident",
            "device_output_to_host",
        ],
        "correctness_digest": simplify_digest(input_terms=op.num_terms, simplified=expected),
        "correctness_passed": False,
        "campaign3_headroom_statuses": dict(CAMPAIGN3_HEADROOM_STATUSES),
    }
    if is_campaign4:
        row.update(workspace_fields_for_case(case))
        row["campaign4_terminal_statuses"] = dict(CAMPAIGN4_TERMINAL_STATUSES)
    for prefix in (
        "hip_simplify_transfer",
        "hip_simplify_device_resident",
        "hip_simplify_to_host",
    ):
        add_null_timing_fields(row, prefix)

    if not strategy_valid:
        row["status"] = "strategy_unavailable"
        row["hip_simplify_strategy_status"] = "unavailable"
        row["hip_simplify_strategy_unavailable_reason"] = (
            "unknown WOLFGANG_HIP_BENCH_DUPLICATE_REDUCTION setting"
        )
        row["hip_simplify_strategy_reason"] = row["hip_simplify_strategy_unavailable_reason"]
        return row

    if is_campaign4 and strategy in UNAVAILABLE_HIP_SIMPLIFY_STRATEGIES:
        row["status"] = "strategy_unavailable"
        row["hip_simplify_strategy_status"] = "unavailable"
        row["hip_simplify_strategy_unavailable_reason"] = strategy_reason
        row["hip_simplify_strategy_reason"] = strategy_reason
        return row

    if not is_campaign4 and strategy != "rocthrust_default":
        row["status"] = "strategy_rejected"
        row["hip_simplify_strategy_status"] = "rejected_with_evidence"
        row["hip_simplify_strategy_unavailable_reason"] = (
            "Campaign 3 retained rocThrust default only; hipCUB/custom probes need "
            "separate implementation evidence before timing"
        )
        row["hip_simplify_strategy_reason"] = row["hip_simplify_strategy_unavailable_reason"]
        return row

    if not hip_status.get("runtime_available", False):
        row["status"] = "hip_unavailable"
        row["hip_simplify_strategy_status"] = "unavailable"
        row["hip_simplify_strategy_unavailable_reason"] = hip_status.get(
            "skip_reason",
            "HIP runtime unavailable",
        )
        row["unavailable_reason"] = hip_status.get("skip_reason", "HIP runtime unavailable")
        return row

    env_updates = simplify_runtime_environment(
        strategy=strategy,
        generic_multiword_parallelism=generic_multiword_parallelism,
    )

    def transfer_inclusive_call() -> wolfgang.PauliSum:
        with forced_environment(env_updates):
            return op.to_device(device=0).simplify().to_host()

    transfer_host, transfer_timing = timed_call(
        transfer_inclusive_call,
        warmup=warmup,
        repeat=repeat,
    )
    assert_simplify_matches(transfer_host, expected, case_name=str(case["name"]))

    device_op = op.to_device(device=0)
    with forced_environment(env_updates):
        simplified_device, device_timing = timed_call(
            lambda: device_op.simplify(),
            warmup=warmup,
            repeat=repeat,
        )
    device_host = simplified_device.to_host()
    assert_simplify_matches(device_host, expected, case_name=str(case["name"]))

    _, to_host_timing = timed_call(
        lambda: simplified_device.to_host(),
        warmup=warmup,
        repeat=repeat,
    )

    row["status"] = "ok"
    row["correctness_passed"] = True
    row["hip_simplify_output_terms"] = int(simplified_device.num_terms)
    row["hip_simplify_output_words"] = words
    add_timing_fields(row, "hip_simplify_transfer", transfer_timing)
    add_timing_fields(row, "hip_simplify_device_resident", device_timing)
    add_timing_fields(row, "hip_simplify_to_host", to_host_timing)
    return row


def campaign5_add_null_timing_fields(row: dict[str, Any]) -> None:
    for prefix in (
        "hip_dlpack_export",
        "consumer_from_dlpack",
        "consumer_sum",
        "hip_device_output_to_host",
        "hip_count_commuting_axis_none",
    ):
        add_null_timing_fields(row, prefix)


def campaign6_base_row(
    *,
    case: dict[str, Any],
    profile: str,
    operation: str,
    mode: str,
    build_info: dict[str, Any],
    hip_status: dict[str, Any],
) -> dict[str, Any]:
    return {
        "campaign": "rocm_mi300x_campaign6",
        "profile": profile,
        "operation": operation,
        "mode": mode,
        "backend": "hip",
        "case": case["name"],
        "device_name": backend_device_name(hip_status),
        "gfx_target": first_device_field(hip_status, "gfx_target"),
        "hip_architectures": build_info.get("hip_architectures", "not_available"),
        "runtime_available": bool(hip_status.get("runtime_available", False)),
        "available_cpu_selector_seconds": {"not_applicable": None},
        "available_cpu_selector_timing": {
            "not_applicable": {
                "status": "not_applicable",
                "reason": "Campaign 6 operations do not have a dedicated optimized CPU selector",
            }
        },
        "campaign6_terminal_statuses": dict(CAMPAIGN6_TERMINAL_STATUSES),
    }


def campaign6_unavailable_row(row: dict[str, Any], hip_status: dict[str, Any]) -> dict[str, Any]:
    row["status"] = "unavailable"
    row["final_status"] = "unavailable"
    row["correctness_passed"] = False
    row["unavailable_reason"] = hip_status.get("skip_reason", "HIP runtime unavailable")
    return row


def benchmark_campaign6_expectation_case(
    case: dict[str, Any],
    *,
    profile: str,
    warmup: int,
    repeat: int,
    build_info: dict[str, Any],
    hip_status: dict[str, Any],
) -> dict[str, Any]:
    op = make_campaign6_expectation_operator(case)
    psi = make_campaign6_statevector(case)
    words = (int(case["num_qubits"]) + 63) // 64
    mode = "profiler_expectation" if profile == "campaign6-profiler" else f"host_{psi.dtype}"

    with forced_cpu_backend("scalar"):
        expected, cpu_scalar = timed_call(
            lambda: op.expectation_statevector(psi),
            warmup=warmup,
            repeat=repeat,
        )

    row = campaign6_base_row(
        case=case,
        profile=profile,
        operation="expectation_statevector",
        mode=mode,
        build_info=build_info,
        hip_status=hip_status,
    )
    row.update(
        {
            "dataset": {
                "num_qubits": int(case["num_qubits"]),
                "num_terms": int(op.num_terms),
                "words": words,
                "term_weight": int(case["term_weight"]),
                "statevector_dtype": str(psi.dtype),
                "state_size": int(psi.size),
                "random_seed": int(case["random_seed"]),
                "operator_construction_method": "deterministic weighted dense labels",
            },
            "timing_boundary": "transfer_inclusive",
            "cpu_scalar_seconds": cpu_scalar["median"],
            "cpu_scalar_timing": cpu_scalar,
            "hip_expectation_input_dtype": str(psi.dtype),
            "hip_expectation_state_size": int(psi.size),
            "hip_expectation_num_terms": int(op.num_terms),
            "hip_expectation_words": words,
            "correctness_digest": expectation_digest(op, expected),
        }
    )
    for prefix in (
        "hip_expectation_transfer",
        "hip_expectation_device_resident",
        "hip_expectation_result_copy",
    ):
        add_null_timing_fields(row, prefix)

    if not hip_status.get("runtime_available", False):
        return campaign6_unavailable_row(row, hip_status)

    tolerance = 1.0e-5 if psi.dtype == np.complex64 else 1.0e-12
    transfer_result, transfer_timing = timed_call(
        lambda: op.to_device(device=0).expectation_statevector(psi),
        warmup=warmup,
        repeat=repeat,
    )
    if not np.allclose(transfer_result, expected, rtol=tolerance, atol=tolerance):
        raise RuntimeError(f"HIP transfer-inclusive expectation mismatch for {case['name']}")

    device_op = op.to_device(device=0)
    device_result, device_timing = timed_call(
        lambda: device_op.expectation_statevector(psi),
        warmup=warmup,
        repeat=repeat,
    )
    if not np.allclose(device_result, expected, rtol=tolerance, atol=tolerance):
        raise RuntimeError(f"HIP device-resident expectation mismatch for {case['name']}")

    row["status"] = "ok"
    row["final_status"] = "retained"
    row["correctness_passed"] = True
    row["timing_boundary"] = "device_resident_kernel"
    row["correctness_digest"] = expectation_digest(op, device_result)
    add_timing_fields(row, "hip_expectation_transfer", transfer_timing)
    add_timing_fields(row, "hip_expectation_device_resident", device_timing)
    return row


def benchmark_campaign6_matmul_case(
    case: dict[str, Any],
    *,
    profile: str,
    warmup: int,
    repeat: int,
    build_info: dict[str, Any],
    hip_status: dict[str, Any],
) -> dict[str, Any]:
    lhs = make_campaign6_matmul_operand(case, side="lhs")
    rhs = make_campaign6_matmul_operand(case, side="rhs")
    simplify_output = bool(case["simplify_output"])
    words = (int(case["num_qubits"]) + 63) // 64
    mode = (
        "profiler_matmul"
        if profile == "campaign6-profiler"
        else ("matmul_simplify_true" if simplify_output else "matmul_simplify_false")
    )

    with forced_cpu_backend("scalar"):
        expected, cpu_scalar = timed_call(
            lambda: lhs.matmul(rhs, simplify=simplify_output),
            warmup=warmup,
            repeat=repeat,
        )

    row = campaign6_base_row(
        case=case,
        profile=profile,
        operation="matmul",
        mode=mode,
        build_info=build_info,
        hip_status=hip_status,
    )
    row.update(
        {
            "dataset": {
                "num_qubits": int(case["num_qubits"]),
                "lhs_terms": int(lhs.num_terms),
                "rhs_terms": int(rhs.num_terms),
                "intermediate_terms": int(lhs.num_terms * rhs.num_terms),
                "words": words,
                "term_weight": int(case["term_weight"]),
                "duplicate_rate": float(case["duplicate_rate"]),
                "simplify_output": simplify_output,
                "random_seed": int(case["random_seed"]),
                "operator_construction_method": "deterministic labels with duplicate pool",
            },
            "timing_boundary": "transfer_inclusive",
            "cpu_scalar_seconds": cpu_scalar["median"],
            "cpu_scalar_timing": cpu_scalar,
            "hip_matmul_lhs_terms": int(lhs.num_terms),
            "hip_matmul_rhs_terms": int(rhs.num_terms),
            "hip_matmul_output_terms": int(expected.num_terms),
            "hip_matmul_words": words,
            "hip_matmul_simplify_output": simplify_output,
            "correctness_digest": operator_digest(expected),
        }
    )
    for prefix in (
        "hip_matmul_transfer",
        "hip_matmul_device_resident",
        "hip_matmul_to_host",
    ):
        add_null_timing_fields(row, prefix)

    if not hip_status.get("runtime_available", False):
        return campaign6_unavailable_row(row, hip_status)

    transfer_host, transfer_timing = timed_call(
        lambda: (
            lhs.to_device(device=0)
            .matmul(rhs.to_device(device=0), simplify=simplify_output)
            .to_host()
        ),
        warmup=warmup,
        repeat=repeat,
    )
    assert_operator_matches(transfer_host, expected, case_name=str(case["name"]))

    lhs_device = lhs.to_device(device=0)
    rhs_device = rhs.to_device(device=0)
    device_product, device_timing = timed_call(
        lambda: lhs_device.matmul(rhs_device, simplify=simplify_output),
        warmup=warmup,
        repeat=repeat,
    )
    device_host = device_product.to_host()
    assert_operator_matches(device_host, expected, case_name=str(case["name"]))
    _, to_host_timing = timed_call(
        lambda: device_product.to_host(),
        warmup=warmup,
        repeat=repeat,
    )

    row["status"] = "ok"
    row["final_status"] = "retained"
    row["correctness_passed"] = True
    row["timing_boundary"] = "device_resident_kernel"
    row["hip_matmul_output_terms"] = int(device_product.num_terms)
    row["correctness_digest"] = operator_digest(device_host)
    add_timing_fields(row, "hip_matmul_transfer", transfer_timing)
    add_timing_fields(row, "hip_matmul_device_resident", device_timing)
    add_timing_fields(row, "hip_matmul_to_host", to_host_timing)
    return row


def campaign7_base_fields(
    *,
    profile: str,
    operation: str,
    mode: str,
    backend: str,
    host_role: str,
    build_info: dict[str, Any],
    hip_status: dict[str, Any],
    final_status: str,
) -> dict[str, Any]:
    compiler = build_info.get("compiler_build_config", {})
    if not isinstance(compiler, dict):
        compiler = {}
    gpu_name = backend_device_name(hip_status)
    return {
        "campaign": "rocm_mi300x_campaign7",
        "profile": profile,
        "operation": operation,
        "mode": mode,
        "backend": backend,
        "host_role": host_role,
        "status": "ok" if final_status in {"passed", "retained"} else final_status,
        "final_status": final_status,
        "rocm_runtime_version": hip_status.get(
            "runtime_version",
            build_info.get("hip_runtime_version", "not_available"),
        ),
        "rocm_toolkit_version": build_info.get("rocm_toolkit_version", "not_available"),
        "hip_compiler_version": compiler.get("CMAKE_HIP_COMPILER_VERSION", "not_available"),
        "gpu_name": gpu_name,
        "device_name": gpu_name,
        "gfx_target": first_device_field(hip_status, "gfx_target"),
        "hip_architectures": build_info.get("hip_architectures", "not_available"),
        "runtime_available": bool(hip_status.get("runtime_available", False)),
        "build_command": CAMPAIGN7_HIP_BUILD_COMMAND,
        "validation_command": CAMPAIGN7_HIP_VALIDATION_COMMAND,
        "profiler_command": CAMPAIGN7_PROFILER_COMMAND,
        "campaign7_terminal_statuses": dict(CAMPAIGN7_TERMINAL_STATUSES),
    }


def campaign7_enrich_row(
    row: dict[str, Any],
    *,
    profile: str,
    operation: str,
    mode: str,
    timing_boundary: str,
    build_info: dict[str, Any],
    hip_status: dict[str, Any],
    final_status: str,
    backend: str = "hip",
    host_role: str = "primary_mi300x",
) -> dict[str, Any]:
    item = dict(row)
    item.update(
        campaign7_base_fields(
            profile=profile,
            operation=operation,
            mode=mode,
            backend=backend,
            host_role=host_role,
            build_info=build_info,
            hip_status=hip_status,
            final_status=final_status,
        )
    )
    item["timing_boundary"] = timing_boundary
    if not hip_status.get("runtime_available", False) and backend == "hip":
        item["status"] = "unavailable"
        item["final_status"] = "unavailable"
        item["unavailable_reason"] = hip_status.get("skip_reason", "HIP runtime unavailable")
    return item


def benchmark_campaign7_transfer_case(
    case: dict[str, Any],
    *,
    profile: str,
    warmup: int,
    repeat: int,
    build_info: dict[str, Any],
    hip_status: dict[str, Any],
) -> dict[str, Any]:
    op = make_operator(case, side="lhs")
    words = (int(case["num_qubits"]) + 63) // 64
    _, cpu_scalar = timed_call(lambda: labels_and_coefficients(op), warmup=warmup, repeat=repeat)
    row = campaign7_base_fields(
        profile=profile,
        operation="retained_operation_smoke",
        mode="retained_transfer",
        backend="hip",
        host_role="primary_mi300x",
        build_info=build_info,
        hip_status=hip_status,
        final_status="retained",
    )
    row.update(
        {
            "case": case["name"],
            "dataset": {
                "num_qubits": int(case["num_qubits"]),
                "num_terms": int(op.num_terms),
                "words": words,
                "term_weight": int(case["term_weight"]),
                "random_seed": int(case["random_seed"]),
                "operator_construction_method": "deterministic weighted dense labels",
            },
            "timing_boundary": "transfer_inclusive",
            "cpu_scalar_seconds": cpu_scalar["median"],
            "cpu_scalar_timing": cpu_scalar,
            "correctness_passed": False,
            "correctness_digest": operator_digest(op),
        }
    )
    for prefix in ("hip_transfer", "hip_to_host"):
        add_null_timing_fields(row, prefix)
    if not hip_status.get("runtime_available", False):
        row["status"] = "unavailable"
        row["final_status"] = "unavailable"
        row["unavailable_reason"] = hip_status.get("skip_reason", "HIP runtime unavailable")
        return row

    device_op, transfer_timing = timed_call(
        lambda: op.to_device(device=0),
        warmup=warmup,
        repeat=repeat,
    )
    host_op, to_host_timing = timed_call(
        lambda: device_op.to_host(),
        warmup=warmup,
        repeat=repeat,
    )
    assert_operator_matches(host_op, op, case_name=str(case["name"]))
    row["status"] = "ok"
    row["final_status"] = "retained"
    row["correctness_passed"] = True
    row["correctness_digest"] = operator_digest(host_op)
    add_timing_fields(row, "hip_transfer", transfer_timing)
    add_timing_fields(row, "hip_to_host", to_host_timing)
    return row


def campaign7_commutation_rows(
    case: dict[str, Any],
    *,
    profile: str,
    warmup: int,
    repeat: int,
    build_info: dict[str, Any],
    hip_status: dict[str, Any],
) -> list[dict[str, Any]]:
    row = benchmark_campaign2_case(
        case,
        profile="commutation-compact-consumers",
        warmup=warmup,
        repeat=repeat,
        build_info=build_info,
        hip_status=hip_status,
    )
    retained_status = "retained"
    timing_boundary = "device_resident"
    if not hip_status.get("runtime_available", False):
        retained_status = "unavailable"

    commutation = campaign7_enrich_row(
        row,
        profile=profile,
        operation="retained_operation_smoke",
        mode="retained_commutation",
        timing_boundary=timing_boundary,
        build_info=build_info,
        hip_status=hip_status,
        final_status=retained_status,
    )
    commutation["case"] = f"{case['name']}_commutation"
    commutation["correctness_passed"] = bool(row.get("correctness_passed", False))
    commutation["correctness_digest"] = row.get("correctness_digest")

    consumers = campaign7_enrich_row(
        row,
        profile=profile,
        operation="retained_operation_smoke",
        mode="retained_device_consumers",
        timing_boundary="compact_consumer",
        build_info=build_info,
        hip_status=hip_status,
        final_status=retained_status,
    )
    consumers["case"] = f"{case['name']}_compact_consumers"
    consumers["correctness_passed"] = bool(row.get("correctness_passed", False))
    consumers["correctness_digest"] = row.get("correctness_digest")
    return [commutation, consumers]


def campaign7_decision_rows(
    *,
    case: dict[str, Any],
    profile: str,
    build_info: dict[str, Any],
    hip_status: dict[str, Any],
) -> list[dict[str, Any]]:
    decisions = [
        (
            "release_source_build",
            "mi300x_repeatability",
            "hip",
            "primary_mi300x",
            "source_build",
            "passed" if hip_status.get("runtime_available", False) else "unavailable",
            "MI300X HIP source build and runtime import lane.",
            CAMPAIGN7_HIP_VALIDATION_COMMAND,
        ),
        (
            "runtime_validation",
            "cpu_only_control",
            "cpu",
            "local_cpu_control",
            "validation_only",
            "passed",
            "CPU-only validation lane keeps WOLFGANG_ENABLE_HIP=OFF independent of ROCm.",
            CAMPAIGN7_CPU_VALIDATION_COMMAND,
        ),
        (
            "runtime_validation",
            "cuda_hip_rejection",
            "none",
            "decision_only",
            "validation_only",
            "passed",
            "WOLFGANG_ENABLE_CUDA=ON with WOLFGANG_ENABLE_HIP=ON remains a configure-time error.",
            "cmake configure with WOLFGANG_ENABLE_CUDA=ON and WOLFGANG_ENABLE_HIP=ON",
        ),
        (
            "ci_runbook",
            "release_runbook",
            "none",
            "decision_only",
            "decision_only",
            "retained",
            "Campaign 7 release-support lane records commands, environment, artifacts, and terminal statuses.",
            "python scripts/run_rocm_release_support_lane.py --print-commands",
        ),
        (
            "packaging_decision",
            "rocm_wheel_policy",
            "none",
            "decision_only",
            "decision_only",
            "unavailable",
            "ROCm wheels remain unavailable until a separate packaging channel is designed and tested.",
            "git diff --check docs/quality/release_and_packaging.md",
        ),
        (
            "portability_lane",
            "alternate_amd_gpu_probe",
            "hip",
            "alternate_amd_gpu",
            "decision_only",
            "blocked_external",
            "No non-MI300X AMD GPU host is available in Campaign 7.",
            "ssh to an alternate AMD GPU host, then run the Campaign 7 source-build lane",
        ),
        (
            "backend_neutral_decision",
            "external_statevector_decision",
            "none",
            "decision_only",
            "decision_only",
            "out_of_scope_with_next_trigger",
            "External HIP statevectors require ownership, stream, and consumer contracts.",
            "create a backend-neutral statevector interop architecture plan",
        ),
        (
            "backend_neutral_decision",
            "multi_gpu_decision",
            "none",
            "decision_only",
            "decision_only",
            "out_of_scope_with_next_trigger",
            "Multi-GPU ROCm requires a backend-neutral device ownership model.",
            "create a Wave 6 multi-accelerator architecture plan",
        ),
        (
            "backend_neutral_decision",
            "simultaneous_cuda_hip_decision",
            "none",
            "decision_only",
            "decision_only",
            "unavailable",
            "CUDA and HIP source builds remain mutually exclusive.",
            "cmake configure with WOLFGANG_ENABLE_CUDA=ON and WOLFGANG_ENABLE_HIP=ON",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for (
        operation,
        mode,
        backend,
        host_role,
        boundary,
        final_status,
        reason,
        validation,
    ) in decisions:
        row = campaign7_base_fields(
            profile=profile,
            operation=operation,
            mode=mode,
            backend=backend,
            host_role=host_role,
            build_info=build_info,
            hip_status=hip_status,
            final_status=final_status,
        )
        row.update(
            {
                "case": f"{case['name']}_{mode}",
                "timing_boundary": boundary,
                "dataset": {},
                "decision_reason": reason,
                "validation_command": validation,
                "correctness_passed": None,
                "correctness_digest": None,
            }
        )
        rows.append(row)
    return rows


def benchmark_campaign7_case(
    case: dict[str, Any],
    *,
    profile: str,
    warmup: int,
    repeat: int,
    build_info: dict[str, Any],
    hip_status: dict[str, Any],
) -> list[dict[str, Any]]:
    operation = str(case["campaign7_operation"])
    if operation == "transfer":
        return [
            benchmark_campaign7_transfer_case(
                case,
                profile=profile,
                warmup=warmup,
                repeat=repeat,
                build_info=build_info,
                hip_status=hip_status,
            )
        ]
    if operation == "commutation_consumers":
        return campaign7_commutation_rows(
            case,
            profile=profile,
            warmup=warmup,
            repeat=repeat,
            build_info=build_info,
            hip_status=hip_status,
        )
    if operation == "decisions":
        return campaign7_decision_rows(
            case=case,
            profile=profile,
            build_info=build_info,
            hip_status=hip_status,
        )

    if operation in {"simplify", "simplify_duplicate_pressure"}:
        source_row = benchmark_simplify_case(
            case,
            profile="simplify-campaign4-generic-multiword",
            warmup=warmup,
            repeat=repeat,
            build_info=build_info,
            hip_status=hip_status,
        )
        mode = (
            "simplify_duplicate_pressure"
            if operation == "simplify_duplicate_pressure"
            else "retained_simplify"
        )
        final_status = (
            "rejected_with_evidence"
            if "duplicate_pressure" in operation
            else ("passed" if profile == "campaign7-profiler" else "retained")
        )
        return [
            campaign7_enrich_row(
                source_row,
                profile=profile,
                operation=(
                    "duplicate_pressure_probe"
                    if "duplicate_pressure" in operation
                    else (
                        "profiler_smoke"
                        if profile == "campaign7-profiler"
                        else "retained_operation_smoke"
                    )
                ),
                mode="rocprof_availability" if profile == "campaign7-profiler" else mode,
                timing_boundary="profiler_only"
                if profile == "campaign7-profiler"
                else ("benchmark_only" if "duplicate_pressure" in operation else "device_resident"),
                build_info=build_info,
                hip_status=hip_status,
                final_status=final_status,
            )
        ]

    if operation in {"expectation", "matmul", "matmul_duplicate_pressure"}:
        if operation == "expectation":
            source_row = benchmark_campaign6_expectation_case(
                case,
                profile="campaign6-expectation-parity",
                warmup=warmup,
                repeat=repeat,
                build_info=build_info,
                hip_status=hip_status,
            )
            mode = "retained_expectation"
        else:
            source_row = benchmark_campaign6_matmul_case(
                case,
                profile="campaign6-matmul-parity",
                warmup=warmup,
                repeat=repeat,
                build_info=build_info,
                hip_status=hip_status,
            )
            mode = (
                "matmul_duplicate_pressure"
                if operation == "matmul_duplicate_pressure"
                else "retained_matmul"
            )
        final_status = (
            "rejected_with_evidence"
            if "duplicate_pressure" in operation
            else ("passed" if profile == "campaign7-profiler" else "retained")
        )
        return [
            campaign7_enrich_row(
                source_row,
                profile=profile,
                operation=(
                    "duplicate_pressure_probe"
                    if "duplicate_pressure" in operation
                    else (
                        "profiler_smoke"
                        if profile == "campaign7-profiler"
                        else "retained_operation_smoke"
                    )
                ),
                mode="rocprof_availability" if profile == "campaign7-profiler" else mode,
                timing_boundary="profiler_only"
                if profile == "campaign7-profiler"
                else ("benchmark_only" if "duplicate_pressure" in operation else "device_resident"),
                build_info=build_info,
                hip_status=hip_status,
                final_status=final_status,
            )
        ]

    raise ValueError(f"unknown Campaign 7 operation: {operation}")


def campaign5_base_row(
    *,
    case: dict[str, Any],
    profile: str,
    operation: str,
    mode: str,
    build_info: dict[str, Any],
    hip_status: dict[str, Any],
) -> dict[str, Any]:
    return {
        "campaign": "rocm_mi300x_campaign5",
        "profile": profile,
        "operation": operation,
        "mode": mode,
        "case": case["name"],
        "backend": "hip",
        "device_name": backend_device_name(hip_status),
        "gfx_target": first_device_field(hip_status, "gfx_target"),
        "hip_architectures": build_info.get("hip_architectures", "not_available"),
        "runtime_available": bool(hip_status.get("runtime_available", False)),
        "campaign5_terminal_statuses": dict(CAMPAIGN5_TERMINAL_STATUSES),
    }


def benchmark_campaign5_interop_case(
    case: dict[str, Any],
    *,
    profile: str,
    warmup: int,
    repeat: int,
    build_info: dict[str, Any],
    hip_status: dict[str, Any],
    consumer_probes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    lhs = make_operator(case, side="lhs")
    rhs = make_operator(case, side="rhs")
    with forced_cpu_backend("scalar"):
        expected = lhs.commutes_with(rhs)
    expected_array = np.asarray(expected, dtype=np.bool_).reshape(lhs.num_terms, rhs.num_terms)

    dataset = {
        "num_qubits": int(case["num_qubits"]),
        "lhs_terms": lhs.num_terms,
        "rhs_terms": rhs.num_terms,
        "entries": int(lhs.num_terms * rhs.num_terms),
        "words": (int(case["num_qubits"]) + 63) // 64,
        "term_weight": int(case["term_weight"]),
        "random_seed": int(case["random_seed"]),
        "operator_construction_method": "deterministic weighted dense labels",
    }

    matrix: Any = None
    dlpack_error = "HIP runtime unavailable"
    cuda_array_interface_error = "HIP runtime unavailable"
    digest = campaign5_commutation_digest(expected_array)
    to_host_timing: dict[str, float] | None = None
    count_timing: dict[str, float] | None = None
    correctness_passed = False

    if hip_status.get("runtime_available", False):
        lhs_device = lhs.to_device(device=0)
        rhs_device = rhs.to_device(device=0)
        matrix = lhs_device.commutes_with_device(rhs_device)

        dense_host, to_host_timing = timed_call(
            lambda: matrix.to_host(),
            warmup=warmup,
            repeat=repeat,
        )
        dense_host = np.asarray(dense_host, dtype=np.bool_).reshape(expected_array.shape)
        if not np.array_equal(dense_host, expected_array):
            raise RuntimeError(f"HIP Campaign 5 to_host mismatch for {case['name']}")

        count, count_timing = timed_call(
            lambda: matrix.count_commuting(),
            warmup=warmup,
            repeat=repeat,
        )
        if int(count) != int(expected_array.astype(np.uint64).sum(dtype=np.uint64)):
            raise RuntimeError(f"HIP Campaign 5 count mismatch for {case['name']}")
        digest = campaign5_commutation_digest(expected_array, consumer_sum=None)
        correctness_passed = True

        try:
            matrix.__dlpack__(max_version=(1, 0))
        except Exception as exc:
            dlpack_error = f"{type(exc).__name__}: {exc}"
        else:
            dlpack_error = "unexpectedly_available"

        try:
            matrix.__cuda_array_interface__
        except Exception as exc:
            cuda_array_interface_error = f"{type(exc).__name__}: {exc}"
        else:
            cuda_array_interface_error = "unexpectedly_available"

    rows: list[dict[str, Any]] = []
    mode_by_consumer = {
        "torch": "dlpack_pytorch",
        "cupy": "dlpack_cupy",
    }
    for consumer_name, mode in mode_by_consumer.items():
        probe = consumer_probes.get(
            consumer_name,
            {
                "consumer_library": consumer_name,
                "consumer_available": False,
                "consumer_import_error": "consumer probe was not run",
                "consumer_version": None,
                "consumer_backend": "unavailable",
                "consumer_device_visible": False,
                "consumer_device_error": "",
            },
        )
        row = campaign5_base_row(
            case=case,
            profile=profile,
            operation="commutation_interop",
            mode=mode,
            build_info=build_info,
            hip_status=hip_status,
        )
        row.update(
            {
                "dataset": dataset,
                "status": "rejected_with_evidence"
                if hip_status.get("runtime_available", False)
                else "unavailable",
                "final_status": "rejected_with_evidence"
                if hip_status.get("runtime_available", False)
                else "unavailable",
                "timing_boundary": "decision_only",
                "hip_dlpack_device_type": None,
                "hip_dlpack_device_type_name": "unavailable",
                "dlpack_unavailable_error": dlpack_error,
                "consumer_correctness_passed": False,
                "consumer_read_only_enforced": False,
                "consumer_mutation_error": "",
                "correctness_passed": correctness_passed,
                "correctness_digest": digest,
                "retention_decision": CAMPAIGN5_DLPACK_REJECTION_REASON,
                "candidate_probe_evidence_kind": "not_run_in_retained_build",
                "candidate_probe_source_file": None,
                "candidate_probe_consumer_correctness_passed": None,
                "candidate_probe_consumer_read_only_enforced": None,
                "candidate_probe_mutation_result": "not_run",
            }
        )
        row.update(probe)
        campaign5_add_null_timing_fields(row)
        if to_host_timing is not None:
            add_timing_fields(row, "hip_device_output_to_host", to_host_timing)
        if count_timing is not None:
            add_timing_fields(row, "hip_count_commuting_axis_none", count_timing)
        rows.append(row)

    guard = campaign5_base_row(
        case=case,
        profile=profile,
        operation="commutation_interop",
        mode="cuda_array_interface_guard",
        build_info=build_info,
        hip_status=hip_status,
    )
    guard.update(
        {
            "dataset": dataset,
            "status": "rejected_with_evidence"
            if hip_status.get("runtime_available", False)
            else "unavailable",
            "final_status": "rejected_with_evidence"
            if hip_status.get("runtime_available", False)
            else "unavailable",
            "timing_boundary": "decision_only",
            "hip_dlpack_device_type": None,
            "hip_dlpack_device_type_name": "unavailable",
            "cuda_array_interface_error": cuda_array_interface_error,
            "consumer_library": None,
            "consumer_version": None,
            "consumer_backend": None,
            "consumer_available": False,
            "consumer_import_error": "",
            "consumer_correctness_passed": False,
            "consumer_read_only_enforced": False,
            "consumer_mutation_error": "",
            "correctness_passed": correctness_passed,
            "correctness_digest": digest,
            "retention_decision": (
                "HIP __cuda_array_interface__ remains unavailable because HIP pointers "
                "must not be presented as CUDA memory."
            ),
        }
    )
    campaign5_add_null_timing_fields(guard)
    if to_host_timing is not None:
        add_timing_fields(guard, "hip_device_output_to_host", to_host_timing)
    if count_timing is not None:
        add_timing_fields(guard, "hip_count_commuting_axis_none", count_timing)
    rows.append(guard)
    return rows


def campaign5_decision_rows(
    *,
    case: dict[str, Any],
    profile: str,
    build_info: dict[str, Any],
    hip_status: dict[str, Any],
) -> list[dict[str, Any]]:
    decisions = [
        (
            "streams",
            "stream_graph_decision",
            "stream_graph_probe",
            "rejected_with_evidence",
            "No public stream handle ownership, synchronization, and error propagation contract is accepted.",
        ),
        (
            "graphs",
            "stream_graph_decision",
            "stream_graph_probe",
            "rejected_with_evidence",
            "No public graph replay lifetime and shape-change contract is accepted.",
        ),
        (
            "workspaces",
            "workspace_decision",
            "workspace_probe",
            "rejected_with_evidence",
            "No ownership-safe public workspace API or 10 percent retained-operation speedup is accepted.",
        ),
        (
            "expectation",
            "expectation_decision",
            "expectation_decision",
            "out_of_scope_with_next_trigger",
            "HIP expectation parity needs CPU/CUDA fixtures promoted in a separate campaign.",
        ),
        (
            "matmul",
            "matmul_decision",
            "matmul_decision",
            "out_of_scope_with_next_trigger",
            "HIP matmul parity needs CPU/CUDA fixtures promoted in a separate campaign.",
        ),
        (
            "portability",
            "portability_decision",
            "portability_decision",
            "out_of_scope_with_next_trigger",
            "Campaign 5 evidence remains MI300X gfx942 only.",
        ),
        (
            "ROCm wheels",
            "packaging_decision",
            "packaging_decision",
            "out_of_scope_with_next_trigger",
            "ROCm binary wheels need a separate packaging and CI support campaign.",
        ),
        (
            "multi-GPU",
            "multi_gpu_decision",
            "multi_gpu_decision",
            "out_of_scope_with_next_trigger",
            "Multi-GPU ROCm needs a backend-neutral device ownership design.",
        ),
        (
            "simultaneous CUDA+HIP",
            "multi_backend_decision",
            "simultaneous_cuda_hip_decision",
            "unavailable",
            "WOLFGANG_ENABLE_CUDA=ON with WOLFGANG_ENABLE_HIP=ON remains a configure-time error.",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for item, operation, mode, final_status, reason in decisions:
        row = campaign5_base_row(
            case=case,
            profile=profile,
            operation=operation,
            mode=mode,
            build_info=build_info,
            hip_status=hip_status,
        )
        row.update(
            {
                "decision_item": item,
                "status": final_status,
                "final_status": final_status,
                "timing_boundary": "decision_only",
                "decision_reason": reason,
                "hip_dlpack_device_type": None,
                "hip_dlpack_device_type_name": "unavailable",
                "consumer_library": None,
                "consumer_version": None,
                "consumer_backend": None,
                "consumer_available": False,
                "consumer_import_error": "",
                "consumer_correctness_passed": False,
                "consumer_read_only_enforced": False,
                "consumer_mutation_error": "",
                "correctness_digest": None,
            }
        )
        campaign5_add_null_timing_fields(row)
        rows.append(row)
    return rows


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    profile = "smoke" if args.smoke else args.profile
    consumer_probes: dict[str, dict[str, Any]] = {}
    if profile in CAMPAIGN5_PROFILES:
        consumer_probes = {
            "torch": probe_pytorch_rocm_consumer(),
            "cupy": probe_cupy_rocm_consumer(),
        }
    build_info = core._build_info()
    hip_status = core._hip_status()
    cases = PROFILE_CASES[profile]
    if profile in CAMPAIGN2_PROFILES:
        results = [
            benchmark_campaign2_case(
                case,
                profile=profile,
                warmup=args.warmup,
                repeat=args.repeat,
                build_info=build_info,
                hip_status=hip_status,
            )
            for case in cases
        ]
    elif profile in CAMPAIGN7_PROFILES:
        results = []
        for case in cases:
            results.extend(
                benchmark_campaign7_case(
                    case,
                    profile=profile,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    build_info=build_info,
                    hip_status=hip_status,
                )
            )
    elif profile in CAMPAIGN6_PROFILES:
        results = []
        for case in cases:
            if "statevector_dtype" in case:
                results.append(
                    benchmark_campaign6_expectation_case(
                        case,
                        profile=profile,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        build_info=build_info,
                        hip_status=hip_status,
                    )
                )
            else:
                results.append(
                    benchmark_campaign6_matmul_case(
                        case,
                        profile=profile,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        build_info=build_info,
                        hip_status=hip_status,
                    )
                )
    elif profile in CAMPAIGN5_PROFILES:
        if profile == "interop-campaign5-stream-workspace-decisions":
            results = campaign5_decision_rows(
                case=cases[0],
                profile=profile,
                build_info=build_info,
                hip_status=hip_status,
            )
        else:
            results = []
            for case in cases:
                results.extend(
                    benchmark_campaign5_interop_case(
                        case,
                        profile=profile,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        build_info=build_info,
                        hip_status=hip_status,
                        consumer_probes=consumer_probes,
                    )
                )
    elif profile in CAMPAIGN3_SIMPLIFY_PROFILES or profile in CAMPAIGN4_SIMPLIFY_PROFILES:
        results = [
            benchmark_simplify_case(
                case,
                profile=profile,
                warmup=args.warmup,
                repeat=args.repeat,
                build_info=build_info,
                hip_status=hip_status,
            )
            for case in cases
        ]
    else:
        results = [
            benchmark_case(
                case,
                warmup=args.warmup,
                repeat=args.repeat,
                build_info=build_info,
                hip_status=hip_status,
            )
            for case in cases
        ]
    return {
        "benchmark": "rocm_kernels",
        "profile": profile,
        "command": command_string(),
        "git_commit": git_commit(),
        "fastpauli_version": wolfgang.__version__,
        "fastpauli_build_info": build_info,
        "hip_status": hip_status,
        "environment": benchmark_environment(build_info, numpy_version=np.__version__),
        "timing_policy": {
            "repeat": args.repeat,
            "warmup": args.warmup,
            "summary": "median seconds",
        },
        "cases": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run the smoke profile")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_CASES),
        default="smoke",
        help="benchmark profile to run",
    )
    parser.add_argument("--repeat", type=int, default=3, help="timed repetitions")
    parser.add_argument("--warmup", type=int, default=1, help="warmup repetitions")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()
    if args.repeat <= 0:
        raise SystemExit("--repeat must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    report = build_report(args)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.json or args.output is None:
        print(payload)


if __name__ == "__main__":
    main()

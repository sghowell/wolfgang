#!/usr/bin/env python3
"""CUDA scaling benchmark for Wolfgang hot paths.

The fixed ``bench_cuda_kernels.py`` profiles are intentionally stable evidence
targets.  This benchmark fans the same correctness-checked datasets across
multiple sizes so profiling runs can see where launch overhead, transfer cost,
kernel work, and CPU baselines cross over.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import wolfgang_quantum as wolfgang
import wolfgang_quantum._wolfgang_core as core

try:
    from _benchmark_metadata import benchmark_environment, command_string, git_commit
    from bench_cuda_kernels import (
        CAMPAIGN7_BITPACKED_DECISION_STATUS,
        CAMPAIGN7_COUNT_SPECIALIZATION_STATUS,
        CAMPAIGN8_REQUIRED_STATUS_FIELDS,
        add_campaign8_device_resident_consumer_timing_fields,
        add_campaign8_row_schema_fields,
        add_cupy_consumer_timing_fields,
        add_cupy_dlpack_consumer_timing_fields,
        add_device_output_consumer_timing_fields,
        add_preallocated_commutation_timing_fields,
        add_private_fused_consumer_timing_fields,
        add_public_conflict_degrees_timing_fields,
        add_public_device_output_commutation_timing_fields,
        add_reused_device_output_commutation_timing_fields,
        add_torch_dlpack_consumer_timing_fields,
        assert_same_operator,
        campaign4_commutation_output_target,
        case_result,
        forced_cpu_backend,
        generate_operator,
        normalized_statevector,
        timed_cupy_device_output_consumer_cuda_commutation,
        timed_cupy_dlpack_consumer_cuda_commutation,
        timed_preallocated_cuda_commutation,
        timed_private_campaign8_device_resident_consumer_cuda_commutation,
        timed_private_fused_consumer_cuda_commutation,
        timed_public_conflict_degrees_cuda_commutation,
        timed_public_device_output_allocate_cuda_commutation,
        timed_public_device_output_consumer_cuda_commutation,
        timed_public_device_output_cuda_array_interface_export,
        timed_public_device_output_reuse_cuda_commutation,
        timed_public_device_output_to_host_cuda_commutation,
        timed_reused_device_output_cuda_commutation,
        timed_torch_dlpack_consumer_cuda_commutation,
    )
except ModuleNotFoundError:
    from benchmarks._benchmark_metadata import (
        benchmark_environment,
        command_string,
        git_commit,
    )
    from benchmarks.bench_cuda_kernels import (
        CAMPAIGN7_BITPACKED_DECISION_STATUS,
        CAMPAIGN7_COUNT_SPECIALIZATION_STATUS,
        CAMPAIGN8_REQUIRED_STATUS_FIELDS,
        add_campaign8_device_resident_consumer_timing_fields,
        add_campaign8_row_schema_fields,
        add_cupy_consumer_timing_fields,
        add_cupy_dlpack_consumer_timing_fields,
        add_device_output_consumer_timing_fields,
        add_preallocated_commutation_timing_fields,
        add_private_fused_consumer_timing_fields,
        add_public_conflict_degrees_timing_fields,
        add_public_device_output_commutation_timing_fields,
        add_reused_device_output_commutation_timing_fields,
        add_torch_dlpack_consumer_timing_fields,
        assert_same_operator,
        campaign4_commutation_output_target,
        case_result,
        forced_cpu_backend,
        generate_operator,
        normalized_statevector,
        timed_cupy_device_output_consumer_cuda_commutation,
        timed_cupy_dlpack_consumer_cuda_commutation,
        timed_preallocated_cuda_commutation,
        timed_private_campaign8_device_resident_consumer_cuda_commutation,
        timed_private_fused_consumer_cuda_commutation,
        timed_public_conflict_degrees_cuda_commutation,
        timed_public_device_output_allocate_cuda_commutation,
        timed_public_device_output_consumer_cuda_commutation,
        timed_public_device_output_cuda_array_interface_export,
        timed_public_device_output_reuse_cuda_commutation,
        timed_public_device_output_to_host_cuda_commutation,
        timed_reused_device_output_cuda_commutation,
        timed_torch_dlpack_consumer_cuda_commutation,
    )


OperationRunner = Callable[
    [dict[str, Any], dict[str, Any], np.random.Generator, int, int, int],
    dict[str, Any],
]

ROOT = Path(__file__).resolve().parents[1]

CAMPAIGN8_MODES_BY_TARGET = {
    "campaign8_device_resident_graph": "device_resident_graph",
    "campaign8_device_grouping_consumer": "device_grouping_consumer",
    "campaign8_interop": "dlpack_consumer",
    "campaign8_stream_graph": "stream_graph_probe",
    "campaign8_scatter_ab": "csr_scatter_ab",
    "campaign8_portability": "portability_check",
}

CAMPAIGN9_FINAL_STATUSES = (
    "accepted",
    "implemented",
    "rejected_with_evidence",
    "passed",
    "failed",
    "blocked_external",
    "not_applicable",
)
CAMPAIGN9_REQUIRED_ROW_FIELDS = (
    "campaign",
    "mode",
    "boundary",
    "campaign8_headroom_item",
    "final_status",
    "deferred_status_allowed",
    "decision_doc",
    "correctness_digest",
    "unavailable_reason",
    "git_revision",
    "cuda_driver",
    "cuda_runtime",
    "cuda_toolkit",
    "compiled_architectures",
    "gpu_name",
    "gpu_compute_capability",
)
CAMPAIGN9_MODE_METADATA = {
    "privileged_ncu": {
        "boundary": "profiler_only",
        "campaign8_headroom_item": 2,
        "decision_doc": "docs/plans/cuda_csr_scatter_campaign9_decision.md",
        "default_final_status": "passed",
        "env_status": "FASTPAULI_CUDA_CAMPAIGN9_PRIVILEGED_NCU_STATUS",
        "unavailable_reason": "",
    },
    "non_h100_portability": {
        "boundary": "private_benchmark_only",
        "campaign8_headroom_item": 1,
        "decision_doc": (
            "docs/benchmarks/reports/cuda_portability_campaign9_non_h100_nvidia_2026-04-29.md"
        ),
        "default_final_status": "blocked_external",
        "env_status": "FASTPAULI_CUDA_CAMPAIGN9_NON_H100_STATUS",
        "unavailable_reason": (
            "No named non-H100 NVIDIA host has been validated for Campaign 9 yet."
        ),
    },
    "public_grouping_api": {
        "boundary": "public_api",
        "campaign8_headroom_item": 3,
        "decision_doc": "docs/plans/cuda_fused_grouping_public_api_campaign9_contract.md",
        "default_final_status": "rejected_with_evidence",
        "env_status": "FASTPAULI_CUDA_CAMPAIGN9_GROUPING_STATUS",
        "unavailable_reason": "",
    },
    "dlpack_interop": {
        "boundary": "framework_consumer",
        "campaign8_headroom_item": 4,
        "decision_doc": "docs/plans/cuda_dlpack_interop_campaign9_contract.md",
        "default_final_status": "implemented",
        "env_status": "FASTPAULI_CUDA_CAMPAIGN9_DLPACK_STATUS",
        "unavailable_reason": "",
    },
    "stream_graph": {
        "boundary": "private_benchmark_only",
        "campaign8_headroom_item": 5,
        "decision_doc": "docs/plans/cuda_stream_graph_campaign9_contract.md",
        "default_final_status": "rejected_with_evidence",
        "env_status": "FASTPAULI_CUDA_CAMPAIGN9_STREAM_GRAPH_STATUS",
        "unavailable_reason": "",
    },
    "csr_scatter_reopen": {
        "boundary": "private_benchmark_only",
        "campaign8_headroom_item": 6,
        "decision_doc": "docs/plans/cuda_csr_scatter_campaign9_decision.md",
        "default_final_status": "rejected_with_evidence",
        "env_status": "FASTPAULI_CUDA_CAMPAIGN9_CSR_STATUS",
        "unavailable_reason": "",
    },
}
CAMPAIGN9_MODES_BY_TARGET = {
    "campaign9_privileged_ncu": "privileged_ncu",
    "campaign9_non_h100_portability": "non_h100_portability",
    "campaign9_public_grouping_api": "public_grouping_api",
    "campaign9_dlpack_interop": "dlpack_interop",
    "campaign9_stream_graph": "stream_graph",
    "campaign9_csr_scatter_reopen": "csr_scatter_reopen",
}
CAMPAIGN10_FINAL_STATUSES = (
    "implemented",
    "passed",
    "rejected_with_evidence",
    "blocked_external",
    "blocked_toolchain",
    "blocked_dependency",
)
CAMPAIGN10_REQUIRED_ROW_FIELDS = (
    "campaign",
    "mode",
    "campaign9_headroom_item",
    "final_status",
    "deferred_status_allowed",
    "decision_doc",
    "provider_instance_type",
    "gpu_name",
    "gpu_compute_capability",
    "cuda_driver",
    "cuda_runtime",
    "cuda_toolkit",
    "compiled_architectures",
    "architecture_compile_status",
    "git_revision",
    "command",
    "correctness_digest",
    "unavailable_reason",
)
CAMPAIGN10_MODE_METADATA = {
    "cross_arch_portability": {
        "boundary": "cross_architecture_source_build",
        "campaign9_headroom_item": 1,
        "decision_doc": "docs/plans/cuda_cross_architecture_campaign10_plan.md",
        "default_final_status": "passed",
        "env_status": "FASTPAULI_CUDA_CAMPAIGN10_PORTABILITY_STATUS",
        "unavailable_reason": "",
    },
    "dlpack_pytorch": {
        "boundary": "framework_consumer",
        "campaign9_headroom_item": 2,
        "decision_doc": "docs/plans/cuda_dlpack_interop_campaign9_contract.md",
        "default_final_status": "passed",
        "env_status": "FASTPAULI_CUDA_CAMPAIGN10_DLPACK_PYTORCH_STATUS",
        "unavailable_reason": "",
    },
    "public_grouping_api": {
        "boundary": "public_api_decision",
        "campaign9_headroom_item": 3,
        "decision_doc": "docs/plans/cuda_grouping_public_api_campaign10_contract.md",
        "default_final_status": "rejected_with_evidence",
        "env_status": "FASTPAULI_CUDA_CAMPAIGN10_GROUPING_STATUS",
        "unavailable_reason": "",
    },
    "stream_graph_reprobe": {
        "boundary": "private_benchmark_only",
        "campaign9_headroom_item": 4,
        "decision_doc": "docs/plans/cuda_stream_graph_campaign10_decision.md",
        "default_final_status": "rejected_with_evidence",
        "env_status": "FASTPAULI_CUDA_CAMPAIGN10_STREAM_GRAPH_STATUS",
        "unavailable_reason": "",
    },
    "csr_scatter_reprobe": {
        "boundary": "private_benchmark_only",
        "campaign9_headroom_item": 5,
        "decision_doc": "docs/plans/cuda_csr_scatter_campaign10_decision.md",
        "default_final_status": "rejected_with_evidence",
        "env_status": "FASTPAULI_CUDA_CAMPAIGN10_CSR_STATUS",
        "unavailable_reason": "",
    },
}
CAMPAIGN10_MODES_BY_TARGET = {
    "campaign10_portability": "cross_arch_portability",
    "campaign10_dlpack_pytorch": "dlpack_pytorch",
    "campaign10_public_grouping_api": "public_grouping_api",
    "campaign10_stream_graph_reprobe": "stream_graph_reprobe",
    "campaign10_csr_scatter_reprobe": "csr_scatter_reprobe",
}


def git_revision_full() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip()


SCALE_PROFILES: dict[str, dict[str, list[dict[str, Any]]]] = {
    "smoke": {
        "simplify_duplicate_pressure": [
            {"scale": "terms_16", "num_qubits": 4, "num_terms": 16, "term_weight": 2, "pool": 8}
        ],
        "statevector_expectation": [
            {"scale": "qubits_3_terms_8", "num_qubits": 3, "num_terms": 8, "term_weight": 2}
        ],
        "pairwise_commutation": [
            {"scale": "terms_8x8", "num_qubits": 4, "terms": 8, "term_weight": 2}
        ],
        "matmul_product_generation_simplify": [
            {"scale": "terms_4x4", "num_qubits": 4, "terms": 4, "term_weight": 2}
        ],
    },
    "default": {
        "simplify_duplicate_pressure": [
            {
                "scale": "terms_10000",
                "num_qubits": 16,
                "num_terms": 10000,
                "term_weight": 3,
                "pool": 1024,
            },
            {
                "scale": "terms_50000",
                "num_qubits": 16,
                "num_terms": 50000,
                "term_weight": 3,
                "pool": 1024,
            },
            {
                "scale": "terms_200000",
                "num_qubits": 16,
                "num_terms": 200000,
                "term_weight": 3,
                "pool": 2048,
            },
        ],
        "statevector_expectation": [
            {
                "scale": "qubits_10_terms_1024",
                "num_qubits": 10,
                "num_terms": 1024,
                "term_weight": 3,
            },
            {
                "scale": "qubits_12_terms_2048",
                "num_qubits": 12,
                "num_terms": 2048,
                "term_weight": 3,
            },
            {
                "scale": "qubits_14_terms_4096",
                "num_qubits": 14,
                "num_terms": 4096,
                "term_weight": 3,
            },
        ],
        "pairwise_commutation": [
            {"scale": "terms_1024x1024", "num_qubits": 16, "terms": 1024, "term_weight": 3},
            {"scale": "terms_2048x2048", "num_qubits": 16, "terms": 2048, "term_weight": 3},
            {"scale": "terms_4096x4096", "num_qubits": 16, "terms": 4096, "term_weight": 3},
        ],
        "matmul_product_generation_simplify": [
            {"scale": "terms_128x128", "num_qubits": 12, "terms": 128, "term_weight": 3},
            {"scale": "terms_256x256", "num_qubits": 12, "terms": 256, "term_weight": 3},
            {"scale": "terms_512x512", "num_qubits": 12, "terms": 512, "term_weight": 3},
        ],
    },
    "stress": {
        "simplify_duplicate_pressure": [
            {
                "scale": "terms_100000",
                "num_qubits": 16,
                "num_terms": 100000,
                "term_weight": 3,
                "pool": 2048,
            },
            {
                "scale": "terms_500000",
                "num_qubits": 16,
                "num_terms": 500000,
                "term_weight": 3,
                "pool": 4096,
            },
            {
                "scale": "terms_1000000",
                "num_qubits": 16,
                "num_terms": 1000000,
                "term_weight": 3,
                "pool": 8192,
            },
        ],
        "statevector_expectation": [
            {
                "scale": "qubits_14_terms_4096",
                "num_qubits": 14,
                "num_terms": 4096,
                "term_weight": 3,
            },
            {
                "scale": "qubits_15_terms_4096",
                "num_qubits": 15,
                "num_terms": 4096,
                "term_weight": 3,
            },
            {
                "scale": "qubits_16_terms_8192",
                "num_qubits": 16,
                "num_terms": 8192,
                "term_weight": 3,
            },
        ],
        "pairwise_commutation": [
            {"scale": "terms_4096x4096", "num_qubits": 16, "terms": 4096, "term_weight": 3},
            {"scale": "terms_8192x8192", "num_qubits": 16, "terms": 8192, "term_weight": 3},
            {"scale": "terms_10000x10000", "num_qubits": 16, "terms": 10000, "term_weight": 3},
        ],
        "matmul_product_generation_simplify": [
            {"scale": "terms_512x512", "num_qubits": 12, "terms": 512, "term_weight": 3},
            {"scale": "terms_1024x1024", "num_qubits": 12, "terms": 1024, "term_weight": 3},
            {"scale": "terms_2048x2048", "num_qubits": 12, "terms": 2048, "term_weight": 3},
        ],
    },
    "extreme": {
        "simplify_duplicate_pressure": [
            {
                "scale": "terms_2000000",
                "num_qubits": 16,
                "num_terms": 2000000,
                "term_weight": 3,
                "pool": 16384,
            },
            {
                "scale": "terms_5000000",
                "num_qubits": 16,
                "num_terms": 5000000,
                "term_weight": 3,
                "pool": 32768,
            },
        ],
        "statevector_expectation": [
            {
                "scale": "qubits_17_terms_8192",
                "num_qubits": 17,
                "num_terms": 8192,
                "term_weight": 3,
            },
            {
                "scale": "qubits_18_terms_8192",
                "num_qubits": 18,
                "num_terms": 8192,
                "term_weight": 3,
            },
        ],
        "pairwise_commutation": [
            {"scale": "terms_12000x12000", "num_qubits": 16, "terms": 12000, "term_weight": 3},
            {"scale": "terms_16384x16384", "num_qubits": 16, "terms": 16384, "term_weight": 3},
        ],
        "matmul_product_generation_simplify": [
            {"scale": "terms_3072x3072", "num_qubits": 12, "terms": 3072, "term_weight": 3},
            {"scale": "terms_4096x4096", "num_qubits": 12, "terms": 4096, "term_weight": 3},
        ],
    },
    "materialization": {
        "simplify_duplicate_pressure": [
            {
                "scale": "terms_200000_low_duplicate",
                "num_qubits": 16,
                "num_terms": 200000,
                "term_weight": 3,
                "pool": 190000,
            },
            {
                "scale": "terms_200000_medium_duplicate",
                "num_qubits": 16,
                "num_terms": 200000,
                "term_weight": 3,
                "pool": 50000,
            },
            {
                "scale": "terms_200000_high_duplicate",
                "num_qubits": 16,
                "num_terms": 200000,
                "term_weight": 3,
                "pool": 4096,
            },
            {
                "scale": "terms_200000_pathological_duplicate",
                "num_qubits": 16,
                "num_terms": 200000,
                "term_weight": 3,
                "pool": 1,
            },
        ],
        "statevector_expectation": [
            {
                "scale": "resident_qubits_14_terms_4096",
                "num_qubits": 14,
                "num_terms": 4096,
                "term_weight": 3,
            },
            {
                "scale": "resident_qubits_16_terms_8192",
                "num_qubits": 16,
                "num_terms": 8192,
                "term_weight": 3,
            },
        ],
        "pairwise_commutation": [
            {
                "scale": "host_output_terms_8192x8192",
                "num_qubits": 16,
                "terms": 8192,
                "term_weight": 3,
            },
            {
                "scale": "host_output_terms_12000x12000",
                "num_qubits": 16,
                "terms": 12000,
                "term_weight": 3,
            },
        ],
        "matmul_product_generation_simplify": [
            {
                "scale": "duplicate_product_terms_1024x1024",
                "num_qubits": 12,
                "terms": 1024,
                "term_weight": 3,
            },
            {
                "scale": "duplicate_product_terms_2048x2048",
                "num_qubits": 12,
                "terms": 2048,
                "term_weight": 3,
            },
        ],
    },
    "campaign4_workspace": {
        "simplify_duplicate_pressure": [
            {
                "scale": "oneword_low_duplicate_terms_200000",
                "num_qubits": 16,
                "num_terms": 200000,
                "term_weight": 3,
                "pool": 190000,
            },
            {
                "scale": "oneword_medium_duplicate_terms_200000",
                "num_qubits": 16,
                "num_terms": 200000,
                "term_weight": 3,
                "pool": 50000,
            },
            {
                "scale": "oneword_high_duplicate_terms_200000",
                "num_qubits": 16,
                "num_terms": 200000,
                "term_weight": 3,
                "pool": 4096,
            },
            {
                "scale": "oneword_pathological_duplicate_terms_200000",
                "num_qubits": 16,
                "num_terms": 200000,
                "term_weight": 3,
                "pool": 1,
            },
            {
                "scale": "multiword_medium_duplicate_terms_100000",
                "num_qubits": 96,
                "num_terms": 100000,
                "term_weight": 3,
                "pool": 25000,
            },
            {
                "scale": "multiword_high_duplicate_terms_100000",
                "num_qubits": 96,
                "num_terms": 100000,
                "term_weight": 3,
                "pool": 4096,
            },
        ],
        "statevector_expectation": [
            {
                "scale": "complex64_qubits_14_terms_4096",
                "num_qubits": 14,
                "num_terms": 4096,
                "term_weight": 3,
                "statevector_dtype": "complex64",
            },
            {
                "scale": "complex128_qubits_14_terms_4096",
                "num_qubits": 14,
                "num_terms": 4096,
                "term_weight": 3,
                "statevector_dtype": "complex128",
            },
            {
                "scale": "complex64_qubits_16_terms_8192",
                "num_qubits": 16,
                "num_terms": 8192,
                "term_weight": 3,
                "statevector_dtype": "complex64",
            },
            {
                "scale": "complex128_qubits_16_terms_8192",
                "num_qubits": 16,
                "num_terms": 8192,
                "term_weight": 3,
                "statevector_dtype": "complex128",
            },
        ],
        "pairwise_commutation": [
            {
                "scale": "host_vector_terms_8192x8192",
                "num_qubits": 16,
                "terms": 8192,
                "term_weight": 3,
                "output_target": "host_vector",
            },
            {
                "scale": "caller_owned_host_bytes_terms_8192x8192",
                "num_qubits": 16,
                "terms": 8192,
                "term_weight": 3,
                "output_target": "caller_owned_host_bytes",
            },
            {
                "scale": "caller_owned_device_bytes_terms_8192x8192",
                "num_qubits": 16,
                "terms": 8192,
                "term_weight": 3,
                "output_target": "caller_owned_device_bytes",
            },
            {
                "scale": "bitpacked_device_words_terms_8192x8192",
                "num_qubits": 16,
                "terms": 8192,
                "term_weight": 3,
                "output_target": "bitpacked_device_words",
            },
        ],
        "matmul_product_generation_simplify": [
            {"scale": "terms_512x512", "num_qubits": 12, "terms": 512, "term_weight": 3},
            {"scale": "terms_1024x1024", "num_qubits": 12, "terms": 1024, "term_weight": 3},
            {"scale": "terms_2048x2048", "num_qubits": 12, "terms": 2048, "term_weight": 3},
            {"scale": "terms_4096x4096", "num_qubits": 12, "terms": 4096, "term_weight": 3},
        ],
    },
    "campaign5_device_output": {
        "pairwise_commutation": [
            {
                "scale": "device_output_terms_1024x1024",
                "num_qubits": 16,
                "terms": 1024,
                "term_weight": 3,
                "output_target": "device_uint8_matrix",
            },
            {
                "scale": "device_output_terms_2048x2048",
                "num_qubits": 16,
                "terms": 2048,
                "term_weight": 3,
                "output_target": "device_uint8_matrix",
            },
            {
                "scale": "device_output_terms_4096x4096",
                "num_qubits": 16,
                "terms": 4096,
                "term_weight": 3,
                "output_target": "device_uint8_matrix",
            },
            {
                "scale": "device_output_terms_8192x8192",
                "num_qubits": 16,
                "terms": 8192,
                "term_weight": 3,
                "output_target": "device_uint8_matrix",
            },
            {
                "scale": "device_output_terms_16384x16384",
                "num_qubits": 16,
                "terms": 16384,
                "term_weight": 3,
                "output_target": "device_uint8_matrix",
            },
        ],
    },
    "campaign6_consumers": {
        "pairwise_commutation": [
            {
                "scale": "dense_consumer_terms_2048x2048",
                "num_qubits": 16,
                "terms": 2048,
                "term_weight": 3,
                "output_target": "device_uint8_matrix_consumer",
            },
            {
                "scale": "dense_consumer_terms_8192x8192",
                "num_qubits": 16,
                "terms": 8192,
                "term_weight": 3,
                "output_target": "device_uint8_matrix_consumer",
            },
            {
                "scale": "dense_consumer_terms_16384x16384",
                "num_qubits": 16,
                "terms": 16384,
                "term_weight": 3,
                "output_target": "device_uint8_matrix_consumer",
            },
        ],
    },
    "campaign7_fused_consumers": {
        "pairwise_commutation": [
            {
                "scale": "fused_consumer_terms_2048x2048",
                "num_qubits": 16,
                "terms": 2048,
                "term_weight": 3,
                "output_target": "device_uint8_matrix_fused_consumer",
            },
            {
                "scale": "fused_consumer_terms_8192x8192",
                "num_qubits": 16,
                "terms": 8192,
                "term_weight": 3,
                "output_target": "device_uint8_matrix_fused_consumer",
            },
            {
                "scale": "fused_consumer_terms_16384x16384",
                "num_qubits": 16,
                "terms": 16384,
                "term_weight": 3,
                "output_target": "device_uint8_matrix_fused_consumer",
            },
        ],
    },
}

SCALE_PROFILES["fused-graph-stress"] = SCALE_PROFILES["campaign7_fused_consumers"]
SCALE_PROFILES["campaign8-device-graph"] = {
    "pairwise_commutation": [
        {
            "scale": "campaign8_graph_terms_2048x2048",
            "num_qubits": 16,
            "terms": 2048,
            "term_weight": 3,
            "output_target": "campaign8_device_resident_graph",
        },
        {
            "scale": "campaign8_graph_terms_8192x8192",
            "num_qubits": 16,
            "terms": 8192,
            "term_weight": 3,
            "output_target": "campaign8_device_resident_graph",
        },
        {
            "scale": "campaign8_graph_terms_16384x16384",
            "num_qubits": 16,
            "terms": 16384,
            "term_weight": 3,
            "output_target": "campaign8_device_resident_graph",
        },
    ]
}
SCALE_PROFILES["campaign8-grouping-consumer"] = {
    "pairwise_commutation": [
        {
            "scale": "campaign8_grouping_terms_2048x2048",
            "num_qubits": 16,
            "terms": 2048,
            "term_weight": 3,
            "output_target": "campaign8_device_grouping_consumer",
        },
        {
            "scale": "campaign8_grouping_terms_8192x8192",
            "num_qubits": 16,
            "terms": 8192,
            "term_weight": 3,
            "output_target": "campaign8_device_grouping_consumer",
        },
        {
            "scale": "campaign8_grouping_terms_16384x16384",
            "num_qubits": 16,
            "terms": 16384,
            "term_weight": 3,
            "output_target": "campaign8_device_grouping_consumer",
        },
    ]
}
SCALE_PROFILES["campaign8-interop"] = {
    "pairwise_commutation": [
        {
            "scale": "campaign8_interop_terms_2048x2048",
            "num_qubits": 16,
            "terms": 2048,
            "term_weight": 3,
            "output_target": "campaign8_interop",
        }
    ]
}
SCALE_PROFILES["campaign8-stream-graph"] = {
    "pairwise_commutation": [
        {
            "scale": "campaign8_stream_graph_terms_2048x2048",
            "num_qubits": 16,
            "terms": 2048,
            "term_weight": 3,
            "output_target": "campaign8_stream_graph",
        }
    ]
}
SCALE_PROFILES["campaign8-scatter-ab"] = {
    "pairwise_commutation": [
        {
            "scale": "campaign8_scatter_ab_terms_2048x2048",
            "num_qubits": 16,
            "terms": 2048,
            "term_weight": 3,
            "output_target": "campaign8_scatter_ab",
        }
    ]
}
SCALE_PROFILES["campaign8-portability"] = {
    "pairwise_commutation": [
        {
            "scale": "campaign8_portability_terms_2048x2048",
            "num_qubits": 16,
            "terms": 2048,
            "term_weight": 3,
            "output_target": "campaign8_portability",
        }
    ]
}
SCALE_PROFILES["campaign9-privileged-ncu"] = {
    "pairwise_commutation": [
        {
            "scale": "campaign9_privileged_ncu_terms_2048x2048",
            "num_qubits": 16,
            "terms": 2048,
            "term_weight": 3,
            "output_target": "campaign9_privileged_ncu",
        },
        {
            "scale": "campaign9_privileged_ncu_terms_8192x8192",
            "num_qubits": 16,
            "terms": 8192,
            "term_weight": 3,
            "output_target": "campaign9_privileged_ncu",
        },
    ]
}
SCALE_PROFILES["campaign9-non-h100-portability"] = {
    "pairwise_commutation": [
        {
            "scale": "campaign9_non_h100_portability_terms_2048x2048",
            "num_qubits": 16,
            "terms": 2048,
            "term_weight": 3,
            "output_target": "campaign9_non_h100_portability",
        }
    ]
}
SCALE_PROFILES["campaign9-public-grouping-api"] = {
    "pairwise_commutation": [
        {
            "scale": "campaign9_public_grouping_api_terms_2048x2048",
            "num_qubits": 16,
            "terms": 2048,
            "term_weight": 3,
            "output_target": "campaign9_public_grouping_api",
        }
    ]
}
SCALE_PROFILES["campaign9-dlpack-interop"] = {
    "pairwise_commutation": [
        {
            "scale": "campaign9_dlpack_interop_terms_2048x2048",
            "num_qubits": 16,
            "terms": 2048,
            "term_weight": 3,
            "output_target": "campaign9_dlpack_interop",
        }
    ]
}
SCALE_PROFILES["campaign9-stream-graph"] = {
    "pairwise_commutation": [
        {
            "scale": "campaign9_stream_graph_terms_2048x2048",
            "num_qubits": 16,
            "terms": 2048,
            "term_weight": 3,
            "output_target": "campaign9_stream_graph",
        }
    ]
}
SCALE_PROFILES["campaign9-csr-scatter-ab"] = {
    "pairwise_commutation": [
        {
            "scale": "campaign9_csr_scatter_reopen_terms_2048x2048",
            "num_qubits": 16,
            "terms": 2048,
            "term_weight": 3,
            "output_target": "campaign9_csr_scatter_reopen",
        }
    ]
}
SCALE_PROFILES["campaign10-portability"] = {
    "pairwise_commutation": [
        {
            "scale": "campaign10_portability_terms_2048x2048",
            "num_qubits": 16,
            "terms": 2048,
            "term_weight": 3,
            "output_target": "campaign10_portability",
        },
        {
            "scale": "campaign10_portability_terms_8192x8192",
            "num_qubits": 16,
            "terms": 8192,
            "term_weight": 3,
            "output_target": "campaign10_portability",
        },
    ]
}
SCALE_PROFILES["campaign10-dlpack-pytorch"] = {
    "pairwise_commutation": [
        {
            "scale": "campaign10_dlpack_pytorch_terms_2048x2048",
            "num_qubits": 16,
            "terms": 2048,
            "term_weight": 3,
            "output_target": "campaign10_dlpack_pytorch",
        }
    ]
}
SCALE_PROFILES["campaign10-public-grouping-api"] = {
    "pairwise_commutation": [
        {
            "scale": "campaign10_public_grouping_api_terms_2048x2048",
            "num_qubits": 16,
            "terms": 2048,
            "term_weight": 3,
            "output_target": "campaign10_public_grouping_api",
        }
    ]
}
SCALE_PROFILES["campaign10-stream-graph-reprobe"] = {
    "pairwise_commutation": [
        {
            "scale": "campaign10_stream_graph_terms_2048x2048",
            "num_qubits": 16,
            "terms": 2048,
            "term_weight": 3,
            "output_target": "campaign10_stream_graph_reprobe",
        },
        {
            "scale": "campaign10_stream_graph_terms_8192x8192",
            "num_qubits": 16,
            "terms": 8192,
            "term_weight": 3,
            "output_target": "campaign10_stream_graph_reprobe",
        },
    ]
}
SCALE_PROFILES["campaign10-csr-scatter-reprobe"] = {
    "pairwise_commutation": [
        {
            "scale": "campaign10_csr_scatter_terms_2048x2048",
            "num_qubits": 16,
            "terms": 2048,
            "term_weight": 3,
            "output_target": "campaign10_csr_scatter_reprobe",
        },
        {
            "scale": "campaign10_csr_scatter_terms_8192x8192",
            "num_qubits": 16,
            "terms": 8192,
            "term_weight": 3,
            "output_target": "campaign10_csr_scatter_reprobe",
        },
    ]
}


def planned_cases(profile: str, operations: list[str]) -> list[dict[str, Any]]:
    profile_scales = SCALE_PROFILES[profile]
    return [
        {
            "name": operation,
            "scales": [scale["scale"] for scale in profile_scales[operation]],
        }
        for operation in operations
    ]


def _scale_seed(seed: int, operation_index: int, scale_index: int) -> int:
    return seed + 1009 * operation_index + 9176 * scale_index


def _campaign9_gpu_metadata(cuda_status: dict[str, Any]) -> tuple[str, str]:
    devices = cuda_status.get("devices") or []
    if not devices:
        return "", ""
    first = devices[0]
    capability = first.get("compute_capability", "")
    if isinstance(capability, (list, tuple)) and len(capability) >= 2:
        capability_value = f"{capability[0]}.{capability[1]}"
    else:
        capability_value = str(capability) if capability else ""
    return str(first.get("name", "")), capability_value


def _campaign9_final_status(mode: str) -> str:
    metadata = CAMPAIGN9_MODE_METADATA[mode]
    status = os.environ.get(str(metadata["env_status"]), str(metadata["default_final_status"]))
    if status == "deferred":
        raise RuntimeError("Campaign 9 rows may not use final_status='deferred'")
    if status not in CAMPAIGN9_FINAL_STATUSES:
        allowed = ", ".join(CAMPAIGN9_FINAL_STATUSES)
        raise RuntimeError(f"invalid Campaign 9 final_status {status!r}; expected one of {allowed}")
    return status


def _campaign9_digest(value: Any) -> str:
    if value in (None, ""):
        return ""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def add_campaign9_row_schema_fields(
    result: dict[str, Any],
    *,
    mode: str,
    build_info: dict[str, Any],
    cuda_status: dict[str, Any],
    git_revision: str,
    unavailable_reason: str = "",
    cuda_unavailable: bool = False,
) -> None:
    metadata = CAMPAIGN9_MODE_METADATA[mode]
    gpu_name, gpu_compute_capability = _campaign9_gpu_metadata(cuda_status)
    result["campaign"] = "cuda_deferred_headroom_campaign9"
    result["mode"] = mode
    result["boundary"] = metadata["boundary"]
    result["campaign8_headroom_item"] = int(metadata["campaign8_headroom_item"])
    result["final_status"] = (
        "blocked_external" if cuda_unavailable else _campaign9_final_status(mode)
    )
    result["deferred_status_allowed"] = False
    result["decision_doc"] = metadata["decision_doc"]
    result["correctness_digest"] = _campaign9_digest(result.get("correctness_digest", ""))
    result["unavailable_reason"] = unavailable_reason or (
        str(metadata["unavailable_reason"]) if cuda_unavailable else ""
    )
    result["git_revision"] = git_revision
    result["cuda_driver"] = str(cuda_status.get("driver_version", ""))
    result["cuda_runtime"] = str(cuda_status.get("runtime_version", ""))
    result["cuda_toolkit"] = str(build_info.get("cuda_toolkit_version", ""))
    result["compiled_architectures"] = str(build_info.get("cuda_architectures", "")).replace(
        ",", ";"
    )
    result["gpu_name"] = gpu_name
    result["gpu_compute_capability"] = gpu_compute_capability
    missing = [field for field in CAMPAIGN9_REQUIRED_ROW_FIELDS if field not in result]
    if missing:
        raise RuntimeError(f"Campaign 9 row omitted required field(s): {missing}")


def _campaign10_final_status(
    mode: str,
    *,
    cuda_unavailable: bool = False,
    dependency_unavailable: bool = False,
) -> str:
    if cuda_unavailable:
        return "blocked_external"
    if dependency_unavailable:
        return "blocked_dependency"
    metadata = CAMPAIGN10_MODE_METADATA[mode]
    status = os.environ.get(str(metadata["env_status"]), str(metadata["default_final_status"]))
    if status == "deferred":
        raise RuntimeError("Campaign 10 rows may not use final_status='deferred'")
    if status not in CAMPAIGN10_FINAL_STATUSES:
        allowed = ", ".join(CAMPAIGN10_FINAL_STATUSES)
        raise RuntimeError(
            f"invalid Campaign 10 final_status {status!r}; expected one of {allowed}"
        )
    return status


def add_campaign10_row_schema_fields(
    result: dict[str, Any],
    *,
    mode: str,
    build_info: dict[str, Any],
    cuda_status: dict[str, Any],
    git_revision: str,
    unavailable_reason: str = "",
    cuda_unavailable: bool = False,
    dependency_unavailable: bool = False,
) -> None:
    metadata = CAMPAIGN10_MODE_METADATA[mode]
    gpu_name, gpu_compute_capability = _campaign9_gpu_metadata(cuda_status)
    result["campaign"] = "cuda_cross_architecture_campaign10"
    result["mode"] = mode
    result["boundary"] = metadata["boundary"]
    result["campaign9_headroom_item"] = int(metadata["campaign9_headroom_item"])
    result["final_status"] = _campaign10_final_status(
        mode,
        cuda_unavailable=cuda_unavailable,
        dependency_unavailable=dependency_unavailable,
    )
    result["deferred_status_allowed"] = False
    result["decision_doc"] = metadata["decision_doc"]
    result["provider_instance_type"] = os.environ.get(
        "FASTPAULI_CAMPAIGN10_PROVIDER_INSTANCE_TYPE",
        "not_available_to_agent",
    )
    result["gpu_name"] = gpu_name
    result["gpu_compute_capability"] = gpu_compute_capability
    result["cuda_driver"] = str(cuda_status.get("driver_version", ""))
    result["cuda_runtime"] = str(cuda_status.get("runtime_version", ""))
    result["cuda_toolkit"] = str(build_info.get("cuda_toolkit_version", ""))
    result["compiled_architectures"] = str(build_info.get("cuda_architectures", "")).replace(
        ",", ";"
    )
    result["architecture_compile_status"] = os.environ.get(
        "FASTPAULI_CAMPAIGN10_ARCHITECTURE_COMPILE_STATUS",
        "not_checked",
    )
    result["git_revision"] = git_revision
    result["command"] = command_string()
    result["correctness_digest"] = _campaign9_digest(result.get("correctness_digest", ""))
    result["unavailable_reason"] = unavailable_reason or (
        str(metadata["unavailable_reason"]) if (cuda_unavailable or dependency_unavailable) else ""
    )
    missing = [field for field in CAMPAIGN10_REQUIRED_ROW_FIELDS if field not in result]
    if missing:
        raise RuntimeError(f"Campaign 10 row omitted required field(s): {missing}")


def run_simplify_scale(
    scale: dict[str, Any],
    build_info: dict[str, Any],
    rng: np.random.Generator,
    seed: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    op, metadata = generate_operator(
        rng=rng,
        num_qubits=int(scale["num_qubits"]),
        num_terms=int(scale["num_terms"]),
        term_weight=int(scale["term_weight"]),
        duplicate_pool_size=int(scale["pool"]),
    )
    metadata = {**metadata, "random_seed": seed, "scale": scale["scale"]}
    expected = op.simplify()
    metadata = {**metadata, "survivor_count": expected.num_terms}
    device_op = op.to_device()
    assert_same_operator(device_op.simplify().to_host(), expected)
    result = case_result(
        name="simplify_duplicate_pressure",
        dataset=metadata,
        cpu_fn=lambda: op.simplify(),
        cuda_transfer_fn=lambda: op.to_device().simplify().to_host(),
        cuda_resident_fn=lambda: device_op.simplify(),
        build_info=build_info,
        warmup=warmup,
        repeat=repeat,
    )
    result["scale"] = scale["scale"]
    return result


def run_expectation_scale(
    scale: dict[str, Any],
    build_info: dict[str, Any],
    rng: np.random.Generator,
    seed: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    op, metadata = generate_operator(
        rng=rng,
        num_qubits=int(scale["num_qubits"]),
        num_terms=int(scale["num_terms"]),
        term_weight=int(scale["term_weight"]),
    )
    psi = normalized_statevector(rng, int(scale["num_qubits"]))
    statevector_dtype = str(scale.get("statevector_dtype", "complex128"))
    if statevector_dtype == "complex64":
        psi = psi.astype(np.complex64)
        rtol = 1.0e-5
        atol = 1.0e-5
    elif statevector_dtype == "complex128":
        rtol = 1.0e-12
        atol = 1.0e-12
    else:
        raise ValueError(f"unsupported statevector_dtype: {statevector_dtype}")
    metadata = {
        **metadata,
        "random_seed": seed,
        "scale": scale["scale"],
        "statevector_length": psi.size,
        "statevector_dtype": statevector_dtype,
        "statevector_random_seed": seed,
    }
    expected = op.expectation_statevector(psi)
    device_op = op.to_device()
    if not np.allclose(device_op.expectation_statevector(psi), expected, rtol=rtol, atol=atol):
        raise RuntimeError("CUDA and CPU statevector expectation differ")
    result = case_result(
        name="statevector_expectation",
        dataset=metadata,
        cpu_fn=lambda: op.expectation_statevector(psi),
        cuda_transfer_fn=lambda: op.to_device().expectation_statevector(psi),
        cuda_resident_fn=lambda: device_op.expectation_statevector(psi),
        build_info=build_info,
        warmup=warmup,
        repeat=repeat,
        notes=["device-resident timing keeps the operator resident but copies host psi"],
    )
    result["scale"] = scale["scale"]
    return result


def run_commutation_scale(
    scale: dict[str, Any],
    build_info: dict[str, Any],
    rng: np.random.Generator,
    seed: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    terms = int(scale["terms"])
    lhs, lhs_metadata = generate_operator(
        rng=rng,
        num_qubits=int(scale["num_qubits"]),
        num_terms=terms,
        term_weight=int(scale["term_weight"]),
    )
    rhs, rhs_metadata = generate_operator(
        rng=rng,
        num_qubits=int(scale["num_qubits"]),
        num_terms=terms,
        term_weight=int(scale["term_weight"]),
    )
    entries = terms * terms
    max_entries = max(entries, 100_000_000)
    metadata = {
        "scale": scale["scale"],
        "output_target": str(scale.get("output_target", campaign4_commutation_output_target())),
        "lhs_terms": terms,
        "rhs_terms": terms,
        "num_qubits": lhs.num_qubits,
        "entries": entries,
        "max_commutation_matrix_entries": max_entries,
        "term_weight_distribution": f"fixed term_weight={scale['term_weight']}",
        "lhs_duplicate_rate": lhs_metadata["duplicate_rate"],
        "rhs_duplicate_rate": rhs_metadata["duplicate_rate"],
        "coefficient_dtype": "complex128",
        "random_seed": seed,
        "operator_construction_method": "deterministic weighted lhs/rhs labels",
    }
    lhs_device = lhs.to_device()
    rhs_device = rhs.to_device()
    with forced_cpu_backend("scalar"):
        expected = lhs.commutes_with(rhs, max_commutation_matrix_entries=max_entries)
    if not np.array_equal(
        lhs_device.commutes_with(rhs_device, max_commutation_matrix_entries=max_entries),
        expected,
    ):
        raise RuntimeError("CUDA and CPU commutation outputs differ")
    result = case_result(
        name="pairwise_commutation",
        dataset=metadata,
        cpu_fn=lambda: lhs.commutes_with(rhs, max_commutation_matrix_entries=max_entries),
        cuda_transfer_fn=lambda: lhs.to_device().commutes_with(
            rhs.to_device(),
            max_commutation_matrix_entries=max_entries,
        ),
        cuda_resident_fn=lambda: lhs_device.commutes_with(
            rhs_device,
            max_commutation_matrix_entries=max_entries,
        ),
        build_info=build_info,
        warmup=warmup,
        repeat=repeat,
        notes=[
            "device-resident timing excludes operand transfers but includes host bool allocation and copy"
        ],
    )
    preallocated_timing = timed_preallocated_cuda_commutation(
        lhs_device=lhs_device,
        rhs_device=rhs_device,
        expected=expected,
        max_entries=max_entries,
        warmup=warmup,
        repeat=repeat,
    )
    add_preallocated_commutation_timing_fields(result, preallocated_timing)
    result["notes"].append(
        "preallocated timing reuses a caller-owned bool output buffer to isolate Python allocation overhead"
    )
    reused_device_output_timing = timed_reused_device_output_cuda_commutation(
        lhs_device=lhs_device,
        rhs_device=rhs_device,
        expected=expected,
        max_entries=max_entries,
        warmup=warmup,
        repeat=repeat,
    )
    add_reused_device_output_commutation_timing_fields(result, reused_device_output_timing)
    result["notes"].append(
        "reused-device-output timing is a private benchmark-only path gated by "
        "FASTPAULI_CUDA_BENCH_REUSE_COMMUTATION_DEVICE_OUTPUT"
    )
    device_output_allocate_timing = timed_public_device_output_allocate_cuda_commutation(
        lhs_device=lhs_device,
        rhs_device=rhs_device,
        expected=expected,
        max_entries=max_entries,
        warmup=warmup,
        repeat=repeat,
    )
    device_output_reuse_timing = timed_public_device_output_reuse_cuda_commutation(
        lhs_device=lhs_device,
        rhs_device=rhs_device,
        expected=expected,
        max_entries=max_entries,
        warmup=warmup,
        repeat=repeat,
    )
    device_output_to_host_timing = timed_public_device_output_to_host_cuda_commutation(
        lhs_device=lhs_device,
        rhs_device=rhs_device,
        expected=expected,
        max_entries=max_entries,
        warmup=warmup,
        repeat=repeat,
    )
    cuda_array_interface_timing = timed_public_device_output_cuda_array_interface_export(
        lhs_device=lhs_device,
        rhs_device=rhs_device,
        max_entries=max_entries,
        warmup=warmup,
        repeat=repeat,
    )
    add_public_device_output_commutation_timing_fields(
        result,
        allocate_timing=device_output_allocate_timing,
        reuse_timing=device_output_reuse_timing,
        to_host_timing=device_output_to_host_timing,
        cuda_array_interface_timing=cuda_array_interface_timing,
    )
    result["notes"].append(
        "public device-output timings keep dense uint8 commutation flags on CUDA until "
        "DeviceCommutationMatrix.to_host() is timed separately"
    )
    result["instrumentation"]["result_materialization_target"] = metadata["output_target"]
    fused_consumers_enabled = (
        metadata["output_target"] == "device_uint8_matrix_fused_consumer"
        or os.environ.get("FASTPAULI_CUDA_BENCH_CAMPAIGN7_FUSED_CONSUMERS", "") == "1"
    )
    if metadata["output_target"] == "device_uint8_matrix_consumer":
        consumer_timing = timed_public_device_output_consumer_cuda_commutation(
            lhs_device=lhs_device,
            rhs_device=rhs_device,
            expected=expected,
            max_entries=max_entries,
            warmup=warmup,
            repeat=repeat,
        )
        add_device_output_consumer_timing_fields(result, consumer_timing)
        cupy_timing = timed_cupy_device_output_consumer_cuda_commutation(
            lhs_device=lhs_device,
            rhs_device=rhs_device,
            expected=expected,
            max_entries=max_entries,
            warmup=warmup,
            repeat=repeat,
        )
        add_cupy_consumer_timing_fields(result, cupy_timing)
        result["notes"].append(
            "Campaign 6 consumer timings reduce DeviceCommutationMatrix on GPU and copy "
            "only compact uint64 count results unless the explicitly labeled CuPy dense "
            "consumer materializes the matrix."
        )
    if metadata["output_target"] == "campaign8_interop":
        cupy_timing = timed_cupy_device_output_consumer_cuda_commutation(
            lhs_device=lhs_device,
            rhs_device=rhs_device,
            expected=expected,
            max_entries=max_entries,
            warmup=warmup,
            repeat=repeat,
        )
        add_cupy_consumer_timing_fields(result, cupy_timing)
        result["notes"].append(
            "Campaign 8 interop keeps DLPack deferred and records the retained "
            "CUDA Array Interface framework consumer path when CuPy is available."
        )
    if fused_consumers_enabled:
        fused_timing = timed_private_fused_consumer_cuda_commutation(
            lhs_device=lhs_device,
            rhs_device=rhs_device,
            expected=expected,
            max_entries=max_entries,
            warmup=warmup,
            repeat=repeat,
        )
        add_private_fused_consumer_timing_fields(result, fused_timing)
        result["notes"].append(
            "Campaign 7 fused-consumer timings are benchmark-only private hooks "
            "over DeviceCommutationMatrix and are not user-facing API promises."
        )
    if metadata["output_target"] in CAMPAIGN8_MODES_BY_TARGET:
        campaign8_timing = timed_private_campaign8_device_resident_consumer_cuda_commutation(
            lhs_device=lhs_device,
            rhs_device=rhs_device,
            expected=expected,
            max_entries=max_entries,
            warmup=warmup,
            repeat=repeat,
        )
        add_campaign8_device_resident_consumer_timing_fields(
            result,
            campaign8_timing,
            mode=CAMPAIGN8_MODES_BY_TARGET[metadata["output_target"]],
            build_info=build_info,
            cuda_status=core._cuda_status(),
            git_revision=git_revision_full(),
        )
        result["notes"].append(
            "Campaign 8 rows use a private benchmark-only device-resident consumer "
            "hook and keep public CUDA APIs synchronous and unchanged."
        )
    if metadata["output_target"] in CAMPAIGN9_MODES_BY_TARGET:
        campaign9_mode = CAMPAIGN9_MODES_BY_TARGET[metadata["output_target"]]
        campaign8_timing = timed_private_campaign8_device_resident_consumer_cuda_commutation(
            lhs_device=lhs_device,
            rhs_device=rhs_device,
            expected=expected,
            max_entries=max_entries,
            warmup=warmup,
            repeat=repeat,
        )
        add_campaign8_device_resident_consumer_timing_fields(
            result,
            campaign8_timing,
            mode="device_resident_graph",
            build_info=build_info,
            cuda_status=core._cuda_status(),
            git_revision=git_revision_full(),
        )
        add_campaign9_row_schema_fields(
            result,
            mode=campaign9_mode,
            build_info=build_info,
            cuda_status=core._cuda_status(),
            git_revision=git_revision_full(),
        )
        result["instrumentation"]["campaign9_deferred_headroom"] = {
            "status": result["final_status"],
            "campaign8_headroom_item": result["campaign8_headroom_item"],
            "decision_doc": result["decision_doc"],
            "private_hook": "wolfgang._wolfgang_core._benchmark_cuda_device_resident_consumer",
            "non_deferred_closeout_required": True,
        }
        result["notes"].append(
            "Campaign 9 rows close Campaign 8 deferred headroom items and are invalid "
            "if final_status is deferred."
        )
        if metadata["output_target"] == "campaign9_public_grouping_api":
            conflict_timing = timed_public_conflict_degrees_cuda_commutation(
                lhs_device=lhs_device,
                rhs_device=rhs_device,
                expected=expected,
                max_entries=max_entries,
                warmup=warmup,
                repeat=repeat,
            )
            add_public_conflict_degrees_timing_fields(result, conflict_timing)
            result["notes"].append(
                "Campaign 9 accepts public DeviceCommutationMatrix.conflict_degrees "
                "as a compact summary API but keeps the true group_commuting_device "
                "surface rejected with evidence."
            )
        if metadata["output_target"] == "campaign9_dlpack_interop":
            cupy_timing = timed_cupy_device_output_consumer_cuda_commutation(
                lhs_device=lhs_device,
                rhs_device=rhs_device,
                expected=expected,
                max_entries=max_entries,
                warmup=warmup,
                repeat=repeat,
            )
            add_cupy_consumer_timing_fields(result, cupy_timing)
            dlpack_timing = timed_cupy_dlpack_consumer_cuda_commutation(
                lhs_device=lhs_device,
                rhs_device=rhs_device,
                expected=expected,
                max_entries=max_entries,
                warmup=warmup,
                repeat=repeat,
            )
            add_cupy_dlpack_consumer_timing_fields(result, dlpack_timing)
            result["notes"].append(
                "Campaign 9 DLPack rows compare the accepted read-only DLPack "
                "producer against the retained CUDA Array Interface CuPy consumer."
            )
    if metadata["output_target"] in CAMPAIGN10_MODES_BY_TARGET:
        campaign10_mode = CAMPAIGN10_MODES_BY_TARGET[metadata["output_target"]]
        campaign8_timing = timed_private_campaign8_device_resident_consumer_cuda_commutation(
            lhs_device=lhs_device,
            rhs_device=rhs_device,
            expected=expected,
            max_entries=max_entries,
            warmup=warmup,
            repeat=repeat,
        )
        add_campaign8_device_resident_consumer_timing_fields(
            result,
            campaign8_timing,
            mode="device_resident_graph",
            build_info=build_info,
            cuda_status=core._cuda_status(),
            git_revision=git_revision_full(),
        )
        dependency_unavailable = False
        dependency_reason = ""
        if metadata["output_target"] == "campaign10_dlpack_pytorch":
            cupy_timing = timed_cupy_device_output_consumer_cuda_commutation(
                lhs_device=lhs_device,
                rhs_device=rhs_device,
                expected=expected,
                max_entries=max_entries,
                warmup=warmup,
                repeat=repeat,
            )
            add_cupy_consumer_timing_fields(result, cupy_timing)
            cupy_dlpack_timing = timed_cupy_dlpack_consumer_cuda_commutation(
                lhs_device=lhs_device,
                rhs_device=rhs_device,
                expected=expected,
                max_entries=max_entries,
                warmup=warmup,
                repeat=repeat,
            )
            add_cupy_dlpack_consumer_timing_fields(result, cupy_dlpack_timing)
            torch_timing = timed_torch_dlpack_consumer_cuda_commutation(
                lhs_device=lhs_device,
                rhs_device=rhs_device,
                expected=expected,
                max_entries=max_entries,
                warmup=warmup,
                repeat=repeat,
            )
            add_torch_dlpack_consumer_timing_fields(result, torch_timing)
            dependency_unavailable = not bool(torch_timing["available"])
            dependency_reason = str(torch_timing["unavailable_reason"] or "")
        if metadata["output_target"] == "campaign10_public_grouping_api":
            conflict_timing = timed_public_conflict_degrees_cuda_commutation(
                lhs_device=lhs_device,
                rhs_device=rhs_device,
                expected=expected,
                max_entries=max_entries,
                warmup=warmup,
                repeat=repeat,
            )
            add_public_conflict_degrees_timing_fields(result, conflict_timing)
        add_campaign10_row_schema_fields(
            result,
            mode=campaign10_mode,
            build_info=build_info,
            cuda_status=core._cuda_status(),
            git_revision=git_revision_full(),
            unavailable_reason=dependency_reason,
            dependency_unavailable=dependency_unavailable,
        )
        result["instrumentation"]["campaign10_cross_architecture"] = {
            "status": result["final_status"],
            "campaign9_headroom_item": result["campaign9_headroom_item"],
            "decision_doc": result["decision_doc"],
            "private_hook": "wolfgang._wolfgang_core._benchmark_cuda_device_resident_consumer",
            "non_deferred_closeout_required": True,
        }
        result["notes"].append(
            "Campaign 10 rows close Campaign 9 remaining headroom and are invalid "
            "if final_status is deferred."
        )
    if metadata["output_target"] in {"caller_owned_device_bytes", "bitpacked_device_words"}:
        result["instrumentation"]["timing_boundary"] = "prototype"
        result["notes"].append(
            "device-output materialization remains a private Campaign 4 prototype label; "
            "public commutation methods still return host bool outputs"
        )
    result["scale"] = scale["scale"]
    return result


def run_matmul_scale(
    scale: dict[str, Any],
    build_info: dict[str, Any],
    rng: np.random.Generator,
    seed: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    terms = int(scale["terms"])
    lhs, lhs_metadata = generate_operator(
        rng=rng,
        num_qubits=int(scale["num_qubits"]),
        num_terms=terms,
        term_weight=int(scale["term_weight"]),
    )
    rhs, rhs_metadata = generate_operator(
        rng=rng,
        num_qubits=int(scale["num_qubits"]),
        num_terms=terms,
        term_weight=int(scale["term_weight"]),
    )
    max_intermediate_terms = max(terms * terms, 50_000_000)
    metadata = {
        "scale": scale["scale"],
        "lhs_terms": terms,
        "rhs_terms": terms,
        "intermediate_terms": terms * terms,
        "max_intermediate_terms": max_intermediate_terms,
        "num_qubits": lhs.num_qubits,
        "simplify": True,
        "term_weight_distribution": f"fixed term_weight={scale['term_weight']}",
        "lhs_duplicate_rate": lhs_metadata["duplicate_rate"],
        "rhs_duplicate_rate": rhs_metadata["duplicate_rate"],
        "coefficient_dtype": "complex128",
        "random_seed": seed,
        "operator_construction_method": "deterministic weighted lhs/rhs labels",
    }
    lhs_device = lhs.to_device()
    rhs_device = rhs.to_device()
    expected = lhs.matmul(rhs, simplify=True, max_intermediate_terms=max_intermediate_terms)
    metadata = {**metadata, "survivor_count": expected.num_terms}
    assert_same_operator(
        lhs_device.matmul(
            rhs_device,
            simplify=True,
            max_intermediate_terms=max_intermediate_terms,
        ).to_host(),
        expected,
    )
    result = case_result(
        name="matmul_product_generation_simplify",
        dataset=metadata,
        cpu_fn=lambda: lhs.matmul(
            rhs, simplify=True, max_intermediate_terms=max_intermediate_terms
        ),
        cuda_transfer_fn=lambda: (
            lhs.to_device()
            .matmul(
                rhs.to_device(),
                simplify=True,
                max_intermediate_terms=max_intermediate_terms,
            )
            .to_host()
        ),
        cuda_resident_fn=lambda: lhs_device.matmul(
            rhs_device,
            simplify=True,
            max_intermediate_terms=max_intermediate_terms,
        ),
        build_info=build_info,
        warmup=warmup,
        repeat=repeat,
    )
    result["scale"] = scale["scale"]
    return result


RUNNERS: dict[str, OperationRunner] = {
    "simplify_duplicate_pressure": run_simplify_scale,
    "statevector_expectation": run_expectation_scale,
    "pairwise_commutation": run_commutation_scale,
    "matmul_product_generation_simplify": run_matmul_scale,
}


def selected_operations(raw_operations: list[str]) -> list[str]:
    if not raw_operations:
        return list(RUNNERS)
    operations: list[str] = []
    for value in raw_operations:
        for item in value.split(","):
            operation = item.strip()
            if operation:
                operations.append(operation)
    unknown = [operation for operation in operations if operation not in RUNNERS]
    if unknown:
        raise SystemExit(f"unknown operation(s): {', '.join(sorted(unknown))}")
    return operations


def campaign8_cpu_unavailable_cases(
    *,
    profile: str,
    operations: list[str],
    build_info: dict[str, Any],
    cuda_status: dict[str, Any],
    git_revision: str,
) -> list[dict[str, Any]]:
    unavailable_reason = cuda_status.get("skip_reason") or (
        "Wolfgang CUDA runtime is unavailable for this Campaign 8 benchmark profile."
    )
    rows: list[dict[str, Any]] = []
    for operation in operations:
        for scale in SCALE_PROFILES[profile].get(operation, []):
            output_target = scale.get("output_target")
            mode = CAMPAIGN8_MODES_BY_TARGET.get(output_target)
            if mode is None:
                continue
            row: dict[str, Any] = {
                "name": operation,
                "scale": scale["scale"],
                "status": "unavailable",
                "dataset": dict(scale),
                "instrumentation": {
                    "campaign8_device_resident_consumer": {
                        "status": "unavailable",
                        "private_hook": (
                            "wolfgang._wolfgang_core._benchmark_cuda_device_resident_consumer"
                        ),
                        "mode": mode,
                    },
                    "result_materialization_target": output_target,
                },
                "results": {},
                "notes": [
                    "Campaign 8 unavailable row emitted without requiring CUDA import "
                    "or runtime availability."
                ],
            }
            add_campaign8_row_schema_fields(
                row,
                mode=mode,
                build_info=build_info,
                cuda_status=cuda_status,
                git_revision=git_revision,
                unavailable_reason=unavailable_reason,
                cuda_unavailable=True,
            )
            rows.append(row)
    return rows


def campaign9_cpu_unavailable_cases(
    *,
    profile: str,
    operations: list[str],
    build_info: dict[str, Any],
    cuda_status: dict[str, Any],
    git_revision: str,
) -> list[dict[str, Any]]:
    unavailable_reason = cuda_status.get("skip_reason") or (
        "Wolfgang CUDA runtime is unavailable for this Campaign 9 benchmark profile."
    )
    rows: list[dict[str, Any]] = []
    for operation in operations:
        for scale in SCALE_PROFILES[profile].get(operation, []):
            output_target = scale.get("output_target")
            mode = CAMPAIGN9_MODES_BY_TARGET.get(output_target)
            if mode is None:
                continue
            row: dict[str, Any] = {
                "name": operation,
                "scale": scale["scale"],
                "status": "unavailable",
                "dataset": dict(scale),
                "instrumentation": {
                    "campaign9_deferred_headroom": {
                        "status": "blocked_external",
                        "mode": mode,
                        "private_hook": (
                            "wolfgang._wolfgang_core._benchmark_cuda_device_resident_consumer"
                        ),
                        "non_deferred_closeout_required": True,
                    },
                    "result_materialization_target": output_target,
                },
                "results": {},
                "notes": [
                    "Campaign 9 unavailable row emitted without requiring CUDA import "
                    "or runtime availability."
                ],
            }
            add_campaign9_row_schema_fields(
                row,
                mode=mode,
                build_info=build_info,
                cuda_status=cuda_status,
                git_revision=git_revision,
                unavailable_reason=unavailable_reason,
                cuda_unavailable=True,
            )
            rows.append(row)
    return rows


def campaign10_cpu_unavailable_cases(
    *,
    profile: str,
    operations: list[str],
    build_info: dict[str, Any],
    cuda_status: dict[str, Any],
    git_revision: str,
) -> list[dict[str, Any]]:
    unavailable_reason = cuda_status.get("skip_reason") or (
        "Wolfgang CUDA runtime is unavailable for this Campaign 10 benchmark profile."
    )
    rows: list[dict[str, Any]] = []
    for operation in operations:
        for scale in SCALE_PROFILES[profile].get(operation, []):
            output_target = scale.get("output_target")
            mode = CAMPAIGN10_MODES_BY_TARGET.get(output_target)
            if mode is None:
                continue
            row: dict[str, Any] = {
                "name": operation,
                "scale": scale["scale"],
                "status": "unavailable",
                "dataset": dict(scale),
                "instrumentation": {
                    "campaign10_cross_architecture": {
                        "status": "blocked_external",
                        "mode": mode,
                        "private_hook": (
                            "wolfgang._wolfgang_core._benchmark_cuda_device_resident_consumer"
                        ),
                        "non_deferred_closeout_required": True,
                    },
                    "result_materialization_target": output_target,
                },
                "results": {},
                "notes": [
                    "Campaign 10 unavailable row emitted without requiring CUDA import "
                    "or runtime availability."
                ],
            }
            add_campaign10_row_schema_fields(
                row,
                mode=mode,
                build_info=build_info,
                cuda_status=cuda_status,
                git_revision=git_revision,
                unavailable_reason=unavailable_reason,
                cuda_unavailable=True,
            )
            rows.append(row)
    return rows


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")

    operations = selected_operations(args.operation)
    if (
        args.profile
        in {
            "campaign5_device_output",
            "campaign6_consumers",
            "campaign7_fused_consumers",
            "fused-graph-stress",
            "campaign8-device-graph",
            "campaign8-grouping-consumer",
            "campaign8-interop",
            "campaign8-stream-graph",
            "campaign8-scatter-ab",
            "campaign8-portability",
            "campaign9-privileged-ncu",
            "campaign9-non-h100-portability",
            "campaign9-public-grouping-api",
            "campaign9-dlpack-interop",
            "campaign9-stream-graph",
            "campaign9-csr-scatter-ab",
            "campaign10-portability",
            "campaign10-dlpack-pytorch",
            "campaign10-public-grouping-api",
            "campaign10-stream-graph-reprobe",
            "campaign10-csr-scatter-reprobe",
        }
        and not args.operation
    ):
        operations = ["pairwise_commutation"]
    build_info = core._build_info()
    cuda_status = core._cuda_status()
    full_revision = git_revision_full()
    report: dict[str, Any] = {
        "benchmark": "cuda_scaling",
        "scale_profile": args.profile,
        "git_commit": git_commit(),
        "command": command_string(),
        "environment": benchmark_environment(build_info, numpy_version=np.__version__),
        "fastpauli_version": wolfgang.__version__,
        "fastpauli_build_info": build_info,
        "cuda_status": cuda_status,
        "timing_policy": {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "summary": "median seconds",
            "cuda_transfer_inclusive": (
                "includes operator transfer and host result conversion where applicable"
            ),
            "cuda_device_resident": (
                "excludes initial operator transfer and final operator to_host conversion"
            ),
        },
        "correctness_checks": {
            "enabled": True,
            "reference": "Wolfgang scalar CPU output on the same deterministic datasets",
            "failure_mode": "raises RuntimeError if CPU/GPU outputs differ",
        },
        "campaign7": {
            "fused_graph_csr": {
                "status": "available when CUDA build exposes the private benchmark hook",
                "mode": "csr_anticommutation_graph",
            },
            "fused_conflict_degrees": {
                "status": "available when CUDA build exposes the private benchmark hook",
                "mode": "conflict_degrees",
            },
            "fused_grouping_summary": {
                "status": "available when CUDA build exposes the private benchmark hook",
                "mode": "grouping_summary",
            },
            "count_specialization_status": os.environ.get(
                "FASTPAULI_CUDA_CAMPAIGN7_COUNT_SPECIALIZATION_STATUS",
                CAMPAIGN7_COUNT_SPECIALIZATION_STATUS,
            ),
            "bitpacked_decision_status": os.environ.get(
                "FASTPAULI_CUDA_CAMPAIGN7_BITPACKED_DECISION_STATUS",
                CAMPAIGN7_BITPACKED_DECISION_STATUS,
            ),
            "portability_gpu": os.environ.get(
                "FASTPAULI_CUDA_PORTABILITY_GPU",
                "h100_sm90_primary_campaign7_host",
            ),
        },
        "campaign8": {
            "required_status_fields": list(CAMPAIGN8_REQUIRED_STATUS_FIELDS),
            "device_resident_graph_status": "retained",
            "public_grouping_api_status": "deferred",
            "dlpack_interop_status": "deferred",
            "non_h100_portability_status": os.environ.get(
                "FASTPAULI_CUDA_NON_H100_PORTABILITY_STATUS",
                "not_run",
            ),
            "stream_graph_status": "deferred",
            "scatter_tuning_status": "rejected_no_consumer",
            "private_hook": "wolfgang._wolfgang_core._benchmark_cuda_device_resident_consumer",
        },
        "campaign9": {
            "campaign": "cuda_deferred_headroom_campaign9",
            "allowed_final_statuses": list(CAMPAIGN9_FINAL_STATUSES),
            "required_row_fields": list(CAMPAIGN9_REQUIRED_ROW_FIELDS),
            "deferred_status_allowed": False,
            "mode_metadata": CAMPAIGN9_MODE_METADATA,
            "private_hook": "wolfgang._wolfgang_core._benchmark_cuda_device_resident_consumer",
        },
        "campaign10": {
            "campaign": "cuda_cross_architecture_campaign10",
            "allowed_final_statuses": list(CAMPAIGN10_FINAL_STATUSES),
            "required_row_fields": list(CAMPAIGN10_REQUIRED_ROW_FIELDS),
            "deferred_status_allowed": False,
            "mode_metadata": CAMPAIGN10_MODE_METADATA,
            "private_hook": "wolfgang._wolfgang_core._benchmark_cuda_device_resident_consumer",
        },
        "planned_cases": planned_cases(args.profile, operations),
        "cases": [],
    }

    if not cuda_status["built"] or not cuda_status["runtime_available"]:
        report["unavailable_reason"] = cuda_status["skip_reason"]
        if args.profile.startswith("campaign9-"):
            report["cases"] = campaign9_cpu_unavailable_cases(
                profile=args.profile,
                operations=operations,
                build_info=build_info,
                cuda_status=cuda_status,
                git_revision=full_revision,
            )
        elif args.profile.startswith("campaign10-"):
            report["cases"] = campaign10_cpu_unavailable_cases(
                profile=args.profile,
                operations=operations,
                build_info=build_info,
                cuda_status=cuda_status,
                git_revision=full_revision,
            )
        else:
            report["cases"] = campaign8_cpu_unavailable_cases(
                profile=args.profile,
                operations=operations,
                build_info=build_info,
                cuda_status=cuda_status,
                git_revision=full_revision,
            )
        return report

    profile_scales = SCALE_PROFILES[args.profile]
    for operation_index, operation in enumerate(operations):
        runner = RUNNERS[operation]
        for scale_index, scale in enumerate(profile_scales[operation]):
            scale_seed = _scale_seed(args.seed, operation_index, scale_index)
            rng = np.random.default_rng(scale_seed)
            report["cases"].append(
                runner(scale, build_info, rng, scale_seed, args.warmup, args.repeat)
            )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--profile",
        choices=tuple(SCALE_PROFILES),
        default="default",
        help="scaling profile to execute",
    )
    parser.add_argument(
        "--operation",
        action="append",
        default=[],
        help="operation name or comma-separated operation list; defaults to all operations",
    )
    parser.add_argument("--repeat", type=int, default=3, help="timed repetitions per scale")
    parser.add_argument("--warmup", type=int, default=1, help="untimed warmup repetitions")
    parser.add_argument("--seed", type=int, default=29051, help="deterministic RNG seed")
    parser.add_argument("--output", type=Path, help="optional path for the emitted JSON report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

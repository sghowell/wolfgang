#!/usr/bin/env python3
"""Deterministic Apple Metal benchmarks for the source-build backend."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import wolfgang_quantum as wolfgang
import wolfgang_quantum._wolfgang_core as core

try:
    from _benchmark_metadata import (
        benchmark_environment,
        benchmark_row_boundary,
        command_string,
        git_commit,
        git_provenance,
    )
except ModuleNotFoundError:
    from benchmarks._benchmark_metadata import (
        benchmark_environment,
        benchmark_row_boundary,
        command_string,
        git_commit,
        git_provenance,
    )


SMOKE_CASES = [
    {
        "name": "metal_smoke_pairwise",
        "profile": "smoke",
        "num_qubits": 16,
        "lhs_terms": 128,
        "rhs_terms": 128,
        "term_weight": 4,
        "random_seed": 19201,
    },
]
SCALING_CASES = [
    {
        "name": "metal_scaling_one_word_256x256",
        "profile": "scaling",
        "num_qubits": 32,
        "lhs_terms": 256,
        "rhs_terms": 256,
        "term_weight": 6,
        "random_seed": 19301,
    },
    {
        "name": "metal_scaling_rectangular_128x1024",
        "profile": "scaling",
        "num_qubits": 48,
        "lhs_terms": 128,
        "rhs_terms": 1024,
        "term_weight": 8,
        "random_seed": 19311,
    },
    {
        "name": "metal_scaling_multiword_192x192",
        "profile": "scaling",
        "num_qubits": 130,
        "lhs_terms": 192,
        "rhs_terms": 192,
        "term_weight": 10,
        "random_seed": 19321,
    },
    {
        "name": "metal_scaling_square_512x512",
        "profile": "scaling",
        "num_qubits": 64,
        "lhs_terms": 512,
        "rhs_terms": 512,
        "term_weight": 8,
        "random_seed": 19331,
    },
]
SPECIALIZATION_CASES = [
    {
        "name": "metal_specialization_words1_512x512",
        "profile": "specialization",
        "num_qubits": 64,
        "lhs_terms": 512,
        "rhs_terms": 512,
        "term_weight": 8,
        "random_seed": 19401,
    },
    {
        "name": "metal_specialization_words2_384x384",
        "profile": "specialization",
        "num_qubits": 96,
        "lhs_terms": 384,
        "rhs_terms": 384,
        "term_weight": 10,
        "random_seed": 19411,
    },
    {
        "name": "metal_specialization_generic_words3_192x192",
        "profile": "specialization",
        "num_qubits": 130,
        "lhs_terms": 192,
        "rhs_terms": 192,
        "term_weight": 10,
        "random_seed": 19421,
    },
]
CAMPAIGN3_CASES = [
    {
        "name": "metal_campaign3_words2_decision_384x384",
        "profile": "campaign3",
        "num_qubits": 96,
        "lhs_terms": 384,
        "rhs_terms": 384,
        "term_weight": 10,
        "random_seed": 19501,
    },
    {
        "name": "metal_campaign3_large_private_storage_1024x1024",
        "profile": "campaign3",
        "num_qubits": 64,
        "lhs_terms": 1024,
        "rhs_terms": 1024,
        "term_weight": 8,
        "random_seed": 19511,
    },
    {
        "name": "metal_campaign3_compact_reduction_512x512",
        "profile": "campaign3",
        "num_qubits": 64,
        "lhs_terms": 512,
        "rhs_terms": 512,
        "term_weight": 8,
        "random_seed": 19521,
    },
]
CAMPAIGN4_CASES = [
    {
        "name": "metal_campaign4_words2_large_768x768",
        "profile": "campaign4",
        "num_qubits": 96,
        "lhs_terms": 768,
        "rhs_terms": 768,
        "term_weight": 10,
        "random_seed": 19601,
    },
    {
        "name": "metal_campaign4_compact_large_2048x2048",
        "profile": "campaign4",
        "num_qubits": 64,
        "lhs_terms": 2048,
        "rhs_terms": 2048,
        "term_weight": 8,
        "random_seed": 19611,
    },
    {
        "name": "metal_campaign4_private_device_boundary_2048x2048",
        "profile": "campaign4",
        "num_qubits": 64,
        "lhs_terms": 2048,
        "rhs_terms": 2048,
        "term_weight": 8,
        "random_seed": 19621,
    },
]
CAMPAIGN5_CASES = [
    {
        "name": "metal_campaign5_simplify_words1_duplicate_heavy_8192_terms",
        "profile": "campaign5",
        "operation": "simplify",
        "num_qubits": 64,
        "num_terms": 8192,
        "term_weight": 8,
        "duplicate_rate": 0.85,
        "random_seed": 19701,
        "atol": 1.0e-12,
        "rtol": 0.0,
    },
    {
        "name": "metal_campaign5_simplify_words1_duplicate_light_8192_terms",
        "profile": "campaign5",
        "operation": "simplify",
        "num_qubits": 64,
        "num_terms": 8192,
        "term_weight": 8,
        "duplicate_rate": 0.05,
        "random_seed": 19711,
        "atol": 1.0e-12,
        "rtol": 0.0,
    },
    {
        "name": "metal_campaign5_simplify_words2_duplicate_heavy_4096_terms",
        "profile": "campaign5",
        "operation": "simplify",
        "num_qubits": 96,
        "num_terms": 4096,
        "term_weight": 10,
        "duplicate_rate": 0.70,
        "random_seed": 19721,
        "atol": 1.0e-12,
        "rtol": 0.0,
    },
    {
        "name": "metal_campaign5_simplify_generic_multiword_2048_terms",
        "profile": "campaign5",
        "operation": "simplify",
        "num_qubits": 130,
        "num_terms": 2048,
        "term_weight": 10,
        "duplicate_rate": 0.50,
        "random_seed": 19731,
        "atol": 1.0e-12,
        "rtol": 0.0,
    },
    {
        "name": "metal_campaign5_simplify_cancellation_4096_terms",
        "profile": "campaign5",
        "operation": "simplify",
        "num_qubits": 64,
        "num_terms": 4096,
        "term_weight": 8,
        "duplicate_rate": 0.75,
        "coefficient_mode": "cancellation_pairs",
        "random_seed": 19741,
        "atol": 1.0e-12,
        "rtol": 0.0,
    },
]
CAMPAIGN6_CASES = [
    {
        "name": "metal_campaign6_simplify_words1_duplicate_heavy_8192_terms",
        "profile": "campaign6",
        "operation": "simplify",
        "num_qubits": 64,
        "num_terms": 8192,
        "term_weight": 8,
        "duplicate_rate": 0.85,
        "random_seed": 19801,
        "atol": 1.0e-12,
        "rtol": 0.0,
    },
    {
        "name": "metal_campaign6_simplify_words1_duplicate_light_8192_terms",
        "profile": "campaign6",
        "operation": "simplify",
        "num_qubits": 64,
        "num_terms": 8192,
        "term_weight": 8,
        "duplicate_rate": 0.05,
        "random_seed": 19811,
        "atol": 1.0e-12,
        "rtol": 0.0,
    },
    {
        "name": "metal_campaign6_simplify_words2_duplicate_heavy_4096_terms",
        "profile": "campaign6",
        "operation": "simplify",
        "num_qubits": 96,
        "num_terms": 4096,
        "term_weight": 10,
        "duplicate_rate": 0.70,
        "random_seed": 19821,
        "atol": 1.0e-12,
        "rtol": 0.0,
    },
    {
        "name": "metal_campaign6_simplify_generic_multiword_2048_terms",
        "profile": "campaign6",
        "operation": "simplify",
        "num_qubits": 130,
        "num_terms": 2048,
        "term_weight": 10,
        "duplicate_rate": 0.50,
        "random_seed": 19831,
        "atol": 1.0e-12,
        "rtol": 0.0,
    },
    {
        "name": "metal_campaign6_simplify_cancellation_4096_terms",
        "profile": "campaign6",
        "operation": "simplify",
        "num_qubits": 64,
        "num_terms": 4096,
        "term_weight": 8,
        "duplicate_rate": 0.75,
        "coefficient_mode": "cancellation_pairs",
        "random_seed": 19841,
        "atol": 1.0e-12,
        "rtol": 0.0,
    },
]
CAMPAIGN7_CASES = [
    {
        "name": "metal_campaign7_simplify_words1_duplicate_heavy_8192_terms",
        "profile": "campaign7",
        "operation": "simplify",
        "num_qubits": 64,
        "num_terms": 8192,
        "term_weight": 8,
        "duplicate_rate": 0.85,
        "coefficient_mode": "fixed_dyadic",
        "random_seed": 19901,
        "atol": 1.0e-12,
        "rtol": 0.0,
    },
    {
        "name": "metal_campaign7_simplify_words1_duplicate_light_8192_terms",
        "profile": "campaign7",
        "operation": "simplify",
        "num_qubits": 64,
        "num_terms": 8192,
        "term_weight": 8,
        "duplicate_rate": 0.05,
        "coefficient_mode": "fixed_dyadic",
        "random_seed": 19911,
        "atol": 1.0e-12,
        "rtol": 0.0,
    },
    {
        "name": "metal_campaign7_simplify_words1_cancellation_4096_terms",
        "profile": "campaign7",
        "operation": "simplify",
        "num_qubits": 64,
        "num_terms": 4096,
        "term_weight": 8,
        "duplicate_rate": 0.75,
        "coefficient_mode": "cancellation_pairs",
        "random_seed": 19921,
        "atol": 1.0e-12,
        "rtol": 0.0,
    },
    {
        "name": "metal_campaign7_simplify_words2_unavailable_4096_terms",
        "profile": "campaign7",
        "operation": "simplify",
        "num_qubits": 96,
        "num_terms": 4096,
        "term_weight": 10,
        "duplicate_rate": 0.70,
        "coefficient_mode": "fixed_dyadic",
        "random_seed": 19931,
        "atol": 1.0e-12,
        "rtol": 0.0,
    },
]
CAMPAIGN8_CASES = [
    {
        "name": "metal_campaign8_simplify_words1_duplicate_heavy_8192_terms",
        "profile": "campaign8",
        "operation": "simplify",
        "num_qubits": 64,
        "num_terms": 8192,
        "term_weight": 8,
        "duplicate_rate": 0.85,
        "coefficient_mode": "fixed_dyadic",
        "random_seed": 20001,
        "atol": 1.0e-12,
        "rtol": 0.0,
    },
    {
        "name": "metal_campaign8_simplify_words1_duplicate_light_8192_terms",
        "profile": "campaign8",
        "operation": "simplify",
        "num_qubits": 64,
        "num_terms": 8192,
        "term_weight": 8,
        "duplicate_rate": 0.05,
        "coefficient_mode": "fixed_dyadic",
        "random_seed": 20011,
        "atol": 1.0e-12,
        "rtol": 0.0,
    },
    {
        "name": "metal_campaign8_simplify_words1_cancellation_4096_terms",
        "profile": "campaign8",
        "operation": "simplify",
        "num_qubits": 64,
        "num_terms": 4096,
        "term_weight": 8,
        "duplicate_rate": 0.75,
        "coefficient_mode": "cancellation_pairs",
        "random_seed": 20021,
        "atol": 1.0e-12,
        "rtol": 0.0,
    },
    {
        "name": "metal_campaign8_simplify_words1_large_duplicate_heavy_16384_terms",
        "profile": "campaign8",
        "operation": "simplify",
        "num_qubits": 64,
        "num_terms": 16384,
        "term_weight": 8,
        "duplicate_rate": 0.90,
        "coefficient_mode": "fixed_dyadic",
        "random_seed": 20031,
        "atol": 1.0e-12,
        "rtol": 0.0,
    },
    {
        "name": "metal_campaign8_simplify_words2_status_only_4096_terms",
        "profile": "campaign8",
        "operation": "simplify",
        "num_qubits": 96,
        "num_terms": 4096,
        "term_weight": 10,
        "duplicate_rate": 0.70,
        "coefficient_mode": "fixed_dyadic",
        "random_seed": 20041,
        "atol": 1.0e-12,
        "rtol": 0.0,
    },
]
PROFILE_CASES = {
    "smoke": SMOKE_CASES,
    "scaling": SCALING_CASES,
    "specialization": SPECIALIZATION_CASES,
    "campaign3": CAMPAIGN3_CASES,
    "campaign4": CAMPAIGN4_CASES,
    "campaign5": CAMPAIGN5_CASES,
    "campaign6": CAMPAIGN6_CASES,
    "campaign7": CAMPAIGN7_CASES,
    "campaign8": CAMPAIGN8_CASES,
}
CPU_BASELINE_SELECTORS = (
    ("cpu_default", "auto"),
    ("cpu_scalar", "scalar"),
    ("cpu_neon", "neon"),
)
METAL_COMMUTATION_THREADGROUP = [16, 16, 1]
METAL_FLAT_THREADGROUP = [256, 1, 1]
METAL_EXPERIMENTAL_KERNEL_ENV = "WOLFGANG_EXPERIMENTAL_METAL_COMMUTATION_KERNEL"
METAL_EXPERIMENTAL_LIBRARY_PATH_ENV = "WOLFGANG_EXPERIMENTAL_METAL_LIBRARY_PATH"
METAL_EXPERIMENTAL_OUTPUT_STORAGE_ENV = "WOLFGANG_EXPERIMENTAL_METAL_OUTPUT_STORAGE"
METAL_EXPERIMENTAL_COMPACT_CONSUMER_ENV = "WOLFGANG_EXPERIMENTAL_METAL_COMPACT_CONSUMER"
METAL_EXPERIMENTAL_SIMPLIFY_STRATEGY_ENV = "WOLFGANG_EXPERIMENTAL_METAL_SIMPLIFY_STRATEGY"
METAL_SIMPLIFY_WORKSPACE_TIMING_ENV = "WOLFGANG_METAL_BENCH_WORKSPACE_TIMING"


def selected_metal_commutation_kernel(
    packed_word_count: int,
    *,
    kernel_selector: str | None = None,
) -> str:
    if kernel_selector == "words1":
        if packed_word_count != 1:
            raise ValueError("words1 Metal kernel selector requires exactly one packed word")
        return "fp_pairwise_commutation_words1"
    if kernel_selector == "words2":
        if packed_word_count != 2:
            raise ValueError("words2 Metal kernel selector requires exactly two packed words")
        return "fp_pairwise_commutation_words2"
    if kernel_selector == "flat_generic":
        return "fp_pairwise_commutation_flat_generic"
    if kernel_selector == "generic_2d":
        return "fp_pairwise_commutation_generic"
    if packed_word_count == 1:
        return "fp_pairwise_commutation_words1"
    return "fp_pairwise_commutation_generic"


def metal_library_metadata() -> dict[str, str]:
    metallib_path = os.environ.get(METAL_EXPERIMENTAL_LIBRARY_PATH_ENV, "")
    if metallib_path:
        return {"library_source": "offline_metallib", "metallib_path": metallib_path}
    return {"library_source": "runtime_source"}


def metal_kernel_execution_metadata(
    case: dict[str, Any],
    *,
    transfer_boundary: str,
    storage_mode: str,
    buffer_allocation_or_reuse_boundary: str,
    kernel_selector: str | None = None,
) -> dict[str, Any]:
    if kernel_selector == "flat_generic":
        return {
            "buffer_allocation_or_reuse_boundary": buffer_allocation_or_reuse_boundary,
            "command_buffer_synchronization": "commit_and_waitUntilCompleted_per_operation",
            "dispatch_api": "dispatchThreads_1d",
            "grid_shape": [case["matrix_entries"], 1, 1],
            "kernel": selected_metal_commutation_kernel(
                int(case["packed_words"]),
                kernel_selector=kernel_selector,
            ),
            "kernel_selector": kernel_selector,
            "storage_mode": storage_mode,
            "threadgroup_size": METAL_FLAT_THREADGROUP,
            "transfer_boundary": transfer_boundary,
            **metal_library_metadata(),
        }
    return {
        "buffer_allocation_or_reuse_boundary": buffer_allocation_or_reuse_boundary,
        "command_buffer_synchronization": "commit_and_waitUntilCompleted_per_operation",
        "dispatch_api": "dispatchThreads_2d",
        "grid_shape": [case["rhs_terms"], case["lhs_terms"], 1],
        "kernel": selected_metal_commutation_kernel(
            int(case["packed_words"]),
            kernel_selector=kernel_selector,
        ),
        "kernel_selector": kernel_selector or "auto",
        "storage_mode": storage_mode,
        "threadgroup_size": METAL_COMMUTATION_THREADGROUP,
        "transfer_boundary": transfer_boundary,
        **metal_library_metadata(),
    }


def metal_reduction_execution_metadata(
    *,
    kernel: str,
    grid_entries: int,
    output_entries: int,
) -> dict[str, Any]:
    return {
        "buffer_allocation_or_reuse_boundary": "shared_count_output_allocation_per_call",
        "command_buffer_synchronization": "commit_and_waitUntilCompleted_per_operation",
        "dispatch_api": "dispatchThreads_1d",
        "grid_shape": [grid_entries, 1, 1],
        "kernel": kernel,
        "storage_mode": "shared_input_shared_count_output",
        "threadgroup_size": METAL_FLAT_THREADGROUP,
        "transfer_boundary": "compact_consumer_gpu_reduction",
        "output_entries": output_entries,
        **metal_library_metadata(),
    }


def metal_parallel_reduction_execution_metadata(
    *,
    kernel: str,
    input_entries: int,
    output_entries: int,
) -> dict[str, Any]:
    return {
        "buffer_allocation_or_reuse_boundary": "shared_partial_count_output_allocation_per_call",
        "command_buffer_synchronization": "commit_and_waitUntilCompleted_per_operation",
        "dispatch_api": "dispatchThreads_1d_threadgroup_block_reduction",
        "grid_shape": [output_entries * METAL_FLAT_THREADGROUP[0], 1, 1],
        "input_entries": input_entries,
        "kernel": kernel,
        "storage_mode": "shared_input_shared_partial_count_output",
        "threadgroup_size": METAL_FLAT_THREADGROUP,
        "transfer_boundary": "compact_consumer_gpu_parallel_block_reduction",
        "output_entries": output_entries,
        **metal_library_metadata(),
    }


def metal_no_kernel_execution_metadata(
    *,
    transfer_boundary: str,
    storage_mode: str,
    operation_boundary: str,
) -> dict[str, Any]:
    return {
        "buffer_allocation_or_reuse_boundary": operation_boundary,
        "command_buffer_synchronization": "not_applicable_no_command_buffer",
        "dispatch_api": "not_applicable",
        "grid_shape": "not_applicable",
        "kernel": "not_applicable_no_metal_kernel_dispatch",
        "storage_mode": storage_mode,
        "threadgroup_size": "not_applicable",
        "transfer_boundary": transfer_boundary,
    }


def metal_status_only_execution_metadata(
    *,
    kernel: str,
    operation_boundary: str,
    transfer_boundary: str = "status_only",
) -> dict[str, Any]:
    return {
        "buffer_allocation_or_reuse_boundary": operation_boundary,
        "command_buffer_synchronization": "not_applicable_no_command_buffer",
        "dispatch_api": "not_applicable",
        "grid_shape": "not_applicable",
        "kernel": kernel,
        "storage_mode": "not_applicable_candidate_not_executed",
        "threadgroup_size": "not_applicable",
        "transfer_boundary": transfer_boundary,
    }


def metal_simplify_device_candidate_execution_metadata(
    *,
    report: dict[str, Any] | None,
    transfer_boundary: str,
) -> dict[str, Any]:
    if report is None or report.get("status") != "ok":
        return metal_status_only_execution_metadata(
            kernel="not_applicable_simplify_candidate_not_executed",
            operation_boundary="private_metal_simplify_candidate_status_only",
            transfer_boundary=transfer_boundary,
        )

    stack = [
        "fp_simplify_words1_init_keys",
        "fp_simplify_words1_bitonic_sort_step",
        "fp_simplify_words1_mark_heads",
        "fp_simplify_prefix_sum_step",
        "fp_simplify_words1_reduce_by_key",
        "fp_simplify_words1_compact_survivors",
    ]
    metadata: dict[str, Any] = {
        "buffer_allocation_or_reuse_boundary": (
            "benchmark_only_private_simplify_scratch_and_device_output_allocation_per_call"
        ),
        "coefficient_domain": "signed_fixed32_dyadic_coefficients_only",
        "command_buffer_synchronization": "commit_and_waitUntilCompleted_per_operation",
        "dispatch_api": "dispatchThreads_1d_multi_kernel_stack",
        "kernel": "fp_simplify_words1_checked_primitive_stack",
        "kernel_stack": stack,
        "storage_mode": "shared_input_shared_scratch_shared_device_output",
        "threadgroup_size": METAL_FLAT_THREADGROUP,
        "transfer_boundary": transfer_boundary,
        **metal_library_metadata(),
    }
    if report is not None:
        metadata["bitonic_passes"] = report.get("bitonic_passes")
        metadata["prefix_sum_passes"] = report.get("prefix_sum_passes")
        metadata["padded_terms"] = report.get("padded_terms")
        metadata["workspace_reserved_bytes"] = report.get("workspace_reserved_bytes")
        if "campaign8_timing_schema" in report:
            metadata["timing_decomposition_source"] = "private_hook_internal_steady_clock"
        if "dispatch_counts" in report:
            metadata["dispatch_counts"] = report.get("dispatch_counts")
        if "pipeline_cache" in report:
            metadata["pipeline_cache"] = report.get("pipeline_cache")
    return metadata


def timing_summary(timings: list[float]) -> dict[str, float]:
    if not timings:
        raise ValueError("at least one timing is required")
    ordered = sorted(timings)
    return {
        "median": statistics.median(timings),
        "min": ordered[0],
        "max": ordered[-1],
    }


def packed_words(num_qubits: int) -> int:
    return (num_qubits + 63) // 64


def case_with_metadata(case: dict[str, Any], *, repeat: int) -> dict[str, Any]:
    metadata = dict(case)
    metadata["packed_words"] = packed_words(case["num_qubits"])
    if "lhs_terms" in case and "rhs_terms" in case:
        metadata["matrix_entries"] = case["lhs_terms"] * case["rhs_terms"]
    metadata["repeat"] = repeat
    return metadata


def list_profiles() -> dict[str, Any]:
    return {
        "benchmark": "apple_metal_kernels",
        "profiles": {
            profile: [case_with_metadata(case, repeat=0) for case in cases]
            for profile, cases in PROFILE_CASES.items()
        },
    }


def timed_call(
    fn: Callable[[], Any], *, repeat: int, warmup: int = 1
) -> tuple[Any, dict[str, float]]:
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
def forced_environment_value(name: str, value: str):
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


@contextmanager
def metal_commutation_selector(selector: str):
    if selector == "auto":
        previous = os.environ.get(METAL_EXPERIMENTAL_KERNEL_ENV)
        os.environ.pop(METAL_EXPERIMENTAL_KERNEL_ENV, None)
        try:
            yield
        finally:
            if previous is not None:
                os.environ[METAL_EXPERIMENTAL_KERNEL_ENV] = previous
        return

    with forced_environment_value(METAL_EXPERIMENTAL_KERNEL_ENV, selector):
        yield


def compile_offline_metallib() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    source = root / "src" / "metal" / "kernels" / "commutation.metal"
    work_dir = Path(tempfile.mkdtemp(prefix="fastpauli-metal-optimization-"))
    air = work_dir / "commutation.air"
    metallib = work_dir / "commutation.metallib"
    commands = [
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(source), "-o", str(air)],
        ["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(metallib)],
    ]
    for command in commands:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            return {
                "status": "skipped",
                "path": None,
                "command": " ".join(command),
                "skip_reason": (
                    completed.stderr or completed.stdout or "xcrun metal failed"
                ).strip(),
            }
    return {
        "status": "ok",
        "path": str(metallib),
        "command": " && ".join(" ".join(command) for command in commands),
    }


def external_baseline_statuses() -> dict[str, dict[str, str]]:
    return {
        "mpsgraph": {
            "status": "skipped",
            "semantic_mapping": (
                "No exact sparse Pauli pairwise-commutation primitive is exposed through MPSGraph; "
                "a dense boolean graph would measure a different materialized problem."
            ),
            "timing_boundary": "not_timed_no_exact_sparse_mapping",
        },
        "pytorch_mps": {
            "status": "skipped",
            "semantic_mapping": (
                "PyTorch MPS does not provide an exact sparse Pauli packed-word commutation path "
                "or raw Metal-buffer interop boundary matching Wolfgang device-resident rows."
            ),
            "timing_boundary": "not_timed_no_exact_sparse_mapping",
        },
    }


def poison_device_commutation_matrix(output: Any, expected: np.ndarray) -> None:
    poison = np.ascontiguousarray(np.logical_not(expected), dtype=np.bool_)
    core._copy_device_commutation_matrix_from_host_for_testing(output, poison)


def partial_count_entries(matrix_entries: int) -> int:
    threads_per_group = METAL_FLAT_THREADGROUP[0]
    return (matrix_entries + threads_per_group - 1) // threads_per_group


def dense_label(num_qubits: int, positions: dict[int, str]) -> str:
    chars = ["I"] * num_qubits
    for qubit, pauli in positions.items():
        chars[num_qubits - 1 - qubit] = pauli
    return "".join(chars)


def random_pauli_sum(
    *,
    num_qubits: int,
    num_terms: int,
    term_weight: int,
    seed: int,
) -> wolfgang.PauliSum:
    rng = np.random.default_rng(seed)
    labels: list[str] = []
    coeffs: list[complex] = []
    paulis = np.array(["X", "Y", "Z"])
    for _ in range(num_terms):
        weight = min(term_weight, num_qubits)
        qubits = rng.choice(num_qubits, size=weight, replace=False)
        local = rng.choice(paulis, size=weight, replace=True)
        labels.append(dense_label(num_qubits, dict(zip(qubits.tolist(), local.tolist()))))
        coeffs.append(complex(float(rng.normal()), float(rng.normal())))
    return wolfgang.PauliSum.from_labels(labels, coeffs)


def random_simplify_pauli_sum(case: dict[str, Any]) -> wolfgang.PauliSum:
    rng = np.random.default_rng(int(case["random_seed"]))
    num_terms = int(case["num_terms"])
    duplicate_rate = float(case["duplicate_rate"])
    pool_size = max(1, min(num_terms, round(num_terms * (1.0 - duplicate_rate))))
    weight = min(int(case["term_weight"]), int(case["num_qubits"]))
    pool = [
        dense_label(
            int(case["num_qubits"]),
            dict(
                zip(
                    rng.choice(
                        int(case["num_qubits"]),
                        size=weight,
                        replace=False,
                    ).tolist(),
                    rng.choice(np.array(["X", "Y", "Z"]), size=weight).tolist(),
                )
            ),
        )
        for _ in range(pool_size)
    ]
    labels: list[str] = []
    coeffs: list[complex] = []
    if case.get("coefficient_mode") == "cancellation_pairs":
        for index in range(num_terms):
            pool_index = index % pool_size
            label = pool[pool_index]
            base = complex(
                1.0 + float(pool_index % 17) * 0.125,
                -0.5 + float(pool_index % 11) * 0.0625,
            )
            labels.append(label)
            coeffs.append(base if (index // pool_size) % 2 == 0 else -base)
        return wolfgang.PauliSum.from_labels(labels, coeffs)

    if case.get("coefficient_mode") == "fixed_dyadic":
        for index in range(num_terms):
            pool_index = index % pool_size
            labels.append(pool[pool_index])
            real = float((index % 17) - 8) / 8.0
            imag = float((index % 11) - 5) / 16.0
            coeffs.append(complex(real, imag))
        return wolfgang.PauliSum.from_labels(labels, coeffs)

    for index in range(num_terms):
        labels.append(pool[index % pool_size])
        coeffs.append(complex(float(rng.normal()), float(rng.normal())))
    return wolfgang.PauliSum.from_labels(labels, coeffs)


def labels_and_coeffs(op: wolfgang.PauliSum) -> tuple[list[str], list[complex]]:
    labels, coeffs = op.to_labels()
    return list(labels), [complex(value) for value in coeffs]


def assert_same_operator(lhs: wolfgang.PauliSum, rhs: wolfgang.PauliSum) -> None:
    lhs_labels, lhs_coeffs = labels_and_coeffs(lhs)
    rhs_labels, rhs_coeffs = labels_and_coeffs(rhs)
    if lhs.num_qubits != rhs.num_qubits or lhs_labels != rhs_labels:
        raise AssertionError(
            "PauliSum mismatch: "
            f"lhs(num_qubits={lhs.num_qubits}, labels={lhs_labels[:8]}) "
            f"rhs(num_qubits={rhs.num_qubits}, labels={rhs_labels[:8]})"
        )
    np.testing.assert_allclose(lhs_coeffs, rhs_coeffs, rtol=1.0e-12, atol=1.0e-12)


def cpu_selector_status(build_info: dict[str, Any], selector: str) -> tuple[bool, str]:
    if selector == "auto":
        return True, ""
    candidates = {
        str(candidate.get("name")): str(candidate.get("status"))
        for candidate in build_info.get("cpu_backend_candidates", [])
    }
    if candidates.get(selector) == "available":
        return True, ""
    unavailable = build_info.get("unavailable_cpu_backends", {})
    reason = str(unavailable.get(selector, candidates.get(selector, "not_compiled")))
    return False, reason


def cpu_selector_status_for_case(
    build_info: dict[str, Any],
    selector: str,
    case: dict[str, Any],
) -> tuple[bool, str]:
    available, reason = cpu_selector_status(build_info, selector)
    if not available:
        return available, reason
    if selector == "neon" and int(case.get("packed_words", 0)) > 2:
        return False, "neon_commutation_supports_one_or_two_packed_words"
    return True, ""


def append_cpu_baseline_rows(
    rows: list[dict[str, Any]],
    *,
    lhs: wolfgang.PauliSum,
    rhs: wolfgang.PauliSum,
    expected: np.ndarray,
    case: dict[str, Any],
    repeat: int,
    build_info: dict[str, Any],
) -> None:
    for label, selector in CPU_BASELINE_SELECTORS:
        available, unavailable_reason = cpu_selector_status_for_case(build_info, selector, case)
        if not available:
            rows.append(
                {
                    "case": case,
                    "operation": "commutes_with",
                    "variant": label,
                    "cpu_backend_selector": selector,
                    "status": "skipped",
                    "skip_reason": unavailable_reason,
                    "timing": None,
                    "correct": None,
                    **benchmark_row_boundary(
                        build_info=build_info,
                        object_backend="cpu",
                        transfer_boundary="host_materialized",
                    ),
                }
            )
            continue

        with forced_cpu_backend(selector):
            result, timing = timed_call(lambda: lhs.commutes_with(rhs), repeat=repeat)
            active_info = core._build_info()
        np.testing.assert_array_equal(result, expected)
        rows.append(
            {
                "case": case,
                "operation": "commutes_with",
                "variant": label,
                "cpu_backend_selector": selector,
                "active_cpu_backend": active_info["active_cpu_backend"],
                "status": "ok",
                "timing": timing,
                "correct": True,
                **benchmark_row_boundary(
                    build_info=active_info,
                    object_backend="cpu",
                    transfer_boundary="host_materialized",
                ),
            }
        )


def simplify_row_fields(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "atol": case["atol"],
        "duplicate_rate": case["duplicate_rate"],
        "num_terms": case["num_terms"],
        "output_terms": case["output_terms"],
        "rtol": case["rtol"],
    }


def simplify_workspace_timing_mode() -> str:
    mode = os.environ.get(METAL_SIMPLIFY_WORKSPACE_TIMING_ENV, "absent")
    if mode in {"absent", "grow_inside_timing", "pre_reserved_outside_timing"}:
        return mode
    return "absent"


def simplify_workspace_model(case: dict[str, Any]) -> dict[str, Any]:
    terms = int(case["num_terms"])
    words = int(case["packed_words"])
    key_bytes = terms * words * 2 * 8
    coeff_bytes = terms * 16
    index_bytes = terms * 8
    flag_bytes = terms * 4
    prefix_bytes = terms * 4
    survivor_output_bytes = key_bytes + coeff_bytes
    scratch_bytes = key_bytes + coeff_bytes + index_bytes + flag_bytes + prefix_bytes
    reservation = scratch_bytes + survivor_output_bytes
    alignment = 256
    aligned_reservation = ((reservation + alignment - 1) // alignment) * alignment
    return {
        "status": "retained_private_model",
        "workspace_timing_mode": simplify_workspace_timing_mode(),
        "alignment_bytes": alignment,
        "input_key_bytes": key_bytes,
        "input_coeff_bytes": coeff_bytes,
        "index_bytes": index_bytes,
        "flag_bytes": flag_bytes,
        "prefix_bytes": prefix_bytes,
        "survivor_output_bytes": survivor_output_bytes,
        "reserved_bytes_estimate": aligned_reservation,
        "candidate_primitives_required": [
            "Metal sort",
            "Metal prefix-sum",
            "Metal reduce-by-key",
        ],
    }


def append_simplify_workspace_probe_row(
    rows: list[dict[str, Any]],
    *,
    case: dict[str, Any],
    build_info: dict[str, Any],
) -> None:
    rows.append(
        {
            "case": case,
            "operation": "simplify",
            "variant": "metal_simplify_workspace_probe",
            "status": "skipped",
            "skip_reason": (
                "device-resident simplify candidate remains blocked until checked "
                "Metal sort/prefix/reduce primitives exist"
            ),
            "timing": None,
            "correct": None,
            "metal_simplify_strategy": "device_candidate",
            "metal_simplify_strategy_status": "rejected_with_evidence",
            "metal_simplify_strategy_reason": (
                "Metal sort/prefix/reduce primitives are not retained in Wolfgang yet; "
                "Campaign 6 retains only the private workspace model and status row"
            ),
            "metal_simplify_workspace_model": simplify_workspace_model(case),
            **simplify_row_fields(case),
            "metal_execution": metal_status_only_execution_metadata(
                kernel="not_applicable_device_resident_simplify_candidate_not_retained",
                operation_boundary="private_metal_workspace_model_status_only",
            ),
            **benchmark_row_boundary(
                build_info=build_info,
                object_backend="metal",
                transfer_boundary="status_only",
            ),
        }
    )


def append_simplify_device_candidate_row(
    rows: list[dict[str, Any]],
    *,
    device_op: Any,
    expected: wolfgang.PauliSum,
    case: dict[str, Any],
    repeat: int,
    build_info: dict[str, Any],
) -> None:
    if not hasattr(core, "_metal_simplify_words1_candidate_for_testing"):
        rows.append(
            {
                "case": case,
                "operation": "simplify",
                "variant": "metal_simplify_device_candidate",
                "status": "unavailable",
                "skip_reason": "Metal source build does not expose the Campaign 7 candidate hook",
                "timing": None,
                "correct": None,
                "metal_simplify_strategy": "device_candidate",
                "metal_simplify_strategy_status": "unavailable",
                "metal_simplify_strategy_reason": "private benchmark-only hook unavailable",
                **simplify_row_fields(case),
                "metal_execution": metal_simplify_device_candidate_execution_metadata(
                    report=None,
                    transfer_boundary="status_only",
                ),
                **benchmark_row_boundary(
                    build_info=build_info,
                    object_backend="metal",
                    transfer_boundary="status_only",
                ),
            }
        )
        return

    check_report = core._metal_simplify_words1_candidate_for_testing(
        device_op,
        atol=float(case["atol"]),
        rtol=float(case["rtol"]),
        include_output=True,
    )
    if check_report.get("status") != "ok":
        reason = str(check_report.get("skip_reason", "Metal simplify candidate unavailable"))
        rows.append(
            {
                "case": case,
                "operation": "simplify",
                "variant": "metal_simplify_device_candidate",
                "status": check_report.get("status", "unavailable"),
                "skip_reason": reason,
                "timing": None,
                "correct": None,
                "metal_simplify_strategy": "device_candidate",
                "metal_simplify_strategy_status": check_report.get(
                    "metal_simplify_strategy_status",
                    "unavailable",
                ),
                "metal_simplify_strategy_reason": reason,
                **simplify_row_fields(case),
                "metal_execution": metal_simplify_device_candidate_execution_metadata(
                    report=check_report,
                    transfer_boundary="status_only",
                ),
                **benchmark_row_boundary(
                    build_info=core._build_info(),
                    object_backend="metal",
                    transfer_boundary="status_only",
                ),
            }
        )
        return

    assert_same_operator(check_report["device_output"].to_host(), expected)
    timed_report, timing = timed_call(
        lambda: core._metal_simplify_words1_candidate_for_testing(
            device_op,
            atol=float(case["atol"]),
            rtol=float(case["rtol"]),
            include_output=False,
        ),
        repeat=repeat,
    )
    if timed_report.get("status") != "ok":
        raise AssertionError(f"Metal simplify candidate became unavailable: {timed_report}")

    primitive_stack = dict(check_report["primitive_stack"])
    timing_report = timed_report if isinstance(timed_report, dict) else check_report
    rows.append(
        {
            "case": case,
            "operation": "simplify",
            "variant": "metal_simplify_device_candidate",
            "status": "ok",
            "timing": timing,
            "correct": True,
            "metal_simplify_strategy": "device_candidate",
            "metal_simplify_strategy_status": "benchmark_only",
            "metal_simplify_strategy_reason": (
                f"{'Campaign 8 timing-decomposed' if case.get('profile') == 'campaign8' else 'Campaign 7 checked'} "
                "one-word Metal sort/prefix/reduce stack; benchmark-only and "
                "limited to signed fixed32 dyadic coefficients inside the "
                "exact uint64 squared-magnitude comparison domain"
            ),
            "metal_simplify_primitive_stack": primitive_stack,
            "metal_simplify_coefficient_domain": "signed_fixed32_dyadic_coefficients_only",
            "padded_terms": int(check_report["padded_terms"]),
            "bitonic_passes": int(check_report["bitonic_passes"]),
            "prefix_sum_passes": int(check_report["prefix_sum_passes"]),
            "workspace_reserved_bytes": int(check_report["workspace_reserved_bytes"]),
            **(
                {
                    "campaign8_timing_schema": timing_report.get("campaign8_timing_schema"),
                    "timing_decomposition_seconds": timing_report.get(
                        "timing_decomposition_seconds"
                    ),
                    "dispatch_counts": timing_report.get("dispatch_counts"),
                    "pipeline_cache": timing_report.get("pipeline_cache"),
                    "performance_decision": timing_report.get("performance_decision"),
                }
                if case.get("profile") == "campaign8"
                else {}
            ),
            **simplify_row_fields(case),
            "metal_execution": metal_simplify_device_candidate_execution_metadata(
                report=timing_report,
                transfer_boundary="device_resident",
            ),
            **benchmark_row_boundary(
                build_info=core._build_info(),
                object_backend="metal",
                transfer_boundary="device_resident",
            ),
        }
    )


def apply_campaign8_simplify_decision(rows: list[dict[str, Any]], case: dict[str, Any]) -> None:
    if case.get("profile") != "campaign8":
        return
    candidate = next(
        (
            row
            for row in rows
            if row.get("variant") == "metal_simplify_device_candidate" and row.get("status") == "ok"
        ),
        None,
    )
    if candidate is None:
        return

    def median_for(variant: str) -> float | None:
        for row in rows:
            if row.get("variant") != variant:
                continue
            timing = row.get("timing")
            if isinstance(timing, dict) and isinstance(timing.get("median"), (int, float)):
                return float(timing["median"])
        return None

    candidate_median = median_for("metal_simplify_device_candidate")
    cpu_default_median = median_for("cpu_default")
    transfer_median = median_for("metal_simplify_transfer_reference")
    is_high_value_workload = (
        float(case.get("duplicate_rate", 0.0)) >= 0.75
        or case.get("coefficient_mode") == "cancellation_pairs"
    )
    if (
        is_high_value_workload
        and candidate_median is not None
        and cpu_default_median is not None
        and transfer_median is not None
        and candidate_median < cpu_default_median
        and candidate_median < transfer_median
    ):
        decision = {
            "candidate_status": "performance_relevant",
            "reason": (
                "Campaign 8 candidate beats same-host CPU default and Metal "
                "transfer-reference on this checked duplicate-heavy or "
                "cancellation workload, but remains private until a broader "
                "public lifetime and correctness design is accepted."
            ),
            "cpu_default_median_seconds": cpu_default_median,
            "transfer_reference_median_seconds": transfer_median,
            "candidate_median_seconds": candidate_median,
        }
    else:
        decision = {
            "candidate_status": "experimental",
            "reason": (
                "Campaign 8 candidate does not beat same-host CPU default and "
                "Metal transfer-reference on this checked workload; promotion "
                "requires lower-pass sort or reusable output/lifetime work."
            ),
            "cpu_default_median_seconds": cpu_default_median,
            "transfer_reference_median_seconds": transfer_median,
            "candidate_median_seconds": candidate_median,
        }
    candidate["performance_decision"] = decision


def append_simplify_cpu_rows(
    rows: list[dict[str, Any]],
    *,
    op: wolfgang.PauliSum,
    expected: wolfgang.PauliSum,
    case: dict[str, Any],
    repeat: int,
    build_info: dict[str, Any],
) -> None:
    for label, selector in CPU_BASELINE_SELECTORS:
        available, unavailable_reason = cpu_selector_status(build_info, selector)
        if available and selector not in {"auto", "scalar"}:
            available = False
            unavailable_reason = (
                f"WOLFGANG_CPU_BACKEND={selector} does not execute scalar-only simplify"
            )
        if not available:
            rows.append(
                {
                    "case": case,
                    "operation": "simplify",
                    "variant": label,
                    "cpu_backend_selector": selector,
                    "status": "skipped",
                    "skip_reason": unavailable_reason,
                    "timing": None,
                    "correct": None,
                    "metal_simplify_strategy": "cpu_reference",
                    "metal_simplify_strategy_status": "unavailable",
                    "metal_simplify_strategy_reason": unavailable_reason,
                    **simplify_row_fields(case),
                    **benchmark_row_boundary(
                        build_info=build_info,
                        object_backend="cpu",
                        transfer_boundary="host_materialized",
                    ),
                }
            )
            continue

        with forced_cpu_backend(selector):
            result, timing = timed_call(
                lambda: op.simplify(atol=float(case["atol"]), rtol=float(case["rtol"])),
                repeat=repeat,
            )
            active_info = core._build_info()
        assert_same_operator(result, expected)
        rows.append(
            {
                "case": case,
                "operation": "simplify",
                "variant": label,
                "cpu_backend_selector": selector,
                "active_cpu_backend": active_info["active_cpu_backend"],
                "status": "ok",
                "timing": timing,
                "correct": True,
                "metal_simplify_strategy": "cpu_reference",
                "metal_simplify_strategy_status": "retained",
                "metal_simplify_strategy_reason": (
                    "host PauliSum.simplify baseline for Campaign 5 transfer-reference comparison"
                ),
                **simplify_row_fields(case),
                **benchmark_row_boundary(
                    build_info=active_info,
                    object_backend="cpu",
                    transfer_boundary="host_materialized",
                ),
            }
        )


def run_simplify_case(
    case: dict[str, Any],
    *,
    repeat: int,
    include_metal: bool,
    build_info: dict[str, Any],
) -> list[dict[str, Any]]:
    case_metadata = case_with_metadata(case, repeat=repeat)
    op = random_simplify_pauli_sum(case)
    expected = op.simplify(atol=float(case["atol"]), rtol=float(case["rtol"]))
    case_metadata["output_terms"] = expected.num_terms

    rows: list[dict[str, Any]] = []
    append_simplify_cpu_rows(
        rows,
        op=op,
        expected=expected,
        case=case_metadata,
        repeat=repeat,
        build_info=build_info,
    )
    if not include_metal:
        return rows

    device_op = op.to_device(backend="metal")
    storage_mode = str(core._metal_status().get("storage_mode", "unknown"))
    checked = device_op.simplify(atol=float(case["atol"]), rtol=float(case["rtol"]))
    assert_same_operator(checked.to_host(), expected)
    simplified_device, timing = timed_call(
        lambda: device_op.simplify(atol=float(case["atol"]), rtol=float(case["rtol"])),
        repeat=repeat,
    )
    if simplified_device.backend != "metal":
        raise AssertionError(f"Metal simplify returned backend={simplified_device.backend!r}")
    assert_same_operator(simplified_device.to_host(), expected)
    rows.append(
        {
            "case": case_metadata,
            "operation": "simplify",
            "variant": "metal_simplify_transfer_reference",
            "status": "ok",
            "timing": timing,
            "correct": True,
            "metal_simplify_strategy": "transfer_reference",
            "metal_simplify_strategy_status": "retained",
            "metal_simplify_strategy_reason": (
                "DevicePauliSum.to_host -> CPU PauliSum.simplify -> DevicePauliSum.from_host"
            ),
            **simplify_row_fields(case_metadata),
            "metal_execution": metal_no_kernel_execution_metadata(
                transfer_boundary="device_to_host_cpu_simplify_host_to_device",
                storage_mode=storage_mode,
                operation_boundary=(
                    "shared_buffer_host_materialization_cpu_simplify_shared_buffer_allocation"
                ),
            ),
            **benchmark_row_boundary(
                build_info=core._build_info(),
                object_backend="metal",
                transfer_boundary="device_to_host_cpu_simplify_host_to_device",
            ),
        }
    )
    if case_metadata["profile"] == "campaign6":
        append_simplify_workspace_probe_row(
            rows,
            case=case_metadata,
            build_info=core._build_info(),
        )
    if case_metadata["profile"] in {"campaign7", "campaign8"}:
        append_simplify_device_candidate_row(
            rows,
            device_op=device_op,
            expected=expected,
            case=case_metadata,
            repeat=repeat,
            build_info=core._build_info(),
        )
    apply_campaign8_simplify_decision(rows, case_metadata)
    return rows


def run_case(
    case: dict[str, Any],
    *,
    repeat: int,
    include_metal: bool,
    build_info: dict[str, Any],
    metallib_path: str | None = None,
) -> list[dict[str, Any]]:
    if case.get("operation") == "simplify":
        return run_simplify_case(
            case,
            repeat=repeat,
            include_metal=include_metal,
            build_info=build_info,
        )

    case_metadata = case_with_metadata(case, repeat=repeat)
    lhs = random_pauli_sum(
        num_qubits=case["num_qubits"],
        num_terms=case["lhs_terms"],
        term_weight=case["term_weight"],
        seed=case["random_seed"],
    )
    rhs = random_pauli_sum(
        num_qubits=case["num_qubits"],
        num_terms=case["rhs_terms"],
        term_weight=case["term_weight"],
        seed=case["random_seed"] + 1,
    )
    expected = np.asarray(lhs.commutes_with(rhs), dtype=np.bool_)

    rows: list[dict[str, Any]] = []
    append_cpu_baseline_rows(
        rows,
        lhs=lhs,
        rhs=rhs,
        expected=expected,
        case=case_metadata,
        repeat=repeat,
        build_info=build_info,
    )
    if not include_metal:
        return rows

    lhs_device = lhs.to_device(backend="metal")
    rhs_device = rhs.to_device(backend="metal")
    storage_mode = str(core._metal_status().get("storage_mode", "unknown"))

    transfer_result, transfer_timing = timed_call(
        lambda: lhs.to_device(backend="metal").commutes_with(rhs.to_device(backend="metal")),
        repeat=repeat,
    )
    np.testing.assert_array_equal(transfer_result, expected)
    rows.append(
        {
            "case": case_metadata,
            "operation": "commutes_with",
            "variant": "metal_transfer_inclusive",
            "status": "ok",
            "timing": transfer_timing,
            "correct": True,
            "metal_execution": metal_kernel_execution_metadata(
                case_metadata,
                transfer_boundary="transfer_inclusive",
                storage_mode=storage_mode,
                buffer_allocation_or_reuse_boundary=(
                    "host_to_shared_buffer_allocation_and_device_output_allocation_per_call"
                ),
            ),
            **benchmark_row_boundary(
                build_info=core._build_info(),
                object_backend="metal",
                transfer_boundary="transfer_inclusive",
            ),
        }
    )

    resident_result, resident_timing = timed_call(
        lambda: lhs_device.commutes_with(rhs_device),
        repeat=repeat,
    )
    np.testing.assert_array_equal(resident_result, expected)
    rows.append(
        {
            "case": case_metadata,
            "operation": "commutes_with",
            "variant": "metal_device_resident",
            "status": "ok",
            "timing": resident_timing,
            "correct": True,
            "metal_execution": metal_kernel_execution_metadata(
                case_metadata,
                transfer_boundary="device_resident",
                storage_mode=storage_mode,
                buffer_allocation_or_reuse_boundary="device_output_allocation_per_call",
            ),
            **benchmark_row_boundary(
                build_info=core._build_info(),
                object_backend="metal",
                transfer_boundary="device_resident",
            ),
        }
    )

    if case_metadata["profile"] in {"campaign3", "campaign4"} and (
        "private_storage" in case_metadata["name"]
        or "private_device_boundary" in case_metadata["name"]
    ):
        with forced_environment_value(METAL_EXPERIMENTAL_OUTPUT_STORAGE_ENV, "private"):
            private_result, private_timing = timed_call(
                lambda: lhs_device.commutes_with(rhs_device),
                repeat=repeat,
            )
            private_execution = metal_kernel_execution_metadata(
                case_metadata,
                transfer_boundary="device_resident_private_output_blit_to_shared_staging",
                storage_mode="private_output_plus_shared_staging",
                buffer_allocation_or_reuse_boundary=(
                    "private_device_output_allocation_and_shared_staging_allocation_per_call"
                ),
            )
        np.testing.assert_array_equal(private_result, expected)
        rows.append(
            {
                "case": case_metadata,
                "operation": "commutes_with",
                "variant": "metal_private_blit_host_output",
                "status": "ok",
                "timing": private_timing,
                "correct": True,
                "metal_execution": private_execution,
                **benchmark_row_boundary(
                    build_info=core._build_info(),
                    object_backend="metal",
                    transfer_boundary="device_resident_private_output_blit_to_shared_staging",
                ),
            }
        )

    matrix, matrix_timing = timed_call(
        lambda: lhs_device.commutes_with_device(rhs_device),
        repeat=repeat,
    )
    np.testing.assert_array_equal(matrix.to_host(), expected)
    rows.append(
        {
            "case": case_metadata,
            "operation": "commutes_with_device",
            "variant": "metal_device_matrix",
            "status": "ok",
            "timing": matrix_timing,
            "correct": True,
            "metal_execution": metal_kernel_execution_metadata(
                case_metadata,
                transfer_boundary="device_output_allocating",
                storage_mode=storage_mode,
                buffer_allocation_or_reuse_boundary="device_matrix_output_allocation_per_call",
            ),
            **benchmark_row_boundary(
                build_info=core._build_info(),
                object_backend="metal",
                transfer_boundary="device_output_allocating",
            ),
        }
    )

    reused_output = wolfgang.DeviceCommutationMatrix.empty(expected.shape, backend="metal")
    reused_matrix, reused_timing = timed_call(
        lambda: lhs_device.commutes_with_device(rhs_device, output=reused_output),
        repeat=repeat,
    )
    np.testing.assert_array_equal(reused_matrix.to_host(), expected)
    rows.append(
        {
            "case": case_metadata,
            "operation": "commutes_with_device",
            "variant": "metal_device_matrix_reuse",
            "status": "ok",
            "timing": reused_timing,
            "correct": True,
            "metal_execution": metal_kernel_execution_metadata(
                case_metadata,
                transfer_boundary="device_output_reused",
                storage_mode=storage_mode,
                buffer_allocation_or_reuse_boundary="caller_provided_device_matrix_output_reused",
            ),
            **benchmark_row_boundary(
                build_info=core._build_info(),
                object_backend="metal",
                transfer_boundary="device_output_reused",
            ),
        }
    )

    if case_metadata["profile"] in {"specialization", "campaign3", "campaign4"}:
        ab_candidates = [
            ("auto", "metal_device_matrix_reuse_auto_ab"),
            ("generic_2d", "metal_device_matrix_reuse_generic2d_baseline"),
            ("flat_generic", "metal_device_matrix_reuse_flat_generic_baseline"),
        ]
        if case_metadata["packed_words"] == 1:
            ab_candidates.append(("words1", "metal_device_matrix_reuse_words1_candidate"))
        if case_metadata["packed_words"] == 2:
            ab_candidates.append(("words2", "metal_device_matrix_reuse_words2_candidate"))

        baseline_timings: dict[str, list[float]] = {
            variant_label: [] for _, variant_label in ab_candidates
        }
        for selector, variant_label in ab_candidates:
            poison_device_commutation_matrix(reused_output, expected)
            with metal_commutation_selector(selector):
                candidate_output = lhs_device.commutes_with_device(
                    rhs_device,
                    output=reused_output,
                )
            np.testing.assert_array_equal(candidate_output.to_host(), expected)
        for iteration in range(repeat):
            rotated_candidates = (
                ab_candidates[iteration % len(ab_candidates) :]
                + ab_candidates[: iteration % len(ab_candidates)]
            )
            for selector, variant_label in rotated_candidates:
                with metal_commutation_selector(selector):
                    start = time.perf_counter()
                    lhs_device.commutes_with_device(
                        rhs_device,
                        output=reused_output,
                    )
                    baseline_timings[variant_label].append(time.perf_counter() - start)

        for selector, variant_label in ab_candidates:
            poison_device_commutation_matrix(reused_output, expected)
            with metal_commutation_selector(selector):
                checked_output = lhs_device.commutes_with_device(
                    rhs_device,
                    output=reused_output,
                )
            np.testing.assert_array_equal(checked_output.to_host(), expected)
            kernel_selector = None if selector == "auto" else selector
            rows.append(
                {
                    "case": case_metadata,
                    "operation": "commutes_with_device",
                    "variant": variant_label,
                    "status": "ok",
                    "timing": timing_summary(baseline_timings[variant_label]),
                    "correct": True,
                    "metal_execution": metal_kernel_execution_metadata(
                        case_metadata,
                        transfer_boundary="device_output_reused",
                        storage_mode=storage_mode,
                        buffer_allocation_or_reuse_boundary=(
                            "caller_provided_device_matrix_output_reused"
                        ),
                        kernel_selector=kernel_selector,
                    ),
                    **benchmark_row_boundary(
                        build_info=core._build_info(),
                        object_backend="metal",
                        transfer_boundary="device_output_reused",
                    ),
                }
            )

        if case_metadata["profile"] in {"campaign3", "campaign4"} and metallib_path:
            poison_device_commutation_matrix(reused_output, expected)
            with forced_environment_value(METAL_EXPERIMENTAL_LIBRARY_PATH_ENV, metallib_path):
                with metal_commutation_selector("auto"):
                    metallib_output, metallib_timing = timed_call(
                        lambda: lhs_device.commutes_with_device(
                            rhs_device,
                            output=reused_output,
                        ),
                        repeat=repeat,
                    )
                    metallib_execution = metal_kernel_execution_metadata(
                        case_metadata,
                        transfer_boundary="device_output_reused",
                        storage_mode=storage_mode,
                        buffer_allocation_or_reuse_boundary=(
                            "caller_provided_device_matrix_output_reused"
                        ),
                    )
            np.testing.assert_array_equal(metallib_output.to_host(), expected)
            rows.append(
                {
                    "case": case_metadata,
                    "operation": "commutes_with_device",
                    "variant": "metal_device_matrix_reuse_metallib_auto",
                    "status": "ok",
                    "timing": metallib_timing,
                    "correct": True,
                    "metal_execution": metallib_execution,
                    **benchmark_row_boundary(
                        build_info=core._build_info(),
                        object_backend="metal",
                        transfer_boundary="device_output_reused",
                    ),
                }
            )

    host_matrix, to_host_timing = timed_call(
        lambda: matrix.to_host(),
        repeat=repeat,
    )
    np.testing.assert_array_equal(host_matrix, expected)
    rows.append(
        {
            "case": case_metadata,
            "operation": "to_host",
            "variant": "metal_device_matrix_to_host",
            "status": "ok",
            "timing": to_host_timing,
            "correct": True,
            "metal_execution": metal_no_kernel_execution_metadata(
                transfer_boundary="device_output_to_host",
                storage_mode=storage_mode,
                operation_boundary="shared_buffer_host_materialization_from_existing_device_matrix",
            ),
            **benchmark_row_boundary(
                build_info=core._build_info(),
                object_backend="metal",
                transfer_boundary="device_output_to_host",
            ),
        }
    )

    count_result, count_timing = timed_call(
        lambda: matrix.count_commuting(),
        repeat=repeat,
    )
    assert count_result == int(expected.sum())
    rows.append(
        {
            "case": case_metadata,
            "operation": "count_commuting",
            "variant": "metal_compact_consumer",
            "status": "ok",
            "timing": count_timing,
            "correct": True,
            "metal_execution": metal_no_kernel_execution_metadata(
                transfer_boundary="compact_consumer",
                storage_mode=storage_mode,
                operation_boundary="cpu_scan_over_shared_metal_storage",
            ),
            **benchmark_row_boundary(
                build_info=core._build_info(),
                object_backend="metal",
                transfer_boundary="compact_consumer",
            ),
        }
    )
    count_axis0, count_axis0_timing = timed_call(
        lambda: matrix.count_commuting(axis=0),
        repeat=repeat,
    )
    np.testing.assert_array_equal(count_axis0, expected.sum(axis=0))
    rows.append(
        {
            "case": case_metadata,
            "operation": "count_commuting",
            "variant": "metal_compact_count_axis0",
            "status": "ok",
            "timing": count_axis0_timing,
            "correct": True,
            "metal_execution": metal_no_kernel_execution_metadata(
                transfer_boundary="compact_consumer",
                storage_mode=storage_mode,
                operation_boundary="cpu_axis0_scan_over_shared_metal_storage",
            ),
            **benchmark_row_boundary(
                build_info=core._build_info(),
                object_backend="metal",
                transfer_boundary="compact_consumer",
            ),
        }
    )

    count_axis1, count_axis1_timing = timed_call(
        lambda: matrix.count_commuting(axis=1),
        repeat=repeat,
    )
    np.testing.assert_array_equal(count_axis1, expected.sum(axis=1))
    rows.append(
        {
            "case": case_metadata,
            "operation": "count_commuting",
            "variant": "metal_compact_count_axis1",
            "status": "ok",
            "timing": count_axis1_timing,
            "correct": True,
            "metal_execution": metal_no_kernel_execution_metadata(
                transfer_boundary="compact_consumer",
                storage_mode=storage_mode,
                operation_boundary="cpu_axis1_scan_over_shared_metal_storage",
            ),
            **benchmark_row_boundary(
                build_info=core._build_info(),
                object_backend="metal",
                transfer_boundary="compact_consumer",
            ),
        }
    )
    if case_metadata["profile"] in {"campaign3", "campaign4"} and (
        "compact_reduction" in case_metadata["name"] or "compact_large" in case_metadata["name"]
    ):
        with forced_environment_value(METAL_EXPERIMENTAL_COMPACT_CONSUMER_ENV, "gpu"):
            gpu_count_result, gpu_count_timing = timed_call(
                lambda: matrix.count_commuting(),
                repeat=repeat,
            )
        assert gpu_count_result == int(expected.sum())
        rows.append(
            {
                "case": case_metadata,
                "operation": "count_commuting",
                "variant": "metal_compact_consumer_gpu_total",
                "status": "ok",
                "timing": gpu_count_timing,
                "correct": True,
                "metal_execution": metal_reduction_execution_metadata(
                    kernel="fp_count_commuting_total_atomic",
                    grid_entries=case_metadata["matrix_entries"],
                    output_entries=1,
                ),
                **benchmark_row_boundary(
                    build_info=core._build_info(),
                    object_backend="metal",
                    transfer_boundary="compact_consumer_gpu_reduction",
                ),
            }
        )

        if case_metadata["profile"] == "campaign4":
            with forced_environment_value(
                METAL_EXPERIMENTAL_COMPACT_CONSUMER_ENV,
                "gpu_parallel_total",
            ):
                gpu_parallel_count_result, gpu_parallel_count_timing = timed_call(
                    lambda: matrix.count_commuting(),
                    repeat=repeat,
                )
            assert gpu_parallel_count_result == int(expected.sum())
            partials = partial_count_entries(case_metadata["matrix_entries"])
            rows.append(
                {
                    "case": case_metadata,
                    "operation": "count_commuting",
                    "variant": "metal_compact_consumer_gpu_parallel_total",
                    "status": "ok",
                    "timing": gpu_parallel_count_timing,
                    "correct": True,
                    "metal_execution": metal_parallel_reduction_execution_metadata(
                        kernel="fp_count_commuting_total_block_sums",
                        input_entries=case_metadata["matrix_entries"],
                        output_entries=partials,
                    ),
                    **benchmark_row_boundary(
                        build_info=core._build_info(),
                        object_backend="metal",
                        transfer_boundary="compact_consumer_gpu_parallel_block_reduction",
                    ),
                }
            )

        with forced_environment_value(METAL_EXPERIMENTAL_COMPACT_CONSUMER_ENV, "gpu"):
            gpu_axis0, gpu_axis0_timing = timed_call(
                lambda: matrix.count_commuting(axis=0),
                repeat=repeat,
            )
        np.testing.assert_array_equal(gpu_axis0, expected.sum(axis=0))
        rows.append(
            {
                "case": case_metadata,
                "operation": "count_commuting",
                "variant": "metal_compact_count_axis0_gpu",
                "status": "ok",
                "timing": gpu_axis0_timing,
                "correct": True,
                "metal_execution": metal_reduction_execution_metadata(
                    kernel="fp_count_commuting_cols",
                    grid_entries=case_metadata["rhs_terms"],
                    output_entries=case_metadata["rhs_terms"],
                ),
                **benchmark_row_boundary(
                    build_info=core._build_info(),
                    object_backend="metal",
                    transfer_boundary="compact_consumer_gpu_reduction",
                ),
            }
        )

        with forced_environment_value(METAL_EXPERIMENTAL_COMPACT_CONSUMER_ENV, "gpu"):
            gpu_axis1, gpu_axis1_timing = timed_call(
                lambda: matrix.count_commuting(axis=1),
                repeat=repeat,
            )
        np.testing.assert_array_equal(gpu_axis1, expected.sum(axis=1))
        rows.append(
            {
                "case": case_metadata,
                "operation": "count_commuting",
                "variant": "metal_compact_count_axis1_gpu",
                "status": "ok",
                "timing": gpu_axis1_timing,
                "correct": True,
                "metal_execution": metal_reduction_execution_metadata(
                    kernel="fp_count_commuting_rows",
                    grid_entries=case_metadata["lhs_terms"],
                    output_entries=case_metadata["lhs_terms"],
                ),
                **benchmark_row_boundary(
                    build_info=core._build_info(),
                    object_backend="metal",
                    transfer_boundary="compact_consumer_gpu_reduction",
                ),
            }
        )
    return rows


def build_report(*, repeat: int, profile: str) -> dict[str, Any]:
    build_info = core._build_info()
    metal_status = core._metal_status()
    provenance = git_provenance()
    report: dict[str, Any] = {
        "benchmark": "apple_metal_kernels",
        "profile": profile,
        "git_commit": git_commit(),
        "git_provenance": provenance,
        "command": command_string(),
        "environment": benchmark_environment(build_info, numpy_version=np.__version__),
        "build_info": build_info,
        "metal_status": metal_status,
        "cases": [],
        "limitations": [
            "Metal support is source-build-only.",
            "Default compact consumers are CPU scans over shared Metal storage.",
            "Timings are local Apple Silicon evidence, not generic wheel claims.",
        ],
    }
    metallib: dict[str, Any] | None = None
    if profile in {"campaign3", "campaign4"} and bool(metal_status["runtime_available"]):
        metallib = compile_offline_metallib()
        report["offline_metallib"] = metallib
        report["external_baselines"] = external_baseline_statuses()
    cases = PROFILE_CASES[profile]
    for case in cases:
        report["cases"].extend(
            run_case(
                case,
                repeat=repeat,
                include_metal=bool(metal_status["runtime_available"]),
                build_info=build_info,
                metallib_path=(
                    str(metallib.get("path"))
                    if metallib is not None and metallib.get("status") == "ok"
                    else None
                ),
            )
        )

    if not metal_status["runtime_available"]:
        report["status"] = "skipped"
        report["skip_reason"] = metal_status["skip_reason"]
        return report

    report["status"] = "ok"
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3, help="Timed repetitions per case.")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_CASES),
        default="smoke",
        help="Benchmark profile to run. Use smoke for validation.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Shorthand for --profile smoke, retained for validation compatibility.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Emit deterministic case metadata without running benchmarks.",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON report.")
    parser.add_argument("--output", type=Path, help="Optional path for the emitted JSON report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_cases:
        report = list_profiles()
    else:
        profile = "smoke" if args.smoke else args.profile
        report = build_report(repeat=args.repeat, profile=profile)
    payload = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.json:
        print(payload)
    else:
        print(payload)


if __name__ == "__main__":
    main()

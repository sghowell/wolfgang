#!/usr/bin/env python3
"""Deterministic Phase 11 CUDA kernel benchmark smoke.

The benchmark is availability-aware: CPU-only builds emit CUDA status and no
timed cases, while CUDA builds compare the same deterministic datasets across
CPU scalar/default paths, transfer-inclusive CUDA calls, and device-resident
CUDA calls where the public CUDA API supports them.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import wolfgang_quantum as wolfgang
import wolfgang_quantum._wolfgang_core as core
from wolfgang_quantum import PauliSum

try:
    from _benchmark_metadata import benchmark_environment, command_string, git_commit
except ModuleNotFoundError:
    from benchmarks._benchmark_metadata import (
        benchmark_environment,
        command_string,
        git_commit,
    )


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


def _cuda_compute_capability() -> tuple[int, int] | None:
    status = core._cuda_status()
    devices = status.get("devices")
    if not isinstance(devices, list) or not devices:
        return None
    first = devices[0]
    if not isinstance(first, dict):
        return None
    capability = first.get("compute_capability")
    if (
        isinstance(capability, (list, tuple))
        and len(capability) == 2
        and all(isinstance(value, int) for value in capability)
    ):
        return int(capability[0]), int(capability[1])
    return None


def _cupy_compile_error_indicates_unsupported_cuda_architecture(
    exc: BaseException,
    *,
    compute_capability: tuple[int, int] | None,
) -> bool:
    message = str(exc)
    if "gpu-architecture" not in message and "NVRTC_ERROR_INVALID_OPTION" not in message:
        return False
    return compute_capability is not None


def _require_supported_cupy_runtime_for_current_cuda_architecture(cupy: Any) -> None:
    compute_capability = _cuda_compute_capability()
    try:
        probe = cupy.asarray([0], dtype=cupy.uint8)
        reduced = cupy.sum(probe)
        if hasattr(reduced, "get"):
            reduced.get()
    except Exception as exc:
        if _cupy_compile_error_indicates_unsupported_cuda_architecture(
            exc,
            compute_capability=compute_capability,
        ):
            assert compute_capability is not None
            import pytest

            pytest.skip(
                "CuPy runtime does not support CUDA compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}: {type(exc).__name__}: {exc}"
            )
        raise


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
def temporary_env_var(name: str, value: str):
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def campaign4_workspace_mode() -> str:
    value = os.environ.get("WOLFGANG_CUDA_BENCH_WORKSPACE_MODE", "absent")
    if value in {"", "absent"}:
        return "absent"
    if value in {"grow_inside_timing", "pre_reserved_outside_timing"}:
        return value
    raise ValueError(
        "WOLFGANG_CUDA_BENCH_WORKSPACE_MODE must be absent, grow_inside_timing, "
        "or pre_reserved_outside_timing"
    )


def campaign4_duplicate_reduction_strategy() -> str:
    value = os.environ.get("WOLFGANG_CUDA_BENCH_DUPLICATE_REDUCTION", "thrust_default")
    if value in {"", "thrust_default"}:
        return "thrust_default"
    if value in {"cub_radix_sort_reduce", "cub_radix_sort_run_length"}:
        return value
    raise ValueError(
        "WOLFGANG_CUDA_BENCH_DUPLICATE_REDUCTION must be thrust_default, "
        "cub_radix_sort_reduce, or cub_radix_sort_run_length"
    )


def campaign4_commutation_output_target() -> str:
    value = os.environ.get("WOLFGANG_CUDA_BENCH_COMMUTATION_OUTPUT", "host_vector")
    if value in {
        "",
        "host_vector",
        "caller_owned_host_bytes",
        "caller_owned_device_bytes",
        "bitpacked_device_words",
    }:
        return "host_vector" if value == "" else value
    raise ValueError(
        "WOLFGANG_CUDA_BENCH_COMMUTATION_OUTPUT must be host_vector, "
        "caller_owned_host_bytes, caller_owned_device_bytes, or bitpacked_device_words"
    )


def campaign4_cub_strategy_for_case(name: str, dataset: dict[str, Any] | None = None) -> str:
    if name not in {"simplify_duplicate_pressure", "matmul_product_generation_simplify"}:
        return "none"
    strategy = campaign4_duplicate_reduction_strategy()
    # The current Campaign 4 prototype is deliberately narrow: it uses a CUB
    # radix-sort boundary only for the packed one-word, <=32-qubit key layout.
    # Wider-key cases still execute the production path and must not be labeled
    # as CUB timings in reports.
    if isinstance(dataset, dict):
        num_qubits = dataset.get("num_qubits")
        if num_qubits is not None and (int(num_qubits) + 63) // 64 != 1:
            return "none"
        if num_qubits is not None and int(num_qubits) > 32:
            return "none"
    if strategy == "cub_radix_sort_reduce":
        return "device_radix_sort_reduce"
    if strategy == "cub_radix_sort_run_length":
        return "none"
    return "none"


def campaign4_materialization_target_for_case(name: str) -> str:
    if name == "pairwise_commutation":
        return campaign4_commutation_output_target()
    if name in {"simplify_duplicate_pressure", "matmul_product_generation_simplify"}:
        return "none"
    if name == "statevector_expectation":
        return "none"
    return "none"


def campaign4_workspace_fields_for_case(
    name: str, dataset: dict[str, Any] | None
) -> dict[str, Any]:
    temporary_storage = (
        _temporary_storage_estimate(name, dataset)
        if dataset is not None
        else {"available": False, "estimated_bytes": 0}
    )
    scratch_bytes = int(temporary_storage.get("estimated_bytes") or 0)
    workspace_mode = campaign4_workspace_mode()
    cub_strategy = campaign4_cub_strategy_for_case(name, dataset)
    strategy = campaign4_duplicate_reduction_strategy()
    workspace_active = workspace_mode != "absent" and cub_strategy != "none"
    unavailable_reason = None
    if strategy == "cub_radix_sort_run_length" and name in {
        "simplify_duplicate_pressure",
        "matmul_product_generation_simplify",
    }:
        unavailable_reason = (
            "CUB run-length duplicate reduction was not implemented; row uses production fallback"
        )
    elif (
        strategy != "thrust_default"
        and cub_strategy == "none"
        and name in {"simplify_duplicate_pressure", "matmul_product_generation_simplify"}
    ):
        unavailable_reason = "prototype supports only one-word <=32-qubit packed-key simplify"
    return {
        "workspace_mode": workspace_mode,
        "workspace_reserved_bytes": scratch_bytes
        if workspace_mode == "pre_reserved_outside_timing" and workspace_active
        else 0,
        "workspace_high_watermark_bytes": scratch_bytes if workspace_active else 0,
        "workspace_allocation_count": 1 if workspace_active else 0,
        "workspace_growth_count": 1 if workspace_active else 0,
        "cub_strategy": cub_strategy,
        "cub_strategy_unavailable_reason": unavailable_reason,
        "scratch_bytes_requested": scratch_bytes,
        "result_materialization_target": campaign4_materialization_target_for_case(name),
        "timing_boundary": "device_resident",
    }


def timed_cpu_backend_call(
    fn: Callable[[], Any],
    *,
    backend: str,
    warmup: int,
    repeat: int,
) -> tuple[Any, dict[str, float]]:
    with forced_cpu_backend(backend):
        return timed_call(fn, warmup=warmup, repeat=repeat)


OPTIMIZED_CPU_SELECTOR_ORDER = ("tbb", "avx512", "avx2", "neon", "sve")
CAMPAIGN7_COUNT_SPECIALIZATION_STATUS = "rejected_not_dominant"
CAMPAIGN7_BITPACKED_DECISION_STATUS = "deferred_no_dense_capacity_or_bandwidth_trigger"
CAMPAIGN8_REQUIRED_STATUS_FIELDS = (
    "device_resident_graph_status",
    "public_grouping_api_status",
    "dlpack_interop_status",
    "non_h100_portability_status",
    "stream_graph_status",
    "scatter_tuning_status",
)
CAMPAIGN8_DEFAULT_STATUSES = {
    "device_resident_graph_status": "not_applicable",
    "public_grouping_api_status": "deferred",
    "dlpack_interop_status": "deferred",
    "non_h100_portability_status": "not_run",
    "stream_graph_status": "deferred",
    "scatter_tuning_status": "rejected_no_consumer",
}
CAMPAIGN8_MODE_BOUNDARIES = {
    "device_resident_graph": ("compact_host_copy", "device_resident_consumer"),
    "device_grouping_consumer": ("private_benchmark_only", "compact_materialization"),
    "dlpack_consumer": ("framework_consumer", "device_resident_consumer"),
    "stream_graph_probe": ("private_benchmark_only", "kernel_only"),
    "csr_scatter_ab": ("private_benchmark_only", "kernel_only"),
    "portability_check": ("private_benchmark_only", "device_resident_consumer"),
}


def campaign8_statuses_for_mode(mode: str, *, cuda_unavailable: bool = False) -> dict[str, str]:
    statuses = dict(CAMPAIGN8_DEFAULT_STATUSES)
    if mode == "device_resident_graph":
        statuses["device_resident_graph_status"] = "unavailable" if cuda_unavailable else "retained"
        statuses["public_grouping_api_status"] = "not_applicable"
        statuses["dlpack_interop_status"] = "not_applicable"
        statuses["stream_graph_status"] = "not_applicable"
    elif mode == "device_grouping_consumer":
        statuses["device_resident_graph_status"] = "unavailable" if cuda_unavailable else "retained"
        statuses["public_grouping_api_status"] = "deferred"
        statuses["dlpack_interop_status"] = "not_applicable"
        statuses["stream_graph_status"] = "not_applicable"
        statuses["scatter_tuning_status"] = "not_applicable"
    elif mode == "dlpack_consumer":
        statuses["device_resident_graph_status"] = "unavailable" if cuda_unavailable else "retained"
        statuses["dlpack_interop_status"] = "unavailable" if cuda_unavailable else "deferred"
        statuses["public_grouping_api_status"] = "not_applicable"
        statuses["stream_graph_status"] = "not_applicable"
        statuses["scatter_tuning_status"] = "not_applicable"
    elif mode == "stream_graph_probe":
        statuses["device_resident_graph_status"] = "not_applicable"
        statuses["public_grouping_api_status"] = "not_applicable"
        statuses["dlpack_interop_status"] = "not_applicable"
        statuses["stream_graph_status"] = "deferred"
        statuses["scatter_tuning_status"] = "not_applicable"
    elif mode == "csr_scatter_ab":
        statuses["device_resident_graph_status"] = "unavailable" if cuda_unavailable else "retained"
        statuses["public_grouping_api_status"] = "not_applicable"
        statuses["dlpack_interop_status"] = "not_applicable"
        statuses["stream_graph_status"] = "not_applicable"
        statuses["scatter_tuning_status"] = "rejected_no_consumer"
    elif mode == "portability_check":
        statuses["device_resident_graph_status"] = "unavailable" if cuda_unavailable else "retained"
        statuses["public_grouping_api_status"] = "not_applicable"
        statuses["dlpack_interop_status"] = "not_applicable"
        statuses["stream_graph_status"] = "not_applicable"
        statuses["scatter_tuning_status"] = "not_applicable"
    return statuses


def _campaign8_gpu_metadata(cuda_status: dict[str, Any]) -> tuple[str, str]:
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


def add_campaign8_row_schema_fields(
    result: dict[str, Any],
    *,
    mode: str,
    build_info: dict[str, Any],
    cuda_status: dict[str, Any],
    git_revision: str,
    unavailable_reason: str = "",
    cuda_unavailable: bool = False,
) -> None:
    boundary, timing_boundary = CAMPAIGN8_MODE_BOUNDARIES[mode]
    gpu_name, gpu_compute_capability = _campaign8_gpu_metadata(cuda_status)
    result["campaign"] = "h100_campaign8"
    result["mode"] = mode
    result["boundary"] = boundary
    result["timing_boundary"] = timing_boundary
    result["unavailable_reason"] = unavailable_reason
    result["git_revision"] = git_revision
    result["cuda_driver"] = str(cuda_status.get("driver_version", ""))
    result["cuda_runtime"] = str(cuda_status.get("runtime_version", ""))
    result["cuda_toolkit"] = str(build_info.get("cuda_toolkit_version", ""))
    result["compiled_architectures"] = str(build_info.get("cuda_architectures", ""))
    result["gpu_name"] = gpu_name
    result["gpu_compute_capability"] = gpu_compute_capability
    for field, value in campaign8_statuses_for_mode(
        mode, cuda_unavailable=cuda_unavailable
    ).items():
        result[field] = value
    result.setdefault("correctness_digest", {})


def cpu_optimized_selectors_for_case(name: str, build_info: dict[str, Any]) -> list[str]:
    if name != "pairwise_commutation":
        return []
    available = set(build_info.get("available_cpu_backends", []))
    kernels = build_info.get("optimized_cpu_kernels", {})
    selectors: list[str] = []
    for selector in OPTIMIZED_CPU_SELECTOR_ORDER:
        covered_kernels = kernels.get(selector, [])
        if selector in available and any(
            str(kernel).startswith("commutes_with") for kernel in covered_kernels
        ):
            selectors.append(selector)
    return selectors


def cpu_optimized_unavailable_for_case(
    name: str,
    build_info: dict[str, Any],
) -> dict[str, str]:
    if name != "pairwise_commutation":
        return {}
    available = set(build_info.get("available_cpu_backends", []))
    unavailable = build_info.get("unavailable_cpu_backends", {})
    kernels = build_info.get("optimized_cpu_kernels", {})
    reasons: dict[str, str] = {}
    for selector in OPTIMIZED_CPU_SELECTOR_ORDER:
        if selector in available and any(
            str(kernel).startswith("commutes_with") for kernel in kernels.get(selector, [])
        ):
            continue
        reasons[selector] = str(unavailable.get(selector, "no covered optimized kernel"))
    return reasons


def _words_for_num_qubits(num_qubits: int | None) -> int | None:
    if num_qubits is None:
        return None
    return (num_qubits + 63) // 64


def _temporary_storage_estimate(name: str, dataset: dict[str, Any]) -> dict[str, Any]:
    num_qubits = dataset.get("num_qubits")
    words = _words_for_num_qubits(int(num_qubits)) if num_qubits is not None else None

    if name == "simplify_duplicate_pressure":
        terms = int(dataset.get("num_terms", 0))
        survivors = int(dataset.get("survivor_count", terms))
        if words == 1 and num_qubits is not None and int(num_qubits) <= 32:
            key_bytes = 8
            implementation_path = "packed_key32_sort_reduce"
        elif words == 1:
            key_bytes = 16
            implementation_path = "key1_struct_sort_reduce"
        elif words == 2:
            key_bytes = 32
            implementation_path = "key2_struct_sort_reduce"
        elif words is not None:
            key_bytes = 8
            implementation_path = "generic_index_sort_reduce"
        else:
            key_bytes = 0
            implementation_path = "unknown"
        value_bytes = 16
        estimated_bytes = (
            terms * (key_bytes + value_bytes)
            + terms * (key_bytes + value_bytes)
            + survivors * (key_bytes + value_bytes)
        )
        return {
            "available": True,
            "source": "static estimate from benchmark shape and CUDA simplify implementation path",
            "implementation_path": implementation_path,
            "estimated_bytes": estimated_bytes,
            "estimated_allocations": 6,
        }

    if name == "matmul_product_generation_simplify":
        intermediate_terms = int(dataset.get("intermediate_terms", 0))
        survivors = int(dataset.get("survivor_count", intermediate_terms))
        if words == 1 and num_qubits is not None and int(num_qubits) <= 32:
            key_bytes = 8
            simplify_path = "packed_key32_sort_reduce"
        elif words == 1:
            key_bytes = 16
            simplify_path = "key1_struct_sort_reduce"
        elif words == 2:
            key_bytes = 32
            simplify_path = "key2_struct_sort_reduce"
        elif words is not None:
            key_bytes = 8
            simplify_path = "generic_index_sort_reduce"
        else:
            key_bytes = 0
            simplify_path = "unknown"
        product_words = intermediate_terms * int(words or 0)
        product_bytes = product_words * 8 * 2 + intermediate_terms * 16
        simplify_bytes = (
            intermediate_terms * (key_bytes + 16)
            + intermediate_terms * (key_bytes + 16)
            + survivors * (key_bytes + 16)
        )
        return {
            "available": True,
            "source": "static estimate from benchmark shape and CUDA matmul+simplify implementation path",
            "implementation_path": f"product_generation_then_{simplify_path}",
            "estimated_bytes": product_bytes + simplify_bytes,
            "estimated_allocations": 9,
        }

    if name == "pairwise_commutation":
        entries = int(dataset.get("entries", 0))
        return {
            "available": True,
            "source": "static estimate from dense commutation output shape",
            "implementation_path": "device_byte_output_then_host_materialization",
            "estimated_bytes": entries,
            "estimated_allocations": 1,
        }

    if name == "statevector_expectation":
        statevector_length = int(dataset.get("statevector_length", 0))
        dtype = str(dataset.get("statevector_dtype", "complex128"))
        element_bytes = 8 if dtype == "complex64" else 16
        return {
            "available": True,
            "source": "static estimate from statevector input dtype and accumulator shape",
            "implementation_path": "host_statevector_copy_or_cuda_array_interface_plus_device_accumulator",
            "estimated_bytes": statevector_length * element_bytes + 16,
            "estimated_allocations": 2,
        }

    return {
        "available": False,
        "unavailable_reason": "no allocation/materialization model is defined for this case",
    }


def campaign2_instrumentation_for_case(
    name: str, dataset: dict[str, Any] | None = None
) -> dict[str, Any]:
    materialization_by_case = {
        "simplify_duplicate_pressure": "device-resident sparse Pauli buffers returned as DevicePauliSum",
        "statevector_expectation": "host scalar complex result copied from device accumulator",
        "pairwise_commutation": "vector-return host bytes plus caller-owned host bool output timing",
        "matmul_product_generation_simplify": (
            "device-resident product buffers followed by simplified DevicePauliSum"
        ),
    }
    campaign4_fields = campaign4_workspace_fields_for_case(name, dataset)
    workspace_active = campaign4_fields["workspace_allocation_count"] > 0
    instrumentation = {
        "workspace": {
            "enabled": workspace_active,
            "mode": campaign4_fields["workspace_mode"],
            "pre_reserved": workspace_active
            and campaign4_fields["workspace_mode"] == "pre_reserved_outside_timing",
            "growth_inside_timing": workspace_active
            and campaign4_fields["workspace_mode"] == "grow_inside_timing",
            "status": (
                "private Campaign 4 benchmark workspace mode is active"
                if workspace_active
                else "no public or internal reusable workspace is active for this case"
            ),
        },
        "temporary_storage_bytes": (
            _temporary_storage_estimate(name, dataset)
            if dataset is not None
            else {
                "available": False,
                "unavailable_reason": "dataset shape is required for a temporary-storage estimate",
            }
        ),
        "allocation_count": (
            {
                "available": True,
                "source": "static estimate from implementation path; CUDA allocator interception is not enabled",
                "estimated_allocations": _temporary_storage_estimate(name, dataset).get(
                    "estimated_allocations"
                ),
            }
            if dataset is not None and _temporary_storage_estimate(name, dataset)["available"]
            else {
                "available": False,
                "unavailable_reason": "dataset shape is required for an allocation-count estimate",
            }
        ),
        "cuda_stream_mode": "default_stream_synchronize_before_return",
        "result_materialization": materialization_by_case.get(
            name, "operation-specific public result"
        ),
        "duplicate_survivor_count": (dataset or {}).get("survivor_count"),
    }
    instrumentation.update(campaign4_fields)
    return instrumentation


def labels_and_coeffs(op: PauliSum) -> tuple[list[str], list[complex]]:
    labels, coeffs = op.to_labels()
    return list(labels), [complex(value) for value in coeffs]


def assert_same_operator(actual: PauliSum, expected: PauliSum) -> None:
    actual_labels, actual_coeffs = labels_and_coeffs(actual)
    expected_labels, expected_coeffs = labels_and_coeffs(expected)
    if actual_labels != expected_labels:
        raise RuntimeError(f"operator labels differ: {actual_labels!r} != {expected_labels!r}")
    if not np.allclose(actual_coeffs, expected_coeffs, rtol=1.0e-12, atol=1.0e-12):
        raise RuntimeError("operator coefficients differ")


PAULIS = np.asarray(["X", "Y", "Z"])


def duplicate_rate(labels: list[str]) -> float:
    if not labels:
        return 0.0
    return 1.0 - (len(set(labels)) / len(labels))


def generate_operator(
    *,
    rng: np.random.Generator,
    num_qubits: int,
    num_terms: int,
    term_weight: int,
    duplicate_pool_size: int | None = None,
) -> tuple[PauliSum, dict[str, Any]]:
    labels: list[str] = []
    pool_size = duplicate_pool_size or num_terms
    pool: list[str] = []
    for _ in range(pool_size):
        chars = ["I"] * num_qubits
        for qubit in rng.choice(num_qubits, size=min(term_weight, num_qubits), replace=False):
            chars[num_qubits - 1 - int(qubit)] = str(rng.choice(PAULIS))
        pool.append("".join(chars))

    for term in range(num_terms):
        labels.append(pool[term % pool_size])
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    return (
        PauliSum.from_labels(labels, np.asarray(coeffs, dtype=np.complex128).tolist()),
        {
            "num_qubits": num_qubits,
            "num_terms": num_terms,
            "term_weight_distribution": f"fixed term_weight={term_weight}",
            "duplicate_rate": duplicate_rate(labels),
            "duplicate_pool_size": pool_size,
            "coefficient_dtype": "complex128",
            "operator_construction_method": "deterministic weighted labels with duplicate pool",
        },
    )


def normalized_statevector(rng: np.random.Generator, num_qubits: int) -> np.ndarray:
    raw = rng.normal(size=1 << num_qubits) + 1j * rng.normal(size=1 << num_qubits)
    psi = np.asarray(raw, dtype=np.complex128)
    return psi / np.linalg.norm(psi)


def case_result(
    *,
    name: str,
    dataset: dict[str, Any],
    cpu_fn: Callable[[], Any],
    cuda_transfer_fn: Callable[[], Any],
    cuda_resident_fn: Callable[[], Any],
    build_info: dict[str, Any],
    warmup: int,
    repeat: int,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    _, cpu_timing = timed_cpu_backend_call(cpu_fn, backend="scalar", warmup=warmup, repeat=repeat)
    _, cpu_default_timing = timed_call(cpu_fn, warmup=warmup, repeat=repeat)
    cpu_optimized_selectors = cpu_optimized_selectors_for_case(name, build_info)
    cpu_optimized_timings: dict[str, dict[str, float]] = {}
    cpu_optimized_status = "not applicable: no named optimized CPU kernel for this operation"
    for selector in cpu_optimized_selectors:
        _, timing = timed_cpu_backend_call(
            cpu_fn,
            backend=selector,
            warmup=warmup,
            repeat=repeat,
        )
        cpu_optimized_timings[selector] = timing
    if cpu_optimized_timings:
        cpu_optimized_status = "available"
    best_cpu_optimized_backend = None
    best_cpu_optimized_timing: dict[str, float] | None = None
    if cpu_optimized_timings:
        best_cpu_optimized_backend = min(
            cpu_optimized_timings,
            key=lambda selector: cpu_optimized_timings[selector]["median"],
        )
        best_cpu_optimized_timing = cpu_optimized_timings[best_cpu_optimized_backend]
    _, cuda_transfer_timing = timed_call(cuda_transfer_fn, warmup=warmup, repeat=repeat)
    _, cuda_resident_timing = timed_call(cuda_resident_fn, warmup=warmup, repeat=repeat)
    return {
        "name": name,
        "dataset": dataset,
        "instrumentation": campaign2_instrumentation_for_case(name, dataset),
        "results": {
            "cpu_scalar_seconds": cpu_timing["median"],
            "cpu_default_seconds": cpu_default_timing["median"],
            "cpu_default_min_seconds": cpu_default_timing["min"],
            "cpu_default_p10_seconds": cpu_default_timing["p10"],
            "cpu_default_p90_seconds": cpu_default_timing["p90"],
            "cpu_default_max_seconds": cpu_default_timing["max"],
            "cpu_optimized_seconds": (
                best_cpu_optimized_timing["median"]
                if best_cpu_optimized_timing is not None
                else None
            ),
            "cpu_optimized_backend": best_cpu_optimized_backend,
            "cpu_optimized_timings": {
                selector: {
                    "seconds": timing["median"],
                    "p10_seconds": timing["p10"],
                    "p90_seconds": timing["p90"],
                    "min_seconds": timing["min"],
                    "max_seconds": timing["max"],
                }
                for selector, timing in cpu_optimized_timings.items()
            },
            "cpu_optimized_unavailable": cpu_optimized_unavailable_for_case(name, build_info),
            "cpu_optimized_status": cpu_optimized_status,
            "cuda_transfer_inclusive_seconds": cuda_transfer_timing["median"],
            "cuda_device_resident_seconds": cuda_resident_timing["median"],
            "cpu_scalar_min_seconds": cpu_timing["min"],
            "cpu_scalar_p10_seconds": cpu_timing["p10"],
            "cpu_scalar_p90_seconds": cpu_timing["p90"],
            "cpu_scalar_max_seconds": cpu_timing["max"],
            "cpu_optimized_min_seconds": (
                best_cpu_optimized_timing["min"] if best_cpu_optimized_timing is not None else None
            ),
            "cpu_optimized_p10_seconds": (
                best_cpu_optimized_timing["p10"] if best_cpu_optimized_timing is not None else None
            ),
            "cpu_optimized_p90_seconds": (
                best_cpu_optimized_timing["p90"] if best_cpu_optimized_timing is not None else None
            ),
            "cpu_optimized_max_seconds": (
                best_cpu_optimized_timing["max"] if best_cpu_optimized_timing is not None else None
            ),
            "cuda_transfer_inclusive_min_seconds": cuda_transfer_timing["min"],
            "cuda_transfer_inclusive_p10_seconds": cuda_transfer_timing["p10"],
            "cuda_transfer_inclusive_p90_seconds": cuda_transfer_timing["p90"],
            "cuda_transfer_inclusive_max_seconds": cuda_transfer_timing["max"],
            "cuda_device_resident_min_seconds": cuda_resident_timing["min"],
            "cuda_device_resident_p10_seconds": cuda_resident_timing["p10"],
            "cuda_device_resident_p90_seconds": cuda_resident_timing["p90"],
            "cuda_device_resident_max_seconds": cuda_resident_timing["max"],
            "repeat_count": repeat,
            "warmup_count": warmup,
        },
        "notes": notes or [],
    }


def timed_preallocated_cuda_commutation(
    *,
    lhs_device: Any,
    rhs_device: Any,
    expected: Any,
    max_entries: int,
    warmup: int,
    repeat: int,
) -> dict[str, float]:
    expected_flat = np.asarray(expected, dtype=np.bool_).reshape(-1)
    output = np.empty(expected_flat.size, dtype=np.bool_)

    _, timing = timed_call(
        lambda: lhs_device.commutes_with_into(
            rhs_device,
            output,
            max_commutation_matrix_entries=max_entries,
        ),
        warmup=warmup,
        repeat=repeat,
    )
    if not np.array_equal(output, expected_flat):
        raise RuntimeError("CUDA preallocated commutation output differs from CPU output")
    return timing


def add_preallocated_commutation_timing_fields(
    result: dict[str, Any],
    preallocated_timing: dict[str, float],
) -> None:
    result["results"]["cuda_device_resident_preallocated_seconds"] = preallocated_timing["median"]
    result["results"]["cuda_device_resident_preallocated_p10_seconds"] = preallocated_timing["p10"]
    result["results"]["cuda_device_resident_preallocated_p90_seconds"] = preallocated_timing["p90"]
    result["results"]["cuda_device_resident_preallocated_min_seconds"] = preallocated_timing["min"]
    result["results"]["cuda_device_resident_preallocated_max_seconds"] = preallocated_timing["max"]


def timed_reused_device_output_cuda_commutation(
    *,
    lhs_device: Any,
    rhs_device: Any,
    expected: Any,
    max_entries: int,
    warmup: int,
    repeat: int,
) -> dict[str, float]:
    expected_flat = np.asarray(expected, dtype=np.bool_).reshape(-1)
    output = np.empty(expected_flat.size, dtype=np.bool_)
    with temporary_env_var("WOLFGANG_CUDA_BENCH_REUSE_COMMUTATION_DEVICE_OUTPUT", "1"):
        _, timing = timed_call(
            lambda: lhs_device.commutes_with_into(
                rhs_device,
                output,
                max_commutation_matrix_entries=max_entries,
            ),
            warmup=warmup,
            repeat=repeat,
        )
    if not np.array_equal(output, expected_flat):
        raise RuntimeError("CUDA reused-device-output commutation output differs from CPU output")
    return timing


def add_reused_device_output_commutation_timing_fields(
    result: dict[str, Any],
    timing: dict[str, float],
) -> None:
    prefix = "cuda_device_resident_reused_device_output"
    result["results"][f"{prefix}_seconds"] = timing["median"]
    result["results"][f"{prefix}_p10_seconds"] = timing["p10"]
    result["results"][f"{prefix}_p90_seconds"] = timing["p90"]
    result["results"][f"{prefix}_min_seconds"] = timing["min"]
    result["results"][f"{prefix}_max_seconds"] = timing["max"]


def _assert_device_commutation_output_matches(output: Any, expected: Any) -> None:
    if not np.array_equal(output.to_host(), np.asarray(expected, dtype=np.bool_)):
        raise RuntimeError("CUDA device commutation output differs from CPU output")


def timed_public_device_output_allocate_cuda_commutation(
    *,
    lhs_device: Any,
    rhs_device: Any,
    expected: Any,
    max_entries: int,
    warmup: int,
    repeat: int,
) -> dict[str, float]:
    matrix, timing = timed_call(
        lambda: lhs_device.commutes_with_device(
            rhs_device,
            max_commutation_matrix_entries=max_entries,
        ),
        warmup=warmup,
        repeat=repeat,
    )
    _assert_device_commutation_output_matches(matrix, expected)
    return timing


def timed_public_device_output_reuse_cuda_commutation(
    *,
    lhs_device: Any,
    rhs_device: Any,
    expected: Any,
    max_entries: int,
    warmup: int,
    repeat: int,
) -> dict[str, float]:
    output = wolfgang.DeviceCommutationMatrix.empty(
        (lhs_device.num_terms, rhs_device.num_terms),
        device=lhs_device.device,
    )
    same, timing = timed_call(
        lambda: lhs_device.commutes_with_device(
            rhs_device,
            max_commutation_matrix_entries=max_entries,
            output=output,
        ),
        warmup=warmup,
        repeat=repeat,
    )
    if same is not output:
        raise RuntimeError("CUDA device commutation output reuse did not return the output object")
    _assert_device_commutation_output_matches(output, expected)
    return timing


def timed_public_device_output_to_host_cuda_commutation(
    *,
    lhs_device: Any,
    rhs_device: Any,
    expected: Any,
    max_entries: int,
    warmup: int,
    repeat: int,
) -> dict[str, float]:
    output = lhs_device.commutes_with_device(
        rhs_device,
        max_commutation_matrix_entries=max_entries,
    )
    host, timing = timed_call(output.to_host, warmup=warmup, repeat=repeat)
    if not np.array_equal(host, np.asarray(expected, dtype=np.bool_)):
        raise RuntimeError("CUDA device commutation to_host output differs from CPU output")
    return timing


def timed_public_device_output_cuda_array_interface_export(
    *,
    lhs_device: Any,
    rhs_device: Any,
    max_entries: int,
    warmup: int,
    repeat: int,
) -> dict[str, float]:
    output = lhs_device.commutes_with_device(
        rhs_device,
        max_commutation_matrix_entries=max_entries,
    )
    interface, timing = timed_call(
        lambda: output.__cuda_array_interface__,
        warmup=warmup,
        repeat=repeat,
    )
    if interface["shape"] != (lhs_device.num_terms, rhs_device.num_terms):
        raise RuntimeError("CUDA device commutation array-interface shape is incorrect")
    if interface["typestr"] != "|u1":
        raise RuntimeError("CUDA device commutation array-interface dtype is incorrect")
    return timing


def _assert_device_commutation_counts_match(output: Any, expected: Any) -> None:
    expected_uint64 = np.asarray(expected, dtype=np.uint64)
    if output.count_commuting() != int(expected_uint64.sum()):
        raise RuntimeError("CUDA device commutation total count differs from CPU output")
    if not np.array_equal(
        output.count_commuting(axis=0),
        expected_uint64.sum(axis=0, dtype=np.uint64),
    ):
        raise RuntimeError("CUDA device commutation column counts differ from CPU output")
    if not np.array_equal(
        output.count_commuting(axis=1),
        expected_uint64.sum(axis=1, dtype=np.uint64),
    ):
        raise RuntimeError("CUDA device commutation row counts differ from CPU output")


def timed_public_device_output_consumer_cuda_commutation(
    *,
    lhs_device: Any,
    rhs_device: Any,
    expected: Any,
    max_entries: int,
    warmup: int,
    repeat: int,
) -> dict[str, dict[str, float] | int]:
    output = lhs_device.commutes_with_device(
        rhs_device,
        max_commutation_matrix_entries=max_entries,
    )
    _assert_device_commutation_counts_match(output, expected)
    _, total_timing = timed_call(output.count_commuting, warmup=warmup, repeat=repeat)
    _, axis0_timing = timed_call(
        lambda: output.count_commuting(axis=0), warmup=warmup, repeat=repeat
    )
    _, axis1_timing = timed_call(
        lambda: output.count_commuting(axis=1), warmup=warmup, repeat=repeat
    )
    return {
        "total": total_timing,
        "axis0": axis0_timing,
        "axis1": axis1_timing,
        "to_host_bytes": int(8 * (1 + lhs_device.num_terms + rhs_device.num_terms)),
    }


def timed_cupy_device_output_consumer_cuda_commutation(
    *,
    lhs_device: Any,
    rhs_device: Any,
    expected: Any,
    max_entries: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    try:
        import cupy
    except Exception as exc:
        return {
            "available": False,
            "unavailable_reason": f"{type(exc).__name__}: {exc}",
        }

    try:
        _require_supported_cupy_runtime_for_current_cuda_architecture(cupy)
    except BaseException as exc:
        import pytest

        if isinstance(exc, pytest.skip.Exception):
            return {
                "available": False,
                "unavailable_reason": str(exc),
            }
        raise

    output = lhs_device.commutes_with_device(
        rhs_device,
        max_commutation_matrix_entries=max_entries,
    )
    expected_uint64 = np.asarray(expected, dtype=np.uint64)
    view, asarray_timing = timed_call(lambda: cupy.asarray(output), warmup=warmup, repeat=repeat)
    if tuple(view.shape) != (lhs_device.num_terms, rhs_device.num_terms):
        raise RuntimeError("CuPy DeviceCommutationMatrix view shape is incorrect")
    if view.dtype != cupy.uint8:
        raise RuntimeError("CuPy DeviceCommutationMatrix view dtype is incorrect")

    compute_capability = _cuda_compute_capability()
    try:
        total, total_timing = timed_call(
            lambda: int(cupy.sum(view).get()),
            warmup=warmup,
            repeat=repeat,
        )
        axis0, axis0_timing = timed_call(
            lambda: cupy.asnumpy(cupy.sum(view, axis=0)),
            warmup=warmup,
            repeat=repeat,
        )
        axis1, axis1_timing = timed_call(
            lambda: cupy.asnumpy(cupy.sum(view, axis=1)),
            warmup=warmup,
            repeat=repeat,
        )
        dense_host, dense_host_timing = timed_call(
            lambda: cupy.asnumpy(view), warmup=warmup, repeat=repeat
        )
    except Exception as exc:
        if _cupy_compile_error_indicates_unsupported_cuda_architecture(
            exc,
            compute_capability=compute_capability,
        ):
            assert compute_capability is not None
            return {
                "available": False,
                "unavailable_reason": (
                    "CuPy runtime does not support CUDA compute capability "
                    f"{compute_capability[0]}.{compute_capability[1]}: {type(exc).__name__}: {exc}"
                ),
            }
        raise

    if int(total) != int(expected_uint64.sum()):
        raise RuntimeError("CuPy DeviceCommutationMatrix total count differs from CPU output")
    if not np.array_equal(axis0, expected_uint64.sum(axis=0, dtype=np.uint64)):
        raise RuntimeError("CuPy DeviceCommutationMatrix column counts differ from CPU output")
    if not np.array_equal(axis1, expected_uint64.sum(axis=1, dtype=np.uint64)):
        raise RuntimeError("CuPy DeviceCommutationMatrix row counts differ from CPU output")
    if not np.array_equal(dense_host.astype(np.bool_), np.asarray(expected, dtype=np.bool_)):
        raise RuntimeError("CuPy DeviceCommutationMatrix dense host copy differs from CPU output")

    return {
        "available": True,
        "unavailable_reason": None,
        "asarray": asarray_timing,
        "total": total_timing,
        "axis0": axis0_timing,
        "axis1": axis1_timing,
        "dense_to_host": dense_host_timing,
    }


def timed_public_conflict_degrees_cuda_commutation(
    *,
    lhs_device: Any,
    rhs_device: Any,
    expected: Any,
    max_entries: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    output = lhs_device.commutes_with_device(
        rhs_device,
        max_commutation_matrix_entries=max_entries,
    )
    expected_conflicts = np.logical_not(np.asarray(expected, dtype=np.bool_)).astype(np.uint64)

    total, total_timing = timed_call(output.conflict_degrees, warmup=warmup, repeat=repeat)
    axis0, axis0_timing = timed_call(
        lambda: output.conflict_degrees(axis=0),
        warmup=warmup,
        repeat=repeat,
    )
    axis1, axis1_timing = timed_call(
        lambda: output.conflict_degrees(axis=1),
        warmup=warmup,
        repeat=repeat,
    )
    dense_conflicts, dense_timing = timed_call(
        lambda: np.logical_not(output.to_host()).astype(np.uint64),
        warmup=warmup,
        repeat=repeat,
    )

    if int(total) != int(expected_conflicts.sum()):
        raise RuntimeError("CUDA public conflict degree total differs from CPU output")
    if not np.array_equal(axis0, expected_conflicts.sum(axis=0, dtype=np.uint64)):
        raise RuntimeError("CUDA public conflict degree column counts differ from CPU output")
    if not np.array_equal(axis1, expected_conflicts.sum(axis=1, dtype=np.uint64)):
        raise RuntimeError("CUDA public conflict degree row counts differ from CPU output")
    if not np.array_equal(dense_conflicts, expected_conflicts):
        raise RuntimeError("CUDA dense-to-host conflict degree baseline differs from CPU output")

    return {
        "axis_none": total_timing,
        "axis0": axis0_timing,
        "axis1": axis1_timing,
        "dense_to_host_plus_numpy": dense_timing,
        "compact_host_bytes": int(8 * (1 + lhs_device.num_terms + rhs_device.num_terms)),
        "dense_host_bytes": int(lhs_device.num_terms * rhs_device.num_terms),
    }


def timed_cupy_dlpack_consumer_cuda_commutation(
    *,
    lhs_device: Any,
    rhs_device: Any,
    expected: Any,
    max_entries: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    try:
        import cupy
    except Exception as exc:
        return {
            "available": False,
            "unavailable_reason": f"{type(exc).__name__}: {exc}",
        }

    try:
        _require_supported_cupy_runtime_for_current_cuda_architecture(cupy)
    except BaseException as exc:
        import pytest

        if isinstance(exc, pytest.skip.Exception):
            return {
                "available": False,
                "unavailable_reason": str(exc),
            }
        raise

    output = lhs_device.commutes_with_device(
        rhs_device,
        max_commutation_matrix_entries=max_entries,
    )
    expected_uint64 = np.asarray(expected, dtype=np.uint64)
    view, from_dlpack_timing = timed_call(
        lambda: cupy.from_dlpack(output),
        warmup=warmup,
        repeat=repeat,
    )
    if tuple(view.shape) != (lhs_device.num_terms, rhs_device.num_terms):
        raise RuntimeError("CuPy DLPack DeviceCommutationMatrix view shape is incorrect")
    if view.dtype != cupy.uint8:
        raise RuntimeError("CuPy DLPack DeviceCommutationMatrix view dtype is incorrect")

    compute_capability = _cuda_compute_capability()
    try:
        total, total_timing = timed_call(
            lambda: int(cupy.sum(view).get()),
            warmup=warmup,
            repeat=repeat,
        )
        dense_host, dense_host_timing = timed_call(
            lambda: cupy.asnumpy(view), warmup=warmup, repeat=repeat
        )
    except Exception as exc:
        if _cupy_compile_error_indicates_unsupported_cuda_architecture(
            exc,
            compute_capability=compute_capability,
        ):
            assert compute_capability is not None
            return {
                "available": False,
                "unavailable_reason": (
                    "CuPy runtime does not support CUDA compute capability "
                    f"{compute_capability[0]}.{compute_capability[1]}: {type(exc).__name__}: {exc}"
                ),
            }
        raise

    if int(total) != int(expected_uint64.sum()):
        raise RuntimeError("CuPy DLPack DeviceCommutationMatrix total differs from CPU output")
    if not np.array_equal(dense_host.astype(np.bool_), np.asarray(expected, dtype=np.bool_)):
        raise RuntimeError(
            "CuPy DLPack DeviceCommutationMatrix dense host copy differs from CPU output"
        )

    return {
        "available": True,
        "unavailable_reason": None,
        "from_dlpack": from_dlpack_timing,
        "sum_total": total_timing,
        "dense_to_host": dense_host_timing,
    }


def timed_torch_dlpack_consumer_cuda_commutation(
    *,
    lhs_device: Any,
    rhs_device: Any,
    expected: Any,
    max_entries: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "available": False,
            "unavailable_reason": f"torch not importable: {type(exc).__name__}: {exc}",
            "torch_version": None,
            "torch_cuda_version": None,
        }
    if not torch.cuda.is_available():
        return {
            "available": False,
            "unavailable_reason": "torch importable but torch.cuda.is_available() is false",
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
        }

    output = lhs_device.commutes_with_device(
        rhs_device,
        max_commutation_matrix_entries=max_entries,
    )
    expected_uint64 = np.asarray(expected, dtype=np.uint64)

    def from_versioned_capsule() -> Any:
        capsule = output.__dlpack__(max_version=(1, 0))
        return torch.utils.dlpack.from_dlpack(capsule)

    try:
        view, from_dlpack_timing = timed_call(
            from_versioned_capsule,
            warmup=warmup,
            repeat=repeat,
        )
    except Exception as exc:
        return {
            "available": False,
            "unavailable_reason": (
                "torch CUDA available but torch.utils.dlpack cannot consume "
                f"the versioned read-only capsule: {type(exc).__name__}: {exc}"
            ),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
        }

    if tuple(view.shape) != (lhs_device.num_terms, rhs_device.num_terms):
        raise RuntimeError("PyTorch DLPack DeviceCommutationMatrix view shape is incorrect")
    if view.dtype != torch.uint8:
        raise RuntimeError("PyTorch DLPack DeviceCommutationMatrix view dtype is incorrect")
    if not view.is_cuda:
        raise RuntimeError("PyTorch DLPack DeviceCommutationMatrix view is not CUDA-resident")

    total, total_timing = timed_call(
        lambda: int(torch.sum(view.to(torch.int64)).item()),
        warmup=warmup,
        repeat=repeat,
    )
    dense_host, dense_host_timing = timed_call(
        lambda: view.cpu().numpy(),
        warmup=warmup,
        repeat=repeat,
    )

    if int(total) != int(expected_uint64.sum()):
        raise RuntimeError("PyTorch DLPack DeviceCommutationMatrix total differs from CPU output")
    if not np.array_equal(dense_host.astype(np.bool_), np.asarray(expected, dtype=np.bool_)):
        raise RuntimeError(
            "PyTorch DLPack DeviceCommutationMatrix dense host copy differs from CPU output"
        )

    return {
        "available": True,
        "unavailable_reason": None,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "from_dlpack": from_dlpack_timing,
        "sum_total": total_timing,
        "dense_to_host": dense_host_timing,
    }


def timed_private_fused_consumer_cuda_commutation(
    *,
    lhs_device: Any,
    rhs_device: Any,
    expected: Any,
    max_entries: int,
    warmup: int,
    repeat: int,
    top_k: int = 8,
) -> dict[str, Any]:
    output = lhs_device.commutes_with_device(
        rhs_device,
        max_commutation_matrix_entries=max_entries,
    )
    expected_bool = np.asarray(expected, dtype=np.bool_)
    expected_conflicts = np.logical_not(expected_bool).astype(np.uint64)
    expected_edge_count = int(expected_conflicts.sum())

    csr_report, csr_timing = timed_call(
        lambda: core._benchmark_cuda_fused_commutation_consumer(
            "csr_anticommutation_graph",
            output,
            include_outputs=False,
            require_cuda=True,
        ),
        warmup=warmup,
        repeat=repeat,
    )
    if int(csr_report["correctness_digest"]["edge_count"]) != expected_edge_count:
        raise RuntimeError("CUDA fused CSR anti-commutation edge count differs from CPU output")

    degree_report, degree_timing = timed_call(
        lambda: core._benchmark_cuda_fused_commutation_consumer(
            "conflict_degrees",
            output,
            include_outputs=True,
            require_cuda=True,
        ),
        warmup=warmup,
        repeat=repeat,
    )
    expected_row_conflicts = expected_conflicts.sum(axis=1, dtype=np.uint64)
    expected_col_conflicts = expected_conflicts.sum(axis=0, dtype=np.uint64)
    if not np.array_equal(
        np.asarray(degree_report["row_conflicts"], dtype=np.uint64),
        expected_row_conflicts,
    ):
        raise RuntimeError("CUDA fused row conflict degrees differ from CPU output")
    if not np.array_equal(
        np.asarray(degree_report["col_conflicts"], dtype=np.uint64),
        expected_col_conflicts,
    ):
        raise RuntimeError("CUDA fused column conflict degrees differ from CPU output")

    grouping_report, grouping_timing = timed_call(
        lambda: core._benchmark_cuda_fused_commutation_consumer(
            "grouping_summary",
            output,
            include_outputs=False,
            top_k=top_k,
            require_cuda=True,
        ),
        warmup=warmup,
        repeat=repeat,
    )
    bitpacked_report = core._benchmark_cuda_fused_commutation_consumer(
        "bitpacked_ab",
        output,
        include_outputs=False,
        require_cuda=True,
    )

    return {
        "csr": csr_timing,
        "csr_report": csr_report,
        "conflict_degrees": degree_timing,
        "conflict_degrees_report": degree_report,
        "grouping_summary": grouping_timing,
        "grouping_summary_report": grouping_report,
        "bitpacked_report": bitpacked_report,
        "edge_count": expected_edge_count,
        "conflict_summary_to_host_bytes": int(degree_report["output_sizes"]["host_bytes"]),
    }


def _assert_campaign8_status_fields(report: dict[str, Any]) -> None:
    missing = [field for field in CAMPAIGN8_REQUIRED_STATUS_FIELDS if field not in report]
    if missing:
        raise RuntimeError(f"Campaign 8 private hook omitted status field(s): {missing}")


def timed_private_campaign8_device_resident_consumer_cuda_commutation(
    *,
    lhs_device: Any,
    rhs_device: Any,
    expected: Any,
    max_entries: int,
    warmup: int,
    repeat: int,
    top_k: int = 8,
) -> dict[str, Any]:
    output = lhs_device.commutes_with_device(
        rhs_device,
        max_commutation_matrix_entries=max_entries,
    )
    expected_bool = np.asarray(expected, dtype=np.bool_)
    expected_conflicts = np.logical_not(expected_bool).astype(np.uint64)
    expected_edge_count = int(expected_conflicts.sum())

    dense_report, dense_timing = timed_call(
        lambda: output.to_host(),
        warmup=warmup,
        repeat=repeat,
    )
    if not np.array_equal(np.asarray(dense_report, dtype=np.bool_), expected_bool):
        raise RuntimeError("Campaign 8 dense validation output differs from CPU output")

    axis_none, axis_none_timing = timed_call(
        lambda: output.count_commuting(),
        warmup=warmup,
        repeat=repeat,
    )
    axis0, axis0_timing = timed_call(
        lambda: output.count_commuting(axis=0),
        warmup=warmup,
        repeat=repeat,
    )
    axis1, axis1_timing = timed_call(
        lambda: output.count_commuting(axis=1),
        warmup=warmup,
        repeat=repeat,
    )
    expected_commuting = expected_bool.astype(np.uint64)
    if int(axis_none) != int(expected_commuting.sum()):
        raise RuntimeError("Campaign 8 count_commuting total differs from CPU output")
    if not np.array_equal(np.asarray(axis0, dtype=np.uint64), expected_commuting.sum(axis=0)):
        raise RuntimeError("Campaign 8 count_commuting axis=0 differs from CPU output")
    if not np.array_equal(np.asarray(axis1, dtype=np.uint64), expected_commuting.sum(axis=1)):
        raise RuntimeError("Campaign 8 count_commuting axis=1 differs from CPU output")

    csr_report, csr_timing = timed_call(
        lambda: core._benchmark_cuda_fused_commutation_consumer(
            "csr_anticommutation_graph",
            output,
            include_outputs=False,
            require_cuda=True,
        ),
        warmup=warmup,
        repeat=repeat,
    )
    if int(csr_report["correctness_digest"]["edge_count"]) != expected_edge_count:
        raise RuntimeError("Campaign 8 CSR baseline edge count differs from CPU output")

    graph_report, graph_timing = timed_call(
        lambda: core._benchmark_cuda_device_resident_consumer(
            "device_resident_graph",
            output,
            include_outputs=False,
            require_cuda=True,
        ),
        warmup=warmup,
        repeat=repeat,
    )
    _assert_campaign8_status_fields(graph_report)
    if int(graph_report["correctness_digest"]["edge_count"]) != expected_edge_count:
        raise RuntimeError("Campaign 8 compact graph edge count differs from CPU output")
    if int(graph_report["output_sizes"]["full_csr_host_bytes"]) != 0:
        raise RuntimeError("Campaign 8 compact graph path unexpectedly exported full CSR")

    grouping_report, grouping_timing = timed_call(
        lambda: core._benchmark_cuda_device_resident_consumer(
            "device_grouping_consumer",
            output,
            include_outputs=False,
            top_k=top_k,
            require_cuda=True,
        ),
        warmup=warmup,
        repeat=repeat,
    )
    _assert_campaign8_status_fields(grouping_report)
    if int(grouping_report["correctness_digest"]["row_conflict_sum"]) != expected_edge_count:
        raise RuntimeError("Campaign 8 grouping row conflict sum differs from CPU output")

    dlpack_report = core._benchmark_cuda_device_resident_consumer(
        "dlpack_consumer",
        output,
        require_cuda=True,
    )
    stream_graph_report = core._benchmark_cuda_device_resident_consumer(
        "stream_graph_probe",
        output,
        require_cuda=True,
    )
    scatter_report = core._benchmark_cuda_device_resident_consumer(
        "csr_scatter_ab",
        output,
        require_cuda=True,
    )
    for report in (dlpack_report, stream_graph_report, scatter_report):
        _assert_campaign8_status_fields(report)

    return {
        "dense_to_host": dense_timing,
        "count_axis_none": axis_none_timing,
        "count_axis0": axis0_timing,
        "count_axis1": axis1_timing,
        "campaign7_csr": csr_timing,
        "campaign7_csr_report": csr_report,
        "device_graph": graph_timing,
        "device_graph_report": graph_report,
        "grouping": grouping_timing,
        "grouping_report": grouping_report,
        "dlpack_report": dlpack_report,
        "stream_graph_report": stream_graph_report,
        "scatter_report": scatter_report,
        "edge_count": expected_edge_count,
    }


def add_public_device_output_commutation_timing_fields(
    result: dict[str, Any],
    *,
    allocate_timing: dict[str, float],
    reuse_timing: dict[str, float],
    to_host_timing: dict[str, float],
    cuda_array_interface_timing: dict[str, float],
) -> None:
    timings = {
        "cuda_device_output_allocate": allocate_timing,
        "cuda_device_output_reuse": reuse_timing,
        "cuda_device_output_to_host": to_host_timing,
        "cuda_device_output_cuda_array_interface_export": cuda_array_interface_timing,
    }
    for prefix, timing in timings.items():
        result["results"][f"{prefix}_seconds"] = timing["median"]
        result["results"][f"{prefix}_p10_seconds"] = timing["p10"]
        result["results"][f"{prefix}_p90_seconds"] = timing["p90"]
        result["results"][f"{prefix}_min_seconds"] = timing["min"]
        result["results"][f"{prefix}_max_seconds"] = timing["max"]
    for suffix in ["seconds", "p10_seconds", "p90_seconds", "min_seconds", "max_seconds"]:
        result["results"][f"cuda_device_output_dense_to_host_{suffix}"] = result["results"][
            f"cuda_device_output_to_host_{suffix}"
        ]

    result["instrumentation"]["public_device_output"] = {
        "status": "experimental_public",
        "result_materialization_target": "device_uint8_matrix",
        "timing_boundaries": [
            "device_output_allocating",
            "device_output_reused",
            "device_output_to_host",
            "device_output_cuda_array_interface_export",
        ],
    }
    result["instrumentation"]["result_materialization_target"] = "device_uint8_matrix"
    result["instrumentation"]["timing_boundary"] = (
        "device_output_allocating,device_output_reused,device_output_to_host"
    )


def add_device_output_consumer_timing_fields(
    result: dict[str, Any],
    consumer_timing: dict[str, dict[str, float] | int],
) -> None:
    timings = {
        "cuda_device_output_consumer_total": consumer_timing["total"],
        "cuda_device_output_consumer_axis0": consumer_timing["axis0"],
        "cuda_device_output_consumer_axis1": consumer_timing["axis1"],
    }
    for prefix, raw_timing in timings.items():
        timing = dict(raw_timing)
        result["results"][f"{prefix}_seconds"] = timing["median"]
        result["results"][f"{prefix}_p10_seconds"] = timing["p10"]
        result["results"][f"{prefix}_p90_seconds"] = timing["p90"]
        result["results"][f"{prefix}_min_seconds"] = timing["min"]
        result["results"][f"{prefix}_max_seconds"] = timing["max"]

    result["results"]["cuda_device_output_consumer_to_host_bytes"] = int(
        consumer_timing["to_host_bytes"]
    )
    result["instrumentation"]["device_output_consumer"] = {
        "status": "public_compact_summary",
        "timing_boundaries": [
            "count_commuting_total",
            "count_commuting_axis0",
            "count_commuting_axis1",
        ],
        "result_materialization": "compact uint64 counts copied to host",
    }


def add_cupy_consumer_timing_fields(
    result: dict[str, Any],
    cupy_timing: dict[str, Any],
) -> None:
    result["results"]["cupy_consumer_available"] = bool(cupy_timing["available"])
    result["results"]["cupy_consumer_unavailable_reason"] = cupy_timing["unavailable_reason"]
    result["instrumentation"]["cupy_cuda_array_interface_consumer"] = {
        "status": "available" if cupy_timing["available"] else "unavailable",
        "timing_boundary": "CuPy consumer through DeviceCommutationMatrix.__cuda_array_interface__",
    }
    if not cupy_timing["available"]:
        return

    timings = {
        "cupy_asarray_export": cupy_timing["asarray"],
        "cupy_sum_total": cupy_timing["total"],
        "cupy_sum_axis0": cupy_timing["axis0"],
        "cupy_sum_axis1": cupy_timing["axis1"],
        "cupy_dense_to_host": cupy_timing["dense_to_host"],
    }
    for prefix, timing in timings.items():
        result["results"][f"{prefix}_seconds"] = timing["median"]
        result["results"][f"{prefix}_p10_seconds"] = timing["p10"]
        result["results"][f"{prefix}_p90_seconds"] = timing["p90"]
        result["results"][f"{prefix}_min_seconds"] = timing["min"]
        result["results"][f"{prefix}_max_seconds"] = timing["max"]


def add_public_conflict_degrees_timing_fields(
    result: dict[str, Any],
    timing: dict[str, Any],
) -> None:
    timings = {
        "conflict_degrees_axis_none": timing["axis_none"],
        "conflict_degrees_axis_0": timing["axis0"],
        "conflict_degrees_axis_1": timing["axis1"],
        "dense_to_host_plus_numpy_conflicts": timing["dense_to_host_plus_numpy"],
    }
    for prefix, raw_timing in timings.items():
        result["results"][f"{prefix}_seconds"] = raw_timing["median"]
        result["results"][f"{prefix}_p10_seconds"] = raw_timing["p10"]
        result["results"][f"{prefix}_p90_seconds"] = raw_timing["p90"]
        result["results"][f"{prefix}_min_seconds"] = raw_timing["min"]
        result["results"][f"{prefix}_max_seconds"] = raw_timing["max"]

    result["results"]["conflict_degrees_compact_host_bytes"] = int(timing["compact_host_bytes"])
    result["results"]["dense_to_host_plus_numpy_conflicts_host_bytes"] = int(
        timing["dense_host_bytes"]
    )
    result["instrumentation"]["campaign9_public_conflict_degrees"] = {
        "status": "implemented",
        "public_api": "DeviceCommutationMatrix.conflict_degrees(axis=None|0|1)",
        "timing_boundaries": list(timings),
        "true_grouping_api_status": "rejected_with_evidence",
    }


def add_cupy_dlpack_consumer_timing_fields(
    result: dict[str, Any],
    timing: dict[str, Any],
) -> None:
    result["results"]["cupy_dlpack_consumer_available"] = bool(timing["available"])
    result["results"]["cupy_dlpack_consumer_unavailable_reason"] = timing["unavailable_reason"]
    result["instrumentation"]["cupy_dlpack_consumer"] = {
        "status": "available" if timing["available"] else "unavailable",
        "timing_boundary": "CuPy consumer through DeviceCommutationMatrix.__dlpack__",
        "ownership": "DLPack capsule is single-consumer; Wolfgang owner is retained by deleter context",
    }
    if not timing["available"]:
        return

    timings = {
        "cupy_dlpack_from_dlpack": timing["from_dlpack"],
        "cupy_dlpack_sum_total": timing["sum_total"],
        "cupy_dlpack_dense_to_host": timing["dense_to_host"],
    }
    for prefix, raw_timing in timings.items():
        result["results"][f"{prefix}_seconds"] = raw_timing["median"]
        result["results"][f"{prefix}_p10_seconds"] = raw_timing["p10"]
        result["results"][f"{prefix}_p90_seconds"] = raw_timing["p90"]
        result["results"][f"{prefix}_min_seconds"] = raw_timing["min"]
        result["results"][f"{prefix}_max_seconds"] = raw_timing["max"]


def add_torch_dlpack_consumer_timing_fields(
    result: dict[str, Any],
    timing: dict[str, Any],
) -> None:
    result["results"]["torch_dlpack_consumer_available"] = bool(timing["available"])
    result["results"]["torch_dlpack_consumer_unavailable_reason"] = timing["unavailable_reason"]
    result["results"]["torch_version"] = timing.get("torch_version")
    result["results"]["torch_cuda_version"] = timing.get("torch_cuda_version")
    result["instrumentation"]["torch_dlpack_consumer"] = {
        "status": "available" if timing["available"] else "unavailable",
        "timing_boundary": (
            "PyTorch CUDA consumer through DeviceCommutationMatrix.__dlpack__(max_version=(1, 0))"
        ),
        "ownership": "DLPack capsule is single-consumer; Wolfgang owner is retained by deleter context",
    }
    if not timing["available"]:
        return

    timings = {
        "torch_dlpack_from_dlpack": timing["from_dlpack"],
        "torch_dlpack_sum_total": timing["sum_total"],
        "torch_dlpack_dense_to_host": timing["dense_to_host"],
    }
    for prefix, raw_timing in timings.items():
        result["results"][f"{prefix}_seconds"] = raw_timing["median"]
        result["results"][f"{prefix}_p10_seconds"] = raw_timing["p10"]
        result["results"][f"{prefix}_p90_seconds"] = raw_timing["p90"]
        result["results"][f"{prefix}_min_seconds"] = raw_timing["min"]
        result["results"][f"{prefix}_max_seconds"] = raw_timing["max"]


def add_private_fused_consumer_timing_fields(
    result: dict[str, Any],
    fused_timing: dict[str, Any],
) -> None:
    count_status = os.environ.get(
        "WOLFGANG_CUDA_CAMPAIGN7_COUNT_SPECIALIZATION_STATUS",
        CAMPAIGN7_COUNT_SPECIALIZATION_STATUS,
    )
    bitpacked_status = os.environ.get(
        "WOLFGANG_CUDA_CAMPAIGN7_BITPACKED_DECISION_STATUS",
        CAMPAIGN7_BITPACKED_DECISION_STATUS,
    )
    timings = {
        "cuda_fused_graph_csr": fused_timing["csr"],
        "cuda_fused_conflict_degrees": fused_timing["conflict_degrees"],
        "cuda_fused_grouping_summary": fused_timing["grouping_summary"],
    }
    for prefix, timing in timings.items():
        result["results"][f"{prefix}_seconds"] = timing["median"]
        result["results"][f"{prefix}_p10_seconds"] = timing["p10"]
        result["results"][f"{prefix}_p90_seconds"] = timing["p90"]
        result["results"][f"{prefix}_min_seconds"] = timing["min"]
        result["results"][f"{prefix}_max_seconds"] = timing["max"]

    result["results"]["cuda_fused_graph_csr_edge_count"] = int(fused_timing["edge_count"])
    result["results"]["cuda_fused_graph_csr_host_bytes"] = int(
        fused_timing["csr_report"]["output_sizes"]["host_bytes"]
    )
    result["results"]["cuda_fused_conflict_degrees_host_bytes"] = int(
        fused_timing["conflict_summary_to_host_bytes"]
    )
    result["results"]["count_specialization_status"] = count_status
    result["results"]["bitpacked_decision_status"] = bitpacked_status
    result["instrumentation"]["campaign7_fused_consumer"] = {
        "status": "private_benchmark_only",
        "private_hook": "wolfgang._wolfgang_core._benchmark_cuda_fused_commutation_consumer",
        "modes": [
            "csr_anticommutation_graph",
            "conflict_degrees",
            "grouping_summary",
        ],
        "timing_boundaries": [
            "cuda_fused_graph_csr",
            "cuda_fused_conflict_degrees",
            "cuda_fused_grouping_summary",
        ],
        "bitpacked_decision_status": result["results"]["bitpacked_decision_status"],
        "count_specialization_status": result["results"]["count_specialization_status"],
    }


def add_campaign8_device_resident_consumer_timing_fields(
    result: dict[str, Any],
    campaign8_timing: dict[str, Any],
    *,
    mode: str,
    build_info: dict[str, Any],
    cuda_status: dict[str, Any],
    git_revision: str,
) -> None:
    add_campaign8_row_schema_fields(
        result,
        mode=mode,
        build_info=build_info,
        cuda_status=cuda_status,
        git_revision=git_revision,
    )
    if mode == "portability_check":
        result["non_h100_portability_status"] = os.environ.get(
            "WOLFGANG_CUDA_NON_H100_PORTABILITY_STATUS",
            "not_run",
        )

    timing_aliases = {
        "dense_to_host": campaign8_timing["dense_to_host"],
        "count_commuting_axis_none": campaign8_timing["count_axis_none"],
        "count_commuting_axis_0": campaign8_timing["count_axis0"],
        "count_commuting_axis_1": campaign8_timing["count_axis1"],
        "campaign7_csr_graph_export": campaign8_timing["campaign7_csr"],
        "campaign8_device_resident_graph_compact": campaign8_timing["device_graph"],
        "campaign8_device_grouping_consumer": campaign8_timing["grouping"],
    }
    for prefix, timing in timing_aliases.items():
        result["results"][f"{prefix}_seconds"] = timing["median"]
        result["results"][f"{prefix}_p10_seconds"] = timing["p10"]
        result["results"][f"{prefix}_p90_seconds"] = timing["p90"]
        result["results"][f"{prefix}_min_seconds"] = timing["min"]
        result["results"][f"{prefix}_max_seconds"] = timing["max"]

    graph_report = campaign8_timing["device_graph_report"]
    grouping_report = campaign8_timing["grouping_report"]
    result["results"]["campaign8_device_resident_graph_edge_count"] = int(
        campaign8_timing["edge_count"]
    )
    result["results"]["campaign8_device_resident_graph_compact_host_bytes"] = int(
        graph_report["output_sizes"]["compact_host_bytes"]
    )
    result["results"]["campaign8_device_resident_graph_full_csr_host_bytes"] = int(
        graph_report["output_sizes"]["full_csr_host_bytes"]
    )
    result["results"]["campaign8_device_grouping_consumer_compact_host_bytes"] = int(
        grouping_report["output_sizes"]["compact_host_bytes"]
    )
    result["results"]["campaign8_device_resident_graph_validation_csr_status"] = (
        "not_run_high_scale_default"
    )
    result["results"]["dlpack_unavailable_reason"] = campaign8_timing["dlpack_report"][
        "unavailable_reason"
    ]
    result["results"]["stream_graph_unavailable_reason"] = campaign8_timing["stream_graph_report"][
        "unavailable_reason"
    ]
    result["results"]["csr_scatter_ab_unavailable_reason"] = campaign8_timing["scatter_report"][
        "unavailable_reason"
    ]
    if mode == "dlpack_consumer":
        result["unavailable_reason"] = result["results"]["dlpack_unavailable_reason"]
    elif mode == "stream_graph_probe":
        result["unavailable_reason"] = result["results"]["stream_graph_unavailable_reason"]
    elif mode == "csr_scatter_ab":
        result["unavailable_reason"] = result["results"]["csr_scatter_ab_unavailable_reason"]
    result["correctness_digest"] = dict(graph_report["correctness_digest"])
    result["instrumentation"]["campaign8_device_resident_consumer"] = {
        "status": "private_benchmark_only",
        "private_hook": "wolfgang._wolfgang_core._benchmark_cuda_device_resident_consumer",
        "mode": mode,
        "device_resident_graph_status": result["device_resident_graph_status"],
        "public_grouping_api_status": result["public_grouping_api_status"],
        "dlpack_interop_status": result["dlpack_interop_status"],
        "non_h100_portability_status": result["non_h100_portability_status"],
        "stream_graph_status": result["stream_graph_status"],
        "scatter_tuning_status": result["scatter_tuning_status"],
        "labels": list(timing_aliases),
        "dlpack_unavailable_reason": result["results"]["dlpack_unavailable_reason"],
        "stream_graph_unavailable_reason": result["results"]["stream_graph_unavailable_reason"],
        "csr_scatter_ab_unavailable_reason": result["results"]["csr_scatter_ab_unavailable_reason"],
    }


def run_simplify_case(
    *,
    warmup: int,
    repeat: int,
    profile: str,
    build_info: dict[str, Any],
    rng: np.random.Generator,
    seed: int,
) -> dict[str, Any]:
    if profile == "smoke":
        labels = ["XX", "ZI", "XX", "II", "YY", "YY", "IZ", "IZ"]
        op = PauliSum.from_labels(
            labels,
            [1.0 + 2.0j, -0.5, -0.25 + 0.5j, 1.0e-14, 2.0j, -2.0j, 3.0, -1.0],
        )
        metadata = {
            "num_qubits": op.num_qubits,
            "num_terms": op.num_terms,
            "term_weight_distribution": "explicit smoke fixture",
            "duplicate_rate": duplicate_rate(labels),
            "duplicate_pool_size": 5,
            "coefficient_dtype": "complex128",
            "random_seed": seed,
            "operator_construction_method": "explicit duplicate/tolerance fixture",
        }
    else:
        num_terms = 50000 if profile == "default" else 100000
        duplicate_pool_size = 1024
        op, metadata = generate_operator(
            rng=rng,
            num_qubits=16,
            num_terms=num_terms,
            term_weight=3,
            duplicate_pool_size=duplicate_pool_size,
        )
    expected = op.simplify()
    metadata = {**metadata, "survivor_count": expected.num_terms}
    device_op = op.to_device()
    assert_same_operator(device_op.simplify().to_host(), expected)

    return case_result(
        name="simplify_duplicate_pressure",
        dataset=metadata,
        cpu_fn=lambda: op.simplify(),
        cuda_transfer_fn=lambda: op.to_device().simplify().to_host(),
        cuda_resident_fn=lambda: device_op.simplify(),
        build_info=build_info,
        warmup=warmup,
        repeat=repeat,
    )


def run_expectation_case(
    *,
    warmup: int,
    repeat: int,
    profile: str,
    build_info: dict[str, Any],
    rng: np.random.Generator,
    seed: int,
) -> dict[str, Any]:
    if profile == "smoke":
        labels = ["ZI", "IZ", "XX", "YY", "XY"]
        op = PauliSum.from_labels(
            labels,
            [1.0, -0.5, 0.25, 0.75j, -0.125 + 0.5j],
        )
        raw = np.asarray([1.0 + 0.25j, -0.5j, 0.75, -0.125 + 0.5j], dtype=np.complex128)
        psi = raw / np.linalg.norm(raw)
        metadata = {
            "num_qubits": op.num_qubits,
            "num_terms": op.num_terms,
            "term_weight_distribution": "explicit mixed-Pauli smoke fixture",
            "duplicate_rate": duplicate_rate(labels),
            "coefficient_dtype": "complex128",
            "statevector_length": psi.size,
            "statevector_dtype": "complex128",
            "random_seed": seed,
            "statevector_random_seed": "explicit smoke fixture",
            "operator_construction_method": "explicit mixed-Pauli expectation fixture",
        }
    else:
        num_qubits = 12 if profile == "default" else 14
        num_terms = 2048 if profile == "default" else 4096
        op, metadata = generate_operator(
            rng=rng,
            num_qubits=num_qubits,
            num_terms=num_terms,
            term_weight=3,
        )
        psi = normalized_statevector(rng, num_qubits)
        metadata = {
            **metadata,
            "statevector_length": psi.size,
            "statevector_dtype": "complex128",
            "random_seed": seed,
            "statevector_random_seed": seed,
        }
    expected = op.expectation_statevector(psi)
    device_op = op.to_device()
    if not np.allclose(
        device_op.expectation_statevector(psi),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError("CUDA and CPU statevector expectation differ")

    resident_notes: list[str] = []
    try:
        import cupy

        device_psi = cupy.asarray(psi)
        cuda_resident_fn = lambda: device_op.expectation_statevector(device_psi)
        if not np.allclose(cuda_resident_fn(), expected, rtol=1.0e-12, atol=1.0e-12):
            raise RuntimeError("CUDA-array-interface and CPU statevector expectation differ")
    except ModuleNotFoundError:
        resident_notes.append(
            "CuPy is unavailable; device-resident timing keeps the operator resident "
            "but copies host psi"
        )
        cuda_resident_fn = lambda: device_op.expectation_statevector(psi)

    return case_result(
        name="statevector_expectation",
        dataset=metadata,
        cpu_fn=lambda: op.expectation_statevector(psi),
        cuda_transfer_fn=lambda: op.to_device().expectation_statevector(psi),
        cuda_resident_fn=cuda_resident_fn,
        build_info=build_info,
        warmup=warmup,
        repeat=repeat,
        notes=resident_notes,
    )


def run_commutation_case(
    *,
    warmup: int,
    repeat: int,
    profile: str,
    build_info: dict[str, Any],
    rng: np.random.Generator,
    seed: int,
) -> dict[str, Any]:
    if profile == "smoke":
        lhs_labels = ["XX", "ZI", "IZ", "XY"]
        rhs_labels = ["YY", "XI", "ZZ"]
        lhs = PauliSum.from_labels(lhs_labels, [1.0, 2.0, -1.0, 0.25j])
        rhs = PauliSum.from_labels(rhs_labels, [1.0, 1.0j, -0.5])
        metadata = {
            "lhs_terms": lhs.num_terms,
            "rhs_terms": rhs.num_terms,
            "num_qubits": lhs.num_qubits,
            "entries": lhs.num_terms * rhs.num_terms,
            "term_weight_distribution": "explicit smoke fixtures",
            "lhs_duplicate_rate": duplicate_rate(lhs_labels),
            "rhs_duplicate_rate": duplicate_rate(rhs_labels),
            "coefficient_dtype": "complex128",
            "random_seed": seed,
            "operator_construction_method": "explicit pairwise commutation fixture",
        }
    else:
        terms = 2048 if profile == "default" else 8192
        lhs, lhs_metadata = generate_operator(
            rng=rng,
            num_qubits=16,
            num_terms=terms,
            term_weight=3,
        )
        rhs, rhs_metadata = generate_operator(
            rng=rng,
            num_qubits=16,
            num_terms=terms,
            term_weight=3,
        )
        metadata = {
            "lhs_terms": lhs.num_terms,
            "rhs_terms": rhs.num_terms,
            "num_qubits": lhs.num_qubits,
            "entries": lhs.num_terms * rhs.num_terms,
            "term_weight_distribution": "fixed term_weight=3",
            "lhs_duplicate_rate": lhs_metadata["duplicate_rate"],
            "rhs_duplicate_rate": rhs_metadata["duplicate_rate"],
            "coefficient_dtype": "complex128",
            "random_seed": seed,
            "operator_construction_method": "deterministic weighted lhs/rhs labels",
        }
    max_entries = max(metadata["entries"], 100_000_000)
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
        "WOLFGANG_CUDA_BENCH_REUSE_COMMUTATION_DEVICE_OUTPUT"
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
    return result


def run_matmul_case(
    *,
    warmup: int,
    repeat: int,
    profile: str,
    build_info: dict[str, Any],
    rng: np.random.Generator,
    seed: int,
) -> dict[str, Any]:
    if profile == "smoke":
        lhs_labels = ["X", "Y", "Z"]
        rhs_labels = ["Y", "Z"]
        lhs = PauliSum.from_labels(lhs_labels, [2.0, -0.5j, 1.25])
        rhs = PauliSum.from_labels(rhs_labels, [3.0, 0.25j])
        metadata = {
            "lhs_terms": lhs.num_terms,
            "rhs_terms": rhs.num_terms,
            "intermediate_terms": lhs.num_terms * rhs.num_terms,
            "num_qubits": lhs.num_qubits,
            "simplify": True,
            "term_weight_distribution": "explicit one-qubit smoke fixtures",
            "lhs_duplicate_rate": duplicate_rate(lhs_labels),
            "rhs_duplicate_rate": duplicate_rate(rhs_labels),
            "coefficient_dtype": "complex128",
            "random_seed": seed,
            "operator_construction_method": "explicit matrix-product phase fixture",
        }
    else:
        terms = 256 if profile == "default" else 512
        lhs, lhs_metadata = generate_operator(
            rng=rng,
            num_qubits=12,
            num_terms=terms,
            term_weight=3,
        )
        rhs, rhs_metadata = generate_operator(
            rng=rng,
            num_qubits=12,
            num_terms=terms,
            term_weight=3,
        )
        metadata = {
            "lhs_terms": lhs.num_terms,
            "rhs_terms": rhs.num_terms,
            "intermediate_terms": lhs.num_terms * rhs.num_terms,
            "num_qubits": lhs.num_qubits,
            "simplify": True,
            "term_weight_distribution": "fixed term_weight=3",
            "lhs_duplicate_rate": lhs_metadata["duplicate_rate"],
            "rhs_duplicate_rate": rhs_metadata["duplicate_rate"],
            "coefficient_dtype": "complex128",
            "random_seed": seed,
            "operator_construction_method": "deterministic weighted lhs/rhs labels",
        }
    lhs_device = lhs.to_device()
    rhs_device = rhs.to_device()
    expected = lhs.matmul(rhs, simplify=True)
    metadata = {**metadata, "survivor_count": expected.num_terms}
    assert_same_operator(lhs_device.matmul(rhs_device, simplify=True).to_host(), expected)

    return case_result(
        name="matmul_product_generation_simplify",
        dataset=metadata,
        cpu_fn=lambda: lhs.matmul(rhs, simplify=True),
        cuda_transfer_fn=lambda: lhs.to_device().matmul(rhs.to_device(), simplify=True).to_host(),
        cuda_resident_fn=lambda: lhs_device.matmul(rhs_device, simplify=True),
        build_info=build_info,
        warmup=warmup,
        repeat=repeat,
    )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")
    if args.warmup is not None and args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")

    profile = "smoke" if args.smoke else args.profile
    warmup = 0 if profile == "smoke" and args.warmup is None else (args.warmup or 0)
    rng = np.random.default_rng(args.seed)
    build_info = core._build_info()
    cuda_status = core._cuda_status()
    report: dict[str, Any] = {
        "benchmark": "cuda_kernels",
        "git_commit": git_commit(),
        "command": command_string(),
        "environment": benchmark_environment(build_info, numpy_version=np.__version__),
        "fastpauli_version": wolfgang.__version__,
        "fastpauli_build_info": build_info,
        "cuda_status": cuda_status,
        "timing_policy": {
            "warmup": warmup,
            "repeat": args.repeat,
            "profile": profile,
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
        "cases": [],
    }

    if not cuda_status["built"] or not cuda_status["runtime_available"]:
        report["unavailable_reason"] = cuda_status["skip_reason"]
        return report

    report["cases"] = [
        run_simplify_case(
            warmup=warmup,
            repeat=args.repeat,
            profile=profile,
            build_info=build_info,
            rng=rng,
            seed=args.seed,
        ),
        run_expectation_case(
            warmup=warmup,
            repeat=args.repeat,
            profile=profile,
            build_info=build_info,
            rng=rng,
            seed=args.seed,
        ),
        run_commutation_case(
            warmup=warmup,
            repeat=args.repeat,
            profile=profile,
            build_info=build_info,
            rng=rng,
            seed=args.seed,
        ),
        run_matmul_case(
            warmup=warmup,
            repeat=args.repeat,
            profile=profile,
            build_info=build_info,
            rng=rng,
            seed=args.seed,
        ),
    ]
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--smoke", action="store_true", help="run tiny validation-sized cases")
    parser.add_argument(
        "--profile",
        choices=("default", "stress"),
        default="default",
        help="non-smoke dataset profile",
    )
    parser.add_argument("--repeat", type=int, default=5, help="timed repetitions per case")
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="untimed warmup repetitions (defaults to 0 for --smoke, 1 otherwise)",
    )
    parser.add_argument("--seed", type=int, default=9051, help="deterministic RNG seed")
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

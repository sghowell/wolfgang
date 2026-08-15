#!/usr/bin/env python3
"""Compare FastPauli hot paths with optional quantum-library baselines.

This harness is intentionally conservative: it runs FastPauli on every case,
uses Qiskit or OpenFermion only when installed, and records unavailable
competitors instead of silently dropping a comparison. Correctness checks run
against the competitor result whenever a competitor is available.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import statistics
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import fastpauli
import numpy as np
from fastpauli import PauliSum

try:
    from _benchmark_metadata import benchmark_environment, command_string, git_commit
except ModuleNotFoundError:
    from benchmarks._benchmark_metadata import (
        benchmark_environment,
        command_string,
        git_commit,
    )


PAULIS = np.asarray(["X", "Y", "Z"])
MATPLOTLIB_CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "fastpauli-matplotlib-cache"


def configure_optional_library_import_environment() -> None:
    # Cirq/OpenFermion can import Matplotlib, which writes cache metadata during
    # import. Pinning the cache to a known writable temp directory keeps the
    # benchmark's JSON stdout clean on locked-down benchmark hosts.
    MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))


def optional_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def first_optional_version(packages: list[str]) -> str | None:
    for package in packages:
        version = optional_version(package)
        if version is not None:
            return version
    return None


def optional_import_status(module_name: str, packages: list[str]) -> dict[str, Any]:
    version = first_optional_version(packages)
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        return {
            "available": False,
            "version": version,
            "import_error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": True,
        "version": version,
        "import_error": None,
    }


def import_cupy_module() -> tuple[Any | None, str | None]:
    try:
        return importlib.import_module("cupy"), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _cupy_compile_error_indicates_unsupported_cuda_architecture(
    exc: BaseException,
    *,
    compute_capability: tuple[int, int] | None,
) -> bool:
    compiler = getattr(exc, "__class__", type(exc))
    module_name = getattr(compiler, "__module__", "")
    name = getattr(compiler, "__name__", "")
    message = str(exc)
    if "gpu-architecture" not in message and "NVRTC_ERROR_INVALID_OPTION" not in message:
        return False
    if "cupy" not in module_name and name != "FakeCompileException":
        return False
    return compute_capability is not None


def _cuda_compute_capability(cuda_status: dict[str, Any]) -> tuple[int, int] | None:
    devices = cuda_status.get("devices")
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


def import_qiskit_quantum_info() -> Any | None:
    try:
        from qiskit import quantum_info
    except ImportError:
        return None
    return quantum_info


def import_openfermion_qubit_operator() -> Any | None:
    try:
        from openfermion.ops import QubitOperator
    except ImportError:
        return None
    return QubitOperator


def import_cuquantum_statevector_stack() -> tuple[Any | None, Any | None, str | None]:
    try:
        cuquantum = importlib.import_module("cuquantum")
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"
    try:
        return cuquantum.custatevec, cuquantum.cudaDataType, None
    except AttributeError as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def make_label(rng: np.random.Generator, num_qubits: int, term_weight: int) -> str:
    chars = ["I"] * num_qubits
    active_qubits = rng.choice(num_qubits, size=min(term_weight, num_qubits), replace=False)
    for qubit in active_qubits:
        chars[num_qubits - 1 - int(qubit)] = str(rng.choice(PAULIS))
    return "".join(chars)


def generate_labels(
    *,
    num_qubits: int,
    num_terms: int,
    term_weight: int,
    duplicate_pool: int,
    seed: int,
) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    pool_size = max(1, min(num_terms, duplicate_pool))
    pool = [make_label(rng, num_qubits, term_weight) for _ in range(pool_size)]
    labels = [pool[index % pool_size] for index in range(num_terms)]
    rng.shuffle(labels)
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    return labels, np.asarray(coeffs, dtype=np.complex128)


def duplicate_rate(labels: list[str]) -> float:
    if not labels:
        return 0.0
    return 1.0 - (len(set(labels)) / len(labels))


def packed_key(label: str) -> tuple[int, ...]:
    num_qubits = len(label)
    words = (num_qubits + 63) // 64
    x_words = [0] * words
    z_words = [0] * words
    for label_offset, pauli in enumerate(label):
        qubit = num_qubits - 1 - label_offset
        bit = 1 << (qubit % 64)
        word = qubit // 64
        if pauli in {"X", "Y"}:
            x_words[word] |= bit
        if pauli in {"Z", "Y"}:
            z_words[word] |= bit

    key: list[int] = []
    for word in range(words):
        key.extend((x_words[word], z_words[word]))
    return tuple(key)


def fastpauli_pairs(op: PauliSum) -> list[tuple[tuple[int, ...], complex]]:
    labels, coeffs = op.to_labels()
    return [
        (packed_key(label), complex(coeff))
        for label, coeff in zip(labels, coeffs, strict=True)
    ]


def assert_pairs_close(
    lhs: list[tuple[tuple[int, ...], complex]],
    rhs: list[tuple[tuple[int, ...], complex]],
) -> None:
    if len(lhs) != len(rhs):
        raise RuntimeError("competitor baseline produced a different term count")
    for (lhs_key, lhs_coeff), (rhs_key, rhs_coeff) in zip(lhs, rhs, strict=True):
        if lhs_key != rhs_key or not np.allclose(lhs_coeff, rhs_coeff, rtol=1.0e-11, atol=1.0e-11):
            raise RuntimeError("competitor baseline produced a different canonical operator")


def timed_call(fn: Callable[[], Any], *, warmup: int, repeat: int) -> tuple[Any, dict[str, float]]:
    result: Any = None
    for _ in range(warmup):
        result = fn()

    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        timings.append(time.perf_counter() - start)
    return result, {
        "median": statistics.median(timings),
        "min": min(timings),
        "max": max(timings),
    }


def normalized_statevector(rng: np.random.Generator, num_qubits: int) -> np.ndarray:
    values = rng.normal(size=1 << num_qubits) + 1j * rng.normal(size=1 << num_qubits)
    norm = np.linalg.norm(values)
    if norm == 0.0:
        raise RuntimeError("generated zero statevector")
    return np.asarray(values / norm, dtype=np.complex128)


def qiskit_simplify(op: Any, *, atol: float) -> Any:
    try:
        return op.simplify(atol=atol, rtol=0.0)
    except TypeError:
        return op.simplify(atol=atol)


def openfermion_from_labels(labels: list[str], coeffs: np.ndarray, qubit_operator_type: Any) -> Any:
    output = qubit_operator_type()
    for label, coeff in zip(labels, coeffs, strict=True):
        term: list[tuple[int, str]] = []
        num_qubits = len(label)
        for label_offset, pauli in enumerate(label):
            if pauli != "I":
                term.append((num_qubits - 1 - label_offset, pauli))
        output += qubit_operator_type(tuple(term), complex(coeff))
    return output


def openfermion_term_count(op: Any) -> int:
    return len(op.terms)


def run_simplify_case(args: argparse.Namespace, qiskit_quantum_info: Any | None) -> dict[str, Any]:
    num_qubits = 8 if args.smoke else args.num_qubits
    num_terms = 32 if args.smoke else args.num_terms
    labels, coeffs = generate_labels(
        num_qubits=num_qubits,
        num_terms=num_terms,
        term_weight=2 if args.smoke else args.term_weight,
        duplicate_pool=max(1, num_terms // 8),
        seed=args.seed,
    )
    fast_op = PauliSum.from_labels(labels, coeffs.tolist())
    fast_result, fast_timings = timed_call(
        lambda: fast_op.simplify(atol=args.atol, rtol=0.0),
        warmup=args.warmup,
        repeat=args.repeat,
    )

    competitor_name = "qiskit.SparsePauliOp.simplify"
    competitor_available = qiskit_quantum_info is not None
    competitor_correctness_checked = False
    competitor_timings: dict[str, float] | None = None
    if competitor_available:
        qiskit_op = qiskit_quantum_info.SparsePauliOp(labels, coeffs=coeffs)
        qiskit_result, competitor_timings = timed_call(
            lambda: qiskit_simplify(qiskit_op, atol=args.atol),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        qiskit_fast = PauliSum.from_qiskit(qiskit_result).simplify(atol=args.atol, rtol=0.0)
        assert_pairs_close(fastpauli_pairs(fast_result), fastpauli_pairs(qiskit_fast))
        competitor_correctness_checked = True

    return {
        "name": "simplify",
        "dataset": {
            "num_qubits": num_qubits,
            "num_terms": num_terms,
            "term_weight": 2 if args.smoke else args.term_weight,
            "duplicate_rate": duplicate_rate(labels),
            "coefficient_dtype": "complex128",
            "random_seed": args.seed,
            "competitor": competitor_name,
        },
        "results": {
            "fastpauli_scalar_seconds": fast_timings["median"],
            "fastpauli_scalar_min_seconds": fast_timings["min"],
            "competitor_available": competitor_available,
            "competitor_correctness_checked": competitor_correctness_checked,
            "competitor_seconds": None if competitor_timings is None else competitor_timings["median"],
            "competitor_min_seconds": None if competitor_timings is None else competitor_timings["min"],
        },
    }


def run_multiply_case(args: argparse.Namespace, qubit_operator_type: Any | None) -> dict[str, Any]:
    num_qubits = 8 if args.smoke else args.num_qubits
    lhs_terms = 8 if args.smoke else args.lhs_terms
    rhs_terms = 6 if args.smoke else args.rhs_terms
    lhs_labels, lhs_coeffs = generate_labels(
        num_qubits=num_qubits,
        num_terms=lhs_terms,
        term_weight=2 if args.smoke else args.term_weight,
        duplicate_pool=lhs_terms,
        seed=args.seed + 10,
    )
    rhs_labels, rhs_coeffs = generate_labels(
        num_qubits=num_qubits,
        num_terms=rhs_terms,
        term_weight=2 if args.smoke else args.term_weight,
        duplicate_pool=rhs_terms,
        seed=args.seed + 11,
    )
    lhs_fast = PauliSum.from_labels(lhs_labels, lhs_coeffs.tolist())
    rhs_fast = PauliSum.from_labels(rhs_labels, rhs_coeffs.tolist())
    fast_result, fast_timings = timed_call(
        lambda: lhs_fast.matmul(rhs_fast, simplify=True, max_intermediate_terms=args.max_intermediate_terms),
        warmup=args.warmup,
        repeat=args.repeat,
    )

    competitor_name = "openfermion.QubitOperator.__mul__"
    competitor_available = qubit_operator_type is not None
    competitor_correctness_checked = False
    competitor_timings: dict[str, float] | None = None
    competitor_lhs_terms: int | None = None
    competitor_rhs_terms: int | None = None
    competitor_intermediate_terms: int | None = None
    if competitor_available:
        lhs_of = openfermion_from_labels(lhs_labels, lhs_coeffs, qubit_operator_type)
        rhs_of = openfermion_from_labels(rhs_labels, rhs_coeffs, qubit_operator_type)
        competitor_lhs_terms = openfermion_term_count(lhs_of)
        competitor_rhs_terms = openfermion_term_count(rhs_of)
        competitor_intermediate_terms = competitor_lhs_terms * competitor_rhs_terms

        def multiply_openfermion() -> Any:
            result = lhs_of * rhs_of
            result.compress(abs_tol=args.atol)
            return result

        of_result, competitor_timings = timed_call(
            multiply_openfermion,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        fast_of = fast_result.to_openfermion()
        if set(fast_of.terms) != set(of_result.terms):
            raise RuntimeError("OpenFermion multiply baseline produced different terms")
        for term in fast_of.terms:
            if not np.allclose(fast_of.terms[term], of_result.terms[term], rtol=1.0e-11, atol=1.0e-11):
                raise RuntimeError("OpenFermion multiply baseline produced different coefficients")
        competitor_correctness_checked = True

    return {
        "name": "multiply",
        "dataset": {
            "num_qubits": num_qubits,
            "lhs_terms": lhs_terms,
            "rhs_terms": rhs_terms,
            "intermediate_terms": lhs_terms * rhs_terms,
            "max_intermediate_terms": args.max_intermediate_terms,
            "term_weight": 2 if args.smoke else args.term_weight,
            "lhs_duplicate_rate": duplicate_rate(lhs_labels),
            "rhs_duplicate_rate": duplicate_rate(rhs_labels),
            "competitor_lhs_terms": competitor_lhs_terms,
            "competitor_rhs_terms": competitor_rhs_terms,
            "competitor_intermediate_terms": competitor_intermediate_terms,
            "competitor_operand_semantics": (
                "OpenFermion QubitOperator canonicalizes duplicate keys during construction"
            ),
            "coefficient_dtype": "complex128",
            "random_seed": args.seed + 10,
            "competitor": competitor_name,
        },
        "results": {
            "fastpauli_scalar_seconds": fast_timings["median"],
            "fastpauli_scalar_min_seconds": fast_timings["min"],
            "competitor_available": competitor_available,
            "competitor_correctness_checked": competitor_correctness_checked,
            "competitor_seconds": None if competitor_timings is None else competitor_timings["median"],
            "competitor_min_seconds": None if competitor_timings is None else competitor_timings["min"],
        },
    }


def qiskit_group_term_count(groups: list[Any]) -> int:
    return sum(len(group) for group in groups)


def fastpauli_group_label_counts(groups: list[PauliSum]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for group in groups:
        group_labels, _ = group.to_labels()
        counts.update(group_labels)
    return counts


def qiskit_group_label_counts(groups: list[Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for group in groups:
        counts.update(label for label, _coeff in group.to_list())
    return counts


def run_qiskit_grouping_case(args: argparse.Namespace, qiskit_quantum_info: Any | None) -> dict[str, Any]:
    num_qubits = 8 if args.smoke else args.num_qubits
    num_terms = 16 if args.smoke else args.group_terms
    labels, coeffs = generate_labels(
        num_qubits=num_qubits,
        num_terms=num_terms,
        term_weight=2 if args.smoke else args.term_weight,
        duplicate_pool=num_terms,
        seed=args.seed + 20,
    )
    fast_op = PauliSum.from_labels(labels, coeffs.tolist())
    fast_result, fast_timings = timed_call(
        lambda: fast_op.group_commuting(mode="full"),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    input_label_counts = Counter(labels)
    if fastpauli_group_label_counts(fast_result) != input_label_counts:
        raise RuntimeError("FastPauli grouping did not preserve the input label multiset")

    competitor_name = "qiskit.SparsePauliOp.group_commuting"
    competitor_available = qiskit_quantum_info is not None
    competitor_correctness_checked = False
    competitor_timings: dict[str, float] | None = None
    if competitor_available:
        qiskit_op = qiskit_quantum_info.SparsePauliOp(labels, coeffs=coeffs)
        qiskit_result, competitor_timings = timed_call(
            lambda: qiskit_op.group_commuting(qubit_wise=False),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        if qiskit_group_term_count(qiskit_result) != num_terms:
            raise RuntimeError("Qiskit grouping baseline lost input terms")
        if qiskit_group_label_counts(qiskit_result) != input_label_counts:
            raise RuntimeError("Qiskit grouping baseline changed the input label multiset")
        competitor_correctness_checked = True

    return {
        "name": "qiskit_grouping",
        "dataset": {
            "num_qubits": num_qubits,
            "num_terms": num_terms,
            "term_weight": 2 if args.smoke else args.term_weight,
            "duplicate_rate": duplicate_rate(labels),
            "coefficient_dtype": "complex128",
            "random_seed": args.seed + 20,
            "grouping_mode": "full",
            "competitor": competitor_name,
        },
        "results": {
            "fastpauli_scalar_seconds": fast_timings["median"],
            "fastpauli_scalar_min_seconds": fast_timings["min"],
            "fastpauli_groups": len(fast_result),
            "competitor_available": competitor_available,
            "competitor_correctness_checked": competitor_correctness_checked,
            "competitor_seconds": None if competitor_timings is None else competitor_timings["median"],
            "competitor_min_seconds": None if competitor_timings is None else competitor_timings["min"],
        },
    }


def cuquantum_pauli_descriptors(labels: list[str], custatevec: Any) -> tuple[list[list[Any]], list[np.ndarray], list[int]]:
    pauli_operators: list[list[Any]] = []
    basis_bits: list[np.ndarray] = []
    basis_sizes: list[int] = []
    num_qubits = len(labels[0]) if labels else 0
    for label in labels:
        paulis: list[Any] = []
        bits: list[int] = []
        for label_offset, pauli in enumerate(label):
            if pauli == "I":
                continue
            paulis.append(getattr(custatevec.Pauli, pauli))
            bits.append(num_qubits - 1 - label_offset)
        pauli_operators.append(paulis)
        basis_bits.append(np.asarray(bits, dtype=np.int32))
        basis_sizes.append(len(bits))
    return pauli_operators, basis_bits, basis_sizes


def run_cuquantum_statevector_case(
    args: argparse.Namespace,
    cupy: Any | None,
    custatevec: Any | None,
    cuda_data_type: Any | None,
    import_error: str | None,
) -> dict[str, Any]:
    num_qubits = 3 if args.smoke else args.statevector_qubits
    num_terms = 8 if args.smoke else args.statevector_terms
    labels, coeffs = generate_labels(
        num_qubits=num_qubits,
        num_terms=num_terms,
        term_weight=2 if args.smoke else args.term_weight,
        duplicate_pool=num_terms,
        seed=args.seed + 30,
    )
    rng = np.random.default_rng(args.seed + 31)
    psi = normalized_statevector(rng, num_qubits)
    fast_op = PauliSum.from_labels(labels, coeffs.tolist())
    fast_result, fast_timings = timed_call(
        lambda: fast_op.expectation_statevector(psi),
        warmup=args.warmup,
        repeat=args.repeat,
    )

    cuda_status = fastpauli._fastpauli_core._cuda_status()
    fastpauli_cuda_available = bool(cuda_status.get("built") and cuda_status.get("runtime_available"))
    fastpauli_cuda_host_statevector_timings: dict[str, float] | None = None
    fastpauli_cuda_device_statevector_timings: dict[str, float] | None = None
    if fastpauli_cuda_available:
        device_op = fast_op.to_device()
        cuda_host_result, fastpauli_cuda_host_statevector_timings = timed_call(
            lambda: device_op.expectation_statevector(psi),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        if not np.allclose(cuda_host_result, fast_result, rtol=1.0e-11, atol=1.0e-11):
            raise RuntimeError("FastPauli CUDA statevector expectation disagrees with scalar CPU")
        if cupy is not None:
            fastpauli_device_psi = cupy.asarray(psi, dtype=cupy.complex128)
            cupy.cuda.Stream.null.synchronize()
            cuda_device_result, fastpauli_cuda_device_statevector_timings = timed_call(
                lambda: device_op.expectation_statevector(fastpauli_device_psi),
                warmup=args.warmup,
                repeat=args.repeat,
            )
            if not np.allclose(cuda_device_result, fast_result, rtol=1.0e-11, atol=1.0e-11):
                raise RuntimeError(
                    "FastPauli CUDA device-statevector expectation disagrees with scalar CPU"
                )

    competitor_name = "cuquantum.custatevec.compute_expectations_on_pauli_basis"
    competitor_available = cupy is not None and custatevec is not None and cuda_data_type is not None
    competitor_correctness_checked = False
    competitor_timings: dict[str, float] | None = None
    competitor_transfer_timings: dict[str, float] | None = None
    competitor_unavailable_reason = import_error

    if competitor_available:
        pauli_operators, basis_bits, basis_sizes = cuquantum_pauli_descriptors(labels, custatevec)
        expectation_values = np.empty(num_terms, dtype=np.float64)
        handle = None
        try:
            handle = custatevec.create()
            device_psi = cupy.asarray(psi, dtype=cupy.complex128)

            def compute_on_device_statevector() -> complex:
                custatevec.compute_expectations_on_pauli_basis(
                    handle,
                    device_psi.data.ptr,
                    cuda_data_type.CUDA_C_64F,
                    num_qubits,
                    expectation_values.ctypes.data,
                    pauli_operators,
                    len(pauli_operators),
                    basis_bits,
                    basis_sizes,
                )
                cupy.cuda.Stream.null.synchronize()
                return complex(np.dot(coeffs, expectation_values))

            def compute_with_statevector_transfer() -> complex:
                transfer_device_psi = cupy.asarray(psi, dtype=cupy.complex128)
                custatevec.compute_expectations_on_pauli_basis(
                    handle,
                    transfer_device_psi.data.ptr,
                    cuda_data_type.CUDA_C_64F,
                    num_qubits,
                    expectation_values.ctypes.data,
                    pauli_operators,
                    len(pauli_operators),
                    basis_bits,
                    basis_sizes,
                )
                cupy.cuda.Stream.null.synchronize()
                return complex(np.dot(coeffs, expectation_values))

            competitor_result, competitor_timings = timed_call(
                compute_on_device_statevector,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            _, competitor_transfer_timings = timed_call(
                compute_with_statevector_transfer,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            if not np.allclose(competitor_result, fast_result, rtol=1.0e-11, atol=1.0e-11):
                raise RuntimeError("cuStateVec statevector expectation disagrees with FastPauli")
            competitor_correctness_checked = True
            competitor_unavailable_reason = None
        except Exception as exc:
            competitor_available = False
            competitor_unavailable_reason = f"{type(exc).__name__}: {exc}"
            competitor_timings = None
            competitor_transfer_timings = None
        finally:
            if handle is not None:
                custatevec.destroy(handle)

    return {
        "name": "cuquantum_statevector_expectation",
        "dataset": {
            "num_qubits": num_qubits,
            "num_terms": num_terms,
            "term_weight": 2 if args.smoke else args.term_weight,
            "duplicate_rate": duplicate_rate(labels),
            "statevector_length": psi.size,
            "statevector_dtype": "complex128",
            "coefficient_dtype": "complex128",
            "random_seed": args.seed + 30,
            "statevector_random_seed": args.seed + 31,
            "competitor": competitor_name,
            "competitor_semantic_mapping": (
                "cuStateVec computes one real Pauli-string expectation per term on the "
                "same normalized statevector; the benchmark combines those values with "
                "FastPauli's complex128 coefficients on the host."
            ),
            "competitor_timing_boundary": (
                "device-resident timing reuses the device statevector, cuStateVec handle, "
                "Pauli descriptors, and host output array; transfer-inclusive timing "
                "copies the statevector before each call."
            ),
            "fastpauli_cuda_timing_boundary": (
                "device-resident timing uses FastPauli's CUDA-array-interface path with "
                "a reused CuPy statevector when CuPy is importable; operator-resident "
                "host-statevector timing keeps only the Pauli operator on device and "
                "copies psi inside each call."
            ),
        },
        "results": {
            "fastpauli_scalar_seconds": fast_timings["median"],
            "fastpauli_scalar_min_seconds": fast_timings["min"],
            "fastpauli_cuda_available": fastpauli_cuda_available,
            "fastpauli_cuda_device_statevector_available": (
                fastpauli_cuda_device_statevector_timings is not None
            ),
            "fastpauli_cuda_device_resident_seconds": (
                None
                if fastpauli_cuda_device_statevector_timings is None
                else fastpauli_cuda_device_statevector_timings["median"]
            ),
            "fastpauli_cuda_device_resident_min_seconds": (
                None
                if fastpauli_cuda_device_statevector_timings is None
                else fastpauli_cuda_device_statevector_timings["min"]
            ),
            "fastpauli_cuda_operator_resident_host_statevector_seconds": (
                None
                if fastpauli_cuda_host_statevector_timings is None
                else fastpauli_cuda_host_statevector_timings["median"]
            ),
            "fastpauli_cuda_operator_resident_host_statevector_min_seconds": (
                None
                if fastpauli_cuda_host_statevector_timings is None
                else fastpauli_cuda_host_statevector_timings["min"]
            ),
            "competitor_available": competitor_available,
            "competitor_unavailable_reason": competitor_unavailable_reason,
            "competitor_correctness_checked": competitor_correctness_checked,
            "competitor_seconds": None if competitor_timings is None else competitor_timings["median"],
            "competitor_min_seconds": None if competitor_timings is None else competitor_timings["min"],
            "competitor_transfer_inclusive_seconds": (
                None if competitor_transfer_timings is None else competitor_transfer_timings["median"]
            ),
            "competitor_transfer_inclusive_min_seconds": (
                None if competitor_transfer_timings is None else competitor_transfer_timings["min"]
            ),
        },
    }


def run_cupy_commutation_consumer_case(
    args: argparse.Namespace,
    cupy: Any | None,
    import_error: str | None,
) -> dict[str, Any]:
    num_qubits = 8 if args.smoke else args.num_qubits
    terms = 8 if args.smoke else min(args.lhs_terms, args.rhs_terms)
    lhs_labels, lhs_coeffs = generate_labels(
        num_qubits=num_qubits,
        num_terms=terms,
        term_weight=2 if args.smoke else args.term_weight,
        duplicate_pool=terms,
        seed=args.seed + 40,
    )
    rhs_labels, rhs_coeffs = generate_labels(
        num_qubits=num_qubits,
        num_terms=terms,
        term_weight=2 if args.smoke else args.term_weight,
        duplicate_pool=terms,
        seed=args.seed + 41,
    )
    lhs = PauliSum.from_labels(lhs_labels, lhs_coeffs.tolist())
    rhs = PauliSum.from_labels(rhs_labels, rhs_coeffs.tolist())
    expected, fast_timings = timed_call(
        lambda: lhs.commutes_with(rhs),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    expected_uint64 = np.asarray(expected, dtype=np.uint64)

    cuda_status = fastpauli._fastpauli_core._cuda_status()
    fastpauli_cuda_available = bool(cuda_status.get("built") and cuda_status.get("runtime_available"))
    fastpauli_count_timings: dict[str, float] | None = None
    competitor_available = fastpauli_cuda_available and cupy is not None
    competitor_correctness_checked = False
    competitor_timings: dict[str, float] | None = None
    competitor_dense_to_host_timings: dict[str, float] | None = None
    competitor_unavailable_reason = import_error
    if not fastpauli_cuda_available:
        competitor_unavailable_reason = str(cuda_status.get("skip_reason", "CUDA unavailable"))
    compute_capability = _cuda_compute_capability(cuda_status)

    if fastpauli_cuda_available:
        lhs_device = lhs.to_device()
        rhs_device = rhs.to_device()
        matrix = lhs_device.commutes_with_device(rhs_device)
        count_result, fastpauli_count_timings = timed_call(
            matrix.count_commuting,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        if count_result != int(expected_uint64.sum()):
            raise RuntimeError("FastPauli compact commutation count disagrees with scalar CPU")

        if cupy is not None:
            try:
                view = cupy.asarray(matrix)
            except Exception as exc:
                if _cupy_compile_error_indicates_unsupported_cuda_architecture(
                    exc,
                    compute_capability=compute_capability,
                ):
                    assert compute_capability is not None
                    competitor_available = False
                    competitor_unavailable_reason = (
                        "CuPy runtime does not support CUDA compute capability "
                        f"{compute_capability[0]}.{compute_capability[1]}: {type(exc).__name__}: {exc}"
                    )
                else:
                    raise
            else:

                def cupy_sum_total() -> int:
                    total = cupy.sum(view)
                    return int(total.get())

                def cupy_dense_to_host() -> np.ndarray:
                    return cupy.asnumpy(view)

                try:
                    cupy_total, competitor_timings = timed_call(
                        cupy_sum_total,
                        warmup=args.warmup,
                        repeat=args.repeat,
                    )
                    dense_host, competitor_dense_to_host_timings = timed_call(
                        cupy_dense_to_host,
                        warmup=args.warmup,
                        repeat=args.repeat,
                    )
                except Exception as exc:
                    if _cupy_compile_error_indicates_unsupported_cuda_architecture(
                        exc,
                        compute_capability=compute_capability,
                    ):
                        assert compute_capability is not None
                        competitor_available = False
                        competitor_timings = None
                        competitor_dense_to_host_timings = None
                        competitor_unavailable_reason = (
                            "CuPy runtime does not support CUDA compute capability "
                            f"{compute_capability[0]}.{compute_capability[1]}: {type(exc).__name__}: {exc}"
                        )
                    else:
                        raise
                else:
                    if cupy_total != int(expected_uint64.sum()):
                        raise RuntimeError("CuPy commutation consumer total disagrees with scalar CPU")
                    if not np.array_equal(dense_host.astype(np.bool_), np.asarray(expected, dtype=np.bool_)):
                        raise RuntimeError("CuPy commutation consumer dense copy disagrees with scalar CPU")
                    competitor_correctness_checked = True
                    competitor_unavailable_reason = None

    return {
        "name": "cupy_commutation_consumer",
        "dataset": {
            "num_qubits": num_qubits,
            "lhs_terms": terms,
            "rhs_terms": terms,
            "matrix_entries": terms * terms,
            "term_weight": 2 if args.smoke else args.term_weight,
            "lhs_duplicate_rate": duplicate_rate(lhs_labels),
            "rhs_duplicate_rate": duplicate_rate(rhs_labels),
            "coefficient_dtype": "complex128",
            "random_seed": args.seed + 40,
            "competitor": "cupy.sum(DeviceCommutationMatrix.__cuda_array_interface__)",
            "competitor_semantic_mapping": (
                "CuPy consumes FastPauli's dense uint8 DeviceCommutationMatrix through "
                "the CUDA Array Interface and reduces commuting flags on the GPU."
            ),
            "competitor_timing_boundary": (
                "CuPy timing starts from an already populated DeviceCommutationMatrix; "
                "it excludes FastPauli commutation fill and includes CuPy reduction work."
            ),
        },
        "results": {
            "fastpauli_scalar_seconds": fast_timings["median"],
            "fastpauli_scalar_min_seconds": fast_timings["min"],
            "fastpauli_cuda_available": fastpauli_cuda_available,
            "fastpauli_cuda_compact_count_seconds": (
                None if fastpauli_count_timings is None else fastpauli_count_timings["median"]
            ),
            "fastpauli_cuda_compact_count_min_seconds": (
                None if fastpauli_count_timings is None else fastpauli_count_timings["min"]
            ),
            "competitor_available": competitor_available,
            "competitor_unavailable_reason": competitor_unavailable_reason,
            "competitor_correctness_checked": competitor_correctness_checked,
            "competitor_seconds": None if competitor_timings is None else competitor_timings["median"],
            "competitor_min_seconds": None if competitor_timings is None else competitor_timings["min"],
            "competitor_dense_to_host_seconds": (
                None
                if competitor_dense_to_host_timings is None
                else competitor_dense_to_host_timings["median"]
            ),
            "competitor_dense_to_host_min_seconds": (
                None
                if competitor_dense_to_host_timings is None
                else competitor_dense_to_host_timings["min"]
            ),
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.smoke:
        args.warmup = 0

    configure_optional_library_import_environment()
    qiskit_quantum_info = import_qiskit_quantum_info()
    qubit_operator_type = import_openfermion_qubit_operator()
    cupy, cupy_import_error = import_cupy_module()
    custatevec, cuda_data_type, cuquantum_import_error = import_cuquantum_statevector_stack()
    cuquantum_case_import_error = cupy_import_error if cupy is None else cuquantum_import_error
    cudaq_status = optional_import_status("cudaq", ["cudaq"])
    qiskit_aer_status = optional_import_status("qiskit_aer", ["qiskit-aer-gpu", "qiskit-aer"])
    build_info = fastpauli._fastpauli_core._build_info()
    cases = [
        run_simplify_case(args, qiskit_quantum_info),
        run_multiply_case(args, qubit_operator_type),
        run_qiskit_grouping_case(args, qiskit_quantum_info),
        run_cuquantum_statevector_case(
            args,
            cupy,
            custatevec,
            cuda_data_type,
            cuquantum_case_import_error,
        ),
        run_cupy_commutation_consumer_case(args, cupy, cupy_import_error),
    ]
    competitor_correctness_checked = any(
        bool(case["results"]["competitor_correctness_checked"]) for case in cases
    )
    return {
        "benchmark": "competitive_baselines",
        "git_commit": git_commit(),
        "command": command_string(),
        "environment": benchmark_environment(build_info, numpy_version=np.__version__),
        "fastpauli_version": fastpauli.__version__,
        "fastpauli_build_info": build_info,
        "competitors": {
            "qiskit": {
                "available": qiskit_quantum_info is not None,
                "version": optional_version("qiskit"),
            },
            "openfermion": {
                "available": qubit_operator_type is not None,
                "version": optional_version("openfermion"),
            },
            "cupy": {
                "available": cupy is not None,
                "version": optional_version("cupy") or optional_version("cupy-cuda12x"),
            },
            "cuquantum": {
                "available": custatevec is not None,
                "version": optional_version("cuquantum-python-cu12") or optional_version("cuquantum"),
                "component": "custatevec",
                "import_error": cuquantum_import_error,
            },
            "cudaq": {
                **cudaq_status,
                "benchmark_status": (
                    "installed/importable but no primitive-equivalent sparse-Pauli baseline retained"
                    if cudaq_status["available"]
                    else "not importable in this environment"
                ),
            },
            "qiskit_aer": {
                **qiskit_aer_status,
                "benchmark_status": (
                    "available for future framework-level GPU baselines"
                    if qiskit_aer_status["available"]
                    else "not importable in this environment"
                ),
            },
        },
        "timing_policy": {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "summary": "median seconds",
        },
        "correctness_checks": {
            "enabled": competitor_correctness_checked,
            "fastpauli_cases_executed": True,
            "competitor_correctness_checked": competitor_correctness_checked,
            "reference": "Qiskit/OpenFermion results when the optional competitor is installed",
            "failure_mode": (
                "raises RuntimeError if an available competitor disagrees semantically; "
                "reports competitor_correctness_checked=false when optional libraries are absent"
            ),
        },
        "baselines": [
            "FastPauli scalar CPU",
            "FastPauli CUDA device-resident when CUDA and CUDA-array-interface input are available",
            "FastPauli CUDA operator-resident host-statevector path when built and runtime-available",
            "Qiskit SparsePauliOp when installed",
            "OpenFermion QubitOperator when installed",
            "NVIDIA cuStateVec Pauli-basis statevector expectation when installed",
            "CuPy CUDA Array Interface commutation consumer when installed",
        ],
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--smoke", action="store_true", help="run tiny validation-sized cases")
    parser.add_argument("--repeat", type=int, default=3, help="timed repetitions per case")
    parser.add_argument("--warmup", type=int, default=1, help="untimed warmup repetitions")
    parser.add_argument("--seed", type=int, default=5153)
    parser.add_argument("--num-qubits", type=int, default=64)
    parser.add_argument("--num-terms", type=int, default=10_000)
    parser.add_argument("--lhs-terms", type=int, default=256)
    parser.add_argument("--rhs-terms", type=int, default=256)
    parser.add_argument("--group-terms", type=int, default=512)
    parser.add_argument("--statevector-qubits", type=int, default=14)
    parser.add_argument("--statevector-terms", type=int, default=4096)
    parser.add_argument("--term-weight", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--max-intermediate-terms", type=int, default=50_000_000)
    parser.add_argument("--output", type=Path, help="optional path for the emitted JSON report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.num_qubits < 1:
        raise SystemExit("--num-qubits must be positive")
    if min(args.num_terms, args.lhs_terms, args.rhs_terms, args.group_terms) < 1:
        raise SystemExit("--num-terms, --lhs-terms, --rhs-terms, and --group-terms must be positive")
    if min(args.statevector_qubits, args.statevector_terms) < 1:
        raise SystemExit("--statevector-qubits and --statevector-terms must be positive")
    if args.term_weight < 0:
        raise SystemExit("--term-weight must be non-negative")

    report = build_report(args)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Apple Metal Campaign 7 checked simplify primitive-stack behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import wolfgang_quantum
import wolfgang_quantum._wolfgang_core as core

ROOT = Path(__file__).resolve().parents[1]


def read(path: str | Path) -> str:
    return (ROOT / path if isinstance(path, str) else path).read_text(encoding="utf-8")


def _require_metal_runtime() -> None:
    status = core._metal_status()
    if not status["runtime_available"]:
        pytest.skip(status["skip_reason"])


def _assert_same_operator(lhs: wolfgang_quantum.PauliSum, rhs: wolfgang_quantum.PauliSum) -> None:
    lhs_labels, lhs_coeffs = lhs.to_labels()
    rhs_labels, rhs_coeffs = rhs.to_labels()
    assert lhs.num_qubits == rhs.num_qubits
    assert list(lhs_labels) == list(rhs_labels)
    np.testing.assert_allclose(
        [complex(value) for value in lhs_coeffs],
        [complex(value) for value in rhs_coeffs],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_campaign7_private_metal_source_layout_and_build_registration() -> None:
    required_paths = (
        "src/metal/simplify_metal.hpp",
        "src/metal/simplify_metal.mm",
        "src/metal/kernels/simplify.metal",
    )
    missing = [path for path in required_paths if not (ROOT / path).exists()]
    assert not missing, "missing Campaign 7 private Metal source files: " + ", ".join(missing)

    cmake = read("CMakeLists.txt")
    assert "src/metal/simplify_metal.mm" in cmake
    validate_source = read("scripts/validate.py")
    assert "src/metal/simplify_metal.hpp" in validate_source

    source = read("src/metal/simplify_metal.mm") + read("src/metal/kernels/simplify.metal")
    for token in (
        "fp_simplify_words1_init_keys",
        "fp_simplify_words1_bitonic_sort_step",
        "fp_simplify_words1_mark_heads",
        "fp_simplify_prefix_sum_step",
        "fp_simplify_words1_reduce_by_key",
        "fp_simplify_words1_compact_survivors",
    ):
        assert token in source

    assert "#include <Metal/Metal.h>" not in read("include/wolfgang/device_pauli_sum.hpp")


def test_campaign7_private_hook_reports_unavailable_for_multiword_when_available() -> None:
    if not hasattr(core, "_metal_simplify_words1_candidate_for_testing"):
        pytest.skip("Metal source build is not active")
    _require_metal_runtime()

    labels = ["X" + ("I" * 64), "X" + ("I" * 64)]
    device_op = wolfgang_quantum.PauliSum.from_labels(labels, [1.0, 2.0]).to_device(backend="metal")
    report = core._metal_simplify_words1_candidate_for_testing(device_op, include_output=False)

    assert report["status"] == "unavailable"
    assert report["metal_simplify_strategy_status"] == "unavailable"
    assert "one packed word" in report["skip_reason"]


def test_campaign7_words1_candidate_matches_cpu_when_available() -> None:
    if not hasattr(core, "_metal_simplify_words1_candidate_for_testing"):
        pytest.skip("Metal source build is not active")
    _require_metal_runtime()

    host = wolfgang_quantum.PauliSum.from_labels(
        ["X", "Z", "X", "Y", "Z", "I", "Y", "X"],
        [1.0, 2.0j, -0.25, 3.0, -2.0j, 0.0, -3.0, 0.5],
    )
    expected = host.simplify(atol=1.0e-12, rtol=0.0)
    report = core._metal_simplify_words1_candidate_for_testing(
        host.to_device(backend="metal"),
        atol=1.0e-12,
        rtol=0.0,
        include_output=True,
    )

    assert report["status"] == "ok"
    assert report["transfer_boundary"] == "device_resident"
    assert report["metal_simplify_strategy"] == "device_candidate"
    assert report["metal_simplify_strategy_status"] == "benchmark_only"
    assert report["primitive_stack"]["sort"] == "bitonic_sort_words1"
    assert report["primitive_stack"]["prefix_sum"] == "hillis_steele_inclusive_scan_uint32"
    assert report["primitive_stack"]["reduce_by_key"] == "head_parallel_duplicate_sum_words1"
    assert report["output_terms"] == expected.num_terms
    assert report["padded_terms"] >= host.num_terms
    assert report["bitonic_passes"] > 0
    assert report["prefix_sum_passes"] > 0

    device_output = report["device_output"]
    assert device_output.backend == "metal"
    assert device_output.num_terms == expected.num_terms
    _assert_same_operator(device_output.to_host(), expected)


def test_campaign7_words1_candidate_uses_complex_magnitude_tolerance_when_available() -> None:
    if not hasattr(core, "_metal_simplify_words1_candidate_for_testing"):
        pytest.skip("Metal source build is not active")
    _require_metal_runtime()

    host = wolfgang_quantum.PauliSum.from_labels(["X"], [0.5 + 0.5j])
    expected = host.simplify(atol=0.6, rtol=0.0)
    report = core._metal_simplify_words1_candidate_for_testing(
        host.to_device(backend="metal"),
        atol=0.6,
        rtol=0.0,
        include_output=True,
    )

    assert report["status"] == "ok"
    assert report["output_terms"] == expected.num_terms == 1
    _assert_same_operator(report["device_output"].to_host(), expected)


def test_campaign7_words1_candidate_rejects_non_dyadic_coefficients_when_available() -> None:
    if not hasattr(core, "_metal_simplify_words1_candidate_for_testing"):
        pytest.skip("Metal source build is not active")
    _require_metal_runtime()

    host = wolfgang_quantum.PauliSum.from_labels(["X", "X"], [0.1, 0.2])
    report = core._metal_simplify_words1_candidate_for_testing(
        host.to_device(backend="metal"),
        atol=1.0e-12,
        rtol=0.0,
        include_output=False,
    )

    assert report["status"] == "rejected_with_evidence"
    assert report["transfer_boundary"] == "status_only"
    assert report["metal_simplify_strategy_status"] == "rejected_with_evidence"
    assert "signed fixed32 dyadic" in report["skip_reason"]


def test_campaign7_words1_candidate_rejects_accumulator_overflow_domain_when_available() -> None:
    if not hasattr(core, "_metal_simplify_words1_candidate_for_testing"):
        pytest.skip("Metal source build is not active")
    _require_metal_runtime()

    host = wolfgang_quantum.PauliSum.from_labels(["X", "X"], [float(2**30), float(2**30)])
    report = core._metal_simplify_words1_candidate_for_testing(
        host.to_device(backend="metal"),
        atol=1.0e-12,
        rtol=0.0,
        include_output=False,
    )

    assert report["status"] == "rejected_with_evidence"
    assert report["transfer_boundary"] == "status_only"
    assert "fixed32 coefficient sum may overflow" in report["skip_reason"]


def test_campaign7_words1_candidate_rejects_unbounded_square_compare_when_available() -> None:
    if not hasattr(core, "_metal_simplify_words1_candidate_for_testing"):
        pytest.skip("Metal source build is not active")
    _require_metal_runtime()

    host = wolfgang_quantum.PauliSum.from_labels(["X"], [0.75 + 0.75j])
    report = core._metal_simplify_words1_candidate_for_testing(
        host.to_device(backend="metal"),
        atol=1.0,
        rtol=0.0,
        include_output=False,
    )

    assert report["status"] == "rejected_with_evidence"
    assert report["transfer_boundary"] == "status_only"
    assert "uint64 square comparison" in report["skip_reason"]

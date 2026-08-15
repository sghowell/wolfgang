"""Apple Metal Campaign 8 simplify timing-decomposition behavior."""

from __future__ import annotations

from pathlib import Path

import fastpauli
import fastpauli._fastpauli_core as core
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _require_metal_runtime() -> None:
    status = core._metal_status()
    if not status["runtime_available"]:
        pytest.skip(status["skip_reason"])


def _assert_same_operator(lhs: fastpauli.PauliSum, rhs: fastpauli.PauliSum) -> None:
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


def test_campaign8_words1_candidate_reports_timing_decomposition_when_available() -> None:
    if not hasattr(core, "_metal_simplify_words1_candidate_for_testing"):
        pytest.skip("Metal source build is not active")
    _require_metal_runtime()

    host = fastpauli.PauliSum.from_labels(
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
    assert report["campaign8_timing_schema"] == "checked_device_resident_simplify_v1"
    assert report["pipeline_cache"]["boundary"] == "prewarmed_static_pipeline_cache"
    assert report["pipeline_cache"]["library_source"] in {"runtime_source", "offline_metallib"}
    assert report["performance_decision"]["candidate_status"] in {
        "benchmark_only",
        "experimental",
        "performance_relevant",
    }

    timing = report["timing_decomposition_seconds"]
    required_timing_keys = {
        "host_preflight",
        "scratch_and_output_allocation",
        "command_encoding",
        "command_execution",
        "output_accounting",
        "total_observed",
    }
    assert set(timing) == required_timing_keys
    assert all(isinstance(timing[key], float) and timing[key] >= 0.0 for key in timing)
    assert timing["total_observed"] >= timing["command_execution"]

    dispatch_counts = report["dispatch_counts"]
    assert dispatch_counts["bitonic_sort"] == report["bitonic_passes"]
    assert dispatch_counts["prefix_sum"] == report["prefix_sum_passes"]
    assert dispatch_counts["total_kernel_dispatches"] >= (
        report["bitonic_passes"] + report["prefix_sum_passes"] + 5
    )

    _assert_same_operator(report["device_output"].to_host(), expected)


def test_campaign8_rejected_rows_keep_status_only_boundary_when_available() -> None:
    if not hasattr(core, "_metal_simplify_words1_candidate_for_testing"):
        pytest.skip("Metal source build is not active")
    _require_metal_runtime()

    host = fastpauli.PauliSum.from_labels(["X", "X"], [0.1, 0.2])
    report = core._metal_simplify_words1_candidate_for_testing(
        host.to_device(backend="metal"),
        atol=1.0e-12,
        rtol=0.0,
        include_output=False,
    )

    assert report["status"] == "rejected_with_evidence"
    assert report["transfer_boundary"] == "status_only"
    assert "timing_decomposition_seconds" not in report
    assert "dispatch_counts" not in report

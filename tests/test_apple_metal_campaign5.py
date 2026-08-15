"""Apple Metal Campaign 5 simplify semantics."""

from __future__ import annotations

import math

import fastpauli
import fastpauli._fastpauli_core as core
import numpy as np
import pytest


def _require_metal_runtime() -> dict:
    status = core._metal_status()
    if not status["runtime_available"]:
        pytest.skip(status["skip_reason"])
    return status


def _multiword_label(num_qubits: int, positions: dict[int, str]) -> str:
    chars = ["I"] * num_qubits
    for qubit, pauli in positions.items():
        chars[num_qubits - 1 - qubit] = pauli
    return "".join(chars)


def _labels_and_coeffs(op: fastpauli.PauliSum) -> tuple[list[str], list[complex]]:
    labels, coeffs = op.to_labels()
    return list(labels), [complex(value) for value in coeffs]


def _assert_same_operator(lhs: fastpauli.PauliSum, rhs: fastpauli.PauliSum) -> None:
    lhs_labels, lhs_coeffs = _labels_and_coeffs(lhs)
    rhs_labels, rhs_coeffs = _labels_and_coeffs(rhs)
    assert lhs.num_qubits == rhs.num_qubits
    assert lhs_labels == rhs_labels
    np.testing.assert_allclose(lhs_coeffs, rhs_coeffs, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize(
    ("labels", "coeffs", "num_qubits", "atol", "rtol"),
    [
        ([], [], 7, 1.0e-12, 0.0),
        (["XYZ"], [1.0 + 0.0j], 3, 1.0e-12, 0.0),
        (["XYZ", "XYZ", "III"], [1.0, 2.0, 0.0], 3, 1.0e-12, 0.0),
        (["XII", "XII", "ZII"], [1.0, -1.0, 0.25], 3, 1.0e-12, 0.0),
        (
            [
                _multiword_label(130, {0: "X", 65: "Z"}),
                _multiword_label(130, {0: "X", 65: "Z"}),
            ],
            [1.0, 3.0j],
            130,
            1.0e-12,
            0.0,
        ),
        (["XYZ", "XYZ"], [1.0e-8, -0.5e-8], 3, 1.0e-9, 0.5),
    ],
)
def test_metal_simplify_matches_cpu_when_available(
    labels: list[str],
    coeffs: list[complex],
    num_qubits: int,
    atol: float,
    rtol: float,
) -> None:
    _require_metal_runtime()

    host = (
        fastpauli.PauliSum.from_labels(labels, coeffs)
        if labels
        else fastpauli.PauliSum.empty(num_qubits=num_qubits)
    )
    expected = host.simplify(atol=atol, rtol=rtol)
    simplified_device = host.to_device(backend="metal").simplify(atol=atol, rtol=rtol)
    actual = simplified_device.to_host()

    assert simplified_device.backend == "metal"
    assert simplified_device.device == 0
    assert simplified_device.num_qubits == expected.num_qubits
    assert simplified_device.num_terms == expected.num_terms
    _assert_same_operator(actual, expected)


@pytest.mark.parametrize("bad_value", [-1.0, math.nan, math.inf])
def test_metal_simplify_rejects_invalid_tolerances_when_available(bad_value: float) -> None:
    _require_metal_runtime()

    device_op = fastpauli.PauliSum.from_labels(["X"], [1.0]).to_device(backend="metal")
    with pytest.raises(ValueError, match="simplify tolerances must be non-negative finite values"):
        device_op.simplify(atol=bad_value)
    with pytest.raises(ValueError, match="simplify tolerances must be non-negative finite values"):
        device_op.simplify(rtol=bad_value)

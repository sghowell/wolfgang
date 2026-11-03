"""Measurement planning: bases, diagonalizing Cliffords, counts estimation, shots.

Clifford results are checked against dense gate unitaries built here, so the
oracle shares no code with the tableau conjugation in wolfgang_quantum.measurement.
"""

from __future__ import annotations

import math
from functools import reduce

import numpy as np
import pytest
from wolfgang_quantum import PauliSum
from wolfgang_quantum.measurement import (
    allocate_shots,
    diagonalizing_clifford,
    qwc_basis,
    to_qiskit_circuit,
)

ONE_QUBIT = {
    "h": np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2),
    "s": np.diag([1, 1j]),
    "sdg": np.diag([1, -1j]),
}


def gate_unitary(name: str, qubits: tuple[int, ...], num_qubits: int) -> np.ndarray:
    dimension = 2**num_qubits
    if name in ONE_QUBIT:
        (q,) = qubits
        factors = [ONE_QUBIT[name] if qubit == q else np.eye(2) for qubit in reversed(range(num_qubits))]
        return reduce(np.kron, factors, np.eye(1))
    unitary = np.zeros((dimension, dimension), dtype=np.complex128)
    for basis in range(dimension):
        if name == "cx":
            control, target = qubits
            unitary[basis ^ (((basis >> control) & 1) << target), basis] = 1.0
        elif name == "cz":
            a, b = qubits
            unitary[basis, basis] = -1.0 if (basis >> a) & (basis >> b) & 1 else 1.0
    return unitary


def circuit_unitary(gates: list, num_qubits: int) -> np.ndarray:
    unitary = np.eye(2**num_qubits, dtype=np.complex128)
    for name, qubits in gates:
        unitary = gate_unitary(name, qubits, num_qubits) @ unitary
    return unitary


def commuting_group(rng: np.random.Generator, num_qubits: int, size: int) -> PauliSum:
    """Greedy random commuting set, filtered with the independent commutes_with kernel."""
    chosen: list[str] = []
    while len(chosen) < size:
        label = "".join(rng.choice(list("IXYZ"), size=num_qubits))
        if set(label) == {"I"} or label in chosen:
            continue
        candidate = PauliSum.from_labels([label], [1.0])
        if all(candidate.commutes_with(PauliSum.from_labels([other], [1.0])) for other in chosen):
            chosen.append(label)
    coeffs = rng.normal(size=size)
    return PauliSum.from_labels(chosen, coeffs)


# --- bases and counts ---------------------------------------------------------------


def test_qwc_basis_and_rejection() -> None:
    group = PauliSum.from_labels(["XIZ", "XYI", "IYZ"], [1.0, 2.0, 3.0])
    assert qwc_basis(group) == "XYZ"
    assert group.qwc_basis() == "XYZ"
    assert PauliSum.from_labels(["IZI"], [1.0]).qwc_basis() == "IZI"
    with pytest.raises(ValueError, match="qubit-wise"):
        PauliSum.from_labels(["XI", "ZI"], [1.0, 1.0]).qwc_basis()
    for group in PauliSum.from_labels(["XX", "YY", "ZZ", "XI", "IZ"], [1, 2, 3, 4, 5]).group_commuting():
        group.qwc_basis()  # every qwc group from the grouping kernel has a basis


def test_estimate_counts_matches_hand_computation() -> None:
    op = PauliSum.from_labels(["X"], [2.0])
    mean, error = op.estimate_counts({"0": 70, "1": 30}, basis="X")
    assert mean == pytest.approx(0.8)
    # Per-shot values +2 (70x) and -2 (30x): population variance 3.36 over 100 shots.
    assert error == pytest.approx(math.sqrt(3.36 * 100 / 99 / 100))


def test_estimate_counts_in_z_basis_matches_expectation_z_counts() -> None:
    op = PauliSum.from_labels(["ZZ", "IZ", "II"], [0.5, -1.0, 2.0])
    counts = {"00": 10, "01": 20, "10": 5, "11": 65}
    mean, _ = op.estimate_counts(counts)
    assert mean == pytest.approx(op.expectation_z_counts(counts))
    assert op.estimate_counts(counts, basis="ZZ")[0] == pytest.approx(mean)


def test_estimate_counts_validation() -> None:
    op = PauliSum.from_labels(["XZ"], [1.0])
    with pytest.raises(ValueError, match="basis letter"):
        op.estimate_counts({"00": 1}, basis="ZZ")
    with pytest.raises(ValueError, match="length"):
        op.estimate_counts({"00": 1}, basis="XZZ")
    with pytest.raises(ValueError, match="only X, Y, Z or I"):
        op.estimate_counts({"00": 1}, basis="XQ")
    with pytest.raises(TypeError):
        op.estimate_counts({"00": 1}, basis=3)  # type: ignore[arg-type]
    mean, error = op.estimate_counts({"01": 1}, basis="XZ")
    assert mean == -1.0 and math.isnan(error)


# --- diagonalizing Cliffords -------------------------------------------------------


def test_qwc_group_uses_only_single_qubit_gates() -> None:
    group = PauliSum.from_labels(["XIY", "XZI", "IZY"], [0.5, -1.0, 2.0])
    gates, diagonal = diagonalizing_clifford(group)
    assert {name for name, _ in gates} <= {"h", "sdg"}
    unitary = circuit_unitary(gates, 3)
    np.testing.assert_allclose(unitary @ group.to_matrix() @ unitary.conj().T, diagonal.to_matrix(), atol=1e-12)


@pytest.mark.parametrize("seed", range(12))
def test_full_commuting_groups_are_diagonalized(seed: int) -> None:
    rng = np.random.default_rng(seed)
    num_qubits = int(rng.integers(2, 6))
    group = commuting_group(rng, num_qubits, int(rng.integers(2, 2 * num_qubits + 1)))
    gates, diagonal = group.diagonalizing_clifford()
    assert {name for name, _ in gates} <= {"h", "s", "sdg", "cx", "cz"}
    x_words, _, _ = diagonal.to_arrays()
    assert not x_words.any()
    assert diagonal.num_terms == group.num_terms
    unitary = circuit_unitary(gates, num_qubits)
    np.testing.assert_allclose(
        unitary @ group.to_matrix() @ unitary.conj().T, diagonal.to_matrix(), atol=1e-12
    )


def test_bell_stabilizers_need_entangling_gates() -> None:
    group = PauliSum.from_labels(["XX", "ZZ", "YY"], [1.0, 1.0, 1.0])
    gates, diagonal = group.diagonalizing_clifford()
    assert any(name in ("cx", "cz") for name, _ in gates)
    unitary = circuit_unitary(gates, 2)
    np.testing.assert_allclose(unitary @ group.to_matrix() @ unitary.conj().T, diagonal.to_matrix(), atol=1e-12)


def test_non_commuting_terms_are_rejected() -> None:
    with pytest.raises(ValueError, match="commute"):
        diagonalizing_clifford(PauliSum.from_labels(["XX", "ZI"], [1.0, 1.0]))


def test_sampled_counts_estimate_matches_exact_expectation() -> None:
    rng = np.random.default_rng(2026)
    group = commuting_group(rng, 4, 6)
    psi = rng.normal(size=16) + 1j * rng.normal(size=16)
    psi /= np.linalg.norm(psi)
    exact = group.expectation_statevector(psi.astype(np.complex128))
    gates, diagonal = group.diagonalizing_clifford()
    rotated = circuit_unitary(gates, 4) @ psi
    probabilities = np.abs(rotated) ** 2
    shots = 200_000
    outcomes = rng.choice(16, size=shots, p=probabilities / probabilities.sum())
    values, frequencies = np.unique(outcomes, return_counts=True)
    counts = {format(int(v), "04b"): int(f) for v, f in zip(values, frequencies, strict=True)}
    mean, error = diagonal.estimate_counts(counts)
    assert abs(mean - exact) < 5 * error
    assert error < 0.05


def test_to_qiskit_circuit_matches_dense_unitary() -> None:
    qiskit_info = pytest.importorskip("qiskit.quantum_info")
    group = PauliSum.from_labels(["XX", "ZZ", "YY"], [1.0, 1.0, 1.0])
    gates, _ = group.diagonalizing_clifford()
    circuit = to_qiskit_circuit(gates, 2)
    np.testing.assert_allclose(qiskit_info.Operator(circuit).data, circuit_unitary(gates, 2), atol=1e-12)


# --- shot allocation ---------------------------------------------------------------


def test_allocate_shots_is_proportional_and_exact() -> None:
    groups = [
        PauliSum.from_labels(["ZI", "IZ"], [3.0, 1.0]),
        PauliSum.from_labels(["XX"], [-1.0]),
        PauliSum.from_labels(["II"], [5.0]),
    ]
    shots = allocate_shots(groups, 1000)
    assert sum(shots) == 1000
    assert shots[2] == 0
    assert shots[0] == pytest.approx(800, abs=2) and shots[1] == pytest.approx(200, abs=2)
    assert allocate_shots(groups, 1000, variances=[1.0, 9.0, 0.0])[1] == pytest.approx(750, abs=1)
    assert allocate_shots(groups, 2) == [1, 1, 0]
    assert allocate_shots(groups, 0) == [0, 0, 0]


def test_allocate_shots_validation() -> None:
    groups = [PauliSum.from_labels(["Z"], [1.0])]
    with pytest.raises(ValueError):
        allocate_shots(groups, -1)
    with pytest.raises(ValueError):
        allocate_shots(groups, 10, variances=[1.0, 2.0])
    with pytest.raises(ValueError):
        allocate_shots(groups, 10, variances=[-1.0])

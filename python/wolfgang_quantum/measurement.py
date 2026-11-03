"""Measurement planning for grouped Pauli operators.

``PauliSum.group_commuting`` splits an operator into groups whose terms can be
estimated from the same shots. This module turns a group into something a
device can run and turns the resulting counts back into an estimate:

``qwc_basis(group)``
    The per-qubit measurement basis of a qubit-wise commuting group.
``diagonalizing_clifford(group)``
    Clifford gates ``U`` and a Z-only operator ``D = U G U^dagger`` for any
    fully commuting group ``G``. Run ``U`` before a computational-basis
    measurement and estimate ``D`` from the counts.
``allocate_shots(groups, total_shots)``
    Integer shot budgets that minimize the variance of the summed estimate.
``to_qiskit_circuit(gates, num_qubits)``
    The gate list as a Qiskit ``QuantumCircuit`` (requires Qiskit).

Gates are ``(name, qubits)`` tuples with names ``h``, ``s``, ``sdg``, ``cx``
(control first) and ``cz``, listed in application order. Qubit indices follow
Wolfgang's convention: qubit 0 is the right-most label character, matching
Qiskit's ordering.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ._wolfgang_core import PauliSum

Gate = tuple[str, tuple[int, ...]]

__all__ = [
    "Gate",
    "allocate_shots",
    "diagonalizing_clifford",
    "qwc_basis",
    "to_qiskit_circuit",
]


def _term_bits(op: PauliSum) -> tuple[np.ndarray, np.ndarray]:
    """Unpack packed words into boolean (num_terms, num_qubits) x and z arrays."""
    x_words, z_words, _ = op.to_arrays()

    def unpack(words: np.ndarray) -> np.ndarray:
        as_bytes = np.ascontiguousarray(words, dtype="<u8").view(np.uint8)
        bits = np.unpackbits(as_bytes, axis=1, bitorder="little")
        return bits[:, : op.num_qubits].astype(bool)

    return unpack(x_words), unpack(z_words)


def _pack_words(bits: np.ndarray) -> np.ndarray:
    """Pack boolean (num_terms, num_qubits) bits into uint64 term-major words."""
    num_terms, num_qubits = bits.shape
    words = math.ceil(num_qubits / 64)
    padded = np.zeros((num_terms, words * 64), dtype=np.uint8)
    padded[:, :num_qubits] = bits
    packed = np.packbits(padded, axis=1, bitorder="little")
    return np.ascontiguousarray(packed).view("<u8").astype(np.uint64).reshape(num_terms, words)


def qwc_basis(op: PauliSum) -> str:
    """Return the measurement basis of a qubit-wise commuting operator.

    The result is a dense-label-order string (right-most character is qubit 0)
    of X, Y or Z for each qubit some term acts on, and I for qubits every term
    leaves alone. Pass it as ``basis`` to ``PauliSum.estimate_counts``. Raises
    ValueError when two terms act on a qubit with different non-identity Paulis.
    """
    x, z = _term_bits(op)
    support = x | z
    basis_x = x.any(axis=0)
    basis_z = z.any(axis=0)
    if not (np.array_equal(x, basis_x & support) and np.array_equal(z, basis_z & support)):
        raise ValueError("terms are not qubit-wise commuting; use diagonalizing_clifford")
    letters = np.array(["I", "X", "Z", "Y"])[basis_x.astype(int) + 2 * basis_z.astype(int)]
    return "".join(letters[::-1])


def _apply_gate(x: np.ndarray, z: np.ndarray, sign: np.ndarray, gate: Gate) -> None:
    """Conjugate every row P -> U P U^dagger in place, tracking the +/- sign.

    Rows are Hermitian Pauli strings; the update rules are the stabilizer
    tableau rules of Aaronson and Gottesman (2004).
    """
    name, qubits = gate
    if name == "h":
        (q,) = qubits
        sign ^= x[:, q] & z[:, q]
        x[:, q], z[:, q] = z[:, q].copy(), x[:, q].copy()
    elif name == "s":
        (q,) = qubits
        sign ^= x[:, q] & z[:, q]
        z[:, q] ^= x[:, q]
    elif name == "sdg":
        for _ in range(3):
            _apply_gate(x, z, sign, ("s", qubits))
    elif name == "cx":
        control, target = qubits
        sign ^= x[:, control] & z[:, target] & ~(x[:, target] ^ z[:, control])
        x[:, target] ^= x[:, control]
        z[:, control] ^= z[:, target]
    elif name == "cz":
        a, b = qubits
        for step in (("h", (b,)), ("cx", (a, b)), ("h", (b,))):
            _apply_gate(x, z, sign, step)
    else:
        raise ValueError(f"unknown gate {name!r}")


def _greedy_diagonalizing_gates(x: np.ndarray, z: np.ndarray) -> list[Gate]:
    """Map each independent term to a fresh single-qubit Z with H, S, CX and CZ.

    Terms are processed in order. A term's Z components on already-used qubits
    are products of earlier generators and are ignored; gates only touch unused
    qubits, so finished generators are never disturbed.
    """
    x = x.copy()
    z = z.copy()
    scratch_sign = np.zeros(x.shape[0], dtype=bool)
    used = np.zeros(x.shape[1], dtype=bool)
    gates: list[Gate] = []

    def apply(gate: Gate) -> None:
        gates.append(gate)
        _apply_gate(x, z, scratch_sign, gate)

    for term in range(x.shape[0]):

        def free_bits() -> tuple[np.ndarray, np.ndarray]:
            return x[term] & ~used, z[term] & ~used

        if (x[term] & used).any():
            raise ValueError("terms do not all commute; split them with group_commuting(mode='full')")
        free_x, free_z = free_bits()
        if not free_x.any() and not free_z.any():
            continue  # dependent on earlier terms
        if free_x.any():
            pivot = int(np.flatnonzero(free_x)[0])
            for other in np.flatnonzero(free_x):
                if other != pivot:
                    apply(("cx", (pivot, int(other))))
            _, free_z = free_bits()
            for other in np.flatnonzero(free_z):
                if other != pivot:
                    apply(("cz", (pivot, int(other))))
            if z[term, pivot]:
                apply(("s", (pivot,)))
            apply(("h", (pivot,)))
        else:
            pivot = int(np.flatnonzero(free_z)[0])
            for other in np.flatnonzero(free_z):
                if other != pivot:
                    apply(("cx", (int(other), pivot)))
        used[pivot] = True
    return gates


def diagonalizing_clifford(op: PauliSum) -> tuple[list[Gate], PauliSum]:
    """Return Clifford gates U and the Z-only operator D = U op U^dagger.

    Every term of ``op`` must commute with every other. Qubit-wise commuting
    operators get only single-qubit basis changes (``h``, or ``sdg`` then
    ``h``); other groups also use ``cx`` and ``cz``. For a state psi,
    ``<psi|op|psi> = <U psi|D|U psi>``, so run the gates, measure every qubit in
    Z, and call ``D.estimate_counts(counts)``. D keeps the term order of ``op``
    with each coefficient multiplied by the sign picked up under conjugation.
    Raises ValueError when the terms do not all commute.
    """
    num_qubits = op.num_qubits
    x, z = _term_bits(op)
    try:
        basis = qwc_basis(op)
    except ValueError:
        basis = None
    gates: list[Gate] = []
    if basis is not None:
        for qubit in range(num_qubits):
            letter = basis[num_qubits - 1 - qubit]
            if letter == "Y":
                gates.append(("sdg", (qubit,)))
            if letter in ("X", "Y"):
                gates.append(("h", (qubit,)))
    else:
        gates = _greedy_diagonalizing_gates(x, z)

    sign = np.zeros(x.shape[0], dtype=bool)
    x_out, z_out = x.copy(), z.copy()
    for gate in gates:
        _apply_gate(x_out, z_out, sign, gate)
    if x_out.any():
        raise ValueError("terms do not all commute; split them with group_commuting(mode='full')")
    coeffs = np.asarray(op.to_arrays()[2]) * np.where(sign, -1.0, 1.0)
    zero_words = np.zeros((x.shape[0], math.ceil(num_qubits / 64)), dtype=np.uint64)
    diagonal = PauliSum.from_arrays(zero_words, _pack_words(z_out), coeffs, num_qubits)
    return gates, diagonal


def allocate_shots(
    groups: Sequence[PauliSum],
    total_shots: int,
    *,
    variances: Sequence[float] | None = None,
) -> list[int]:
    """Split ``total_shots`` across groups to minimize the summed estimate's variance.

    Estimating group g with N_g shots contributes sigma_g^2 / N_g, which is
    minimized under sum N_g = total_shots by N_g proportional to sigma_g. With
    ``variances`` (per-shot variances, for example from a pilot run or
    ``variance_statevector``) sigma_g = sqrt(variance_g); otherwise the bound
    sigma_g <= sum |c| over the group's non-identity terms is used. Identity-only
    groups are deterministic and get zero shots. Counts are rounded by largest
    remainder, and every group with nonzero weight gets at least one shot when
    the budget allows.
    """
    if isinstance(total_shots, bool) or not isinstance(total_shots, (int, np.integer)) or total_shots < 0:
        raise ValueError("total_shots must be a non-negative integer")
    if variances is not None and len(variances) != len(groups):
        raise ValueError("variances must have one entry per group")

    weights = np.zeros(len(groups))
    for index, group in enumerate(groups):
        if variances is not None:
            variance = float(variances[index])
            if not math.isfinite(variance) or variance < 0.0:
                raise ValueError("variances must be finite and non-negative")
            weights[index] = math.sqrt(variance)
        else:
            x_words, z_words, coeffs = group.to_arrays()
            non_identity = (x_words | z_words).any(axis=1)
            weights[index] = float(np.abs(coeffs[non_identity]).sum())

    shots = np.zeros(len(groups), dtype=np.int64)
    active = np.flatnonzero(weights > 0.0)
    if active.size == 0 or total_shots == 0:
        return shots.tolist()
    budget = int(total_shots)
    if budget >= active.size:
        shots[active] = 1  # every measurable group needs at least one shot
        budget -= active.size
    ideal = weights[active] / weights[active].sum() * budget
    floor = np.floor(ideal).astype(np.int64)
    shots[active] += floor
    remainder = budget - int(floor.sum())
    order = np.argsort(-(ideal - floor), kind="stable")[:remainder]
    shots[active[order]] += 1
    return shots.tolist()


def to_qiskit_circuit(gates: Sequence[Gate], num_qubits: int) -> Any:
    """Return the gate list as a Qiskit ``QuantumCircuit`` without measurements."""
    try:
        from qiskit import QuantumCircuit
    except ImportError as error:  # pragma: no cover - exercised without Qiskit
        raise ImportError(
            "to_qiskit_circuit requires Qiskit; install it with "
            "`python -m pip install 'wolfgang-quantum[qiskit]'`"
        ) from error
    circuit = QuantumCircuit(num_qubits)
    for name, qubits in gates:
        if name not in ("h", "s", "sdg", "cx", "cz"):
            raise ValueError(f"unknown gate {name!r}")
        getattr(circuit, name)(*qubits)
    return circuit


def _install_pauli_sum_measurement_methods() -> None:
    PauliSum.qwc_basis = qwc_basis  # type: ignore[attr-defined]
    PauliSum.diagonalizing_clifford = diagonalizing_clifford  # type: ignore[attr-defined]

"""Optional Qiskit adapter for :class:`wolfgang_quantum.PauliSum`.

The adapter deliberately imports Qiskit only inside adapter calls. This keeps
the base Wolfgang import light and preserves Qiskit as an optional dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._wolfgang_core import PauliSum

QISKIT_INSTALL_HINT = "Install Wolfgang with the qiskit extra to use this adapter."

_PHASE_PREFIX_FACTORS: tuple[tuple[str, complex], ...] = (
    ("-i", -1.0j),
    ("+i", 1.0j),
    ("i", 1.0j),
    ("-", -1.0 + 0.0j),
    ("+", 1.0 + 0.0j),
)


def _import_qiskit_quantum_info() -> Any:
    try:
        from qiskit import quantum_info
    except ImportError as exc:
        raise ImportError(QISKIT_INSTALL_HINT) from exc
    return quantum_info


def _strip_qiskit_phase_prefix(label: str) -> tuple[str, complex]:
    """Return a zero-phase dense label and the coefficient factor it implies."""

    for prefix, factor in _PHASE_PREFIX_FACTORS:
        if label.startswith(prefix):
            return label[len(prefix) :], factor
    return label, 1.0 + 0.0j


def from_qiskit(op: Any) -> PauliSum:
    """Build a :class:`PauliSum` from a Qiskit ``SparsePauliOp``.

    Conversion uses Qiskit public Pauli labels and coefficients; it does not
    materialize dense matrices. Any explicit Pauli phase prefixes are folded
    into the Wolfgang coefficients before construction.
    """

    quantum_info = _import_qiskit_quantum_info()
    if not isinstance(op, quantum_info.SparsePauliOp):
        raise TypeError("PauliSum.from_qiskit expects qiskit.quantum_info.SparsePauliOp")

    labels = list(op.paulis.to_labels())
    coeffs = np.asarray(op.coeffs, dtype=np.complex128)
    if len(labels) != len(coeffs):
        raise ValueError("SparsePauliOp label and coefficient counts differ")
    if not labels:
        return PauliSum.empty(int(op.num_qubits))

    normalized_labels: list[str] = []
    normalized_coeffs = np.empty(len(coeffs), dtype=np.complex128)
    for index, (label, coeff) in enumerate(zip(labels, coeffs, strict=True)):
        normalized_label, phase_factor = _strip_qiskit_phase_prefix(str(label))
        normalized_labels.append(normalized_label)
        normalized_coeffs[index] = coeff * phase_factor

    return PauliSum.from_labels(normalized_labels, normalized_coeffs.tolist())


def to_qiskit(self: PauliSum) -> Any:
    """Export a :class:`PauliSum` as a Qiskit ``SparsePauliOp``.

    Exported Qiskit Pauli strings have zero explicit Pauli phase; all phase
    information is represented in the returned operator coefficients.
    """

    if not isinstance(self, PauliSum):
        raise TypeError("PauliSum.to_qiskit expects a PauliSum receiver")

    quantum_info = _import_qiskit_quantum_info()
    labels, coeffs = self.to_labels()
    coeff_array = np.asarray(coeffs, dtype=np.complex128)

    if self.num_terms == 0:
        paulis = quantum_info.PauliList.from_symplectic(
            np.zeros((0, self.num_qubits), dtype=bool),
            np.zeros((0, self.num_qubits), dtype=bool),
        )
        return quantum_info.SparsePauliOp(paulis, coeffs=coeff_array)

    return quantum_info.SparsePauliOp(labels, coeffs=coeff_array)


def _install_pauli_sum_qiskit_methods() -> None:
    PauliSum.from_qiskit = staticmethod(from_qiskit)  # type: ignore[attr-defined]
    PauliSum.to_qiskit = to_qiskit  # type: ignore[attr-defined]


__all__ = [
    "QISKIT_INSTALL_HINT",
    "from_qiskit",
    "to_qiskit",
]

"""Optional OpenFermion adapter for :class:`wolfgang_quantum.PauliSum`.

The adapter imports OpenFermion only inside adapter calls. This keeps the base
Wolfgang import light and preserves OpenFermion as an optional dependency.
"""

from __future__ import annotations

from typing import Any

from ._wolfgang_core import PauliSum

OPENFERMION_INSTALL_HINT = "Install Wolfgang with the openfermion extra to use this adapter."


def _import_qubit_operator_type() -> Any:
    try:
        from openfermion.ops.operators.qubit_operator import QubitOperator
    except ImportError as exc:
        raise ImportError(OPENFERMION_INSTALL_HINT) from exc
    return QubitOperator


def _checked_num_qubits(num_qubits: int | None) -> int | None:
    if num_qubits is None:
        return None
    if not isinstance(num_qubits, int):
        raise TypeError("num_qubits must be an integer or None")
    if num_qubits < 0:
        raise ValueError("num_qubits must be non-negative")
    return num_qubits


def _infer_num_qubits(terms: dict[tuple[tuple[int, str], ...], complex]) -> int:
    max_index = -1
    for term in terms:
        for qubit, _pauli in term:
            max_index = max(max_index, int(qubit))
    return max_index + 1 if max_index >= 0 else 0


def from_openfermion(op: Any, num_qubits: int | None = None) -> PauliSum:
    """Build a :class:`PauliSum` from an OpenFermion ``QubitOperator``.

    Conversion uses the sparse term dictionary exposed by OpenFermion and does
    not materialize dense matrices. When ``num_qubits`` is omitted, the width is
    inferred from the largest non-identity qubit index, with identity-only and
    zero operators inferred as width zero.
    """

    qubit_operator_type = _import_qubit_operator_type()
    if not isinstance(op, qubit_operator_type):
        raise TypeError("PauliSum.from_openfermion expects openfermion.ops.QubitOperator")

    explicit_num_qubits = _checked_num_qubits(num_qubits)
    width = explicit_num_qubits if explicit_num_qubits is not None else _infer_num_qubits(op.terms)

    if not op.terms:
        return PauliSum.empty(width)

    sparse_terms: list[tuple[str, list[int], complex]] = []
    for term, coeff in op.terms.items():
        local_paulis: list[str] = []
        qubit_indices: list[int] = []
        for qubit, pauli in term:
            qubit_index = int(qubit)
            if qubit_index < 0 or qubit_index >= width:
                raise ValueError("OpenFermion term qubit index is out of range for num_qubits")
            local_paulis.append(str(pauli))
            qubit_indices.append(qubit_index)
        sparse_terms.append(("".join(local_paulis), qubit_indices, complex(coeff)))

    return PauliSum.from_sparse_list(sparse_terms, width)


def to_openfermion(self: PauliSum) -> Any:
    """Export a :class:`PauliSum` as an OpenFermion ``QubitOperator``.

    Identity terms export as the OpenFermion identity term. Zero-term operators
    export as an additive zero ``QubitOperator``.
    """

    if not isinstance(self, PauliSum):
        raise TypeError("PauliSum.to_openfermion expects a PauliSum receiver")

    qubit_operator_type = _import_qubit_operator_type()
    output = qubit_operator_type()
    for local_paulis, qubit_indices, coeff in self.to_sparse_list():
        if local_paulis:
            term = tuple(
                (int(qubit), str(pauli))
                for pauli, qubit in zip(local_paulis, qubit_indices, strict=True)
            )
        else:
            term = ()
        output += qubit_operator_type(term, complex(coeff))
    return output


def _install_pauli_sum_openfermion_methods() -> None:
    PauliSum.from_openfermion = staticmethod(from_openfermion)  # type: ignore[attr-defined]
    PauliSum.to_openfermion = to_openfermion  # type: ignore[attr-defined]


__all__ = [
    "OPENFERMION_INSTALL_HINT",
    "from_openfermion",
    "to_openfermion",
]

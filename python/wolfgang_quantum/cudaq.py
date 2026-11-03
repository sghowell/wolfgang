"""Optional CUDA-Q adapter for :class:`wolfgang_quantum.PauliSum`.

The adapter imports CUDA-Q only inside adapter calls. This keeps the base
Wolfgang import light and preserves CUDA-Q as an optional dependency.

Both directions move a whole operator through ``SpinOperator.serialize()`` and
the ``SpinOperator(list[float])`` constructor, one flat list per operator
instead of one Python call per Pauli. Since CUDA-Q 0.11 the list is::

    [num_terms, (re, im, num_ops, (target, code) * num_ops) * num_terms]

with ``code = 2 * x + z`` (I = 0, Z = 1, X = 2, Y = 3). CUDA-Q does not promise
this layout, so the adapter checks it once per process against a known
operator and otherwise uses the slower public term API.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

from ._wolfgang_core import PauliSum

CUDAQ_INSTALL_HINT = "Install Wolfgang with the cudaq extra to use this adapter."

# 2.0 * X0 Z2 + 3j * Y1 - 1.5, serialized by CUDA-Q 0.11 through 0.16.
_PROBE_SERIALIZED = (
    3.0,
    2.0, 0.0, 2.0, 0.0, 2.0, 2.0, 1.0,
    0.0, 3.0, 1.0, 1.0, 3.0,
    -1.5, 0.0, 0.0,
)  # fmt: skip


def _import_cudaq() -> Any:
    try:
        import cudaq
    except ImportError as exc:
        raise ImportError(CUDAQ_INSTALL_HINT) from exc
    return cudaq


@functools.cache
def _serialized_format_supported() -> bool:
    """Return whether this CUDA-Q writes and reads the layout described above."""

    cudaq = _import_cudaq()
    spin = cudaq.spin
    try:
        probe = 2.0 * spin.x(0) * spin.z(2) + 3j * spin.y(1) - 1.5
        written = tuple(float(value) for value in probe.serialize())
        read = tuple(float(value) for value in cudaq.SpinOperator(list(_PROBE_SERIALIZED)).serialize())
    except Exception:  # noqa: BLE001 - any failure means the layout is not the one we parse
        return False
    return written == _PROBE_SERIALIZED and read == _PROBE_SERIALIZED


def _checked_num_qubits(num_qubits: int | None) -> int | None:
    if num_qubits is None:
        return None
    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int):
        raise TypeError("num_qubits must be an integer or None")
    if num_qubits < 0:
        raise ValueError("num_qubits must be non-negative")
    return num_qubits


def _parse_serialized(data: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a serialized operator into per-term coefficients and flat (term, target, code) ops."""

    values = np.asarray(data, dtype=np.float64)
    num_terms = int(data[0]) if data else 0
    starts = np.empty(num_terms, dtype=np.int64)
    counts = np.empty(num_terms, dtype=np.int64)
    position = 1
    for term in range(num_terms):
        starts[term] = position
        count = int(data[position + 2])
        counts[term] = count
        position += 3 + 2 * count
    if position != len(values):
        raise RuntimeError("CUDA-Q serialized the operator in an unexpected layout")
    coeffs = values[starts] + 1j * values[starts + 1]
    term_of_op = np.repeat(np.arange(num_terms, dtype=np.int64), counts)
    rank_in_term = np.arange(term_of_op.size, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
    op_positions = np.repeat(starts + 3, counts) + 2 * rank_in_term
    targets = values[op_positions]
    codes = values[op_positions + 1]
    if not (
        np.array_equal(targets, np.floor(targets))
        and np.all(targets >= 0)
        and np.all(np.isin(codes, (0.0, 1.0, 2.0, 3.0)))
    ):
        raise RuntimeError("CUDA-Q serialized the operator in an unexpected layout")
    return coeffs, term_of_op, targets.astype(np.int64), codes.astype(np.int64)


def _terms_from_api(op: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Same output as ``_parse_serialized``, through CUDA-Q's public term API."""

    coeffs: list[complex] = []
    term_of_op: list[int] = []
    targets: list[int] = []
    codes: list[int] = []
    for index, term in enumerate(op):
        coeffs.append(complex(term.evaluate_coefficient()))
        # x bits for qubits 0..max_degree, then z bits for the same qubits.
        # Explicit identities stay as code 0 so both paths infer one width.
        form = term.get_binary_symplectic_form()
        width = len(form) // 2
        mentioned = set(term.degrees)
        for qubit in range(width):
            code = 2 * int(form[qubit]) + int(form[width + qubit])
            if code or qubit in mentioned:
                term_of_op.append(index)
                targets.append(qubit)
                codes.append(code)
    return (
        np.asarray(coeffs, dtype=np.complex128),
        np.asarray(term_of_op, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
        np.asarray(codes, dtype=np.int64),
    )


def from_cudaq(op: Any, num_qubits: int | None = None) -> PauliSum:
    """Build a :class:`PauliSum` from a CUDA-Q ``SpinOperator`` or ``SpinOperatorTerm``.

    Terms keep CUDA-Q's order and coefficients exactly, including terms whose
    coefficient is zero. When ``num_qubits`` is omitted, the width is one more
    than the largest qubit index the operator mentions, including explicit
    identities such as ``spin.i(5)`` (zero for identity-only and empty
    operators). Parameterized coefficients raise ``ValueError``.
    """

    cudaq = _import_cudaq()
    if isinstance(op, cudaq.SpinOperatorTerm):
        op = cudaq.SpinOperator(op)
    elif not isinstance(op, cudaq.SpinOperator):
        raise TypeError("PauliSum.from_cudaq expects cudaq.SpinOperator or cudaq.SpinOperatorTerm")
    explicit_num_qubits = _checked_num_qubits(num_qubits)
    if op.parameters:
        raise ValueError("PauliSum.from_cudaq needs numeric coefficients; evaluate parameters first")

    if _serialized_format_supported():
        coeffs, term_of_op, targets, codes = _parse_serialized(list(op.serialize()))
    else:
        coeffs, term_of_op, targets, codes = _terms_from_api(op)

    inferred = int(targets.max()) + 1 if targets.size else 0
    width = explicit_num_qubits if explicit_num_qubits is not None else inferred
    if inferred > width:
        raise ValueError("CUDA-Q term qubit index is out of range for num_qubits")
    words = (width + 63) // 64
    x = np.zeros(coeffs.size * words, dtype=np.uint64)
    z = np.zeros(coeffs.size * words, dtype=np.uint64)
    flat_word = term_of_op * words + targets // 64
    bits = np.left_shift(np.uint64(1), (targets % 64).astype(np.uint64))
    np.bitwise_or.at(x, flat_word[(codes & 2) != 0], bits[(codes & 2) != 0])
    np.bitwise_or.at(z, flat_word[(codes & 1) != 0], bits[(codes & 1) != 0])
    shape = (coeffs.size, words)
    return PauliSum.from_arrays(x.reshape(shape), z.reshape(shape), coeffs, width)


def _serialize(op: PauliSum) -> list[float]:
    """Write ``op`` in CUDA-Q's serialized layout, targets ascending within each term."""

    x_words, z_words, coeffs = op.to_arrays()
    num_terms = op.num_terms
    if num_terms == 0:
        return [0.0]
    # Unpack each term's words to one x and one z bit per qubit (little-endian bit order).
    x_bytes = x_words.astype("<u8", copy=False).view(np.uint8)
    z_bytes = z_words.astype("<u8", copy=False).view(np.uint8)
    x_bits = np.unpackbits(x_bytes, axis=1, bitorder="little")[:, : op.num_qubits]
    z_bits = np.unpackbits(z_bytes, axis=1, bitorder="little")[:, : op.num_qubits]
    term_of_op, targets = np.nonzero(x_bits | z_bits)  # row-major: terms in order, targets ascending
    codes = 2 * x_bits[term_of_op, targets] + z_bits[term_of_op, targets]
    counts = np.bincount(term_of_op, minlength=num_terms)
    header = 1 + 3 * np.arange(num_terms, dtype=np.int64) + 2 * (np.cumsum(counts) - counts)
    out = np.empty(1 + 3 * num_terms + 2 * term_of_op.size, dtype=np.float64)
    out[0] = num_terms
    out[header] = coeffs.real
    out[header + 1] = coeffs.imag
    out[header + 2] = counts
    rank_in_term = np.arange(term_of_op.size, dtype=np.int64) - (np.cumsum(counts) - counts)[term_of_op]
    op_positions = header[term_of_op] + 3 + 2 * rank_in_term
    out[op_positions] = targets
    out[op_positions + 1] = codes
    return out.tolist()


def _build_with_api(cudaq: Any, op: PauliSum) -> Any:
    spin = cudaq.spin
    factories = {"X": spin.x, "Y": spin.y, "Z": spin.z}
    total = cudaq.SpinOperator.empty()
    for local_paulis, qubit_indices, coeff in op.to_sparse_list():
        term = complex(coeff) * cudaq.SpinOperator.identity()
        for pauli, qubit in zip(local_paulis, qubit_indices, strict=True):
            term = term * factories[pauli](int(qubit))
        total += term
    return total


def to_cudaq(self: PauliSum) -> Any:
    """Export a :class:`PauliSum` as a CUDA-Q ``SpinOperator``.

    CUDA-Q merges duplicate terms, keeping the first occurrence's position and
    summing coefficients in input order, the same sums as
    :meth:`PauliSum.simplify` with ``atol=0``. CUDA-Q operators carry no width,
    so ``PauliSum.from_cudaq(op.to_cudaq(), op.num_qubits)`` restores it.
    """

    if not isinstance(self, PauliSum):
        raise TypeError("PauliSum.to_cudaq expects a PauliSum receiver")
    cudaq = _import_cudaq()
    if self.num_terms == 0:
        return cudaq.SpinOperator.empty()
    if _serialized_format_supported():
        return cudaq.SpinOperator(_serialize(self))
    return _build_with_api(cudaq, self)


def _install_pauli_sum_cudaq_methods() -> None:
    PauliSum.from_cudaq = staticmethod(from_cudaq)  # type: ignore[attr-defined]
    PauliSum.to_cudaq = to_cudaq  # type: ignore[attr-defined]


__all__ = [
    "CUDAQ_INSTALL_HINT",
    "from_cudaq",
    "to_cudaq",
]

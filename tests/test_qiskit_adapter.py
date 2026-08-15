"""Phase 3 optional Qiskit adapter tests."""

from __future__ import annotations

import builtins
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
from wolfgang_quantum import PauliSum

QISKIT_INSTALL_HINT = "Install Wolfgang with the qiskit extra to use this adapter."


@pytest.fixture
def qiskit_quantum_info() -> Any:
    return pytest.importorskip("qiskit.quantum_info")


def empty_sparse_pauli_op(num_qubits: int, qiskit_quantum_info: Any) -> Any:
    pauli_list = qiskit_quantum_info.PauliList.from_symplectic(
        np.zeros((0, num_qubits), dtype=bool),
        np.zeros((0, num_qubits), dtype=bool),
    )
    return qiskit_quantum_info.SparsePauliOp(pauli_list, coeffs=[])


def random_labels(rng: np.random.Generator, num_terms: int, num_qubits: int) -> list[str]:
    alphabet = np.asarray(["I", "X", "Y", "Z"])
    return ["".join(rng.choice(alphabet, size=num_qubits).tolist()) for _ in range(num_terms)]


def assert_qiskit_matrices_close(lhs: Any, rhs: Any) -> None:
    np.testing.assert_allclose(lhs.to_matrix(), rhs.to_matrix(), rtol=1e-12, atol=1e-12)


def test_importing_fastpauli_does_not_import_qiskit() -> None:
    code = "import sys; import fastpauli; raise SystemExit('qiskit' in sys.modules)"

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("label", ["I", "X", "Y", "Z"])
def test_single_qubit_sparse_pauli_op_round_trip(label: str, qiskit_quantum_info: Any) -> None:
    source = qiskit_quantum_info.SparsePauliOp([label], coeffs=[1.25 - 0.5j])

    op = PauliSum.from_qiskit(source)
    exported = op.to_qiskit()

    labels, coeffs = op.to_labels()
    assert labels == [label]
    np.testing.assert_array_equal(coeffs, np.asarray([1.25 - 0.5j], dtype=np.complex128))
    assert exported.paulis.to_labels() == [label]
    np.testing.assert_array_equal(exported.coeffs, source.coeffs)
    assert_qiskit_matrices_close(exported, source)


def test_multi_qubit_endianness_round_trip(qiskit_quantum_info: Any) -> None:
    source = qiskit_quantum_info.SparsePauliOp(["XYZ", "IIZ"], coeffs=[2.0, -1.0j])

    op = PauliSum.from_qiskit(source)

    assert op.to_sparse_list() == [
        ("ZYX", [0, 1, 2], 2.0 + 0.0j),
        ("Z", [0], -1.0j),
    ]
    assert op.to_qiskit().paulis.to_labels() == ["XYZ", "IIZ"]
    assert_qiskit_matrices_close(op.to_qiskit(), source)


def test_identity_operator_round_trip(qiskit_quantum_info: Any) -> None:
    source = qiskit_quantum_info.SparsePauliOp(["III"], coeffs=[3.5 + 0.25j])

    op = PauliSum.from_qiskit(source)
    exported = op.to_qiskit()

    assert op.to_sparse_list() == [("", [], 3.5 + 0.25j)]
    assert exported.num_qubits == 3
    assert exported.paulis.to_labels() == ["III"]
    np.testing.assert_array_equal(exported.coeffs, source.coeffs)


def test_empty_operator_round_trip(qiskit_quantum_info: Any) -> None:
    source = empty_sparse_pauli_op(4, qiskit_quantum_info)

    op = PauliSum.from_qiskit(source)
    exported = op.to_qiskit()

    assert op.num_qubits == 4
    assert op.num_terms == 0
    assert exported.num_qubits == 4
    assert len(exported) == 0
    assert exported.paulis.to_labels() == []
    np.testing.assert_array_equal(exported.coeffs, np.asarray([], dtype=np.complex128))


@pytest.mark.parametrize(
    ("phase", "expected_factor"),
    [
        (0, 1.0 + 0.0j),
        (1, -1.0j),
        (2, -1.0 + 0.0j),
        (3, 1.0j),
    ],
)
def test_phased_pauli_inputs_fold_into_coefficients(
    phase: int,
    expected_factor: complex,
    qiskit_quantum_info: Any,
) -> None:
    x = np.asarray([[True, False, True]], dtype=bool)
    z = np.asarray([[False, True, True]], dtype=bool)
    paulis = qiskit_quantum_info.PauliList.from_symplectic(
        x,
        z,
        phase=np.asarray([phase], dtype=np.int8),
    )
    source = qiskit_quantum_info.SparsePauliOp(paulis, coeffs=[2.0])

    op = PauliSum.from_qiskit(source)
    exported = op.to_qiskit()

    labels, coeffs = op.to_labels()
    assert labels == ["YXZ"]
    np.testing.assert_allclose(coeffs, np.asarray([2.0 * expected_factor], dtype=np.complex128))
    assert exported.paulis.to_labels() == ["YXZ"]
    assert np.all(exported.paulis.phase == 0)
    assert_qiskit_matrices_close(exported, source)


def test_duplicate_terms_preserve_operator_semantics(qiskit_quantum_info: Any) -> None:
    source = qiskit_quantum_info.SparsePauliOp(
        ["XX", "XX", "IZ"],
        coeffs=[1.0, -0.25, 2.0j],
    )

    exported = PauliSum.from_qiskit(source).to_qiskit()

    assert exported.paulis.to_labels() == ["XX", "XX", "IZ"]
    np.testing.assert_array_equal(exported.coeffs, source.coeffs)
    assert_qiskit_matrices_close(exported, source)


@pytest.mark.parametrize("num_qubits", [1, 2, 4, 8])
def test_small_random_operators_match_dense_matrix_semantics(
    num_qubits: int,
    qiskit_quantum_info: Any,
) -> None:
    rng = np.random.default_rng(24601 + num_qubits)
    labels = random_labels(rng, num_terms=12, num_qubits=num_qubits)
    coeffs = rng.normal(size=12) + 1j * rng.normal(size=12)
    source = qiskit_quantum_info.SparsePauliOp(labels, coeffs=coeffs)

    exported = PauliSum.from_qiskit(source).to_qiskit()

    assert exported.num_qubits == source.num_qubits
    assert exported.paulis.to_labels() == source.paulis.to_labels()
    np.testing.assert_allclose(exported.coeffs, source.coeffs, rtol=0.0, atol=0.0)
    assert_qiskit_matrices_close(exported, source)


def test_to_qiskit_rejects_non_paulisum_receiver() -> None:
    with pytest.raises(TypeError):
        PauliSum.to_qiskit(object())  # type: ignore[misc]


def test_from_qiskit_rejects_wrong_input_type(qiskit_quantum_info: Any) -> None:
    with pytest.raises(TypeError, match="SparsePauliOp"):
        PauliSum.from_qiskit(qiskit_quantum_info.Pauli("X"))


def test_missing_qiskit_dependency_raises_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def fail_qiskit_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "qiskit" or name.startswith("qiskit."):
            raise ModuleNotFoundError("No module named 'qiskit'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_qiskit_import)

    with pytest.raises(ImportError, match=QISKIT_INSTALL_HINT):
        PauliSum.empty(1).to_qiskit()

    with pytest.raises(ImportError, match=QISKIT_INSTALL_HINT):
        PauliSum.from_qiskit(object())


def test_qiskit_phase_prefix_labels_are_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    import wolfgang_quantum.qiskit as adapter

    class FakePaulis:
        def to_labels(self) -> list[str]:
            return ["-iXYZ", "-III", "iIIZ"]

    class FakeSparsePauliOp:
        num_qubits = 3

        def __init__(self) -> None:
            self.paulis = FakePaulis()
            self.coeffs = np.asarray([2.0, 3.0, -4.0j], dtype=np.complex128)

    class FakeQuantumInfo:
        SparsePauliOp = FakeSparsePauliOp

    def import_fake_qiskit() -> FakeQuantumInfo:
        return FakeQuantumInfo()

    monkeypatch.setattr(adapter, "_import_qiskit_quantum_info", import_fake_qiskit)

    labels, coeffs = PauliSum.from_qiskit(FakeSparsePauliOp()).to_labels()

    assert labels == ["XYZ", "III", "IIZ"]
    np.testing.assert_array_equal(coeffs, np.asarray([-2.0j, -3.0, 4.0], dtype=np.complex128))

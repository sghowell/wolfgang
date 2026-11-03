"""Statevector tools against a Kronecker-product matrix oracle.

Dense labels list qubit n-1 first, so kron(label[0], ..., label[-1]) puts
qubit q on bit q of the basis index, the convention every tool documents.
"""

from __future__ import annotations

import sys
from functools import reduce

import numpy as np
import pytest
from wolfgang_quantum import PauliSum

PAULI = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def oracle_matrix(op: PauliSum) -> np.ndarray:
    labels, coeffs = op.to_labels()
    dimension = 2**op.num_qubits
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for label, coeff in zip(labels, coeffs, strict=True):
        matrix += coeff * reduce(np.kron, [PAULI[ch] for ch in label], np.eye(1))
    return matrix


def random_op(rng: np.random.Generator, num_qubits: int, num_terms: int) -> PauliSum:
    labels = ["".join(rng.choice(list("IXYZ"), size=num_qubits)) for _ in range(num_terms)]
    labels += labels[: num_terms // 3]  # duplicates exercise per-x-group merging
    coeffs = rng.normal(size=len(labels)) + 1j * rng.normal(size=len(labels))
    return PauliSum.from_labels(labels, coeffs)


def random_state(rng: np.random.Generator, num_qubits: int, batch: int | None = None) -> np.ndarray:
    shape = (2**num_qubits,) if batch is None else (batch, 2**num_qubits)
    psi = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return (psi / np.linalg.norm(psi, axis=-1, keepdims=True)).astype(np.complex128)


@pytest.mark.parametrize("num_qubits", [1, 2, 3, 5])
def test_dense_matrix_matches_oracle(num_qubits: int) -> None:
    op = random_op(np.random.default_rng(num_qubits), num_qubits, 9)
    np.testing.assert_allclose(op.to_matrix(), oracle_matrix(op), atol=1e-12)


def test_zero_qubit_and_empty_matrices() -> None:
    identity = PauliSum.from_sparse_list([("", [], 2.0 - 1.0j), ("", [], 0.5)], num_qubits=0)
    np.testing.assert_array_equal(identity.to_matrix(), [[2.5 - 1.0j]])
    assert not PauliSum.empty(2).to_matrix().any()
    assert PauliSum.empty(2).to_matrix().shape == (4, 4)


def test_dense_matrix_guard_rejects_before_allocating() -> None:
    op = PauliSum.from_labels(["X" * 13], [1.0])
    with pytest.raises(ValueError, match="max_entries"):
        op.to_matrix()
    assert op.to_matrix(max_entries=4**13).shape == (2**13, 2**13)
    with pytest.raises(ValueError, match="max_entries"):
        PauliSum.from_labels(["Z" * 40], [1.0]).to_matrix(max_entries=2**62)


@pytest.mark.parametrize("num_qubits", [1, 3, 6])
def test_sparse_matrix_matches_dense_with_sorted_indices(num_qubits: int) -> None:
    pytest.importorskip("scipy.sparse")
    op = random_op(np.random.default_rng(40 + num_qubits), num_qubits, 11)
    sparse = op.to_matrix(sparse=True)
    assert sparse.format == "csr" and sparse.shape == (2**num_qubits, 2**num_qubits)
    assert sparse.has_sorted_indices
    np.testing.assert_allclose(sparse.toarray(), op.to_matrix(), atol=1e-12)


def test_sparse_matrix_omits_exact_cancellation_and_guards_size() -> None:
    pytest.importorskip("scipy.sparse")
    # XX + YY maps |00> to |11> - |11> = 0, so half of each row cancels exactly.
    op = PauliSum.from_labels(["XX", "YY"], [1.0, 1.0])
    sparse = op.to_matrix(sparse=True)
    assert sparse.nnz == 2
    np.testing.assert_allclose(sparse.toarray(), oracle_matrix(op), atol=0)
    with pytest.raises(ValueError, match="max_entries"):
        PauliSum.from_labels(["X" * 20, "Z" * 20], [1.0, 1.0]).to_matrix(sparse=True, max_entries=2**20)


def test_sparse_matrix_reports_missing_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "scipy.sparse", None)
    with pytest.raises(ImportError, match=r"wolfgang-quantum\[scipy\]") as raised:
        PauliSum.from_labels(["X"], [1.0]).to_matrix(sparse=True)
    assert raised.value.__cause__ is not None


def test_sparse_guard_counts_row_pointers_for_empty_operators() -> None:
    with pytest.raises(ValueError, match="max_entries"):
        PauliSum.empty(30).to_matrix(sparse=True)
    pytest.importorskip("scipy.sparse")
    assert PauliSum.empty(3).to_matrix(sparse=True).nnz == 0


def test_importing_wolfgang_does_not_import_scipy() -> None:
    import subprocess

    code = "import sys, wolfgang_quantum; assert 'scipy' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("num_qubits", [0, 1, 4, 7])
def test_apply_matches_oracle_for_single_and_batched_states(num_qubits: int) -> None:
    rng = np.random.default_rng(70 + num_qubits)
    op = (
        PauliSum.from_sparse_list([("", [], 1.5j)], num_qubits=0)
        if num_qubits == 0
        else random_op(rng, num_qubits, 8)
    )
    matrix = oracle_matrix(op)
    psi = random_state(rng, num_qubits)
    np.testing.assert_allclose(op.apply_statevector(psi), matrix @ psi, atol=1e-12)
    batch = random_state(rng, num_qubits, batch=3)
    applied = op.apply_statevector(batch)
    assert applied.shape == batch.shape and applied.dtype == np.complex128
    np.testing.assert_allclose(applied, batch @ matrix.T, atol=1e-12)
    np.testing.assert_allclose(
        op.apply_statevector(psi.astype(np.complex64)), matrix @ psi, atol=1e-5
    )


def test_batched_expectation_and_variance_match_per_row_oracle() -> None:
    rng = np.random.default_rng(11)
    op = random_op(rng, 4, 10)
    hermitian = PauliSum.from_labels(["XZIY", "ZZII", "IIXX"], [0.5, -1.0, 2.0])
    batch = random_state(rng, 4, batch=5)
    values = op.expectation_statevector(batch)
    assert isinstance(values, np.ndarray) and values.shape == (5,)
    for row, value in zip(batch, values, strict=True):
        assert value == pytest.approx(op.expectation_statevector(row))
    matrix = oracle_matrix(hermitian)
    variances = hermitian.variance_statevector(batch)
    assert variances.shape == (5,) and variances.dtype == np.float64
    for row, variance in zip(batch, variances, strict=True):
        mean = np.vdot(row, matrix @ row)
        expected = np.vdot(matrix @ row, matrix @ row).real - abs(mean) ** 2
        assert variance == pytest.approx(expected, abs=1e-12)
        assert hermitian.variance_statevector(row) == pytest.approx(expected, abs=1e-12)
    assert isinstance(hermitian.variance_statevector(batch[0]), float)


def test_variance_uses_the_normalized_state() -> None:
    op = PauliSum.from_labels(["X", "Z"], [1.0, 0.5])
    psi = np.array([0.6, 0.8j], dtype=np.complex128)
    assert op.variance_statevector(3.0 * psi) == pytest.approx(op.variance_statevector(psi))
    with pytest.raises(ValueError, match="nonzero"):
        op.variance_statevector(np.zeros(2, dtype=np.complex128))


def test_variance_is_zero_in_an_eigenstate() -> None:
    op = PauliSum.from_labels(["ZZ", "ZI"], [1.0, 0.5])
    basis_state = np.zeros(4, dtype=np.complex128)
    basis_state[2] = 1.0
    assert op.variance_statevector(basis_state) == 0.0


@pytest.mark.parametrize("method", ["apply_statevector", "expectation_statevector", "variance_statevector"])
def test_statevector_input_validation(method: str) -> None:
    op = PauliSum.from_labels(["ZI"], [1.0])
    call = getattr(op, method)
    with pytest.raises(ValueError, match="1-dimensional"):
        call(np.ones((2, 2, 4), dtype=np.complex128))
    with pytest.raises(ValueError, match="2 \\*\\* num_qubits"):
        call(np.ones((2, 2), dtype=np.complex128))
    with pytest.raises(TypeError, match="complex64 or complex128"):
        call(np.ones(4, dtype=np.float64))
    with pytest.raises(TypeError, match="C-contiguous"):
        call(np.ones((4, 2), dtype=np.complex128)[:, 0])
    empty_batch = call(np.ones((0, 4), dtype=np.complex128))
    assert len(empty_batch) == 0
    with pytest.raises(ValueError, match="2 \\*\\* num_qubits"):
        call(np.ones((0, 7), dtype=np.complex128))
    with pytest.raises(ValueError, match="num_qubits <= 63"):
        getattr(PauliSum.from_labels(["Z" * 70], [1.0]), method)(np.ones((0, 4), dtype=np.complex128))

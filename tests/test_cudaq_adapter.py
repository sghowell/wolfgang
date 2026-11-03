"""Optional CUDA-Q adapter tests: PauliSum.from_cudaq and PauliSum.to_cudaq."""

from __future__ import annotations

import builtins
import subprocess
import sys
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import wolfgang_quantum
import wolfgang_quantum.cudaq as cudaq_adapter
from wolfgang_quantum import PauliSum

CUDAQ_INSTALL_HINT = "Install Wolfgang with the cudaq extra to use this adapter."


@pytest.fixture
def cudaq() -> Any:
    return pytest.importorskip("cudaq")


@pytest.fixture(params=["serialized", "term_api"])
def conversion_path(cudaq: Any, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run each conversion test on the fast serialized path and the term-API fallback."""
    if request.param == "term_api":
        monkeypatch.setattr(cudaq_adapter, "_serialized_format_supported", lambda: False)
    else:
        # A failed layout probe would silently run the fallback twice.
        assert cudaq_adapter._serialized_format_supported(), cudaq.__version__
    return str(request.param)


def _random_operator(num_qubits: int, num_terms: int, seed: int) -> PauliSum:
    rng = np.random.default_rng(seed)
    letters = np.asarray(list("IXYZ"), dtype="<U1")
    chars = letters[rng.integers(0, 4, size=(num_terms, num_qubits))]
    labels = np.ascontiguousarray(chars).view(f"<U{num_qubits}").ravel().tolist()
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    return PauliSum.from_labels(labels, coeffs)


def _cudaq_terms(op: Any) -> dict[str, complex]:
    return {term.get_pauli_word(op.qubit_count): term.evaluate_coefficient() for term in op}


def test_importing_wolfgang_does_not_import_cudaq() -> None:
    code = "import sys; import wolfgang_quantum; raise SystemExit('cudaq' in sys.modules)"
    completed = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_known_operator_converts_both_ways(cudaq: Any, conversion_path: str) -> None:
    spin = cudaq.spin
    native = 0.5 * spin.x(0) * spin.z(2) + 1j * spin.y(1) - 2.0
    op = PauliSum.from_cudaq(native)
    assert op.num_qubits == 3
    expected = PauliSum.from_sparse_list([("XZ", [0, 2], 0.5), ("Y", [1], 1j), ("", [], -2.0)], 3)
    assert op == expected
    back = op.to_cudaq()
    assert isinstance(back, cudaq.SpinOperator)
    assert _cudaq_terms(back) == _cudaq_terms(native)


@pytest.mark.parametrize("num_qubits", [1, 7, 64, 65, 130])
def test_random_operators_round_trip(cudaq: Any, conversion_path: str, num_qubits: int) -> None:
    op = _random_operator(num_qubits, 40, seed=num_qubits).simplify(atol=0.0)
    if num_qubits > 64:
        # Terms whose only Paulis sit in the second word exercise the sparse writer.
        qubits = sorted({64, num_qubits - 1})
        op = op + PauliSum.from_sparse_list([("Y" * len(qubits), qubits, 0.5j)], num_qubits)
    native = op.to_cudaq()
    assert native.term_count == op.num_terms
    rebuilt = PauliSum.from_cudaq(native, num_qubits=num_qubits)
    assert rebuilt == op  # term order and coefficients are preserved exactly


def test_term_and_single_pauli_inputs(cudaq: Any, conversion_path: str) -> None:
    spin = cudaq.spin
    assert PauliSum.from_cudaq(spin.z(5)) == PauliSum.from_sparse_list([("Z", [5], 1.0)], 6)
    term = 0.25j * spin.x(0) * spin.y(3)
    assert PauliSum.from_cudaq(term, num_qubits=4) == PauliSum.from_sparse_list([("XY", [0, 3], 0.25j)], 4)


def test_identity_empty_and_width(cudaq: Any, conversion_path: str) -> None:
    identity = PauliSum.from_cudaq(cudaq.SpinOperator.identity())
    assert identity.num_qubits == 0 and identity.num_terms == 1
    assert identity.to_arrays()[2].tolist() == [1.0]
    padded = PauliSum.from_cudaq(cudaq.SpinOperator.identity(), num_qubits=3)
    assert padded == PauliSum.from_labels(["III"], [1.0])
    empty = PauliSum.from_cudaq(cudaq.SpinOperator.empty(), num_qubits=4)
    assert empty.num_terms == 0 and empty.num_qubits == 4
    assert PauliSum.empty(5).to_cudaq().term_count == 0
    assert _cudaq_terms(PauliSum.from_labels(["II"], [3.0]).to_cudaq()) == {"": 3.0}


def test_duplicates_sum_in_input_order_like_simplify(cudaq: Any, conversion_path: str) -> None:
    op = PauliSum.from_labels(["XZ", "IY", "XZ", "XZ"], [0.1, 2.0, 0.2, 0.3])
    native = op.to_cudaq()
    assert native.term_count == 2
    # CUDA-Q keeps first-occurrence order and sums each duplicate in input
    # order, so after sorting the coefficients equal simplify's bit for bit.
    assert PauliSum.from_cudaq(native, num_qubits=2).simplify(atol=0.0) == op.simplify(atol=0.0)


def test_zero_coefficients_are_kept(cudaq: Any, conversion_path: str) -> None:
    spin = cudaq.spin
    native = spin.x(0) - spin.x(0) + spin.z(1)
    op = PauliSum.from_cudaq(native)
    assert op.num_terms == 2
    assert op.to_arrays()[2].tolist() == [0.0, 1.0]


def test_explicit_identities_set_the_inferred_width(cudaq: Any, conversion_path: str) -> None:
    spin = cudaq.spin
    padded = PauliSum.from_cudaq(spin.i(5))
    assert padded == PauliSum.from_labels(["IIIIII"], [1.0])
    mixed = PauliSum.from_cudaq(spin.i(5) * spin.x(1))
    assert mixed == PauliSum.from_sparse_list([("X", [1], 1.0)], 6)
    with pytest.raises(ValueError, match="out of range"):
        PauliSum.from_cudaq(spin.i(5) * spin.x(1), num_qubits=5)


def test_coefficients_are_read_exactly(cudaq: Any, conversion_path: str) -> None:
    # Identity-only terms reach CUDA-Q's list constructor without rescaling.
    infinite = PauliSum.from_cudaq(cudaq.SpinOperator([1.0, 1.0, np.inf, 0.0])).to_arrays()[2]
    assert infinite[0].real == 1.0 and np.isposinf(infinite[0].imag)
    signed = PauliSum.from_cudaq(cudaq.SpinOperator([1.0, 2.0, -0.0, 0.0])).to_arrays()[2]
    assert signed[0].real == 2.0 and np.signbit(signed[0].imag)


def test_parameterized_coefficients_raise(cudaq: Any, conversion_path: str) -> None:
    parameterized = cudaq.ScalarOperator(lambda t: t) * cudaq.spin.x(0)
    with pytest.raises(ValueError, match="numeric coefficients"):
        PauliSum.from_cudaq(parameterized)


def test_from_cudaq_validates_inputs(cudaq: Any, conversion_path: str) -> None:
    spin = cudaq.spin
    with pytest.raises(TypeError, match="SpinOperator"):
        PauliSum.from_cudaq(object())
    with pytest.raises(ValueError, match="out of range"):
        PauliSum.from_cudaq(spin.x(4), num_qubits=4)
    with pytest.raises(ValueError, match="non-negative"):
        PauliSum.from_cudaq(spin.x(0), num_qubits=-1)
    with pytest.raises(TypeError, match="integer"):
        PauliSum.from_cudaq(spin.x(0), num_qubits=1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integer"):
        PauliSum.from_cudaq(spin.x(0), num_qubits=True)
    assert PauliSum.from_cudaq(spin.x(0), num_qubits=np.int64(3)).num_qubits == 3


def test_to_cudaq_rejects_non_paulisum_receiver() -> None:
    with pytest.raises(TypeError):
        PauliSum.to_cudaq(object())  # type: ignore[misc]


@pytest.fixture
def double_precision_target(cudaq: Any) -> Iterator[Any]:
    # On a GPU host CUDA-Q defaults to the single-precision nvidia target.
    cudaq.set_target("qpp-cpu")
    yield cudaq
    cudaq.reset_target()


def test_expectations_match_cudaq_observe(double_precision_target: Any) -> None:
    cudaq = double_precision_target
    num_qubits = 5
    op = _random_operator(num_qubits, 30, seed=11)
    hermitian = (op + op.adjoint()).simplify()
    native = hermitian.to_cudaq()
    angles = np.random.default_rng(5).uniform(0, np.pi, size=2 * num_qubits).tolist()

    @cudaq.kernel
    def ansatz(theta: list[float]):
        qubits = cudaq.qvector(5)
        for i in range(5):
            ry(theta[i], qubits[i])  # noqa: F821 - CUDA-Q kernel builtin
        for i in range(4):
            x.ctrl(qubits[i], qubits[i + 1])  # noqa: F821
        for i in range(5):
            rz(theta[5 + i], qubits[i])  # noqa: F821

    state = np.asarray(cudaq.get_state(ansatz, angles), dtype=np.complex128)
    expected = cudaq.observe(ansatz, native, angles).expectation()
    got = hermitian.expectation_statevector(state)
    assert got.real == pytest.approx(expected, rel=1e-10, abs=1e-10)
    assert abs(got.imag) < 1e-10


def test_missing_cudaq_dependency_raises_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def fail_cudaq_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "cudaq" or name.startswith("cudaq."):
            raise ModuleNotFoundError("No module named 'cudaq'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_cudaq_import)
    with pytest.raises(ImportError, match=CUDAQ_INSTALL_HINT):
        PauliSum.empty(1).to_cudaq()
    with pytest.raises(ImportError, match=CUDAQ_INSTALL_HINT):
        PauliSum.from_cudaq(object())


def test_adapter_module_is_public() -> None:
    assert wolfgang_quantum.cudaq is cudaq_adapter
    assert cudaq_adapter.CUDAQ_INSTALL_HINT == CUDAQ_INSTALL_HINT

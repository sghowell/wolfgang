"""Basic package smoke tests for FastPauli."""

import importlib

import pytest


def test_fastpauli_imports() -> None:
    module = importlib.import_module("fastpauli")

    assert module.__name__ == "fastpauli"


def test_version_is_exposed() -> None:
    import wolfgang_quantum as fastpauli

    assert isinstance(fastpauli.__version__, str)
    assert fastpauli.__version__


def test_pauli_sum_type_is_exposed() -> None:
    import wolfgang_quantum as fastpauli

    assert isinstance(fastpauli.PauliSum, type)


def test_pauli_sum_metadata_constructor_properties() -> None:
    import wolfgang_quantum as fastpauli

    op = fastpauli.PauliSum(num_qubits=3, num_terms=0)

    assert op.num_qubits == 3
    assert op.num_terms == 0


def test_phase_six_public_apis_are_exposed() -> None:
    import wolfgang_quantum as fastpauli

    expected_apis = {
        "commutes_with",
        "empty",
        "expectation_statevector",
        "expectation_z_counts",
        "from_labels",
        "from_openfermion",
        "from_sparse_list",
        "group_commuting",
        "matmul",
        "simplify",
        "to_labels",
        "to_openfermion",
        "to_sparse_list",
    }

    for api_name in expected_apis:
        assert hasattr(fastpauli.PauliSum, api_name)


def test_pauli_sum_rejects_negative_metadata() -> None:
    import wolfgang_quantum as fastpauli

    with pytest.raises(ValueError):
        fastpauli.PauliSum(num_qubits=-1)

    with pytest.raises(ValueError):
        fastpauli.PauliSum(num_qubits=1, num_terms=-1)

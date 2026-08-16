"""Basic package smoke tests for Wolfgang."""

import pytest


def test_fastpauli_import_fails() -> None:
    with pytest.raises(ModuleNotFoundError):
        __import__("fastpauli")


def test_version_is_exposed() -> None:
    import wolfgang_quantum

    assert isinstance(wolfgang_quantum.__version__, str)
    assert wolfgang_quantum.__version__


def test_pauli_sum_type_is_exposed() -> None:
    import wolfgang_quantum

    assert isinstance(wolfgang_quantum.PauliSum, type)


def test_pauli_sum_metadata_constructor_properties() -> None:
    import wolfgang_quantum

    op = wolfgang_quantum.PauliSum(num_qubits=3, num_terms=0)

    assert op.num_qubits == 3
    assert op.num_terms == 0


def test_phase_six_public_apis_are_exposed() -> None:
    import wolfgang_quantum

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
        assert hasattr(wolfgang_quantum.PauliSum, api_name)


def test_pauli_sum_rejects_negative_metadata() -> None:
    import wolfgang_quantum

    with pytest.raises(ValueError):
        wolfgang_quantum.PauliSum(num_qubits=-1)

    with pytest.raises(ValueError):
        wolfgang_quantum.PauliSum(num_qubits=1, num_terms=-1)

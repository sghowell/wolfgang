"""Zero-copy device arrays: DevicePauliSum.device_arrays() and from_device_arrays().

On CUDA, device_arrays() returns read-only views of the x words, z words and
coefficients that CuPy (and PyTorch) consume through DLPack or the CUDA Array
Interface without a copy, and that keep the operator alive.
from_device_arrays() copies CuPy arrays into a new operator without a host
round trip. HIP and Metal raise until a validated consumer enforces the
read-only contract.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest
import wolfgang_quantum
import wolfgang_quantum._wolfgang_core as core


def _cuda_and_cupy():
    status = dict(core._cuda_status())
    if not status.get("built") or not status.get("runtime_available"):
        pytest.skip(str(status.get("skip_reason", "CUDA is unavailable")))
    return pytest.importorskip("cupy", reason="CuPy is required for CUDA interop tests")


def _operator(num_qubits: int, num_terms: int, seed: int) -> wolfgang_quantum.PauliSum:
    rng = np.random.default_rng(seed)
    letters = np.asarray(list("IXYZ"), dtype="<U1")
    chars = letters[rng.integers(0, 4, size=(num_terms, num_qubits))]
    labels = np.ascontiguousarray(chars).view(f"<U{num_qubits}").ravel().tolist()
    return wolfgang_quantum.PauliSum.from_labels(labels, rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms))


@pytest.mark.parametrize("num_qubits", [5, 64, 130])
def test_device_arrays_are_zero_copy_views_cupy_reads(num_qubits: int) -> None:
    cupy = _cuda_and_cupy()
    op = _operator(num_qubits, 300, seed=num_qubits)
    device = op.to_device(backend="cuda")
    x_view, z_view, coeffs_view = device.device_arrays()
    words = (num_qubits + 63) // 64
    assert x_view.shape == (300, words) and z_view.shape == (300, words)
    assert coeffs_view.shape == (300,)
    assert (x_view.dtype, coeffs_view.dtype) == ("uint64", "complex128")
    host_x, host_z, host_coeffs = op.to_arrays()
    for view, expected in ((x_view, host_x), (z_view, host_z), (coeffs_view, host_coeffs)):
        via_dlpack = cupy.from_dlpack(view)
        via_interface = cupy.asarray(view)
        np.testing.assert_array_equal(cupy.asnumpy(via_dlpack), expected)
        np.testing.assert_array_equal(cupy.asnumpy(via_interface), expected)
        # Zero copy: both consumers see the operator's own buffer.
        assert via_dlpack.data.ptr == via_interface.data.ptr == view.__cuda_array_interface__["data"][0]
    assert x_view.__dlpack_device__() == (2, 0)


def test_device_array_views_keep_the_operator_alive() -> None:
    cupy = _cuda_and_cupy()
    op = _operator(40, 200, seed=3)
    device = op.to_device(backend="cuda")
    coeffs = cupy.from_dlpack(device.device_arrays()[2])
    del device
    gc.collect()
    np.testing.assert_array_equal(cupy.asnumpy(coeffs), op.to_arrays()[2])
    del coeffs
    gc.collect()  # releasing the last consumer synchronizes and frees


@pytest.mark.parametrize("num_qubits", [3, 64, 100, 200])
def test_from_device_arrays_round_trips_without_host_copy(num_qubits: int) -> None:
    cupy = _cuda_and_cupy()
    op = _operator(num_qubits, 257, seed=num_qubits + 1)
    x, z, coeffs = (cupy.asarray(array) for array in op.to_arrays())
    rebuilt = wolfgang_quantum.DevicePauliSum.from_device_arrays(x, z, coeffs, num_qubits)
    assert rebuilt.backend == "cuda"
    assert rebuilt.to_host() == op
    # The result owns its buffers: changing the source does not change it.
    coeffs[:] = 0
    assert rebuilt.to_host() == op
    # Exported views feed straight back in.
    again = wolfgang_quantum.DevicePauliSum.from_device_arrays(*rebuilt.device_arrays(), num_qubits)
    assert again.to_host() == op


def test_from_device_arrays_validates_inputs() -> None:
    cupy = _cuda_and_cupy()
    op = _operator(40, 50, seed=8)
    x, z, coeffs = (cupy.asarray(array) for array in op.to_arrays())
    from_arrays = wolfgang_quantum.DevicePauliSum.from_device_arrays
    bad_x = x.copy()
    bad_x[3, 0] |= np.uint64(1) << np.uint64(50)  # a bit above num_qubits = 40
    with pytest.raises(ValueError, match="at or above num_qubits"):
        from_arrays(bad_x, z, coeffs, 40)
    with pytest.raises(TypeError, match="dtype"):
        from_arrays(x.astype(cupy.int64), z, coeffs, 40)
    with pytest.raises(TypeError, match="dtype"):
        from_arrays(x, z, coeffs.astype(cupy.complex64), 40)
    with pytest.raises(ValueError, match="shape"):
        from_arrays(x[:10], z, coeffs, 40)
    with pytest.raises(TypeError, match="C-contiguous"):
        from_arrays(cupy.concatenate([x, x], axis=1)[:, ::2], z, coeffs, 40)
    with pytest.raises(TypeError, match="__cuda_array_interface__"):
        from_arrays(op.to_arrays()[0], z, coeffs, 40)
    empty = from_arrays(x[:0], z[:0], coeffs[:0], 40)
    assert empty.num_terms == 0 and empty.num_qubits == 40


@pytest.mark.parametrize("backend", ["hip", "metal"])
def test_device_arrays_unavailable_on_hip_and_metal(backend: str) -> None:
    status = dict(getattr(core, f"_{backend}_status")())
    if not status.get("built") or not status.get("runtime_available"):
        pytest.skip(f"{backend} is unavailable")
    device = _operator(10, 4, seed=1).to_device(backend=backend)
    with pytest.raises(RuntimeError, match="DLPack|Array Interface"):
        device.device_arrays()

"""Zero-copy device arrays: DevicePauliSum.device_arrays() and from_device_arrays().

On CUDA, device_arrays() returns views of the x words, z words and
coefficients, flagged read-only, that CuPy consumes through DLPack or the CUDA
Array Interface and PyTorch through DLPack, without a copy, and that keep the
operator alive. from_device_arrays() copies CUDA arrays into a new operator
without a host round trip, waiting for producers on any stream. HIP and Metal
raise.
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


class _FakeInterface:
    """A minimal __cuda_array_interface__ producer for malformed-input tests."""

    def __init__(self, pointer: int, shape: tuple[int, ...], typestr: str = "<c16", **extra: object) -> None:
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": typestr,
            "data": (pointer, False),
            "version": 3,
            "strides": None,
            **extra,
        }


# Spins for `cycles` clocks, then copies n doubles: keeps a write pending on a stream.
_DELAYED_COPY = r"""
extern "C" __global__ void delayed_copy(long long cycles, double* out, const double* src, int n) {
  const long long start = clock64();
  while (clock64() - start < cycles) {
  }
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    out[i] = src[i];
  }
}
"""


def test_from_device_arrays_waits_for_a_non_blocking_producer() -> None:
    cupy = _cuda_and_cupy()
    op = _operator(64, 4096, seed=21)
    host_x, host_z, host_coeffs = op.to_arrays()
    kernel = cupy.RawKernel(_DELAYED_COPY, "delayed_copy")
    stream = cupy.cuda.Stream(non_blocking=True)
    with stream:
        x, z = cupy.asarray(host_x), cupy.asarray(host_z)
        source = cupy.asarray(host_coeffs)
        coeffs = cupy.zeros_like(source)
        stream.synchronize()
        # About 50 ms on current GPUs before the coefficients land.
        kernel((1,), (256,), (cupy.int64(100_000_000), coeffs.view(cupy.float64), source.view(cupy.float64),
                              cupy.int32(2 * len(host_coeffs))))
    rebuilt = wolfgang_quantum.DevicePauliSum.from_device_arrays(x, z, coeffs, 64)  # no stream sync
    assert rebuilt.to_host() == op


def test_from_device_arrays_finishes_before_returning() -> None:
    cupy = _cuda_and_cupy()
    op = _operator(64, 4096, seed=22)
    x, z, coeffs = (cupy.asarray(array) for array in op.to_arrays())
    kernel = cupy.RawKernel(_DELAYED_COPY, "delayed_copy")
    cupy.cuda.runtime.deviceSynchronize()
    with cupy.cuda.Stream.null:  # occupy the legacy stream, so a queued copy would lag
        kernel((1,), (1,), (cupy.int64(100_000_000), coeffs.view(cupy.float64), coeffs.view(cupy.float64),
                            cupy.int32(0)))
    rebuilt = wolfgang_quantum.DevicePauliSum.from_device_arrays(x, z, coeffs, 64)
    stream = cupy.cuda.Stream(non_blocking=True)
    with stream:  # overwrite the sources at once, on a stream unordered with the copy
        coeffs.fill(0)
        x.fill(0)
        z.fill(0)
    stream.synchronize()
    assert rebuilt.to_host() == op


def test_cuda_array_interface_consumers_keep_the_operator_alive() -> None:
    cupy = _cuda_and_cupy()
    op = _operator(70, 120, seed=4)
    device = op.to_device(backend="cuda")
    x = cupy.asarray(device.device_arrays()[0])
    del device
    gc.collect()
    np.testing.assert_array_equal(cupy.asnumpy(x), op.to_arrays()[0])


def test_zero_term_and_zero_width_operators_export_and_import() -> None:
    cupy = _cuda_and_cupy()
    empty = wolfgang_quantum.PauliSum.empty(5).to_device(backend="cuda")
    for view, shape in zip(empty.device_arrays(), ((0, 1), (0, 1), (0,)), strict=True):
        assert view.shape == shape
        assert cupy.from_dlpack(view).shape == shape
        assert cupy.asarray(view).shape == shape
    zero_width = wolfgang_quantum.PauliSum.from_arrays(
        np.zeros((3, 0), dtype=np.uint64), np.zeros((3, 0), dtype=np.uint64), np.arange(3) + 1j, 0
    )
    device = zero_width.to_device(backend="cuda")
    x, z, coeffs = (cupy.from_dlpack(view) for view in device.device_arrays())
    assert x.shape == (3, 0)
    assert wolfgang_quantum.DevicePauliSum.from_device_arrays(x, z, coeffs, 0).to_host() == zero_width


def test_from_device_arrays_rejects_bad_metadata_and_memory() -> None:
    cupy = _cuda_and_cupy()
    op = _operator(40, 50, seed=8)
    x, z, coeffs = (cupy.asarray(array) for array in op.to_arrays())
    from_arrays = wolfgang_quantum.DevicePauliSum.from_device_arrays
    empty_words = cupy.empty((1, 0), dtype=cupy.uint64)
    with pytest.raises(ValueError, match="too large"):
        from_arrays(empty_words, empty_words, cupy.ones(1, dtype=cupy.complex128), 2**64 - 1)
    with pytest.raises(ValueError, match="shape"):
        from_arrays(x, z[:10], coeffs, 40)
    with pytest.raises(ValueError, match="dimensions"):
        from_arrays(x.ravel(), z, coeffs, 40)
    with pytest.raises(ValueError, match="requested CUDA device"):
        from_arrays(x, z, coeffs, 40, device=len(wolfgang_quantum.cuda_devices()))
    with pytest.raises(ValueError, match="device must be -1"):
        from_arrays(x, z, coeffs, 40, device=-2)
    with pytest.raises(ValueError, match="non-null"):
        from_arrays(x[:2], z[:2], _FakeInterface(0, (2,)), 40)
    with pytest.raises(TypeError, match="masked"):
        from_arrays(x[:2], z[:2], _FakeInterface(coeffs.data.ptr, (2,), mask=(1, False)), 40)
    pinned = cupy.cuda.alloc_pinned_memory(16 * 2)
    with pytest.raises(ValueError, match="device memory"):
        from_arrays(x[:2], z[:2], _FakeInterface(pinned.ptr, (2,)), 40)
    # Managed memory is device-accessible and accepted.
    managed = cupy.ndarray(coeffs.shape, cupy.complex128, cupy.cuda.malloc_managed(coeffs.nbytes))
    managed[:] = coeffs
    assert from_arrays(x, z, managed, 40).to_host() == op


def test_device_array_dlpack_arguments() -> None:
    cupy = _cuda_and_cupy()
    view = _operator(10, 8, seed=2).to_device(backend="cuda").device_arrays()[2]
    with pytest.raises(BufferError, match="max_version"):
        view.__dlpack__()
    with pytest.raises(BufferError, match="DeviceArrayView.__dlpack__ does not support copy=True"):
        view.__dlpack__(max_version=(1, 0), copy=True)
    with pytest.raises(BufferError, match="another device"):
        view.__dlpack__(max_version=(1, 0), dl_device=(1, 0))
    assert view.__dlpack__(max_version=(1, 1), dl_device=(2, 0)) is not None
    assert cupy.from_dlpack(view).shape == (8,)


def test_torch_reads_device_arrays_through_dlpack() -> None:
    _cuda_and_cupy()
    torch = pytest.importorskip("torch", reason="PyTorch is required for the PyTorch interop test")
    if not torch.cuda.is_available():
        pytest.skip("PyTorch has no CUDA runtime")
    op = _operator(64, 64, seed=9)
    device = op.to_device(backend="cuda")
    coeffs = torch.from_dlpack(device.device_arrays()[2])
    assert coeffs.dtype == torch.complex128
    np.testing.assert_array_equal(coeffs.cpu().numpy(), op.to_arrays()[2])
    x = torch.from_dlpack(device.device_arrays()[0])
    np.testing.assert_array_equal(x.cpu().numpy(), op.to_arrays()[0])
    # PyTorch refuses read-only CUDA Array Interface input; DLPack is the PyTorch path.
    with pytest.raises(TypeError, match="read only"):
        torch.as_tensor(device.device_arrays()[2], device="cuda")
    # Errors raised by a producer's attribute propagate unchanged.
    graded = torch.zeros(64, dtype=torch.complex128, device="cuda", requires_grad=True)
    with pytest.raises(RuntimeError, match="requires grad"):
        wolfgang_quantum.DevicePauliSum.from_device_arrays(
            torch.from_dlpack(device.device_arrays()[0]), torch.from_dlpack(device.device_arrays()[1]),
            graded, 64,
        )


def test_from_device_arrays_needs_a_cuda_build() -> None:
    if dict(core._cuda_status()).get("built"):
        pytest.skip("CUDA build")
    with pytest.raises(RuntimeError, match="CUDA build"):
        wolfgang_quantum.DevicePauliSum.from_device_arrays(object(), object(), object(), 1)


@pytest.mark.parametrize("backend", ["hip", "metal"])
def test_device_arrays_unavailable_on_hip_and_metal(backend: str) -> None:
    status = dict(getattr(core, f"_{backend}_status")())
    if not status.get("built") or not status.get("runtime_available"):
        pytest.skip(f"{backend} is unavailable")
    device = _operator(10, 4, seed=1).to_device(backend=backend)
    with pytest.raises(RuntimeError, match="DLPack|Array Interface"):
        device.device_arrays()

"""CUDA operators on more than one GPU.

Every test needs two visible CUDA devices and skips otherwise. Operations on
the second device must match the CPU and leave the caller's current device
alone; operands on different devices are rejected; buffers exported from one
device must be readable by a kernel on a peer that can access it (the
Wolfgang memory pool maps exported memory into those peers); and device calls
release the GIL, so Python threads can drive several GPUs at once.

The peer-read tests run in a subprocess: an illegal address is sticky and
would poison the CUDA context for every later test in this process.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time

import numpy as np
import pytest
import wolfgang_quantum
import wolfgang_quantum._wolfgang_core as core


def _two_devices() -> None:
    status = dict(core._cuda_status())
    if not status.get("built") or not status.get("runtime_available"):
        pytest.skip(str(status.get("skip_reason", "CUDA is unavailable")))
    if len(wolfgang_quantum.cuda_devices()) < 2:
        pytest.skip("needs two visible CUDA devices")


def _cupy():
    return pytest.importorskip("cupy", reason="CuPy is required for this multi-GPU test")


def _operator(num_qubits: int, num_terms: int, seed: int, duplicates: float = 0.5) -> wolfgang_quantum.PauliSum:
    rng = np.random.default_rng(seed)
    words = (num_qubits + 63) // 64
    distinct = max(1, int(num_terms * (1 - duplicates)))
    x = rng.integers(0, 2**63, size=(distinct, words), dtype=np.uint64)
    z = rng.integers(0, 2**63, size=(distinct, words), dtype=np.uint64)
    if num_qubits % 64:
        tail = np.uint64((1 << (num_qubits % 64)) - 1)
        x[:, -1] &= tail
        z[:, -1] &= tail
    pick = rng.integers(0, distinct, size=num_terms)
    coeffs = rng.normal(size=num_terms) + 1j * rng.normal(size=num_terms)
    return wolfgang_quantum.PauliSum.from_arrays(x[pick], z[pick], coeffs, num_qubits)


# Sizes cover the fused small kernel (up to 1,024 terms), the one- and two-word
# pipeline paths and the generic wide path.
@pytest.mark.parametrize(("num_qubits", "num_terms"), [(24, 128), (24, 50_000), (100, 3_000), (300, 2_000)])
def test_simplify_on_the_second_device_matches_the_cpu(num_qubits: int, num_terms: int) -> None:
    _two_devices()
    op = _operator(num_qubits, num_terms, seed=num_terms + num_qubits)
    device_op = op.to_device(device=1)
    assert device_op.device == 1
    result = device_op.simplify()
    assert result.device == 1
    assert result.to_host().equiv(op.simplify())


def test_other_operations_on_the_second_device_match_the_cpu() -> None:
    _two_devices()
    lhs, rhs = _operator(24, 128, seed=1), _operator(24, 96, seed=2)
    dl, dr = lhs.to_device(device=1), rhs.to_device(device=1)

    product = dl.matmul(dr)
    assert product.device == 1
    assert product.to_host().equiv(lhs.matmul(rhs))

    expected = np.asarray(lhs.commutes_with(rhs))
    assert np.array_equal(np.asarray(dl.commutes_with(dr)), expected)
    matrix = dl.commutes_with_device(dr)
    assert matrix.device == 1
    assert matrix.__dlpack_device__() == (2, 1)
    assert np.array_equal(matrix.to_host(), expected)
    assert matrix.count_commuting() == int(expected.sum())
    assert matrix.conflict_degrees() == int((~expected).sum())

    small = _operator(10, 32, seed=3)
    rng = np.random.default_rng(4)
    psi = rng.normal(size=1 << 10) + 1j * rng.normal(size=1 << 10)
    psi /= np.linalg.norm(psi)
    assert small.to_device(device=1).expectation_statevector(psi) == pytest.approx(
        small.expectation_statevector(psi), abs=1e-10
    )


def test_device_calls_leave_the_current_device_alone() -> None:
    _two_devices()
    cupy = _cupy()
    op = _operator(24, 256, seed=5)
    for current, target in ((0, 1), (1, 0)):
        cupy.cuda.runtime.setDevice(current)
        device_op = op.to_device(device=target)
        device_op.simplify().to_host()
        device_op.matmul(device_op).commutes_with_device(device_op.simplify()).count_commuting()
        device_op.device_arrays()
        del device_op
        assert cupy.cuda.runtime.getDevice() == current
    cupy.cuda.runtime.setDevice(0)


def test_operands_on_different_devices_are_rejected() -> None:
    _two_devices()
    op = _operator(24, 64, seed=6)
    on0, on1 = op.to_device(device=0), op.to_device(device=1)
    with pytest.raises(ValueError, match="same device"):
        on0.matmul(on1)
    with pytest.raises(ValueError, match="same device"):
        on0.commutes_with(on1)
    with pytest.raises(ValueError, match="same device"):
        on1.commutes_with_device(on1, output=wolfgang_quantum.DeviceCommutationMatrix.empty((64, 64), device=0))


def test_device_arrays_round_trip_on_the_second_device() -> None:
    _two_devices()
    cupy = _cupy()
    op = _operator(24, 256, seed=7)
    device_op = op.to_device(device=1)
    x, z, coeffs = (cupy.from_dlpack(view) for view in device_op.device_arrays())
    assert x.device.id == z.device.id == coeffs.device.id == 1
    rebuilt = wolfgang_quantum.DevicePauliSum.from_device_arrays(x, z, coeffs, op.num_qubits)
    assert rebuilt.device == 1
    assert rebuilt.to_host() == op
    with pytest.raises(ValueError, match="requested CUDA device"):
        wolfgang_quantum.DevicePauliSum.from_device_arrays(x, z, coeffs, op.num_qubits, device=0)


_SECOND_DEVICE_ONLY = textwrap.dedent(
    """
    import os
    import subprocess

    import numpy as np
    import wolfgang_quantum


    def memory_on_gpu0_mib():
        uuid = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=uuid", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout.strip()
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
        for line in apps.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if fields[:2] == [uuid, str(os.getpid())]:
                return float(fields[2])
        return 0.0


    rng = np.random.default_rng(10)
    labels = ["".join(rng.choice(list("IXYZ"), 24)) for _ in range(2048)]
    op = wolfgang_quantum.PauliSum.from_labels(labels, rng.normal(size=2048) + 1j * rng.normal(size=2048))
    device_op = op.to_device(device=1)
    device_op.simplify().to_host()
    device_op.matmul(device_op.simplify()).simplify()
    device_op.commutes_with(device_op)
    matrix = device_op.commutes_with_device(device_op)
    matrix.count_commuting()
    matrix.to_host()
    psi = np.zeros(1 << 24, dtype=np.complex128)
    psi[0] = 1.0
    device_op.expectation_statevector(psi)
    views = device_op.device_arrays()
    del views, matrix, device_op
    print(memory_on_gpu0_mib())
    """
)


def test_a_process_using_only_the_second_device_leaves_the_first_alone() -> None:
    # Since CUDA 12, cudaSetDevice creates the device's primary context; switching
    # back to an unused device 0 after each call used to cost about 500 MiB there.
    _two_devices()
    result = subprocess.run(
        [sys.executable, "-c", _SECOND_DEVICE_ONLY], capture_output=True, text=True, timeout=300, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert float(result.stdout.strip().splitlines()[-1]) == 0.0, result.stdout


_PEER_READ = textwrap.dedent(
    """
    import cupy
    import numpy as np
    import wolfgang_quantum

    if not cupy.cuda.runtime.deviceCanAccessPeer(0, 1):
        print("SKIP no peer access between devices 0 and 1")
        raise SystemExit(0)
    rng = np.random.default_rng(8)
    labels = ["".join(rng.choice(list("IXYZ"), 24)) for _ in range(512)]
    op = wolfgang_quantum.PauliSum.from_labels(labels, rng.normal(size=512) + 1j * rng.normal(size=512))
    device_op = op.to_device(device=1)
    if "{source}" == "arrays":
        source = cupy.from_dlpack(device_op.device_arrays()[2])
        expected = op.to_arrays()[2]
    else:
        source = cupy.from_dlpack(device_op.commutes_with_device(device_op))
        expected = np.asarray(op.commutes_with(op)).astype(np.uint8)
    copy = cupy.ElementwiseKernel("T x", "T y", "y = x", "wolfgang_peer_copy")
    with cupy.cuda.Device(0):
        cupy.cuda.runtime.deviceEnablePeerAccess(1)
        view = cupy.ndarray(source.shape, source.dtype, cupy.cuda.MemoryPointer(
            cupy.cuda.UnownedMemory(source.data.ptr, source.nbytes, source, 1), 0))
        out = cupy.empty(source.shape, source.dtype)
        copy(view, out)
        cupy.cuda.Device(0).synchronize()
        assert np.array_equal(out.get(), expected)
    print("OK")
    """
)


@pytest.mark.parametrize("source", ["arrays", "matrix"])
def test_a_peer_device_kernel_reads_exported_buffers(source: str) -> None:
    _two_devices()
    _cupy()
    result = subprocess.run(
        [sys.executable, "-c", _PEER_READ.replace("{source}", source)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.stdout.startswith("SKIP"):
        pytest.skip(result.stdout.strip())
    assert result.returncode == 0 and result.stdout.strip().endswith("OK"), result.stdout + result.stderr


def test_threads_on_two_devices_give_cpu_results() -> None:
    _two_devices()
    errors: list[str] = []

    def work(device: int, seed: int) -> None:
        try:
            for step in range(15):
                op = _operator(24 if step % 2 else 100, 2_000 + 500 * step, seed=seed * 100 + step)
                device_op = op.to_device(device=device)
                assert device_op.simplify().to_host().equiv(op.simplify())
                # Device duplicate sums may associate differently from the CPU's
                # (docs/architecture/cuda_backend.md); products of these sizes
                # differ by about 1e-12 on coefficients summed from many terms.
                product = device_op.matmul(device_op.simplify()).to_host()
                assert product.equiv(op.matmul(op.simplify()), atol=1e-9)
        except Exception as error:  # noqa: BLE001 - reported below
            errors.append(f"device {device}: {error!r}")

    threads = [threading.Thread(target=work, args=(device, seed)) for seed, device in enumerate((0, 1, 0, 1))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []


def test_a_long_device_call_releases_the_gil() -> None:
    _two_devices()
    op = _operator(100, 4_000_000, seed=9, duplicates=0.0)
    device_op = op.to_device(device=1)
    device_op.to_host()  # warm the staging pool
    started = threading.Event()
    window: dict[str, float] = {}

    def call() -> None:
        started.set()
        window["begin"] = time.perf_counter()
        device_op.to_host()
        window["end"] = time.perf_counter()

    stamps: list[float] = []
    worker = threading.Thread(target=call)
    worker.start()
    started.wait()
    while worker.is_alive():
        stamps.append(time.perf_counter())
    worker.join()
    # This thread's first stamp can only follow the worker's "begin" once the
    # worker lets go of the GIL, so measure every gap inside the call's window,
    # including the one from "begin" to the first stamp. With the GIL held for
    # the whole copy, that first gap would span the call.
    begin, end = window["begin"], window["end"]
    inside = [begin] + [t for t in stamps if begin < t < end] + [end]
    largest_gap = max(later - earlier for earlier, later in zip(inside, inside[1:]))
    assert end - begin > 0.02, window
    assert largest_gap < 0.5 * (end - begin), (largest_gap, end - begin)

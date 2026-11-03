"""Wolfgang Python package.

Phase 11 exposes packed Pauli-sum storage, dense-label and sparse-list I/O, the
optional Qiskit SparsePauliOp and OpenFermion QubitOperator adapters, scalar
simplify/canonical ordering, addition without implicit simplify, scalar
multiplication, Pauli-sum multiplication with phase-correct matrix-product
semantics, guarded pairwise commutation, and deterministic greedy QWC/full
commuting groups, plus scalar CPU statevector and diagonal Z-count expectation
kernels. CPU backend dispatch metadata, forced scalar execution, and optional
oneTBB/SIMD selectors for covered commutation and grouping kernels are
available through the `WOLFGANG_CPU_BACKEND` environment variable when the
optimized paths are compiled and runtime-available. `set_num_threads()` and the
`WOLFGANG_NUM_THREADS` environment variable cap the oneTBB threads.

CUDA source builds can additionally expose explicit host/device transfers through
`PauliSum.to_device()` and `DevicePauliSum.to_host()`, plus CUDA simplify,
statevector expectation, pairwise commutation, experimental device-resident
commutation matrices, and matrix-product kernels on `DevicePauliSum`. ROCm/HIP
source builds expose the same transfer entrypoint on HIP-only builds as bring-up
evidence lands. Apple Metal source builds expose the backend-neutral transfer,
pairwise commutation, device-resident commutation matrix surfaces, and
simplify: a GPU-sort hybrid for one-word operators of at least 12,288 terms,
otherwise the source-build simplify transfer-reference correctness bridge.
"""

from . import _wolfgang_core as _wolfgang_core
from . import measurement as measurement
from ._capabilities import (
    BackendCapabilities,
    CpuCapabilities,
    WolfgangCapabilities,
    capabilities,
)
from ._version import __version__
from ._wolfgang_core import (
    DeviceArrayView,
    DeviceCommutationMatrix,
    DevicePauliSum,
    PauliSum,
    cuda_available,
    cuda_devices,
    get_num_threads,
    hip_available,
    hip_devices,
    metal_available,
    metal_devices,
    set_num_threads,
)
from .cudaq import _install_pauli_sum_cudaq_methods
from .measurement import _install_pauli_sum_measurement_methods
from .openfermion import _install_pauli_sum_openfermion_methods
from .qiskit import _install_pauli_sum_qiskit_methods

_install_pauli_sum_qiskit_methods()
_install_pauli_sum_openfermion_methods()
_install_pauli_sum_cudaq_methods()
_install_pauli_sum_measurement_methods()

__all__ = [
    "BackendCapabilities",
    "CpuCapabilities",
    "DeviceArrayView",
    "DeviceCommutationMatrix",
    "DevicePauliSum",
    "WolfgangCapabilities",
    "PauliSum",
    "__version__",
    "_wolfgang_core",
    "capabilities",
    "cuda_available",
    "cuda_devices",
    "get_num_threads",
    "hip_available",
    "hip_devices",
    "measurement",
    "metal_available",
    "metal_devices",
    "set_num_threads",
]

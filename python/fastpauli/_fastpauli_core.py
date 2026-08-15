from wolfgang_quantum import _wolfgang_core as _wolfgang_core
from wolfgang_quantum._wolfgang_core import *  # noqa: F403

_build_info = _wolfgang_core._build_info
_cuda_status = _wolfgang_core._cuda_status
_hip_status = _wolfgang_core._hip_status
_metal_status = _wolfgang_core._metal_status

for _name in (
    "_accelerator_status",
    "_accelerator_backend_selection_for_testing",
    "_accelerator_context_validation_for_testing",
    "_benchmark_cuda_fused_commutation_consumer",
    "_benchmark_cuda_device_resident_consumer",
):
    _value = getattr(_wolfgang_core, _name, None)
    if _value is not None:
        globals()[_name] = _value

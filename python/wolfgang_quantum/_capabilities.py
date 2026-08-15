"""Stable, immutable capability discovery for Wolfgang builds."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Literal

from ._version import __version__

_PACKAGE_NAME = __package__ or "wolfgang_quantum"
_core = importlib.import_module(f"{_PACKAGE_NAME}._wolfgang_core")

BackendName = Literal["cuda", "hip", "metal"]


@dataclass(frozen=True, slots=True)
class CpuCapabilities:
    """Compiled and runtime-visible CPU dispatch state."""

    requested: str
    active: str
    compiled: tuple[str, ...]
    available: tuple[str, ...]
    optimized_kernels: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """One accelerator backend's compiled and runtime state."""

    name: BackendName
    compiled: bool
    runtime_available: bool
    device_count: int
    devices: tuple[str, ...]
    runtime_version: str
    driver_or_platform_version: str
    reason: str


@dataclass(frozen=True, slots=True)
class WolfgangCapabilities:
    """Immutable snapshot of one imported Wolfgang native build."""

    version: str
    build_mode: str
    cpu: CpuCapabilities
    accelerators: tuple[BackendCapabilities, ...]

    def accelerator(self, name: BackendName) -> BackendCapabilities:
        """Return one accelerator capability record by canonical backend name."""

        for backend in self.accelerators:
            if backend.name == name:
                return backend
        raise ValueError(f"unknown accelerator backend: {name}")


def _device_names(status: dict[str, object]) -> tuple[str, ...]:
    devices = status.get("devices", [])
    if not isinstance(devices, list):
        return ()
    names: list[str] = []
    for index, device in enumerate(devices):
        if isinstance(device, dict):
            name = device.get("name") or device.get("device_name")
            names.append(str(name) if name else f"device:{index}")
        else:
            names.append(f"device:{index}")
    return tuple(names)


def _accelerator_record(name: BackendName, status: dict[str, object]) -> BackendCapabilities:
    if name == "metal":
        runtime_version = ""
        platform_version = str(status.get("macos_version", ""))
    else:
        runtime_version = str(status.get("runtime_version", ""))
        platform_version = str(status.get("driver_version", ""))

    compiled = bool(status.get("built", False))
    runtime_available = bool(status.get("runtime_available", False))
    reason = str(status.get("skip_reason", ""))
    if not runtime_available and not reason:
        reason = f"{name} runtime is not available in this build environment"

    raw_device_count = status.get("device_count", 0)
    device_count = raw_device_count if isinstance(raw_device_count, int) else 0
    return BackendCapabilities(
        name=name,
        compiled=compiled,
        runtime_available=runtime_available,
        device_count=device_count,
        devices=_device_names(status),
        runtime_version=runtime_version,
        driver_or_platform_version=platform_version,
        reason=reason,
    )


def capabilities() -> WolfgangCapabilities:
    """Describe compiled and runtime-visible capabilities without trial calls.

    The result is an immutable snapshot. It distinguishes a backend compiled into
    the extension from a runtime-visible device, avoiding exception-driven feature
    discovery and preventing source-build evidence from being mistaken for wheel
    availability.
    """

    build = _core._build_info()
    optimized = build.get("optimized_cpu_kernels", {})
    optimized_items: list[tuple[str, tuple[str, ...]]] = []
    if isinstance(optimized, dict):
        for backend, kernels in sorted(optimized.items()):
            kernel_tuple = tuple(str(kernel) for kernel in kernels) if isinstance(kernels, list) else ()
            optimized_items.append((str(backend), kernel_tuple))

    cpu = CpuCapabilities(
        requested=str(build.get("requested_cpu_backend", "auto")),
        active=str(build.get("active_cpu_backend", build.get("cpu_backend", "scalar"))),
        compiled=tuple(str(item) for item in build.get("compiled_cpu_backends", ["scalar"])),
        available=tuple(str(item) for item in build.get("available_cpu_backends", ["scalar"])),
        optimized_kernels=tuple(optimized_items),
    )

    accelerators = (
        _accelerator_record("cuda", _core._cuda_status()),
        _accelerator_record("hip", _core._hip_status()),
        _accelerator_record("metal", _core._metal_status()),
    )
    return WolfgangCapabilities(
        version=__version__,
        build_mode=str(build.get("accelerator_build_mode", "cpu_only")),
        cpu=cpu,
        accelerators=accelerators,
    )


# One-transition alias for the legacy exported name.
FastPauliCapabilities = WolfgangCapabilities

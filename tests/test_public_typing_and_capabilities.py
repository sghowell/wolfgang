from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import wolfgang_quantum as wolfgang
from wolfgang_quantum import _capabilities

ROOT = Path(__file__).resolve().parents[1]


def test_capabilities_reports_cpu_and_all_accelerator_backends() -> None:
    report = wolfgang.capabilities()

    assert report.version == wolfgang.__version__
    assert report.cpu.active in report.cpu.available
    assert "scalar" in report.cpu.compiled
    assert tuple(backend.name for backend in report.accelerators) == ("cuda", "hip", "metal")
    assert all(backend.device_count >= 0 for backend in report.accelerators)


def test_capability_report_is_immutable() -> None:
    report = wolfgang.capabilities()
    try:
        report.cpu.active = "invented"  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("capability reports must be immutable snapshots")


def test_capabilities_explain_unavailable_backend() -> None:
    report = wolfgang.capabilities()
    unavailable = [backend for backend in report.accelerators if not backend.runtime_available]
    assert all(backend.reason for backend in unavailable)


def test_capabilities_fall_back_when_shipped_extension_lacks_status_helpers(monkeypatch) -> None:
    fake_core = SimpleNamespace(
        _build_info=lambda: {
            "requested_cpu_backend": "auto",
            "active_cpu_backend": "scalar",
            "compiled_cpu_backends": ["scalar"],
            "available_cpu_backends": ["scalar"],
            "optimized_cpu_kernels": {},
            "accelerator_build_mode": "cpu_only",
            "cuda_enabled": False,
            "hip_enabled": False,
            "metal_enabled": False,
            "native_enabled": False,
            "compiled_backends": ["cpu"],
        }
    )
    monkeypatch.setattr(_capabilities, "_core", fake_core)

    report = _capabilities.capabilities()

    assert report.cpu.active == "scalar"
    assert tuple(backend.name for backend in report.accelerators) == ("cuda", "hip", "metal")
    assert all(not backend.compiled for backend in report.accelerators)
    assert all(not backend.runtime_available for backend in report.accelerators)
    assert all("not compiled into this build" in backend.reason for backend in report.accelerators)


def test_pep561_files_are_shipped_in_source_package() -> None:
    package = ROOT / "python" / "wolfgang_quantum"
    marker = package / "py.typed"
    stub = package / "_wolfgang_core.pyi"

    assert marker.is_file()
    assert stub.is_file()
    text = stub.read_text(encoding="utf-8")
    for public_type in ("class PauliSum", "class DevicePauliSum", "class DeviceCommutationMatrix"):
        assert public_type in text


def test_public_exports_are_declared_in_package_stub() -> None:
    stub = (ROOT / "python" / "wolfgang_quantum" / "__init__.pyi").read_text(encoding="utf-8")
    for public_name in (
        "PauliSum",
        "DevicePauliSum",
        "DeviceCommutationMatrix",
        "BackendCapabilities",
        "CpuCapabilities",
        "WolfgangCapabilities",
        "capabilities",
    ):
        assert public_name in stub
    assert "FastPauliCapabilities" not in stub


def test_canonical_package_does_not_export_fastpauli_capability_alias() -> None:
    assert not hasattr(wolfgang, "FastPauliCapabilities")


def test_native_stub_matches_polymorphic_commutation_and_sparse_sequences() -> None:
    stub = (ROOT / "python" / "wolfgang_quantum" / "_wolfgang_core.pyi").read_text(
        encoding="utf-8"
    )

    assert stub.count("-> bool | npt.NDArray[np.bool_]") == 2
    assert "tuple[str, Sequence[int], complex]" in stub
    assert "output: npt.NDArray[np.bool_]" in stub

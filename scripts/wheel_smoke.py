#!/usr/bin/env python3
"""Smoke-test an installed Wolfgang CPU wheel.

This script is intentionally import-only from the installed environment. When
cibuildwheel runs it as ``python {project}/scripts/wheel_smoke.py``, Python
places the repository's ``scripts`` directory on ``sys.path`` rather than the
package source root, so the checks exercise the wheel that was just installed.
"""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
import warnings
from typing import Any


def smoke_wolfgang_wheel(
    expected_version: str | None = None,
    *,
    require_cpu_safe: bool = True,
) -> dict[str, Any]:
    import wolfgang_quantum
    import wolfgang_quantum._wolfgang_core as core
    from wolfgang_quantum import PauliSum

    installed_version = importlib_metadata.version("wolfgang-quantum")
    if expected_version is not None:
        assert wolfgang_quantum.__version__ == expected_version, wolfgang_quantum.__version__
        assert installed_version == expected_version, installed_version
    else:
        assert wolfgang_quantum.__version__ == installed_version, (
            wolfgang_quantum.__version__,
            installed_version,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import fastpauli

    assert fastpauli.__version__ == wolfgang_quantum.__version__, (
        fastpauli.__version__,
        wolfgang_quantum.__version__,
    )

    info = core._build_info()
    if require_cpu_safe:
        assert info["accelerator_build_mode"] == "cpu_only", info
        assert info["cuda_enabled"] is False, info
        assert info["hip_enabled"] is False, info
        assert info["metal_enabled"] is False, info
        assert info["native_enabled"] is False, info
        assert info["compiled_backends"] == ["cpu"], info
        assert "scalar" in info["compiled_cpu_backends"], info

    labels, coeffs = (
        PauliSum.from_labels(["X", "X", "Z"], [1.0, -0.5, 2.0])
        .simplify()
        .to_labels()
    )
    assert labels == ["Z", "X"], labels
    assert coeffs[0] == 2.0 + 0.0j, coeffs
    assert coeffs[1] == 0.5 + 0.0j, coeffs

    return {
        "accelerator_build_mode": info["accelerator_build_mode"],
        "compiled_backends": list(info["compiled_backends"]),
        "compiled_cpu_backends": list(info["compiled_cpu_backends"]),
        "cuda_enabled": info["cuda_enabled"],
        "fastpauli_compat_version": fastpauli.__version__,
        "hip_enabled": info["hip_enabled"],
        "installed_version": installed_version,
        "metal_enabled": info["metal_enabled"],
        "native_enabled": info["native_enabled"],
        "package_import": "wolfgang_quantum",
        "project_distribution": "wolfgang-quantum",
        "wolfgang_version": wolfgang_quantum.__version__,
    }


def smoke_fastpauli_wheel(expected_version: str | None = None) -> dict[str, Any]:
    """Legacy compatibility alias for older tests and tooling."""
    return smoke_wolfgang_wheel(expected_version)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-version",
        help="Optional exact Wolfgang version to require from both package metadata and __version__.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(smoke_wolfgang_wheel(args.expected_version), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

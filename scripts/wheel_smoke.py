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
import os
import subprocess
import sys
from collections.abc import Callable
from typing import Any

# Simplifies a fixed 20,000-term operator with duplicates (above the parallel
# threshold) and prints the active CPU backend and a digest of the result.
_BACKEND_PROBE = '''
import hashlib, json
import numpy as np
import wolfgang_quantum._wolfgang_core as core
from wolfgang_quantum import PauliSum
rng = np.random.default_rng(20261103)
x = rng.integers(0, 1 << 20, size=(20000, 1), dtype=np.uint64)
z = rng.integers(0, 1 << 20, size=(20000, 1), dtype=np.uint64)
x[1::2] = x[::2]
z[1::2] = z[::2]
coeffs = rng.normal(size=20000) + 1j * rng.normal(size=20000)
out = PauliSum.from_arrays(x, z, coeffs, 20).simplify()
digest = hashlib.sha256(b"".join(array.tobytes() for array in out.to_arrays())).hexdigest()
print(json.dumps({"backend": core._build_info()["active_cpu_backend"], "terms": out.num_terms, "digest": digest}))
'''


def _run_backend_probe(backend: str) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-c", _BACKEND_PROBE],
        env={**os.environ, "WOLFGANG_CPU_BACKEND": backend},
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def check_bundled_onetbb(
    info: dict[str, Any],
    run_probe: Callable[[str], dict[str, Any]] = _run_backend_probe,
) -> dict[str, Any]:
    """Require the bundled oneTBB backend and its bit-identity with scalar simplify."""

    assert "tbb" in info["compiled_cpu_backends"], info
    assert "tbb" in info["available_cpu_backends"], info
    assert info["oneTBB_version"] not in {"not_available", "unknown"}, info
    parallel = run_probe("tbb")
    scalar = run_probe("scalar")
    assert parallel["backend"] == "tbb" and scalar["backend"] == "scalar", (parallel, scalar)
    assert parallel["terms"] == scalar["terms"] and parallel["digest"] == scalar["digest"], (parallel, scalar)
    return {"oneTBB_version": info["oneTBB_version"], "tbb_simplify_matches_scalar": True}


def smoke_wolfgang_wheel(
    expected_version: str | None = None,
    *,
    require_cpu_safe: bool = True,
    require_tbb: bool = False,
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

    try:
        __import__("fastpauli")
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError("fastpauli import should fail after compatibility removal")

    info = core._build_info()
    capability_report = wolfgang_quantum.capabilities()
    if require_cpu_safe:
        assert info["accelerator_build_mode"] == "cpu_only", info
        assert info["cuda_enabled"] is False, info
        assert info["hip_enabled"] is False, info
        assert info["metal_enabled"] is False, info
        assert info["native_enabled"] is False, info
        assert info["compiled_backends"] == ["cpu"], info
        assert "scalar" in info["compiled_cpu_backends"], info
        assert capability_report.build_mode == "cpu_only", capability_report
        assert tuple(backend.name for backend in capability_report.accelerators) == (
            "cuda",
            "hip",
            "metal",
        )
        assert all(not backend.compiled for backend in capability_report.accelerators), capability_report

    labels, coeffs = (
        PauliSum.from_labels(["X", "X", "Z"], [1.0, -0.5, 2.0])
        .simplify()
        .to_labels()
    )
    assert labels == ["Z", "X"], labels
    assert coeffs[0] == 2.0 + 0.0j, coeffs
    assert coeffs[1] == 0.5 + 0.0j, coeffs

    tbb_summary = check_bundled_onetbb(info) if require_tbb else {}
    return {
        **tbb_summary,
        "accelerator_build_mode": info["accelerator_build_mode"],
        "capabilities_build_mode": capability_report.build_mode,
        "capabilities_backends": [backend.name for backend in capability_report.accelerators],
        "compiled_backends": list(info["compiled_backends"]),
        "compiled_cpu_backends": list(info["compiled_cpu_backends"]),
        "cuda_enabled": info["cuda_enabled"],
        "hip_enabled": info["hip_enabled"],
        "installed_version": installed_version,
        "metal_enabled": info["metal_enabled"],
        "native_enabled": info["native_enabled"],
        "package_import": "wolfgang_quantum",
        "project_distribution": "wolfgang-quantum",
        "wolfgang_version": wolfgang_quantum.__version__,
    }
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-version",
        help="Optional exact Wolfgang version to require from both package metadata and __version__.",
    )
    parser.add_argument(
        "--require-tbb",
        action="store_true",
        help="Require the bundled oneTBB backend and bit-identical parallel and scalar simplify.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = smoke_wolfgang_wheel(args.expected_version, require_tbb=args.require_tbb)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

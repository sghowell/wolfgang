#!/usr/bin/env python3
"""Smoke-test an installed Wolfgang CPU wheel.

This script is intentionally import-only from the installed environment. When
cibuildwheel runs it as ``python {project}/scripts/wheel_smoke.py --require-tbb``, Python
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

BUNDLED_ONETBB_VERSION = "2022.3.0"
ONETBB_NOTICES = ("third_party/onetbb/LICENSE.txt", "third_party/onetbb/third-party-programs.txt")

# Simplifies fixed operators above the parallel threshold at 20, 100 and 130
# qubits (one, two and three words), with duplicate runs of one to five terms of
# mixed magnitude, and prints the active backend and a digest of each result.
_BACKEND_PROBE = """
import hashlib, json
import numpy as np
import wolfgang_quantum._wolfgang_core as core
from wolfgang_quantum import PauliSum
digests = []
for num_qubits in (20, 100, 130):
    rng = np.random.default_rng(20261103 + num_qubits)
    words = (num_qubits + 63) // 64
    unique = 6000
    ux = rng.integers(0, 2**63, size=(unique, words), dtype=np.uint64)
    uz = rng.integers(0, 2**63, size=(unique, words), dtype=np.uint64)
    tail = num_qubits % 64
    ux[:, -1] &= np.uint64((1 << tail) - 1)
    uz[:, -1] &= np.uint64((1 << tail) - 1)
    pick = np.repeat(np.arange(unique), rng.integers(1, 6, size=unique))
    rng.shuffle(pick)
    coeffs = rng.normal(size=pick.size) * 10.0 ** rng.integers(-8, 8, size=pick.size)
    out = PauliSum.from_arrays(ux[pick], uz[pick], coeffs + 1j * rng.normal(size=pick.size), num_qubits).simplify()
    digests.append(hashlib.sha256(b"".join(array.tobytes() for array in out.to_arrays())).hexdigest())
print(json.dumps({"backend": core._build_info()["active_cpu_backend"], "digests": digests}))
"""

# The parent simplifies through oneTBB and forks; the child must finish an auto
# simplify (oneTBB does not support fork, so Wolfgang falls back to scalar there).
_FORK_PROBE = """
import os, signal
import numpy as np
from wolfgang_quantum import PauliSum
rng = np.random.default_rng(5)
x = rng.integers(0, 1 << 20, size=(20000, 1), dtype=np.uint64)
op = PauliSum.from_arrays(x, x[::-1].copy(), rng.normal(size=20000) + 0j, 20)
os.environ["WOLFGANG_CPU_BACKEND"] = "tbb"
expected = op.simplify().to_arrays()
pid = os.fork()
if pid == 0:
    signal.alarm(30)
    os.environ["WOLFGANG_CPU_BACKEND"] = "auto"
    got = op.simplify().to_arrays()
    os._exit(0 if all(np.array_equal(a, b) for a, b in zip(got, expected)) else 3)
_, status = os.waitpid(pid, 0)
raise SystemExit(0 if status == 0 else 1)
"""


def _run_probe(code: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=180, check=False
    )
    if completed.returncode != 0:
        raise AssertionError(f"probe failed ({completed.returncode}): {completed.stderr}")
    return completed


def _run_backend_probe(backend: str) -> dict[str, Any]:
    completed = _run_probe(_BACKEND_PROBE, {**os.environ, "WOLFGANG_CPU_BACKEND": backend})
    return json.loads(completed.stdout)


def _run_fork_probe() -> bool:
    if not hasattr(os, "fork"):
        return False
    _run_probe(_FORK_PROBE, dict(os.environ))
    return True


def _installed_license_files() -> list[str]:
    files = importlib_metadata.distribution("wolfgang-quantum").files or []
    return [str(path) for path in files]


def check_bundled_onetbb(
    info: dict[str, Any],
    run_probe: Callable[[str], dict[str, Any]] = _run_backend_probe,
    license_files: Callable[[], list[str]] = _installed_license_files,
    run_fork_probe: Callable[[], bool] = _run_fork_probe,
) -> dict[str, Any]:
    """Require the bundled oneTBB: its backend, notices, scalar bit-identity and fork safety."""

    assert "tbb" in info["compiled_cpu_backends"], info
    assert "tbb" in info["available_cpu_backends"], info
    assert info["oneTBB_version"] == BUNDLED_ONETBB_VERSION, info
    installed = license_files()
    for notice in ONETBB_NOTICES:
        assert any(path.endswith(notice) for path in installed), (notice, installed)
    parallel = run_probe("tbb")
    scalar = run_probe("scalar")
    assert parallel["backend"] == "tbb" and scalar["backend"] == "scalar", (parallel, scalar)
    assert parallel["digests"] == scalar["digests"], (parallel, scalar)
    forked = run_fork_probe()
    return {
        "oneTBB_version": info["oneTBB_version"],
        "tbb_simplify_matches_scalar": True,
        "fork_after_tbb_checked": forked,
    }


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

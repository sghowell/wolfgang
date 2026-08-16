#!/usr/bin/env python3
"""Build and smoke-test Wolfgang CPU release artifacts.

The validator proves the release-candidate packaging boundary: build a source
distribution and CPU wheel with accelerator and native CPU tuning disabled, then
install the produced wheel into a clean virtual environment and run an import
plus metadata smoke test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SDIST_ARTIFACT_PREFIX = "wolfgang-quantum-"
WHEEL_ARTIFACT_PREFIX = "wolfgang_quantum-"
PROJECT_DISTRIBUTION = "wolfgang-quantum"


def print_step(name: str) -> None:
    print(f"\n==> {name}", flush=True)


def run_command(
    name: str,
    command: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> None:
    print_step(name)
    print("$ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {completed.returncode}")


def cmake_executable_for_build() -> str | None:
    try:
        import cmake as cmake_package  # type: ignore[import-not-found]
    except Exception:
        cmake_path = shutil.which("cmake")
        return cmake_path

    cmake_path = Path(cmake_package.CMAKE_BIN_DIR) / "cmake"
    if cmake_path.exists():
        return str(cmake_path)
    return shutil.which("cmake")


def cpu_safe_build_env() -> dict[str, str]:
    env = os.environ.copy()
    env["WOLFGANG_ENABLE_CUDA"] = "OFF"
    env["WOLFGANG_ENABLE_HIP"] = "OFF"
    env["WOLFGANG_ENABLE_METAL"] = "OFF"
    env["WOLFGANG_ENABLE_NATIVE"] = "OFF"

    cmake_executable = cmake_executable_for_build()
    if cmake_executable is not None:
        env["CMAKE_EXECUTABLE"] = cmake_executable
    return env


def project_version() -> str:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    if match is None:
        raise SystemExit("could not find project version in pyproject.toml")
    return match.group(1)


def build_artifact_command(output_dir: Path, *, python_executable: str) -> list[str]:
    return [
        python_executable,
        "-m",
        "build",
        "--sdist",
        "--wheel",
        "--outdir",
        str(output_dir),
        "--config-setting",
        "cmake.define.WOLFGANG_ENABLE_CUDA=OFF",
        "--config-setting",
        "cmake.define.WOLFGANG_ENABLE_HIP=OFF",
        "--config-setting",
        "cmake.define.WOLFGANG_ENABLE_METAL=OFF",
        "--config-setting",
        "cmake.define.WOLFGANG_ENABLE_NATIVE=OFF",
    ]


def clean_prior_release_artifacts(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if not (
            path.name.startswith(SDIST_ARTIFACT_PREFIX)
            or path.name.startswith(WHEEL_ARTIFACT_PREFIX)
        ):
            continue
        if path.suffix == ".whl" or path.name.endswith(".tar.gz"):
            path.unlink()


def find_single_artifact(output_dir: Path, suffix: str) -> Path:
    artifacts = sorted(
        path
        for path in output_dir.iterdir()
        if path.name.startswith(SDIST_ARTIFACT_PREFIX) or path.name.startswith(WHEEL_ARTIFACT_PREFIX)
    )
    matching = [path for path in artifacts if path.name.endswith(suffix)]
    if len(matching) != 1:
        names = ", ".join(path.name for path in artifacts) or "none"
        raise SystemExit(f"expected exactly one Wolfgang {suffix} artifact, found: {names}")
    return matching[0]


def validate_artifact_versions(sdist: Path, wheel: Path, expected_version: str) -> None:
    expected_sdist_names = (
        f"{SDIST_ARTIFACT_PREFIX}{expected_version}.tar.gz",
        f"{WHEEL_ARTIFACT_PREFIX}{expected_version}.tar.gz",
    )
    expected_wheel_prefix = f"{WHEEL_ARTIFACT_PREFIX}{expected_version}-"

    if sdist.name not in expected_sdist_names:
        expected_sdist_display = " or ".join(expected_sdist_names)
        raise SystemExit(
            f"expected source distribution named {expected_sdist_display}, "
            f"found {sdist.name}"
        )

    if not wheel.name.startswith(expected_wheel_prefix) or not wheel.name.endswith(".whl"):
        raise SystemExit(
            f"expected wheel basename starting with {expected_wheel_prefix!r} "
            f"and ending with '.whl', found {wheel.name}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifacts(output_dir: Path, *, python_executable: str) -> tuple[Path, Path]:
    clean_prior_release_artifacts(output_dir)
    run_command(
        "build CPU source distribution and wheel",
        build_artifact_command(output_dir, python_executable=python_executable),
        env=cpu_safe_build_env(),
    )
    sdist = find_single_artifact(output_dir, ".tar.gz")
    wheel = find_single_artifact(output_dir, ".whl")
    return sdist, wheel


def venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def wheel_smoke_script(expected_version: str | None = None) -> str:
    version = expected_version if expected_version is not None else project_version()
    return "\n".join(
        [
            "import importlib.metadata as importlib_metadata",
            "import wolfgang_quantum",
            "import wolfgang_quantum._wolfgang_core as core",
            "from wolfgang_quantum import PauliSum",
            "import importlib.util",
            "info = core._build_info()",
            "print(info)",
            "assert info['accelerator_build_mode'] == 'cpu_only', info",
            "assert info['cuda_enabled'] is False, info",
            "assert info['hip_enabled'] is False, info",
            "assert info['metal_enabled'] is False, info",
            "assert info['native_enabled'] is False, info",
            "assert info['compiled_backends'] == ['cpu'], info",
            "assert 'scalar' in info['compiled_cpu_backends'], info",
            "labels, coeffs = PauliSum.from_labels(['X', 'X', 'Z'], [1.0, -0.5, 2.0]).simplify().to_labels()",
            "assert labels == ['Z', 'X'], labels",
            "assert coeffs[0] == 2.0 + 0.0j, coeffs",
            "assert coeffs[1] == 0.5 + 0.0j, coeffs",
            f"assert wolfgang_quantum.__version__ == {version!r}, wolfgang_quantum.__version__",
            f"assert importlib_metadata.version('{PROJECT_DISTRIBUTION}') == {version!r}, importlib_metadata.version('{PROJECT_DISTRIBUTION}')",
            "assert importlib.util.find_spec('fastpauli') is None",
            "print(wolfgang_quantum.__version__)",
        ]
    )


def install_and_smoke_wheel(output_dir: Path, wheel: Path, *, python_executable: str) -> Path:
    venv_dir = output_dir / "wheel-smoke-venv"
    if venv_dir.exists():
        shutil.rmtree(venv_dir)

    run_command(
        "create clean wheel smoke virtual environment",
        [python_executable, "-m", "venv", str(venv_dir)],
    )
    venv_python = venv_python_path(venv_dir)
    run_command(
        "upgrade wheel smoke installer",
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
    )
    run_command(
        "install produced Wolfgang wheel",
        [str(venv_python), "-m", "pip", "install", str(wheel)],
    )
    run_command(
        "import produced Wolfgang wheel and verify CPU-safe metadata",
        [str(venv_python), "-c", wheel_smoke_script()],
    )
    return venv_python


def validate_release_artifacts(output_dir: Path, *, python_executable: str) -> dict[str, str]:
    output_dir = output_dir.resolve()
    expected_version = project_version()
    sdist, wheel = build_artifacts(output_dir, python_executable=python_executable)
    validate_artifact_versions(sdist, wheel, expected_version)
    venv_python = install_and_smoke_wheel(
        output_dir,
        wheel,
        python_executable=python_executable,
    )

    summary = {
        "output_dir": str(output_dir),
        "sdist": str(sdist),
        "sdist_name": sdist.name,
        "sdist_sha256": sha256_file(sdist),
        "wheel": str(wheel),
        "wheel_name": wheel.name,
        "wheel_sha256": sha256_file(wheel),
        "venv_python": str(venv_python),
        "build_mode": "cpu_only",
        "native_enabled": "false",
        "expected_version": expected_version,
    }
    print_step("release artifact validation summary")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "_skbuild" / "validate-release-artifacts",
        help="Directory for generated artifacts and the clean wheel-smoke virtual environment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_release_artifacts(args.output_dir, python_executable=sys.executable)


if __name__ == "__main__":
    main()

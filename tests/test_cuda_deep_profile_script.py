"""Deep CUDA profiling orchestration tests."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cuda_deep_profile.py"


def _load_cuda_deep_profile_module():
    spec = importlib.util.spec_from_file_location("cuda_deep_profile_under_test", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cuda_deep_profile_dry_run_emits_required_tool_ladder(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dry-run",
            "--json",
            "--profile",
            "smoke",
            "--operation",
            "pairwise_commutation",
            "--output-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["execute"] is False
    assert report["profile"] == "smoke"
    assert report["operations"] == ["pairwise_commutation"]

    step_names = {step["name"] for step in report["steps"]}
    assert "cuda validation" in step_names
    assert "cuda scaling benchmark" in step_names
    assert "nsys cuda api timeline" in step_names
    assert "ncu pairwise_commutation detailed" in step_names
    assert "compute sanitizer memcheck" in step_names
    assert "cuobjdump sass inventory" in step_names
    assert "nvdisasm sass listing" in step_names

    commands = "\n".join(step["command"] for step in report["steps"])
    assert "FASTPAULI_VALIDATE_CUDA=1" in commands
    assert "benchmarks/bench_cuda_scaling.py" in commands
    assert "--kernel-name regex:commutation_kernel" in commands


def test_cuda_deep_profile_can_include_gpu_competitor_setup(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dry-run",
            "--json",
            "--profile",
            "smoke",
            "--operation",
            "statevector_expectation",
            "--competitor-set",
            "gpu",
            "--output-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    commands = "\n".join(step["command"] for step in report["steps"])
    assert "cupy-cuda12x" in commands
    assert "cuquantum-python-cu12" in commands
    assert "cudaq" in commands
    assert "qiskit-aer-gpu" in commands
    assert "benchmarks/bench_competitive_baselines.py" in commands


def test_cuda_deep_profile_profiles_all_kernels_for_thrust_heavy_operations(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dry-run",
            "--json",
            "--profile",
            "smoke",
            "--operation",
            "simplify_duplicate_pressure,matmul_product_generation_simplify",
            "--output-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    ncu_commands = [
        step["command"]
        for step in report["steps"]
        if step["name"].startswith("ncu ")
    ]
    assert len(ncu_commands) == 2
    assert all("--kernel-name regex:" not in command for command in ncu_commands)
    assert any("--launch-count 16" in command for command in ncu_commands)
    assert any("--launch-count 24" in command for command in ncu_commands)


def test_cuda_deep_profile_can_require_profiler_artifacts_for_completion_runs(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dry-run",
            "--json",
            "--profile",
            "smoke",
            "--operation",
            "pairwise_commutation",
            "--require-profiler-artifacts",
            "--output-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["require_profiler_artifacts"] is True
    profiler_steps = [
        step
        for step in report["steps"]
        if step["name"] in {"nsys cuda api timeline", "ncu pairwise_commutation detailed"}
    ]
    assert profiler_steps
    assert all(step["required"] is True for step in profiler_steps)


def test_cuda_deep_profile_extension_path_falls_back_without_installed_extension(monkeypatch) -> None:
    module = _load_cuda_deep_profile_module()

    def raise_module_not_found(name: str):
        assert name == "fastpauli._fastpauli_core"
        raise ModuleNotFoundError("No module named 'fastpauli'")

    monkeypatch.setattr(module.importlib.util, "find_spec", raise_module_not_found)

    assert module.resolve_extension_path() == "<fastpauli_extension_path>"

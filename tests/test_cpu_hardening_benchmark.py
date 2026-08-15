"""CPU hardening benchmark orchestration tests."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def load_cpu_hardening_module() -> ModuleType:
    script = ROOT / "benchmarks" / "bench_cpu_hardening.py"
    spec = importlib.util.spec_from_file_location("bench_cpu_hardening", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flag_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_cpu_hardening_benchmark_smoke_covers_all_hot_paths() -> None:
    script = ROOT / "benchmarks" / "bench_cpu_hardening.py"
    assert script.exists()

    completed = subprocess.run(
        [sys.executable, str(script), "--profile", "smoke", "--repeat", "1", "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["benchmark"] == "cpu_hardening"
    assert report["profile"] == "smoke"
    assert {operation["benchmark"] for operation in report["operations"]} == {
        "simplify",
        "multiply",
        "grouping",
        "expectation",
    }
    for operation in report["operations"]:
        assert operation["returncode"] == 0
        assert operation["cases"]
        assert operation["correctness_checked"] is True
        assert operation["correctness_checks"]["enabled"] is True
        assert "reference" in operation["correctness_checks"]


def test_cpu_hardening_stress_profile_covers_two_word_packed_paths() -> None:
    module = load_cpu_hardening_module()

    for operation in ("simplify", "multiply", "grouping"):
        args = module.PROFILE_ARGS["stress"][operation]
        assert flag_value(args, "--num-qubits") == "65"

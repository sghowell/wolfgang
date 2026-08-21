"""CPU dispatch-threshold benchmark tests."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def load_script_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_thresholds_benchmark_smoke_reports_threshold_and_correctness() -> None:
    script = ROOT / "benchmarks" / "bench_cpu_thresholds.py"
    assert script.exists()

    completed = subprocess.run(
        [sys.executable, str(script), "--smoke", "--repeat", "1", "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["benchmark"] == "cpu_thresholds"
    assert report["thresholds"]["tbb_pairwise_entries"] == 331776
    assert report["thresholds"]["neon_full_grouping_scalar_min_entries"] == 1024
    assert report["correctness_checks"]["enabled"] is True
    assert report["cases"]
    assert {case["dataset"]["threshold_region"] for case in report["cases"]} == {
        "below",
        "at_or_above",
    }
    for case in report["cases"]:
        assert case["results"]["matches_forced_scalar"] is True
        assert "auto_effective_backend_hint" in case["dataset"]


def test_cpu_threshold_hint_respects_simd_word_coverage() -> None:
    threshold_module = load_script_module(
        ROOT / "benchmarks" / "bench_cpu_thresholds.py",
        "bench_cpu_thresholds_for_test",
    )
    dispatch_module = load_script_module(
        ROOT / "benchmarks" / "bench_cpu_dispatch.py",
        "bench_cpu_dispatch_for_test",
    )
    build_info = {
        "cpu_auto_dispatch_thresholds": {
            "tbb_pairwise_entries": 331776,
            "neon_full_grouping_scalar_min_entries": 1024,
        },
        "cpu_backend_candidates": [
            {"name": "scalar", "status": "available"},
            {"name": "tbb", "status": "available"},
            {"name": "avx512", "status": "available"},
        ],
    }

    assert threshold_module.auto_pairwise_backend_hint(build_info, 82944, 65) == "avx512"
    assert threshold_module.auto_pairwise_backend_hint(build_info, 82944, 193) == "scalar"
    assert threshold_module.auto_pairwise_backend_hint(build_info, 331776, 193) == "tbb"
    assert dispatch_module.infer_auto_pairwise_backend(build_info, 82944, 193) == "scalar"
    assert dispatch_module.infer_auto_full_grouping_backend(build_info, 193, 128) == "scalar"


def test_full_grouping_hint_prefers_scalar_for_neon_small_graph_threshold() -> None:
    dispatch_module = load_script_module(
        ROOT / "benchmarks" / "bench_cpu_dispatch.py",
        "bench_cpu_dispatch_for_full_grouping_threshold_test",
    )
    build_info = {
        "cpu_auto_dispatch_thresholds": {
            "tbb_pairwise_entries": 331776,
            "neon_full_grouping_scalar_min_entries": 1024,
        },
        "cpu_backend_candidates": [
            {"name": "scalar", "status": "available"},
            {"name": "neon", "status": "available"},
        ],
    }

    assert dispatch_module.infer_auto_full_grouping_backend(build_info, 65, 16) == "neon"
    assert dispatch_module.infer_auto_full_grouping_backend(build_info, 65, 32) == "scalar"

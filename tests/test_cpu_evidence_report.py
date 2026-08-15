"""CPU benchmark evidence report rendering tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_cpu_evidence_report_renderer_summarizes_required_inputs(tmp_path: Path) -> None:
    dispatch = {
        "benchmark": "cpu_dispatch",
        "git_commit": "abc1234",
        "command": "python benchmarks/bench_cpu_dispatch.py --smoke --json",
        "environment": {
            "cpu_architecture": "arm64",
            "cpu_vendor_or_soc": "Apple M4 Pro",
            "operating_system": "macOS-test",
            "compiler": "AppleClang test",
            "compiled_fastpauli_cpu_backends": ["scalar", "neon"],
            "available_fastpauli_cpu_backends": ["scalar", "neon"],
            "unavailable_fastpauli_cpu_backends": {"tbb": "not_compiled"},
            "oneTBB": {"enabled": False, "version": "not_available"},
            "thread_settings": {"controlled_thread_count": "not_controlled"},
        },
        "fastpauli_build_info": {
            "optimized_cpu_kernels": {"neon": ["commutes_with_words_1_2"]},
            "cpu_auto_dispatch_thresholds": {"tbb_pairwise_entries": 331776},
        },
        "cases": [
            {
                "name": "forced_scalar_pairwise_commutation",
                "dataset": {"matrix_entries": 48},
                "results": {"fastpauli_seconds": 1.0e-5, "matches_forced_scalar": True},
            }
        ],
    }
    hardening = {
        "benchmark": "cpu_hardening",
        "profile": "smoke",
        "operations": [
            {
                "benchmark": "simplify",
                "correctness_checked": True,
                "cases": [
                    {
                        "name": "low_duplicate",
                        "dataset": {
                            "num_qubits": 8,
                            "num_terms": 32,
                            "duplicate_rate": 0.25,
                            "random_seed": 1729,
                        },
                        "fastpauli_scalar_seconds": 1.0e-5,
                        "python_baseline_seconds": 5.0e-5,
                    }
                ],
            }
        ],
    }
    thresholds = {
        "benchmark": "cpu_thresholds",
        "thresholds": {"tbb_pairwise_entries": 331776},
        "cases": [
            {
                "name": "below_threshold",
                "dataset": {
                    "matrix_entries": 65536,
                    "threshold_region": "below",
                    "auto_effective_backend_hint": "neon",
                    "num_qubits": 65,
                    "lhs_terms": 256,
                    "rhs_terms": 256,
                    "random_seed": 1234,
                },
                "results": {
                    "auto_seconds": 2.0e-5,
                    "scalar_seconds": 3.0e-5,
                    "matches_forced_scalar": True,
                    "optimized_backends": {
                        "neon": {"seconds": 2.1e-5, "matches_forced_scalar": True}
                    },
                },
            }
        ],
    }

    dispatch_path = tmp_path / "dispatch.json"
    hardening_path = tmp_path / "hardening.json"
    thresholds_path = tmp_path / "thresholds.json"
    output_path = tmp_path / "report.md"
    write_json(dispatch_path, dispatch)
    write_json(hardening_path, hardening)
    write_json(thresholds_path, thresholds)

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "render_cpu_evidence_report.py"),
            "--title",
            "CPU Evidence Test",
            "--dispatch-json",
            str(dispatch_path),
            "--hardening-json",
            str(hardening_path),
            "--threshold-json",
            str(thresholds_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    rendered = output_path.read_text(encoding="utf-8")
    assert "# CPU Evidence Test" in rendered
    assert "abc1234" in rendered
    assert "Apple M4 Pro" in rendered
    assert "scalar, neon" in rendered
    assert "below_threshold" in rendered
    assert "num_qubits=65" in rendered
    assert "neon=2.1e-05" in rendered
    assert "simplify" in rendered
    assert "random_seed=1729" in rendered

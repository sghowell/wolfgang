from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "h100_validation_artifacts.py"


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("h100_validation_artifacts_under_test", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_classify_per_test_memcheck_distinguishes_non_cuda_cases() -> None:
    module = load_module()

    assert (
        module.classify_per_test_memcheck(
            exit_code=255,
            log_text="========= Error: Target application terminated before first instrumented API call\n",
        )
        == "no_cuda_api_call"
    )
    assert (
        module.classify_per_test_memcheck(
            exit_code=0,
            log_text="========= ERROR SUMMARY: 0 errors\n",
        )
        == "cuda_api_clean"
    )
    assert (
        module.classify_per_test_memcheck(
            exit_code=0,
            log_text="nanobind: leaked 1 instances!\n",
        )
        == "nanobind_leak_diagnostic"
    )


def test_parse_pytest_summary_counts_skips_in_total() -> None:
    module = load_module()

    assert module.parse_pytest_summary("48 passed, 6 skipped, 1 warning in 8.62s") == {
        "passed": 48,
        "skipped": 6,
        "failed": 0,
        "total": 54,
    }


def test_artifact_deriver_refreshes_public_outputs_from_exact_evidence(tmp_path: Path) -> None:
    evidence_root = tmp_path / "remote-artifacts"
    public = evidence_root / "public"
    logs = evidence_root / "private" / "logs"
    raw = evidence_root / "private" / "raw"
    profiler = evidence_root / "private" / "profiler"
    public.mkdir(parents=True)
    logs.mkdir(parents=True)
    raw.mkdir(parents=True)
    profiler.mkdir(parents=True)

    (public / "qualification_manifest.json").write_text(
        json.dumps(
            {
                "source": {"commit": "capture_on_remote", "short_commit": "capture", "tree_state": "capture_on_remote"},
                "runtime": {"driver": "capture_on_remote", "toolkit": {"cuda": "capture_on_remote"}, "device": "capture_on_remote", "os": "capture_on_remote", "python": "capture_on_remote"},
                "build": {"compiler": {"command": "nvcc --version", "value": "capture_on_remote"}, "artifact_hashes": {}, "build_flags": []},
                "test_counts": {"total": "capture_on_remote", "passed": "capture_on_remote", "failed": 0, "skipped": "capture_on_remote"},
                "diagnostics": {"required": ["compute_sanitizer_memcheck", "compute_sanitizer_racecheck"], "status": "capture_on_remote", "summary": "capture_on_remote"},
                "interop_checks": {},
                "benchmarks": {"policy": "public/benchmark_policy.json", "result_summary": "capture_on_remote"},
                "reproducibility": {"same_image_reruns": 2, "fresh_provision_reruns": 1, "test_count_match_required": True},
                "cleanup": {"fail_closed": True, "note": "capture_on_remote"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (public / "benchmark_policy.json").write_text(
        json.dumps({"median_variance_limit_percent": 5}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (logs / "inventory.log").write_text("id: ubuntu\nPython 3.10.12\n", encoding="utf-8")
    (logs / "gpu-summary.csv").write_text("NVIDIA H100 PCIe, 9.0, 570.148.08, 81559 MiB\n", encoding="utf-8")
    (logs / "build-info.txt").write_text(
        "{'git_commit': 'cb08a0a5bf04c5274ad4478350df14d7e592daf4', 'cuda_toolkit_version': '12.8.93', 'cuda_architectures': '90'}\n"
        "{'devices': [{'name': 'NVIDIA H100 PCIe'}]}\n",
        encoding="utf-8",
    )
    (logs / "nvcc-version.log").write_text("Build cuda_12.8.r12.8/compiler.35583870_0\n", encoding="utf-8")
    (logs / "validate-cuda.log").write_text("All checks passed\n", encoding="utf-8")
    (logs / "cuda-contracts.log").write_text("48 passed, 6 skipped, 1 warning in 8.62s\n", encoding="utf-8")
    (profiler / "compute_sanitizer_memcheck.log").write_text("========= ERROR SUMMARY: 0 errors\n", encoding="utf-8")
    (profiler / "compute_sanitizer_racecheck.log").write_text(
        "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n",
        encoding="utf-8",
    )
    (raw / "per_test_memcheck_leaks.json").write_text(
        json.dumps(
            [
                {
                    "i": 1,
                    "nodeid": "tests/test_phase11_cuda_kernels.py::test_phase_eleven_public_cuda_kernel_surface_is_exposed",
                    "exit_code": 255,
                    "leak": False,
                    "error_summary_zero": False,
                    "summary": "1 passed, 1 warning in 0.25s",
                },
                {
                    "i": 2,
                    "nodeid": "tests/test_phase11_cuda_kernels.py::test_cuda_binding_lifecycle_subprocess_exits_without_nanobind_leaks",
                    "exit_code": 0,
                    "leak": False,
                    "error_summary_zero": True,
                    "summary": "1 passed, 1 warning in 2.86s",
                },
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (profiler / "per-test-memcheck-01.log").write_text(
        "========= Error: Target application terminated before first instrumented API call\n",
        encoding="utf-8",
    )
    (profiler / "per-test-memcheck-02.log").write_text(
        "========= ERROR SUMMARY: 0 errors\n",
        encoding="utf-8",
    )
    for run_idx, seconds in enumerate((2.0, 2.1, 1.9), start=1):
        (raw / f"bench_cuda_kernels_default_run{run_idx}.json").write_text(
            json.dumps(
                {
                    "cases": [
                        {
                            "name": "statevector_expectation",
                            "results": {"cuda_device_resident_seconds": seconds},
                        }
                    ]
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    (raw / "bench_cuda_scaling_smoke.json").write_text(
        json.dumps({"cases": [{"name": "scale_a"}, {"name": "scale_b"}]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--evidence-root",
            str(evidence_root),
            "--commit",
            "cb08a0a5bf04c5274ad4478350df14d7e592daf4",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr

    manifest = json.loads((public / "qualification_manifest.json").read_text(encoding="utf-8"))
    sanitized_manifest = json.loads((public / "qualification_manifest.sanitized.json").read_text(encoding="utf-8"))
    summary = json.loads((public / "summary.json").read_text(encoding="utf-8"))
    markdown = (public / "sanitized_h100_validation_summary.md").read_text(encoding="utf-8")

    assert manifest["test_counts"] == {"passed": 48, "skipped": 6, "failed": 0, "total": 54}
    assert sanitized_manifest == manifest
    assert manifest["diagnostics"]["summary"]["per_test_no_cuda_api_call_count"] == 1
    assert manifest["diagnostics"]["summary"]["per_test_no_cuda_api_call_nodeids"] == [
        "tests/test_phase11_cuda_kernels.py::test_phase_eleven_public_cuda_kernel_surface_is_exposed"
    ]
    assert summary["validation"]["compute_sanitizer_racecheck"].startswith("RACECHECK SUMMARY")
    assert summary["reproducibility"]["worst_case_variance_percent"] > 0
    assert summary["status"] == "NO-GO"
    assert "Status: NO-GO" in markdown
    assert "Per-test no-CUDA-API-call classifications: 1" in markdown

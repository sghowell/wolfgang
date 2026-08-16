from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cloud_hardware_qualification_harness.py"
ARCHIVE_PORTABILITY = ROOT / "scripts" / "archive_portability.py"
DOC_PATH = "docs/release/cloud_hardware_qualification_harness.md"
CUDA_COLLECTOR = ROOT / "tools" / "remote" / "collect_cuda_inventory.sh"
ROCM_COLLECTOR = ROOT / "tools" / "remote" / "collect_rocm_inventory.sh"


def load_validate_module() -> ModuleType:
    validate_path = ROOT / "scripts/validate.py"
    spec = importlib.util.spec_from_file_location("wolfgang_validate", validate_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hardware_harness_docs_are_registered() -> None:
    validate = load_validate_module()

    assert SCRIPT.exists()
    assert (ROOT / DOC_PATH).exists()
    assert CUDA_COLLECTOR.exists()
    assert ROCM_COLLECTOR.exists()
    assert DOC_PATH in validate.SOURCE_OF_TRUTH_PATHS

    for path in (
        "docs/release/README.md",
        "docs/release/support_matrix.md",
        "docs/quality/release_and_packaging.md",
        "docs/research/provenance.md",
        "docs/roadmap.md",
    ):
        text = (ROOT / path).read_text(encoding="utf-8")
        assert DOC_PATH in text, path


def test_harness_dry_run_writes_public_manifest_and_fail_closed_runbook(tmp_path: Path) -> None:
    output_dir = tmp_path / "qualification-harness"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "bundle",
            "--lane",
            "mi300x",
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "mi300x" in completed.stdout

    manifest = json.loads((output_dir / "public" / "qualification_manifest.json").read_text(encoding="utf-8"))
    benchmark_policy = json.loads((output_dir / "public" / "benchmark_policy.json").read_text(encoding="utf-8"))
    runbook = (output_dir / "RUNBOOK.md").read_text(encoding="utf-8")
    runner = (output_dir / "scripts" / "run_lane.sh").read_text(encoding="utf-8")
    private_readme = (output_dir / "private" / "README.md").read_text(encoding="utf-8")

    assert manifest["lane"] == "mi300x"
    assert manifest["backend"] == "rocm"
    assert manifest["architecture"] == "gfx942"
    assert manifest["source"]["commit"]
    assert "tree_state" in manifest["source"]
    assert "compiler" in manifest["build"]
    assert "driver" in manifest["runtime"]
    assert "toolkit" in manifest["runtime"]
    assert "device" in manifest["runtime"]
    assert "test_counts" in manifest
    assert "diagnostics" in manifest
    assert "numerical_parity" in manifest
    assert "interop_checks" in manifest
    assert manifest["public_artifact_policy"]["raw_environment_dumps_in_public_tree"] == "forbidden"
    assert manifest["public_artifact_policy"]["raw_profiler_data_in_public_tree"] == "forbidden"

    assert benchmark_policy["warmup_iterations"] == 10
    assert benchmark_policy["timed_iterations"] == 30
    assert benchmark_policy["cross_architecture_comparisons"] == "forbidden"

    assert "fail-closed" in runbook.lower()
    assert "terminate the cloud instance" in runbook.lower()
    assert "sanitized derived evidence" in runbook
    assert "private/" in private_readme
    assert "public/" in private_readme
    assert "set -euo pipefail" in runner
    assert "trap cleanup EXIT" in runner
    assert "scripts/audit_public_artifacts.py --path" in runner
    assert "collect_rocm_inventory.sh" in runner
    assert "qualification_manifest.json" in runner


def test_hopper_harness_uses_supported_smoke_benchmark_cli(tmp_path: Path) -> None:
    output_dir = tmp_path / "qualification-harness"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "bundle",
            "--lane",
            "hopper",
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr

    benchmark_policy = json.loads((output_dir / "public" / "benchmark_policy.json").read_text(encoding="utf-8"))
    runner = (output_dir / "scripts" / "run_lane.sh").read_text(encoding="utf-8")

    assert "bench_cuda_kernels.py" in benchmark_policy["benchmark_command"]
    assert "--smoke" in benchmark_policy["benchmark_command"]
    assert "--profile smoke" not in benchmark_policy["benchmark_command"]
    assert "bench_cuda_kernels.py" in runner
    assert "--smoke" in runner
    assert "--profile smoke" not in runner


def test_blackwell_defaults_exclude_kepler_and_match_lane_targets() -> None:
    validate = load_validate_module()
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert validate.DEFAULT_CUDA_ARCHITECTURES == "80,86,89,90,100-real,120"
    assert (
        '_wolfgang_string_option(WOLFGANG_CUDA_ARCHITECTURES WOLFGANG_CUDA_ARCHITECTURES '
        '"80;86;89;90;100-real;120" "CUDA architectures for WOLFGANG_ENABLE_CUDA=ON source builds")'
    ) in cmake
    assert '"70;75;80;86;89;90"' not in cmake


def test_harness_archive_mode_requires_explicit_source_identity(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive-root"
    scripts_dir = archive_root / "scripts"
    scripts_dir.mkdir(parents=True)
    shutil.copy2(SCRIPT, scripts_dir / SCRIPT.name)
    shutil.copy2(ARCHIVE_PORTABILITY, scripts_dir / ARCHIVE_PORTABILITY.name)

    completed = subprocess.run(
        [
            sys.executable,
            str(scripts_dir / SCRIPT.name),
            "bundle",
            "--lane",
            "mi300x",
            "--output-dir",
            str(tmp_path / "bundle"),
        ],
        cwd=archive_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "source" in (completed.stdout + completed.stderr).lower()
    assert "commit" in (completed.stdout + completed.stderr).lower()


def test_harness_archive_mode_accepts_explicit_source_identity(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive-root"
    scripts_dir = archive_root / "scripts"
    scripts_dir.mkdir(parents=True)
    shutil.copy2(SCRIPT, scripts_dir / SCRIPT.name)
    shutil.copy2(ARCHIVE_PORTABILITY, scripts_dir / ARCHIVE_PORTABILITY.name)
    output_dir = tmp_path / "bundle"
    source_commit = "0123456789abcdef0123456789abcdef01234567"

    completed = subprocess.run(
        [
            sys.executable,
            str(scripts_dir / SCRIPT.name),
            "bundle",
            "--lane",
            "mi300x",
            "--output-dir",
            str(output_dir),
            "--source-commit",
            source_commit,
            "--source-tree-state",
            "archive",
        ],
        cwd=archive_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest = json.loads((output_dir / "public" / "qualification_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == {
        "commit": source_commit,
        "short_commit": source_commit[:7],
        "tree_state": "archive",
    }


def test_cuda_inventory_collector_uses_metadata_allowlist() -> None:
    collector = CUDA_COLLECTOR.read_text(encoding="utf-8")

    assert "nvidia-smi -q" not in collector
    assert "env |" not in collector
    assert "hostname" not in collector
    assert "--query-gpu=uuid" not in collector
    assert "nvidia-smi --query-gpu=name,driver_version,compute_cap,pci.bus_id,memory.total" in collector
    assert "nvcc --version" in collector

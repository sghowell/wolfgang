from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "scripts/audit_public_artifacts.py"
ARCHIVE_PORTABILITY = ROOT / "scripts/archive_portability.py"


def run_auditor(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(AUDITOR), *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_auditor_accepts_sanitized_text(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text(
        '{"device": "AMD MI300X", "result_seconds": 0.001}', encoding="utf-8"
    )

    completed = run_auditor("--path", str(tmp_path))

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "public artifact audit passed" in completed.stdout


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("private-path.txt", "/" + "home" + "/researcher/private/run.json"),
        ("private-macos-path.txt", "/" + "Users" + "/researcher/private/run.json"),
        ("address.txt", ".".join(("198", "51", "100", "42"))),
        ("ssh-target.json", '{"ssh_' + 'target": "researcher@benchmark-node"}'),
        ("hostname.json", '{"host' + 'name": "benchmark-node.internal"}'),
        ("environment.txt", "API_" + "TOKEN=private-value\nPATH=/private/bin"),
        ("gpu-uuid.json", '{"gpu_' + 'uuid": "GPU-private-identifier"}'),
    ],
)
def test_auditor_rejects_private_metadata_without_echoing_it(
    tmp_path: Path, filename: str, content: str
) -> None:
    (tmp_path / filename).write_text(content, encoding="utf-8")

    completed = run_auditor("--path", str(tmp_path))

    assert completed.returncode == 1
    assert filename in completed.stdout
    assert content not in completed.stdout


@pytest.mark.parametrize("suffix", [".nsys-rep", ".ncu-rep", ".sqlite", ".db", ".profraw"])
def test_auditor_rejects_profiler_binary_formats(tmp_path: Path, suffix: str) -> None:
    artifact = tmp_path / f"profile{suffix}"
    artifact.write_bytes(b"opaque profiler data")

    completed = run_auditor("--path", str(tmp_path))

    assert completed.returncode == 1
    assert artifact.name in completed.stdout


def test_tracked_tree_obeys_public_artifact_policy() -> None:
    completed = run_auditor("--tracked")

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_tracked_tree_archive_mode_requires_manifest_without_git(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(AUDITOR, scripts / AUDITOR.name)
    shutil.copy2(ARCHIVE_PORTABILITY, scripts / ARCHIVE_PORTABILITY.name)
    (repository / "README.md").write_text("safe\n", encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(scripts / AUDITOR.name), "--tracked"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "manifest" in (completed.stdout + completed.stderr).lower()


def test_tracked_tree_archive_mode_uses_explicit_manifest(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(AUDITOR, scripts / AUDITOR.name)
    shutil.copy2(ARCHIVE_PORTABILITY, scripts / ARCHIVE_PORTABILITY.name)
    (repository / "README.md").write_text("safe\n", encoding="utf-8")
    manifest = repository / "tracked-files.txt"
    manifest.write_text("README.md\nscripts/audit_public_artifacts.py\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(scripts / AUDITOR.name),
            "--tracked",
            "--tracked-manifest",
            str(manifest),
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "public artifact audit passed" in completed.stdout


def test_tracked_tree_rejects_review_artifact_evidence_roots(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(AUDITOR, scripts / AUDITOR.name)
    shutil.copy2(ARCHIVE_PORTABILITY, scripts / ARCHIVE_PORTABILITY.name)
    review_artifact = repository / "review-artifacts" / "candidate" / "remote-artifacts" / "public"
    review_artifact.mkdir(parents=True)
    (review_artifact / "summary.json").write_text('{"status": "sanitized"}\n', encoding="utf-8")
    manifest = repository / "tracked-files.txt"
    manifest.write_text(
        "review-artifacts/candidate/remote-artifacts/public/summary.json\n"
        "scripts/audit_public_artifacts.py\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(scripts / AUDITOR.name),
            "--tracked",
            "--tracked-manifest",
            str(manifest),
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "review-artifacts/candidate/remote-artifacts/public/summary.json: review-artifacts-root" in completed.stdout


def test_history_audit_detects_sensitive_content_removed_from_tip(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(AUDITOR, scripts / AUDITOR.name)
    shutil.copy2(ARCHIVE_PORTABILITY, scripts / ARCHIVE_PORTABILITY.name)
    subprocess.run(["git", "init", "-b", "main"], cwd=repository, check=True, capture_output=True)
    environment = os.environ | {
        "GIT_AUTHOR_NAME": "Policy Test",
        "GIT_AUTHOR_EMAIL": "policy@example.invalid",
        "GIT_COMMITTER_NAME": "Policy Test",
        "GIT_COMMITTER_EMAIL": "policy@example.invalid",
    }
    removed = repository / "removed.txt"
    removed.write_text("/" + "home" + "/researcher/private/result.json", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repository, check=True, env=environment)
    subprocess.run(
        ["git", "commit", "-m", "historical evidence"],
        cwd=repository,
        check=True,
        env=environment,
        capture_output=True,
    )
    removed.unlink()
    subprocess.run(["git", "add", "-u"], cwd=repository, check=True, env=environment)
    subprocess.run(
        ["git", "commit", "-m", "remove evidence"],
        cwd=repository,
        check=True,
        env=environment,
        capture_output=True,
    )

    completed = subprocess.run(
        [sys.executable, str(scripts / AUDITOR.name), "--history"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "removed.txt: private-home-path" in completed.stdout
    assert "researcher" not in completed.stdout


def test_remote_inventory_collector_uses_metadata_allowlist() -> None:
    collector = (ROOT / "tools/remote/collect_rocm_inventory.sh").read_text(encoding="utf-8")

    assert "uname -a" not in collector
    assert "rocminfo" not in collector
    assert "df -h" not in collector
    assert "env |" not in collector
    assert "--showuniqueid" not in collector
    assert "uname -s" in collector
    assert "--showproductname" in collector


def test_repo_validation_runs_public_artifact_gate() -> None:
    validator = (ROOT / "scripts/validate.py").read_text(encoding="utf-8")

    assert '"public artifact policy"' in validator
    assert '"scripts/audit_public_artifacts.py", "--tracked"' in validator


def test_sdist_contains_only_build_material(tmp_path: Path) -> None:
    archive = tmp_path / "wolfgang-quantum-example.tar.gz"
    with tarfile.open(archive, "w:gz") as package:
        safe_file = tmp_path / "pyproject.toml"
        safe_file.write_text("[build-system]\n", encoding="utf-8")
        package.add(safe_file, arcname="wolfgang-quantum-example/pyproject.toml")
        forbidden_file = tmp_path / "test_internal.py"
        forbidden_file.write_text("internal", encoding="utf-8")
        package.add(forbidden_file, arcname="wolfgang-quantum-example/tests/test_internal.py")

    completed = run_auditor("--sdist", str(archive))

    assert completed.returncode == 1
    assert "tests/test_internal.py" in completed.stdout


def test_pyproject_excludes_gitattributes_from_sdist() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '".gitattributes"' in pyproject

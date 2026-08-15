from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ARCHIVE_SOURCE_IDENTITY = Path("scripts/archive_source_identity.json")
TRACKED_FILES_MANIFEST = Path("scripts/tracked_files_manifest.txt")


def validate_commit(commit: str) -> str:
    normalized = commit.strip().lower()
    if not COMMIT_RE.fullmatch(normalized):
        raise SystemExit(
            "source commit must be an exact 40-character lowercase hexadecimal Git commit id"
        )
    return normalized


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _read_archive_identity(root: Path) -> dict[str, str] | None:
    path = root / ARCHIVE_SOURCE_IDENTITY
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    commit = str(payload.get("commit", "")).strip().lower()
    tree_state = str(payload.get("tree_state", "")).strip()
    if not commit or commit.startswith("$format:"):
        return None
    return {
        "commit": validate_commit(commit),
        "short_commit": str(payload.get("short_commit", "")).strip() or commit[:7],
        "tree_state": tree_state or "archive",
    }


def resolve_source_snapshot(
    root: Path,
    *,
    source_commit: str | None = None,
    source_tree_state: str | None = None,
) -> dict[str, str]:
    if source_commit is not None:
        commit = validate_commit(source_commit)
        return {
            "commit": commit,
            "short_commit": commit[:7],
            "tree_state": source_tree_state or "archive",
        }
    try:
        commit = validate_commit(_git_output(root, "rev-parse", "HEAD"))
        short = _git_output(root, "rev-parse", "--short", "HEAD")
        dirty = "dirty" if _git_output(root, "status", "--short") else "clean"
        return {"commit": commit, "short_commit": short, "tree_state": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        archive_identity = _read_archive_identity(root)
        if archive_identity is not None:
            return archive_identity
        raise SystemExit(
            "source commit identity is unavailable without .git metadata; rerun with --source-commit <40-hex> "
            "or include scripts/archive_source_identity.json from a git archive export-subst bundle"
        )


def default_tracked_manifest(root: Path) -> Path:
    return root / TRACKED_FILES_MANIFEST

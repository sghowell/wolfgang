#!/usr/bin/env python3
"""Audit tracked files and source distributions for public-release safety.

The auditor reports only paths and rule names. It intentionally never prints
matched text, so it is safe to run while investigating a possible disclosure.
"""

from __future__ import annotations

import argparse
import io
import ipaddress
import json
import re
import subprocess
import sys
import tarfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from archive_portability import default_tracked_manifest

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = (".nsys-rep", ".ncu-rep", ".sqlite", ".db", ".profraw", ".profdata")
FORBIDDEN_EVIDENCE_PARTS = {"raw", "logs", "profiler"}
FORBIDDEN_TRACKED_ROOTS = ("review-artifacts/",)
PRIVATE_PATH = re.compile(r"/(?:Users|home)/[^/\s]+/")
IPV4_CANDIDATE = re.compile(r"(?<![\w.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![\w.])")
ENV_ASSIGNMENT = re.compile(r"(?m)^[A-Z][A-Z0-9_]{2,}=[^\n]+$")
SENSITIVE_JSON_KEYS = {
    "device_uuid",
    "gpu_uuid",
    "host_display_name",
    "hostname",
    "node_name",
    "remote_path",
    "ssh_target",
}
SDIST_ALLOWED_FILES = {
    "CHANGELOG.md",
    "CITATION.cff",
    "CMakeLists.txt",
    "CODE_OF_CONDUCT.md",
    "GOVERNANCE.md",
    "LICENSE",
    "PKG-INFO",
    "README.md",
    "SECURITY.md",
    "SUPPORT.md",
    "mkdocs.yml",
    "pyproject.toml",
}
SDIST_ALLOWED_ROOTS = {"bindings", "include", "python", "src", "third_party"}
SDIST_MAX_FILES = 150
SDIST_MAX_UNCOMPRESSED_BYTES = 5 * 1024 * 1024
SELF_SCAN_EXCLUSIONS = {
    "scripts/audit_public_artifacts.py",
    "tests/test_public_artifact_policy.py",
}


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    rule: str


def valid_ipv4_present(text: str) -> bool:
    for match in IPV4_CANDIDATE.finditer(text):
        try:
            ipaddress.ip_address(match.group(0))
        except ValueError:
            continue
        return True
    return False


def sensitive_json_key_present(value: object) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in SENSITIVE_JSON_KEYS:
                return True
            if sensitive_json_key_present(child):
                return True
    elif isinstance(value, list):
        return any(sensitive_json_key_present(child) for child in value)
    return False


def scan_content(display_path: str, data: bytes) -> list[Finding]:
    findings: list[Finding] = []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return findings

    if PRIVATE_PATH.search(text):
        findings.append(Finding(display_path, "private-home-path"))
    if valid_ipv4_present(text):
        findings.append(Finding(display_path, "ip-address"))

    suffix = PurePosixPath(display_path).suffix.lower()
    if suffix in {".json"}:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = None
        if sensitive_json_key_present(value):
            findings.append(Finding(display_path, "sensitive-metadata-key"))

    evidence_file = suffix in {".log", ".stdout", ".stderr", ".txt"}
    if evidence_file and ENV_ASSIGNMENT.search(text):
        findings.append(Finding(display_path, "environment-dump"))
    return findings


def scan_files(files: Iterable[tuple[str, bytes]], *, tracked: bool = False) -> list[Finding]:
    findings: set[Finding] = set()
    for display_path, data in files:
        normalized = display_path.replace("\\", "/")
        path = PurePosixPath(normalized)
        lower_name = path.name.lower()
        if tracked and normalized.startswith(FORBIDDEN_TRACKED_ROOTS):
            findings.add(Finding(normalized, "review-artifacts-root"))
        if lower_name.endswith(FORBIDDEN_SUFFIXES):
            findings.add(Finding(normalized, "forbidden-profiler-format"))
        if tracked and normalized.startswith("docs/benchmarks/data/"):
            relative_parts = set(path.parts[3:-1])
            if relative_parts & FORBIDDEN_EVIDENCE_PARTS:
                findings.add(Finding(normalized, "raw-evidence-directory"))
        if not (tracked and normalized in SELF_SCAN_EXCLUSIONS):
            findings.update(scan_content(normalized, data))
    return sorted(findings)


def _tracked_files_from_names(root: Path, names: Iterable[str]) -> list[tuple[str, bytes]]:
    result: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    for name in names:
        normalized = name.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        path = root / normalized
        if not path.is_file():
            raise SystemExit(
                f"tracked manifest entry does not exist in this archive checkout: {normalized}"
            )
        result.append((normalized, path.read_bytes()))
    return result


def tracked_files(root: Path = ROOT, *, tracked_manifest: Path | None = None) -> list[tuple[str, bytes]]:
    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        manifest_path = tracked_manifest or default_tracked_manifest(root)
        if not manifest_path.is_file():
            raise SystemExit(
                "tracked tree enumeration requires .git metadata or an explicit tracked manifest; "
                "rerun with --tracked-manifest <path> or include scripts/tracked_files_manifest.txt in the archive"
            )
        return _tracked_files_from_names(
            root,
            manifest_path.read_text(encoding="utf-8").splitlines(),
        )

    return _tracked_files_from_names(
        root,
        [raw_name.decode("utf-8") for raw_name in completed.stdout.split(b"\0") if raw_name],
    )


def history_files(root: Path = ROOT) -> list[tuple[str, bytes]]:
    """Return each path/blob pair reachable from any local Git ref."""
    listed = subprocess.run(
        ["git", "rev-list", "--objects", "--all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    paths_by_object: dict[str, set[str]] = {}
    for line in listed.stdout.splitlines():
        fields = line.split(" ", 1)
        if len(fields) == 2:
            paths_by_object.setdefault(fields[0], set()).add(fields[1])

    object_ids = sorted(paths_by_object)
    if not object_ids:
        return []
    batch_input = "".join(f"{object_id}\n" for object_id in object_ids)
    checked = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype)"],
        cwd=root,
        input=batch_input,
        check=True,
        capture_output=True,
        text=True,
    )
    blob_ids = [
        object_id
        for object_id, object_type in (line.split() for line in checked.stdout.splitlines())
        if object_type == "blob"
    ]
    contents = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=root,
        input="".join(f"{object_id}\n" for object_id in blob_ids).encode(),
        check=True,
        capture_output=True,
    ).stdout
    stream = io.BytesIO(contents)
    result: list[tuple[str, bytes]] = []
    for expected_id in blob_ids:
        header = stream.readline().decode("ascii").strip().split()
        if len(header) != 3 or header[0] != expected_id or header[1] != "blob":
            raise RuntimeError("unexpected response from git cat-file --batch")
        data = stream.read(int(header[2]))
        if stream.read(1) != b"\n":
            raise RuntimeError("malformed response from git cat-file --batch")
        result.extend((path, data) for path in sorted(paths_by_object[expected_id]))
    return result


def directory_files(root: Path) -> list[tuple[str, bytes]]:
    if root.is_file():
        return [(root.name, root.read_bytes())]
    return [
        (path.relative_to(root).as_posix(), path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def strip_sdist_root(name: str) -> str:
    parts = PurePosixPath(name).parts
    return PurePosixPath(*parts[1:]).as_posix() if len(parts) > 1 else ""


def scan_sdist(path: Path) -> list[Finding]:
    findings: set[Finding] = set()
    members: list[tuple[str, bytes]] = []
    total_size = 0
    with tarfile.open(path, "r:gz") as archive:
        regular = [member for member in archive.getmembers() if member.isfile()]
        if len(regular) > SDIST_MAX_FILES:
            findings.add(Finding(path.name, "sdist-file-count"))
        for member in regular:
            relative = strip_sdist_root(member.name)
            total_size += member.size
            relative_path = PurePosixPath(relative)
            if relative and not (
                relative in SDIST_ALLOWED_FILES
                or (relative_path.parts and relative_path.parts[0] in SDIST_ALLOWED_ROOTS)
            ):
                findings.add(Finding(relative, "sdist-non-build-material"))
            extracted = archive.extractfile(member)
            if extracted is not None:
                members.append((relative, extracted.read()))
    if total_size > SDIST_MAX_UNCOMPRESSED_BYTES:
        findings.add(Finding(path.name, "sdist-uncompressed-size"))
    findings.update(scan_files(members))
    return sorted(findings)


def print_result(findings: list[Finding]) -> int:
    if not findings:
        print("public artifact audit passed")
        return 0
    print(f"public artifact audit failed: {len(findings)} finding(s)")
    for finding in findings:
        print(f"{finding.path}: {finding.rule}")
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tracked", action="store_true", help="scan Git-tracked files")
    source.add_argument("--history", action="store_true", help="scan blobs reachable from Git refs")
    source.add_argument("--path", type=Path, help="scan a file or directory")
    source.add_argument("--sdist", type=Path, help="scan a .tar.gz source distribution")
    parser.add_argument(
        "--tracked-manifest",
        type=Path,
        help="newline-delimited tracked file list for archive/no-.git --tracked scans",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.tracked:
        return print_result(scan_files(tracked_files(tracked_manifest=args.tracked_manifest), tracked=True))
    if args.history:
        return print_result(scan_files(history_files(), tracked=True))
    if args.path is not None:
        return print_result(scan_files(directory_files(args.path)))
    return print_result(scan_sdist(args.sdist))


if __name__ == "__main__":
    sys.exit(main())

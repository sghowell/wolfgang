#!/usr/bin/env python3
"""Write a sorted SHA256 manifest for Wolfgang release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

SDIST_ARTIFACT_PREFIX = "wolfgang-quantum-"
WHEEL_ARTIFACT_PREFIX = "wolfgang_quantum-"
EXPECTED_PYTHON_TAGS = ("cp310", "cp311", "cp312")
EXPECTED_PLATFORM_TARGETS = ("manylinux_x86_64", "macos_arm64")
EXPECTED_CPU_WHEEL_TARGETS = frozenset(
    (python_tag, platform_target)
    for python_tag in EXPECTED_PYTHON_TAGS
    for platform_target in EXPECTED_PLATFORM_TARGETS
)


def artifact_paths(dist_dir: Path) -> list[Path]:
    paths = []
    for path in dist_dir.iterdir():
        if not path.is_file():
            continue
        if not (
            path.name.startswith(SDIST_ARTIFACT_PREFIX)
            or path.name.startswith(WHEEL_ARTIFACT_PREFIX)
        ):
            continue
        if path.name.endswith(".checksums.txt"):
            continue
        if path.suffix == ".whl" or path.name.endswith(".tar.gz"):
            paths.append(path)
    return sorted(paths, key=lambda path: path.name)


def artifact_version(path: Path) -> str:
    if path.name.endswith(".tar.gz"):
        match = re.fullmatch(r"wolfgang(?:-|_)quantum-(.+)\.tar\.gz", path.name)
    else:
        match = re.fullmatch(r"wolfgang_quantum-(.+?)-[^/]+\.whl", path.name)
    if match is None:
        raise SystemExit(f"could not infer Wolfgang version from artifact: {path.name}")
    return match.group(1)


def classify_platform_tag(platform_tag: str) -> str | None:
    platform_tags = platform_tag.split(".")
    if any(
        tag.startswith("manylinux") and tag.endswith("_x86_64")
        for tag in platform_tags
    ):
        return "manylinux_x86_64"
    if any(
        tag.startswith("macosx") and tag.endswith("_arm64")
        for tag in platform_tags
    ):
        return "macos_arm64"
    return None


def wheel_target(path: Path) -> tuple[str, str]:
    match = re.fullmatch(
        r"wolfgang_quantum-(?P<version>.+?)-(?P<python>cp\d+)-(?P<abi>[^-]+)-(?P<platform>.+)\.whl",
        path.name,
    )
    if match is None:
        raise SystemExit(f"could not parse Wolfgang wheel tag: {path.name}")

    python_tag = match.group("python")
    abi_tag = match.group("abi")
    if abi_tag != python_tag:
        raise SystemExit(
            f"expected wheel ABI tag to match Python tag for {path.name}; "
            f"found {abi_tag!r}"
        )

    platform_target = classify_platform_tag(match.group("platform"))
    if platform_target is None:
        raise SystemExit(f"unsupported CPU wheel platform tag in {path.name}")
    return python_tag, platform_target


def format_target(target: tuple[str, str]) -> str:
    return f"{target[0]}/{target[1]}"


def validate_cpu_wheelhouse(artifacts: list[Path]) -> None:
    sdists = [path for path in artifacts if path.name.endswith(".tar.gz")]
    wheels = [path for path in artifacts if path.suffix == ".whl"]

    if len(sdists) != 1:
        names = ", ".join(path.name for path in artifacts) or "none"
        raise SystemExit(
            "expected exactly one Wolfgang source distribution in CPU wheelhouse; "
            f"found: {names}"
        )

    targets: dict[tuple[str, str], Path] = {}
    for wheel in wheels:
        target = wheel_target(wheel)
        if target in targets:
            raise SystemExit(
                "duplicate CPU wheel target "
                f"{format_target(target)}: {targets[target].name}, {wheel.name}"
            )
        targets[target] = wheel

    observed_targets = set(targets)
    missing = EXPECTED_CPU_WHEEL_TARGETS - observed_targets
    extra = observed_targets - EXPECTED_CPU_WHEEL_TARGETS
    if missing or extra:
        message_parts = []
        if missing:
            message_parts.append(
                "missing CPU wheel targets: "
                + ", ".join(format_target(target) for target in sorted(missing))
            )
        if extra:
            message_parts.append(
                "unexpected CPU wheel targets: "
                + ", ".join(format_target(target) for target in sorted(extra))
            )
        raise SystemExit("; ".join(message_parts))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksum_manifest(
    dist_dir: Path,
    *,
    require_cpu_wheelhouse: bool = False,
) -> Path:
    dist_dir = dist_dir.resolve()
    artifacts = artifact_paths(dist_dir)
    if not artifacts:
        raise SystemExit(f"no Wolfgang release artifacts found in {dist_dir}")

    versions = {artifact_version(path) for path in artifacts}
    if len(versions) != 1:
        raise SystemExit(
            "release artifact directory contains multiple Wolfgang versions: "
            + ", ".join(sorted(versions))
        )
    if require_cpu_wheelhouse:
        validate_cpu_wheelhouse(artifacts)
    version = next(iter(versions))
    manifest = dist_dir / f"wolfgang-quantum-{version}.checksums.txt"

    lines = [f"{sha256_file(path)}  {path.name}" for path in artifacts]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=Path("dist"),
        help="Directory containing Wolfgang release artifacts.",
    )
    parser.add_argument(
        "--require-cpu-wheelhouse",
        action="store_true",
        help=(
            "Require the final 0.1.0 CPU wheelhouse shape: one sdist plus "
            "CPython 3.10-3.12 wheels for manylinux x86_64 and macOS arm64."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_checksum_manifest(
        args.dist_dir,
        require_cpu_wheelhouse=args.require_cpu_wheelhouse,
    )


if __name__ == "__main__":
    main()

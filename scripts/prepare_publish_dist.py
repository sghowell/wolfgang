#!/usr/bin/env python3
"""Prepare the package-index upload directory from collected release artifacts."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from write_release_checksums import artifact_paths


def prepare_publish_dist(dist_dir: Path, publish_dir: Path) -> list[Path]:
    dist_dir = dist_dir.resolve()
    publish_dir = publish_dir.resolve()
    if dist_dir == publish_dir:
        raise SystemExit("publish directory must be separate from collected dist")

    artifacts = artifact_paths(dist_dir)
    if not artifacts:
        raise SystemExit(f"no package-index artifacts found in {dist_dir}")

    publish_dir.mkdir(parents=True, exist_ok=True)
    for path in publish_dir.iterdir():
        if path.is_file() and (path.suffix == ".whl" or path.name.endswith(".tar.gz")):
            path.unlink()

    copied = []
    for artifact in artifacts:
        target = publish_dir / artifact.name
        shutil.copy2(artifact, target)
        copied.append(target)
        print(target)
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=Path("dist"),
        help="Collected release artifact directory.",
    )
    parser.add_argument(
        "--publish-dir",
        type=Path,
        default=Path("publish-dist"),
        help="Output directory containing only .whl and .tar.gz upload artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_publish_dist(args.dist_dir, args.publish_dir)


if __name__ == "__main__":
    main()

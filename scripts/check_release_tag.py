#!/usr/bin/env python3
"""Require a publish ref to exactly match the version in pyproject.toml."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"


def project_version() -> str:
    """Read the single authoritative project version without extra dependencies."""
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', text, re.MULTILINE)
    if match is None:
        raise SystemExit("could not find project version in pyproject.toml")
    return match.group(1)


def expected_release_tag() -> str:
    """Return the only tag authorized to publish this source tree."""
    return f"v{project_version()}"


def check_release_tag(ref_type: str, ref_name: str) -> list[str]:
    """Return publication-binding failures for the supplied GitHub ref."""
    if ref_type != "tag":
        return [f"package-index publication requires a tag ref; got {ref_type}"]
    expected = expected_release_tag()
    if ref_name != expected:
        return [
            "release tag must exactly match project version: "
            f"expected {expected}, got {ref_name}"
        ]
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref_type")
    parser.add_argument("ref_name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures = check_release_tag(args.ref_type, args.ref_name)
    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(1)
    print(f"release ref matches project version: {args.ref_name}")


if __name__ == "__main__":
    main()

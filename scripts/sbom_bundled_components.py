#!/usr/bin/env python3
"""Add the native code Wolfgang ships to the release CycloneDX SBOM.

`cyclonedx-py requirements` lists the Python dependencies only. Release CPU
wheels also bundle the oneTBB runtime, and every build compiles in the vendored
DLPack header. This script reads their versions from the files that pin them
(`scripts/build_onetbb.sh`, `third_party/dlpack/include/dlpack/dlpack.h`) and
adds them as components to an existing CycloneDX JSON document:

    python scripts/sbom_bundled_components.py sbom/wolfgang-runtime.cdx.json

Running it again replaces the same components instead of duplicating them.
With --validate it then checks the result against the document's CycloneDX
schema, using cyclonedx-python-lib (installed with cyclonedx-bom).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BUNDLED_PROPERTY = "wolfgang:bundled-in"


def _shell_variable(text: str, name: str) -> str:
    match = re.search(rf'^{name}="([^"]+)"$', text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"{name} not found in scripts/build_onetbb.sh")
    return match.group(1)


def onetbb_component(root: Path = ROOT) -> dict[str, Any]:
    script = (root / "scripts" / "build_onetbb.sh").read_text(encoding="utf-8")
    tag = _shell_variable(script, "ONETBB_TAG")
    commit = _shell_variable(script, "ONETBB_COMMIT")
    repository = _shell_variable(script, "ONETBB_REPOSITORY")
    version = tag.removeprefix("v")
    return {
        "type": "library",
        "bom-ref": f"pkg:github/uxlfoundation/oneTBB@{tag}",
        "group": "uxlfoundation",
        "name": "oneTBB",
        "version": version,
        "purl": f"pkg:github/uxlfoundation/oneTBB@{tag}",
        "licenses": [{"license": {"id": "Apache-2.0"}}],
        "externalReferences": [{"type": "vcs", "url": repository, "comment": f"commit {commit}"}],
        "properties": [
            {"name": BUNDLED_PROPERTY, "value": "release CPU wheels (core tbb runtime library)"},
            {"name": "wolfgang:source-commit", "value": commit},
            {"name": "wolfgang:third-party-notices", "value": "third_party/onetbb/third-party-programs.txt"},
        ],
    }


def dlpack_component(root: Path = ROOT) -> dict[str, Any]:
    header = (root / "third_party" / "dlpack" / "include" / "dlpack" / "dlpack.h").read_text(encoding="utf-8")
    major = re.search(r"^#define DLPACK_MAJOR_VERSION (\d+)$", header, flags=re.MULTILINE)
    minor = re.search(r"^#define DLPACK_MINOR_VERSION (\d+)$", header, flags=re.MULTILINE)
    if major is None or minor is None:
        raise ValueError("DLPack version macros not found in dlpack.h")
    tag = f"v{major.group(1)}.{minor.group(1)}"
    return {
        "type": "library",
        "bom-ref": f"pkg:github/dmlc/dlpack@{tag}",
        "group": "dmlc",
        "name": "dlpack",
        "version": tag.removeprefix("v"),
        "purl": f"pkg:github/dmlc/dlpack@{tag}",
        "licenses": [{"license": {"id": "Apache-2.0"}}],
        "externalReferences": [{"type": "vcs", "url": "https://github.com/dmlc/dlpack"}],
        "properties": [{"name": BUNDLED_PROPERTY, "value": "every build (vendored header, compiled in)"}],
    }


def bundled_components(root: Path = ROOT) -> list[dict[str, Any]]:
    return [onetbb_component(root), dlpack_component(root)]


def merge(document: dict[str, Any], components: list[dict[str, Any]]) -> dict[str, Any]:
    if document.get("bomFormat") != "CycloneDX":
        raise ValueError("not a CycloneDX JSON document")
    refs = {component["bom-ref"] for component in components}
    kept = [c for c in document.get("components", []) if c.get("bom-ref") not in refs]
    document["components"] = kept + components
    return document


def validate(text: str, spec_version: str) -> None:
    from cyclonedx.schema import SchemaVersion
    from cyclonedx.validation.json import JsonStrictValidator

    schema = SchemaVersion.from_version(spec_version)
    error = JsonStrictValidator(schema).validate_str(text)
    if error is not None:
        raise ValueError(f"SBOM fails CycloneDX {spec_version} validation: {error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sbom", type=Path, help="CycloneDX JSON file to update in place")
    parser.add_argument("--validate", action="store_true", help="check the result against the CycloneDX schema")
    args = parser.parse_args(argv)
    document = json.loads(args.sbom.read_text(encoding="utf-8"))
    merged = merge(document, bundled_components())
    text = json.dumps(merged, indent=2) + "\n"
    if args.validate:
        validate(text, str(merged.get("specVersion", "")))
    args.sbom.write_text(text, encoding="utf-8")
    names = ", ".join(f"{c['name']} {c['version']}" for c in bundled_components())
    sys.stdout.write(f"added {names} to {args.sbom}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

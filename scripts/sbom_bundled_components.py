#!/usr/bin/env python3
"""Add the native code Wolfgang ships to the release CycloneDX SBOM.

`cyclonedx-py requirements` lists the Python dependencies only. Release CPU
wheels also bundle the oneTBB runtime, and the Python extension compiles in the
vendored DLPack header. This script reads their versions from the files that
pin them (`scripts/build_onetbb.sh`, `third_party/dlpack/include/dlpack/dlpack.h`)
and adds them as components, with dependency-graph entries, to an existing
CycloneDX JSON document:

    python scripts/sbom_bundled_components.py --validate sbom/wolfgang-runtime.cdx.json

Components with the same name replace any earlier copy, whatever its version.
With --validate the result is checked against the document's CycloneDX schema,
using cyclonedx-python-lib (installed with cyclonedx-bom), before it is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BUNDLED_PROPERTY = "wolfgang:bundled-in"
DLPACK_HEADER = Path("third_party") / "dlpack" / "include" / "dlpack" / "dlpack.h"


def _shell_variable(text: str, name: str, shape: str) -> str:
    match = re.search(rf'^{name}="([^"]+)"$', text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"{name} not found in scripts/build_onetbb.sh")
    value = match.group(1)
    if re.fullmatch(shape, value) is None:
        raise ValueError(f"{name}={value!r} in scripts/build_onetbb.sh does not look like a pinned value")
    return value


def onetbb_component(root: Path = ROOT) -> dict[str, Any]:
    script = (root / "scripts" / "build_onetbb.sh").read_text(encoding="utf-8")
    tag = _shell_variable(script, "ONETBB_TAG", r"v\d+\.\d+\.\d+")
    commit = _shell_variable(script, "ONETBB_COMMIT", r"[0-9a-f]{40}")
    repository = _shell_variable(script, "ONETBB_REPOSITORY", r"https://github\.com/[\w.-]+/[\w.-]+?(\.git)?")
    owner, name = re.fullmatch(r"https://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?", repository).groups()
    purl = f"pkg:github/{owner.lower()}/{name.lower()}@{tag}"  # purl-spec: github names are lowercase
    web = f"https://github.com/{owner}/{name}"
    return {
        "type": "library",
        "bom-ref": purl,
        "group": owner,
        "name": name,
        "version": tag.removeprefix("v"),
        "purl": purl,
        # Core oneTBB is Apache-2.0. On x86_64 the runtime also compiles in the
        # ITT Notify static API (BSD-3-Clause); arm64 builds leave it out.
        "licenses": [{"expression": "Apache-2.0 AND BSD-3-Clause"}],
        "externalReferences": [
            {"type": "vcs", "url": repository, "comment": f"commit {commit}"},
            {"type": "license", "url": f"{web}/blob/{commit}/LICENSE.txt"},
            {"type": "license", "url": f"{web}/blob/{commit}/third-party-programs.txt",
             "comment": "BSD-3-Clause ITT Notify notice; applies to x86_64 builds"},
        ],
        "properties": [
            {"name": BUNDLED_PROPERTY, "value": "release CPU wheels (core tbb runtime library)"},
            {"name": "wolfgang:source-commit", "value": commit},
        ],
    }


def dlpack_component(root: Path = ROOT) -> dict[str, Any]:
    header_bytes = (root / DLPACK_HEADER).read_bytes()
    header = header_bytes.decode("utf-8")
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
        "hashes": [{"alg": "SHA-256", "content": hashlib.sha256(header_bytes).hexdigest()}],
        "licenses": [{"license": {"id": "Apache-2.0"}}],
        "externalReferences": [{"type": "vcs", "url": "https://github.com/dmlc/dlpack"}],
        "properties": [
            {"name": BUNDLED_PROPERTY, "value": "the Python extension in every build (vendored header, compiled in)"},
        ],
    }


def bundled_components(root: Path = ROOT) -> list[dict[str, Any]]:
    return [onetbb_component(root), dlpack_component(root)]


def merge(document: dict[str, Any], components: list[dict[str, Any]]) -> dict[str, Any]:
    if document.get("bomFormat") != "CycloneDX":
        raise ValueError("not a CycloneDX JSON document")
    names = {component["name"].lower() for component in components}
    stale = {c.get("bom-ref") for c in document.get("components", []) if str(c.get("name", "")).lower() in names}
    document["components"] = [
        c for c in document.get("components", []) if str(c.get("name", "")).lower() not in names
    ] + components
    dependencies = [d for d in document.get("dependencies", []) if d.get("ref") not in stale]
    document["dependencies"] = dependencies + [{"ref": c["bom-ref"]} for c in components]
    return document


def validate(text: str, spec_version: str) -> None:
    if not spec_version:
        raise ValueError("the SBOM has no specVersion to validate against")
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

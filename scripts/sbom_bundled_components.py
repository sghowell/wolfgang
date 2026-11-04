#!/usr/bin/env python3
"""Write one CycloneDX SBOM per release artifact kind.

`cyclonedx-py requirements` lists the Python dependencies only, as unpinned
components. The release ships three kinds of artifact with different contents:
the sdist, which carries the vendored DLPack header as source; manylinux x86_64
wheels, whose extension compiles in DLPack and nanobind's support library (with
nanobind's vendored tsl-robin-map) and which bundle the oneTBB runtime with
oneTBB's BSD-3-Clause ITT Notify code compiled in; and macOS arm64 wheels, the
same except that their oneTBB leaves ITT Notify out. This script reads the
cyclonedx-py document and writes, for each kind,

- wolfgang-quantum as `metadata.component`, with the version, license
  expression and repository from pyproject.toml;
- each Python dependency as an external component (`isExternal`) with the
  version range pyproject.toml accepts, in vers syntax, since none is bundled;
- the native code that kind ships, with versions read from the files that pin
  them (`scripts/build_onetbb.sh`, `third_party/dlpack/include/dlpack/dlpack.h`,
  `third_party/nanobind/README.md` checked against the `pyproject.toml` pin);
- a dependency graph rooted at wolfgang-quantum.

    cyclonedx-py requirements runtime-requirements.txt --spec-version 1.7 \\
        --output-format JSON --output-file requirements.cdx.json
    python scripts/sbom_bundled_components.py --validate \\
        --requirements requirements.cdx.json --out-dir sbom

The output files are named `wolfgang-quantum-<kind>.cdx.json`. The release
workflow attests each one against the files matched by its kind's
`subject_path`. Components in the input with the name of a bundled component
are replaced, whatever their version. With --validate each document is checked
against the CycloneDX schema, using cyclonedx-python-lib (installed with
cyclonedx-bom), before anything is written.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any, NamedTuple

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
BUNDLED_PROPERTY = "wolfgang:bundled-in"
DLPACK_HEADER = Path("third_party") / "dlpack" / "include" / "dlpack" / "dlpack.h"
NANOBIND_README = Path("third_party") / "nanobind" / "README.md"
# External components with a version range need CycloneDX 1.7.
SPEC_VERSION = "1.7"


class ArtifactKind(NamedTuple):
    # "sdist", or the build-wheels matrix platform the wheels come from.
    name: str
    # Glob for the release files this kind's SBOM is attested against.
    subject_path: str
    # License expression of the bundled oneTBB runtime; None when not bundled.
    onetbb_licenses: str | None
    # Where the vendored DLPack header ends up.
    dlpack_bundled_in: str
    # Whether the artifact contains the compiled extension (and so nanobind).
    compiled: bool


ARTIFACT_KINDS = (
    ArtifactKind(
        "sdist",
        "dist/*.tar.gz",
        None,
        "the source distribution (vendored header source)",
        False,
    ),
    ArtifactKind(
        "manylinux-x86_64",
        "dist/*-manylinux*_x86_64.whl",
        # Core oneTBB is Apache-2.0; on x86_64 the runtime also compiles in the
        # ITT Notify static API (BSD-3-Clause).
        "Apache-2.0 AND BSD-3-Clause",
        "the Python extension (vendored header, compiled in)",
        True,
    ),
    ArtifactKind(
        "macos-arm64",
        "dist/*-macosx_*_arm64.whl",
        # arm64 builds leave ITT Notify out.
        "Apache-2.0",
        "the Python extension (vendored header, compiled in)",
        True,
    ),
)


def _shell_variable(text: str, name: str, shape: str) -> str:
    match = re.search(rf'^{name}="([^"]+)"$', text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"{name} not found in scripts/build_onetbb.sh")
    value = match.group(1)
    if re.fullmatch(shape, value) is None:
        raise ValueError(f"{name}={value!r} in scripts/build_onetbb.sh does not look like a pinned value")
    return value


def onetbb_component(root: Path = ROOT, *, licenses: str) -> dict[str, Any]:
    script = (root / "scripts" / "build_onetbb.sh").read_text(encoding="utf-8")
    tag = _shell_variable(script, "ONETBB_TAG", r"v\d+\.\d+\.\d+")
    commit = _shell_variable(script, "ONETBB_COMMIT", r"[0-9a-f]{40}")
    repository = _shell_variable(script, "ONETBB_REPOSITORY", r"https://github\.com/[\w.-]+/[\w.-]+?(\.git)?")
    owner, name = re.fullmatch(r"https://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?", repository).groups()
    purl = f"pkg:github/{owner.lower()}/{name.lower()}@{tag}"  # purl-spec: github names are lowercase
    web = f"https://github.com/{owner}/{name}"
    references = [
        {"type": "vcs", "url": repository, "comment": f"commit {commit}"},
        {"type": "license", "url": f"{web}/blob/{commit}/LICENSE.txt"},
    ]
    if "BSD-3-Clause" in licenses:
        references.append({"type": "license", "url": f"{web}/blob/{commit}/third-party-programs.txt",
                           "comment": "BSD-3-Clause ITT Notify notice"})
    return {
        "type": "library",
        "bom-ref": purl,
        "group": owner,
        "name": name,
        "version": tag.removeprefix("v"),
        "purl": purl,
        "licenses": [{"expression": licenses}],
        "externalReferences": references,
        "properties": [
            {"name": BUNDLED_PROPERTY, "value": "release CPU wheels (core tbb runtime library)"},
            {"name": "wolfgang:source-commit", "value": commit},
        ],
    }


def dlpack_component(root: Path = ROOT, *, bundled_in: str = "the Python extension") -> dict[str, Any]:
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
        "properties": [{"name": BUNDLED_PROPERTY, "value": bundled_in}],
    }


def _build_requirement_pin(root: Path, name: str) -> str:
    requires = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["build-system"]["requires"]
    for requirement in requires:
        match = re.fullmatch(rf"{re.escape(name)}==(\d+\.\d+\.\d+)", requirement.replace(" ", ""))
        if match is not None:
            return match.group(1)
    raise ValueError(f"pyproject.toml [build-system] requires must pin {name} exactly")


def nanobind_components(root: Path = ROOT) -> list[dict[str, Any]]:
    """nanobind's support library and the tsl-robin-map it vendors, both compiled into the extension."""
    readme = (root / NANOBIND_README).read_text(encoding="utf-8")
    nanobind = re.search(r"^nanobind (\d+\.\d+\.\d+) commit ([0-9a-f]{40})$", readme, flags=re.MULTILINE)
    robin_map = re.search(r"^robin-map (\d+\.\d+\.\d+) commit ([0-9a-f]{40})$", readme, flags=re.MULTILINE)
    if nanobind is None or robin_map is None:
        raise ValueError(f"nanobind and robin-map pins not found in {NANOBIND_README}")
    version, commit = nanobind.groups()
    pinned = _build_requirement_pin(root, "nanobind")
    if pinned != version:
        raise ValueError(f"{NANOBIND_README} names nanobind {version}, but pyproject.toml pins {pinned}")
    map_version, map_commit = robin_map.groups()
    map_purl = f"pkg:github/tessil/robin-map@v{map_version}"
    return [
        {
            "type": "library",
            "bom-ref": f"pkg:pypi/nanobind@{version}",
            "name": "nanobind",
            "version": version,
            "purl": f"pkg:pypi/nanobind@{version}",
            "licenses": [{"license": {"id": "BSD-3-Clause"}}],
            "externalReferences": [
                {"type": "vcs", "url": "https://github.com/wjakob/nanobind", "comment": f"commit {commit}"},
            ],
            "properties": [
                {"name": BUNDLED_PROPERTY, "value": "the Python extension (support library, compiled in statically)"},
                {"name": "wolfgang:source-commit", "value": commit},
            ],
        },
        {
            "type": "library",
            "bom-ref": map_purl,
            "group": "Tessil",
            "name": "robin-map",
            "version": map_version,
            "purl": map_purl,
            "licenses": [{"license": {"id": "MIT"}}],
            "externalReferences": [
                {"type": "vcs", "url": "https://github.com/Tessil/robin-map", "comment": f"commit {map_commit}"},
            ],
            "properties": [
                {"name": BUNDLED_PROPERTY, "value": "the Python extension, through nanobind (header-only hash table)"},
                {"name": "wolfgang:source-commit", "value": map_commit},
            ],
        },
    ]


def bundled_components(kind: ArtifactKind, root: Path = ROOT) -> list[dict[str, Any]]:
    components = []
    if kind.onetbb_licenses is not None:
        components.append(onetbb_component(root, licenses=kind.onetbb_licenses))
    components.append(dlpack_component(root, bundled_in=kind.dlpack_bundled_in))
    if kind.compiled:
        components.extend(nanobind_components(root))
    return components


# Components that another listed component pulls in, by name: the root depends
# on everything else directly.
_INDIRECT = {"robin-map": "nanobind"}


def load_project(root: Path = ROOT) -> dict[str, Any]:
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def wolfgang_component(project: dict[str, Any], kind: ArtifactKind) -> dict[str, Any]:
    purl = f"pkg:pypi/{project['name']}@{project['version']}"
    urls = project.get("urls", {})
    references = [{"type": "vcs", "url": urls["Repository"]}] if "Repository" in urls else []
    if "Homepage" in urls:
        references.append({"type": "website", "url": urls["Homepage"]})
    return {
        "type": "library",
        "bom-ref": purl,
        "name": project["name"],
        "version": project["version"],
        "purl": purl,
        "description": project["description"],
        "licenses": [{"expression": project["license"]}],
        "externalReferences": references,
        "properties": [{"name": "wolfgang:release-artifact", "value": kind.name}],
    }


_VERS_COMPARATORS = {">=": ">=", ">": ">", "<=": "<=", "<": "<", "!=": "!=", "==": ""}


def vers_range(ecosystem: str, specifier: str) -> str:
    """Convert a PEP 440 specifier set to a vers range (https://github.com/package-url/vers-spec).

    Accepts one interval: at most one lower bound (``>=``, ``>``) below at most
    one upper bound (``<=``, ``<``), with ``!=`` exclusions strictly inside it
    (SpecifierSet already merges equal ones), or a lone ``==``. vers reads its constraints differently from
    PEP 440's intersection, so anything else (two lower bounds, an empty or
    one-version interval, ``==`` beside other clauses, ``~=``, ``===``,
    wildcards, no constraint at all) raises ValueError rather than producing a
    range that means something else.
    """
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    specifiers = list(SpecifierSet(specifier))
    if not specifiers:
        raise ValueError("an unconstrained requirement has no version range")
    lower = upper = exact = None
    excluded: list[tuple[Version, str]] = []
    for item in specifiers:
        if item.operator not in _VERS_COMPARATORS or "*" in item.version:
            raise ValueError(f"vers cannot express the specifier {item}")
        constraint = (Version(item.version), _VERS_COMPARATORS[item.operator] + item.version)
        if item.operator in (">=", ">"):
            if lower is not None:
                raise ValueError(f"{specifier!r} has more than one lower bound")
            lower = constraint
        elif item.operator in ("<=", "<"):
            if upper is not None:
                raise ValueError(f"{specifier!r} has more than one upper bound")
            upper = constraint
        elif item.operator == "==":
            exact = constraint
        else:
            excluded.append(constraint)
    if exact is not None:
        if len(specifiers) != 1:
            raise ValueError(f"{specifier!r} combines == with other clauses")
        return f"vers:{ecosystem}/{exact[1]}"
    if lower is not None and upper is not None and not lower[0] < upper[0]:
        raise ValueError(f"{specifier!r} is not an interval of more than one version")
    for version, _ in excluded:
        if (lower is not None and not version > lower[0]) or (upper is not None and not version < upper[0]):
            raise ValueError(f"{specifier!r} excludes a version outside its interval")
    # vers lists constraints in ascending version order.
    ordered = [lower] if lower is not None else []
    ordered += sorted(excluded)
    ordered += [upper] if upper is not None else []
    return f"vers:{ecosystem}/" + "|".join(text for _, text in ordered)


def mark_external_requirements(components: list[dict[str, Any]], project: dict[str, Any]) -> None:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    requirements = [Requirement(line) for line in project.get("dependencies", [])]
    names = [canonicalize_name(requirement.name) for requirement in requirements]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        # One component per name: markers that split a dependency would need one each.
        raise ValueError(f"dependencies listed more than once: {', '.join(repeated)}")
    for name, requirement in zip(names, requirements):
        matching = [
            c for c in components
            if canonicalize_name(str(c.get("name", ""))) == name and str(c.get("purl", "")).startswith("pkg:pypi/")
        ]
        if len(matching) != 1:
            raise ValueError(f"expected one {name} component in the requirements SBOM, found {len(matching)}")
        (component,) = matching
        component.pop("version", None)  # version and versionRange are mutually exclusive
        component["bom-ref"] = component["purl"]  # in place of cyclonedx-py's "requirements-L<line>"
        component["isExternal"] = True
        component["versionRange"] = vers_range("pypi", str(requirement.specifier))
        if requirement.marker is not None:
            component.setdefault("properties", []).append(
                {"name": "wolfgang:environment-marker", "value": str(requirement.marker)}
            )


def artifact_sbom(
    requirements: dict[str, Any],
    project: dict[str, Any],
    kind: ArtifactKind,
    root: Path = ROOT,
) -> dict[str, Any]:
    if requirements.get("bomFormat") != "CycloneDX":
        raise ValueError("not a CycloneDX JSON document")
    if requirements.get("specVersion") != SPEC_VERSION:
        raise ValueError(
            f"the requirements SBOM is CycloneDX {requirements.get('specVersion')}; "
            f"run cyclonedx-py with --spec-version {SPEC_VERSION}"
        )
    document = copy.deepcopy(requirements)
    bundled = bundled_components(kind, root)
    # Any kind's bundled names: a stale oneTBB must not survive in the sdist's SBOM.
    names = {c["name"].lower() for other in ARTIFACT_KINDS for c in bundled_components(other, root)}
    python = [c for c in document.get("components", []) if str(c.get("name", "")).lower() not in names]
    mark_external_requirements(python, project)
    subject = wolfgang_component(project, kind)
    components = python + bundled
    document["serialNumber"] = uuid.uuid4().urn
    document.setdefault("metadata", {})["component"] = subject
    document["components"] = components
    refs = {c["name"]: c["bom-ref"] for c in components}
    children: dict[str, list[str]] = {c["bom-ref"]: [] for c in components}
    direct = []
    for component in components:
        parent = _INDIRECT.get(component["name"])
        if parent in refs:
            children[refs[parent]].append(component["bom-ref"])
        else:
            direct.append(component["bom-ref"])
    document["dependencies"] = [{"ref": subject["bom-ref"], "dependsOn": direct}] + [
        {"ref": ref, "dependsOn": dependencies} if dependencies else {"ref": ref}
        for ref, dependencies in children.items()
    ]
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
    parser.add_argument("--requirements", type=Path, required=True,
                        help="CycloneDX 1.7 JSON written by cyclonedx-py requirements")
    parser.add_argument("--out-dir", type=Path, required=True, help="directory for the per-artifact SBOMs")
    parser.add_argument("--validate", action="store_true", help="check each document against the CycloneDX schema")
    args = parser.parse_args(argv)
    requirements = json.loads(args.requirements.read_text(encoding="utf-8"))
    project = load_project()
    texts = {}
    for kind in ARTIFACT_KINDS:
        document = artifact_sbom(requirements, project, kind)
        text = json.dumps(document, indent=2) + "\n"
        if args.validate:
            validate(text, str(document.get("specVersion", "")))
        texts[kind.name] = text
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        path = args.out_dir / f"wolfgang-quantum-{name}.cdx.json"
        path.write_text(text, encoding="utf-8")
        sys.stdout.write(f"wrote {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

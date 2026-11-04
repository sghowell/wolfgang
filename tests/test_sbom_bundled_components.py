"""Each release artifact's SBOM lists what that artifact ships.

`cyclonedx-py` covers Python dependencies only; `scripts/sbom_bundled_components.py`
writes one CycloneDX document per artifact kind (the sdist and each wheel
platform), with wolfgang-quantum as `metadata.component`, the Python
dependencies as external components with their accepted version ranges, and
the native code the artifact bundles: the vendored DLPack header everywhere,
the oneTBB runtime in wheels, with the ITT Notify license only on x86_64.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sbom_bundled_components", ROOT / "scripts" / "sbom_bundled_components.py")
sbom = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sbom)
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
KINDS = ("sdist", "manylinux-x86_64", "macos-arm64")


def test_onetbb_component_matches_the_build_pins_and_notices() -> None:
    script = (ROOT / "scripts" / "build_onetbb.sh").read_text(encoding="utf-8")
    tag = re.search(r'^ONETBB_TAG="([^"]+)"$', script, flags=re.MULTILINE).group(1)
    commit = re.search(r'^ONETBB_COMMIT="([^"]+)"$', script, flags=re.MULTILINE).group(1)
    component = sbom.onetbb_component(licenses="Apache-2.0 AND BSD-3-Clause")
    assert component["version"] == tag.removeprefix("v")
    assert component["purl"] == f"pkg:github/uxlfoundation/onetbb@{tag}"  # lowercase per purl-spec
    assert {"name": "wolfgang:source-commit", "value": commit} in component["properties"]
    # Apache-2.0 core plus the BSD-3-Clause ITT Notify code compiled in on x86_64.
    assert component["licenses"] == [{"expression": "Apache-2.0 AND BSD-3-Clause"}]
    assert any(commit in ref["url"] for ref in component["externalReferences"] if ref["type"] == "license")
    notices = (ROOT / "third_party" / "onetbb" / "README.md").read_text(encoding="utf-8")
    assert tag in notices and commit in notices
    assert "Apache License" in (ROOT / "third_party" / "onetbb" / "LICENSE.txt").read_text(encoding="utf-8")


def test_dlpack_component_matches_the_vendored_header() -> None:
    component = sbom.dlpack_component()
    header = (ROOT / "third_party" / "dlpack" / "include" / "dlpack" / "dlpack.h").read_text(encoding="utf-8")
    assert f"#define DLPACK_MAJOR_VERSION {component['version'].split('.')[0]}" in header
    assert f"#define DLPACK_MINOR_VERSION {component['version'].split('.')[1]}" in header
    assert "Apache License" in (ROOT / "third_party" / "dlpack" / "LICENSE").read_text(encoding="utf-8")


def _requirements_sbom(path: Path, extra: list[dict] | None = None, spec_version: str = "1.7") -> None:
    # The shape cyclonedx-py requirements writes for an unpinned requirement.
    components = [{
        "type": "library",
        "bom-ref": "requirements-L1",
        "name": "numpy",
        "purl": "pkg:pypi/numpy",
        "description": "requirements line 1: numpy>=1.23",
    }] + (extra or [])
    path.write_text(
        json.dumps({
            "bomFormat": "CycloneDX",
            "specVersion": spec_version,
            "serialNumber": "urn:uuid:00000000-0000-4000-8000-000000000000",
            "version": 1,
            "metadata": {"timestamp": "2026-11-03T00:00:00+00:00"},
            "components": components,
            "dependencies": [{"ref": c["bom-ref"]} for c in components],
        }),
        encoding="utf-8",
    )


def _write(tmp_path: Path, extra: list[dict] | None = None) -> dict[str, dict]:
    requirements = tmp_path / "requirements.cdx.json"
    _requirements_sbom(requirements, extra)
    out = tmp_path / "sbom"
    assert sbom.main(["--requirements", str(requirements), "--out-dir", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == sorted(f"wolfgang-quantum-{kind}.cdx.json" for kind in KINDS)
    return {kind: json.loads((out / f"wolfgang-quantum-{kind}.cdx.json").read_text(encoding="utf-8")) for kind in KINDS}


def _by_name(document: dict) -> dict[str, dict]:
    return {c["name"]: c for c in document["components"]}


def test_artifact_kinds_match_the_wheel_build_matrix() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release-wheelhouse.yml").read_text(encoding="utf-8")
    platforms = re.findall(r"^\s+- platform: (\S+)$", workflow, flags=re.MULTILINE)
    assert [kind.name for kind in sbom.ARTIFACT_KINDS] == ["sdist", *platforms]


def test_each_sbom_describes_wolfgang_quantum_as_its_subject(tmp_path: Path) -> None:
    documents = _write(tmp_path)
    serials = {document["serialNumber"] for document in documents.values()}
    assert len(serials) == len(KINDS)  # distinct documents get distinct serial numbers
    for kind, document in documents.items():
        root = document["metadata"]["component"]
        purl = f"pkg:pypi/wolfgang-quantum@{PROJECT['version']}"
        assert (root["name"], root["version"], root["purl"], root["bom-ref"]) == (
            "wolfgang-quantum", PROJECT["version"], purl, purl,
        )
        assert root["licenses"] == [{"expression": PROJECT["license"]}]
        assert {"name": "wolfgang:release-artifact", "value": kind} in root["properties"]
        assert document["metadata"]["timestamp"] == "2026-11-03T00:00:00+00:00"
        # The root depends on every listed component, and each has its own node.
        graph = {node["ref"]: node.get("dependsOn", []) for node in document["dependencies"]}
        refs = [c["bom-ref"] for c in document["components"]]
        assert sorted(graph[purl]) == sorted(refs)
        assert set(graph) == {purl, *refs}


def test_python_dependencies_are_external_with_their_accepted_range(tmp_path: Path) -> None:
    (numpy_requirement,) = [d for d in PROJECT["dependencies"] if d.startswith("numpy")]
    for document in _write(tmp_path).values():
        numpy = _by_name(document)["numpy"]
        assert numpy["bom-ref"] == "pkg:pypi/numpy"
        assert numpy["isExternal"] is True
        assert numpy["versionRange"] == sbom.vers_range("pypi", numpy_requirement.removeprefix("numpy"))
        assert "version" not in numpy


@pytest.mark.parametrize(
    ("specifier", "expected"),
    [
        (">=1.23", "vers:pypi/>=1.23"),
        (">=1.23,<3", "vers:pypi/>=1.23|<3"),
        ("<3,>=1.23", "vers:pypi/>=1.23|<3"),  # vers lists constraints in version order
        ("!=1.25.0,>=1.23", "vers:pypi/>=1.23|!=1.25.0"),
        ("==2.0", "vers:pypi/2.0"),
    ],
)
def test_pep440_specifiers_become_vers_ranges(specifier: str, expected: str) -> None:
    assert sbom.vers_range("pypi", specifier) == expected


@pytest.mark.parametrize("specifier", ["~=1.4", "==1.*", "===1.0", ""])
def test_specifiers_vers_cannot_express_exactly_are_rejected(specifier: str) -> None:
    with pytest.raises(ValueError):
        sbom.vers_range("pypi", specifier)


def test_bundled_native_code_depends_on_the_artifact(tmp_path: Path) -> None:
    documents = _write(tmp_path)
    sdist = _by_name(documents["sdist"])
    assert "oneTBB" not in sdist  # source builds use a system oneTBB, if any
    assert sdist["dlpack"]["properties"][0]["value"].startswith("the source distribution")
    linux = _by_name(documents["manylinux-x86_64"])["oneTBB"]
    mac = _by_name(documents["macos-arm64"])["oneTBB"]
    assert linux["licenses"] == [{"expression": "Apache-2.0 AND BSD-3-Clause"}]  # ITT Notify on x86_64
    assert mac["licenses"] == [{"expression": "Apache-2.0"}]
    for kind in ("manylinux-x86_64", "macos-arm64"):
        names = [c["name"] for c in documents[kind]["components"]]
        assert names == ["numpy", "oneTBB", "dlpack"]
        dlpack = _by_name(documents[kind])["dlpack"]
        assert dlpack["hashes"][0]["content"] in (ROOT / "third_party" / "dlpack" / "README.md").read_text(encoding="utf-8")


def test_stale_bundled_components_in_the_input_are_replaced(tmp_path: Path) -> None:
    stale = {"type": "library", "bom-ref": "pkg:github/uxlfoundation/onetbb@v2021.1.0", "name": "oneTBB"}
    documents = _write(tmp_path, [stale])
    assert [c["name"] for c in documents["sdist"]["components"]] == ["numpy", "dlpack"]
    assert stale["bom-ref"] not in {node["ref"] for node in documents["sdist"]["dependencies"]}


def test_other_documents_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "not-sbom.json"
    path.write_text(json.dumps({"components": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="CycloneDX"):
        sbom.main(["--requirements", str(path), "--out-dir", str(tmp_path / "out")])


def test_requirements_sbom_must_be_cyclonedx_1_7(tmp_path: Path) -> None:
    # External components with version ranges are new in CycloneDX 1.7.
    path = tmp_path / "requirements.cdx.json"
    _requirements_sbom(path, spec_version="1.6")
    with pytest.raises(ValueError, match="1.7"):
        sbom.main(["--requirements", str(path), "--out-dir", str(tmp_path / "out")])


def test_unpinned_values_are_rejected(tmp_path: Path) -> None:
    # An override such as ONETBB_TAG="${ONETBB_TAG:-v2022.3.0}" would otherwise
    # be copied into the SBOM as the version.
    shutil.copytree(ROOT / "third_party", tmp_path / "third_party")
    (tmp_path / "scripts").mkdir()
    script = (ROOT / "scripts" / "build_onetbb.sh").read_text(encoding="utf-8")
    tag = re.search(r'^ONETBB_TAG="([^"]+)"$', script, flags=re.MULTILINE).group(1)
    (tmp_path / "scripts" / "build_onetbb.sh").write_text(
        script.replace(f'ONETBB_TAG="{tag}"', 'ONETBB_TAG="${ONETBB_TAG:-' + tag + '}"'), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="pinned"):
        sbom.onetbb_component(tmp_path, licenses="Apache-2.0")


def test_validation_accepts_every_sbom_and_rejects_a_bad_license(tmp_path: Path) -> None:
    pytest.importorskip("cyclonedx.validation.json")
    requirements = tmp_path / "requirements.cdx.json"
    _requirements_sbom(requirements)
    assert sbom.main(["--validate", "--requirements", str(requirements), "--out-dir", str(tmp_path / "ok")]) == 0
    bad = tmp_path / "bad.cdx.json"
    _requirements_sbom(bad)
    document = json.loads(bad.read_text(encoding="utf-8"))
    document["components"][0]["licenses"] = [{"license": {"id": "Not-A-License"}}]
    bad.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="validation"):
        sbom.main(["--validate", "--requirements", str(bad), "--out-dir", str(tmp_path / "bad")])


def test_validation_needs_a_spec_version() -> None:
    with pytest.raises(ValueError, match="specVersion"):
        sbom.validate("{}", "")


def test_release_workflow_writes_and_attests_one_sbom_per_artifact_kind() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release-wheelhouse.yml").read_text(encoding="utf-8")
    # CycloneDX 1.7 needs cyclonedx-python-lib 11, which pip-audit 2.9 excluded.
    assert '"cyclonedx-python-lib==11.' in workflow
    generate = workflow.index("cyclonedx-py requirements runtime-requirements.txt")
    assert "--spec-version 1.7" in workflow[generate:generate + 300]
    write = workflow.index("python scripts/sbom_bundled_components.py --validate")
    upload = workflow.index("name: software-bill-of-materials")
    assert generate < write < upload
    attest_job = workflow[workflow.index("  attest-release:"):workflow.index("  validate-publish-ref:")]
    assert "name: software-bill-of-materials" in attest_job  # downloaded before attesting
    for kind in sbom.ARTIFACT_KINDS:
        step = re.search(
            r"uses: actions/attest@[0-9a-f]{40} # v\S+\n\s+with:\n"
            rf"\s+subject-path: {re.escape(kind.subject_path)}\n"
            rf"\s+sbom-path: sbom/wolfgang-quantum-{re.escape(kind.name)}\.cdx\.json\n"
            r"\s+create-storage-record: false\n",
            attest_job,
        )
        assert step is not None, kind.name

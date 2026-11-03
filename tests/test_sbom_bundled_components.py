"""The release SBOM lists the native code Wolfgang ships.

`cyclonedx-py` covers Python dependencies only; `scripts/sbom_bundled_components.py`
adds the bundled oneTBB runtime and the vendored DLPack header, reading their
versions from the files that pin them.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sbom_bundled_components", ROOT / "scripts" / "sbom_bundled_components.py")
sbom = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sbom)


def test_onetbb_component_matches_the_build_pins_and_notices() -> None:
    script = (ROOT / "scripts" / "build_onetbb.sh").read_text(encoding="utf-8")
    tag = re.search(r'^ONETBB_TAG="([^"]+)"$', script, flags=re.MULTILINE).group(1)
    commit = re.search(r'^ONETBB_COMMIT="([^"]+)"$', script, flags=re.MULTILINE).group(1)
    component = sbom.onetbb_component()
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


def _fixture(path: Path, extra: list[dict] | None = None) -> None:
    components = [{"type": "library", "bom-ref": "numpy", "name": "numpy"}] + (extra or [])
    path.write_text(
        json.dumps({
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "version": 1,
            "components": components,
            "dependencies": [{"ref": c["bom-ref"]} for c in components],
        }),
        encoding="utf-8",
    )


def test_merge_adds_components_and_graph_nodes_once(tmp_path: Path) -> None:
    path = tmp_path / "sbom.cdx.json"
    # An older pinned oneTBB from a previous run is replaced, not kept beside the new one.
    _fixture(path, [{"type": "library", "bom-ref": "pkg:github/uxlfoundation/onetbb@v2021.1.0", "name": "oneTBB"}])
    assert sbom.main([str(path)]) == 0
    assert sbom.main([str(path)]) == 0
    document = json.loads(path.read_text(encoding="utf-8"))
    assert [c["name"] for c in document["components"]] == ["numpy", "oneTBB", "dlpack"]
    refs = [d["ref"] for d in document["dependencies"]]
    assert refs == ["numpy"] + [c["bom-ref"] for c in document["components"][1:]]
    dlpack = document["components"][2]
    assert dlpack["hashes"][0]["alg"] == "SHA-256"
    assert dlpack["hashes"][0]["content"] in (ROOT / "third_party" / "dlpack" / "README.md").read_text(encoding="utf-8")


def test_merge_rejects_other_documents(tmp_path: Path) -> None:
    path = tmp_path / "not-sbom.json"
    path.write_text(json.dumps({"components": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="CycloneDX"):
        sbom.main([str(path)])


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
        sbom.onetbb_component(tmp_path)


def test_validation_accepts_the_merged_sbom_and_rejects_a_bad_license(tmp_path: Path) -> None:
    pytest.importorskip("cyclonedx.validation.json")
    path = tmp_path / "sbom.cdx.json"
    _fixture(path)
    assert sbom.main(["--validate", str(path)]) == 0
    bad = tmp_path / "bad.cdx.json"
    _fixture(bad)
    document = json.loads(bad.read_text(encoding="utf-8"))
    document["components"][0]["licenses"] = [{"license": {"id": "Not-A-License"}}]
    bad.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="validation"):
        sbom.main(["--validate", str(bad)])


def test_validation_needs_a_spec_version() -> None:
    with pytest.raises(ValueError, match="specVersion"):
        sbom.validate("{}", "")


def test_release_workflow_adds_bundled_components_to_the_sbom() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release-wheelhouse.yml").read_text(encoding="utf-8")
    generate = workflow.index("--output-file sbom/wolfgang-runtime.cdx.json")
    add = workflow.index("python scripts/sbom_bundled_components.py --validate sbom/wolfgang-runtime.cdx.json")
    upload = workflow.index("name: software-bill-of-materials")
    assert generate < add < upload

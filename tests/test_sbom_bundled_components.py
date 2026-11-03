"""The release SBOM lists the native code Wolfgang ships.

`cyclonedx-py` covers Python dependencies only; `scripts/sbom_bundled_components.py`
adds the bundled oneTBB runtime and the vendored DLPack header, reading their
versions from the files that pin them.
"""

from __future__ import annotations

import importlib.util
import json
import re
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
    assert component["purl"] == f"pkg:github/uxlfoundation/oneTBB@{tag}"
    assert {"name": "wolfgang:source-commit", "value": commit} in component["properties"]
    assert component["licenses"] == [{"license": {"id": "Apache-2.0"}}]
    notices = (ROOT / "third_party" / "onetbb" / "README.md").read_text(encoding="utf-8")
    assert tag in notices and commit in notices
    assert "Apache License" in (ROOT / "third_party" / "onetbb" / "LICENSE.txt").read_text(encoding="utf-8")


def test_dlpack_component_matches_the_vendored_header() -> None:
    component = sbom.dlpack_component()
    header = (ROOT / "third_party" / "dlpack" / "include" / "dlpack" / "dlpack.h").read_text(encoding="utf-8")
    assert f"#define DLPACK_MAJOR_VERSION {component['version'].split('.')[0]}" in header
    assert f"#define DLPACK_MINOR_VERSION {component['version'].split('.')[1]}" in header
    assert "Apache License" in (ROOT / "third_party" / "dlpack" / "LICENSE").read_text(encoding="utf-8")


def test_merge_adds_components_once(tmp_path: Path) -> None:
    path = tmp_path / "sbom.cdx.json"
    path.write_text(
        json.dumps({"bomFormat": "CycloneDX", "specVersion": "1.6", "components": [{"bom-ref": "numpy", "name": "numpy"}]}),
        encoding="utf-8",
    )
    assert sbom.main([str(path)]) == 0
    assert sbom.main([str(path)]) == 0
    names = [component["name"] for component in json.loads(path.read_text(encoding="utf-8"))["components"]]
    assert names == ["numpy", "oneTBB", "dlpack"]


def test_merge_rejects_other_documents(tmp_path: Path) -> None:
    path = tmp_path / "not-sbom.json"
    path.write_text(json.dumps({"components": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="CycloneDX"):
        sbom.main([str(path)])


def test_release_workflow_adds_bundled_components_to_the_sbom() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release-wheelhouse.yml").read_text(encoding="utf-8")
    generate = workflow.index("--output-file sbom/wolfgang-runtime.cdx.json")
    add = workflow.index("python scripts/sbom_bundled_components.py sbom/wolfgang-runtime.cdx.json")
    upload = workflow.index("name: software-bill-of-materials")
    assert generate < add < upload

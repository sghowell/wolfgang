"""Execute onboarding and check that source-map validation reads the entrypoint."""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_installation_verification_example_executes() -> None:
    source = (ROOT / "docs/getting-started/installation.md").read_text()
    examples = re.findall(r"python - <<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    assert examples, "installation verification example must exist"
    for example in examples:
        result = subprocess.run([sys.executable, "-c", example], capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert "Wolfgang" in result.stdout


def test_agent_source_map_cannot_reference_a_missing_document(tmp_path, monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location("validate_entrypoint", ROOT / "scripts/validate.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "README.md").write_text("# Example\n")
    (tmp_path / "AGENTS.md").write_text("# Sources\n```text\ndocs/missing.md\n```\n")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "SOURCE_OF_TRUTH_PATHS", ())
    with pytest.raises(SystemExit):
        module.check_markdown_links()

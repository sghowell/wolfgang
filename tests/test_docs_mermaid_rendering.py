from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MERMAID_RUNTIME = "javascripts/vendor/mermaid-11.4.1.min.js"
MERMAID_INIT = "javascripts/mermaid-init.js"
PAGES_WITH_DIAGRAMS = (
    "guide/architecture/index.html",
    "guide/agent-driven-engineering/index.html",
)


def build_docs(site_dir: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir", str(site_dir)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_mermaid_fences_build_as_runtime_backed_diagrams(tmp_path: Path) -> None:
    site_dir = tmp_path / "site"
    build_docs(site_dir)

    for relative_path in PAGES_WITH_DIAGRAMS:
        html = (site_dir / relative_path).read_text(encoding="utf-8")
        assert '<div class="mermaid">' in html, f"missing Mermaid container in {relative_path}"
        assert "flowchart LR" in html, f"missing diagram source in {relative_path}"
        assert '<div class="highlight"><pre><span></span><code>flowchart LR' not in html
        assert MERMAID_RUNTIME in html, f"missing Mermaid runtime on {relative_path}"
        assert MERMAID_INIT in html, f"missing Mermaid initializer on {relative_path}"

    initializer = (site_dir / MERMAID_INIT).read_text(encoding="utf-8")
    assert "document$.subscribe" in initializer
    assert "mermaid.run" in initializer
    assert "data-md-color-scheme" in initializer

    runtime = site_dir / MERMAID_RUNTIME
    assert runtime.is_file(), f"missing vendored Mermaid runtime: {MERMAID_RUNTIME}"
    assert runtime.stat().st_size > 1_000_000

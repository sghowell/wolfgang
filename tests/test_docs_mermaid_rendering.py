from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MERMAID_RUNTIME = ROOT / "docs/javascripts/vendor/mermaid-11.4.1.min.js"
MERMAID_RUNTIME_SITE_PATH = "javascripts/vendor/mermaid-11.4.1.min.js"
MERMAID_INIT = ROOT / "docs/javascripts/mermaid-init.js"
MERMAID_INIT_SITE_PATH = "javascripts/mermaid-init.js"
MERMAID_SOURCE_PAGES = (
    ROOT / "docs/guide/architecture.md",
    ROOT / "docs/guide/agent-driven-engineering.md",
)


def test_mermaid_docs_contract_is_pinned_and_source_backed() -> None:
    mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")

    assert "name: mermaid" in mkdocs
    assert "class: mermaid" in mkdocs
    assert MERMAID_RUNTIME_SITE_PATH in mkdocs
    assert MERMAID_INIT_SITE_PATH in mkdocs
    assert "stylesheets/mermaid.css" in mkdocs

    for path in MERMAID_SOURCE_PAGES:
        source = path.read_text(encoding="utf-8")
        assert "```mermaid" in source, f"missing Mermaid fence in {path.relative_to(ROOT)}"
        assert "flowchart LR" in source, f"missing Mermaid flowchart in {path.relative_to(ROOT)}"

    assert MERMAID_RUNTIME.is_file(), (
        f"missing vendored Mermaid runtime: {MERMAID_RUNTIME.relative_to(ROOT)}"
    )
    assert MERMAID_RUNTIME.stat().st_size > 1_000_000

    initializer = MERMAID_INIT.read_text(encoding="utf-8")
    assert "document$.subscribe" in initializer
    assert "mermaid.run" in initializer
    assert "data-md-color-scheme" in initializer

"""The README figures are generated from committed benchmark data and stay current."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_figures_match_their_data() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "render_readme_figures.py"), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout

"""The README quickstart runs as written, so the documentation cannot drift from the code."""

from __future__ import annotations

import re
import subprocess  # nosec
import sys
from pathlib import Path

PACKAGE = Path(__file__).parent.parent
DOSSIER_A = Path(__file__).parent / "fixtures" / "dossier_a"


def test_the_quickstart_runs_and_prints_a_cited_total() -> None:
    readme = (PACKAGE / "README.md").read_text()
    code = re.search(r"## Quickstart\n\n```python\n(.*?)```", readme, re.S)
    assert code, "README has no python quickstart block"
    snippet = code.group(1).replace('"path/to/dossier"', repr(str(DOSSIER_A)))
    # A subprocess: the snippet defines a live policy, and one per process is the rule. It runs
    # our own README, not external input.
    run = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True)  # nosec B603
    assert run.returncode == 0, run.stderr
    assert "45385.06" in run.stdout and "GL.txt@" in run.stdout, run.stdout

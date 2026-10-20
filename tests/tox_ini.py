"""tox.ini section lookup shared by the tests."""

from __future__ import annotations

import re
from pathlib import Path

_TOX_INI = Path(__file__).resolve().parents[1] / "tox.ini"


def tox_block(env_name: str) -> str:
    """Body of a tox.ini ``[testenv:<name>]`` section."""
    match = re.search(
        rf"^\[testenv:{re.escape(env_name)}\]\n(.*?)(?=\n\[|\Z)",
        _TOX_INI.read_text(),
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"tox.ini has no [testenv:{env_name}] section"
    return match.group(1)

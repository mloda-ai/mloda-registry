"""tox.ini section lookup shared by the tests."""

from __future__ import annotations

import re
from pathlib import Path

_TOX_INI = Path(__file__).resolve().parents[1] / "tox.ini"

# A leading tox 4 factor condition on a commands line: "lint: cmd", "!tests: cmd", "py311,py312-lint: cmd".
_FACTOR_CONDITION = re.compile(r"^(?P<condition>!?[\w.]+(?:[-,]!?[\w.]+)*):\s+(?P<command>.*)$")


def split_factor_condition(line: str) -> tuple[str | None, str]:
    """A tox commands line split into its leading factor condition (None when unconditional) and the command."""
    stripped = line.strip()
    match = _FACTOR_CONDITION.match(stripped)
    if match is None:
        return None, stripped
    return match.group("condition"), match.group("command")


def tox_block(env_name: str) -> str:
    """Body of a tox.ini ``[testenv:<name>]`` section."""
    match = re.search(
        rf"^\[testenv:{re.escape(env_name)}\]\n(.*?)(?=\n\[|\Z)",
        _TOX_INI.read_text(),
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"tox.ini has no [testenv:{env_name}] section"
    return match.group(1)


def tox_deps(env_name: str) -> list[str]:
    """Stripped dep lines of a tox.ini ``[testenv:<name>]`` multi-line ``deps =`` block."""
    match = re.search(r"^deps\s*=\s*\n((?:[ \t]+\S.*\n?)+)", tox_block(env_name), re.MULTILINE)
    assert match is not None, f"[testenv:{env_name}] has no multi-line deps block"
    return [line.strip() for line in match.group(1).splitlines()]

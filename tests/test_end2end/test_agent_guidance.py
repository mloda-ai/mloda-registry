"""Repository-level guarantees for agent guidance files."""

import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_CLAUDE_GUIDANCE = _REPO_ROOT / "CLAUDE.md"
_AGENT_GUIDANCE = _REPO_ROOT / "AGENTS.md"


def test_agent_guidance_files_are_byte_identical() -> None:
    """Tool-specific entry points must expose exactly the same guidance."""
    assert _CLAUDE_GUIDANCE.read_bytes() == _AGENT_GUIDANCE.read_bytes()


def test_supply_chain_guidance_documents_package_exemptions() -> None:
    """The cooldown exemptions must stay documented alongside the cooldown itself.

    Only CLAUDE.md is read; the byte-identical check above covers AGENTS.md.
    The assertions deliberately pin the setting name and nothing else, so the
    surrounding prose stays free to be reworded.
    """
    bullets = [
        line
        for line in _CLAUDE_GUIDANCE.read_text(encoding="utf-8").splitlines()
        if line.startswith("- **Supply chain**:")
    ]

    assert len(bullets) == 1, f"expected exactly one supply chain bullet, found {len(bullets)}"
    assert "exclude-newer-package" in bullets[0]


def test_type_hints_guidance_matches_ruff_select() -> None:
    """Every UP rule named in the type hints guidance must be selected in pyproject.toml."""
    bullets = [
        line
        for line in _CLAUDE_GUIDANCE.read_text(encoding="utf-8").splitlines()
        if line.startswith("- **Type hints**:")
    ]

    assert len(bullets) == 1, f"expected exactly one type hints bullet, found {len(bullets)}"
    codes = re.findall(r"UP\d+", bullets[0])
    assert codes, "type hints bullet names no UP rule"

    selected = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["tool"]["ruff"]["lint"]["select"]
    for code in codes:
        assert code in selected, f"{code} is documented but not in [tool.ruff.lint] select"

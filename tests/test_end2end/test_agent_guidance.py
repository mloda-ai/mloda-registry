"""Repository-level guarantees for agent guidance files."""

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
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

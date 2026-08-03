"""Repository-level guarantees for agent guidance files."""

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLAUDE_GUIDANCE = _REPO_ROOT / "CLAUDE.md"
_AGENT_GUIDANCE = _REPO_ROOT / "AGENTS.md"


def test_agent_guidance_files_are_byte_identical() -> None:
    """Tool-specific entry points must expose exactly the same guidance."""
    assert _CLAUDE_GUIDANCE.read_bytes() == _AGENT_GUIDANCE.read_bytes()


def test_supply_chain_guidance_documents_package_exemptions() -> None:
    """The cooldown exemptions and their rationale must remain discoverable."""
    supply_chain_bullet = next(
        line
        for line in _CLAUDE_GUIDANCE.read_text(encoding="utf-8").splitlines()
        if line.startswith("- **Supply chain**:")
    ).lower()

    assert "exclude-newer-package" in supply_chain_bullet
    assert "mloda" in supply_chain_bullet
    assert "uv" in supply_chain_bullet
    assert "7-day" in supply_chain_bullet or "7 days" in supply_chain_bullet
    assert "first-party" in supply_chain_bullet
    assert "resolver" in supply_chain_bullet

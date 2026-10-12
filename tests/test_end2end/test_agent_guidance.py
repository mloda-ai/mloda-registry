"""Repository-level guarantees for agent guidance files."""

import re
from pathlib import Path

from tests.toml_loader import loads_toml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_CLAUDE_GUIDANCE = _REPO_ROOT / "CLAUDE.md"
_AGENT_GUIDANCE = _REPO_ROOT / "AGENTS.md"
_PACKAGING_DOC = _REPO_ROOT / "docs" / "packaging.md"
_RELEASE_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "release.yaml"
_UV_PIN = r"(?<![\w-])uv==(\d+\.\d+\.\d+)"


def _single_bullet(prefix: str) -> str:
    """Return the one CLAUDE.md line starting with prefix, asserting there is exactly one."""
    bullets = [line for line in _CLAUDE_GUIDANCE.read_text(encoding="utf-8").splitlines() if line.startswith(prefix)]
    assert len(bullets) == 1, f"expected exactly one {prefix!r} bullet, found {len(bullets)}"
    return bullets[0]


def test_agent_guidance_files_are_byte_identical() -> None:
    """Tool-specific entry points must expose exactly the same guidance."""
    assert _CLAUDE_GUIDANCE.read_bytes() == _AGENT_GUIDANCE.read_bytes()


def test_supply_chain_guidance_documents_package_exemptions() -> None:
    """The cooldown exemptions must stay documented alongside the cooldown itself.

    Only CLAUDE.md is read; the byte-identical check above covers AGENTS.md.
    The assertions deliberately pin the setting name and nothing else, so the
    surrounding prose stays free to be reworded.
    """
    assert "exclude-newer-package" in _single_bullet("- **Supply chain**:")


def test_type_hints_guidance_matches_ruff_select() -> None:
    """The UP rules documented in the type hints guidance must match the ones selected in pyproject.toml."""
    codes = set(re.findall(r"UP\d+", _single_bullet("- **Type hints**:")))
    assert codes, "type hints bullet names no UP rule"

    select = loads_toml(_PYPROJECT.read_text(encoding="utf-8"))["tool"]["ruff"]["lint"]["select"]
    assert codes == {rule for rule in select if rule.startswith("UP")}


def test_relock_guidance_names_the_uv_version_pinned_by_the_release_job() -> None:
    """The relock guidance must name exactly the uv version the release job installs.

    The pin is read from the ``pip install uv==...`` line of the release workflow,
    not from the comment block above it. CLAUDE.md and docs/packaging.md must
    each name only that version.
    """
    pins = set(re.findall(r"pip install uv==(\d+\.\d+\.\d+)", _RELEASE_WORKFLOW.read_text(encoding="utf-8")))
    assert len(pins) == 1, f"expected one 'pip install uv==X.Y.Z' pin in {_RELEASE_WORKFLOW.name}, found {pins}"

    update_hint = (
        f"the release job pins uv=={next(iter(pins))}; if you bumped it in {_RELEASE_WORKFLOW.name}, "
        "update the uv==X.Y.Z version in the CLAUDE.md '- **Relocking**:' bullet (and AGENTS.md) and in docs/packaging.md"
    )
    claude_versions = set(re.findall(_UV_PIN, _single_bullet("- **Relocking**:")))
    assert claude_versions == pins, f"CLAUDE.md Relocking bullet names {claude_versions}: {update_hint}"

    doc_versions = set(re.findall(_UV_PIN, _PACKAGING_DOC.read_text(encoding="utf-8")))
    assert doc_versions == pins, f"docs/packaging.md names {doc_versions}: {update_hint}"

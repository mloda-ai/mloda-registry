"""Drift check: skill files must stay in sync with the guides/repos they reference."""

import importlib.util
import os
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GUIDES_ROOT = _REPO_ROOT / "docs" / "guides"
_PLUGINS_SKILL = _REPO_ROOT / ".claude" / "skills" / "mloda-plugins" / "SKILL.md"
_CORE_SKILL = _REPO_ROOT / ".claude" / "skills" / "mloda-core" / "SKILL.md"

_MLODA_PLUGINS_PATH_PATTERN = re.compile(r"`(mloda_plugins/[^`\n]*)`")

# The only basename that collides under docs/guides/: four "index.md" files (top-level
# and one per pattern subdirectory) all share this filename. They are MkDocs nav-mirror
# pages that just re-list the numbered guides, each of which already has a globally
# unique basename and is checked individually below, so bare "index.md" is excluded here
# rather than pretending path-level (as opposed to filename-level) coverage this check
# doesn't actually provide.
_EXCLUDED_GUIDE_FILENAMES = {"index.md"}


def _find_mloda_core_repo() -> Path | None:
    """Locate mloda_plugins via MLODA_PATH, a sibling mloda repo, or (CI fallback) the installed package."""
    candidate = os.environ.get("MLODA_PATH")
    if candidate:
        path = Path(candidate)
        if (path / "mloda_plugins").is_dir():
            return path

    sibling = _REPO_ROOT.parent / "mloda"
    if (sibling / "mloda_plugins").is_dir():
        return sibling

    spec = importlib.util.find_spec("mloda_plugins")
    if spec is not None and spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).parent

    return None


def test_every_guide_file_is_referenced_by_filename_in_plugins_skill() -> None:
    """Every guide under docs/guides/ must be mentioned (by filename) in the mloda-plugins skill."""
    skill_text = _PLUGINS_SKILL.read_text(encoding="utf-8")

    guide_files = sorted(_GUIDES_ROOT.rglob("*.md"))
    missing = sorted(
        {f.name for f in guide_files if f.name not in _EXCLUDED_GUIDE_FILENAMES and f.name not in skill_text}
    )

    assert missing == [], (
        f"These guide filenames under {_GUIDES_ROOT.relative_to(_REPO_ROOT)} are not referenced anywhere in "
        f"{_PLUGINS_SKILL.relative_to(_REPO_ROOT)}:\n  " + "\n  ".join(missing)
    )


def test_core_skill_key_locations_mloda_plugins_dirs_exist() -> None:
    """Every mloda_plugins/... path in the mloda-core skill's key-locations table must exist on disk."""
    core_repo = _find_mloda_core_repo()
    if core_repo is None:
        pytest.skip("mloda core repo not found via MLODA_PATH, sibling dir, or installed package")

    skill_text = _CORE_SKILL.read_text(encoding="utf-8")
    paths = sorted(set(_MLODA_PLUGINS_PATH_PATTERN.findall(skill_text)))

    assert paths, "no mloda_plugins/... paths found in the key-locations table"

    missing = sorted(p for p in paths if not (core_repo / p).exists())

    assert missing == [], (
        f"These mloda_plugins/... paths from {_CORE_SKILL.relative_to(_REPO_ROOT)} do not exist under "
        f"{core_repo}:\n  " + "\n  ".join(missing)
    )

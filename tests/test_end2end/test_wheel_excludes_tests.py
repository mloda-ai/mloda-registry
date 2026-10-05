"""Published wheels must not ship in-tree test packages.

``[tool.setuptools.packages.find]`` includes ``mloda*`` with namespace
discovery. Without an exclude list, every ``tests`` directory under ``mloda/``
becomes a packaged module. See issue #594.
"""

from __future__ import annotations

from pathlib import Path

from tests.toml_loader import loads_toml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_PACKAGE_ROOT = _REPO_ROOT / "mloda"

# Patterns required by issue #594. ``*.tests`` matches the tests package itself;
# ``*.tests.*`` matches modules and subpackages under it.
_REQUIRED_EXCLUDES = {"*.tests", "*.tests.*"}


def _setuptools_config() -> dict[object, object]:
    data = loads_toml(_PYPROJECT.read_text(encoding="utf-8"))
    return data["tool"]["setuptools"]  # type: ignore[no-any-return]


def test_include_package_data_is_disabled() -> None:
    """Stale egg-info SOURCES.txt re-adds excluded tests when package data is on."""
    setuptools = _setuptools_config()
    assert setuptools.get("include-package-data") is False


def test_packages_find_excludes_tests_patterns() -> None:
    setuptools = _setuptools_config()
    find = setuptools["packages"]["find"]  # type: ignore[index]
    excludes = set(find.get("exclude") or [])
    missing = _REQUIRED_EXCLUDES - excludes
    assert not missing, f"pyproject.toml is missing exclude patterns: {sorted(missing)}"


def test_exclude_patterns_cover_every_tests_directory() -> None:
    """Every tests/ directory under mloda/ must match *.tests or *.tests.*"""
    tests_dirs = sorted(path for path in _PACKAGE_ROOT.rglob("tests") if path.is_dir())
    assert tests_dirs, "expected in-tree tests directories under mloda/"

    uncovered: list[str] = []
    for tests_dir in tests_dirs:
        relative = tests_dir.relative_to(_REPO_ROOT)
        dotted = ".".join(relative.parts)
        # ``foo.bar.tests`` matches ``*.tests``; a nested ``foo.bar.tests.unit``
        # would match ``*.tests.*``.
        covered = dotted.endswith(".tests") or ".tests." in dotted
        if not covered:
            uncovered.append(str(relative))

    assert not uncovered, f"tests directories not covered by *.tests / *.tests.*: {uncovered}"

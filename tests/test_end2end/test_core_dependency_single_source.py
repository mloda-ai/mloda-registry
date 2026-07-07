"""Tests that the mloda-core version pin is declared in a single source of truth.

The mloda-core dependency specifier (e.g. ``mloda>=0.9.0``) must live in exactly
ONE place: ``config/shared.toml`` under ``[defaults].core_dependency``. Everywhere
else it must be referenced, never re-typed:

* ``config/packages.toml`` uses the placeholder token ``"{core_dependency}"`` in
  each package's ``dependencies`` array instead of a literal ``mloda>=...`` string.
* ``scripts/generate_pyproject.py`` substitutes that placeholder (and writes the
  root ``pyproject.toml`` mloda entry) from the shared value.

These tests encode that contract. They must fail while the pin is still duplicated
across ``config/packages.toml`` and the root ``pyproject.toml``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHARED_CONFIG = _REPO_ROOT / "config" / "shared.toml"
_PACKAGES_CONFIG = _REPO_ROOT / "config" / "packages.toml"
_ROOT_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# A pinned mloda-core specifier: the package name ``mloda`` (not ``mloda-...``)
# followed by a floor version constraint.
_CORE_PIN_RE = re.compile(r"^mloda\b.*>=")

# Matches a literal core pin such as ``mloda>=`` or the spaced form
# ``mloda >= 0.9.0`` (any comparison operator), where ``mloda`` is NOT part of a
# longer hyphenated name like ``mloda-community-...``. This must NOT match
# ``mloda-community-data-operations>=0.2.12`` or ``mloda-testing``.
_CORE_LITERAL_RE = re.compile(r"(?<![\w-])mloda\s*(>=|==|~=|!=|<=|<|>)")


def _load_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _dep_name(spec: str) -> str:
    """Extract the bare package name from a PEP 508 requirement string."""
    return re.split(r"[<>=!~\s\[]", spec, maxsplit=1)[0].strip()


def _core_dependency() -> str:
    """Return the single-source core specifier, failing clearly if absent."""
    shared = _load_toml(_SHARED_CONFIG)
    defaults = shared.get("defaults", {})
    assert "core_dependency" in defaults, (
        "config/shared.toml [defaults] must define 'core_dependency' as the single "
        "source of truth for the mloda-core version pin."
    )
    value = defaults["core_dependency"]
    assert isinstance(value, str), "core_dependency must be a string"
    return value


def test_shared_defines_core_dependency() -> None:
    """[defaults].core_dependency exists and is a pinned mloda specifier."""
    value = _core_dependency()
    assert _CORE_PIN_RE.match(value), (
        f"core_dependency must be a pinned mloda-core specifier (e.g. 'mloda>=0.9.0'), got {value!r}."
    )


def test_packages_config_has_no_literal_core_pin() -> None:
    """config/packages.toml must reference the core pin only via the placeholder."""
    raw = _PACKAGES_CONFIG.read_text()
    matches = _CORE_LITERAL_RE.findall(raw)
    assert matches == [], (
        f"config/packages.toml still contains {len(matches)} literal mloda-core pin(s); "
        "each package must use the '{core_dependency}' placeholder instead."
    )


def test_root_pyproject_uses_core_dependency() -> None:
    """Root pyproject.toml [project].dependencies contains exactly core_dependency."""
    core = _core_dependency()
    root = _load_toml(_ROOT_PYPROJECT)
    deps = root.get("project", {}).get("dependencies", [])
    assert core in deps, (
        f"Root pyproject.toml [project].dependencies must contain the shared value {core!r}; got {deps!r}."
    )
    core_entries = [d for d in deps if _dep_name(d) == "mloda"]
    assert core_entries == [core], (
        f"Root pyproject.toml must declare the mloda-core dependency exactly as {core!r}; got {core_entries!r}."
    )


def test_generated_pyprojects_use_core_dependency() -> None:
    """Every generated per-package pyproject.toml uses exactly core_dependency for mloda-core."""
    core = _core_dependency()
    packages = _load_toml(_PACKAGES_CONFIG).get("packages", {})

    checked_core = 0
    for pkg_name, pkg_config in packages.items():
        pyproject_path = _REPO_ROOT / pkg_config["path"] / "pyproject.toml"
        assert pyproject_path.exists(), f"{pyproject_path} is missing (run scripts/generate_pyproject.py)."
        parsed = _load_toml(pyproject_path)
        deps = parsed.get("project", {}).get("dependencies", [])
        core_entries = [d for d in deps if _dep_name(d) == "mloda"]
        for entry in core_entries:
            checked_core += 1
            assert entry == core, (
                f"{pkg_name}: generated {pyproject_path} declares mloda-core as {entry!r}, "
                f"but the single source of truth is {core!r}."
            )

    assert checked_core > 0, "Expected at least one generated pyproject.toml to depend on mloda-core."

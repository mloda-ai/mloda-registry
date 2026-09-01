"""``[tool.uv.sources]`` generation for top-level packages.

``scripts/generate_pyproject.py`` only ever emitted ``mloda-testing = { workspace = true }``
for top-level packages (path depth <= 2) that receive the default dev extras. ``mloda-enterprise``
is a top-level bundle that does not receive default dev deps, and its ``mloda-community`` dependency
is a workspace member, so ``uv lock`` fails with "mloda-community is included as a workspace member,
but is missing an entry in tool.uv.sources".

Required behaviour: for a package whose path depth is <= 2 and which has no ``workspace_deps``, the
generator emits one ``[tool.uv.sources]`` table with one ``{ workspace = true }`` line per configured
package the package depends on (by PEP 503 normalized name, extras/markers/version specifiers
stripped), plus ``mloda-testing`` when the package receives the default dev deps, sorted by name and
deduplicated. External names and the ``{core_dependency}`` placeholder never get an entry. Nested
packages (path depth > 2) never get a ``[tool.uv.sources]`` table (see docs/packaging.md, "UV
workspace sources"). Packages with ``workspace_deps`` keep their existing block unchanged.

The generator lives at ``scripts/generate_pyproject.py`` (a script, not an installed package), so it
is loaded here by file path through ``tests.script_loader``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

from tests.script_loader import load_script

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GEN_PATH = _REPO_ROOT / "scripts" / "generate_pyproject.py"

gen = load_script("generate_pyproject", _GEN_PATH)

_SANDBOX_NAME = "mloda-sandbox"
_MIXED_DEPENDENCIES = [
    "{core_dependency}",
    "Mloda_Community>=0.4.5",
    "opentelemetry-api>=1.30,<2",
    "mloda-registry[all]>=0.4.0 ; python_version >= '3.10'",
]


def _generate(pkg_name: str) -> str:
    """Load the real configs and generate the pyproject text for a single package."""
    shared, packages_config = gen.load_configs()
    packages: dict[str, dict[str, Any]] = packages_config["packages"]
    return str(gen.generate_pyproject(pkg_name, packages[pkg_name], shared, packages))


def _sources(content: str) -> dict[str, Any] | None:
    """Parse generated pyproject text and return its ``[tool.uv.sources]`` table, if any."""
    data: dict[str, Any] = tomllib.loads(content)
    tool = data.get("tool")
    if not isinstance(tool, dict):
        return None
    uv = tool.get("uv")
    if not isinstance(uv, dict):
        return None
    sources = uv.get("sources")
    if not isinstance(sources, dict):
        return None
    return sources


def _all_packages_with(name: str, cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Real configured packages plus one synthetic package registered under ``name``."""
    _shared, packages_config = gen.load_configs()
    packages: dict[str, dict[str, Any]] = dict(packages_config["packages"])
    packages[name] = cfg
    return packages


def _sandbox_content(dependencies: list[str]) -> str:
    """Generated pyproject text for a synthetic top-level sandbox package with no workspace_deps."""
    shared, _packages_config = gen.load_configs()
    cfg: dict[str, Any] = {
        "description": "sandbox",
        "path": "mloda/sandbox",
        "dependencies": dependencies,
    }
    all_packages = _all_packages_with(_SANDBOX_NAME, cfg)
    return str(gen.generate_pyproject(_SANDBOX_NAME, cfg, shared, all_packages))


def test_enterprise_bundle_gets_source_entry_for_its_community_dependency() -> None:
    """mloda-enterprise has no default dev deps but depends on the mloda-community workspace member."""
    content = _generate("mloda-enterprise")
    assert _sources(content) == {"mloda-community": {"workspace": True}}, content


def test_registry_keeps_its_existing_default_dev_deps_source() -> None:
    """mloda-registry has no configured-package dependency; it keeps the existing mloda-testing entry."""
    content = _generate("mloda-registry")
    assert _sources(content) == {"mloda-testing": {"workspace": True}}, content


def test_community_bundle_has_no_sources_table() -> None:
    """mloda-community has no configured-package dependency and no default dev deps."""
    content = _generate("mloda-community")
    assert _sources(content) is None, content


def test_nested_package_with_configured_dependency_has_no_sources_table() -> None:
    """A nested package (depth > 2) never gets a sources table, even with a configured-package dependency."""
    content = _generate("mloda-community-aggregation")
    assert _sources(content) is None, content


def test_synthetic_top_level_package_scans_dependencies_for_sources() -> None:
    """Normalization, extras and markers are handled; external names get no entry."""
    content = _sandbox_content(_MIXED_DEPENDENCIES)
    assert _sources(content) == {
        "mloda-community": {"workspace": True},
        "mloda-registry": {"workspace": True},
        "mloda-testing": {"workspace": True},
    }, content


def test_synthetic_top_level_package_with_only_core_dependency_gets_default_dev_source_only() -> None:
    """With no configured-package dependency, only the default mloda-testing dev entry is emitted."""
    content = _sandbox_content(["{core_dependency}"])
    assert _sources(content) == {"mloda-testing": {"workspace": True}}, content


def test_synthetic_mloda_testing_package_gets_no_self_entry() -> None:
    """A package literally named mloda-testing gets no default dev entry and no self-reference."""
    shared, _packages_config = gen.load_configs()
    cfg: dict[str, Any] = {
        "description": "sandbox mloda-testing",
        "path": "mloda/sandbox_testing",
        "dependencies": ["{core_dependency}", "mloda-registry"],
    }
    all_packages = _all_packages_with("mloda-testing", cfg)

    content = str(gen.generate_pyproject("mloda-testing", cfg, shared, all_packages))

    assert _sources(content) == {"mloda-registry": {"workspace": True}}, content


def test_sources_table_is_emitted_once_and_entries_are_sorted() -> None:
    """The table appears exactly once and its entry lines are in sorted-by-name order."""
    content = _sandbox_content(_MIXED_DEPENDENCIES)

    assert content.count("[tool.uv.sources]") == 1, content

    community_idx = content.index("mloda-community = { workspace = true }")
    registry_idx = content.index("mloda-registry = { workspace = true }")
    testing_idx = content.index("mloda-testing = { workspace = true }")
    assert community_idx < registry_idx < testing_idx, content

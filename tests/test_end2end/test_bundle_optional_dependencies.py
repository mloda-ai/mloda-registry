"""mloda-community must not hard-require openlineage-python; it ships as the ``openlineage`` extra,
pinned in exactly one place: packages.mloda-community-openlineage's own dependency.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

import pytest

from tests.script_loader import load_script

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGES_CONFIG = _REPO_ROOT / "config" / "packages.toml"
_COMMUNITY_PYPROJECT = _REPO_ROOT / "mloda" / "community" / "pyproject.toml"
_GEN_PATH = _REPO_ROOT / "scripts" / "generate_pyproject.py"

# Distribution name -> the import root that provides it. Extend this when a bundle covers a new
# nested leaf's dependency only through an extra.
_IMPORT_ROOT_OF_DISTRIBUTION = {"openlineage-python": "openlineage"}

# Package name -> attribute name(s) the leaf's own __init__ must not expose when the extra-only
# dependency above is absent. Extend this alongside _IMPORT_ROOT_OF_DISTRIBUTION.
_EXPOSED_EXTENDER_NAMES: dict[str, list[str]] = {"mloda-community-openlineage": ["OpenLineageExtender"]}

gen = load_script("generate_pyproject", _GEN_PATH)


def _load_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _dep_name(spec: str) -> str:
    """Extract the bare package name from a PEP 508 requirement string."""
    return re.split(r"[<>=!~;\s\[(@]", spec.strip(), maxsplit=1)[0]


def _packages() -> dict[str, dict[str, Any]]:
    packages: dict[str, dict[str, Any]] = _load_toml(_PACKAGES_CONFIG)["packages"]
    return packages


def _external_dependency_names(deps: list[str], packages: dict[str, dict[str, Any]]) -> set[str]:
    """Bare distribution names among ``deps``, dropping placeholders (e.g. ``{core_dependency}``) and
    names that resolve to a configured package."""
    names = set()
    for dep in deps:
        if dep.strip().startswith("{"):
            continue
        name = _dep_name(dep)
        if name in packages or name == "mloda":
            continue
        names.add(name)
    return names


def test_mloda_community_dependencies_do_not_pin_openlineage_python() -> None:
    community = _packages()["mloda-community"]
    hard_deps = community.get("dependencies", [])
    offending = [dep for dep in hard_deps if _dep_name(dep) == "openlineage-python"]
    assert not offending, (
        f"packages.mloda-community.dependencies must not pin openlineage-python directly, found {offending!r}; "
        "it belongs in optional_dependencies.openlineage instead"
    )


def test_mloda_community_declares_openlineage_extra_matching_pin_source() -> None:
    packages = _packages()
    community_extra = packages["mloda-community"].get("optional_dependencies", {}).get("openlineage")
    assert community_extra is not None, "packages.mloda-community.optional_dependencies.openlineage must be declared"

    matching = [dep for dep in community_extra if _dep_name(dep) == "openlineage-python"]
    assert len(matching) == 1, (
        f"packages.mloda-community.optional_dependencies.openlineage must contain exactly one "
        f"openlineage-python entry, got {community_extra!r}"
    )

    leaf_deps = packages["mloda-community-openlineage"]["dependencies"]
    leaf_matching = [dep for dep in leaf_deps if _dep_name(dep) == "openlineage-python"]
    assert len(leaf_matching) == 1, (
        f"packages.mloda-community-openlineage.dependencies must declare exactly one openlineage-python "
        f"entry, got {leaf_deps!r}"
    )
    assert matching[0] == leaf_matching[0], (
        "the openlineage-python pin must live in one place: packages.mloda-community.optional_dependencies."
        f"openlineage ({matching[0]!r}) must equal packages.mloda-community-openlineage.dependencies "
        f"({leaf_matching[0]!r})"
    )


def test_generated_pyproject_lists_openlineage_python_only_under_optional_extra() -> None:
    assert _COMMUNITY_PYPROJECT.is_file(), f"generated pyproject not found at {_COMMUNITY_PYPROJECT}"
    project = _load_toml(_COMMUNITY_PYPROJECT)["project"]

    hard_deps = project.get("dependencies", [])
    offending = [dep for dep in hard_deps if _dep_name(dep) == "openlineage-python"]
    assert not offending, (
        f"{_COMMUNITY_PYPROJECT}: [project].dependencies must not pin openlineage-python, found {offending!r}"
    )

    optional_deps: dict[str, list[str]] = project.get("optional-dependencies", {})
    groups_with_openlineage = {
        group: deps
        for group, deps in optional_deps.items()
        if any(_dep_name(dep) == "openlineage-python" for dep in deps)
    }
    assert groups_with_openlineage == {"openlineage": optional_deps.get("openlineage", [])}, (
        f"{_COMMUNITY_PYPROJECT}: openlineage-python must appear only under "
        f"[project.optional-dependencies].openlineage, found it under {sorted(groups_with_openlineage)}"
    )
    assert len(optional_deps.get("openlineage", [])) == 1


def test_extra_only_bundle_dependencies_have_import_safe_manifests(monkeypatch: pytest.MonkeyPatch) -> None:
    """A nested leaf's runtime dependency covered only through its bundle's own extra is optional at
    runtime: the leaf's manifest must import cleanly without it and degrade to empty entry-point lists.
    """
    packages = _packages()
    checked = 0

    for bundle_name, bundle_cfg in packages.items():
        if bundle_cfg.get("entry_point_bundle") is not True:
            continue
        prefix = bundle_cfg["path"] + "/"

        hard_names = _external_dependency_names(bundle_cfg.get("dependencies", []), packages)
        extra_names: set[str] = set()
        for extra_name, extra_deps in bundle_cfg.get("optional_dependencies", {}).items():
            if extra_name == "dev":
                continue
            extra_names |= _external_dependency_names(extra_deps, packages)
        extra_only = extra_names - hard_names

        for leaf_name, leaf_cfg in packages.items():
            if leaf_name == bundle_name or not leaf_cfg["path"].startswith(prefix):
                continue
            groups = leaf_cfg.get("entry_point_groups")
            if not groups:
                continue

            leaf_names = _external_dependency_names(leaf_cfg.get("dependencies", []), packages)
            covered_only_by_extra = sorted(leaf_names & extra_only)
            if not covered_only_by_extra:
                continue

            checked += 1
            dotted = leaf_cfg["path"].replace("/", ".")

            for dist_name in covered_only_by_extra:
                root = _IMPORT_ROOT_OF_DISTRIBUTION.get(dist_name)
                assert root is not None, (
                    f"{leaf_name} depends on {dist_name!r}, which {bundle_name} covers only through its "
                    f"'{bundle_name}' optional extra; add {dist_name!r} to _IMPORT_ROOT_OF_DISTRIBUTION "
                    "in this test so its manifest can be checked for an import-safe degrade"
                )
                monkeypatch.setitem(sys.modules, root, None)
                for name in list(sys.modules):
                    if name == root or name.startswith(f"{root}."):
                        monkeypatch.setitem(sys.modules, name, None)

            for name in list(sys.modules):
                if name == dotted or name.startswith(f"{dotted}."):
                    monkeypatch.delitem(sys.modules, name, raising=False)

            manifest_name = f"{dotted}.manifest"
            # monkeypatch.delitem is a no-op (untracked) when the key is absent, but the imports below
            # insert fresh modules directly into sys.modules; setitem always tracks a restore, even for
            # an absent key, so this guarantees teardown removes whatever we cold-import next, however
            # it was cached (or not) before this test ran.
            for fresh_name in (dotted, manifest_name):
                monkeypatch.setitem(sys.modules, fresh_name, None)
                monkeypatch.delitem(sys.modules, fresh_name, raising=False)

            module = importlib.import_module(manifest_name)

            for group in groups:
                attr = gen.ENTRY_POINT_ATTRS[group]
                assert getattr(module, attr) == [], (
                    f"{dotted}.manifest.{attr} is not an empty list after blocking "
                    f"{covered_only_by_extra}; {leaf_name} is covered only through {bundle_name}'s "
                    "extra, so its manifest must degrade to an empty list when that dependency is absent"
                )

            exposed_attrs = _EXPOSED_EXTENDER_NAMES.get(leaf_name)
            assert exposed_attrs is not None, (
                f"{leaf_name} has no entry in _EXPOSED_EXTENDER_NAMES in this test; add the attribute "
                "name(s) its package __init__ must not expose when the extra-only dependency is absent"
            )
            leaf_module = importlib.import_module(dotted)
            for attr in exposed_attrs:
                assert not hasattr(leaf_module, attr), (
                    f"{dotted} still exposes {attr!r} after blocking {covered_only_by_extra}; "
                    f"{leaf_name} is covered only through {bundle_name}'s extra, so the package's "
                    "__init__ must not import it eagerly"
                )

    assert checked, (
        "expected at least one entry-point-bundle nested leaf whose dependency is covered only through "
        "the bundle's own optional extra (e.g. mloda-community-openlineage / openlineage-python)"
    )

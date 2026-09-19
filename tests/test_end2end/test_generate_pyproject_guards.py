"""Robustness guards for scripts/generate_pyproject.py.

The generator is the single source of truth for every package's
``pyproject.toml`` and for the root mloda-core pin. Eight silent-failure modes
must be turned into loud failures:

Guard 1 -- a missing ``[defaults].core_dependency`` must raise, not silently
substitute ``""`` and emit ``dependencies = [""]``.

Guard 2 -- write mode must exit non-zero when the root mloda-core dependency
entry cannot be synced, instead of returning 0 and leaving a stale pin.

Guard 3 -- a meta-package (``workspace_deps``) flagged ``py_typed`` must raise,
instead of emitting ``packages = []`` and a wheel without its PEP 561 marker.

Guard 4 -- ``published`` and ``optional_dependency_indexes`` must be mutually
exclusive, since a built wheel's metadata carries no uv index scoping.

Guard 5 -- an ``optional_dependency_indexes`` entry missing ``explicit = true``
must raise, instead of letting the index shadow PyPI for other packages too.

Guard 6 -- an ``optional_dependency_indexes`` key naming no declared optional
dependency must raise, instead of leaving the real dependency to resolve from
the default index.

Guard 7 -- a dotted spelling of an ``optional_dependency_indexes`` key must
still emit a flat ``[tool.uv.sources]`` entry, instead of an unquoted dot
turning it into a nested TOML table with no source entry for the real
dependency.

Guard 8 -- an ``optional_dependency_indexes`` key colliding with an existing
``[tool.uv.sources]`` workspace entry must raise, naming the collision,
instead of emitting a duplicate TOML key.

The generator lives at ``scripts/generate_pyproject.py`` (a script, not an
installed package), so it is loaded here by file path.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.script_loader import load_script

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GEN_PATH = _REPO_ROOT / "scripts" / "generate_pyproject.py"
_ROOT_PYPROJECT = _REPO_ROOT / "pyproject.toml"

gen = load_script("generate_pyproject", _GEN_PATH)


def _tool_uv_table(content: str) -> dict[str, Any]:
    """Parse ``content`` with tomllib and return its ``[tool.uv]`` table, or ``{}`` if absent."""
    parsed: dict[str, Any] = tomllib.loads(content)
    tool: dict[str, Any] = parsed.get("tool", {})
    uv_table: dict[str, Any] = tool.get("uv", {})
    return uv_table


def test_generate_raises_when_core_dependency_missing() -> None:
    """generate_pyproject must fail loudly if core_dependency is absent.

    Today ``defaults.get("core_dependency", "")`` turns a missing key into an
    empty string, so ``"{core_dependency}"`` placeholders collapse to ``""``
    and packages get an invalid ``dependencies = [""]``. This must raise
    ``ValueError`` instead.
    """
    shared, packages_config = gen.load_configs()
    packages = packages_config.get("packages", {})
    assert "core_dependency" in shared["defaults"], "fixture assumption: shared config defines core_dependency"

    # Simulate a misconfigured shared.toml with the pin removed.
    shared["defaults"].pop("core_dependency", None)

    pkg_config = packages["mloda-registry"]
    assert "{core_dependency}" in pkg_config.get("dependencies", []), (
        "fixture assumption: mloda-registry deps use the {core_dependency} placeholder"
    )

    with pytest.raises(ValueError):
        content = gen.generate_pyproject("mloda-registry", pkg_config, shared, packages)
        # Belt and suspenders: if it did NOT raise, it must at least not have
        # emitted an empty dependency entry. This makes the current failure
        # mode (silent "") produce a descriptive assertion rather than a bare
        # "DID NOT RAISE".
        assert '[""]' not in content and 'dependencies = [""]' not in content, (
            f"generate_pyproject silently produced an empty core dependency:\n{content}"
        )


def test_write_mode_exits_nonzero_when_root_entry_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Write mode must return non-zero when the root mloda-core entry is unsyncable.

    The generator's file-path globals are CWD-relative, so an isolated
    workspace under ``tmp_path`` (real configs copied in, a root
    ``pyproject.toml`` with no ``mloda`` core entry) fully sandboxes the run.
    ``update_root_core_dependency`` then returns ``(False, ...)``. Today
    ``main()`` ignores that in write mode and returns 0; it must return 1.
    """
    # Snapshot the real root pyproject to prove the run never touches the repo.
    real_root_before = _ROOT_PYPROJECT.read_text()

    # Build an isolated workspace: real configs, sandboxed root pyproject.
    (tmp_path / "config").mkdir()
    shutil.copy(_REPO_ROOT / "config" / "shared.toml", tmp_path / "config" / "shared.toml")
    shutil.copy(_REPO_ROOT / "config" / "packages.toml", tmp_path / "config" / "packages.toml")

    # Root pyproject deliberately lacks any ``mloda`` core dependency entry,
    # so update_root_core_dependency cannot find one to sync.
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "sandbox-root"\ndependencies = ["requests>=2.0"]\n',
    )

    # CWD-relative globals now resolve inside the sandbox.
    monkeypatch.chdir(tmp_path)
    # Write mode: no --check flag.
    monkeypatch.setattr(sys, "argv", ["generate_pyproject.py"])

    return_code = gen.main()

    assert return_code == 1, (
        "generate_pyproject.main() in write mode must exit non-zero when the root "
        f"mloda-core dependency entry cannot be synced, but it returned {return_code!r}."
    )

    # The real repository must be untouched by the sandboxed run.
    assert _ROOT_PYPROJECT.read_text() == real_root_before, (
        "the sandboxed generator run modified the real repository root pyproject.toml"
    )


def _meta_package_config(py_typed: bool | None = None) -> dict[str, Any]:
    """Synthetic meta-package config, optionally flagged ``py_typed``."""
    pkg_config: dict[str, Any] = {
        "description": "Meta package aggregating workspace members",
        "path": "mloda/meta",
        "dependencies": [],
        "workspace_deps": ["mloda-registry"],
    }
    if py_typed is not None:
        pkg_config["py_typed"] = py_typed
    return pkg_config


def test_generate_raises_when_meta_package_is_flagged_py_typed() -> None:
    """The ``workspace_deps`` branch emits ``packages = []`` and never consults ``py_typed``."""
    shared, _packages_config = gen.load_configs()
    pkg_config = _meta_package_config(py_typed=True)
    all_packages: dict[str, dict[str, Any]] = {"mloda-meta": pkg_config}

    with pytest.raises(ValueError, match="py_typed"):
        gen.generate_pyproject("mloda-meta", pkg_config, shared, all_packages)


def test_meta_package_without_py_typed_still_generates() -> None:
    """The guard fires on the combination only, not on every meta-package."""
    shared, _packages_config = gen.load_configs()
    pkg_config = _meta_package_config()
    all_packages: dict[str, dict[str, Any]] = {"mloda-meta": pkg_config}

    content: str = gen.generate_pyproject("mloda-meta", pkg_config, shared, all_packages)

    assert "packages = []" in content, content
    assert "package-data" not in content, content


def test_generate_raises_when_a_published_package_uses_optional_dependency_indexes() -> None:
    """Guard 4 -- ``published`` and ``optional_dependency_indexes`` must be mutually exclusive.

    A built wheel's metadata carries only the bare requirement (e.g.
    ``mloda-example-binary>=0.1.0,<0.2.0``); uv's own index configuration does not survive into
    it. If a *published* package also points an extra at a non-default index (TestPyPI, say),
    ``pip install pkg[extra]`` from a real install resolves that bare name against production
    PyPI instead, which may be empty or squatted: a dependency-confusion vector.
    """
    shared, packages_config = gen.load_configs()
    packages: dict[str, dict[str, Any]] = packages_config["packages"]
    pkg_config: dict[str, Any] = {
        "description": "synthetic published package pointing an extra at a non-default index",
        "path": "mloda/sandbox_published_with_index",
        "dependencies": ["{core_dependency}"],
        "published": True,
        "optional_dependencies": {"wheel": ["mloda-example-binary>=0.1.0,<0.2.0"]},
        "optional_dependency_indexes": {"mloda-example-binary": "testpypi"},
    }
    all_packages: dict[str, dict[str, Any]] = {**packages, "mloda-sandbox-published-with-index": pkg_config}

    with pytest.raises(ValueError, match="published"):
        gen.generate_pyproject("mloda-sandbox-published-with-index", pkg_config, shared, all_packages)


def test_generate_accepts_published_package_without_optional_dependency_indexes() -> None:
    """Guard 4 fires on the ``published`` + ``optional_dependency_indexes`` combination only."""
    shared, packages_config = gen.load_configs()
    packages: dict[str, dict[str, Any]] = packages_config["packages"]
    pkg_config: dict[str, Any] = {
        "description": "synthetic published package with no index dependency",
        "path": "mloda/sandbox_published_plain",
        "dependencies": ["{core_dependency}"],
        "published": True,
    }
    all_packages: dict[str, dict[str, Any]] = {**packages, "mloda-sandbox-published-plain": pkg_config}

    content = gen.generate_pyproject("mloda-sandbox-published-plain", pkg_config, shared, all_packages)
    uv_table = _tool_uv_table(content)
    assert "index" not in uv_table, uv_table


def test_generate_raises_when_optional_dependency_indexes_key_is_not_declared_in_any_extra() -> None:
    """A typo'd optional_dependency_indexes key names a dependency absent from every extra."""
    shared, packages_config = gen.load_configs()
    packages: dict[str, dict[str, Any]] = packages_config["packages"]
    pkg_config: dict[str, Any] = {
        "description": "synthetic package pointing an index at a mistyped dependency name",
        "path": "mloda/sandbox_typo_index",
        "dependencies": ["{core_dependency}"],
        "optional_dependencies": {"wheel": ["mloda-example-binary>=0.1.0,<0.2.0"]},
        "optional_dependency_indexes": {"mloda-exampl-binary": "testpypi"},
    }
    all_packages: dict[str, dict[str, Any]] = {**packages, "mloda-sandbox-typo-index": pkg_config}

    with pytest.raises(ValueError, match=re.escape("mloda-exampl-binary")):
        gen.generate_pyproject("mloda-sandbox-typo-index", pkg_config, shared, all_packages)


def test_generate_accepts_optional_dependency_indexes_key_normalised_against_underscore_extra() -> None:
    """A PEP 503 equivalent spelling (underscore extra entry, hyphenated index key) is not a typo."""
    shared, packages_config = gen.load_configs()
    packages: dict[str, dict[str, Any]] = packages_config["packages"]
    pkg_config: dict[str, Any] = {
        "description": "synthetic package pointing an index at an underscore-spelled extra entry",
        "path": "mloda/sandbox_normalised_index",
        "dependencies": ["{core_dependency}"],
        "optional_dependencies": {"wheel": ["mloda_example_binary>=0.1.0,<0.2.0"]},
        "optional_dependency_indexes": {"mloda-example-binary": "testpypi"},
    }
    all_packages: dict[str, dict[str, Any]] = {**packages, "mloda-sandbox-normalised-index": pkg_config}

    content = gen.generate_pyproject("mloda-sandbox-normalised-index", pkg_config, shared, all_packages)
    uv_table = _tool_uv_table(content)
    assert uv_table.get("sources", {}).get("mloda-example-binary") == {"index": "testpypi"}, uv_table
    testpypi_url = shared["defaults"]["uv_indexes"]["testpypi"]["url"]
    assert {"name": "testpypi", "url": testpypi_url, "explicit": True} in uv_table.get("index", []), uv_table


def test_generate_accepts_optional_dependency_indexes_key_declared_only_via_shared_defaults() -> None:
    """The declared-dependency check runs against the merged extras, including the shared dev default."""
    shared, packages_config = gen.load_configs()
    packages: dict[str, dict[str, Any]] = packages_config["packages"]
    pkg_config: dict[str, Any] = {
        "description": "synthetic package pointing an index at a dependency contributed only by the shared default",
        "path": "mloda/sandbox_default_index",
        "dependencies": ["{core_dependency}"],
        "optional_dependency_indexes": {"pytest": "testpypi"},
    }
    all_packages: dict[str, dict[str, Any]] = {**packages, "mloda-sandbox-default-index": pkg_config}

    content = gen.generate_pyproject("mloda-sandbox-default-index", pkg_config, shared, all_packages)
    uv_table = _tool_uv_table(content)
    assert uv_table.get("sources", {}).get("pytest") == {"index": "testpypi"}, uv_table
    testpypi_url = shared["defaults"]["uv_indexes"]["testpypi"]["url"]
    assert {"name": "testpypi", "url": testpypi_url, "explicit": True} in uv_table.get("index", []), uv_table


def test_generate_accepts_optional_dependency_indexes_key_declared_only_via_published_children() -> None:
    """The declared-dependency check runs against the merged extras, after {published_children} expansion."""
    shared, packages_config = gen.load_configs()
    packages: dict[str, dict[str, Any]] = packages_config["packages"]
    real_data_operations = packages["mloda-community-data-operations"]
    assert real_data_operations.get("optional_dependencies", {}).get("all") == [gen.PUBLISHED_CHILDREN], (
        "fixture assumption: mloda-community-data-operations 'all' extra uses the {published_children} placeholder"
    )
    pkg_config: dict[str, Any] = {
        **real_data_operations,
        "published": False,
        "optional_dependency_indexes": {"mloda-community-aggregation": "testpypi"},
    }
    all_packages: dict[str, dict[str, Any]] = {**packages, "mloda-community-data-operations": pkg_config}

    content = gen.generate_pyproject("mloda-community-data-operations", pkg_config, shared, all_packages)
    uv_table = _tool_uv_table(content)
    assert uv_table.get("sources", {}).get("mloda-community-aggregation") == {"index": "testpypi"}, uv_table
    testpypi_url = shared["defaults"]["uv_indexes"]["testpypi"]["url"]
    assert {"name": "testpypi", "url": testpypi_url, "explicit": True} in uv_table.get("index", []), uv_table


def test_generate_dotted_index_key_produces_a_flat_uv_source_entry() -> None:
    """A PEP 503 dotted spelling of the index key must still emit a single flat [tool.uv.sources]
    entry for the real dependency name, not a nested TOML table from the unquoted dots."""
    shared, packages_config = gen.load_configs()
    packages: dict[str, dict[str, Any]] = packages_config["packages"]
    pkg_config: dict[str, Any] = {
        "description": "synthetic package pointing an index at a dotted-spelled dependency name",
        "path": "mloda/sandbox_dotted_index",
        "dependencies": ["{core_dependency}"],
        "optional_dependencies": {"wheel": ["mloda-example-binary>=0.1.0,<0.2.0"]},
        "optional_dependency_indexes": {"mloda.example.binary": "testpypi"},
    }
    all_packages: dict[str, dict[str, Any]] = {**packages, "mloda-sandbox-dotted-index": pkg_config}

    content = gen.generate_pyproject("mloda-sandbox-dotted-index", pkg_config, shared, all_packages)

    uv_table = _tool_uv_table(content)
    sources = uv_table.get("sources", {})
    assert sources.get("mloda-example-binary") == {"index": "testpypi"}, sources
    assert "mloda" not in sources, sources


def test_generate_raises_when_index_key_collides_with_a_workspace_source_name() -> None:
    """An optional_dependency_indexes key equal to an existing [tool.uv.sources] workspace entry
    must raise, naming the colliding name, instead of emitting a duplicate TOML key."""
    shared, packages_config = gen.load_configs()
    packages: dict[str, dict[str, Any]] = packages_config["packages"]
    pkg_config: dict[str, Any] = {
        "description": "synthetic unpublished package pointing an index at a name already used by a workspace source",
        "path": "mloda/sandbox_collision_index",
        "dependencies": ["{core_dependency}"],
        "optional_dependency_indexes": {"mloda-testing": "testpypi"},
    }
    all_packages: dict[str, dict[str, Any]] = {**packages, "mloda-sandbox-collision-index": pkg_config}

    with pytest.raises(ValueError, match="mloda-testing"):
        gen.generate_pyproject("mloda-sandbox-collision-index", pkg_config, shared, all_packages)


def test_real_config_optional_dependency_indexes_reference_a_declared_dependency() -> None:
    """The real mloda-enterprise-binary-example wheel extra satisfies the declared-dependency guard."""
    shared, packages_config = gen.load_configs()
    packages: dict[str, dict[str, Any]] = packages_config["packages"]
    pkg_config = packages["mloda-enterprise-binary-example"]
    assert pkg_config.get("optional_dependency_indexes") == {"mloda-example-binary": "testpypi"}, (
        "fixture assumption: mloda-enterprise-binary-example pins mloda-example-binary to testpypi"
    )

    content = gen.generate_pyproject("mloda-enterprise-binary-example", pkg_config, shared, packages)
    uv_table = _tool_uv_table(content)
    assert uv_table.get("sources", {}).get("mloda-example-binary") == {"index": "testpypi"}, uv_table
    testpypi_url = shared["defaults"]["uv_indexes"]["testpypi"]["url"]
    assert {"name": "testpypi", "url": testpypi_url, "explicit": True} in uv_table.get("index", []), uv_table


def test_uv_index_blocks_raises_when_explicit_is_missing() -> None:
    """Guard 5 -- an index config missing ``explicit = true`` must be rejected.

    ``explicit = true`` is the actual dependency-confusion safeguard: without it, uv's
    first-index strategy lets the index shadow PyPI for *other* packages too, not just the one
    dependency naming it via ``[tool.uv.sources]``.
    """
    index_deps = {"mloda-example-binary": "testpypi"}
    uv_indexes: dict[str, Any] = {"testpypi": {"url": "https://test.pypi.org/simple/"}}

    with pytest.raises(ValueError, match="explicit"):
        gen.uv_index_blocks("mloda-sandbox", index_deps, uv_indexes)


def test_uv_index_blocks_raises_when_explicit_is_false() -> None:
    """Guard 5 -- an index config with ``explicit = false`` is rejected the same as a missing key."""
    index_deps = {"mloda-example-binary": "testpypi"}
    uv_indexes: dict[str, Any] = {"testpypi": {"url": "https://test.pypi.org/simple/", "explicit": False}}

    with pytest.raises(ValueError, match="explicit"):
        gen.uv_index_blocks("mloda-sandbox", index_deps, uv_indexes)


def test_uv_index_blocks_accepts_explicit_true() -> None:
    """Guard 5 fires on the missing/false case only; a correctly configured index is accepted."""
    index_deps = {"mloda-example-binary": "testpypi"}
    uv_indexes: dict[str, Any] = {"testpypi": {"url": "https://test.pypi.org/simple/", "explicit": True}}

    lines = gen.uv_index_blocks("mloda-sandbox", index_deps, uv_indexes)
    assert "explicit = true" in lines, lines


def test_real_shared_toml_uv_indexes_all_declare_explicit_true() -> None:
    """Regression guard on the real config: every configured uv index in config/shared.toml must
    declare explicit = true, not merely something Guard 5 could theoretically enforce."""
    shared, _packages_config = gen.load_configs()
    uv_indexes: dict[str, Any] = shared.get("defaults", {}).get("uv_indexes", {})
    assert uv_indexes, "expected at least one configured uv index; check is vacuous"
    for name, index_cfg in uv_indexes.items():
        assert index_cfg.get("explicit") is True, f"{name}: uv index must declare explicit = true, got {index_cfg!r}"


def test_discover_packages_excludes_real_egg_info_dirs(tmp_path: Path) -> None:
    """A planted ``<package>.egg-info/__init__.py`` must not be discovered as a package.

    ``discover_packages`` excluded directories via an exact-part match
    against the literal string ``".egg-info"``. Real egg-info directories are
    named ``<package>.egg-info`` (e.g. ``foo.egg-info``), so the exact
    comparison never matched and the entry was dead: a synthetic
    ``foo.egg-info`` package directory got discovered like any other package.
    """
    pkg_root = tmp_path / "mloda" / "demo"
    real_pkg = pkg_root / "widgets"
    real_pkg.mkdir(parents=True)
    (real_pkg / "__init__.py").write_text("")

    egg_info_pkg = pkg_root / "widgets.egg-info"
    egg_info_pkg.mkdir(parents=True)
    (egg_info_pkg / "__init__.py").write_text("")

    discovered = gen.discover_packages(str(pkg_root))

    assert any(pkg.endswith("widgets") and "egg-info" not in pkg for pkg in discovered), (
        f"expected the real widgets package to be discovered, got: {discovered}"
    )
    assert not any("egg-info" in pkg for pkg in discovered), (
        f"a synthetic '.egg-info' package leaked into discovery: {discovered}"
    )

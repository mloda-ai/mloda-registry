"""Robustness guards for scripts/generate_pyproject.py.

The generator is the single source of truth for every package's
``pyproject.toml`` and for the root mloda-core pin. Five silent-failure modes
must be turned into loud failures:

Guard 1 -- a missing ``[defaults].core_dependency`` must raise, not silently
substitute ``""`` and emit ``dependencies = [""]``.

Guard 2 -- write mode must exit non-zero when the root mloda-core dependency
entry cannot be synced, instead of returning 0 and leaving a stale pin.

Guard 3 -- a meta-package (``workspace_deps``) flagged ``py_typed`` must raise,
instead of emitting ``packages = []`` and a wheel without its PEP 561 marker.

Guard 4 -- a configured package path with no Python package of its own must raise, not yield ``packages = []``.

Guard 5 -- the root core-dependency marker comment must be enforced: a missing
or stale marker fails --check, and write mode restores it. The marker must sit
directly above the entry, a commented-out pin is never matched, and multiple entries fail.

The generator lives at ``scripts/generate_pyproject.py`` (a script, not an
installed package), so it is loaded here by file path.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.script_loader import load_script
from tests.toml_loader import load_toml, loads_toml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GEN_PATH = _REPO_ROOT / "scripts" / "generate_pyproject.py"
_ROOT_PYPROJECT = _REPO_ROOT / "pyproject.toml"

gen = load_script("generate_pyproject", _GEN_PATH)


def test_generate_quotes_all_configured_toml_string_values() -> None:
    """Every configured string surface must survive TOML generation unchanged."""
    special = 'value with "quotes" and \\backslashes'
    configured_path = f"missing/{special}"
    dotted_path = configured_path.replace("/", ".")
    entry_point_target = f"{dotted_path}.manifest:FEATURE_GROUPS"
    shared = {
        "build-system": {"requires": [special], "build-backend": special},
        "project": {
            "version": special,
            "authors": [{"name": special, "email": special}],
            "requires-python": special,
            "urls": {"Homepage": special},
        },
        "defaults": {"license": special, "optional_dependencies": {"default-extra": [special]}},
    }
    pkg_config = {
        "description": special,
        "path": configured_path,
        "dependencies": [special],
        "optional_dependencies": {"package-extra": [special]},
        "entry_point_groups": ["mloda.feature_groups"],
        "py_typed": True,
    }

    generated = gen.generate_pyproject(special, pkg_config, shared, {special: pkg_config})
    parsed = loads_toml(generated)

    assert parsed["build-system"] == {"requires": [special], "build-backend": special}
    assert parsed["project"] == {
        "name": special,
        "version": special,
        "description": special,
        "license": special,
        "authors": [{"name": special, "email": special}],
        "dependencies": [special],
        "requires-python": special,
        "optional-dependencies": {"default-extra": [special], "package-extra": [special]},
        "urls": {"Homepage": special},
        "entry-points": {"mloda.feature_groups": {special: entry_point_target}},
    }
    assert parsed["tool"]["setuptools"] == {
        "package-dir": {"": "../.."},
        "packages": [dotted_path],
        "package-data": {dotted_path: ["py.typed"]},
    }


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
    # Every configured package path needs a Python package so the path guard does not fire first.
    real_packages = load_toml(_REPO_ROOT / "config" / "packages.toml")["packages"]
    for real_cfg in real_packages.values():
        (tmp_path / real_cfg["path"]).mkdir(parents=True, exist_ok=True)
        (tmp_path / real_cfg["path"] / "__init__.py").write_text("")

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


_SYNTHETIC_SHARED = {"defaults": {"core_dependency": "mloda>=0.15.0,<0.16.0"}}
_CORE_ENTRY_LINE = '    "mloda>=0.15.0,<0.16.0",\n'


def _root_content(marker_line: str | None) -> str:
    """Synthetic root pyproject text with an optional line above the core entry."""
    above = "" if marker_line is None else marker_line + "\n"
    return f'[project]\nname = "sandbox-root"\ndependencies = [\n    "requests>=2.0",\n{above}{_CORE_ENTRY_LINE}]\n'


@pytest.mark.parametrize(
    "marker_line",
    [None, "    # Generated from config/shared.toml (old wording)"],
    ids=["missing_marker", "stale_marker"],
)
def test_check_mode_fails_when_root_core_marker_missing_or_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker_line: str | None
) -> None:
    """Check mode must report out of date for a missing or stale marker and not write."""
    root = tmp_path / "pyproject.toml"
    content = _root_content(marker_line)
    root.write_text(content)
    monkeypatch.setattr(gen, "ROOT_PYPROJECT", root)

    ok, msg = gen.update_root_core_dependency(_SYNTHETIC_SHARED, check=True)

    assert ok is False
    assert "out of date" in msg
    assert root.read_text() == content


def test_write_mode_inserts_root_core_marker_idempotently(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Write mode inserts the marker above the entry, then is a byte-identical no-op."""
    root = tmp_path / "pyproject.toml"
    root.write_text(_root_content(None))
    monkeypatch.setattr(gen, "ROOT_PYPROJECT", root)

    assert gen.update_root_core_dependency(_SYNTHETIC_SHARED, check=False) == (True, "updated")
    lines = root.read_text().splitlines()
    entry_index = lines.index(_CORE_ENTRY_LINE.rstrip("\n"))
    assert lines[entry_index - 1] == "    " + gen.CORE_DEPENDENCY_MARKER

    after_first = root.read_bytes()
    assert gen.update_root_core_dependency(_SYNTHETIC_SHARED, check=False) == (True, "up-to-date")
    assert root.read_bytes() == after_first
    assert gen.update_root_core_dependency(_SYNTHETIC_SHARED, check=True) == (True, "up-to-date")


def test_write_mode_ignores_commented_out_core_pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A commented-out pin above the entry must stay a comment, not become a live dependency."""
    commented = '    # "mloda>=0.13.0",'
    root = tmp_path / "pyproject.toml"
    root.write_text(_root_content(commented))
    monkeypatch.setattr(gen, "ROOT_PYPROJECT", root)

    ok, _msg = gen.update_root_core_dependency(_SYNTHETIC_SHARED, check=False)

    text = root.read_text()
    assert ok is True
    deps = loads_toml(text)["project"]["dependencies"]
    assert [d for d in deps if d.startswith("mloda")] == [_SYNTHETIC_SHARED["defaults"]["core_dependency"]], text
    assert commented in text.splitlines()


@pytest.mark.parametrize("check", [True, False], ids=["check", "write"])
def test_root_core_dependency_fails_when_multiple_entries_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, check: bool
) -> None:
    """More than one live mloda entry must fail without writing."""
    root = tmp_path / "pyproject.toml"
    content = _root_content(None) + '[project.optional-dependencies]\nextra = [\n    "mloda>=0.14.0",\n]\n'
    root.write_text(content)
    monkeypatch.setattr(gen, "ROOT_PYPROJECT", root)

    ok, _msg = gen.update_root_core_dependency(_SYNTHETIC_SHARED, check=check)

    assert ok is False
    assert root.read_text() == content


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


def test_generate_emits_flat_workspace_source_for_dotted_package_name() -> None:
    """An unquoted dot in a workspace source key would parse as a nested table."""
    shared, _packages_config = gen.load_configs()
    pkg_config = _meta_package_config()
    pkg_config["workspace_deps"] = ["mloda.foo", "mloda-registry"]
    all_packages: dict[str, dict[str, Any]] = {"mloda-meta": pkg_config}

    content: str = gen.generate_pyproject("mloda-meta", pkg_config, shared, all_packages)
    parsed = loads_toml(content)

    assert parsed["tool"]["uv"]["sources"] == {
        "mloda.foo": {"workspace": True},
        "mloda-registry": {"workspace": True},
    }, content


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


def _leaf_config(path: str) -> dict[str, Any]:
    """Synthetic non-bundle package config at ``path``."""
    return {"description": "Leaf", "path": path, "dependencies": []}


@pytest.mark.parametrize(
    ("create", "init", "expect_raise"),
    [
        pytest.param(False, False, True, id="path-missing"),
        pytest.param(True, False, True, id="init-missing"),
        pytest.param(True, True, False, id="package-with-code"),
    ],
)
def test_validate_package_paths_requires_code_at_the_configured_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, create: bool, init: bool, expect_raise: bool
) -> None:
    """A missing path or a directory without ``__init__.py`` raises naming package and path; real code passes."""
    monkeypatch.chdir(tmp_path)
    if create:
        (tmp_path / "mloda" / "pkg").mkdir(parents=True)
    if init:
        (tmp_path / "mloda" / "pkg" / "__init__.py").write_text("")
    packages = {"mloda-pkg": _leaf_config("mloda/pkg")}

    if not expect_raise:
        gen.validate_package_paths(packages)
        return
    with pytest.raises(ValueError, match="mloda-pkg") as excinfo:
        gen.validate_package_paths(packages)
    assert "mloda/pkg" in str(excinfo.value)


def test_validate_package_paths_skips_bundle_and_meta_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An entry_point_bundle and a workspace_deps meta-package may have no own code."""
    monkeypatch.chdir(tmp_path)
    packages = {
        "mloda-bundle": {**_leaf_config("mloda/bundle"), "entry_point_bundle": True},
        "mloda-meta": _meta_package_config(),
    }

    gen.validate_package_paths(packages)


def test_validate_package_paths_raises_when_only_nested_package_has_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Code under a nested configured package's path is excluded, so the parent discovers nothing of its own."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mloda" / "parent" / "child").mkdir(parents=True)
    (tmp_path / "mloda" / "parent" / "child" / "__init__.py").write_text("")
    packages = {
        "mloda-parent": _leaf_config("mloda/parent"),
        "mloda-child": _leaf_config("mloda/parent/child"),
    }

    with pytest.raises(ValueError, match="mloda-parent"):
        gen.validate_package_paths(packages)


def test_main_raises_when_configured_package_has_no_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """main() must surface a code-less configured package before generating anything."""
    (tmp_path / "config").mkdir()
    shutil.copy(_REPO_ROOT / "config" / "shared.toml", tmp_path / "config" / "shared.toml")
    (tmp_path / "config" / "packages.toml").write_text(
        '[packages.mloda-ghost]\ndescription = "Ghost"\npath = "mloda/ghost"\ndependencies = []\n'
    )
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "sandbox-root"\ndependencies = []\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["generate_pyproject.py"])

    with pytest.raises(ValueError, match="mloda-ghost"):
        gen.main()

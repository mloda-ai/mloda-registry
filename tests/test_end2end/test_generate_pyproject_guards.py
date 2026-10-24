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

import copy
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

_LICENSE_SOURCE_FILES = {
    "LICENSE-APACHE": b"apache text\n",
    "NOTICE.md": b"notice text\n",
    "mloda/enterprise/LICENSE": b"prop\n",
}


def _write_license_sources(root: Path) -> None:
    """Create the license source files the generator copies from, under ``root``."""
    for rel, data in _LICENSE_SOURCE_FILES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)


def _sandbox_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, packages_toml: str, root_pyproject: str, *argv: str
) -> None:
    """Isolated generator workspace: real shared.toml, given packages.toml and root pyproject, chdir, argv."""
    (tmp_path / "config").mkdir()
    shutil.copy(_REPO_ROOT / "config" / "shared.toml", tmp_path / "config" / "shared.toml")
    (tmp_path / "config" / "packages.toml").write_text(packages_toml)
    (tmp_path / "pyproject.toml").write_text(root_pyproject)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["generate_pyproject.py", *argv])


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
            "keywords": [special, "shared-kw"],
            "classifiers": [special],
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
        "keywords": ["pkg-kw", special],
    }

    generated = gen.generate_pyproject(special, pkg_config, shared, {special: pkg_config})
    parsed = loads_toml(generated)

    assert parsed["build-system"] == {"requires": [special], "build-backend": special}
    assert parsed["project"] == {
        "name": special,
        "version": special,
        "description": special,
        "license": special,
        "license-files": ["LICENSE", "NOTICE"],
        "authors": [{"name": special, "email": special}],
        "dependencies": [special],
        "requires-python": special,
        "optional-dependencies": {"default-extra": [special], "package-extra": [special]},
        "urls": {"Homepage": special},
        "entry-points": {"mloda.feature_groups": {special: entry_point_target}},
        "keywords": [special, "shared-kw", "pkg-kw"],
        "classifiers": [special, "Typing :: Typed"],
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


@pytest.mark.parametrize(
    ("where", "bad_keywords"),
    [
        pytest.param("package", "tracing", id="package-string"),
        pytest.param("package", ["tracing", 1], id="package-non-string-item"),
        pytest.param("shared", "mloda", id="shared-string"),
    ],
)
def test_generate_raises_when_keywords_not_a_list_of_strings(where: str, bad_keywords: Any) -> None:
    """A mistyped keywords value must raise ValueError naming keywords, not be split into characters."""
    shared, packages_config = gen.load_configs()
    packages = copy.deepcopy(packages_config.get("packages", {}))
    shared = copy.deepcopy(shared)
    pkg_config = packages["mloda-registry"]
    if where == "package":
        pkg_config["keywords"] = bad_keywords
    else:
        shared["project"]["keywords"] = bad_keywords

    with pytest.raises(ValueError, match="keywords"):
        gen.generate_pyproject("mloda-registry", pkg_config, shared, packages)


def _authors_fixture(pkg_authors: Any = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """Shared config with one email author, plus a leaf package config, optionally with per-package authors."""
    shared, packages_config = gen.load_configs()
    shared = copy.deepcopy(shared)
    shared["project"]["authors"] = [{"name": "Tom Kaltofen", "email": "info@mloda.ai"}]
    pkg_config = copy.deepcopy(packages_config["packages"]["mloda-registry"])
    pkg_config.pop("authors", None)
    if pkg_authors is not None:
        pkg_config["authors"] = pkg_authors
    return shared, pkg_config, {"mloda-registry": pkg_config}


def test_generate_appends_package_authors_after_shared_authors() -> None:
    """Per-package authors follow the shared ones; a name-only entry renders without email."""
    shared, pkg_config, packages = _authors_fixture([{"name": "Martin Kaiser"}, {"name": "Ada", "email": "a@b.c"}])

    generated = gen.generate_pyproject("mloda-registry", pkg_config, shared, packages)

    assert (
        'authors = [{ name = "Tom Kaltofen", email = "info@mloda.ai" }, { name = "Martin Kaiser" }, '
        '{ name = "Ada", email = "a@b.c" }]'
    ) in generated.splitlines(), generated
    assert loads_toml(generated)["project"]["authors"] == [
        {"name": "Tom Kaltofen", "email": "info@mloda.ai"},
        {"name": "Martin Kaiser"},
        {"name": "Ada", "email": "a@b.c"},
    ]


def test_generate_without_package_authors_emits_only_shared_authors() -> None:
    """A package without authors keeps exactly the shared authors."""
    shared, pkg_config, packages = _authors_fixture()

    generated = gen.generate_pyproject("mloda-registry", pkg_config, shared, packages)

    assert 'authors = [{ name = "Tom Kaltofen", email = "info@mloda.ai" }]' in generated.splitlines(), generated


@pytest.mark.parametrize(
    "bad_authors",
    [
        pytest.param("Martin Kaiser", id="string"),
        pytest.param(["Martin Kaiser"], id="entry-not-a-table"),
        pytest.param([{"email": "m@k.de"}], id="entry-missing-name"),
        pytest.param([{"name": 1}], id="non-string-name"),
        pytest.param([{"name": "Martin Kaiser", "email": 1}], id="non-string-email"),
    ],
)
def test_generate_raises_when_package_authors_malformed(bad_authors: Any) -> None:
    """A mistyped per-package authors value must raise ValueError naming authors and the package."""
    shared, pkg_config, packages = _authors_fixture(bad_authors)

    with pytest.raises(ValueError, match=r"(?s)(?=.*authors)(?=.*mloda-registry)"):
        gen.generate_pyproject("mloda-registry", pkg_config, shared, packages)


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

    # Root pyproject deliberately lacks any ``mloda`` core dependency entry,
    # so update_root_core_dependency cannot find one to sync.
    packages_toml = (_REPO_ROOT / "config" / "packages.toml").read_text()
    _sandbox_workspace(
        tmp_path,
        monkeypatch,
        packages_toml,
        '[project]\nname = "sandbox-root"\ndependencies = ["requests>=2.0"]\n',
    )
    # Every configured package path needs a Python package so the path guard does not fire first.
    for real_cfg in load_toml(_REPO_ROOT / "config" / "packages.toml")["packages"].values():
        (tmp_path / real_cfg["path"]).mkdir(parents=True, exist_ok=True)
        (tmp_path / real_cfg["path"] / "__init__.py").write_text("")
    _write_license_sources(tmp_path)

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


def test_dependency_closure_follows_dependencies_only() -> None:
    packages = {
        "a": {"dependencies": ["b>=1"], "optional_dependencies": {"x": ["d"]}},
        "b": {"dependencies": ["C.C~=2"]},
        "c_c": {"dependencies": []},
        "d": {"dependencies": []},
    }
    assert gen.dependency_closure(packages["a"], packages) == {"b", "c_c"}


def test_dependency_closure_terminates_on_a_cycle() -> None:
    packages = {"a": {"dependencies": ["b"]}, "b": {"dependencies": ["a"]}}
    assert gen.dependency_closure(packages["a"], packages) == {"a", "b"}


def test_dependency_closure_skips_names_not_in_all_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    packages = {"a": {"dependencies": ["b"]}, "b": {"dependencies": []}}
    monkeypatch.setattr(gen, "sibling_dependency_names", lambda deps, all_packages: ["b", "mloda.foo"])
    assert gen.dependency_closure(packages["a"], packages) == {"b"}


@pytest.mark.parametrize(
    ("pkg_name", "expected"),
    [
        pytest.param("under", True, id="depends-on-typed-ancestor-path"),
        pytest.param("sibling", False, id="path-prefix-boundary"),
        pytest.param("extra-only", False, id="extra-dependency-ignored"),
        pytest.param("typed", True, id="flagged-itself"),
    ],
)
def test_has_typed_ancestor_on_synthetic_packages(pkg_name: str, expected: bool) -> None:
    """Typed via own flag or a runtime dependency whose path equals or is an ancestor, on whole segments only."""
    packages: dict[str, dict[str, Any]] = {
        "typed": {"path": "a/b", "dependencies": [], "py_typed": True},
        "under": {"path": "a/b/c", "dependencies": ["typed"]},
        "sibling": {"path": "a/bc", "dependencies": ["typed"]},
        "extra-only": {"path": "a/b/d", "dependencies": [], "optional_dependencies": {"x": ["typed"]}},
    }
    assert gen.has_typed_ancestor(packages[pkg_name], packages) is expected


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
    _sandbox_workspace(
        tmp_path,
        monkeypatch,
        '[packages.mloda-ghost]\ndescription = "Ghost"\npath = "mloda/ghost"\ndependencies = []\n',
        '[project]\nname = "sandbox-root"\ndependencies = []\n',
    )

    with pytest.raises(ValueError, match="mloda-ghost"):
        gen.main()


_LIC_PKG_PATH = "mloda/lic"


def _license_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *argv: str, pkg_path: str = _LIC_PKG_PATH
) -> Path:
    """Sandbox with one package at ``pkg_path``, a valid root pyproject and the license sources."""
    core = load_toml(_REPO_ROOT / "config" / "shared.toml")["defaults"]["core_dependency"]
    _sandbox_workspace(
        tmp_path,
        monkeypatch,
        f'[packages.mloda-lic]\ndescription = "Lic"\npath = "{pkg_path}"\ndependencies = []\n',
        f'[project]\nname = "sandbox-root"\ndependencies = [\n    {gen.CORE_DEPENDENCY_MARKER}\n    "{core}",\n]\n',
        *argv,
    )
    (tmp_path / pkg_path).mkdir(parents=True)
    (tmp_path / pkg_path / "__init__.py").write_text("")
    _write_license_sources(tmp_path)
    return tmp_path / pkg_path


_FORM_CASES = [
    pytest.param(_LIC_PKG_PATH, {"LICENSE": "LICENSE-APACHE", "NOTICE": "NOTICE.md"}, id="apache"),
    pytest.param("mloda/enterprise/x", {"LICENSE": "mloda/enterprise/LICENSE"}, id="proprietary"),
]


@pytest.mark.parametrize("stale", [None, b"stale\n"], ids=["absent", "stale"])
@pytest.mark.parametrize(("pkg_path", "copies"), _FORM_CASES)
def test_write_mode_writes_license_copies_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pkg_path: str,
    copies: dict[str, str],
    stale: bytes | None,
) -> None:
    """Write mode makes byte-equal copies (replacing a stale one), reports each written copy and the count,
    is idempotent (then reporting zero copies written), and check then passes."""
    pkg_dir = _license_workspace(tmp_path, monkeypatch, pkg_path=pkg_path)
    if stale is not None:
        (pkg_dir / "LICENSE").write_bytes(stale)

    assert gen.main() == 0
    out = capsys.readouterr().out
    for name, source in copies.items():
        assert (pkg_dir / name).read_bytes() == _LICENSE_SOURCE_FILES[source]
        assert f"  ✓ {pkg_path}/{name}" in out.splitlines(), out
    assert f"license copies written: {len(copies)}" in out, out
    assert not (pkg_dir / "NOTICE").exists() or "NOTICE" in copies
    before = {p.name: p.read_bytes() for p in pkg_dir.iterdir() if p.is_file()}
    assert gen.main() == 0
    out = capsys.readouterr().out
    assert "license copies written: 0" in out, out
    assert f"✓ {pkg_path}/LICENSE" not in out, out
    assert {p.name: p.read_bytes() for p in pkg_dir.iterdir() if p.is_file()} == before
    monkeypatch.setattr(sys, "argv", ["generate_pyproject.py", "--check"])
    assert gen.main() == 0


@pytest.mark.parametrize(("stale", "reason"), [(None, "missing"), (b"stale\n", "out of date")])
def test_check_mode_fails_on_missing_or_stale_license_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stale: bytes | None,
    reason: str,
) -> None:
    """--check reports the copy as missing or out of date and exits 1 without writing it."""
    pkg_dir = _license_workspace(tmp_path, monkeypatch)
    assert gen.main() == 0
    (pkg_dir / "LICENSE").unlink()
    if stale is not None:
        (pkg_dir / "LICENSE").write_bytes(stale)
    monkeypatch.setattr(sys, "argv", ["generate_pyproject.py", "--check"])
    capsys.readouterr()

    assert gen.main() == 1
    out = capsys.readouterr().out
    assert f"{_LIC_PKG_PATH}/LICENSE: {reason}" in out
    assert "Files out of date" in out
    assert (pkg_dir / "LICENSE").exists() == (stale is not None)


@pytest.mark.parametrize("argv", [(), ("--check",)], ids=["write", "check"])
def test_missing_license_source_raises_naming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: tuple[str, ...]
) -> None:
    """A missing license source raises ValueError naming its path, before any file is written."""
    pkg_dir = _license_workspace(tmp_path, monkeypatch, *argv)
    (tmp_path / "NOTICE.md").unlink()

    with pytest.raises(ValueError, match="NOTICE.md"):
        gen.main()
    assert not (pkg_dir / "pyproject.toml").exists(), "pyproject.toml was written before the source check"
    assert not (pkg_dir / "LICENSE").exists(), "the LICENSE copy was written before the source check"

"""Tests that the set of distributions published to PyPI is declared in a single source of truth.

The released set lives in exactly ONE place: a ``published = true`` flag per package in
``config/packages.toml``. The release workflow, the ``verify-published``
tox env and the data-operations ``all`` extra all derive from it through
``scripts/published_packages.py``. Re-typed copies are how five distributions reached
three of the four and never the build array.

The flag governs the released set only. Wheel boundaries come from the configured layout:
a nested package stays out of its parent's wheel, published or not, with the
``entry_point_bundle`` packages the deliberate exception.
"""

from __future__ import annotations

import os
import re
import subprocess  # nosec
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.script_loader import load_script, version_tuple
from tests.test_end2end.test_verify_independent_installs import _FakeCompletedProcess
from tests.toml_loader import load_toml, loads_toml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHARED_CONFIG = _REPO_ROOT / "config" / "shared.toml"
_PACKAGES_CONFIG = _REPO_ROOT / "config" / "packages.toml"
_GEN_PATH = _REPO_ROOT / "scripts" / "generate_pyproject.py"
_PUBLISHED_SCRIPT = _REPO_ROOT / "scripts" / "published_packages.py"
_IMPORTS_SCRIPT = _REPO_ROOT / "scripts" / "verify_published_imports.py"
_INDEPENDENT_SCRIPT = _REPO_ROOT / "scripts" / "verify_independent_installs.py"
_EXTRAS_SCRIPT = _REPO_ROOT / "scripts" / "verify_extras.py"
_RELEASE_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "release.yaml"
_TOX_INI = _REPO_ROOT / "tox.ini"

# The bundle distributions, always part of the released set.
_BUNDLES = ["mloda-registry", "mloda-testing", "mloda-community", "mloda-enterprise"]

# The released set, in config order: registry, the shared extenders package, the examples, the otel and
# openlineage extenders, the data-operations base plus its plugin packages, the two bundles, which own
# (dependencies/extras) every published package nested under their path, and finally testing, whose
# binary-model extra pins mloda-community, so the published order is dependency-first.
_EXPECTED_PUBLISHED = [
    "mloda-registry",
    "mloda-community-extenders-shared",
    "mloda-community-example",
    "mloda-community-example-a",
    "mloda-community-otel",
    "mloda-community-openlineage",
    "mloda-community-data-operations",
    "mloda-community-aggregation",
    "mloda-community-rank",
    "mloda-community-offset",
    "mloda-community-window-aggregation",
    "mloda-community-frame-aggregate",
    "mloda-community-scalar-aggregate",
    "mloda-community-scalar-arithmetic",
    "mloda-community-point-arithmetic",
    "mloda-community-datetime",
    "mloda-community-string",
    "mloda-community-binning",
    "mloda-community-percentile",
    "mloda-community-time-bucketization",
    "mloda-community-ffill",
    "mloda-community-ema",
    "mloda-community-sessionization",
    "mloda-community-resample",
    "mloda-community",
    "mloda-enterprise",
    "mloda-testing",
]

# Every unpublished package, which reaches users only inside the community and enterprise bundle wheels.
_BUNDLE_ONLY = [
    "mloda-community-binary-model",
    "mloda-enterprise-binary-example",
    "mloda-enterprise-anonymizer",
    "mloda-enterprise-audit",
    "mloda-enterprise-lineage",
]

_DATA_OPERATIONS = "mloda-community-data-operations"

_COMMUNITY_EXAMPLE = "mloda-community-example"

# The child whose flag the wheel-boundary tests drop.
_UNPUBLISH_PROBE = "mloda-community-ema"

# The only wheels that ship nested code.
_ENTRY_POINT_BUNDLES = ["mloda-community", "mloda-enterprise"]

_PUBLISHED_CHILDREN = "{published_children}"

# The tox envs that install the released set from PyPI.
_TOX_PUBLISHED_ENVS = ["verify-published"]

# The tox env that import-checks every installed distribution.
_VERIFY_PUBLISHED_ENV = "verify-published"

# The tox env that installs each top-level distribution into its own venv.
_INDEPENDENT_ENV = "verify-published-independent"

# The tox env that installs internal extras and import-checks their members.
_EXTRAS_ENV = "verify-extras"

# Every tox env that installs distributions by name; none may hand-type them.
_TOX_DERIVED_SET_ENVS = [*_TOX_PUBLISHED_ENVS, _INDEPENDENT_ENV, _EXTRAS_ENV]

_SCRIPT_INVOCATION = "scripts/published_packages.py"

_IMPORTS_INVOCATION = "scripts/verify_published_imports.py"
_INDEPENDENT_INVOCATION = "scripts/verify_independent_installs.py"
_EXTRAS_INVOCATION = "scripts/verify_extras.py"

# Each PyPI-verifying env with the config-derived script that carries its checks.
_TOX_VERIFY_SCRIPTS = [
    (_VERIFY_PUBLISHED_ENV, _IMPORTS_INVOCATION),
    (_INDEPENDENT_ENV, _INDEPENDENT_INVOCATION),
    (_EXTRAS_ENV, _EXTRAS_INVOCATION),
]

_BUILD_STEP = "Build packages"

_PUBLISH_STEP = "Publish to PyPI"

# A line that fills a shell variable from the config-derived script, whatever the variable is named
# (the build step uses 'packages', the publish step may use another name).
_SCRIPT_FILL_RE = re.compile(rf"^[^\n#]*=\s*\$\([^\n]*{re.escape(_SCRIPT_INVOCATION)}", re.MULTILINE)

# A literal bash glob over every file in dist/, the pre-fix upload source (bash order puts
# mloda_community-*.whl before mloda_community_extenders_shared-*.whl).
_DIST_STAR_GLOB_RE = re.compile(r"\(\s*dist/\*\s*\)")

# The publish step's old bash re-implementation of PEP 427/503 distribution name escaping.
_BASH_NAME_ESCAPE_RE = re.compile(r"//-/_")

# The publish step's old per-package 'dist/<escaped>-*.whl' glob, re-implementing verify_builds.find_wheels.
_DIST_NAME_GLOB_RE = re.compile(r"dist/\S*-\*\.whl")

# A hand-written distribution name, quoted or bare. The leading boundary keeps the
# ``/tmp/mloda-verify*`` paths and the dotted ``mloda.community....`` imports out.
_DISTRIBUTION_NAME_RE = re.compile(r"(?<![\w/-])mloda-(?:registry|testing|community|enterprise)[a-z0-9-]*")

# The array filled from the script on one line. A comment cannot match.
_ARRAY_FROM_SCRIPT_RE = re.compile(rf"^[^\n#]*\bpackages\b[^\n#]*{re.escape(_SCRIPT_INVOCATION)}", re.MULTILINE)

gen = load_script("generate_pyproject", _GEN_PATH)


def _packages() -> dict[str, dict[str, Any]]:
    """Config-declared packages, in config order."""
    packages: dict[str, dict[str, Any]] = load_toml(_PACKAGES_CONFIG)["packages"]
    return packages


def _config_published() -> list[str]:
    """Distribution names flagged ``published = true``, in config order."""
    return [name for name, cfg in _packages().items() if cfg.get("published")]


def _dotted_path(pkg_name: str) -> str:
    """Dotted import path of a configured package."""
    path: str = _packages()[pkg_name]["path"]
    return path.replace("/", ".")


def _expected_surface(path: str) -> list[str]:
    """The import surface of a package path: the dotted root, base when the checkout ships base.py, and
    manifest when the checkout ships manifest.py."""
    dotted = path.replace("/", ".")
    surface = [dotted, f"{dotted}.base"] if (_REPO_ROOT / path / "base.py").exists() else [dotted]
    if (_REPO_ROOT / path / "manifest.py").exists():
        surface.append(f"{dotted}.manifest")
    return surface


def _published_children() -> list[str]:
    """Published packages nested under the data-operations path, in config order."""
    packages = _packages()
    prefix = packages[_DATA_OPERATIONS]["path"] + "/"
    return [name for name in _EXPECTED_PUBLISHED if packages[name]["path"].startswith(prefix)]


def _published_script() -> ModuleType:
    """The single-source reader the release workflow, tox and the tests all share."""
    assert _PUBLISHED_SCRIPT.exists(), (
        f"{_PUBLISHED_SCRIPT} is missing; it must expose the released set read from config/packages.toml"
    )
    return load_script("published_packages", _PUBLISHED_SCRIPT)


def _published_packages_fn() -> Callable[[dict[str, dict[str, Any]]], list[str]]:
    """The set reader that published_packages must expose."""
    reader: Callable[[dict[str, dict[str, Any]]], list[str]] | None = getattr(
        _published_script(), "published_packages", None
    )
    assert callable(reader), "published_packages.published_packages must be a callable"
    return reader


def _script_fn(path: Path, attr: str, purpose: str) -> Callable[..., Any]:
    """A callable exposed by a config-derived verify script."""
    assert path.exists(), f"{path} is missing; it must {purpose}"
    fn: Callable[..., Any] | None = getattr(load_script(path.stem, path), attr, None)
    assert callable(fn), f"{path.name} must expose a callable {attr}"
    return fn


def _import_modules(packages: dict[str, dict[str, Any]]) -> list[str]:
    """The smoke-import list verify_published_imports derives from a config."""
    modules: list[str] = _script_fn(
        _IMPORTS_SCRIPT, "import_modules", "derive one smoke import per configured package from config/packages.toml"
    )(packages)
    return modules


def _independent_distributions(packages: dict[str, dict[str, Any]]) -> list[str]:
    """The independently installed distributions verify_independent_installs derives from a config."""
    names: list[str] = _script_fn(
        _INDEPENDENT_SCRIPT,
        "independent_distributions",
        "derive the independently installed distributions from config/packages.toml",
    )(packages)
    return names


def _probe_modules(name: str, packages: dict[str, dict[str, Any]]) -> list[str]:
    """The modules verify_independent_installs probes after installing one distribution on its own."""
    modules: list[str] = _script_fn(
        _INDEPENDENT_SCRIPT,
        "probe_modules",
        "derive the import surface each independent install probes from config/packages.toml",
    )(name, packages)
    return modules


def _internal_extra_entries(packages: dict[str, dict[str, Any]]) -> list[tuple[str, str, list[str]]]:
    """The (package, extra, members) entries verify_extras derives from a config, normalized to tuples."""
    entries = _script_fn(
        _EXTRAS_SCRIPT, "internal_extra_members", "derive the internal extras to verify from config/packages.toml"
    )(packages)
    return [(package, extra, list(members)) for package, extra, members in entries]


def _verification_jobs(
    entries: list[tuple[str, str, list[str]]], version: str
) -> list[tuple[str, str, bool, list[str]]]:
    """The (owner, specifier, expect_import, members) install jobs verify_extras derives from internal extra
    entries."""
    jobs = _script_fn(
        _EXTRAS_SCRIPT, "verification_jobs", "derive the deduplicated install jobs from internal_extra_members"
    )(entries, version)
    return [(owner, specifier, expect_import, list(members)) for owner, specifier, expect_import, members in jobs]


def _cli(monkeypatch: pytest.MonkeyPatch, cwd: Path, argv: list[str]) -> Callable[[], int]:
    """The CLI entry point, wired to run from ``cwd`` with ``argv``."""
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(sys, "argv", ["published_packages.py", *argv])
    main: Callable[[], int] | None = getattr(_published_script(), "main", None)
    assert callable(main), "published_packages.main must be a callable returning an exit code"
    return main


def _cli_exit_code(monkeypatch: pytest.MonkeyPatch, cwd: Path, argv: list[str]) -> int:
    """Run the CLI and return its exit code, treating an argparse ``SystemExit`` as that code."""
    try:
        return _cli(monkeypatch, cwd, argv)()
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1


def _cli_lines(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], argv: list[str]) -> list[str]:
    """Run the CLI in-process from the repo root and return its non-empty output lines."""
    exit_code = _cli(monkeypatch, _REPO_ROOT, argv)()
    assert exit_code == 0, f"published_packages.py {' '.join(argv)} exited {exit_code!r}, expected 0"
    return [line.strip() for line in capsys.readouterr().out.splitlines() if line.strip()]


def _tox_block(env_name: str) -> str:
    """Body of a tox.ini ``[testenv:<name>]`` section."""
    match = re.search(
        rf"^\[testenv:{re.escape(env_name)}\]\n(.*?)(?=\n\[|\Z)",
        _TOX_INI.read_text(),
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"tox.ini has no [testenv:{env_name}] section"
    return match.group(1)


def _workflow_step_body(step_name: str) -> str:
    """Body of the ``run:`` block of a named step in the release workflow."""
    match = re.search(
        rf"^(?P<indent>\s*)- name: {re.escape(step_name)}\n(?P<body>.*?)(?=^(?P=indent)- name:|\Z)",
        _RELEASE_WORKFLOW.read_text(),
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f".github/workflows/release.yaml has no '- name: {step_name}' step"
    _, separator, run_block = match.group("body").partition("run: |\n")
    assert separator, f".github/workflows/release.yaml step '{step_name}' has no 'run: |' block"
    return run_block


def _workflow_build_step() -> str:
    """Body of the ``run:`` block of the release workflow's build step."""
    return _workflow_step_body(_BUILD_STEP)


def _generated(pkg_name: str, packages: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Parsed pyproject document the generator emits for a package under the given config."""
    shared = load_toml(_SHARED_CONFIG)
    content: str = gen.generate_pyproject(pkg_name, packages[pkg_name], shared, packages)
    return loads_toml(content)


def _wheel_packages(pkg_name: str, packages: dict[str, dict[str, Any]]) -> list[str]:
    """``[tool.setuptools] packages`` the generator emits for a package under the given config."""
    listed: list[str] = _generated(pkg_name, packages)["tool"]["setuptools"]["packages"]
    assert listed, (
        f"the generator discovered no modules for {pkg_name}; its paths are relative to the working "
        "directory, so run pytest from the repository root"
    )
    return listed


def _generated_data_operations() -> dict[str, Any]:
    """Parsed pyproject document the generator emits for the data-operations base package."""
    return _generated(_DATA_OPERATIONS, _packages())


def _committed_data_operations() -> dict[str, Any]:
    """Parsed committed pyproject.toml of the data-operations base package."""
    pyproject_path = _REPO_ROOT / _packages()[_DATA_OPERATIONS]["path"] / "pyproject.toml"
    assert pyproject_path.exists(), f"{pyproject_path} is missing (run scripts/generate_pyproject.py)"
    return load_toml(pyproject_path)


def _entries_under(listed: list[str], dotted: str) -> list[str]:
    """Entries of a ``[tool.setuptools] packages`` list that sit at or below a dotted path."""
    return [entry for entry in listed if entry == dotted or entry.startswith(dotted + ".")]


def _leaked_child_packages(listed: list[str]) -> list[str]:
    """Entries of a ``[tool.setuptools] packages`` list that belong to a published child package."""
    children = [_dotted_path(name) for name in _published_children()]
    return sorted({entry for child in children for entry in _entries_under(listed, child)})


# The name a simple, extras/marker-free PEP 508 dependency string starts with. "{core_dependency}" starts
# with none of these characters, so it never matches and is treated as unparseable (skipped).
_DEP_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")

# Only the lower bound matters here: ">=1.30,<2" and " >= 1.30, <2" both floor at 1.30.
_DEP_FLOOR_RE = re.compile(r">=\s*([^\s,;]+)")


def _normalize_dep_name(name: str) -> str:
    """PEP 503 normal form: lowercase, runs of '-', '_', '.' collapsed to '-'."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_dependency(dep: str) -> tuple[str, str | None] | None:
    """(normalized_name, floor_or_None) for a simple 'name>=X,<Y' dependency string, or None if it is not
    a bare name (e.g. the "{core_dependency}" placeholder, which starts with '{')."""
    match = _DEP_NAME_RE.match(dep.strip())
    if match is None:
        return None
    name = _normalize_dep_name(match.group(1))
    floor_match = _DEP_FLOOR_RE.search(dep)
    return name, (floor_match.group(1) if floor_match else None)


def _entry_point_bundles(packages: dict[str, dict[str, Any]]) -> list[str]:
    """Configured packages flagged 'entry_point_bundle = true', in config order."""
    return [name for name, cfg in packages.items() if cfg.get("entry_point_bundle") is True]


def _nested_under(bundle_name: str, packages: dict[str, dict[str, Any]]) -> list[str]:
    """Configured packages, other than the bundle itself, whose path sits under the bundle's own path."""
    prefix = packages[bundle_name]["path"] + "/"
    return [name for name in packages if name != bundle_name and packages[name]["path"].startswith(prefix)]


def _bundle_extra_only_owned_names(bundle_name: str, packages: dict[str, dict[str, Any]]) -> set[str]:
    """Nested packages a bundle owns only through a non-dev extra, not also in its own ``dependencies``."""
    cfg = packages[bundle_name]
    owned = set(gen.bundle_owned_names(cfg, packages))
    return owned - set(gen.sibling_dependency_names(cfg.get("dependencies", []), packages))


def test_config_declares_a_published_set() -> None:
    """config/packages.toml carries the released set as a per-package 'published' flag."""
    assert _config_published(), (
        "config/packages.toml declares no package with 'published = true'; that flag is the single "
        "source of truth for the distributions that ship to PyPI."
    )


def test_published_set_contains_the_bundles() -> None:
    """The four bundle wheels are always released."""
    flagged = _config_published()
    assert set(_BUNDLES) <= set(flagged), (
        f"config/packages.toml does not flag bundles {sorted(set(_BUNDLES) - set(flagged))} as 'published = true'"
    )


def test_published_flag_marks_exactly_the_released_distributions() -> None:
    """The flagged set is the release workflow array plus the five distributions it had drifted from."""
    flagged = _config_published()
    assert sorted(flagged) == sorted(_EXPECTED_PUBLISHED), (
        "config/packages.toml must flag exactly the released set with 'published = true': missing "
        f"{sorted(set(_EXPECTED_PUBLISHED) - set(flagged))}, unexpected {sorted(set(flagged) - set(_EXPECTED_PUBLISHED))}"
    )


def test_bundle_only_lists_exactly_the_unpublished_packages() -> None:
    """_BUNDLE_ONLY is the unpublished packages of config/packages.toml, in config order."""
    unpublished = [name for name, cfg in _packages().items() if not cfg.get("published")]
    assert _BUNDLE_ONLY == unpublished, (
        "_BUNDLE_ONLY must list exactly the packages without 'published = true', in config order: missing "
        f"{sorted(set(unpublished) - set(_BUNDLE_ONLY))}, unexpected {sorted(set(_BUNDLE_ONLY) - set(unpublished))}"
    )


def test_published_packages_returns_the_flagged_names_in_config_order() -> None:
    """published_packages() is the one reader of the flag; order follows config declaration order."""
    names = _published_packages_fn()(_packages())
    assert names == _EXPECTED_PUBLISHED, (
        f"published_packages() returned {names!r}, expected the flagged distributions in config "
        f"declaration order {_EXPECTED_PUBLISHED!r}"
    )


def test_published_set_is_dependency_first() -> None:
    """published_packages() itself enforces the dependency-first order (every published sibling named in a
    package's own runtime dependencies, or in a non-dev extra whose raw config entry carries the {version}
    placeholder, must appear earlier in the published order), so calling it on the real, already-ordered
    config must succeed. A bare {published_children}/'all' entry (e.g. data-operations[all], example[all])
    is unpinned, resolves at any already-published version, and creates no ordering requirement; its
    children depend on the base anyway, so they still come after it."""
    packages = _packages()

    published = _published_packages_fn()(packages)  # must not raise: config/packages.toml is already ordered

    checked_community = False
    for name in published:
        cfg = packages[name]
        raw_deps = list(cfg.get("dependencies", []))
        for extra_name, deps in cfg.get("optional_dependencies", {}).items():
            if extra_name == "dev":
                continue
            raw_deps.extend(dep for dep in deps if "{version}" in dep)

        siblings = gen.sibling_dependency_names(raw_deps, packages)
        if name == "mloda-community" and siblings:
            checked_community = True

    assert checked_community, (
        "expected mloda-community to name at least one published sibling at a pinned {version}, or this "
        "test is vacuous for the bundle ownership edges"
    )


def _synthetic_pkg(
    path: str,
    *,
    published: bool | None = True,
    dependencies: list[str] | None = None,
    optional_dependencies: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """A minimal synthetic packages.toml entry for the published-order tests."""
    cfg: dict[str, Any] = {"description": "sandbox", "path": path}
    if published is not None:
        cfg["published"] = published
    if dependencies is not None:
        cfg["dependencies"] = dependencies
    if optional_dependencies is not None:
        cfg["optional_dependencies"] = optional_dependencies
    return cfg


_ORDER_VIOLATION_CASES = [
    pytest.param(
        {
            "pkg-a": _synthetic_pkg("p/a", dependencies=["pkg-b>=1.0"]),
            "pkg-b": _synthetic_pkg("p/b"),
        },
        id="dependencies-entry",
    ),
    pytest.param(
        {
            "pkg-a": _synthetic_pkg("p/a", optional_dependencies={"otel": ["pkg-b=={version}"]}),
            "pkg-b": _synthetic_pkg("p/b"),
        },
        id="pinned-non-dev-extra-entry",
    ),
]


@pytest.mark.parametrize("packages", _ORDER_VIOLATION_CASES)
def test_published_packages_rejects_a_published_sibling_listed_later(packages: dict[str, dict[str, Any]]) -> None:
    """A published package naming a published sibling, either in 'dependencies' or in a non-dev extra entry
    carrying the {version} placeholder, that does not appear earlier in config order must raise, naming
    both packages."""
    with pytest.raises(ValueError) as exc_info:
        _published_packages_fn()(packages)

    message = str(exc_info.value)
    assert "pkg-a" in message and "pkg-b" in message, f"error message must name both pkg-a and pkg-b, got: {message}"


_ORDER_ACCEPTED_CASES = [
    pytest.param(
        {
            "base": _synthetic_pkg("p/base", optional_dependencies={"all": ["child"]}),
            "child": _synthetic_pkg("p/child"),
        },
        id="bare-extra-entry-child-after",
    ),
    pytest.param(
        {
            "base": _synthetic_pkg("p/base", optional_dependencies={"dev": ["child=={version}"]}),
            "child": _synthetic_pkg("p/child"),
        },
        id="dev-extra-entry-child-after",
    ),
    pytest.param(
        {
            "pkg-a": _synthetic_pkg("p/a", dependencies=["pkg-b>=1.0"]),
            "pkg-b": _synthetic_pkg("p/b", published=None),
        },
        id="unpublished-sibling-listed-later",
    ),
    pytest.param(
        {
            "pkg-a": _synthetic_pkg("p/a", optional_dependencies={"all": ["pkg-a[x]>={version}"]}),
        },
        id="pinned-extra-names-its-own-package",
    ),
]


@pytest.mark.parametrize("packages", _ORDER_ACCEPTED_CASES)
def test_published_packages_accepts_orderings_with_no_ordering_requirement(
    packages: dict[str, dict[str, Any]],
) -> None:
    """A bare (unpinned) extra entry, a 'dev' extra entry, a dependency on an unpublished sibling, and a
    pinned extra naming the package's own name never create an ordering requirement, whatever position the
    named package sits at."""
    _published_packages_fn()(packages)  # must not raise


def test_no_dotted_package_is_listed_by_two_published_distributions() -> None:
    """Definition-of-done: every path shipped by a published wheel has exactly one published owner."""
    packages = _packages()
    published = _published_packages_fn()(packages)

    owners: dict[str, str] = {}
    conflicts: dict[str, list[str]] = {}
    for name in published:
        for entry in _wheel_packages(name, packages):
            previous = owners.get(entry)
            if previous is not None and previous != name:
                conflicts.setdefault(entry, [previous]).append(name)
            else:
                owners[entry] = name

    assert conflicts == {}, (
        f"published distributions' generated [tool.setuptools] packages both list {conflicts}; each "
        "shipped path must have exactly one published owner"
    )


def test_bundle_owned_names_matches_the_published_nested_packages_exactly() -> None:
    """A bundle owns exactly its published nested packages. ``gen.bundle_owned_names`` and the set of
    published nested packages must match in both directions; the generator itself enforces the
    '<name>[extras]=={version}' spelling that ownership requires (see the synthetic tests in
    test_sibling_floor_placeholder.py), so generation succeeding here proves the real bundles comply."""
    packages = _packages()
    shared = load_toml(_SHARED_CONFIG)
    checked_bundles: set[str] = set()

    for bundle_name in _entry_point_bundles(packages):
        owned = set(gen.bundle_owned_names(packages[bundle_name], packages))
        published_nested = {name for name in _nested_under(bundle_name, packages) if packages[name].get("published")}
        if published_nested:
            checked_bundles.add(bundle_name)
        assert owned == published_nested, (
            f"{bundle_name} owns {sorted(owned)}, but its published nested packages are "
            f"{sorted(published_nested)}: these must match exactly"
        )
        gen.generate_pyproject(bundle_name, packages[bundle_name], shared, packages)  # must not raise

    assert "mloda-community" in checked_bundles, (
        "expected at least one published package nested under mloda-community, or this test is vacuous"
    )


def test_bundle_shipped_names_are_the_nested_packages_the_bundle_does_not_own() -> None:
    """Nested packages minus owned ones, in config order; a non-bundle yields []."""
    packages: dict[str, dict[str, Any]] = {
        "sb": {
            **_synthetic_pkg("sb", dependencies=["sb-owned=={version}"]),
            "entry_point_bundle": True,
        },
        "sb-shipped-a": _synthetic_pkg("sb/a", published=None),
        "sb-owned": _synthetic_pkg("sb/owned"),
        "sb-shipped-b": _synthetic_pkg("sb/b", published=None),
        "other": _synthetic_pkg("other"),
    }
    shipped = gen.bundle_shipped_names(packages["sb"], packages)

    assert shipped == ["sb-shipped-a", "sb-shipped-b"], f"bundle_shipped_names() returned {shipped!r}"
    assert gen.bundle_shipped_names(packages["other"], packages) == []


def test_published_packages_rejects_a_non_boolean_flag() -> None:
    """A truthiness test publishes on 'published = "false"', so a non-boolean flag must be rejected."""
    packages: dict[str, dict[str, Any]] = {
        "mloda-registry": {"description": "sandbox", "path": "mloda/registry", "published": "false"},
    }

    with pytest.raises(ValueError, match="published"):
        _published_packages_fn()(packages)


def test_published_packages_counts_only_a_true_flag() -> None:
    """'published = false' and a missing flag both keep a package out of the released set."""
    packages: dict[str, dict[str, Any]] = {
        "mloda-registry": {"description": "sandbox", "path": "mloda/registry", "published": True},
        "mloda-testing": {"description": "sandbox", "path": "mloda/testing", "published": False},
        "mloda-community": {"description": "sandbox", "path": "mloda/community"},
    }

    names = _published_packages_fn()(packages)

    assert names == ["mloda-registry"], f"published_packages() returned {names!r}, expected ['mloda-registry']"


def test_cli_prints_one_distribution_per_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The bare CLI feeds the release workflow build array."""
    lines = _cli_lines(monkeypatch, capsys, [])
    assert lines == _EXPECTED_PUBLISHED, (
        f"'python scripts/published_packages.py' printed {lines!r}, expected one distribution name per "
        f"line in config order {_EXPECTED_PUBLISHED!r}"
    )


def test_cli_pin_appends_the_version_to_every_name(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--pin`` feeds the tox install lines, which need ``name==version`` specifiers."""
    lines = _cli_lines(monkeypatch, capsys, ["--pin", "9.9.9"])
    expected = [f"{name}==9.9.9" for name in _EXPECTED_PUBLISHED]
    assert lines == expected, (
        f"'python scripts/published_packages.py --pin 9.9.9' printed {lines!r}, expected {expected!r}"
    )


def test_cli_exclude_newer_exempt_prints_one_flag_per_distribution(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--exclude-newer-exempt`` exempts the released set from the 7-day ``exclude-newer`` filter."""
    lines = _cli_lines(monkeypatch, capsys, ["--exclude-newer-exempt"])
    expected = [f"--exclude-newer-package={name}=false" for name in _EXPECTED_PUBLISHED]
    assert lines == expected, (
        f"'python scripts/published_packages.py --exclude-newer-exempt' printed {lines!r}, expected {expected!r}"
    )


def test_cli_exits_non_zero_when_nothing_is_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty set would silently publish nothing, so it must fail loudly instead."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "packages.toml").write_text(
        '[packages.mloda-registry]\ndescription = "sandbox"\npath = "mloda/registry"\n'
    )

    exit_code = _cli_exit_code(monkeypatch, tmp_path, [])

    assert exit_code != 0, "published_packages.main() must exit non-zero when no package carries 'published = true'"


def test_cli_rejects_an_empty_pin(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """tox renders '--pin ' when MLODA_REGISTRY_VERSION is unset; bare 'name==' specifiers must not reach uv."""
    exit_code = _cli_exit_code(monkeypatch, _REPO_ROOT, ["--pin", ""])
    captured = capsys.readouterr()

    assert exit_code != 0, (
        f"'python {_SCRIPT_INVOCATION} --pin ' exited {exit_code!r}; an empty version must be rejected "
        "instead of printing unusable 'name==' specifiers"
    )
    assert "==" not in captured.out, (
        f"'python {_SCRIPT_INVOCATION} --pin ' printed {captured.out.splitlines()[:3]}; an empty version "
        "must produce no specifiers at all"
    )
    assert "--pin" in captured.err, (
        f"'python {_SCRIPT_INVOCATION} --pin ' failed without naming --pin on stderr, got {captured.err!r}; "
        "the message must say which argument is empty"
    )


def test_release_workflow_has_no_hardcoded_distribution_list() -> None:
    """The build array must be generated, not typed out in any quoting style; the copy is what drifted."""
    names = sorted(set(_DISTRIBUTION_NAME_RE.findall(_workflow_build_step())))
    assert names == [], (
        f".github/workflows/release.yaml step '{_BUILD_STEP}' names {len(names)} distribution(s) itself, "
        f"starting with {names[:3]}; fill 'packages=( ... )' from 'python {_SCRIPT_INVOCATION}' instead."
    )


def test_release_workflow_builds_from_the_published_script() -> None:
    """Naming the script in a comment proves nothing: the build array itself must be filled from it."""
    assert _ARRAY_FROM_SCRIPT_RE.search(_workflow_build_step()) is not None, (
        f".github/workflows/release.yaml step '{_BUILD_STEP}' does not fill its 'packages' array from "
        f"'python {_SCRIPT_INVOCATION}', so the set it builds is a second copy of the released set."
    )


def test_release_workflow_publish_step_fills_its_list_from_the_published_script() -> None:
    """The upload order must come from the same single source as the build array, not a re-derived list."""
    body = _workflow_step_body(_PUBLISH_STEP)
    assert _SCRIPT_FILL_RE.search(body) is not None, (
        f".github/workflows/release.yaml step '{_PUBLISH_STEP}' does not fill its upload list from "
        f"'python {_SCRIPT_INVOCATION}', like the '{_BUILD_STEP}' step does."
    )


def test_release_workflow_publish_step_matches_wheels_through_published_wheels() -> None:
    """Bash glob order over dist/* puts mloda_community-* before mloda_community_extenders_shared-*,
    uploading the bundle before the package it depends on. The step must not re-implement wheel matching in
    bash (name escaping, per-package glob) either: that duplicates and can drift from
    scripts/verify_builds.py's escape_distribution_name/find_wheels, so it must match wheels through
    'python scripts/published_packages.py --wheels dist' instead."""
    body = _workflow_step_body(_PUBLISH_STEP)
    assert _DIST_STAR_GLOB_RE.search(body) is None, (
        f".github/workflows/release.yaml step '{_PUBLISH_STEP}' still iterates every file in dist/ in "
        "bash glob order instead of one wheel per published name in script order."
    )
    assert "--wheels" in body, (
        f".github/workflows/release.yaml step '{_PUBLISH_STEP}' must locate its wheels through "
        f"'python {_SCRIPT_INVOCATION} --wheels dist', not a hand-written bash glob."
    )
    assert _BASH_NAME_ESCAPE_RE.search(body) is None, (
        f".github/workflows/release.yaml step '{_PUBLISH_STEP}' must not re-implement distribution name "
        'escaping in bash ("${pkg//-/_}"); scripts/published_packages.py\'s published_wheels() (backed by '
        "verify_builds.escape_distribution_name) does the matching."
    )
    assert _DIST_NAME_GLOB_RE.search(body) is None, (
        f".github/workflows/release.yaml step '{_PUBLISH_STEP}' must not glob 'dist/<name>-*.whl' itself; "
        f"'python {_SCRIPT_INVOCATION} --wheels dist' returns the matched wheel paths directly."
    )
    for token in ("--skip-existing", "--verbose", "PYPI_UPLOAD_DELAY_SECONDS"):
        assert token in body, f".github/workflows/release.yaml step '{_PUBLISH_STEP}' must keep {token!r}"


@pytest.mark.parametrize("env_name", _TOX_DERIVED_SET_ENVS)
def test_tox_env_names_no_distribution_itself(env_name: str) -> None:
    """tox must not re-type any distribution name; every installed set is derived from the config."""
    names = sorted(set(_DISTRIBUTION_NAME_RE.findall(_tox_block(env_name))))
    assert names == [], (
        f"tox.ini [testenv:{env_name}] names {len(names)} distribution(s) by hand, starting with "
        f"{names[:3]}; derive the list from config/packages.toml through a scripts/ helper instead."
    )


@pytest.mark.parametrize("pkg_name", _EXPECTED_PUBLISHED)
def test_verify_published_imports_every_published_distribution(pkg_name: str) -> None:
    """Installing the released set proves nothing about importing it, so every distribution needs a smoke import."""
    dotted = _dotted_path(pkg_name)
    assert dotted in _import_modules(_packages()), (
        f"verify_published_imports.import_modules() never yields {dotted}, so {pkg_name} is installed "
        "from PyPI and never import-checked."
    )


@pytest.mark.parametrize("pkg_name", _BUNDLE_ONLY)
def test_verify_published_imports_every_bundle_only_package(pkg_name: str) -> None:
    """Bundle-only code has no wheel of its own, so its smoke import is what proves the bundles carry it."""
    dotted = _dotted_path(pkg_name)
    assert dotted in _import_modules(_packages()), (
        f"verify_published_imports.import_modules() never yields {dotted}, so nothing proves the bundle "
        f"wheels carry the {pkg_name} code."
    )


def test_import_modules_covers_every_configured_package_in_config_order() -> None:
    """Every configured package ships in some wheel, so its whole import surface gets a smoke import."""
    packages = _packages()
    expected = [module for cfg in packages.values() for module in _expected_surface(cfg["path"])]
    modules = _import_modules(packages)
    assert modules == expected, (
        f"import_modules() returned {modules!r}, expected the import surface (root plus base module "
        f"where <path>/base.py exists) of every configured package in config order {expected!r}"
    )


@pytest.mark.parametrize(("env_name", "script"), _TOX_VERIFY_SCRIPTS)
def test_tox_verify_env_runs_its_config_derived_script(env_name: str, script: str) -> None:
    """Each PyPI-verifying env invokes the config-reading script with an argument, not in a comment."""
    invocation = re.compile(rf"^[^\n;#]*{re.escape(script)}\s+\S", re.MULTILINE)
    assert invocation.search(_tox_block(env_name)) is not None, (
        f"tox.ini [testenv:{env_name}] does not invoke {script} with an argument on a non-comment "
        "line, so its checks are a hand-written copy of the configured package set."
    )


def test_independent_distributions_matches_the_published_set() -> None:
    """On the real config, independent_distributions() matches published_packages() exactly."""
    packages = _packages()
    names = _independent_distributions(packages)
    expected = _published_packages_fn()(packages)
    assert names == expected, f"independent_distributions() returned {names!r}, expected {expected!r}"


def test_independent_distributions_includes_a_nested_published_leaf() -> None:
    """A nested published leaf must still be probed independently: it can be released without its base changing."""
    names = _independent_distributions(_packages())
    assert "mloda-community-aggregation" in names, f"expected mloda-community-aggregation in {names!r}"


def test_independent_distributions_excludes_an_unpublished_package() -> None:
    """A package without 'published = true' (flag missing or explicitly false) ships no independent wheel."""
    packages: dict[str, dict[str, Any]] = {
        "mloda-registry": {"description": "sandbox", "path": "mloda/registry", "published": True},
        "mloda-sandbox": {"description": "sandbox", "path": "mloda/sandbox"},
        "mloda-other": {"description": "sandbox", "path": "mloda/other", "published": False},
    }

    names = _independent_distributions(packages)

    assert names == ["mloda-registry"], f"independent_distributions() returned {names!r}, expected ['mloda-registry']"


def test_every_unpublished_package_is_nested_under_an_entry_point_bundle() -> None:
    """import_modules() probes every configured package in the released venv; an unpublished package
    only reaches that venv inside a bundle wheel, so none may sit outside every bundle path."""
    packages = _packages()
    bundle_prefixes = [cfg["path"] + "/" for cfg in packages.values() if cfg.get("entry_point_bundle") is True]
    stranded = [
        name
        for name, cfg in packages.items()
        if not cfg.get("published") and not any(cfg["path"].startswith(prefix) for prefix in bundle_prefixes)
    ]
    assert stranded == [], (
        f"configured packages {stranded} are neither published nor nested under a package flagged "
        "'entry_point_bundle = true'; no wheel ships their code, so the verify-published smoke "
        "imports would fail for them."
    )


@pytest.mark.parametrize("bundle", _ENTRY_POINT_BUNDLES)
def test_probe_modules_covers_every_surface_nested_under_a_bundle(bundle: str) -> None:
    """A bare install of the bundle ships (or pulls in via 'dependencies') every nested package it does not
    own only through an extra; an extra-only owned child has no code in a bare install, so its surface is
    probed through its own distribution instead."""
    packages = _packages()
    prefix = packages[bundle]["path"] + "/"
    extra_only = _bundle_extra_only_owned_names(bundle, packages)
    nested = [name for name, cfg in packages.items() if cfg["path"].startswith(prefix) and name not in extra_only]
    assert nested, f"fixture assumption: {bundle} has configured packages nested under {prefix}"
    expected = [module for name in [bundle, *nested] for module in _expected_surface(packages[name]["path"])]

    modules = _probe_modules(bundle, packages)

    assert modules == expected, (
        f"probe_modules({bundle!r}) returned {modules!r}, expected its own import surface plus every "
        f"nested configured package's surface it does not own only through an extra, in config order {expected!r}"
    )


def test_probe_modules_for_a_top_level_leaf_is_its_own_surface() -> None:
    """A distribution with nothing nested under its path probes exactly its own import surface."""
    packages = _packages()

    modules = _probe_modules("mloda-registry", packages)

    expected = _expected_surface(packages["mloda-registry"]["path"])
    assert modules == expected, (
        f"probe_modules('mloda-registry') returned {modules!r}, expected only its own import surface {expected!r}"
    )


@pytest.mark.parametrize("base", [_DATA_OPERATIONS, _COMMUNITY_EXAMPLE])
def test_probe_modules_for_a_base_package_is_its_own_surface(base: str) -> None:
    """A base package's wheel excludes its nested packages, so probing them fails the independent install."""
    packages = _packages()
    prefix = packages[base]["path"] + "/"
    nested = [name for name, cfg in packages.items() if cfg["path"].startswith(prefix)]
    assert nested, f"fixture assumption: {base} has configured packages nested under {prefix}"
    assert packages[base].get("entry_point_bundle") is not True, (
        f"fixture assumption: {base} is not flagged 'entry_point_bundle = true'"
    )

    modules = _probe_modules(base, packages)

    expected = _expected_surface(packages[base]["path"])
    assert modules == expected, (
        f"probe_modules({base!r}) returned {modules!r}, expected only its own import surface {expected!r}; "
        f"the {base} wheel excludes its nested packages, so probing them makes the independent install fail"
    )


def test_probe_modules_adds_nested_surfaces_only_for_an_entry_point_bundle() -> None:
    """Nested surfaces join the probe only when the parent ships them, i.e. is flagged 'entry_point_bundle = true'."""
    packages: dict[str, dict[str, Any]] = {
        "mloda-sandbox": {"description": "sandbox", "path": "mloda/sandbox", "published": True},
        "mloda-sandbox-child": {"description": "sandbox", "path": "mloda/sandbox/child", "published": True},
    }

    modules = _probe_modules("mloda-sandbox", packages)
    assert modules == ["mloda.sandbox"], (
        f"probe_modules('mloda-sandbox') returned {modules!r}, expected only ['mloda.sandbox']; without "
        "'entry_point_bundle = true' the base wheel excludes its nested child"
    )

    packages["mloda-sandbox"]["entry_point_bundle"] = True
    bundled = _probe_modules("mloda-sandbox", packages)
    assert bundled == ["mloda.sandbox", "mloda.sandbox.child"], (
        f"probe_modules('mloda-sandbox') returned {bundled!r} after flagging 'entry_point_bundle = true', "
        "expected ['mloda.sandbox', 'mloda.sandbox.child']"
    )


def test_probe_modules_skips_a_nested_child_owned_only_through_an_extra() -> None:
    """A dependency-owned child is still installed by a bare bundle install and stays in the probe; a
    child owned only through an extra has no code in a bare install and must be left out."""
    packages: dict[str, dict[str, Any]] = {
        "mloda-sandbox": {
            "description": "sandbox",
            "path": "mloda/sandbox",
            "published": True,
            "entry_point_bundle": True,
            "dependencies": ["mloda-sandbox-dep-child==0.0.0"],
            "optional_dependencies": {"extra": ["mloda-sandbox-extra-child==0.0.0"]},
        },
        "mloda-sandbox-dep-child": {"description": "sandbox", "path": "mloda/sandbox/dep_child", "published": True},
        "mloda-sandbox-extra-child": {
            "description": "sandbox",
            "path": "mloda/sandbox/extra_child",
            "published": True,
        },
    }

    modules = _probe_modules("mloda-sandbox", packages)

    assert modules == ["mloda.sandbox", "mloda.sandbox.dep_child"], (
        f"probe_modules('mloda-sandbox') returned {modules!r}, expected the dependency-owned child kept and "
        "the extra-owned child left out: ['mloda.sandbox', 'mloda.sandbox.dep_child']"
    )


def test_internal_extra_members_yields_exactly_the_internal_extras() -> None:
    """The extras that pull in configured packages are the only ones verify-extras must exercise. A member
    is parsed from the requirement string (e.g. 'mloda-community-otel=={version}'), not required to be a
    bare package name."""
    entries = _internal_extra_entries(_packages())
    expected = [
        (_COMMUNITY_EXAMPLE, "all", ["mloda-community-example-a"]),
        (_DATA_OPERATIONS, "all", _published_children()),
        ("mloda-community", "otel", ["mloda-community-otel"]),
        ("mloda-community", "openlineage", ["mloda-community-openlineage"]),
        ("mloda-community", "all", ["mloda-community-otel", "mloda-community-openlineage"]),
        ("mloda-enterprise", "openlineage", ["mloda-community-openlineage"]),
        ("mloda-testing", "binary-model", ["mloda-community"]),
    ]
    assert entries == expected, f"internal_extra_members() yielded {entries!r}, expected exactly {expected!r}"


def test_internal_extra_members_always_skips_the_dev_extra() -> None:
    """The dev extra is tooling, never shipped code, so it is skipped even when it names a configured package."""
    packages: dict[str, dict[str, Any]] = {
        "mloda-registry": {
            "description": "sandbox",
            "path": "mloda/registry",
            "published": True,
            "optional_dependencies": {"dev": ["mloda-testing"], "all": ["mloda-testing"]},
        },
        "mloda-testing": {"description": "sandbox", "path": "mloda/testing", "published": True},
    }

    entries = _internal_extra_entries(packages)

    assert entries == [("mloda-registry", "all", ["mloda-testing"])], (
        f"internal_extra_members() yielded {entries!r}; the dev extra must never be verified, even when "
        "it lists a configured package."
    )


def test_internal_extra_members_drops_external_names() -> None:
    """External extras (pyarrow, pandas, ...) resolve on PyPI anyway; only internal members need the check."""
    packages = _packages()
    entries = _internal_extra_entries(packages)
    stray = sorted({member for _, _, members in entries for member in members if member not in packages})
    assert stray == [], (
        f"internal_extra_members() yielded external names {stray} as members; only configured package "
        "names can be import-checked through their dotted paths."
    )
    named = sorted({package for package, _, _ in entries})
    assert "mloda-community-aggregation" not in named, (
        "internal_extra_members() yielded mloda-community-aggregation, whose non-dev extras list only "
        "external names; a package with no internal members has nothing to verify."
    )


def test_verification_jobs_yields_one_bare_job_per_package_then_its_gated_jobs() -> None:
    """A bare install carries every extra's members, so it need only run once per owning package; each
    extra still needs its own gated job to prove it alone gates its members."""
    entries = _internal_extra_entries(_packages())
    jobs = _verification_jobs(entries, "9.9.9")
    expected = [
        (_COMMUNITY_EXAMPLE, f"{_COMMUNITY_EXAMPLE}==9.9.9", False, ["mloda-community-example-a"]),
        (_COMMUNITY_EXAMPLE, f"{_COMMUNITY_EXAMPLE}[all]==9.9.9", True, ["mloda-community-example-a"]),
        (_DATA_OPERATIONS, f"{_DATA_OPERATIONS}==9.9.9", False, _published_children()),
        (_DATA_OPERATIONS, f"{_DATA_OPERATIONS}[all]==9.9.9", True, _published_children()),
        ("mloda-community", "mloda-community==9.9.9", False, ["mloda-community-otel", "mloda-community-openlineage"]),
        ("mloda-community", "mloda-community[otel]==9.9.9", True, ["mloda-community-otel"]),
        ("mloda-community", "mloda-community[openlineage]==9.9.9", True, ["mloda-community-openlineage"]),
        (
            "mloda-community",
            "mloda-community[all]==9.9.9",
            True,
            ["mloda-community-otel", "mloda-community-openlineage"],
        ),
        ("mloda-enterprise", "mloda-enterprise==9.9.9", False, ["mloda-community-openlineage"]),
        ("mloda-enterprise", "mloda-enterprise[openlineage]==9.9.9", True, ["mloda-community-openlineage"]),
        ("mloda-testing", "mloda-testing==9.9.9", False, ["mloda-community"]),
        ("mloda-testing", "mloda-testing[binary-model]==9.9.9", True, ["mloda-community"]),
    ]
    assert jobs == expected, f"verification_jobs() yielded {jobs!r}, expected exactly {expected!r}"


def test_verification_jobs_dedupes_overlapping_extras_and_keeps_package_order() -> None:
    """Two extras of one package that share a member must not install the bare package twice; entries of
    two packages must keep the bare/gated jobs of the first package before the second's."""
    entries: list[tuple[str, str, list[str]]] = [
        ("pkg-a", "x", ["pkg-a-member-1", "pkg-a-member-2"]),
        ("pkg-a", "y", ["pkg-a-member-2", "pkg-a-member-3"]),
        ("pkg-b", "z", ["pkg-b-member-1"]),
    ]
    jobs = _verification_jobs(entries, "9.9.9")
    expected = [
        ("pkg-a", "pkg-a==9.9.9", False, ["pkg-a-member-1", "pkg-a-member-2", "pkg-a-member-3"]),
        ("pkg-a", "pkg-a[x]==9.9.9", True, ["pkg-a-member-1", "pkg-a-member-2"]),
        ("pkg-a", "pkg-a[y]==9.9.9", True, ["pkg-a-member-2", "pkg-a-member-3"]),
        ("pkg-b", "pkg-b==9.9.9", False, ["pkg-b-member-1"]),
        ("pkg-b", "pkg-b[z]==9.9.9", True, ["pkg-b-member-1"]),
    ]
    assert jobs == expected, (
        f"verification_jobs() yielded {jobs!r}, expected {expected!r}: one deduplicated bare job per package "
        "in first-appearance order, then its extras' gated jobs in entry order"
    )


def test_verification_jobs_of_an_empty_entries_list_is_empty() -> None:
    """No internal extras means no install jobs at all."""
    jobs = _verification_jobs([], "9.9.9")
    assert jobs == [], f"verification_jobs([], ...) returned {jobs!r}, expected an empty list"


_EXTERNAL_EXTRAS_FN = "external_bundle_extras"
_EXTERNAL_JOB_FN = "_install_and_probe_external"


def _external_extra_entries(
    packages: dict[str, dict[str, Any]],
) -> list[tuple[str, str, list[str], list[str]]]:
    """The (bundle, extra, external distributions, binary wheel distributions) entries verify_extras derives."""
    entries = _script_fn(
        _EXTRAS_SCRIPT, _EXTERNAL_EXTRAS_FN, "derive the third-party bundle extras to verify from config/packages.toml"
    )(packages)
    return [(bundle, extra, list(names), list(binaries)) for bundle, extra, names, binaries in entries]


def _external_extra_probe(distributions: list[str], binary_distributions: list[str]) -> str:
    """The probe program source verify_extras runs in the venv of an external-extra install."""
    source: str = _script_fn(
        _EXTRAS_SCRIPT, "external_extra_probe", "build the probe program for an external extra install"
    )(distributions, binary_distributions)
    return source


def _run_probe(
    distributions: list[str], binary_distributions: list[str], env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the probe program with the dev interpreter (optionally with an explicit environment)."""
    source = _external_extra_probe(distributions, binary_distributions)
    return subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, env=env)  # nosec


def _fake_dist(
    root: Path,
    *,
    binary_exists: bool = True,
    broken_module: bool = False,
    broken_entry_point: bool = False,
) -> dict[str, str]:
    """Install a fake ``fakebin`` distribution under ``root``; returns an env putting it on PYTHONPATH.

    It registers a ``mloda.feature_groups`` entry point to ``fake_fg`` (one class naming the ``fake_plugin``
    binary module). ``broken_module`` makes its top-level module ``fakebin_mod`` raise on import;
    ``broken_entry_point`` makes the entry point target a missing attribute."""
    info = root / "fakebin-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: fakebin\nVersion: 0.1\n")
    top_level = ["fake_fg", "fake_plugin", "fakebin_mod"]
    (info / "top_level.txt").write_text("\n".join(top_level) + "\n")
    target = "FEATURE_GROUPS_MISSING" if broken_entry_point else "FEATURE_GROUPS"
    (info / "entry_points.txt").write_text(f"[mloda.feature_groups]\nfakebin = fake_fg:{target}\n")
    (root / "fake_fg.py").write_text(
        "class FakeFG:\n"
        '    BINARY_WHEEL_DISTRIBUTION = "fakebin"\n'
        '    BINARY_PLUGIN_ID = "fake_plugin"\n\n\n'
        "FEATURE_GROUPS = [FakeFG]\n"
    )
    (root / "fakebin_mod.py").write_text("raise ImportError('fakebin_mod is broken')\n" if broken_module else "")
    binary_name = "__init__.py" if binary_exists else "absent.bin"
    plugin = root / "fake_plugin"
    plugin.mkdir()
    (plugin / "__init__.py").write_text(
        "from pathlib import Path\n\n\n"
        f"def binary_path() -> Path:\n    return Path(__file__).parent / {binary_name!r}\n"
    )
    return {**os.environ, "PYTHONPATH": str(root)}


def _fake_external_run(
    monkeypatch: pytest.MonkeyPatch, failing: Callable[[list[list[str]]], bool] | None = None, stderr: str = ""
) -> list[list[str]]:
    """Replace the script's subprocess.run with a recorder; ``failing(calls)`` is checked after each call is recorded."""
    calls: list[list[str]] = []
    module = load_script(_EXTRAS_SCRIPT.stem, _EXTRAS_SCRIPT)

    def _fake_run(command: list[str], *args: Any, **kwargs: Any) -> _FakeCompletedProcess:
        calls.append(command)
        if failing is not None and failing(calls):
            return _FakeCompletedProcess(1, stderr=stderr)
        return _FakeCompletedProcess(0)

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    return calls


def _run_external_job(tmp_path: Path, binaries: list[str]) -> tuple[list[str], list[str]]:
    """Run one external job for the enterprise anonymizer extra against the (faked) subprocess.run."""
    result: tuple[list[str], list[str]] = _script_fn(
        _EXTRAS_SCRIPT, _EXTERNAL_JOB_FN, "install one third-party extra and probe it"
    )(
        "mloda-enterprise[anonymizer]==9.9.9",
        ("mloda.enterprise",),
        ["mloda-anonymizer-binary", "pyarrow"],
        binaries,
        str(tmp_path),
    )
    return result


def test_external_bundle_extras_yields_exactly_the_third_party_bundle_extras() -> None:
    """The bundle extras that name third-party distributions are the ones never installed by the internal jobs."""
    entries = _external_extra_entries(_packages())
    expected = [
        ("mloda-enterprise", "ed25519", ["cryptography"], []),
        ("mloda-enterprise", "otel", ["opentelemetry-api"], []),
        ("mloda-enterprise", "anonymizer", ["mloda-anonymizer-binary", "pyarrow"], ["mloda-anonymizer-binary"]),
    ]
    assert entries == expected, f"external_bundle_extras() yielded {entries!r}, expected exactly {expected!r}"


def test_external_bundle_extras_synthetic_config_applies_every_rule() -> None:
    """Skips dev, configured-only extras, non-bundle and unpublished-bundle packages; keeps only the external
    names of a mixed extra; takes binary wheels only from a shipped leaf's wheel extra, not an owned leaf's."""
    packages: dict[str, dict[str, Any]] = {
        "sb": {
            "description": "sandbox",
            "path": "sb",
            "published": True,
            "entry_point_bundle": True,
            "dependencies": ["sb-owned=={version}"],
            "optional_dependencies": {
                "dev": ["pytest>=9"],
                "internal": ["sb-owned=={version}"],
                "mixed": ["  Foo_Bar>=1", "sb-owned=={version}", "bar-binary>=0.1", "ownedbin>=1"],
                "plain": ["baz>=2"],
            },
        },
        "sb-owned": {
            "description": "sandbox",
            "path": "sb/owned",
            "published": True,
            "optional_dependencies": {"wheel": ["ownedbin>=1"]},
        },
        "sb-shipped": {
            "description": "sandbox",
            "path": "sb/shipped",
            "entry_point_groups": ["mloda.feature_groups"],
            "optional_dependencies": {"wheel": ["bar-binary>=0.1,<0.2"]},
        },
        "plain-pkg": {
            "description": "sandbox",
            "path": "plain",
            "published": True,
            "optional_dependencies": {"x": ["notbundle>=1"]},
        },
        "hidden": {
            "description": "sandbox",
            "path": "hidden",
            "entry_point_bundle": True,
            "optional_dependencies": {"x": ["hiddenext>=1"]},
        },
    }

    entries = _external_extra_entries(packages)

    expected = [
        ("sb", "mixed", ["Foo_Bar", "bar-binary", "ownedbin"], ["bar-binary"]),
        ("sb", "plain", ["baz"], []),
    ]
    assert entries == expected, f"external_bundle_extras() yielded {entries!r}, expected exactly {expected!r}"


def test_external_job_runs_venv_install_owner_import_then_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The external job creates a fresh venv, installs the extra, imports the owner surface, then runs the probe."""
    calls = _fake_external_run(monkeypatch)

    messages, errors = _run_external_job(tmp_path, ["mloda-anonymizer-binary"])

    assert errors == [], f"_install_and_probe_external() reported {errors!r} for an all-success fake"
    assert len(calls) == 4, f"expected venv, install, owner import and probe (4 commands), got {calls!r}"
    assert calls[0][:2] == ["uv", "venv"], f"first command must create the venv, got {calls[0]!r}"
    assert calls[1][:3] == ["uv", "pip", "install"], f"second command must install, got {calls[1]!r}"
    assert calls[1][-1] == "mloda-enterprise[anonymizer]==9.9.9", f"install must end with the specifier: {calls[1]!r}"
    assert calls[2][1:] == ["-c", "import mloda.enterprise"], f"third command must import the owner: {calls[2]!r}"
    probe = _external_extra_probe(["mloda-anonymizer-binary", "pyarrow"], ["mloda-anonymizer-binary"])
    assert calls[3][1] == "-c" and calls[3][2] == probe, f"last command must run the probe program: {calls[3]!r}"
    assert calls[2][0] == calls[3][0], "owner import and probe must use the same venv python"


def test_external_job_stops_when_the_install_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A failed install is reported against the specifier and nothing runs after it."""
    calls = _fake_external_run(
        monkeypatch, failing=lambda calls: calls[-1][:3] == ["uv", "pip", "install"], stderr="boom"
    )

    messages, errors = _run_external_job(tmp_path, [])

    assert len(calls) == 2, f"commands after the failed install must not run, got {calls!r}"
    assert len(errors) == 1 and "mloda-enterprise[anonymizer]==9.9.9" in errors[0] and "boom" in errors[0], errors


def test_external_job_reports_a_failing_probe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A probe that exits non-zero becomes an error carrying the specifier and the probe's message."""
    _fake_external_run(monkeypatch, failing=lambda calls: len(calls) == 4, stderr="missing-dist")

    messages, errors = _run_external_job(tmp_path, ["mloda-anonymizer-binary"])

    assert len(errors) == 1 and "mloda-enterprise[anonymizer]==9.9.9" in errors[0] and "missing-dist" in errors[0], (
        errors
    )


def test_external_probe_fails_naming_a_missing_binary_wheel() -> None:
    """A binary wheel that is not installed makes the probe exit non-zero and name it."""
    missing = "nonexistent-dist-binary"
    result = _run_probe(["pyarrow"], [missing])

    assert result.returncode != 0, f"probe exited 0 although {missing} is not installed"
    assert missing in result.stderr, f"probe stderr must name the distribution: {result.stderr!r}"


def test_external_probe_passes_for_installed_distributions_without_a_binary() -> None:
    """An installed external distribution with no binary wheel needs only the version lookup."""
    result = _run_probe(["pyarrow"], [])

    assert result.returncode == 0, f"probe exited {result.returncode} for installed pyarrow: {result.stderr!r}"


def test_external_probe_passes_for_a_fake_distribution_whose_binary_exists(tmp_path: Path) -> None:
    """Already passes today: the binary check is unchanged in meaning."""
    result = _run_probe(["fakebin"], ["fakebin"], _fake_dist(tmp_path))

    assert result.returncode == 0, f"probe exited {result.returncode}: {result.stderr!r}"


def test_external_probe_fails_naming_a_fake_distribution_with_a_missing_binary(tmp_path: Path) -> None:
    """Already passes today (binary_path().exists() is False -> exit naming the distribution)."""
    result = _run_probe(["fakebin"], ["fakebin"], _fake_dist(tmp_path, binary_exists=False))

    assert result.returncode != 0 and "fakebin" in result.stderr, result.stderr


def test_external_probe_fails_naming_a_distribution_whose_module_does_not_import(tmp_path: Path) -> None:
    """The probe imports every top-level module of each distribution, not just its metadata."""
    result = _run_probe(["fakebin"], [], _fake_dist(tmp_path, broken_module=True))

    assert result.returncode != 0, "probe exited 0 although a top-level module of fakebin fails to import"
    assert "fakebin" in result.stderr, f"probe stderr must name the distribution: {result.stderr!r}"
    assert "Traceback" not in result.stderr, f"failure must be a clean message, not a traceback: {result.stderr!r}"


def test_external_probe_fails_naming_an_entry_point_that_does_not_load(tmp_path: Path) -> None:
    """Every entry point of the mloda groups is loaded; a failing load names the entry point."""
    result = _run_probe(["fakebin"], [], _fake_dist(tmp_path, broken_entry_point=True))

    assert result.returncode != 0, "probe exited 0 although the fakebin entry point fails to load"
    assert "fakebin" in result.stderr, f"probe stderr must name the entry point: {result.stderr!r}"
    assert "Traceback" not in result.stderr, f"failure must be a clean message, not a traceback: {result.stderr!r}"


def test_external_probe_fails_for_a_distribution_mapping_to_no_module(tmp_path: Path) -> None:
    """A distribution that is installed but owns no top-level module is reported by name."""
    env = _fake_dist(tmp_path)
    (tmp_path / "fakebin-0.1.dist-info" / "top_level.txt").write_text("")
    result = _run_probe(["fakebin"], [], env)

    assert result.returncode != 0 and "fakebin" in result.stderr, result.stderr


@pytest.mark.parametrize("env_name", _TOX_PUBLISHED_ENVS)
def test_tox_env_installs_from_the_published_script(env_name: str) -> None:
    """Both PyPI-installing envs read the released set, and its exclude-newer exemption, from the single source."""
    install_lines = [
        line for line in _tox_block(env_name).splitlines() if "uv pip install" in line and _SCRIPT_INVOCATION in line
    ]
    assert len(install_lines) == 1, (
        f"tox.ini [testenv:{env_name}] must have exactly one 'uv pip install' line invoking "
        f"{_SCRIPT_INVOCATION}, found {len(install_lines)}: {install_lines!r}"
    )
    install_line = install_lines[0]
    assert re.search(rf"{re.escape(_SCRIPT_INVOCATION)}\s+--pin\b", install_line) is not None, (
        f"tox.ini [testenv:{env_name}] 'uv pip install' line does not invoke {_SCRIPT_INVOCATION} with "
        f"--pin: {install_line!r}"
    )
    assert re.search(rf"{re.escape(_SCRIPT_INVOCATION)}\s+--exclude-newer-exempt\b", install_line) is not None, (
        f"tox.ini [testenv:{env_name}] 'uv pip install' line does not invoke {_SCRIPT_INVOCATION} with "
        f"--exclude-newer-exempt, so 'exclude-newer' filters out our own freshly published "
        f"distributions: {install_line!r}"
    )


@pytest.mark.parametrize("base", [_DATA_OPERATIONS, _COMMUNITY_EXAMPLE])
def test_base_package_extra_uses_the_published_children_placeholder(base: str) -> None:
    """The 'all' extra is derived from the flag, so an unpublished package can never enter it."""
    extra = _packages()[base].get("optional_dependencies", {}).get("all")
    assert extra == [_PUBLISHED_CHILDREN], (
        f"config/packages.toml must declare the {base} 'all' extra as ['{_PUBLISHED_CHILDREN}'], got {extra!r}"
    )


def test_no_published_package_requires_an_unpublished_sibling() -> None:
    """A published package's dependencies or a non-dev extra may never name an unpublished configured
    sibling; the 'dev' extra is exempt (tooling only, never ships)."""
    packages = _packages()
    published = set(_config_published())
    offending: list[str] = []
    checked = False

    for name in published:
        cfg = packages[name]
        expanded = gen.expand_published_children(cfg, packages)
        raw = list(cfg.get("dependencies", []))
        for extra_name, deps in expanded.items():
            if extra_name != "dev":
                raw.extend(deps)
        siblings = gen.sibling_dependency_names(raw, packages)
        checked = True
        offending.extend(f"{name} -> {sibling}" for sibling in siblings if sibling not in published)

    assert checked, "expected at least one published package, or this test is vacuous"
    assert offending == [], (
        f"published packages must never require an unpublished sibling outside the 'dev' extra: {offending}"
    )


def test_generator_expands_published_children() -> None:
    """The placeholder expands to the published packages under the package path, in config order."""
    expected = _published_children()
    extra: list[str] = _generated_data_operations()["project"]["optional-dependencies"]["all"]
    assert extra == expected, (
        f"the generator emitted 'all' = {extra!r} for {_DATA_OPERATIONS}, expected the published "
        f"packages nested under its path in config order {expected!r}"
    )


def test_generator_keeps_published_children_out_of_the_base_wheel() -> None:
    """The placeholder must expand before exclude_paths, or every child leaks into the base wheel."""
    listed: list[str] = _generated_data_operations()["tool"]["setuptools"]["packages"]
    leaked = _leaked_child_packages(listed)
    assert leaked == [], (
        f"the generated {_DATA_OPERATIONS} wheel would ship child packages {leaked}; expand "
        f'"{_PUBLISHED_CHILDREN}" before the exclude_paths are computed from the extras.'
    )


def test_unpublishing_a_child_keeps_it_out_of_the_base_wheel() -> None:
    """'published' governs the released set, never wheel contents: a nested package stays its own wheel."""
    packages = deepcopy(_packages())
    assert packages[_UNPUBLISH_PROBE].pop("published", None) is not None, (
        f"fixture assumption: {_UNPUBLISH_PROBE} carries the published flag"
    )

    leaked = _entries_under(_wheel_packages(_DATA_OPERATIONS, packages), _dotted_path(_UNPUBLISH_PROBE))

    assert leaked == [], (
        f"dropping 'published' from {_UNPUBLISH_PROBE} absorbed {leaked} into the {_DATA_OPERATIONS} "
        "wheel; derive the wheel exclusions from the configured package layout, not from the expanded "
        f'"{_PUBLISHED_CHILDREN}" extra, or the same modules ship in two distributions.'
    )


def test_shrinking_an_extra_keeps_a_configured_child_out_of_the_base_wheel() -> None:
    """A configured child's wheel boundary comes from the layout, not from any extra: emptying the example
    base's 'all' extra must not pull example-a into the base wheel."""
    packages = deepcopy(_packages())
    example_a = "mloda-community-example-a"
    packages[_COMMUNITY_EXAMPLE]["optional_dependencies"]["all"] = []

    listed = _wheel_packages(_COMMUNITY_EXAMPLE, packages)

    assert _entries_under(listed, _dotted_path(example_a)) == [], (
        f"the {_COMMUNITY_EXAMPLE} wheel must not ship {example_a} after the 'all' extra is emptied"
    )


def _bundle_dependency_names(bundle: str, packages: dict[str, dict[str, Any]]) -> set[str]:
    """Configured packages (normalized name to config name) the bundle names in its own ``dependencies``."""
    configured = {_normalize_dep_name(name): name for name in packages}
    named = set()
    for dep in packages[bundle].get("dependencies", []):
        parsed = _parse_dependency(dep)
        if parsed is not None and parsed[0] in configured:
            named.add(configured[parsed[0]])
    return named


@pytest.mark.parametrize("bundle", _ENTRY_POINT_BUNDLES)
def test_bundle_wheel_still_ships_every_nested_package(bundle: str) -> None:
    """Bundles ship all nested code, published or not, except a nested package the bundle owns through its
    own dependencies or a non-dev extra."""
    packages = _packages()
    prefix = packages[bundle]["path"] + "/"
    owned = set(gen.bundle_owned_names(packages[bundle], packages))
    nested = {
        name: cfg["path"].replace("/", ".")
        for name, cfg in packages.items()
        if cfg["path"].startswith(prefix) and name not in owned
    }
    assert nested, f"fixture assumption: {bundle} has configured packages nested under {prefix}"

    listed = _wheel_packages(bundle, packages)

    missing = sorted(name for name, dotted in nested.items() if dotted not in listed)
    assert missing == [], (
        f"the generated {bundle} wheel no longer ships nested packages {missing}; a package flagged "
        "'entry_point_bundle = true' must keep including every nested module it does not own."
    )


_SHARED_EXTENDERS = "mloda-community-extenders-shared"
_COMMUNITY_BUNDLE = "mloda-community"
_ENTERPRISE_BUNDLE = "mloda-enterprise"


def test_community_bundle_wheel_leaves_the_shared_extenders_to_their_own_wheel() -> None:
    """mloda-community depends on the shared package, so only the shared wheel ships mloda.community.extenders.shared."""
    packages = _packages()
    shared_dotted = _dotted_path(_SHARED_EXTENDERS)

    bundle_entries = _entries_under(_wheel_packages(_COMMUNITY_BUNDLE, packages), shared_dotted)
    shared_entries = _entries_under(_wheel_packages(_SHARED_EXTENDERS, packages), shared_dotted)

    assert bundle_entries == [], f"the {_COMMUNITY_BUNDLE} wheel must not ship {shared_dotted}, lists {bundle_entries}"
    assert shared_dotted in shared_entries, f"the {_SHARED_EXTENDERS} wheel must ship {shared_dotted}"


@pytest.mark.parametrize("bundle", _ENTRY_POINT_BUNDLES)
def test_bundle_dependencies_on_nested_packages_are_published_and_absent_from_the_bundle_wheel(bundle: str) -> None:
    """Every nested package a bundle owns (dependencies or a non-dev extra) owns its own files: published,
    not in the bundle wheel."""
    packages = _packages()
    nested_owned = sorted(gen.bundle_owned_names(packages[bundle], packages))
    listed = set(_wheel_packages(bundle, packages))

    unpublished = [name for name in nested_owned if packages[name].get("published") is not True]
    # The bundle excludes exactly the owned package's own wheel packages, not everything nested
    # under it, so an unowned package nested under an owned one (still shipped by the bundle) must not
    # be flagged here.
    leaked = {name: sorted(listed & set(_wheel_packages(name, packages))) for name in nested_owned}
    leaked = {name: entries for name, entries in leaked.items() if entries}

    assert unpublished == [], f"{bundle} owns nested packages that are not published: {unpublished}"
    assert leaked == {}, f"the {bundle} wheel ships nested packages it owns: {leaked}"


def test_community_bundle_depends_on_the_nested_shared_extenders() -> None:
    """Guard the two tests above against passing vacuously: the community bundle names the nested shared package."""
    packages = _packages()

    assert _SHARED_EXTENDERS in _nested_under(_COMMUNITY_BUNDLE, packages)
    assert _SHARED_EXTENDERS in _bundle_dependency_names(_COMMUNITY_BUNDLE, packages), (
        f"{_COMMUNITY_BUNDLE} must list {_SHARED_EXTENDERS}=={{version}} in its own dependencies"
    )


def test_enterprise_bundle_wheel_is_unchanged_by_its_shared_extenders_dependency() -> None:
    """The shared package is not nested under mloda/enterprise, so the enterprise wheel never shipped it."""
    packages = _packages()
    assert _SHARED_EXTENDERS not in _nested_under(_ENTERPRISE_BUNDLE, packages)

    listed = _wheel_packages(_ENTERPRISE_BUNDLE, packages)
    nested = [_dotted_path(name) for name in _nested_under(_ENTERPRISE_BUNDLE, packages)]

    assert _entries_under(listed, _dotted_path(_SHARED_EXTENDERS)) == []
    assert all(dotted in listed for dotted in nested), f"the enterprise wheel must ship every nested package {nested}"


def test_bundle_declares_every_nested_leaf_external_runtime_dependency() -> None:
    """An entry_point_bundle wheel ships a nested, unowned package's code without inheriting its
    pyproject.toml's ``dependencies``: nothing else installs such a leaf's real (non-mloda,
    non-internal-registry) runtime dependency for it. So every such external dependency an unowned
    nested package declares must also appear, at an equal-or-higher floor, in its bundle's OWN
    ``dependencies`` or in one of the bundle's OWN non-dev ``optional_dependencies`` extras. An owned
    nested package is skipped: its own metadata installs its own dependencies instead. This guards the
    invariant for the FUTURE: nothing else stops a new unowned bundled leaf with an external runtime
    dependency from being added without covering it in the bundle again. Likewise for a sibling outside
    the bundle: the bundle must list it too."""
    packages = _packages()
    core_placeholder = "{core_dependency}"

    for bundle_name in _entry_point_bundles(packages):
        bundle_cfg = packages[bundle_name]
        bundle_dep_sources = list(bundle_cfg.get("dependencies", []))
        for extra_name, extra_deps in bundle_cfg.get("optional_dependencies", {}).items():
            if extra_name == "dev":
                continue
            bundle_dep_sources.extend(extra_deps)

        bundle_floors: dict[str, str | None] = {}
        bundle_siblings: set[str] = set()
        for dep in bundle_dep_sources:
            if dep.strip() == core_placeholder:
                continue
            parsed = _parse_dependency(dep)
            if parsed is None:
                continue
            name, floor = parsed
            if name in packages:
                bundle_siblings.add(name)
            if name in packages or name == "mloda":
                continue  # internal-registry dependency, or the core dependency's own expansion
            if name not in bundle_floors:
                bundle_floors[name] = floor
            elif floor is not None:
                existing = bundle_floors[name]
                if existing is None or version_tuple(floor) > version_tuple(existing):
                    bundle_floors[name] = floor

        nested_names = _nested_under(bundle_name, packages)
        owned_names = set(gen.bundle_owned_names(packages[bundle_name], packages))
        for nested_name in nested_names:
            if nested_name in owned_names:
                continue  # the owned package's own metadata installs its own dependencies
            for dep in packages[nested_name].get("dependencies", []):
                if dep.strip() == core_placeholder:
                    continue
                parsed = _parse_dependency(dep)
                if parsed is None:
                    continue
                name, floor = parsed
                if name in packages and name != bundle_name and name not in nested_names:
                    # A sibling outside the bundle: the "{version}" floor reads the same on both sides.
                    assert name in bundle_siblings, (
                        f"{nested_name} declares sibling dependency {dep!r}, outside {bundle_name}, but "
                        f"{bundle_name} does not list {name!r} in its own 'dependencies' or in one of its extras; "
                        f"{bundle_name} ships {nested_name}'s code without its pyproject.toml, so nothing else "
                        f"installs {name!r} for it."
                    )
                if name in packages or name == "mloda":
                    continue  # internal-registry dependency, or the core dependency's own expansion

                assert name in bundle_floors, (
                    f"{nested_name} declares external dependency {dep!r}, but its bundle {bundle_name} "
                    f"does not declare {name!r} in its own 'dependencies' or in one of its extras; "
                    f"{bundle_name} ships {nested_name}'s code without inheriting its pyproject.toml, "
                    f"so nothing else installs {name!r} for it."
                )

                bundle_floor = bundle_floors[name]
                if floor is not None:
                    assert bundle_floor is not None and version_tuple(bundle_floor) >= version_tuple(floor), (
                        f"{nested_name} declares {dep!r} (floor {floor!r}), but {bundle_name} declares "
                        f"{name!r} at floor {bundle_floor!r} in its 'dependencies' or extras, which is "
                        f"lower; bump {bundle_name}'s dependency floor to at least {floor!r}."
                    )


def test_committed_data_operations_extra_lists_the_published_children() -> None:
    """``tox -e check-generated`` runs only in the package-integrity workflow, not in this gate."""
    expected = _published_children()
    extra: list[str] = _committed_data_operations()["project"]["optional-dependencies"]["all"]
    assert extra == expected, (
        f"the committed {_DATA_OPERATIONS} pyproject.toml declares 'all' = {extra!r}, expected "
        f"{expected!r} (run scripts/generate_pyproject.py)"
    )


def test_committed_base_wheel_excludes_the_published_children() -> None:
    """The committed base wheel must not carry the code of its child distributions."""
    listed: list[str] = _committed_data_operations()["tool"]["setuptools"]["packages"]
    leaked = _leaked_child_packages(listed)
    assert leaked == [], (
        f"the committed {_DATA_OPERATIONS} pyproject.toml lists child packages {leaked} in "
        "[tool.setuptools] packages (run scripts/generate_pyproject.py)"
    )

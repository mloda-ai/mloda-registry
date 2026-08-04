"""Tests that the set of distributions published to PyPI is declared in a single source of truth.

The released set must live in exactly ONE place: a ``published = true`` flag per
package in ``config/packages.toml``. Everywhere else it must be derived, never
re-typed:

* ``scripts/published_packages.py`` reads the flag and prints the set, plain or
  pinned to a version.
* ``.github/workflows/release.yaml`` fills its ``packages=( ... )`` build array
  from that script instead of a hand-written list.
* ``tox.ini`` builds the ``verify-published`` and ``security`` install lists from
  the same script.
* ``config/packages.toml`` declares the data-operations ``all`` extra as the
  ``"{published_children}"`` placeholder, which ``scripts/generate_pyproject.py``
  expands to the published packages nested under that path.

These tests encode that contract. Re-typed copies are how five distributions
reached three of the copies but never the build array, so
``pip install mloda-community-data-operations[all]`` could not resolve.

The flag governs the RELEASE SET only. Wheel boundaries come from the configured
package layout: every configured package nested under another package's path stays
out of that package's wheel, published or not. The ``entry_point_bundle`` packages
are the deliberate exception and ship all nested code. Deriving the wheel exclusions
from the expanded extra instead couples the two, so dropping ``published`` from a
plugin silently absorbs its modules into the base wheel and ships them twice.

The remaining tests keep the derivation honest: the "no hand-written copy" guards
must catch bare and single-quoted names as well as pinned ones and must look at the
build array itself rather than at a comment naming the script, and every published
distribution must be imported by ``tox -e verify-published``, or a newly released
plugin installs from PyPI without anyone ever importing it.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

import pytest

from tests.script_loader import load_script

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHARED_CONFIG = _REPO_ROOT / "config" / "shared.toml"
_PACKAGES_CONFIG = _REPO_ROOT / "config" / "packages.toml"
_GEN_PATH = _REPO_ROOT / "scripts" / "generate_pyproject.py"
_PUBLISHED_SCRIPT = _REPO_ROOT / "scripts" / "published_packages.py"
_RELEASE_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "release.yaml"
_TOX_INI = _REPO_ROOT / "tox.ini"

# The bundle distributions, always part of the released set.
_BUNDLES = ["mloda-registry", "mloda-testing", "mloda-community", "mloda-enterprise"]

# The released set, in config declaration order: the 4 bundles, the 2 examples kept for end-to-end
# PyPI dependency-resolution coverage, and the data-operations base plus its 17 plugin packages.
_EXPECTED_PUBLISHED = [
    *_BUNDLES,
    "mloda-community-example",
    "mloda-community-example-a",
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
]

# Example/demo packages that reach users only inside the community and enterprise bundle wheels.
_BUNDLE_ONLY = [
    "mloda-enterprise-example",
    "mloda-community-example-b",
    "mloda-community-compute-frameworks-example",
    "mloda-community-extenders-example",
    "mloda-enterprise-compute-frameworks-example",
    "mloda-enterprise-extenders-example",
]

_DATA_OPERATIONS = "mloda-community-data-operations"

_COMMUNITY_EXAMPLE = "mloda-community-example"

_EXAMPLE_B = "mloda-community-example-b"

# The published child whose flag the wheel-boundary tests drop to prove 'published' does not
# control wheel contents.
_UNPUBLISH_PROBE = "mloda-community-ema"

# The packages flagged ``entry_point_bundle = true``: the only wheels that ship nested code.
_ENTRY_POINT_BUNDLES = ["mloda-community", "mloda-enterprise"]

# The placeholder config/packages.toml uses instead of listing the nested published packages.
_PUBLISHED_CHILDREN = "{published_children}"

# The tox envs that install the released set from PyPI.
_TOX_PUBLISHED_ENVS = ["verify-published", "security"]

# The tox env that import-checks every installed distribution.
_VERIFY_PUBLISHED_ENV = "verify-published"

_SCRIPT_INVOCATION = "scripts/published_packages.py"

# The release workflow step that builds the wheels.
_BUILD_STEP = "Build packages"

# A distribution name written out by hand, quoted or bare, e.g. ``"mloda-community-ffill"``. The
# leading boundary keeps the ``/tmp/mloda-verify*`` venv paths and the dotted ``mloda.community....``
# import paths of the same tox blocks out.
_DISTRIBUTION_NAME_RE = re.compile(r"(?<![\w/-])mloda-(?:registry|testing|community|enterprise)[a-z0-9-]*")

# The build array filled from the script on one line, e.g.
# ``mapfile -t packages < <(python scripts/published_packages.py)``. A comment cannot match.
_ARRAY_FROM_SCRIPT_RE = re.compile(rf"^[^\n#]*\bpackages\b[^\n#]*{re.escape(_SCRIPT_INVOCATION)}", re.MULTILINE)

gen = load_script("generate_pyproject", _GEN_PATH)


def _load_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _packages() -> dict[str, dict[str, Any]]:
    """Config-declared packages, in config order."""
    packages: dict[str, dict[str, Any]] = _load_toml(_PACKAGES_CONFIG)["packages"]
    return packages


def _config_published() -> list[str]:
    """Distribution names flagged ``published = true``, in config order."""
    return [name for name, cfg in _packages().items() if cfg.get("published")]


def _dotted_path(pkg_name: str) -> str:
    """Dotted import path of a configured package."""
    path: str = _packages()[pkg_name]["path"]
    return path.replace("/", ".")


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


def _workflow_build_step() -> str:
    """Body of the ``run:`` block of the release workflow's build step."""
    match = re.search(
        rf"^(?P<indent>\s*)- name: {re.escape(_BUILD_STEP)}\n(?P<body>.*?)(?=^(?P=indent)- name:|\Z)",
        _RELEASE_WORKFLOW.read_text(),
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f".github/workflows/release.yaml has no '- name: {_BUILD_STEP}' step"
    _, separator, run_block = match.group("body").partition("run: |\n")
    assert separator, f".github/workflows/release.yaml step '{_BUILD_STEP}' has no 'run: |' block"
    return run_block


def _generated(pkg_name: str, packages: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Parsed pyproject document the generator emits for a package under the given config."""
    shared = _load_toml(_SHARED_CONFIG)
    content: str = gen.generate_pyproject(pkg_name, packages[pkg_name], shared, packages)
    return tomllib.loads(content)


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
    return _load_toml(pyproject_path)


def _entries_under(listed: list[str], dotted: str) -> list[str]:
    """Entries of a ``[tool.setuptools] packages`` list that sit at or below a dotted path."""
    return [entry for entry in listed if entry == dotted or entry.startswith(dotted + ".")]


def _leaked_child_packages(listed: list[str]) -> list[str]:
    """Entries of a ``[tool.setuptools] packages`` list that belong to a published child package."""
    children = [_dotted_path(name) for name in _published_children()]
    return sorted({entry for child in children for entry in _entries_under(listed, child)})


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


def test_bundle_only_packages_are_not_published() -> None:
    """Example and demo packages ship inside the bundle wheels, never as standalone distributions."""
    packages = _packages()
    missing = [name for name in _BUNDLE_ONLY if name not in packages]
    assert missing == [], f"config/packages.toml no longer declares bundle-only packages {missing}"
    flagged = [name for name in _BUNDLE_ONLY if packages[name].get("published")]
    assert flagged == [], (
        f"config/packages.toml flags bundle-only packages {flagged} as 'published = true'; their code "
        "reaches users through the mloda-community / mloda-enterprise wheels."
    )


def test_published_packages_returns_the_flagged_names_in_config_order() -> None:
    """published_packages() is the one reader of the flag; order follows config declaration order."""
    names = _published_packages_fn()(_packages())
    assert names == _EXPECTED_PUBLISHED, (
        f"published_packages() returned {names!r}, expected the flagged distributions in config "
        f"declaration order {_EXPECTED_PUBLISHED!r}"
    )


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


@pytest.mark.parametrize("env_name", _TOX_PUBLISHED_ENVS)
def test_tox_env_names_no_distribution_itself(env_name: str) -> None:
    """tox must not re-type the released set to install it, pinned or unpinned."""
    names = sorted(set(_DISTRIBUTION_NAME_RE.findall(_tox_block(env_name))))
    assert names == [], (
        f"tox.ini [testenv:{env_name}] names {len(names)} distribution(s) by hand, starting with "
        f"{names[:3]}; build the list from 'python {_SCRIPT_INVOCATION} --pin ...' instead."
    )


@pytest.mark.parametrize("pkg_name", _EXPECTED_PUBLISHED)
def test_verify_published_imports_every_published_distribution(pkg_name: str) -> None:
    """Installing the released set proves nothing about importing it, so every distribution needs a smoke import."""
    dotted = _dotted_path(pkg_name)
    assert dotted in _tox_block(_VERIFY_PUBLISHED_ENV), (
        f"tox.ini [testenv:{_VERIFY_PUBLISHED_ENV}] never imports {dotted}, so {pkg_name} is installed "
        "from PyPI and never import-checked; add a smoke import line for it."
    )


@pytest.mark.parametrize("env_name", _TOX_PUBLISHED_ENVS)
def test_tox_env_installs_from_the_published_script(env_name: str) -> None:
    """Both PyPI-installing envs read the released set from the single source."""
    assert _SCRIPT_INVOCATION in _tox_block(env_name), (
        f"tox.ini [testenv:{env_name}] does not invoke {_SCRIPT_INVOCATION}, so its package list is a "
        "second copy of the released set."
    )


def test_data_operations_extra_uses_the_published_children_placeholder() -> None:
    """The 'all' extra is derived from the flag, so an unpublished package can never enter it."""
    extra = _packages()[_DATA_OPERATIONS].get("optional_dependencies", {}).get("all")
    assert extra == [_PUBLISHED_CHILDREN], (
        f"config/packages.toml must declare the {_DATA_OPERATIONS} 'all' extra as "
        f'["{_PUBLISHED_CHILDREN}"], got {extra!r}'
    )


def test_community_example_extra_keeps_the_unpublished_example_b() -> None:
    """example-b is unpublished but still resolvable at its last version, and tox -e verify-extras installs it."""
    packages = _packages()
    extra = packages[_COMMUNITY_EXAMPLE].get("optional_dependencies", {}).get("all", [])
    assert _EXAMPLE_B in extra, (
        f"config/packages.toml dropped {_EXAMPLE_B} from the {_COMMUNITY_EXAMPLE} 'all' extra, got "
        f"{extra!r}; tox -e verify-extras installs that extra and imports example_b from it."
    )
    assert not packages[_EXAMPLE_B].get("published"), (
        f"{_EXAMPLE_B} is flagged published; this test pins that an unpublished package may stay in a "
        "hand-written extra, so the extra must not be converted to the published-children placeholder."
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
    """Same coupling through a hand-written extra: the example base must not absorb example-b either."""
    packages = deepcopy(_packages())
    extra: list[str] = packages[_COMMUNITY_EXAMPLE]["optional_dependencies"]["all"]
    assert _EXAMPLE_B in extra, f"fixture assumption: the {_COMMUNITY_EXAMPLE} 'all' extra lists {_EXAMPLE_B}"
    packages[_COMMUNITY_EXAMPLE]["optional_dependencies"]["all"] = [dep for dep in extra if dep != _EXAMPLE_B]

    leaked = _entries_under(_wheel_packages(_COMMUNITY_EXAMPLE, packages), _dotted_path(_EXAMPLE_B))

    assert leaked == [], (
        f"dropping {_EXAMPLE_B} from the {_COMMUNITY_EXAMPLE} 'all' extra absorbed {leaked} into the "
        f"{_COMMUNITY_EXAMPLE} wheel; a configured package nested under another package's path belongs "
        "to its own wheel whatever the extras say."
    )


@pytest.mark.parametrize("bundle", _ENTRY_POINT_BUNDLES)
def test_bundle_wheel_still_ships_every_nested_package(bundle: str) -> None:
    """Bundles are the deliberate exception to the layout rule: they ship all nested code, published or not."""
    packages = _packages()
    prefix = packages[bundle]["path"] + "/"
    nested = {name: cfg["path"].replace("/", ".") for name, cfg in packages.items() if cfg["path"].startswith(prefix)}
    assert nested, f"fixture assumption: {bundle} has configured packages nested under {prefix}"

    listed = _wheel_packages(bundle, packages)

    missing = sorted(name for name, dotted in nested.items() if dotted not in listed)
    assert missing == [], (
        f"the generated {bundle} wheel no longer ships nested packages {missing}; a package flagged "
        "'entry_point_bundle = true' must keep including every nested module."
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

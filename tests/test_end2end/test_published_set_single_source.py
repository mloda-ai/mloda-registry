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

These tests encode that contract. They must fail while the released set is still
copied across release.yaml, tox.ini and the data-operations extra: that drift is
how five distributions reached three of the copies but never the build array, so
``pip install mloda-community-data-operations[all]`` cannot resolve.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
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

# The placeholder config/packages.toml uses instead of listing the nested published packages.
_PUBLISHED_CHILDREN = "{published_children}"

# The tox envs that install the released set from PyPI.
_TOX_PUBLISHED_ENVS = ["verify-published", "security"]

_SCRIPT_INVOCATION = "scripts/published_packages.py"

# A distribution name written out in a workflow build array, e.g. ``"mloda-community-ffill"``.
_QUOTED_DISTRIBUTION_RE = re.compile(r'"(mloda-[A-Za-z0-9._-]+)"')

# A hand-pinned distribution specifier, e.g. ``mloda-community-ffill==``.
_PINNED_DISTRIBUTION_RE = re.compile(r"mloda-[A-Za-z0-9._-]*==")

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


def _generated_data_operations() -> dict[str, Any]:
    """Parsed pyproject document the generator emits for the data-operations base package."""
    shared = _load_toml(_SHARED_CONFIG)
    packages = _packages()
    content: str = gen.generate_pyproject(_DATA_OPERATIONS, packages[_DATA_OPERATIONS], shared, packages)
    return tomllib.loads(content)


def _committed_data_operations() -> dict[str, Any]:
    """Parsed committed pyproject.toml of the data-operations base package."""
    pyproject_path = _REPO_ROOT / _packages()[_DATA_OPERATIONS]["path"] / "pyproject.toml"
    assert pyproject_path.exists(), f"{pyproject_path} is missing (run scripts/generate_pyproject.py)"
    return _load_toml(pyproject_path)


def _leaked_child_packages(listed: list[str]) -> list[str]:
    """Entries of a ``[tool.setuptools] packages`` list that belong to a published child package."""
    packages = _packages()
    children = [packages[name]["path"].replace("/", ".") for name in _published_children()]
    return [entry for entry in listed if any(entry == child or entry.startswith(child + ".") for child in children)]


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

    exit_code = _cli(monkeypatch, tmp_path, [])()

    assert exit_code != 0, "published_packages.main() must exit non-zero when no package carries 'published = true'"


def test_release_workflow_has_no_hardcoded_distribution_list() -> None:
    """The build array must be generated, not typed out; the typed-out copy is what drifted."""
    names = sorted(set(_QUOTED_DISTRIBUTION_RE.findall(_RELEASE_WORKFLOW.read_text())))
    assert names == [], (
        f".github/workflows/release.yaml still names {len(names)} distribution(s) itself, starting with "
        f"{names[:3]}; fill 'packages=( ... )' from 'python {_SCRIPT_INVOCATION}' instead."
    )


def test_release_workflow_builds_from_the_published_script() -> None:
    """The workflow reads the released set from the single source."""
    assert _SCRIPT_INVOCATION in _RELEASE_WORKFLOW.read_text(), (
        f".github/workflows/release.yaml does not invoke {_SCRIPT_INVOCATION}, so its build array is a "
        "second copy of the released set."
    )


@pytest.mark.parametrize("env_name", _TOX_PUBLISHED_ENVS)
def test_tox_env_holds_no_pinned_distribution_list(env_name: str) -> None:
    """tox must not re-type the released set to install it."""
    pins = sorted(set(_PINNED_DISTRIBUTION_RE.findall(_tox_block(env_name))))
    assert pins == [], (
        f"tox.ini [testenv:{env_name}] still pins {len(pins)} distribution(s) by hand, starting with "
        f"{pins[:3]}; build the list from 'python {_SCRIPT_INVOCATION} --pin ...' instead."
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

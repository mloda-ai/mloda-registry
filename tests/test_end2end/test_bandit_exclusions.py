"""The bandit gate in tox.ini must scan scripts/verify_builds.py and scripts/verify_build_floor.py.
Bandit applies each exclusion entry both as an fnmatch glob and as a plain substring of the full
path, so a bare entry like "build" silently drops every path containing that substring, while genuine
artifact directories must stay excluded. The gate therefore passes an explicit -x glob list: without
one, bandit appends its built-in defaults (.svn, CVS, .eggs, ...) bare."""

from __future__ import annotations

import configparser
import shlex
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

import pytest
from bandit.core import config as b_config
from bandit.core import constants as b_constants
from bandit.core import manager as b_manager

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_TOX_INI = _REPO_ROOT / "tox.ini"

# The real-world artifact the "build" entry exists for: setuptools mirrors sources into build/lib.
_NESTED_BUILD_ARTIFACT = (
    "mloda/community/feature_groups/data_operations/aggregation/build/lib/"
    "mloda/community/feature_groups/data_operations/aggregation/base.py"
)

# Artifact paths the gate must keep excluding, in both bare and walk-style ("./") forms.
_ARTIFACT_PATHS = [
    _NESTED_BUILD_ARTIFACT,
    "./" + _NESTED_BUILD_ARTIFACT,
    "dist/mloda_registry-0.1.0/pkg/module.py",
    ".venv/lib/python3.12/site-packages/module.py",
    "__pycache__/module.py",
    "pkg/dep.egg-info",
    "migrations/0001_initial.py",
    "./.devcontainer/setup.py",
    ".vscode/settings.json",
    "attribution/report.py",
]

# One synthetic lookalike per bandit built-in default: the name occurs only as a substring of a
# path component, never as a component itself.
_DEFAULT_NAME_LOOKALIKES = {
    ".svn": "mloda/pkg.svnx/module.py",
    "CVS": "mloda/tools/CVS_reader.py",
    ".bzr": "mloda/pkg.bzrx/module.py",
    ".hg": "mloda/pkg.hgx/module.py",
    ".git": ".github/scripts/module.py",
    "__pycache__": "mloda/__pycache__x/module.py",
    ".tox": "mloda/pkg.toxic/module.py",
    ".eggs": "mloda/data.eggs_reader/module.py",
    "*.egg": "mloda/pkg.egg_helper/module.py",
}

# Genuine default-named artifact directories the gate excludes must keep out of the scan.
_DEFAULT_NAMED_ARTIFACT_PATHS = [
    ".git/hooks/update.py",
    "mloda/.svn/text-base/module.py",
    "CVS/module.py",
    "mloda/CVS/module.py",
    "mloda/.hg/store/module.py",
    "mloda/.bzr/checkout/module.py",
    ".eggs/pytest_runner-6.0.0/setup.py",
    "mloda/pkg/foo.egg/module.py",
]


def _gate_cli_excludes() -> str:
    """The comma-separated -x exclusion list the tox.ini bandit gate passes."""
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(_TOX_INI)
    for line in parser["testenv"]["commands"].splitlines():
        tokens = shlex.split(line)
        if not tokens or tokens[0] != "bandit":
            continue
        if "-x" in tokens[:-1]:
            return tokens[tokens.index("-x") + 1]
        pytest.fail(
            "the tox.ini bandit gate passes no explicit -x, so bandit appends its built-in default "
            "excludes bare and drops any path merely containing one as a substring"
        )
    pytest.fail("no bandit command found in tox.ini [testenv] commands")


def _discover(targets: list[str], config_file: Path | None) -> Any:
    """BanditManager after file discovery, wired exactly like the CLI gate."""
    conf = b_config.BanditConfig(config_file=str(config_file)) if config_file else b_config.BanditConfig()
    mgr = b_manager.BanditManager(conf, "file")
    mgr.discover_files(targets, True, _gate_cli_excludes())
    return mgr


def _configured_exclude_dirs() -> list[str]:
    """The [tool.bandit] exclude_dirs entries declared in the repo pyproject.toml."""
    entries: list[str] = tomllib.loads(_PYPROJECT.read_text())["tool"]["bandit"]["exclude_dirs"]
    return entries


def test_gate_scans_the_verify_build_scripts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The two verify scripts must be discovered, not swallowed by a substring exclude."""
    monkeypatch.chdir(_REPO_ROOT)
    mgr = _discover(["scripts"], _PYPROJECT)
    expected = {"scripts/verify_builds.py", "scripts/verify_build_floor.py"}
    missing = expected - set(mgr.files_list)
    dropped = [path for path in mgr.excluded_files if path.endswith(".py")]
    assert not missing, (
        f"the bandit gate never scans {sorted(missing)}: an exclude_dirs entry matches them as a plain "
        f"substring of the path (excluded .py files: {dropped})"
    )


def test_gate_scans_the_verify_build_scripts_in_walk_form(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate walks from ".", so the scripts surface as "./"-prefixed paths and must stay included."""
    monkeypatch.chdir(_REPO_ROOT)
    targets = ["./scripts/verify_builds.py", "./scripts/verify_build_floor.py"]
    mgr = _discover(targets, _PYPROJECT)
    dropped = set(targets) & set(mgr.excluded_files)
    assert not dropped, f'an exclude_dirs entry matches the "./"-prefixed form the gate produces: {sorted(dropped)}'
    # bandit re-anchors explicit file targets under "." before adding them to files_list.
    missing = {"./" + target for target in targets} - set(mgr.files_list)
    assert not missing, (
        f"the verify scripts never landed in the scan list: {sorted(missing)} (files_list: {mgr.files_list})"
    )


def test_artifact_paths_stay_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build, dist, venv, pycache, and egg-info artifacts must stay out of the scan."""
    monkeypatch.chdir(_REPO_ROOT)
    mgr = _discover(list(_ARTIFACT_PATHS), _PYPROJECT)
    for path in _ARTIFACT_PATHS:
        assert path in mgr.excluded_files, f"{path} is an artifact path and must stay excluded from the gate"
    assert not mgr.files_list, f"artifact paths leaked into the scan: {mgr.files_list}"


def test_every_configured_exclude_entry_excludes_a_representative(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each exclude_dirs entry keeps rejecting a path derived from it."""
    monkeypatch.chdir(_REPO_ROOT)
    entries = _configured_exclude_dirs()
    assert entries, "[tool.bandit] exclude_dirs vanished from pyproject.toml"

    representatives = {
        entry: entry.replace("*", "sample") if "*" in entry else f"{entry}/artifact.py" for entry in entries
    }
    mgr = _discover(list(representatives.values()), _PYPROJECT)
    for entry, representative in representatives.items():
        assert representative in mgr.excluded_files, (
            f"exclude_dirs entry {entry!r} no longer excludes {representative!r}"
        )


def test_build_output_exclusion_comes_from_the_pyproject_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bandit defaults alone do not exclude build outputs; pyproject.toml must carry that entry."""
    monkeypatch.chdir(_REPO_ROOT)
    target = "mloda/x/build/lib/y.py"

    with_config = _discover([target], _PYPROJECT)
    assert target in with_config.excluded_files, f"{target} must be excluded by the pyproject bandit config"

    without_config = _discover([target], None)
    assert "./" + target in without_config.files_list, (
        f"{target} is already excluded without the pyproject config; this guard no longer proves the "
        "config carries the build-output exclusion"
    )


def test_effective_exclusions_are_globs_not_bare_substrings() -> None:
    """Every effective exclusion entry must contain a "*" so bandit's substring branch stays inert."""
    entries = _configured_exclude_dirs() + _gate_cli_excludes().split(",")
    bare = [entry for entry in entries if "*" not in entry]
    assert not bare, f"bare exclusion entries substring-match every path containing them: {bare}"


def test_paths_merely_containing_a_default_name_stay_scanned(monkeypatch: pytest.MonkeyPatch) -> None:
    """A path containing a bandit default name only as a substring must still be scanned."""
    monkeypatch.chdir(_REPO_ROOT)
    assert set(_DEFAULT_NAME_LOOKALIKES) == set(b_constants.EXCLUDE), (
        "bandit's built-in default exclude list changed; update _DEFAULT_NAME_LOOKALIKES and mirror "
        "the new name as anchored globs in the tox.ini bandit -x list"
    )
    targets = [form for path in _DEFAULT_NAME_LOOKALIKES.values() for form in (path, "./" + path)]
    mgr = _discover(targets, _PYPROJECT)
    dropped = set(targets) & set(mgr.excluded_files)
    assert not dropped, f"paths merely containing a default name were dropped from the scan: {sorted(dropped)}"
    missing = {"./" + target for target in targets} - set(mgr.files_list)
    assert not missing, f"lookalike paths never landed in the scan list: {sorted(missing)}"


def test_default_named_artifact_paths_stay_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Genuine VCS and egg artifact directories must stay out of the scan under the gate excludes."""
    monkeypatch.chdir(_REPO_ROOT)
    targets = [form for path in _DEFAULT_NAMED_ARTIFACT_PATHS for form in (path, "./" + path)]
    mgr = _discover(targets, _PYPROJECT)
    for path in targets:
        assert path in mgr.excluded_files, f"{path} is a default-named artifact path and must stay excluded"
    assert not mgr.files_list, f"default-named artifact paths leaked into the scan: {mgr.files_list}"

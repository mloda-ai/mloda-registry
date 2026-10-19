"""Tests that scripts/verify_backend_extras.py proves each backend extra is enough on its own, and that the
tox env and the release workflow gate on it."""

from __future__ import annotations

import re
import subprocess  # nosec
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.script_loader import load_script
from tests.toml_loader import load_toml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "verify_backend_extras.py"
_PACKAGES_CONFIG = _REPO_ROOT / "config" / "packages.toml"
_RELEASE_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "release.yaml"
_TOX_INI = _REPO_ROOT / "tox.ini"

_BASE = "mloda-community-data-operations"
_RANK_MODULE = "mloda.community.feature_groups.data_operations.row_preserving.rank.sqlite_rank"
_NON_BACKEND_EXTRAS = {"all", "dev"}


def _script() -> ModuleType:
    assert _SCRIPT_PATH.exists(), f"{_SCRIPT_PATH} is missing"
    return load_script("verify_backend_extras", _SCRIPT_PATH)


def _fn(name: str) -> Callable[..., Any]:
    fn: Callable[..., Any] | None = getattr(_script(), name, None)
    assert callable(fn), f"verify_backend_extras.{name} must be a callable"
    return fn


def _packages() -> dict[str, dict[str, Any]]:
    packages: dict[str, dict[str, Any]] = load_toml(_PACKAGES_CONFIG)["packages"]
    return packages


def _tox_block(env_name: str) -> str:
    match = re.search(
        rf"^\[testenv:{re.escape(env_name)}\]\n(.*?)(?=\n\[|\Z)", _TOX_INI.read_text(), re.MULTILINE | re.DOTALL
    )
    assert match is not None, f"tox.ini has no [testenv:{env_name}] section"
    return match.group(1)


def _job_block(job: str) -> str:
    match = re.search(rf"^  {re.escape(job)}:\n(.*?)(?=^  \w+:\n|\Z)", _RELEASE_WORKFLOW.read_text(), re.M | re.S)
    assert match is not None, f".github/workflows/release.yaml has no job '{job}'"
    return match.group(1)


def _run_probe(source: str, pythonpath: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = {"PATH": "/usr/bin:/bin"}
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    return subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, cwd=_REPO_ROOT, env=env)  # nosec


def test_backend_jobs_cover_every_backend_and_each_leaf_declared_extra() -> None:
    script = _script()
    packages = _packages()
    jobs = script.backend_jobs(packages)

    assert set(jobs) == {backend for _, backend in script.BACKEND_PREFIXES}
    sqlite_leaves = {leaf: classes for leaf, classes in jobs["sqlite"]}
    assert (_RANK_MODULE, "SqliteRank") in sqlite_leaves["mloda-community-rank"]
    for backend, entries in jobs.items():
        declared = {
            n for n, entry in script.leaf_packages(packages) if backend in entry.get("optional_dependencies", {})
        }
        assert {leaf for leaf, _ in entries} == declared, f"{backend}: leaves differ from config extras"
    for name, entry in script.leaf_packages(packages):
        backends = {b for b, entries in jobs.items() if name in {leaf for leaf, _ in entries}}
        assert backends == set(entry.get("optional_dependencies", {})) - _NON_BACKEND_EXTRAS, name


def test_leaf_packages_excludes_a_path_without_a_manifest() -> None:
    packages = _packages()
    base_path = packages[_BASE]["path"]
    real_leaf = next(
        n
        for n, e in packages.items()
        if e["path"].startswith(base_path + "/") and (_REPO_ROOT / e["path"] / "manifest.py").is_file()
    )
    config = {
        _BASE: packages[_BASE],
        "x": {"path": "mloda/community/feature_groups/data_operations/no_such_leaf"},
        real_leaf: packages[real_leaf],
    }

    assert [name for name, _ in _fn("leaf_packages")(config)] == [real_leaf]


def test_backend_jobs_rejects_a_backend_without_leaves() -> None:
    with pytest.raises(ValueError):
        _fn("backend_jobs")({})


def test_backend_for_module_rejects_unknown_prefix() -> None:
    assert _fn("backend_for_module")("sqlite_rank") == "sqlite"
    with pytest.raises(ValueError):
        _fn("backend_for_module")("mystery_rank")


@pytest.mark.parametrize(
    "content",
    ["X = 1\n", "load_plugin_classes(__package__, SPECS)\n", 'load_plugin_classes(__package__, [(name, "A")])\n'],
    ids=["no-call", "non-literal-list", "non-literal-module"],
)
def test_manifest_specs_rejects_bad_manifests(tmp_path: Path, content: str) -> None:
    manifest = tmp_path / "manifest.py"
    manifest.write_text(content)
    with pytest.raises(ValueError):
        _fn("manifest_specs")(manifest)


def test_manifest_specs_reads_literal_pairs(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.py"
    manifest.write_text('load_plugin_classes(__package__, [("sqlite_rank", "SqliteRank")])\n')
    assert _fn("manifest_specs")(manifest) == [("sqlite_rank", "SqliteRank")]


def test_install_requirements_use_absolute_file_uris(tmp_path: Path) -> None:
    wheels = {"mloda-community-rank": tmp_path / "rank.whl", _BASE: tmp_path / "base.whl"}
    reqs = _fn("install_requirements")("sqlite", ["mloda-community-rank"], wheels)

    assert sorted(reqs) == sorted(
        [
            f"mloda-community-rank[sqlite] @ {(tmp_path / 'rank.whl').resolve().as_uri()}",
            f"{_BASE} @ {(tmp_path / 'base.whl').resolve().as_uri()}",
        ]
    )
    assert all("@ file:///" in req for req in reqs)


def test_probe_source_passes_for_a_real_class() -> None:
    result = _run_probe(_fn("probe_source")([(_RANK_MODULE, "SqliteRank")], []))
    assert result.returncode == 0, result.stderr


def test_probe_source_fails_naming_a_class_with_an_unavailable_framework(tmp_path: Path) -> None:
    (tmp_path / "fake_backend_mod.py").write_text(
        textwrap.dedent(
            """
            class Framework:
                @classmethod
                def is_available(cls):
                    return False

            class BrokenClass:
                @classmethod
                def compute_framework_rule(cls):
                    return {Framework}
            """
        )
    )
    result = _run_probe(_fn("probe_source")([("fake_backend_mod", "BrokenClass")], []), pythonpath=tmp_path)
    assert result.returncode != 0
    assert "BrokenClass" in result.stdout + result.stderr


def test_probe_source_fails_when_an_expected_absent_module_is_importable() -> None:
    result = _run_probe(_fn("probe_source")([(_RANK_MODULE, "SqliteRank")], ["pyarrow"]))
    assert result.returncode != 0
    assert "pyarrow" in result.stdout + result.stderr


def test_absent_in_bare_install_names_the_backend_packages() -> None:
    assert {"pyarrow", "polars", "pandas", "duckdb"} <= set(_script().ABSENT_IN_BARE_INSTALL)


def test_build_wheels_issues_uv_build_per_package_and_reports_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = _script()
    calls: list[list[str]] = []

    def fake_run(command: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        name = command[command.index("--package") + 1]
        if name == "pkg-ok":
            (tmp_path / "pkg_ok-1.0-py3-none-any.whl").write_text("")
        return subprocess.CompletedProcess(command, 1 if name == "pkg-bad" else 0, "", "boom")

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    wheels, errors = script.build_wheels(["pkg-ok", "pkg-bad", "pkg-missing"], tmp_path)

    assert [
        "uv",
        "build",
        "--package",
        "pkg-ok",
        "--out-dir",
        str(tmp_path),
        "--wheel",
        "--no-build-isolation",
    ] in calls
    assert set(wheels) == {"pkg-ok"}
    assert len(errors) == 2
    assert any("pkg-bad" in e for e in errors) and any("pkg-missing" in e for e in errors)


def _recording_run(
    calls: list[tuple[list[str], dict[str, Any]]], fail_on: str | None = None
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def fake_run(command: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        failed = fail_on is not None and fail_on in " ".join(command) + str(kwargs.get("input", ""))
        return subprocess.CompletedProcess(command, 1 if failed else 0, "", "probe exploded" if failed else "")

    return fake_run


_CLASSES = [("sqlite-leaf", [(_RANK_MODULE, "SqliteRank")])]


def test_verify_extra_installs_and_probes_from_an_isolated_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    script = _script()
    calls: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.setattr(script.subprocess, "run", _recording_run(calls))
    wheels = {"sqlite-leaf": tmp_path / "leaf.whl", _BASE: tmp_path / "base.whl"}

    assert script.verify_extra("sqlite", _CLASSES, wheels) is None

    cwds = {Path(kwargs["cwd"]) for _, kwargs in calls}
    assert len(cwds) == 1 and _REPO_ROOT not in cwds
    commands = [" ".join(command) for command, _ in calls]
    assert commands[0].startswith("uv venv")
    install = next(c for c in commands if c.startswith("uv pip install"))
    for requirement in script.install_requirements("sqlite", ["sqlite-leaf"], wheels):
        assert requirement in install
    expected_probe = script.probe_source([(_RANK_MODULE, "SqliteRank")], [])
    assert any(expected_probe in c + str(k.get("input", "")) for c, k in [(" ".join(c), k) for c, k in calls])


def test_verify_extra_reports_probe_failure_naming_the_extra(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    script = _script()
    marker = "SqliteRank"
    monkeypatch.setattr(script.subprocess, "run", _recording_run([], fail_on=marker))
    wheels = {"sqlite-leaf": tmp_path / "leaf.whl", _BASE: tmp_path / "base.whl"}

    error = script.verify_extra("sqlite", _CLASSES, wheels)

    assert error is not None and "sqlite" in error and "probe exploded" in error


def test_verify_extra_python_dict_expects_backend_packages_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = _script()
    calls: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.setattr(script.subprocess, "run", _recording_run(calls))
    wheels = {"dict-leaf": tmp_path / "leaf.whl", _BASE: tmp_path / "base.whl"}
    classes = [("dict-leaf", [("mod.python_dict_rank", "PythonDictRank")])]

    assert script.verify_extra("python_dict", classes, wheels) is None

    expected = script.probe_source([("mod.python_dict_rank", "PythonDictRank")], list(script.ABSENT_IN_BARE_INSTALL))
    assert any(expected in " ".join(c) + str(k.get("input", "")) for c, k in calls)


def test_main_runs_every_extra_after_one_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    script = _script()
    monkeypatch.chdir(tmp_path)  # main() chdirs to the repo root; monkeypatch restores the cwd
    ran: list[str] = []

    def fake_verify(extra: str, leaves_and_classes: Any, wheels: Any) -> str | None:
        ran.append(extra)
        return f"extra {extra} failed: boom" if extra == "sqlite" else None

    monkeypatch.setattr(script, "build_wheels", lambda names, out_dir: ({}, []))
    monkeypatch.setattr(script, "verify_extra", fake_verify)

    assert script.main() == 1
    assert set(ran) == {backend for _, backend in script.BACKEND_PREFIXES}
    assert "sqlite" in capsys.readouterr().out


def test_tox_env_runs_the_script_without_installing_the_project() -> None:
    block = _tox_block("verify-backend-extras")
    assert re.search(r"^skip_install\s*=\s*true$", block, re.MULTILINE)
    assert re.search(r"^runner\s*=\s*uv-venv-runner$", block, re.MULTILINE)
    deps = re.search(r"^deps\s*=\s*\n((?:[ \t]+\S.*\n?)+)", block, re.MULTILINE)
    assert deps is not None and "setuptools" in deps.group(1) and "tomli" in deps.group(1)
    assert re.search(r"^[^\n#]*python scripts/verify_backend_extras\.py\s*$", block, re.MULTILINE)


def test_release_workflow_gates_github_release_on_the_verified_sha() -> None:
    job = _job_block("verify_backend_extras")
    assert re.search(r"permissions:\s*\n\s+contents: read", job)
    assert re.search(r"^[^\n#]*tox -e verify-backend-extras", job, re.MULTILINE)
    assert re.search(r"outputs:\s*\n\s+sha:", job)
    assert "continue-on-error" not in job

    release = _job_block("github_release")
    assert re.search(r"needs:\s*(\[?\s*)verify_backend_extras", release)
    assert "needs.verify_backend_extras.outputs.sha" in release
    assert re.search(r"git rev-parse HEAD", release)

"""Packaging contract helpers: distribution completeness (build stubbed) and cold imports (run for real)."""

from __future__ import annotations

import importlib.util
import io
import shutil
import subprocess  # nosec
import sys
import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

import mloda.testing.packaging as packaging
from mloda.testing.packaging import assert_cold_imports, assert_distributions_complete

_FULL_MODULE_FILES = [
    "acme/__init__.py",
    "acme/feature_groups/__init__.py",
    "acme/feature_groups/core/__init__.py",
    "acme/feature_groups/core/logic.py",
]
_MISSING_CORE = ["acme/feature_groups/core/__init__.py", "acme/feature_groups/core/logic.py"]


def _write(path: Path, content: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _make_project(root: Path, files: Iterable[str] = _FULL_MODULE_FILES, src_layout: bool = False) -> Path:
    """Project dir with a pyproject.toml and the given package files (under ``src/`` for a src layout)."""
    _write(root / "pyproject.toml", "[project]\nname = 'acme'\nversion = '0.1.0'\n")
    for rel in files:
        _write(root / "src" / rel if src_layout else root / rel)
    return root


def _write_wheel(path: Path, names: Iterable[str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name in [*names, "acme-0.1.0.dist-info/METADATA"]:
            zf.writestr(name, "")


def _write_sdist(path: Path, members: Iterable[str]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name in [*members, "acme-0.1.0/PKG-INFO"]:
            info = tarfile.TarInfo(name)
            info.size = 0
            tf.addfile(info, io.BytesIO(b""))


class _Build:
    """Records what the stubbed build saw; contract: ``[uv, "build", "--out-dir", out, copy_dir, *build_args]``."""

    def __init__(self) -> None:
        self.cmd: list[str] = []
        self.copy_dir: Path | None = None
        self.copy_files: set[str] = set()
        self.wheel_files: list[str] = list(_FULL_MODULE_FILES)
        self.sdist_files: list[str] = list(_FULL_MODULE_FILES)
        self.src_layout = False
        self.returncode = 0
        self.stderr = ""

    def __call__(self, cmd: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.cmd = [str(part) for part in cmd]
        out_index = self.cmd.index("--out-dir")
        out_dir = Path(self.cmd[out_index + 1])
        self.copy_dir = Path(self.cmd[out_index + 2])
        self.copy_files = {str(p.relative_to(self.copy_dir)) for p in self.copy_dir.rglob("*") if p.is_file()} | {
            str(p.relative_to(self.copy_dir)) for p in self.copy_dir.rglob("*") if p.is_dir()
        }
        if self.returncode == 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            prefix = "acme-0.1.0/src/" if self.src_layout else "acme-0.1.0/"
            _write_wheel(out_dir / "acme-0.1.0-py3-none-any.whl", self.wheel_files)
            _write_sdist(out_dir / "acme-0.1.0.tar.gz", [prefix + f for f in self.sdist_files])
        return subprocess.CompletedProcess(self.cmd, self.returncode, stdout="", stderr=self.stderr)


@pytest.fixture
def build(monkeypatch: pytest.MonkeyPatch) -> _Build:
    fake = _Build()
    monkeypatch.setattr(packaging.subprocess, "run", fake)
    return fake


def _failure(project: Path, **kwargs: Any) -> str:
    with pytest.raises(AssertionError) as excinfo:
        assert_distributions_complete(project, "acme", **kwargs)
    return str(excinfo.value)


def _section(message: str, heading: str, other: str) -> str:
    """Text of ``message`` from ``heading`` up to the ``other`` heading, or the end (case-insensitive)."""
    lowered = message.lower()
    start = lowered.index(heading)
    end = lowered.find(other, start + len(heading))
    return message[start:] if end == -1 else message[start:end]


# --------------------------------------------------------------------------------------
# assert_distributions_complete (build stubbed)
# --------------------------------------------------------------------------------------


def test_complete_wheel_and_sdist_pass(tmp_path: Path, build: _Build) -> None:
    assert_distributions_complete(_make_project(tmp_path), "acme")
    assert build.cmd, "the stubbed build must have been invoked"


def test_wheel_missing_module_is_reported_under_wheel_only(tmp_path: Path, build: _Build) -> None:
    build.wheel_files = [f for f in _FULL_MODULE_FILES if f not in _MISSING_CORE]
    message = _failure(_make_project(tmp_path))

    assert "wheel" in message.lower()
    assert "acme.feature_groups.core" in message
    assert "acme.feature_groups.core.logic" in message
    assert "sdist" not in message.lower()


def test_wheel_and_sdist_failures_are_reported_under_their_own_heading(tmp_path: Path, build: _Build) -> None:
    build.wheel_files = [f for f in _FULL_MODULE_FILES if f not in _MISSING_CORE]
    build.sdist_files = [f for f in _FULL_MODULE_FILES if f != "acme/feature_groups/core/logic.py"]
    message = _failure(_make_project(tmp_path))

    wheel_part = _section(message, "wheel", "sdist")
    sdist_part = _section(message, "sdist", "wheel")
    assert "acme.feature_groups.core.logic" in wheel_part
    assert "acme.feature_groups.core.logic" in sdist_part
    assert wheel_part.replace("acme.feature_groups.core.logic", "").count("acme.feature_groups.core") >= 1
    assert "acme.feature_groups.core" not in sdist_part.replace("acme.feature_groups.core.logic", "")


def test_sdist_missing_module_is_reported_under_sdist_only(tmp_path: Path, build: _Build) -> None:
    build.sdist_files = [f for f in _FULL_MODULE_FILES if f != "acme/feature_groups/core/logic.py"]
    message = _failure(_make_project(tmp_path))

    assert "sdist" in message.lower()
    assert "acme.feature_groups.core.logic" in message
    assert "wheel" not in message.lower()


def test_test_code_and_main_are_not_required(tmp_path: Path, build: _Build) -> None:
    extras = [
        "acme/tests/__init__.py",
        "acme/tests/test_core.py",
        "acme/feature_groups/test_logic.py",
        "acme/feature_groups/conftest.py",
        "acme/__main__.py",
    ]
    project = _make_project(tmp_path, [*_FULL_MODULE_FILES, *extras])

    assert_distributions_complete(project, "acme")


def test_exclude_drops_package_and_children(tmp_path: Path, build: _Build) -> None:
    optional = ["acme/optional/__init__.py", "acme/optional/deep/__init__.py", "acme/optional/deep/mod.py"]
    project = _make_project(tmp_path, [*_FULL_MODULE_FILES, *optional])

    assert "acme.optional" in _failure(project)
    assert_distributions_complete(project, "acme", exclude=("acme.optional",))


def test_src_layout(tmp_path: Path, build: _Build) -> None:
    build.src_layout = True
    project = _make_project(tmp_path, src_layout=True)
    assert_distributions_complete(project, "acme")

    build.wheel_files = [f for f in _FULL_MODULE_FILES if f not in _MISSING_CORE]
    assert "acme.feature_groups.core" in _failure(project)


def test_namespace_package_dir_modules_are_required(tmp_path: Path, build: _Build) -> None:
    files = ["acme/__init__.py", "acme/ns/plugin.py"]
    project = _make_project(tmp_path, files)
    build.wheel_files = ["acme/__init__.py"]
    build.sdist_files = files

    message = _failure(project)

    assert "acme.ns.plugin" in message
    assert "sdist" not in message.lower()


def test_build_copy_excludes_artifacts_and_leaves_project_untouched(tmp_path: Path, build: _Build) -> None:
    project = _make_project(tmp_path)
    _write(project / "build/lib/acme/feature_groups/core/logic.py")
    for junk in ("dist/old.whl", "acme.egg-info/PKG-INFO", ".venv/pyvenv.cfg", ".git/HEAD", ".tox/x/log"):
        _write(project / junk)
    before = sorted(str(p.relative_to(project)) for p in project.rglob("*"))

    assert_distributions_complete(project, "acme")

    assert build.copy_dir is not None and build.copy_dir != project
    seen = {Path(p).parts[0] for p in build.copy_files}
    assert seen.isdisjoint({"build", "dist", "acme.egg-info", ".venv", ".git", ".tox"}), seen
    assert "acme" in seen and "pyproject.toml" in seen
    assert sorted(str(p.relative_to(project)) for p in project.rglob("*")) == before


def test_build_failure_raises_with_stderr(tmp_path: Path, build: _Build) -> None:
    build.returncode = 2
    build.stderr = "error: backend exploded"

    assert "backend exploded" in _failure(_make_project(tmp_path))


def test_build_args_are_appended_to_the_command(tmp_path: Path, build: _Build) -> None:
    assert_distributions_complete(_make_project(tmp_path), "acme", build_args=("--no-build-isolation", "--quiet"))

    assert build.cmd[-2:] == ["--no-build-isolation", "--quiet"]
    assert "build" in build.cmd
    assert build.cmd[build.cmd.index("--out-dir") + 2] == str(build.copy_dir)


def test_unknown_package_dir_raises_value_error(tmp_path: Path, build: _Build) -> None:
    project = _make_project(tmp_path)

    with pytest.raises(ValueError):
        assert_distributions_complete(project, "nope")
    with pytest.raises(ValueError):
        assert_cold_imports(project, "nope")


# --------------------------------------------------------------------------------------
# assert_cold_imports (real subprocesses)
# --------------------------------------------------------------------------------------

_NEEDS_SIBLING = "import acme.feature_groups\n\nVALUE = acme.feature_groups.plugin.VALUE\n"


def _make_cold_project(root: Path, extra: dict[str, str] | None = None) -> Path:
    _write(root / "acme/__init__.py")
    _write(root / "acme/feature_groups/__init__.py")
    _write(root / "acme/feature_groups/plugin.py", "VALUE = 1\n")
    _write(root / "acme/feature_groups/needs_sibling.py", _NEEDS_SIBLING)
    _write(root / "acme/clean.py", "X = 1\n")
    for rel, content in (extra or {}).items():
        _write(root / rel, content)
    return root


def test_toy_sibling_module_works_once_the_plugin_is_imported(tmp_path: Path) -> None:
    project = _make_cold_project(tmp_path)
    code = "import acme.feature_groups.plugin, acme.feature_groups.needs_sibling"
    result = subprocess.run([sys.executable, "-c", code], cwd=project, capture_output=True, text=True)  # nosec

    assert result.returncode == 0, result.stderr


def test_cold_import_reports_sibling_dependent_module(tmp_path: Path) -> None:
    project = _make_cold_project(tmp_path)

    with pytest.raises(AssertionError) as excinfo:
        assert_cold_imports(project, "acme")

    message = str(excinfo.value)
    assert "acme.feature_groups.needs_sibling" in message
    assert "AttributeError" in message
    assert "acme.clean" not in message
    assert "acme.feature_groups.plugin" not in message


def test_cold_import_exclude_passes(tmp_path: Path) -> None:
    project = _make_cold_project(tmp_path)

    assert_cold_imports(project, "acme", exclude=("acme.feature_groups.needs_sibling",))


def test_cold_import_all_clean_passes(tmp_path: Path) -> None:
    project = _make_cold_project(tmp_path)
    (project / "acme/feature_groups/needs_sibling.py").unlink()

    assert_cold_imports(project, "acme")


def test_cold_import_timeout_is_reported(tmp_path: Path) -> None:
    project = _make_cold_project(tmp_path, {"acme/slow.py": "import time\ntime.sleep(30)\n"})
    (project / "acme/feature_groups/needs_sibling.py").unlink()

    with pytest.raises(AssertionError) as excinfo:
        assert_cold_imports(project, "acme", timeout=1)

    message = str(excinfo.value)
    assert "acme.slow" in message
    assert "timeout" in message.lower() or "timed out" in message.lower()
    assert "acme.clean" not in message


# --------------------------------------------------------------------------------------
# Real build (skipped without setuptools or uv; runs in the verify-builds tox env)
# --------------------------------------------------------------------------------------


def _uv_available() -> bool:
    return shutil.which("uv") is not None or importlib.util.find_spec("uv") is not None


_TOY_PYPROJECT = """\
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "acme"
version = "0.1.0"

[tool.setuptools.packages.find]
where = ["."]
include = ["acme", "acme.feature_groups"]
"""


def test_real_build_reports_package_dropped_by_find_include(tmp_path: Path) -> None:
    pytest.importorskip("setuptools")
    if not _uv_available():
        pytest.skip("uv is not available")
    project = _make_project(tmp_path)
    _write(project / "pyproject.toml", _TOY_PYPROJECT)

    message = _failure(project, build_args=("--no-build-isolation",))

    assert "acme.feature_groups.core" in message
    assert "wheel" in message.lower()
    assert "sdist" in message.lower()

"""Packaging contract helpers: distribution completeness (build stubbed) and cold imports (run for real)."""

from __future__ import annotations

import importlib.util
import io
import re
import shutil
import subprocess  # nosec
import sys
import tarfile
import types
import zipfile
from collections.abc import Callable, Iterable
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
    """Stubbed build seams.

    ``mloda.testing.packaging._find_uv()`` returns a fake path; ``subprocess.run`` inside the module is replaced and
    the command is ``[uv, "build", "--python", sys.executable, "--out-dir", out_dir, copy_dir, *build_args]``, so the
    copy dir immediately follows the out dir. The stub writes a ``.whl`` (zip) and a ``.tar.gz`` sdist (top dir
    ``acme-0.1.0/``, plus ``src/`` for a src layout) into the out dir.
    """

    def __init__(self) -> None:
        self.cmd: list[str] = []
        self.copy_dir: Path | None = None
        self.copy_files: set[str] = set()
        self.wheel_files: list[str] = list(_FULL_MODULE_FILES)
        self.sdist_files: list[str] = list(_FULL_MODULE_FILES)
        self.src_layout = False
        self.extra_wheel_files: list[str] | None = None
        self.skip_sdist = False
        self.returncode = 0
        self.stderr = ""
        self.raises: BaseException | None = None

    def __call__(self, cmd: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.cmd = [str(part) for part in cmd]
        if self.raises is not None:
            raise self.raises
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
            if self.extra_wheel_files is not None:
                _write_wheel(out_dir / "acme-0.1.0-cp3-none-any.whl", self.extra_wheel_files)
            if not self.skip_sdist:
                _write_sdist(out_dir / "acme-0.1.0.tar.gz", [prefix + f for f in self.sdist_files])
        return subprocess.CompletedProcess(self.cmd, self.returncode, stdout="", stderr=self.stderr)


@pytest.fixture
def build(monkeypatch: pytest.MonkeyPatch) -> _Build:
    fake = _Build()
    monkeypatch.setattr(packaging, "_find_uv", lambda: "/fake/uv")
    monkeypatch.setattr("mloda.testing.packaging.subprocess.run", fake)
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


_LOGIC = ["acme/feature_groups/core/logic.py"]
_CORE_PKG = r"acme\.feature_groups\.core(?!\.)"
_CORE_LOGIC = r"acme\.feature_groups\.core\.logic"


@pytest.mark.parametrize(
    "wheel_drop, sdist_drop, wheel_expect, sdist_expect",
    [
        (_MISSING_CORE, [], [_CORE_PKG, _CORE_LOGIC], None),
        ([], _LOGIC, None, [_CORE_LOGIC]),
        (_MISSING_CORE, _LOGIC, [_CORE_PKG, _CORE_LOGIC], [_CORE_LOGIC]),
    ],
    ids=["wheel_only", "sdist_only", "both"],
)
def test_failures_are_reported_under_their_own_heading(
    tmp_path: Path,
    build: _Build,
    wheel_drop: list[str],
    sdist_drop: list[str],
    wheel_expect: list[str] | None,
    sdist_expect: list[str] | None,
) -> None:
    build.wheel_files = [f for f in _FULL_MODULE_FILES if f not in wheel_drop]
    build.sdist_files = [f for f in _FULL_MODULE_FILES if f not in sdist_drop]
    message = _failure(_make_project(tmp_path))

    for heading, other, expected in (("wheel", "sdist", wheel_expect), ("sdist", "wheel", sdist_expect)):
        if expected is None:
            assert heading not in message.lower()
            continue
        section = _section(message, heading, other)
        assert all(re.search(pattern, section) for pattern in expected), section
    if wheel_expect is not None and sdist_expect is not None:
        assert not re.search(_CORE_PKG, _section(message, "sdist", "wheel"))


def test_test_code_and_main_are_not_required(tmp_path: Path, build: _Build) -> None:
    extras = [
        "acme/tests/__init__.py",
        "acme/tests/test_core.py",
        "acme/feature_groups/test_logic.py",
        "acme/feature_groups/conftest.py",
        "acme/__main__.py",
        "acme/.ipynb_checkpoints/logic-checkpoint.py",
        "acme/feature_groups/my-script.py",
    ]
    project = _make_project(tmp_path, [*_FULL_MODULE_FILES, *extras])

    assert_distributions_complete(project, "acme")


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


def test_missing_sdist_is_reported_as_no_artifact(tmp_path: Path, build: _Build) -> None:
    build.skip_sdist = True

    message = _failure(_make_project(tmp_path))

    assert "sdist" in message and "no artifact" in message


def test_build_args_are_appended_to_the_command(tmp_path: Path, build: _Build) -> None:
    assert_distributions_complete(_make_project(tmp_path), "acme", build_args=("--no-build-isolation", "--quiet"))

    assert build.cmd[-2:] == ["--no-build-isolation", "--quiet"]
    assert "build" in build.cmd
    assert build.cmd[build.cmd.index("--python") + 1] == sys.executable
    assert build.cmd[build.cmd.index("--out-dir") + 2] == str(build.copy_dir)

    assert_distributions_complete(_make_project(tmp_path), "acme", build_args="--no-build-isolation")

    assert build.cmd[-1:] == ["--no-build-isolation"]
    assert "--quiet" not in build.cmd


def test_unknown_package_dir_raises_value_error(tmp_path: Path, build: _Build) -> None:
    project = _make_project(tmp_path)

    with pytest.raises(ValueError):
        assert_distributions_complete(project, "nope")
    with pytest.raises(ValueError):
        assert_cold_imports(project, "nope")


@pytest.mark.parametrize("name", ["build", "dist", "venv", "my_cache"])
def test_subpackage_named_like_a_build_dir_is_copied_and_required(tmp_path: Path, build: _Build, name: str) -> None:
    project = _make_project(tmp_path, [*_FULL_MODULE_FILES, f"acme/{name}/__init__.py"])

    assert f"acme.{name}" in _failure(project)
    assert f"acme/{name}" in build.copy_files


def test_cache_dirs_are_excluded_from_the_copy_at_any_depth(tmp_path: Path, build: _Build) -> None:
    project = _make_project(tmp_path)
    junk_files = (
        "acme/sub/__pycache__/x.pyc",
        "acme/sub/acme.egg-info/PKG-INFO",
        "acme/sub/.pytest_cache/f",
        ".mypy_cache/f",
    )
    for junk in junk_files:
        _write(project / junk)

    assert_distributions_complete(project, "acme")

    assert not [p for p in build.copy_files if "__pycache__" in p or "egg-info" in p or "_cache" in p]


_EXCLUDE_FORMS: dict[str, Callable[[str], Any]] = {
    "tuple": lambda name: (name,),
    "generator": lambda name: (x for x in [name]),
    "bare_str": lambda name: name,
}


@pytest.mark.parametrize("form", list(_EXCLUDE_FORMS))
def test_exclude_forms_drop_the_named_module(tmp_path: Path, form: str) -> None:
    project = _make_cold_project(tmp_path, {"acme/optional.py": "raise RuntimeError('optional dependency missing')\n"})
    _write(project / "acme/feature_groups/needs_sibling.py")

    with pytest.raises(AssertionError, match="acme.optional"):
        assert_cold_imports(project, "acme")
    assert_cold_imports(project, "acme", exclude=_EXCLUDE_FORMS[form]("acme.optional"))


def test_exclude_drops_package_and_children(tmp_path: Path) -> None:
    boom = "raise RuntimeError('optional dependency missing')\n"
    project = _make_cold_project(
        tmp_path, {"acme/optional/__init__.py": boom, "acme/optional/deep/mod.py": boom}, clean=True
    )

    with pytest.raises(AssertionError, match="acme.optional"):
        assert_cold_imports(project, "acme")
    assert_cold_imports(project, "acme", exclude=("acme.optional",))
    with pytest.raises(TypeError):
        assert_distributions_complete(project, "acme", exclude=("acme.optional",))  # type: ignore[call-arg]


def test_timeout_is_reported_as_assertion_error(tmp_path: Path, build: _Build) -> None:
    build.raises = subprocess.TimeoutExpired(["uv"], 1)
    project = _make_project(tmp_path)

    with pytest.raises(AssertionError) as excinfo:
        assert_distributions_complete(project, "acme", timeout=1)

    assert "timed out" in str(excinfo.value).lower() or "timeout" in str(excinfo.value).lower()


def test_modules_in_any_wheel_count_as_present(tmp_path: Path, build: _Build) -> None:
    build.wheel_files = [f for f in _FULL_MODULE_FILES if f not in _MISSING_CORE]
    build.extra_wheel_files = _MISSING_CORE

    assert_distributions_complete(_make_project(tmp_path), "acme")


def test_package_without_python_files_raises(tmp_path: Path, build: _Build) -> None:
    project = _make_project(tmp_path, ["acme/data.txt"])

    with pytest.raises(AssertionError):
        assert_distributions_complete(project, "acme")
    with pytest.raises(AssertionError):
        assert_cold_imports(project, "acme")
    with pytest.raises(AssertionError, match="exclude"):
        assert_cold_imports(_make_cold_project(tmp_path / "cold", clean=True), "acme", exclude=("acme",))


def test_package_named_tests_still_requires_its_modules(tmp_path: Path, build: _Build) -> None:
    files = ["tests/__init__.py", "tests/core.py"]
    project = _make_project(tmp_path, files)
    build.wheel_files = ["tests/__init__.py"]
    build.sdist_files = files

    with pytest.raises(AssertionError) as excinfo:
        assert_distributions_complete(project, "tests")

    assert "tests.core" in str(excinfo.value)


def test_cold_import_package_named_tests_checks_its_modules(tmp_path: Path) -> None:
    project = _write(tmp_path / "tests/__init__.py").parent.parent
    _write(project / "tests/core.py", "raise RuntimeError('boom')\n")

    with pytest.raises(AssertionError, match="tests.core"):
        assert_cold_imports(project, "tests")


# --------------------------------------------------------------------------------------
# uv discovery
# --------------------------------------------------------------------------------------


def _fake_uv_module(find_uv_bin: Callable[[], str]) -> types.ModuleType:
    module = types.ModuleType("uv")
    module.find_uv_bin = find_uv_bin  # type: ignore[attr-defined]
    return module


def _raise_not_found() -> str:
    raise FileNotFoundError("no uv binary")


def test_find_uv_prefers_the_uv_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "uv", _fake_uv_module(lambda: "/pkg/uv"))
    monkeypatch.setattr("mloda.testing.packaging.shutil.which", lambda name: "/path/uv")

    assert packaging._find_uv() == "/pkg/uv"


def test_find_uv_falls_back_to_path_when_find_uv_bin_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "uv", _fake_uv_module(_raise_not_found))
    monkeypatch.setattr("mloda.testing.packaging.shutil.which", lambda name: "/path/uv")

    assert packaging._find_uv() == "/path/uv"


@pytest.mark.parametrize("uv_module", [None, "raising"], ids=["package_missing", "find_uv_bin_raises"])
def test_missing_uv_raises_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, uv_module: str | None
) -> None:
    monkeypatch.setitem(sys.modules, "uv", None if uv_module is None else _fake_uv_module(_raise_not_found))
    monkeypatch.setattr("mloda.testing.packaging.shutil.which", lambda name: None)

    with pytest.raises(RuntimeError, match="uv"):
        assert_distributions_complete(_make_project(tmp_path), "acme")


# --------------------------------------------------------------------------------------
# assert_cold_imports (real subprocesses)
# --------------------------------------------------------------------------------------

_NEEDS_SIBLING = "import acme.feature_groups\n\nVALUE = acme.feature_groups.plugin.VALUE\n"


def _make_cold_project(
    root: Path, extra: dict[str, str] | None = None, clean: bool = False, src_layout: bool = False
) -> Path:
    base = root / "src" if src_layout else root
    _write(base / "acme/__init__.py")
    _write(base / "acme/feature_groups/__init__.py")
    _write(base / "acme/feature_groups/plugin.py", "VALUE = 1\n")
    if not clean:
        _write(base / "acme/feature_groups/needs_sibling.py", _NEEDS_SIBLING)
    _write(base / "acme/clean.py", "X = 1\n")
    for rel, content in (extra or {}).items():
        _write(base / rel, content)
    return root


def test_toy_sibling_module_works_once_the_plugin_is_imported(tmp_path: Path) -> None:
    project = _make_cold_project(tmp_path)
    code = "import acme.feature_groups.plugin, acme.feature_groups.needs_sibling"
    result = subprocess.run([sys.executable, "-c", code], cwd=project, capture_output=True, text=True, timeout=60)  # nosec

    assert result.returncode == 0, result.stderr


_BAD_BYTES = "import sys\nsys.stderr.buffer.write(b'\\xff bad bytes\\n')\nsys.stderr.flush()\nraise SystemExit(1)\n"


@pytest.mark.parametrize("src_layout", [False, True], ids=["flat", "src"])
def test_cold_import_reports_sibling_dependent_module(tmp_path: Path, src_layout: bool) -> None:
    project = _make_cold_project(tmp_path, {"acme/badbytes.py": _BAD_BYTES}, src_layout=src_layout)

    with pytest.raises(AssertionError) as excinfo:
        assert_cold_imports(project, "acme")

    message = str(excinfo.value)
    assert "acme.feature_groups.needs_sibling" in message
    assert "AttributeError" in message
    assert "acme.badbytes" in message
    assert "acme.clean" not in message
    assert "acme.feature_groups.plugin" not in message


def test_cold_import_all_clean_passes(tmp_path: Path) -> None:
    boom = "raise RuntimeError('x')\n"
    extra = {"acme/.ipynb_checkpoints/logic-checkpoint.py": boom, "acme/feature_groups/my-script.py": boom}
    project = _make_cold_project(tmp_path, extra, clean=True)

    assert_cold_imports(project, "acme")

    assert not list(project.rglob("__pycache__"))


def test_cold_import_timeout_is_reported(tmp_path: Path) -> None:
    project = _make_cold_project(tmp_path, {"acme/slow.py": "import time\ntime.sleep(60)\n"}, clean=True)

    with pytest.raises(AssertionError) as excinfo:
        assert_cold_imports(project, "acme", timeout=5)

    message = str(excinfo.value)
    assert "acme.slow" in message
    assert "timeout" in message.lower() or "timed out" in message.lower()
    assert "acme.clean" not in message


def test_cold_import_ignores_a_shadow_package_in_the_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _make_cold_project(tmp_path / "real", clean=True)
    shadow = tmp_path / "shadow"
    _write(shadow / "acme/__init__.py", "raise RuntimeError('shadow')\n")
    monkeypatch.chdir(shadow)

    assert_cold_imports(project, "acme")


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

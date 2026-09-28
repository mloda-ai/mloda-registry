"""The mypy gate must skip local setuptools build/ and dist/ output at any depth, yet keep checking
lookalike names such as rebuild/, build_utils/, distutils_like/, and a module file build.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from mypy.config_parser import parse_config_file
from mypy.find_sources import create_source_list
from mypy.options import Options

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

_ARTIFACTS = ["mloda/pkg/build/lib/mloda/pkg/real.py", "build/lib/top.py", "dist/x/mod.py"]
_CHECKED = [
    "mloda/pkg/real.py",
    "mloda/rebuild/mod.py",
    "mloda/build_utils/mod.py",
    "mloda/pkg/distutils_like/mod.py",
    "mloda/tools/build.py",
]


def test_mypy_skips_build_output_but_checks_lookalikes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Build/dist copies are excluded; the real file and lookalikes are still collected."""
    # parse_config_file sets MYPY_CONFIG_FILE_DIR process-wide; register it so teardown restores it.
    monkeypatch.setenv("MYPY_CONFIG_FILE_DIR", str(_REPO_ROOT))
    options = Options()
    parse_config_file(options, lambda: None, str(_PYPROJECT))

    for rel in _ARTIFACTS + _CHECKED:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x: int = 1\n")

    monkeypatch.chdir(tmp_path)
    sources = create_source_list(["."], options)
    found = {Path(source.path).as_posix().removeprefix("./") for source in sources if source.path}

    leaked = sorted(found & set(_ARTIFACTS))
    assert not leaked, f"mypy would check build/dist output: {leaked}"
    missing = sorted(set(_CHECKED) - found)
    assert not missing, f"mypy skips real files or lookalikes: {missing}"

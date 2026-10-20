"""Shared clean-venv install, probe and local wheel-build helpers for the verify scripts."""

from __future__ import annotations

import runpy
import subprocess  # nosec
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent

# Not a plain import: this script is also loaded by file path in tests, where scripts/ is not on sys.path.
_load_sibling: Callable[[str], ModuleType] = runpy.run_path(str(REPO_ROOT / "scripts" / "script_loader.py"))[
    "load_sibling"
]
venv_python: Callable[[Path], Path] = _load_sibling("verify_build_floor").venv_python
find_wheels: Callable[[Path, str], list[Path]] = _load_sibling("published_packages").find_wheels


def venv_dir(tmpdir: str) -> Path:
    """The venv location inside a verify temp dir."""
    return Path(tmpdir) / "venv"


def install(requirements: list[str], tmpdir: str) -> str | None:
    """Create a fresh venv at <tmpdir>/venv and install the requirements; the error message, else None."""
    venv = venv_dir(tmpdir)
    commands = [
        ["uv", "venv", "--python", sys.executable, str(venv)],
        ["uv", "pip", "install", "--python", str(venv_python(venv)), *requirements],
    ]
    for command in commands:
        # cwd is the temp dir, so the checkout cannot shadow the installed packages.
        result = subprocess.run(command, capture_output=True, text=True, cwd=tmpdir)  # nosec
        if result.returncode != 0:
            return f"{' '.join(command)} failed:\n{(result.stderr or result.stdout)[-500:]}"
    return None


def install_and_probe(requirements: list[str], probe: str) -> str | None:
    """Install the requirements into a clean venv and run the probe in it; the error message, else None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        error = install(requirements, tmpdir)
        if error is not None:
            return error
        command = [str(venv_python(venv_dir(tmpdir))), "-c", probe]
        result = subprocess.run(command, capture_output=True, text=True, cwd=tmpdir)  # nosec
        if result.returncode != 0:
            return f"probe failed:\n{(result.stderr or result.stdout)[-500:]}"
    return None


def build_wheel(name: str, out_dir: Path) -> tuple[Path | None, str | None]:
    """Build one package's wheel into out_dir; (wheel, None) or (None, error message)."""
    command = ["uv", "build", "--package", name, "--out-dir", str(out_dir), "--wheel", "--no-build-isolation"]
    result = subprocess.run(command, capture_output=True, text=True, cwd=REPO_ROOT)  # nosec
    if result.returncode != 0:
        return None, f"{name}: build failed:\n{(result.stderr or result.stdout)[-500:]}"
    wheels = find_wheels(out_dir, name)
    if not wheels:
        return None, f"{name}: no wheel produced (no wheel in the out-dir carries this distribution)"
    if len(wheels) > 1:
        return None, f"{name}: ambiguous wheels in the out-dir ({', '.join(w.name for w in wheels)})"
    return wheels[0], None

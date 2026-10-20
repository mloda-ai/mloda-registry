"""Packaging contract helpers: every module ships in the wheel and sdist, and imports cold."""

from __future__ import annotations

import os
import shutil
import subprocess  # nosec
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_SKIP_DIRS = {"tests", "test"}
_SKIP_FILES = {"conftest.py", "__main__.py"}
_ROOT_ONLY_IGNORE = {"build", "dist", ".venv", "venv", ".git", ".tox"}
_ANY_DEPTH_IGNORE = shutil.ignore_patterns("__pycache__", "*.egg-info", ".*_cache")


def _copy_ignore(project_dir: Path) -> Callable[[str, list[str]], set[str]]:
    """Ignore callable: build artifacts only at the project root, caches at any depth."""

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored = set(_ANY_DEPTH_IGNORE(directory, names))
        if Path(directory) == project_dir:
            ignored |= _ROOT_ONLY_IGNORE.intersection(names)
        return ignored

    return ignore


def _normalize_exclude(exclude: Iterable[str]) -> tuple[str, ...]:
    return (exclude,) if isinstance(exclude, str) else tuple(exclude)


def _source_root(project_dir: Path, package: str) -> Path:
    """Directory holding the package: the project dir (flat) or its ``src`` subdir."""
    rel = Path(*package.split("."))
    for root in (project_dir, project_dir / "src"):
        if (root / rel).is_dir():
            return root
    raise ValueError(f"package {package!r} not found in {project_dir} or {project_dir / 'src'}")


def _is_excluded(name: str, exclude: Iterable[str]) -> bool:
    return any(name == ex or name.startswith(ex + ".") for ex in exclude)


def _discover_modules(project_dir: Path, package: str, exclude: Iterable[str]) -> tuple[Path, list[str]]:
    root = _source_root(project_dir, package)
    names: set[str] = set()
    package_dir = root / Path(*package.split("."))
    for path in package_dir.rglob("*.py"):
        rel = path.relative_to(root)
        if any(part in _SKIP_DIRS for part in path.relative_to(package_dir).parts[:-1]):
            continue
        if path.name in _SKIP_FILES or path.name.startswith("test_"):
            continue
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        names.add(".".join(parts))
    modules = sorted(n for n in names if not _is_excluded(n, exclude))
    if not modules:
        raise AssertionError(f"no modules were found for package {package!r} in {project_dir}")
    return root, modules


def _modules_in(paths: Iterable[str]) -> set[str]:
    names: set[str] = set()
    for raw in paths:
        if not raw.endswith(".py"):
            continue
        parts = list(Path(raw).with_suffix("").parts)
        if parts and parts[0] == "src":
            parts = parts[1:]
        if parts and parts[-1] == "__init__":
            parts.pop()
        if parts:
            names.add(".".join(parts))
    return names


def _find_uv() -> str:
    try:
        import uv

        return str(uv.find_uv_bin())
    except Exception:  # ImportError, or find_uv_bin() failing to locate the binary
        found = shutil.which("uv")
        if found is None:
            raise RuntimeError("uv is required to build distributions but was not found; install it") from None
        return found


def assert_distributions_complete(
    project_dir: Path,
    package: str,
    *,
    exclude: Iterable[str] = (),
    build_args: Iterable[str] = (),
    timeout: float = 600,
) -> None:
    """Build the wheel and sdist from a temp copy and assert both contain every module of ``package``.

    The project dir must be self-contained (flat or ``src/`` layout). ``.git`` is not copied, so VCS-derived
    versions are unavailable.
    """
    project_dir = Path(project_dir)
    exclude = _normalize_exclude(exclude)
    _, modules = _discover_modules(project_dir, package, exclude)
    with tempfile.TemporaryDirectory() as tmp:
        copy_dir = Path(tmp) / "project"
        out_dir = Path(tmp) / "dist"
        shutil.copytree(project_dir, copy_dir, ignore=_copy_ignore(project_dir))
        cmd = [_find_uv(), "build", "--python", sys.executable, "--out-dir", str(out_dir), str(copy_dir), *build_args]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)  # nosec B603
        except subprocess.TimeoutExpired:
            raise AssertionError(f"uv build timed out after {timeout}s") from None
        if result.returncode != 0:
            raise AssertionError(f"uv build failed ({result.returncode}):\n{result.stderr}")

        found: dict[str, set[str]] = {}
        for wheel in out_dir.glob("*.whl"):
            with zipfile.ZipFile(wheel) as zf:
                found.setdefault("wheel", set()).update(_modules_in(zf.namelist()))
        for sdist in out_dir.glob("*.tar.gz"):
            with tarfile.open(sdist) as tf:
                found.setdefault("sdist", set()).update(
                    _modules_in(n.split("/", 1)[1] for n in tf.getnames() if "/" in n)
                )

    problems = []
    for kind in ("wheel", "sdist"):
        if kind not in found:
            problems.append(f"{kind}: no artifact was built")
            continue
        missing = [m for m in modules if m not in found[kind]]
        if missing:
            problems.append(f"{kind} is missing modules:\n  " + "\n  ".join(missing))
    if problems:
        raise AssertionError("\n".join(problems))


def _cold_import(module: str, root: Path, timeout: float) -> str | None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(root), env.get("PYTHONPATH", "")) if p)
    try:
        result = subprocess.run(  # nosec B603
            [sys.executable, "-c", "import importlib, sys; importlib.import_module(sys.argv[1])", module],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=root,
        )
    except subprocess.TimeoutExpired:
        return f"timed out after {timeout}s"
    if result.returncode == 0:
        return None
    lines = [line for line in result.stderr.splitlines() if line.strip()]
    return lines[-1] if lines else f"exit code {result.returncode}"


def assert_cold_imports(project_dir: Path, package: str, *, exclude: Iterable[str] = (), timeout: float = 60) -> None:
    """Assert every module of ``package`` imports in a fresh interpreter.

    Modules are imported from the source tree, not from a built wheel.
    """
    root, modules = _discover_modules(Path(project_dir), package, _normalize_exclude(exclude))
    with ThreadPoolExecutor(max_workers=min(4, len(modules) or 1)) as pool:
        results = list(pool.map(lambda m: (m, _cold_import(m, root, timeout)), modules))
    failures = sorted((m, err) for m, err in results if err is not None)
    if failures:
        raise AssertionError("modules failing a cold import:\n" + "\n".join(f"  {m}: {err}" for m, err in failures))

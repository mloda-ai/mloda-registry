#!/usr/bin/env python3
"""Prove each data-operations backend extra is enough on its own, installed into a clean venv from local wheels.

The probe checks is_available(), not imports: core guards its optional backend imports. One venv per extra is
safe because test_backend_optional_dependencies.py forces every leaf's same-named extra to the same packages.

Run: python scripts/verify_backend_extras.py
Exit code: 1 if any wheel fails to build or any extra is insufficient, 0 otherwise.
"""

from __future__ import annotations

import ast
import concurrent.futures
import os
import runpy
import subprocess  # nosec
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# Not a plain import: this script is also loaded by file path in tests, where scripts/ is not on sys.path.
_load_sibling: Callable[[str], ModuleType] = runpy.run_path(str(REPO_ROOT / "scripts" / "script_loader.py"))[
    "load_sibling"
]

BASE = "mloda-community-data-operations"
MAX_WORKERS: int = _load_sibling("verify_independent_installs").MAX_WORKERS

# Longer, more specific prefixes first.
BACKEND_PREFIXES = [
    ("polars_lazy_", "polars"),
    ("python_dict_", "python_dict"),
    ("pyarrow_", "pyarrow"),
    ("duckdb_", "duckdb"),
    ("pandas_", "pandas"),
    ("sqlite_", "sqlite"),
]

# Packages a python_dict-only install must not pull in, else the check is vacuous.
ABSENT_IN_BARE_INSTALL = ["pyarrow", "polars", "pandas", "duckdb"]

LeafClasses = list[tuple[str, list[tuple[str, str]]]]


def backend_for_module(module: str) -> str:
    """Backend named by the module's prefix."""
    last = module.rsplit(".", 1)[-1]
    for prefix, backend in BACKEND_PREFIXES:
        if last.startswith(prefix):
            return backend
    raise ValueError(f"module {module!r} has no known backend prefix {[p for p, _ in BACKEND_PREFIXES]}")


def leaf_packages(packages: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Packages under the base path that ship a manifest.py."""
    if BASE not in packages:
        return []
    prefix = str(packages[BASE]["path"]).rstrip("/") + "/"
    return [
        (name, entry)
        for name, entry in packages.items()
        if str(entry["path"]).startswith(prefix) and (REPO_ROOT / str(entry["path"]) / "manifest.py").is_file()
    ]


def manifest_specs(manifest: Path) -> list[tuple[str, str]]:
    """The literal (module, class) pairs of a manifest's load_plugin_classes call; ValueError otherwise."""
    for node in ast.walk(ast.parse(manifest.read_text())):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "load_plugin_classes"):
            continue
        if len(node.args) < 2 or not isinstance(node.args[1], ast.List):
            raise ValueError(f"{manifest}: load_plugin_classes needs a literal list of (module, class) pairs")
        specs: list[tuple[str, str]] = []
        for element in node.args[1].elts:
            if not (
                isinstance(element, ast.Tuple)
                and len(element.elts) == 2
                and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in element.elts)
            ):
                raise ValueError(f"{manifest}: every entry must be a literal (module, class) string pair")
            specs.append((element.elts[0].value, element.elts[1].value))  # type: ignore[attr-defined]
        return specs
    raise ValueError(f"{manifest}: no load_plugin_classes call found")


def backend_jobs(packages: dict[str, dict[str, Any]]) -> dict[str, LeafClasses]:
    """Per backend, each leaf declaring that extra with its manifest classes of that backend."""
    jobs: dict[str, LeafClasses] = {backend: [] for _, backend in BACKEND_PREFIXES}
    for name, entry in leaf_packages(packages):
        path = str(entry["path"]).rstrip("/")
        by_backend: dict[str, list[tuple[str, str]]] = {}
        for module, cls in manifest_specs(REPO_ROOT / path / "manifest.py"):
            full = path.replace("/", ".") + "." + module
            by_backend.setdefault(backend_for_module(module), []).append((full, cls))
        for backend in jobs:
            if backend in entry.get("optional_dependencies", {}):
                jobs[backend].append((name, by_backend.get(backend, [])))
    empty = [backend for backend, entries in jobs.items() if not entries]
    if empty:
        raise ValueError(f"no leaf declares an extra for backend(s) {empty}")
    return jobs


def install_requirements(backend: str, leaves: list[str], wheels: dict[str, Path]) -> list[str]:
    reqs = [f"{leaf}[{backend}] @ {wheels[leaf].resolve().as_uri()}" for leaf in leaves]
    reqs.append(f"{BASE} @ {wheels[BASE].resolve().as_uri()}")
    return reqs


def probe_source(classes: list[tuple[str, str]], absent: list[str]) -> str:
    """Probe program: every class's compute frameworks are available and `absent` is not importable."""
    return f"""import importlib
import importlib.util
import sys

failures = []
for module, name in {classes!r}:
    cls = getattr(importlib.import_module(module), name)
    for framework in cls.compute_framework_rule():
        if not framework.is_available():
            failures.append(f"{{name}}: {{framework.__name__}} is not available")
for module in {absent!r}:
    if importlib.util.find_spec(module) is not None:
        failures.append(f"{{module}} is importable but must be absent")
if failures:
    print("\\n".join(failures), file=sys.stderr)
    sys.exit(1)
"""


def build_wheels(names: list[str], out_dir: Path) -> tuple[dict[str, Path], list[str]]:
    """Build one wheel per package."""
    find_wheels: Callable[[Path, str], list[Path]] = _load_sibling("published_packages").find_wheels
    wheels: dict[str, Path] = {}
    errors: list[str] = []
    for name in names:
        command = ["uv", "build", "--package", name, "--out-dir", str(out_dir), "--wheel", "--no-build-isolation"]
        result = subprocess.run(command, capture_output=True, text=True, cwd=REPO_ROOT)  # nosec
        if result.returncode != 0:
            errors.append(f"{name}: build failed:\n{result.stderr[-500:]}")
            continue
        found = find_wheels(out_dir, name)
        if len(found) != 1:
            errors.append(f"{name}: expected exactly one wheel, got {[w.name for w in found]}")
            continue
        wheels[name] = found[0]
    return wheels, errors


def verify_extra(backend: str, leaves_and_classes: LeafClasses, wheels: dict[str, Path]) -> str | None:
    """Install the backend's leaves into a clean venv and probe them; error message or None."""
    venv_python: Callable[[Path], Path] = _load_sibling("verify_build_floor").venv_python
    classes = [pair for _, pairs in leaves_and_classes for pair in pairs]
    absent = list(ABSENT_IN_BARE_INSTALL) if backend == "python_dict" else []
    requirements = install_requirements(backend, [leaf for leaf, _ in leaves_and_classes], wheels)
    with tempfile.TemporaryDirectory() as tmpdir:
        python = str(venv_python(Path(tmpdir) / "venv"))
        commands = [
            ["uv", "venv", "--python", sys.executable, str(Path(tmpdir) / "venv")],
            ["uv", "pip", "install", "--python", python, *requirements],
            [python, "-c", probe_source(classes, absent)],
        ]
        for command in commands:
            # cwd is the temp dir, so the checkout cannot shadow the installed packages.
            result = subprocess.run(command, capture_output=True, text=True, cwd=tmpdir)  # nosec
            if result.returncode != 0:
                return f"extra {backend}: {' '.join(command[:4])} failed:\n{(result.stderr or result.stdout)[-500:]}"
    return None


def main() -> int:
    # The reused helpers resolve paths relative to cwd.
    os.chdir(REPO_ROOT)
    builds = _load_sibling("verify_builds")
    cleanup_build_dirs: Callable[[], int] = builds.cleanup_build_dirs
    cleanup_egg_info: Callable[[], int] = builds.cleanup_egg_info

    load_packages_config: Callable[[], dict[str, dict[str, Any]]] = _load_sibling(
        "published_packages"
    ).load_packages_config
    packages = load_packages_config()
    jobs = backend_jobs(packages)
    names = [BASE, *dict.fromkeys(leaf for entries in jobs.values() for leaf, _ in entries)]

    errors: list[str] = []
    cleanup_build_dirs()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"\nBuilding {len(names)} wheels...")
            wheels, build_errors = build_wheels(names, Path(tmpdir))
            errors.extend(build_errors)
            if not build_errors:
                backends = list(jobs)
                workers = min(len(backends), MAX_WORKERS)
                print(f"Verifying {len(backends)} backend extras, {workers} at a time...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    results = executor.map(lambda b: verify_extra(b, jobs[b], wheels), backends)
                    for backend, error in zip(backends, results):
                        if error is None:
                            print(f"  ✓ {backend}: extra is sufficient")
                        else:
                            print(f"  ✗ {backend}: failed")
                            errors.append(error)
    finally:
        cleanup_egg_info()
        cleanup_build_dirs()

    if errors:
        print("\n❌ Errors:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("\n✅ every backend extra is sufficient on its own")
    return 0


if __name__ == "__main__":
    sys.exit(main())

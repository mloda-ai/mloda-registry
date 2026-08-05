#!/usr/bin/env python3
"""Install each internal extra and prove it gates exactly its members' imports.

Internal extras are the non-dev extras of published packages whose members are configured package
keys. ``{published_children}`` expands as the generator does; the shared default extras from
config/shared.toml declare only ``dev``, which is skipped, so they are never merged here.

Run: python scripts/verify_extras.py <version>
Exit code: 1 if any member imports without its extra or fails to import with it, 0 otherwise.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess  # nosec
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# The dev extra is tooling, never shipped code.
DEV_EXTRA = "dev"


def _load_sibling(name: str) -> ModuleType:
    """Load a sibling scripts/ module by file path, so an arbitrary cwd cannot break the import."""
    path = Path(__file__).resolve().parent / f"{name}.py"
    if not path.exists():
        raise ImportError(f"{path} is missing; {Path(__file__).name} derives its checks from it")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def internal_extra_members(packages: dict[str, dict[str, Any]]) -> list[tuple[str, str, list[str]]]:
    """(package, extra, members) per non-dev extra of a published package with configured members, in config order."""
    expand: Callable[[dict[str, Any], dict[str, dict[str, Any]]], dict[str, list[str]]] = _load_sibling(
        "generate_pyproject"
    ).expand_published_children
    entries: list[tuple[str, str, list[str]]] = []
    for pkg_name, pkg_config in packages.items():
        if pkg_config.get("published") is not True:
            continue
        expanded = expand(pkg_config, packages)
        for extra, deps in expanded.items():
            if extra == DEV_EXTRA:
                continue
            members = [dep for dep in deps if dep in packages]
            if members:
                entries.append((pkg_name, extra, members))
    return entries


def _install_and_probe(
    specifier: str,
    owner_modules: tuple[str, ...],
    expect_import: bool,
    member_modules: dict[str, str],
    tmpdir: str,
) -> list[str]:
    """Install one specifier into a fresh venv, probe the owner's surface, then check every member."""
    from verify_build_floor import venv_python

    venv = Path(tmpdir) / "venv"
    setup = [
        ["uv", "venv", "--python", sys.executable, str(venv)],
        ["uv", "pip", "install", "--python", str(venv_python(venv)), specifier],
    ]
    for command in setup:
        # cwd is the temp dir, so the checkout cannot shadow the installed packages.
        result = subprocess.run(command, capture_output=True, text=True, cwd=tmpdir)  # nosec
        if result.returncode != 0:
            return [f"{specifier}: {' '.join(command)} failed:\n{result.stderr[-500:]}"]

    errors: list[str] = []
    # The owning package itself must import with and without its extra.
    for module in owner_modules:
        command = [str(venv_python(venv)), "-c", f"import {module}"]
        result = subprocess.run(command, capture_output=True, text=True, cwd=tmpdir)  # nosec
        if result.returncode != 0:
            errors.append(f"{specifier}: import {module} failed:\n{result.stderr[-500:]}")
        else:
            print(f"  ✓ base package OK: {module}")
    for member, module in member_modules.items():
        command = [str(venv_python(venv)), "-c", f"import {module}"]
        result = subprocess.run(command, capture_output=True, text=True, cwd=tmpdir)  # nosec
        if expect_import:
            if result.returncode == 0:
                print(f"  ✓ {member}: imports")
            else:
                errors.append(f"{specifier}: import {module} failed:\n{result.stderr[-500:]}")
        elif result.returncode == 0:
            errors.append(f"{specifier}: {member} ({module}) imports without the extra")
        elif "ModuleNotFoundError" in result.stderr and f"'{module}'" in result.stderr:
            # Only a ModuleNotFoundError naming the member proves the extra gates it; anything
            # else (a broken parent, a SyntaxError, a crashed interpreter) is a real failure.
            print(f"  ✓ {member}: correctly not installed")
        else:
            errors.append(
                f"{specifier}: import {module} failed, but not with ModuleNotFoundError for "
                f"{module}:\n{result.stderr[-500:]}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify each internal extra gates exactly its members' imports")
    parser.add_argument("version", nargs="?", default="", help="Released version to install every package at")
    args = parser.parse_args()

    # tox renders an empty argument when MLODA_REGISTRY_VERSION is unset.
    if not args.version.strip():
        parser.error("version must not be empty (is MLODA_REGISTRY_VERSION set?)")

    # The reused published_packages helper resolves the config relative to cwd.
    os.chdir(REPO_ROOT)
    from published_packages import load_packages_config

    packages = load_packages_config()
    entries = internal_extra_members(packages)

    # An empty set would silently verify nothing.
    if not entries:
        print("❌ config/packages.toml declares no internal extras")
        return 1

    # The single derivation point for import surfaces lives in verify_published_imports.
    surface: Callable[[str], tuple[str, ...]] = _load_sibling("verify_published_imports").import_surface

    errors: list[str] = []
    for package, extra, members in entries:
        owner_modules = surface(str(packages[package]["path"]))
        member_modules = {member: str(packages[member]["path"]).replace("/", ".") for member in members}
        bare = (f"{package}=={args.version}", False)
        gated = (f"{package}[{extra}]=={args.version}", True)
        for specifier, expect_import in (bare, gated):
            print(f"\nInstalling {specifier}...")
            with tempfile.TemporaryDirectory() as tmpdir:
                errors.extend(_install_and_probe(specifier, owner_modules, expect_import, member_modules, tmpdir))

    if errors:
        print("\n❌ Errors:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"\n✅ every internal extra gates exactly its members at {args.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

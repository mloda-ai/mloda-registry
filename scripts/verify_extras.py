#!/usr/bin/env python3
"""Install each internal extra and prove it gates exactly its members' imports.

Internal extras are the non-dev extras of published packages whose members are configured package
keys. ``{published_children}`` expands as the generator does; the shared default extras from
config/shared.toml declare only ``dev``, which is skipped, so they are never merged here.

Jobs run concurrently through a bounded thread pool: one bare install per owning package (its
extras' members combined), then one gated install per extra.

Run: python scripts/verify_extras.py <version>
Exit code: 1 if any member imports without its extra or fails to import with it, 0 otherwise.
"""

from __future__ import annotations

import argparse
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

# The dev extra is tooling, never shipped code.
DEV_EXTRA = "dev"

# Not a plain import: this script is also loaded by file path in tests, where scripts/ is not on sys.path.
_load_sibling: Callable[[str], ModuleType] = runpy.run_path(str(REPO_ROOT / "scripts" / "script_loader.py"))[
    "load_sibling"
]


def internal_extra_members(packages: dict[str, dict[str, Any]]) -> list[tuple[str, str, list[str]]]:
    """(package, extra, members) per non-dev extra of a published package with configured members, in config
    order. A member is parsed from the requirement string (e.g. 'mloda-community-otel=={version}'), not
    required to be a bare package name."""
    gen = _load_sibling("generate_pyproject")
    expand: Callable[[dict[str, Any], dict[str, dict[str, Any]]], dict[str, list[str]]] = gen.expand_published_children
    sibling_name: Callable[[str, dict[str, dict[str, Any]], dict[str, str]], str | None] = gen.sibling_dependency_name
    normalize: Callable[[str], str] = gen.normalize_package_name

    # Built once, not per package: the normalized name lookup sibling_dependency_name would otherwise
    # rebuild on every call.
    configured = {normalize(name): name for name in packages}

    entries: list[tuple[str, str, list[str]]] = []
    for pkg_name, pkg_config in packages.items():
        if pkg_config.get("published") is not True:
            continue
        expanded = expand(pkg_config, packages)
        for extra, deps in expanded.items():
            if extra == DEV_EXTRA:
                continue
            # In list order (unlike sibling_dependency_names, which sorts).
            members = [name for name in (sibling_name(dep, packages, configured) for dep in deps) if name is not None]
            if members:
                entries.append((pkg_name, extra, members))
    return entries


def verification_jobs(
    entries: list[tuple[str, str, list[str]]], version: str
) -> list[tuple[str, str, bool, list[str]]]:
    """(owner, specifier, expect_import, members) install jobs, deduplicated per owning package: one bare
    job (the union of that package's extras' members, first-appearance order), then one gated job per
    extra, both in entry order. A package installs bare exactly once, however many extras it has. Each
    job carries its owning package, so a caller never has to re-derive it by parsing the specifier."""
    bare_members: dict[str, list[str]] = {}
    gated_jobs: dict[str, list[tuple[str, str, list[str]]]] = {}
    for package, extra, members in entries:
        seen = bare_members.setdefault(package, [])
        for member in members:
            if member not in seen:
                seen.append(member)
        gated_jobs.setdefault(package, []).append((package, extra, members))

    jobs: list[tuple[str, str, bool, list[str]]] = []
    for package, members in bare_members.items():
        jobs.append((package, f"{package}=={version}", False, members))
        for _, extra, extra_members in gated_jobs[package]:
            jobs.append((package, f"{package}[{extra}]=={version}", True, extra_members))
    return jobs


def _install_and_probe(
    specifier: str,
    owner_modules: tuple[str, ...],
    expect_import: bool,
    member_modules: dict[str, str],
    tmpdir: str,
) -> tuple[list[str], list[str]]:
    """Install one specifier into a fresh venv, probe the owner's surface, then check every member.

    Prints nothing, so callers running several of these concurrently control all output themselves.
    Returns (messages, errors).
    """
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
            return [], [f"{specifier}: {' '.join(command)} failed:\n{result.stderr[-500:]}"]

    messages: list[str] = []
    errors: list[str] = []
    # The owning package itself must import with and without its extra.
    for module in owner_modules:
        command = [str(venv_python(venv)), "-c", f"import {module}"]
        result = subprocess.run(command, capture_output=True, text=True, cwd=tmpdir)  # nosec
        if result.returncode != 0:
            errors.append(f"{specifier}: import {module} failed:\n{result.stderr[-500:]}")
        else:
            messages.append(f"  ✓ base package OK: {module}")
    for member, module in member_modules.items():
        command = [str(venv_python(venv)), "-c", f"import {module}"]
        result = subprocess.run(command, capture_output=True, text=True, cwd=tmpdir)  # nosec
        if expect_import:
            if result.returncode == 0:
                messages.append(f"  ✓ {member}: imports")
            else:
                errors.append(f"{specifier}: import {module} failed:\n{result.stderr[-500:]}")
        elif result.returncode == 0:
            errors.append(f"{specifier}: {member} ({module}) imports without the extra")
        elif "ModuleNotFoundError" in result.stderr and f"'{module}'" in result.stderr:
            # Only a ModuleNotFoundError naming the member proves the extra gates it; anything
            # else (a broken parent, a SyntaxError, a crashed interpreter) is a real failure.
            messages.append(f"  ✓ {member}: correctly not installed")
        else:
            errors.append(
                f"{specifier}: import {module} failed, but not with ModuleNotFoundError for "
                f"{module}:\n{result.stderr[-500:]}"
            )
    return messages, errors


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
    max_workers: int = _load_sibling("verify_independent_installs").MAX_WORKERS

    jobs = verification_jobs(entries, args.version)
    workers = min(len(jobs), max_workers)
    print(f"\nInstalling {len(jobs)} jobs at {args.version}, {workers} at a time...")

    def _run(job: tuple[str, str, bool, list[str]]) -> tuple[list[str], list[str]]:
        package, specifier, expect_import, members = job
        owner_modules = surface(str(packages[package]["path"]))
        member_modules = {member: str(packages[member]["path"]).replace("/", ".") for member in members}
        with tempfile.TemporaryDirectory() as tmpdir:
            return _install_and_probe(specifier, owner_modules, expect_import, member_modules, tmpdir)

    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        # map() preserves job order, so per-job output stays deterministic.
        for job, (messages, job_errors) in zip(jobs, executor.map(_run, jobs)):
            print(f"\nInstalling {job[1]}...")
            for message in messages:
                print(message)
            errors.extend(job_errors)

    if errors:
        print("\n❌ Errors:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"\n✅ every internal extra gates exactly its members at {args.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

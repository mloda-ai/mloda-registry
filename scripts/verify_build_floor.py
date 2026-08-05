#!/usr/bin/env python3
"""Build at the setuptools floor config/shared.toml declares, in an environment pinned to exactly it.

Run: python scripts/verify_build_floor.py
Exit code: 1 if the floor cannot build every license form the generator emits, 0 otherwise.
"""

from __future__ import annotations

import os
import re
import subprocess  # nosec
import sys
import tempfile
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

REPO_ROOT = Path(__file__).resolve().parent.parent
SHARED_CONFIG = REPO_ROOT / "config" / "shared.toml"

# Only the lower bound matters: "setuptools>=77.0.1" and "setuptools >= 77.0.1, < 90" both floor at 77.0.1.
SETUPTOOLS_FLOOR_RE = re.compile(r"^setuptools\s*>=\s*([^\s,;]+)")

# The path prefix the generator reads as proprietary, which emits a LicenseRef instead of an SPDX id.
PROPRIETARY_PATH_PREFIX = "mloda/enterprise"

LICENSE_FORMS = ("apache", "proprietary")


def declared_setuptools_floor(requires: list[str] | None = None) -> str:
    """The setuptools lower bound of a [build-system].requires list, config/shared.toml's by default."""
    if requires is None:
        with open(SHARED_CONFIG, "rb") as f:
            requires = tomllib.load(f)["build-system"]["requires"]
    for entry in requires:
        match = SETUPTOOLS_FLOOR_RE.match(entry.strip())
        if match is not None:
            return match.group(1)
    raise ValueError(f"no setuptools lower bound in build-system requires: {requires!r}")


def license_form(path: str) -> str:
    """The license form the generator emits for a package path."""
    return "proprietary" if path.startswith(PROPRIETARY_PATH_PREFIX) else "apache"


def license_sample_packages(packages: dict[str, dict[str, Any]]) -> dict[str, str]:
    """First package name per license form; setuptools validates a LicenseRef on a different path than an SPDX id."""
    samples: dict[str, str] = {}
    for pkg_name, pkg_config in packages.items():
        samples.setdefault(license_form(str(pkg_config["path"])), pkg_name)
    return samples


def venv_python(venv: Path) -> Path:
    """Interpreter of a uv-created virtual environment."""
    return venv / "bin" / "python"


def create_floor_env(venv: Path, floor: str) -> str | None:
    """Create a venv holding exactly setuptools==floor; returns an error message, or None on success."""
    commands = [
        ["uv", "venv", "--python", sys.executable, str(venv)],
        ["uv", "pip", "install", "--python", str(venv_python(venv)), f"setuptools=={floor}"],
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)  # nosec
        if result.returncode != 0:
            return f"{' '.join(command)} failed:\n{result.stderr[-500:]}"
    return None


def build_wheel_with(python: Path, pkg_dir: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
    """Build one package by calling the backend directly, so only the setuptools in ``python`` is used."""
    code = f"from setuptools import build_meta; build_meta.build_wheel({str(out_dir)!r})"
    return subprocess.run([str(python), "-c", code], cwd=pkg_dir, capture_output=True, text=True)  # nosec


def main() -> int:
    # The reused verify_builds helpers resolve the config and the build directories relative to cwd.
    os.chdir(REPO_ROOT)
    from verify_builds import (
        cleanup_build_dirs,
        cleanup_egg_info,
        find_wheels,
        get_wheel_metadata,
        load_packages_config,
    )

    floor = declared_setuptools_floor()
    print(f"Verifying the declared build floor: setuptools=={floor}")

    packages = load_packages_config()
    samples = license_sample_packages(packages)
    missing = [form for form in LICENSE_FORMS if form not in samples]
    if missing:
        print(f"\n❌ config/packages.toml declares no package for license form(s): {missing}")
        return 1

    errors: list[str] = []
    cleanup_build_dirs()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            venv = Path(tmpdir) / "venv"
            out_dir = Path(tmpdir) / "wheels"
            out_dir.mkdir()

            setup_error = create_floor_env(venv, floor)
            if setup_error is not None:
                print(f"\n❌ {setup_error}")
                return 1

            for form in LICENSE_FORMS:
                pkg_name = samples[form]
                print(f"\nBuilding {pkg_name} ({form} license) at setuptools=={floor}...")
                result = build_wheel_with(venv_python(venv), REPO_ROOT / packages[pkg_name]["path"], out_dir)
                if result.returncode != 0:
                    errors.append(f"{pkg_name}: build failed at setuptools=={floor}\n{result.stderr[-500:]}")
                    continue

                wheels = find_wheels(out_dir, pkg_name)
                if len(wheels) != 1:
                    errors.append(f"{pkg_name}: expected exactly one wheel, got {[wheel.name for wheel in wheels]}")
                    continue

                # A dropped license field would still build, so the metadata has to carry the form back.
                if "License-Expression:" not in get_wheel_metadata(wheels[0]):
                    errors.append(f"{pkg_name}: {wheels[0].name} METADATA has no 'License-Expression:' line")
                    continue

                print(f"  ✓ {wheels[0].name}")
    finally:
        cleanup_egg_info()
        cleanup_build_dirs()

    if errors:
        print("\n❌ Errors:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"\n✅ setuptools=={floor} builds every license form the generator emits")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Verify all workspace packages build correctly with consistent versions."""

from __future__ import annotations

import configparser
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

CONFIG_DIR = Path("config")
PACKAGES_CONFIG = CONFIG_DIR / "packages.toml"

# Valid manifest attributes for the mloda plugin entry-point groups (issue #271).
_VALID_ENTRY_POINT_ATTRS = {"FEATURE_GROUPS", "COMPUTE_FRAMEWORKS", "EXTENDERS"}


def load_packages_config() -> dict[str, dict[str, Any]]:
    """Return the raw [packages] table of config/packages.toml, in config order."""
    with open(PACKAGES_CONFIG, "rb") as f:
        data = tomllib.load(f)
    packages: dict[str, dict[str, Any]] = data.get("packages", {})
    return packages


def load_packages_from_config() -> list[tuple[str, str]]:
    """Return [(pkg_name, pyproject_path), ...] for every configured package, in config order."""
    return [(name, f"{cfg['path']}/pyproject.toml") for name, cfg in load_packages_config().items()]


PACKAGES = load_packages_from_config()


def escape_distribution_name(name: str) -> str:
    """Escape a distribution name the way a wheel filename carries it (PEP 427/503)."""
    return re.sub(r"[-_.]+", "_", name.lower())


def find_wheels(out_dir: Path, pkg_name: str) -> list[Path]:
    """Wheels in out_dir whose distribution segment is exactly pkg_name, so prefix siblings never match."""
    escaped = escape_distribution_name(pkg_name)
    return sorted((path for path in out_dir.glob("*.whl") if path.name.split("-")[0] == escaped), key=lambda p: p.name)


def namespaced_entry_point_error(group: str, name: str, value: str) -> str | None:
    """Return an error message if an entry-point target is not a valid namespaced manifest, else None."""
    if ":" not in value:
        return f"{group}: entry point {name!r} value {value!r} has no ':' manifest-attribute separator"

    module, _, attr = value.partition(":")

    if not (module.startswith("mloda.community.") or module.startswith("mloda.enterprise.")):
        return (
            f"{group}: entry point {name!r} module {module!r} is not under the "
            "'mloda.community.' or 'mloda.enterprise.' namespace"
        )

    if not module.endswith(".manifest"):
        return f"{group}: entry point {name!r} module {module!r} does not end with '.manifest'"

    if attr not in _VALID_ENTRY_POINT_ATTRS:
        return f"{group}: entry point {name!r} attribute {attr!r} is not one of {sorted(_VALID_ENTRY_POINT_ATTRS)}"

    return None


def get_versions_from_pyproject() -> dict[str, str]:
    """Read versions from all pyproject.toml files."""
    versions = {}
    for pkg_name, pyproject_path in PACKAGES:
        path = Path(pyproject_path)
        if path.exists():
            with open(path, "rb") as f:
                data = tomllib.load(f)
                versions[pkg_name] = data.get("project", {}).get("version", "unknown")
    return versions


def check_version_consistency() -> tuple[bool, str]:
    """Check all packages have the same version."""
    versions = get_versions_from_pyproject()
    unique_versions = set(versions.values())

    if len(unique_versions) == 1:
        return True, list(unique_versions)[0]

    print("❌ Version mismatch detected:")
    for pkg, ver in versions.items():
        print(f"  {pkg}: {ver}")
    return False, ""


def get_wheel_metadata(wheel_path: Path) -> str:
    """Extract METADATA content from wheel."""
    with zipfile.ZipFile(wheel_path) as zf:
        for name in zf.namelist():
            if name.endswith("METADATA"):
                return zf.read(name).decode()
    return ""


def get_wheel_top_level(wheel_path: Path) -> list[str]:
    """Extract top_level.txt content from wheel."""
    with zipfile.ZipFile(wheel_path) as zf:
        for name in zf.namelist():
            if name.endswith("top_level.txt"):
                return zf.read(name).decode().strip().split("\n")
    return []


def get_wheel_files(wheel_path: Path) -> list[str]:
    """Get list of files in wheel."""
    with zipfile.ZipFile(wheel_path) as zf:
        return zf.namelist()


def verify_wheel_version(wheel_path: Path, expected: str) -> bool:
    """Check wheel metadata contains expected version."""
    metadata = get_wheel_metadata(wheel_path)
    return f"Version: {expected}" in metadata


def verify_wheel_metadata(wheels: dict[str, Path]) -> list[str]:
    """Verify wheel metadata for namespace package compliance.

    Checks:
    - top_level.txt contains 'mloda'
    - No __init__.py in namespace directories (mloda/, mloda/community/, mloda/enterprise/)
    """
    errors = []
    namespace_dirs = ["mloda/", "mloda/community/", "mloda/enterprise/"]

    for pkg_name, wheel_path in wheels.items():
        # Check top_level.txt
        top_level = get_wheel_top_level(wheel_path)
        if not top_level:
            errors.append(f"{pkg_name}: missing top_level.txt")
        elif "mloda" not in top_level:
            errors.append(f"{pkg_name}: top_level.txt should contain 'mloda', found: {top_level}")

        # Check no __init__.py in namespace directories (PEP 420 compliance)
        files = get_wheel_files(wheel_path)
        for ns_dir in namespace_dirs:
            init_file = f"{ns_dir}__init__.py"
            if init_file in files:
                errors.append(f"{pkg_name}: contains {init_file} (breaks PEP 420 namespace package)")

    return errors


def get_wheel_entry_points(wheel_path: Path) -> dict[str, dict[str, str]]:
    """Read a wheel's dist-info entry_points.txt, returning only mloda.* groups.

    Returns {group: {name: value}} for every entry-point group whose name starts
    with ``mloda.``. Returns {} if the wheel has no entry_points.txt.
    """
    with zipfile.ZipFile(wheel_path) as zf:
        entry_points_name = None
        for name in zf.namelist():
            if name.endswith(".dist-info/entry_points.txt"):
                entry_points_name = name
                break
        if entry_points_name is None:
            return {}
        content = zf.read(entry_points_name).decode()

    # interpolation=None so a literal "%" in an entry-point value (e.g. a module
    # path or object ref) cannot raise InterpolationSyntaxError.
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(content)

    result: dict[str, dict[str, str]] = {}
    for group in parser.sections():
        if not group.startswith("mloda."):
            continue
        result[group] = {name: value for name, value in parser.items(group)}
    return result


def verify_entry_points(built_wheels: dict[str, Path]) -> list[str]:
    """Verify built wheels declare the mloda.* entry points from their generated pyproject.toml."""
    errors: list[str] = []
    pyproject_paths = dict(load_packages_from_config())

    for pkg_name, wheel_path in built_wheels.items():
        pyproject_path = pyproject_paths.get(pkg_name)
        if pyproject_path is None or not Path(pyproject_path).exists():
            continue

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        all_entry_points = data.get("project", {}).get("entry-points", {})
        expected = {group: mapping for group, mapping in all_entry_points.items() if group.startswith("mloda.")}

        actual = get_wheel_entry_points(wheel_path)

        if expected != actual:
            errors.append(f"{pkg_name}: entry points mismatch (expected {expected}, wheel has {actual})")

        for group, mapping in actual.items():
            for name, value in mapping.items():
                error = namespaced_entry_point_error(group, name, value)
                if error is not None:
                    errors.append(f"{pkg_name}: {error}")

    return errors


def verify_py_typed_markers(built_wheels: dict[str, Path]) -> list[str]:
    """Verify wheels of packages flagged ``py_typed = true`` ship their PEP 561 marker."""
    errors: list[str] = []
    packages = load_packages_config()

    for pkg_name, wheel_path in built_wheels.items():
        pkg_config = packages.get(pkg_name)
        if pkg_config is None or not pkg_config.get("py_typed"):
            continue

        marker = f"{pkg_config['path']}/py.typed"
        if marker not in get_wheel_files(wheel_path):
            errors.append(f"{pkg_name}: wheel is missing {marker} (PEP 561 marker)")

    return errors


def verify_dependency_relationships(wheels: dict[str, Path]) -> list[str]:
    """Verify dependency relationships in built wheels.

    Checks:
    - mloda-community-example has 'all' extra with example-a and example-b
    - mloda-community-example-a depends on mloda-community-example
    - mloda-community-example-b depends on mloda-community-example

    Note: mloda-community and mloda-enterprise are bundled packages that include
    all sub-package code directly, so they don't have dependencies on sub-packages.
    """
    errors = []

    # Check mloda-community-example has 'all' extra
    if "mloda-community-example" in wheels:
        metadata = get_wheel_metadata(wheels["mloda-community-example"])
        if "Provides-Extra: all" not in metadata:
            errors.append("mloda-community-example: missing 'all' extra")
        if 'mloda-community-example-a; extra == "all"' not in metadata:
            errors.append("mloda-community-example: 'all' extra missing example-a")
        if 'mloda-community-example-b; extra == "all"' not in metadata:
            errors.append("mloda-community-example: 'all' extra missing example-b")

    # Check example-a depends on base
    if "mloda-community-example-a" in wheels:
        metadata = get_wheel_metadata(wheels["mloda-community-example-a"])
        if "mloda-community-example" not in metadata:
            errors.append("mloda-community-example-a: missing dependency on mloda-community-example")

    # Check example-b depends on base
    if "mloda-community-example-b" in wheels:
        metadata = get_wheel_metadata(wheels["mloda-community-example-b"])
        if "mloda-community-example" not in metadata:
            errors.append("mloda-community-example-b: missing dependency on mloda-community-example")

    return errors


def verify_pep420_source_compliance() -> list[str]:
    """Verify PEP 420 compliance in source tree.

    Checks that no __init__.py exists in namespace directories.
    """
    errors = []
    namespace_dirs = [
        Path("mloda"),
        Path("mloda/community"),
        Path("mloda/enterprise"),
    ]

    for ns_dir in namespace_dirs:
        init_file = ns_dir / "__init__.py"
        if init_file.exists():
            errors.append(f"Source tree contains {init_file} (breaks PEP 420 namespace package)")

    return errors


def cleanup_build_dirs() -> int:
    """Remove the build/ trees setuptools leaves behind; a stale tree is copied into the next wheel."""
    count = 0
    for pkg_config in load_packages_config().values():
        build_dir = Path(pkg_config["path"]) / "build"
        if build_dir.is_dir():
            shutil.rmtree(build_dir)
            count += 1
    return count


def cleanup_egg_info() -> int:
    """Remove all egg-info directories created by builds."""
    count = 0
    # Clean root level
    for egg_info in Path(".").glob("*.egg-info"):
        shutil.rmtree(egg_info)
        count += 1
    # Clean mloda/ level
    for egg_info in Path("mloda").glob("*.egg-info"):
        shutil.rmtree(egg_info)
        count += 1
    # Clean any nested mloda/mloda artifacts
    mloda_mloda = Path("mloda/mloda")
    if mloda_mloda.exists():
        shutil.rmtree(mloda_mloda)
        count += 1
    return count


def main() -> int:
    # Stale build/ trees from a previous run would be copied into this run's wheels
    stale = cleanup_build_dirs()
    if stale:
        print(f"🧹 Removed {stale} stale build directory(ies)")

    # First check version consistency
    consistent, expected_version = check_version_consistency()
    if not consistent or not expected_version:
        return 1

    print(f"All packages declare version: {expected_version}")

    with tempfile.TemporaryDirectory() as tmpdir:
        errors = []
        built_wheels: dict[str, Path] = {}

        for pkg_name, _ in PACKAGES:
            print(f"\nBuilding {pkg_name}...")
            result = subprocess.run(
                ["uv", "build", "--package", pkg_name, "--out-dir", tmpdir, "--wheel", "--no-build-isolation"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                errors.append(f"{pkg_name}: build failed\n{result.stderr[:500]}")
                continue

            # Find and verify wheel
            wheels = find_wheels(Path(tmpdir), pkg_name)
            if not wheels:
                errors.append(f"{pkg_name}: no wheel produced (no wheel in the out-dir carries this distribution)")
                continue
            if len(wheels) > 1:
                candidates = ", ".join(wheel.name for wheel in wheels)
                errors.append(f"{pkg_name}: ambiguous wheels in the out-dir ({candidates})")
                continue

            wheel = wheels[0]
            if not verify_wheel_version(wheel, expected_version):
                errors.append(f"{pkg_name}: version mismatch in wheel (expected {expected_version}, got {wheel.name})")
            else:
                print(f"  ✓ {wheel.name}")
                built_wheels[pkg_name] = wheel

        # Verify dependency relationships
        print("\nVerifying dependency relationships...")
        dep_errors = verify_dependency_relationships(built_wheels)
        if dep_errors:
            errors.extend(dep_errors)
        else:
            print("  ✓ package dependencies correct")

        # Verify wheel metadata (top_level.txt, namespace compliance)
        print("\nVerifying wheel metadata...")
        metadata_errors = verify_wheel_metadata(built_wheels)
        if metadata_errors:
            errors.extend(metadata_errors)
        else:
            print("  ✓ wheel metadata correct")

        # Verify mloda plugin entry points (issue #271)
        print("\nVerifying entry points...")
        entry_point_errors = verify_entry_points(built_wheels)
        if entry_point_errors:
            errors.extend(entry_point_errors)
        else:
            print("  ✓ entry points correct")

        # Verify PEP 561 py.typed markers
        print("\nVerifying py.typed markers...")
        py_typed_errors = verify_py_typed_markers(built_wheels)
        if py_typed_errors:
            errors.extend(py_typed_errors)
        else:
            print("  ✓ py.typed markers present")

    # Verify PEP 420 source compliance (outside temp dir context)
    print("\nVerifying PEP 420 source compliance...")
    pep420_errors = verify_pep420_source_compliance()
    if pep420_errors:
        errors.extend(pep420_errors)
    else:
        print("  ✓ PEP 420 namespace package structure correct")

    if errors:
        print("\n❌ Errors:")
        for e in errors:
            print(f"  - {e}")
        return 1

    # A leftover build/ tree makes the next pytest run collect phantom build.lib.* modules
    cleaned = cleanup_egg_info() + cleanup_build_dirs()
    if cleaned:
        print(f"\n🧹 Cleaned up {cleaned} build artifact(s)")

    print(f"\n✅ All {len(PACKAGES)} packages built with version {expected_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

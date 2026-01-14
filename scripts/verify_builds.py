#!/usr/bin/env python3
"""Verify all workspace packages build correctly with consistent versions."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found]

PACKAGES = [
    ("mloda-registry", "mloda/registry/pyproject.toml"),
    ("mloda-testing", "mloda/testing/pyproject.toml"),
    ("mloda-community", "mloda/community/pyproject.toml"),
    ("mloda-enterprise", "mloda/enterprise/pyproject.toml"),
    ("mloda-community-example", "mloda/community/feature_groups/example/pyproject.toml"),
    ("mloda-enterprise-example", "mloda/enterprise/feature_groups/example/pyproject.toml"),
]


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


def verify_wheel_version(wheel_path: Path, expected: str) -> bool:
    """Check wheel metadata contains expected version."""
    with zipfile.ZipFile(wheel_path) as zf:
        for name in zf.namelist():
            if name.endswith("METADATA"):
                metadata = zf.read(name).decode()
                return f"Version: {expected}" in metadata
    return False


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
    # First check version consistency
    consistent, expected_version = check_version_consistency()
    if not consistent or not expected_version:
        return 1

    print(f"All packages declare version: {expected_version}")

    with tempfile.TemporaryDirectory() as tmpdir:
        errors = []
        for pkg_name, _ in PACKAGES:
            print(f"\nBuilding {pkg_name}...")
            result = subprocess.run(
                ["uv", "build", "--package", pkg_name, "--out-dir", tmpdir],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                errors.append(f"{pkg_name}: build failed\n{result.stderr[:500]}")
                continue

            # Find and verify wheel
            wheels = list(Path(tmpdir).glob(f"{pkg_name.replace('-', '_')}*.whl"))
            if not wheels:
                errors.append(f"{pkg_name}: no wheel produced")
                continue

            if not verify_wheel_version(wheels[0], expected_version):
                errors.append(f"{pkg_name}: version mismatch in wheel (expected {expected_version})")
            else:
                print(f"  ✓ {wheels[0].name}")

    if errors:
        print("\n❌ Errors:")
        for e in errors:
            print(f"  - {e}")
        return 1

    # Clean up egg-info directories on success
    cleaned = cleanup_egg_info()
    if cleaned:
        print(f"\n🧹 Cleaned up {cleaned} build artifact(s)")

    print(f"\n✅ All {len(PACKAGES)} packages built with version {expected_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

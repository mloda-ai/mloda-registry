"""Backend extras declared by data-operation leaf distributions."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

import pytest

_PACKAGES_CONFIG = Path(__file__).resolve().parents[2] / "config" / "packages.toml"

_FULL_BACKEND_EXTRAS = {
    "pyarrow": ["pyarrow"],
    "sqlite": [],
    "python_dict": [],
    "duckdb": ["duckdb"],
    "polars": ["polars"],
    "pandas": ["pandas"],
    "all": ["pyarrow", "polars", "pandas", "duckdb"],
}

_FULL_BACKEND_PACKAGES = ["mloda-community-datetime", "mloda-community-string"]


def _packages() -> dict[str, dict[str, Any]]:
    with open(_PACKAGES_CONFIG, "rb") as f:
        data = tomllib.load(f)
    packages: dict[str, dict[str, Any]] = data["packages"]
    return packages


@pytest.mark.parametrize("package_name", _FULL_BACKEND_PACKAGES)
def test_leaf_package_declares_all_backend_extras(package_name: str) -> None:
    """Each shipped backend is independently installable through its distribution extra."""
    actual = _packages()[package_name].get("optional_dependencies", {})
    assert actual == _FULL_BACKEND_EXTRAS, (
        f"{package_name}: config/packages.toml must declare the full backend extras "
        f"{_FULL_BACKEND_EXTRAS!r}, got {actual!r}"
    )

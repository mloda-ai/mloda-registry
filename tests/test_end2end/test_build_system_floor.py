"""The declared setuptools build floor must cover the bare PEP 639 SPDX license string the generator
emits; setuptools rejects that form before 77.0.1. The floor is proven by ``tox -e verify-build-floor``,
which really builds at it; this module only guards the declaration against drift."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

import pytest

from tests.script_loader import load_script

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GEN_PATH = _REPO_ROOT / "scripts" / "generate_pyproject.py"
_FLOOR_PATH = _REPO_ROOT / "scripts" / "verify_build_floor.py"
_SHARED_CONFIG = _REPO_ROOT / "config" / "shared.toml"
_PACKAGES_CONFIG = _REPO_ROOT / "config" / "packages.toml"

# First setuptools release accepting a PEP 639 SPDX license string (77.0.0 was never published).
_PEP639_SETUPTOOLS = "77.0.1"

# One package per emitted license form; the generator infers proprietary from an mloda/enterprise path.
_LICENSE_SAMPLES = [("mloda-registry", "Apache-2.0"), ("mloda-enterprise", "LicenseRef-Proprietary")]

gen = load_script("generate_pyproject", _GEN_PATH)


def _declared_setuptools_floor() -> Callable[..., str]:
    """The floor parser scripts/verify_build_floor.py must expose, shared with tox -e verify-build-floor."""
    assert _FLOOR_PATH.exists(), f"{_FLOOR_PATH} is missing; it is what 'tox -e verify-build-floor' runs"
    module = load_script("verify_build_floor", _FLOOR_PATH)
    parser: Callable[..., str] | None = getattr(module, "declared_setuptools_floor", None)
    assert callable(parser), "verify_build_floor.declared_setuptools_floor must be a callable"
    return parser


def _load_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _version_tuple(version: str) -> tuple[int, ...]:
    """Comparable form of a numeric release version."""
    return tuple(int(part) for part in version.split("."))


def _generated_license(pkg_name: str) -> Any:
    """The ``[project].license`` value the generator emits for a package."""
    shared = _load_toml(_SHARED_CONFIG)
    packages = _load_toml(_PACKAGES_CONFIG)["packages"]
    content: str = gen.generate_pyproject(pkg_name, packages[pkg_name], shared, packages)
    return tomllib.loads(content).get("project", {}).get("license")


@pytest.mark.parametrize(("pkg_name", "expected_license"), _LICENSE_SAMPLES)
def test_build_floor_covers_the_emitted_license_form(pkg_name: str, expected_license: str) -> None:
    """Both emitted forms are bare SPDX strings, which setuptools accepts only from 77.0.1 on."""
    license_value = _generated_license(pkg_name)
    assert license_value == expected_license, (
        f"{pkg_name}: expected the PEP 639 SPDX string {expected_license!r}, got {license_value!r}; "
        "the setuptools floor asserted below exists for that form, revisit it"
    )

    floor = _declared_setuptools_floor()()
    assert _version_tuple(floor) >= _version_tuple(_PEP639_SETUPTOOLS), (
        f'config/shared.toml declares setuptools>={floor}, but the generator emits license = "{license_value}", '
        f"which setuptools accepts only from {_PEP639_SETUPTOOLS} on. Run 'tox -e verify-build-floor': that env "
        "builds at the declared floor and is what proves it, this assertion only guards the declaration."
    )


def test_declared_floor_defaults_to_the_shared_config() -> None:
    """With no argument the parser reads config/shared.toml, the single source of the floor."""
    floor = _declared_setuptools_floor()()
    requires = _load_toml(_SHARED_CONFIG)["build-system"]["requires"]
    assert any(f">={floor}" in entry for entry in requires), (
        f"declared_setuptools_floor() returned {floor!r}, absent from config/shared.toml requires {requires!r}"
    )


@pytest.mark.parametrize(
    ("requires", "expected"),
    [
        (["setuptools>=77.0.1"], "77.0.1"),
        (["setuptools>=77.0.1,<90"], "77.0.1"),
        (["setuptools >= 77.0.1, < 90"], "77.0.1"),
        (["wheel", "setuptools>=80.0"], "80.0"),
    ],
)
def test_declared_floor_reads_the_setuptools_lower_bound(requires: list[str], expected: str) -> None:
    """An upper bound alongside the floor is a legitimate pin, not a malformed config."""
    floor = _declared_setuptools_floor()(requires)
    assert floor == expected, f"parsing {requires!r} produced floor {floor!r}, expected {expected!r}"


def test_declared_floor_rejects_requires_without_setuptools() -> None:
    """A build-system block that lost its setuptools pin must fail loudly, not fall back to a default."""
    with pytest.raises(ValueError, match="setuptools"):
        _declared_setuptools_floor()(["wheel"])


def test_generated_pyprojects_carry_the_shared_build_floor() -> None:
    """A floor bump reaches wheels only once every generated pyproject.toml carries it."""
    parse_floor = _declared_setuptools_floor()
    shared_floor = parse_floor()
    packages = _load_toml(_PACKAGES_CONFIG)["packages"]

    checked = 0
    for pkg_name, pkg_config in packages.items():
        pyproject_path = _REPO_ROOT / pkg_config["path"] / "pyproject.toml"
        assert pyproject_path.exists(), f"{pyproject_path} is missing (run scripts/generate_pyproject.py)."
        floor = parse_floor(_load_toml(pyproject_path).get("build-system", {}).get("requires", []))
        assert floor == shared_floor, (
            f"{pkg_name}: generated {pyproject_path} requires setuptools>={floor}, but "
            f"config/shared.toml declares setuptools>={shared_floor} (run scripts/generate_pyproject.py)."
        )
        checked += 1

    assert checked > 0, "expected at least one generated pyproject.toml to declare a setuptools floor."

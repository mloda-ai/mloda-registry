"""The declared setuptools build floor must cover the bare PEP 639 SPDX license string the generator
emits; setuptools rejects that form before 77.0.1. The floor is proven by ``tox -e verify-build-floor``,
which really builds at it; this module only guards the declaration against drift. It also guards the
license metadata: per-package SPDX string, ``license-files``, and the committed license copies."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.script_loader import load_script
from tests.toml_loader import load_toml, loads_toml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GEN_PATH = _REPO_ROOT / "scripts" / "generate_pyproject.py"
_FLOOR_PATH = _REPO_ROOT / "scripts" / "verify_build_floor.py"
_SHARED_CONFIG = _REPO_ROOT / "config" / "shared.toml"
_PACKAGES_CONFIG = _REPO_ROOT / "config" / "packages.toml"

# First setuptools release accepting a PEP 639 SPDX license string (77.0.0 was never published).
_PEP639_SETUPTOOLS = "77.0.1"

_SPDX_BY_FORM = {"apache": "Apache-2.0", "proprietary": "LicenseRef-Proprietary"}
_LICENSE_FILES_BY_FORM = {"apache": ["LICENSE", "NOTICE"], "proprietary": ["LICENSE"]}
_LICENSE_SOURCES = ["LICENSE", "LICENSE-APACHE", "NOTICE.md", "mloda/enterprise/LICENSE"]
_LICENSE_HOLDER = "Tom Kaltofen (mloda)"
_PACKAGES = load_toml(_PACKAGES_CONFIG)["packages"]
_SHARED = load_toml(_SHARED_CONFIG)

gen = load_script("generate_pyproject", _GEN_PATH)


def _declared_setuptools_floor() -> Callable[..., str]:
    """The floor parser scripts/verify_build_floor.py must expose, shared with tox -e verify-build-floor."""
    assert _FLOOR_PATH.exists(), f"{_FLOOR_PATH} is missing; it is what 'tox -e verify-build-floor' runs"
    module = load_script("verify_build_floor", _FLOOR_PATH)
    parser: Callable[..., str] | None = getattr(module, "declared_setuptools_floor", None)
    assert callable(parser), "verify_build_floor.declared_setuptools_floor must be a callable"
    return parser


def _version_tuple(version: str) -> tuple[int, ...]:
    """Comparable form of a numeric release version."""
    return tuple(int(part) for part in version.split("."))


def _generated_project(pkg_name: str) -> dict[str, Any]:
    """The ``[project]`` table the generator emits for a package."""
    content: str = gen.generate_pyproject(pkg_name, _PACKAGES[pkg_name], _SHARED, _PACKAGES)
    project: dict[str, Any] = loads_toml(content).get("project", {})
    return project


@pytest.mark.parametrize("pkg_name", list(_PACKAGES))
def test_emitted_license_and_files_match_the_package_form(pkg_name: str) -> None:
    """The SPDX string and ``license-files`` are chosen by the package path (bare SPDX needs the floor below)."""
    form = gen.license_form(_PACKAGES[pkg_name]["path"])
    project = _generated_project(pkg_name)
    expected_license = _SPDX_BY_FORM[form]
    license_value = project.get("license")
    assert license_value == expected_license, (
        f"{pkg_name}: expected the PEP 639 SPDX string {expected_license!r}, got {license_value!r}; "
        "the setuptools floor test below exists for that form, revisit it"
    )
    license_files = project.get("license-files")
    assert license_files == _LICENSE_FILES_BY_FORM[form], f"{pkg_name}: got {license_files!r}"


def test_build_floor_covers_the_emitted_license_form() -> None:
    """Bare SPDX license strings are accepted by setuptools only from 77.0.1 on."""
    floor = _declared_setuptools_floor()()
    assert _version_tuple(floor) >= _version_tuple(_PEP639_SETUPTOOLS), (
        f"config/shared.toml declares setuptools>={floor}, but the generator emits the bare SPDX strings "
        "Apache-2.0 and LicenseRef-Proprietary, which setuptools accepts only from "
        f"{_PEP639_SETUPTOOLS} on. Run 'tox -e verify-build-floor': that env "
        "builds at the declared floor and is what proves it, this assertion only guards the declaration."
    )


def test_package_directories_carry_byte_equal_license_copies() -> None:
    """Every package directory holds each file its license-files names, equal to its source."""
    checked = 0
    for pkg_name, pkg_config in _PACKAGES.items():
        for name, source in gen.license_copies(pkg_config).items():
            copy_path = _REPO_ROOT / pkg_config["path"] / name
            assert copy_path.exists(), f"{pkg_name}: {copy_path} is missing (run scripts/generate_pyproject.py)."
            assert copy_path.read_bytes() == (_REPO_ROOT / source).read_bytes(), (
                f"{pkg_name}: {copy_path} differs from {source} (run scripts/generate_pyproject.py)."
            )
            checked += 1

    assert checked > 0, "expected at least one license copy to be checked."


@pytest.mark.parametrize("source", _LICENSE_SOURCES)
def test_license_sources_name_the_holder_and_avoid_dashes(source: str) -> None:
    """Each license source carries the copyright holder line and no em or en dash."""
    text = (_REPO_ROOT / source).read_text(encoding="utf-8")
    assert any("copyright" in line.lower() and _LICENSE_HOLDER in line for line in text.splitlines()), (
        f"{source}: no copyright line naming {_LICENSE_HOLDER!r}"
    )
    assert "\u2014" not in text and "\u2013" not in text, f"{source} contains an em or en dash"


def test_declared_floor_defaults_to_the_shared_config() -> None:
    """With no argument the parser reads config/shared.toml, the single source of the floor."""
    floor = _declared_setuptools_floor()()
    requires = load_toml(_SHARED_CONFIG)["build-system"]["requires"]
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
    packages = load_toml(_PACKAGES_CONFIG)["packages"]

    checked = 0
    for pkg_name, pkg_config in packages.items():
        pyproject_path = _REPO_ROOT / pkg_config["path"] / "pyproject.toml"
        assert pyproject_path.exists(), f"{pyproject_path} is missing (run scripts/generate_pyproject.py)."
        floor = parse_floor(load_toml(pyproject_path).get("build-system", {}).get("requires", []))
        assert floor == shared_floor, (
            f"{pkg_name}: generated {pyproject_path} requires setuptools>={floor}, but "
            f"config/shared.toml declares setuptools>={shared_floor} (run scripts/generate_pyproject.py)."
        )
        checked += 1

    assert checked > 0, "expected at least one generated pyproject.toml to declare a setuptools floor."

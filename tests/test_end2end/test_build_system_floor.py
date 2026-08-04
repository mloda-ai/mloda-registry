"""The declared setuptools build floor must cover the license form the generator emits.

``scripts/generate_pyproject.py`` writes the PEP 639 form ``license = "Apache-2.0"``, a bare SPDX
string rather than the legacy ``{ text = ... }`` / ``{ file = ... }`` table. setuptools 76.1.0
rejects that form with ``ValueError: invalid pyproject.toml config: project.license``; 77.0.1 is the
first release that accepts it (77.0.0 was never published). So the ``[build-system].requires`` floor
in ``config/shared.toml``, inherited by every generated ``pyproject.toml``, must be at least 77.0.1.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GEN_PATH = _REPO_ROOT / "scripts" / "generate_pyproject.py"
_SHARED_CONFIG = _REPO_ROOT / "config" / "shared.toml"
_PACKAGES_CONFIG = _REPO_ROOT / "config" / "packages.toml"

# First setuptools release accepting a PEP 639 SPDX license string.
_PEP639_SETUPTOOLS = "77.0.1"

# Any package works: the license line and the build-system block come from the shared config.
_SAMPLE_PACKAGE = "mloda-registry"

_SETUPTOOLS_FLOOR_RE = re.compile(r"^setuptools\s*>=\s*(?P<version>[0-9][0-9.]*)$")


def _load_module(name: str, path: Path) -> ModuleType:
    """Import a loose script by file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"could not load spec for {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load_module("generate_pyproject", _GEN_PATH)


def _load_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _version_tuple(version: str) -> tuple[int, ...]:
    """Comparable form of a numeric release version."""
    return tuple(int(part) for part in version.split("."))


def _setuptools_floor(source: str, requires: list[str]) -> str:
    """The ``X`` of the single ``setuptools>=X`` entry of a ``[build-system].requires`` array."""
    matches = [_SETUPTOOLS_FLOOR_RE.match(entry.strip()) for entry in requires]
    floors = [match.group("version") for match in matches if match is not None]
    assert len(floors) == 1, f"{source}: expected exactly one 'setuptools>=X' build requirement, got {requires!r}"
    return floors[0]


def _shared_floor() -> str:
    """The setuptools floor every generated pyproject.toml inherits."""
    return _setuptools_floor(str(_SHARED_CONFIG), _load_toml(_SHARED_CONFIG)["build-system"]["requires"])


def _generated_license(pkg_name: str) -> Any:
    """The ``[project].license`` value the generator emits for a package."""
    shared = _load_toml(_SHARED_CONFIG)
    packages = _load_toml(_PACKAGES_CONFIG)["packages"]
    content: str = gen.generate_pyproject(pkg_name, packages[pkg_name], shared, packages)
    return tomllib.loads(content).get("project", {}).get("license")


def test_generator_emits_a_bare_spdx_license() -> None:
    """The emitted form, pinned here because the build floor below exists for it."""
    license_value = _generated_license(_SAMPLE_PACKAGE)
    assert isinstance(license_value, str), (
        f"{_SAMPLE_PACKAGE}: expected a PEP 639 SPDX license string, got {license_value!r}"
    )


def test_build_floor_covers_the_emitted_license_form() -> None:
    """A PEP 639 license string builds only from setuptools 77.0.1 on."""
    license_value = _generated_license(_SAMPLE_PACKAGE)
    assert isinstance(license_value, str), (
        f"{_SAMPLE_PACKAGE}: the generator no longer emits a PEP 639 SPDX license string but "
        f"{license_value!r}; the setuptools floor asserted below exists for that form, revisit it"
    )

    floor = _shared_floor()
    assert _version_tuple(floor) >= _version_tuple(_PEP639_SETUPTOOLS), (
        f"config/shared.toml declares setuptools>={floor}, but the generator emits "
        f'license = "{license_value}", which setuptools accepts only from {_PEP639_SETUPTOOLS} on '
        "(76.1.0 raises 'invalid pyproject.toml config: project.license')."
    )


def test_generated_pyprojects_carry_the_shared_build_floor() -> None:
    """A floor bump reaches wheels only once every generated pyproject.toml carries it."""
    shared_floor = _shared_floor()
    packages = _load_toml(_PACKAGES_CONFIG)["packages"]

    checked = 0
    for pkg_name, pkg_config in packages.items():
        pyproject_path = _REPO_ROOT / pkg_config["path"] / "pyproject.toml"
        assert pyproject_path.exists(), f"{pyproject_path} is missing (run scripts/generate_pyproject.py)."
        requires = _load_toml(pyproject_path).get("build-system", {}).get("requires", [])
        floor = _setuptools_floor(str(pyproject_path), requires)
        assert floor == shared_floor, (
            f"{pkg_name}: generated {pyproject_path} requires setuptools>={floor}, but "
            f"config/shared.toml declares setuptools>={shared_floor} (run scripts/generate_pyproject.py)."
        )
        checked += 1

    assert checked > 0, "expected at least one generated pyproject.toml to declare a setuptools floor."

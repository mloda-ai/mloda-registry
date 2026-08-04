"""Internal dependency floors in scripts/verify_floor_installs.py. Published leaves floor in-repo
distributions; a floor naming a version that never shipped makes the floored install unresolvable,
so every declared floor must be a released version at or below the workspace version."""

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
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "verify_floor_installs.py"
_SHARED_CONFIG = _REPO_ROOT / "config" / "shared.toml"
_PACKAGES_CONFIG = _REPO_ROOT / "config" / "packages.toml"

# First data-operations release shipping the shared guard helpers the leaves import; 0.3.3 never shipped.
_DATA_OPERATIONS_FIRST_RELEASE = (0, 4, 0)

# First published mloda-community-example release; 0.2.0 never reached an index.
_EXAMPLE_FIRST_RELEASE = (0, 2, 4)

_DEP = "mloda-community-data-operations"
_LEAF = "mloda-community-aggregation"
_LEAF_MODULE = "mloda.community.feature_groups.data_operations.aggregation"


def _internal_floor_pairs() -> Callable[[dict[str, dict[str, Any]]], list[Any]]:
    """The pair extractor scripts/verify_floor_installs.py must expose."""
    assert _SCRIPT_PATH.exists(), f"{_SCRIPT_PATH} is missing; it is what proves the declared floors install"
    module = load_script("verify_floor_installs", _SCRIPT_PATH)
    pairs: Callable[[dict[str, dict[str, Any]]], list[Any]] | None = getattr(module, "internal_floor_pairs", None)
    assert callable(pairs), "verify_floor_installs.internal_floor_pairs(packages) must be a callable"
    return pairs


def _load_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _version_tuple(version: str) -> tuple[int, ...]:
    """Comparable form of a numeric release version."""
    return tuple(int(part) for part in version.split("."))


def _synthetic_packages(
    leaf_dependency: str = f"{_DEP}>=0.4.0", *, published: bool | None = True
) -> dict[str, dict[str, Any]]:
    """A minimal packages table shaped like config/packages.toml; ``published=None`` omits the flag."""
    leaf: dict[str, Any] = {
        "description": "leaf",
        "dependencies": [leaf_dependency],
        "path": "mloda/community/feature_groups/data_operations/aggregation",
    }
    if published is not None:
        leaf["published"] = published
    return {
        _DEP: {
            "description": "base",
            "dependencies": ["{core_dependency}"],
            "path": "mloda/community/feature_groups/data_operations",
            "published": True,
        },
        _LEAF: leaf,
    }


def _real_pairs() -> list[Any]:
    """Pairs extracted from the real config, resolved against the repo root, not the cwd."""
    return _internal_floor_pairs()(_load_toml(_PACKAGES_CONFIG)["packages"])


def test_a_published_leaf_yields_its_internal_floor_pair() -> None:
    """The pair carries the leaf, the floored distribution, the floor, and the leaf's dotted module path."""
    pairs = _internal_floor_pairs()(_synthetic_packages())

    assert len(pairs) == 1, f"expected exactly one pair, got {pairs!r}"
    pair = pairs[0]
    assert pair.package == _LEAF, f"pair.package must be the depending distribution, got {pair.package!r}"
    assert pair.dependency == _DEP, f"pair.dependency must be the floored distribution, got {pair.dependency!r}"
    assert pair.floor == "0.4.0", f"pair.floor must be the '>=' bound, got {pair.floor!r}"
    assert pair.module == _LEAF_MODULE, f"pair.module must derive from the path field, got {pair.module!r}"
    assert tuple(pair) == (_LEAF, _DEP, "0.4.0", _LEAF_MODULE), (
        "FloorPair must be a NamedTuple ordered (package, dependency, floor, module)"
    )


@pytest.mark.parametrize("published", [None, False], ids=["flag-omitted", "flag-false"])
def test_an_unpublished_leaf_yields_no_pair(published: bool | None) -> None:
    """Unpublished packages ship only inside the bundle wheels, so their floors are never installed alone."""
    pairs = _internal_floor_pairs()(_synthetic_packages(published=published))
    assert pairs == [], f"unpublished {_LEAF} must yield no pairs, got {pairs!r}"


@pytest.mark.parametrize("dependency", ["{core_dependency}", "mloda>=0.10.0,<0.11.0"])
def test_an_external_dependency_yields_no_pair(dependency: str) -> None:
    """Only names that are themselves keys of the packages table are in-repo floors."""
    pairs = _internal_floor_pairs()(_synthetic_packages(dependency))
    assert pairs == [], f"{dependency!r} names no in-repo distribution, got pairs {pairs!r}"


@pytest.mark.parametrize("dependency", [_DEP, f"{_DEP}==0.4.0"], ids=["bare", "exact-pin"])
def test_an_internal_dependency_without_a_floor_is_rejected(dependency: str) -> None:
    """An in-repo dependency must declare a '>=' floor; anything else fails loudly, naming the package."""
    with pytest.raises(ValueError, match=_LEAF):
        _internal_floor_pairs()(_synthetic_packages(dependency))


@pytest.mark.parametrize("dependency", [f"{_DEP} >= 0.4.0", f"{_DEP}>=0.4.0,<1"], ids=["whitespace", "upper-bound"])
def test_floor_extraction_tolerates_spec_variants(dependency: str) -> None:
    """Whitespace around '>=' and a trailing spec after a comma are legitimate PEP 508 spellings."""
    pairs = _internal_floor_pairs()(_synthetic_packages(dependency))
    assert [(pair.dependency, pair.floor) for pair in pairs] == [(_DEP, "0.4.0")], (
        f"parsing {dependency!r} produced {pairs!r}"
    )


def test_real_floors_are_numeric_and_at_most_the_workspace_version() -> None:
    """A floor above config/shared.toml [project].version names a release that cannot exist yet."""
    workspace = _load_toml(_SHARED_CONFIG)["project"]["version"]
    pairs = _real_pairs()
    assert pairs, "expected config/packages.toml to declare at least one internal floor"
    for pair in pairs:
        parts = pair.floor.split(".")
        assert len(parts) == 3 and all(part.isdigit() for part in parts), (
            f"{pair.package}: floor {pair.floor!r} on {pair.dependency} is not three dot-separated integers"
        )
        assert _version_tuple(pair.floor) <= _version_tuple(workspace), (
            f"{pair.package}: floors {pair.dependency} at {pair.floor}, above the workspace version {workspace}; "
            "per docs/packaging.md that floor is undeclarable"
        )


def test_data_operations_floors_point_at_a_released_version() -> None:
    """0.3.3 was never released; 0.4.0 first shipped the shared guard helpers the leaves import."""
    pairs = [pair for pair in _real_pairs() if pair.dependency == _DEP]
    assert pairs, f"expected published leaves flooring {_DEP} in config/packages.toml"
    for pair in pairs:
        assert _version_tuple(pair.floor) >= _DATA_OPERATIONS_FIRST_RELEASE, (
            f"{pair.package}: floors {_DEP} at {pair.floor}, which was never released; the first release "
            "shipping the shared guard helpers is 0.4.0, so an install at the floor could not import"
        )


def test_example_a_floors_the_first_published_example_release() -> None:
    """config/packages.toml says 0.2.0, but mloda-community-example first shipped at 0.2.4."""
    pairs = [pair for pair in _real_pairs() if pair.package == "mloda-community-example-a"]
    assert pairs, "expected mloda-community-example-a to floor an in-repo distribution"
    for pair in pairs:
        assert _version_tuple(pair.floor) >= _EXAMPLE_FIRST_RELEASE, (
            f"mloda-community-example-a: floors {pair.dependency} at {pair.floor}, which was never published; "
            "the first published mloda-community-example release is 0.2.4"
        )

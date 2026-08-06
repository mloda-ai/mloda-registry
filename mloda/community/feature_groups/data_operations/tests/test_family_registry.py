"""Lint: the data-operation family registry must match the packages on disk.

``FAMILY_BASE_MODULES`` in ``catalog.py`` is the single registry of the built-in
data-operation families, and every family base class describes itself through the
``DataOperationFamily`` mixin. This module guards that setup against drift: the
registry against the package tree, each family's self-description against the catalog
entry derived from it, the framework table and the manifests against the files on disk.

The negative tests (``test_*_reports_planted_*``, ``test_*_skips_*``) feed planted
inputs to the discovery and comparison helpers so the guards are proven to fire
rather than merely to pass.
"""

from __future__ import annotations

import dataclasses
import importlib
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.options import Options

from mloda.community.feature_groups.data_operations import DataOperationsCatalog
from mloda.community.feature_groups.data_operations.catalog import (
    FAMILY_BASE_MODULES,
    FRAMEWORKS,
    FrameworkInfo,
    _module_local_subclasses,
    installed_family_classes,
    operations_in_declaration_order,
)
from mloda.community.feature_groups.data_operations.family import DataOperationFamily
from mloda.community.feature_groups.data_operations.row_preserving.rank.base import RankFeatureGroup
from mloda.community.feature_groups.data_operations.tests.test_framework_support_matrix import is_artifact_path


REPO_ROOT = Path(__file__).resolve().parents[5]
DATA_OPERATIONS_ROOT = REPO_ROOT / "mloda" / "community" / "feature_groups" / "data_operations"

# Matches a class-level ``PREFIX_PATTERN = ...`` / ``PREFIX_PATTERN: ...`` definition,
# not a mere mention in a docstring or comment.
_PREFIX_DEF_RE = re.compile(r"^\s*PREFIX_PATTERN\s*[:=]", re.MULTILINE)

# Structural-only options, shared by the sibling data-operation lints: enough for families
# that require ``partition_by`` / ``order_by`` / ``time_column`` to accept their string-based
# names, but deliberately free of any operation-type key (``aggregation_type``,
# ``frame_type``, ...) so no family can match via the config-based path.
PERMISSIVE_OPTIONS = Options(context={"partition_by": ["g"], "order_by": "t", "time_column": "t"})

FAMILY_CLASSES: tuple[type[Any], ...] = installed_family_classes()
FAMILY_IDS: list[str] = [str(cls.FAMILY_NAME) for cls in FAMILY_CLASSES]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def family_package(base_module: str) -> str:
    """Dotted package of a family base module (drops the trailing ``.base``)."""
    return base_module.rsplit(".", 1)[0]


def package_dir(package: str) -> Path:
    """Filesystem directory of a dotted package inside this repo."""
    return REPO_ROOT.joinpath(*package.split("."))


FAMILY_PACKAGES: dict[str, str] = {str(cls.FAMILY_NAME): family_package(cls.__module__) for cls in FAMILY_CLASSES}


def module_local_family_classes(module: ModuleType) -> list[type[Any]]:
    """Classes defined in *module* itself that mix in :class:`DataOperationFamily`.

    Delegates to the production discovery helper so the lint cannot drift from it.
    """
    return _module_local_subclasses(module, DataOperationFamily)


def non_empty_unique_strings(values: Any) -> list[str]:
    """Problems with *values* as a non-empty tuple of unique non-empty strings."""
    if not isinstance(values, tuple) or not values:
        return [f"not a non-empty tuple: {values!r}"]
    problems: list[str] = []
    seen: list[Any] = []
    for value in values:
        if not isinstance(value, str) or not value:
            problems.append(f"not a non-empty string: {value!r}")
        elif value in seen:
            problems.append(f"duplicate entry: {value!r}")
        else:
            seen.append(value)
    return problems


def discover_family_base_modules(root: Path = DATA_OPERATIONS_ROOT, repo_root: Path = REPO_ROOT) -> set[str]:
    """Dotted modules of every non-artifact ``base.py`` under *root* that defines a ``PREFIX_PATTERN``."""
    modules: set[str] = set()
    for base_py in root.rglob("base.py"):
        rel = base_py.relative_to(repo_root)
        if is_artifact_path(rel):
            continue
        if not _PREFIX_DEF_RE.search(base_py.read_text()):
            continue
        modules.add(".".join(rel.with_suffix("").parts))
    return modules


def discover_uncovered_families(root: Path = DATA_OPERATIONS_ROOT, repo_root: Path = REPO_ROOT) -> list[str]:
    """Dotted paths of ``base.py`` files that define a ``PREFIX_PATTERN`` but are absent from
    :data:`FAMILY_BASE_MODULES`. Build artifact copies are skipped."""
    return sorted(discover_family_base_modules(root, repo_root) - set(FAMILY_BASE_MODULES))


def discover_manifest_packages(root: Path = DATA_OPERATIONS_ROOT, repo_root: Path = REPO_ROOT) -> set[str]:
    """Dotted packages of every non-artifact ``manifest.py`` under *root*."""
    packages: set[str] = set()
    for manifest_py in root.rglob("manifest.py"):
        rel = manifest_py.relative_to(repo_root)
        if is_artifact_path(rel):
            continue
        packages.add(".".join(rel.parent.parts))
    return packages


def discover_backend_prefixes(directory: Path, dirname: str) -> set[str]:
    """Filename prefixes of the ``<prefix>_<dirname>.py`` backend modules directly in *directory*."""
    suffix = f"_{dirname}.py"
    return {path.name[: -len(suffix)] for path in directory.glob(f"*{suffix}")}


def backend_modules_on_disk(package: str) -> set[str]:
    """Backend module names of *package* whose filename prefix is a known framework."""
    dirname = package.rsplit(".", 1)[-1]
    known = {framework.module_prefix for framework in FRAMEWORKS}
    return {f"{prefix}_{dirname}" for prefix in discover_backend_prefixes(package_dir(package), dirname) & known}


def compare_backend_modules(manifest_modules: set[str], disk_modules: set[str]) -> list[str]:
    """Drift between a family's manifest ``BACKENDS`` submodules and its backend files on disk."""
    return [f"manifest-only: {name}" for name in sorted(manifest_modules - disk_modules)] + [
        f"on-disk-only: {name}" for name in sorted(disk_modules - manifest_modules)
    ]


def load_backends(package: str) -> list[tuple[str, str]]:
    """The ``BACKENDS`` constant of a family package's manifest."""
    backends: list[tuple[str, str]] = importlib.import_module(f"{package}.manifest").BACKENDS
    return backends


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------


def test_every_registry_module_declares_exactly_one_family_class() -> None:
    problems: list[str] = []
    for base_module in FAMILY_BASE_MODULES:
        try:
            module = importlib.import_module(base_module)
        except ModuleNotFoundError:
            # Production installed_family_classes() reads this as "family not installed" and
            # skips it; the lint follows the same contract. Registry entries with no module on
            # disk are caught by test_every_registry_module_is_discovered_on_disk.
            continue
        except ImportError as exc:
            problems.append(f"{base_module}: not importable ({exc})")
            continue
        found = module_local_family_classes(module)
        if len(found) != 1:
            problems.append(f"{base_module}: {[cls.__name__ for cls in found]}")
    assert problems == [], (
        "These FAMILY_BASE_MODULES entries do not declare exactly one module-local "
        "DataOperationFamily subclass (fix the registry entry or the base module):\n  " + "\n  ".join(problems)
    )


def test_family_names_are_declared_locally_and_unique() -> None:
    inherited = [cls.__name__ for cls in FAMILY_CLASSES if "FAMILY_NAME" not in cls.__dict__]
    assert inherited == [], (
        "These family classes inherit FAMILY_NAME instead of declaring their own "
        "(set FAMILY_NAME in the class body):\n  " + "\n  ".join(inherited)
    )
    duplicates = sorted({name for name in FAMILY_IDS if FAMILY_IDS.count(name) > 1})
    assert duplicates == [], "These FAMILY_NAME values are claimed by more than one family:\n  " + "\n  ".join(
        duplicates
    )


def test_family_names_match_catalog_names() -> None:
    catalog_names = {info.name for info in DataOperationsCatalog.list()}
    family_names = set(FAMILY_IDS)
    assert sorted(family_names ^ catalog_names) == [], (
        "FAMILY_NAME values and DataOperationsCatalog.list() names diverge: "
        f"family-only={sorted(family_names - catalog_names)} catalog-only={sorted(catalog_names - family_names)}"
    )


def test_operations_in_declaration_order_follows_the_registry() -> None:
    ordered = [info.name for info in operations_in_declaration_order()]
    assert ordered == FAMILY_IDS, (
        "operations_in_declaration_order() does not follow FAMILY_BASE_MODULES order: "
        f"got {ordered}, expected {FAMILY_IDS}"
    )


def test_subtype_labels_match_the_catalog() -> None:
    problems: list[str] = []
    for cls in FAMILY_CLASSES:
        label = cls.SUBTYPE_LABEL
        if not isinstance(label, str) or not label:
            problems.append(f"{cls.FAMILY_NAME}: SUBTYPE_LABEL is not a non-empty string ({label!r})")
            continue
        catalog_label = DataOperationsCatalog.get(str(cls.FAMILY_NAME)).subtype_label
        if label != catalog_label:
            problems.append(f"{cls.FAMILY_NAME}: SUBTYPE_LABEL {label!r} != catalog subtype_label {catalog_label!r}")
    assert problems == [], "SUBTYPE_LABEL diverges from the catalog:\n  " + "\n  ".join(problems)


# ---------------------------------------------------------------------------
# Per-family self-description
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILY_CLASSES, ids=FAMILY_IDS)
def test_catalog_subtypes_match_the_catalog(family: type[Any]) -> None:
    subtypes = family.catalog_subtypes()
    if subtypes is not None:
        problems = non_empty_unique_strings(subtypes)
        assert problems == [], f"{family.FAMILY_NAME}: catalog_subtypes() is malformed:\n  " + "\n  ".join(problems)
    catalog_subtypes = DataOperationsCatalog.get(str(family.FAMILY_NAME)).subtypes
    assert subtypes == catalog_subtypes, (
        f"{family.FAMILY_NAME}: catalog_subtypes() {subtypes!r} != catalog subtypes {catalog_subtypes!r}"
    )


@pytest.mark.parametrize("family", FAMILY_CLASSES, ids=FAMILY_IDS)
def test_catalog_probe_returns_a_name_and_options_per_subtype(family: type[Any]) -> None:
    subtypes = family.catalog_subtypes()
    if subtypes is None:
        pytest.skip(f"{family.FAMILY_NAME} has no subtype axis")
    problems: list[str] = []
    for subtype in subtypes:
        probe = family.catalog_probe(subtype)
        if not isinstance(probe, tuple) or len(probe) != 2:
            problems.append(f"{subtype}: catalog_probe did not return a 2-tuple ({probe!r})")
            continue
        feature_name, options = probe
        if not isinstance(feature_name, str) or not feature_name:
            problems.append(f"{subtype}: probe feature name is not a non-empty string ({feature_name!r})")
        if not isinstance(options, Options):
            problems.append(f"{subtype}: probe options are not an Options instance ({options!r})")
    assert problems == [], f"{family.FAMILY_NAME}: catalog_probe() is malformed:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("family", FAMILY_CLASSES, ids=FAMILY_IDS)
def test_example_feature_names_are_accepted_by_their_family(family: type[Any]) -> None:
    names = family.example_feature_names()
    problems = non_empty_unique_strings(names)
    assert problems == [], f"{family.FAMILY_NAME}: example_feature_names() is malformed:\n  " + "\n  ".join(problems)
    rejected = [name for name in names if not family.match_feature_group_criteria(name, PERMISSIVE_OPTIONS)]
    assert rejected == [], (
        f"{family.FAMILY_NAME}: these example_feature_names() are rejected by the family's own "
        "match_feature_group_criteria (fix the vocabulary or the matcher):\n  " + "\n  ".join(rejected)
    )


@pytest.mark.parametrize("family", FAMILY_CLASSES, ids=FAMILY_IDS)
def test_example_feature_names_cover_every_catalog_subtype(family: type[Any]) -> None:
    """Per-family vocabulary floor: at least one example name per catalog subtype.

    ``example_feature_names()`` is the sole input of the sibling lints
    (``test_prefix_pattern_collisions``, ``test_return_data_type_rule_invariants``), and their
    non-vacuity guards are global totals: one family collapsing its vocabulary to a single name
    would still clear them. This floor is the per-family half of that guard.
    """
    names = family.example_feature_names()
    subtypes = family.catalog_subtypes() or ()
    assert len(names) >= len(subtypes), (
        f"{family.FAMILY_NAME}: example_feature_names() publishes {len(names)} name(s) for "
        f"{len(subtypes)} catalog subtype(s); a family's vocabulary must at least cover its own "
        "subtype axis, otherwise the lints reading it lose coverage silently."
    )


@pytest.mark.parametrize("family", FAMILY_CLASSES, ids=FAMILY_IDS)
def test_matching_patterns_cover_the_example_feature_names(family: type[Any]) -> None:
    patterns = family.matching_patterns()
    problems = non_empty_unique_strings(patterns)
    assert problems == [], f"{family.FAMILY_NAME}: matching_patterns() is malformed:\n  " + "\n  ".join(problems)
    compiled: list[re.Pattern[str]] = []
    uncompilable: list[str] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error as exc:
            uncompilable.append(f"{pattern!r}: {exc}")
    assert uncompilable == [], f"{family.FAMILY_NAME}: matching_patterns() are not valid regexes:\n  " + "\n  ".join(
        uncompilable
    )
    unmatched = [
        name for name in family.example_feature_names() if not any(pattern.search(name) for pattern in compiled)
    ]
    assert unmatched == [], (
        f"{family.FAMILY_NAME}: these example_feature_names() are matched by none of matching_patterns() "
        "(the declared patterns no longer describe the family's vocabulary):\n  " + "\n  ".join(unmatched)
    )


@pytest.mark.parametrize("family", FAMILY_CLASSES, ids=FAMILY_IDS)
def test_prefix_pattern_is_declared_among_matching_patterns(family: type[Any]) -> None:
    """Every family lists its PREFIX_PATTERN among matching_patterns(), multi-pattern ones included.

    PREFIX_PATTERN is published as ``OperationInfo.prefix_pattern``, so a family that routes on
    patterns not containing it would document a pattern it does not actually match on.
    """
    patterns = family.matching_patterns()
    assert str(family.PREFIX_PATTERN) in patterns, (
        f"{family.FAMILY_NAME}: matching_patterns() {patterns!r} omits PREFIX_PATTERN "
        f"{family.PREFIX_PATTERN!r}, which the catalog publishes as OperationInfo.prefix_pattern; "
        "a family matching on further patterns must list them all, PREFIX_PATTERN included."
    )


def test_parametric_rank_family_without_a_representative_n_is_rejected() -> None:
    """rank's PARAMETRIC_RANK_N is hand-listed, so a parametric family with no entry must fail loudly.

    Both self-description surfaces build their name token through ``_rank_token``; a missing N
    would otherwise emit a probe name and an example name that nothing matches.
    """

    class _RankWithUnmappedParametricFamily(RankFeatureGroup):
        PARAMETRIC_RANK_FAMILIES = (*RankFeatureGroup.PARAMETRIC_RANK_FAMILIES, "quartile")

    message = "PARAMETRIC_RANK_N has no representative N for parametric family 'quartile'"
    with pytest.raises(ValueError, match=message):
        _RankWithUnmappedParametricFamily.example_feature_names()
    with pytest.raises(ValueError, match=message):
        _RankWithUnmappedParametricFamily.catalog_probe("quartile")


# ---------------------------------------------------------------------------
# Framework table
# ---------------------------------------------------------------------------


def test_framework_table_entries_are_framework_infos() -> None:
    field_names = {field.name for field in dataclasses.fields(FrameworkInfo)}
    assert field_names == {"module_prefix", "catalog_key", "label"}, (
        f"FrameworkInfo exposes {sorted(field_names)}, expected module_prefix, catalog_key and label"
    )
    wrong = [repr(framework) for framework in FRAMEWORKS if not isinstance(framework, FrameworkInfo)]
    assert wrong == [], "These FRAMEWORKS entries are not FrameworkInfo instances:\n  " + "\n  ".join(wrong)


def test_framework_table_fields_are_unique() -> None:
    problems: list[str] = []
    for field_name in ("module_prefix", "catalog_key", "label"):
        values = [str(getattr(framework, field_name)) for framework in FRAMEWORKS]
        duplicates = sorted({value for value in values if values.count(value) > 1})
        problems += [f"{field_name}: {value!r}" for value in duplicates]
    assert problems == [], "These FRAMEWORKS values are claimed by more than one framework:\n  " + "\n  ".join(problems)


def test_framework_module_prefixes_match_the_backend_files_on_disk() -> None:
    declared = {str(framework.module_prefix) for framework in FRAMEWORKS}
    on_disk: set[str] = set()
    for base_module in FAMILY_BASE_MODULES:
        package = family_package(base_module)
        on_disk |= discover_backend_prefixes(package_dir(package), package.rsplit(".", 1)[-1])
    undeclared = sorted(on_disk - declared)
    assert undeclared == [], (
        "These backend module prefixes exist under the family packages but are missing from FRAMEWORKS "
        "(add a FrameworkInfo for each):\n  " + "\n  ".join(undeclared)
    )
    unused = sorted(declared - on_disk)
    assert unused == [], (
        "These FRAMEWORKS module prefixes back no backend module on disk (remove the stale "
        "FrameworkInfo):\n  " + "\n  ".join(unused)
    )


def test_catalog_framework_keys_are_declared_in_the_framework_table() -> None:
    declared = {str(framework.catalog_key) for framework in FRAMEWORKS}
    reported = {key for info in DataOperationsCatalog.list() for key in info.frameworks}
    undeclared = sorted(reported - declared)
    assert undeclared == [], (
        "These compute frameworks appear in OperationInfo.frameworks but have no FRAMEWORKS "
        "catalog_key:\n  " + "\n  ".join(undeclared)
    )


# ---------------------------------------------------------------------------
# Registry drift against the package tree
# ---------------------------------------------------------------------------


def test_every_family_on_disk_is_in_the_registry() -> None:
    uncovered = discover_uncovered_families()
    assert uncovered == [], (
        "These data-operation base.py modules define a PREFIX_PATTERN but are missing from "
        "FAMILY_BASE_MODULES (extend the registry so they are covered):\n  " + "\n  ".join(uncovered)
    )


def test_every_registry_module_is_discovered_on_disk() -> None:
    missing = sorted(set(FAMILY_BASE_MODULES) - discover_family_base_modules())
    assert missing == [], (
        "These FAMILY_BASE_MODULES entries were not discovered on disk; an over-broad build-artifact "
        "skip (is_artifact_path) would hide real family modules like this:\n  " + "\n  ".join(missing)
    )


def test_discover_uncovered_families_reports_planted_family(tmp_path: Path) -> None:
    """A planted base.py defining a PREFIX_PATTERN under root is reported by its dotted path."""
    base_py = tmp_path / "pkg" / "some_family" / "base.py"
    base_py.parent.mkdir(parents=True)
    base_py.write_text("PREFIX_PATTERN = r'__op$'\n")
    uncovered = discover_uncovered_families(root=tmp_path / "pkg", repo_root=tmp_path)
    assert uncovered == ["pkg.some_family.base"]


@pytest.mark.parametrize("artifact_part", ["build", "dist", "something.egg-info", ".tox", ".venv"])
def test_discover_uncovered_families_skips_build_artifact_dirs(tmp_path: Path, artifact_part: str) -> None:
    """A base.py copy under a build artifact directory is not reported alongside the real module."""
    body = "PREFIX_PATTERN = r'__op$'\n"
    real = tmp_path / "pkg" / "some_family" / "base.py"
    real.parent.mkdir(parents=True)
    real.write_text(body)
    copy = tmp_path / "pkg" / "some_family" / artifact_part / "lib" / "pkg" / "some_family" / "base.py"
    copy.parent.mkdir(parents=True)
    copy.write_text(body)
    uncovered = discover_uncovered_families(root=tmp_path / "pkg", repo_root=tmp_path)
    assert uncovered == ["pkg.some_family.base"]


# ---------------------------------------------------------------------------
# Manifest agreement
# ---------------------------------------------------------------------------


def test_every_family_manifest_exposes_backends() -> None:
    problems: list[str] = []
    for base_module in FAMILY_BASE_MODULES:
        package = family_package(base_module)
        backends = load_backends(package)
        if not isinstance(backends, list) or not backends:
            problems.append(f"{package}: BACKENDS is not a non-empty list ({backends!r})")
            continue
        seen: list[str] = []
        for entry in backends:
            if not isinstance(entry, tuple) or len(entry) != 2 or not all(isinstance(part, str) for part in entry):
                problems.append(f"{package}: BACKENDS entry is not a (submodule, class_name) pair ({entry!r})")
                continue
            if entry[0] in seen:
                problems.append(f"{package}: duplicate BACKENDS submodule {entry[0]!r}")
            seen.append(entry[0])
    assert problems == [], (
        "Every data-operation family manifest must expose BACKENDS as a non-empty list of "
        "(submodule, class_name) pairs with unique submodules:\n  " + "\n  ".join(problems)
    )


def test_manifest_backends_match_the_backend_files_on_disk() -> None:
    problems: list[str] = []
    for base_module in FAMILY_BASE_MODULES:
        package = family_package(base_module)
        manifest_modules = {submodule for submodule, _class_name in load_backends(package)}
        problems += [
            f"{package}: {problem}"
            for problem in compare_backend_modules(manifest_modules, backend_modules_on_disk(package))
        ]
    assert problems == [], (
        "Manifest BACKENDS drifted from the backend modules on disk (add the missing manifest entry "
        "or delete the stale one):\n  " + "\n  ".join(problems)
    )


def test_catalog_frameworks_have_a_manifest_backend() -> None:
    prefix_by_catalog_key = {str(framework.catalog_key): str(framework.module_prefix) for framework in FRAMEWORKS}
    problems: list[str] = []
    for info in DataOperationsCatalog.list():
        package = FAMILY_PACKAGES[info.name]
        dirname = package.rsplit(".", 1)[-1]
        manifest_modules = {submodule for submodule, _class_name in load_backends(package)}
        for catalog_key in info.frameworks:
            prefix = prefix_by_catalog_key.get(catalog_key)
            if prefix is None:
                # Unknown catalog keys are reported by the framework-table test.
                continue
            expected = f"{prefix}_{dirname}"
            if expected not in manifest_modules:
                problems.append(f"{info.name}: catalog reports {catalog_key} but BACKENDS has no {expected!r} entry")
    assert problems == [], (
        "The catalog reports frameworks that the family manifest does not register:\n  " + "\n  ".join(problems)
    )


def test_every_manifest_on_disk_belongs_to_a_registry_family() -> None:
    registry_packages = {family_package(base_module) for base_module in FAMILY_BASE_MODULES}
    orphans = sorted(discover_manifest_packages() - registry_packages)
    assert orphans == [], (
        "These packages ship a data_operations manifest.py but are not registry families "
        "(add them to FAMILY_BASE_MODULES or drop the manifest):\n  " + "\n  ".join(orphans)
    )


def test_compare_backend_modules_reports_planted_on_disk_backend(tmp_path: Path) -> None:
    """A backend file on disk that no BACKENDS entry lists is reported by the comparison helper."""
    (tmp_path / "pandas_some_family.py").write_text("")
    (tmp_path / "duckdb_some_family.py").write_text("")
    disk = {f"{prefix}_some_family" for prefix in discover_backend_prefixes(tmp_path, "some_family")}
    assert compare_backend_modules({"pandas_some_family"}, disk) == ["on-disk-only: duckdb_some_family"]


def test_compare_backend_modules_reports_manifest_only_backend() -> None:
    """A BACKENDS entry with no backend file on disk is reported by the comparison helper."""
    assert compare_backend_modules({"pandas_some_family", "sqlite_some_family"}, {"pandas_some_family"}) == [
        "manifest-only: sqlite_some_family"
    ]

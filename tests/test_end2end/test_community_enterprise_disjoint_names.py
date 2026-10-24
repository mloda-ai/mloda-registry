"""Community and enterprise FeatureGroups never accept the same feature name (core has no precedence).
Examples are sampled under each side's own options; groups needing extra options rely on coverage and review."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from mloda.provider import DefaultOptionKeys, FeatureChainParserMixin, FeatureGroup, Implementation, property_spec
from mloda.user import Options

from mloda.community.feature_groups.data_operations.tests.test_prefix_pattern_collisions import (
    FAMILIES,
    PERMISSIVE_OPTIONS,
    _load_class,
    generate_valid_names,
)
from mloda.community.feature_groups.example.community_example_feature_group import CommunityExampleFeatureGroup
from mloda.community.feature_groups.example.example_a.example_a_feature_group import ExampleAFeatureGroup
from mloda.community.feature_groups.experimental.german_ledger import reader as german_ledger_reader
from mloda.community.feature_groups.experimental.german_ledger.datev import DatevExtfReader
from mloda.community.feature_groups.experimental.german_ledger.reader import JOURNAL, GdpduReader
from mloda.community.feature_groups.experimental.german_ledger.skr import SkrAccountFeatureGroup
from mloda.community.feature_groups.experimental.german_ledger.sources import SourcesFeatureGroup
from mloda.enterprise.feature_groups.anonymizer.anonymizer_feature_group import AnonymizerFeatureGroup
from mloda.enterprise.feature_groups.binary_example.binary_example_feature_group import BinaryExampleFeatureGroup
from tests.test_end2end.test_entry_points import _plugin_packages

_FEATURE_GROUPS = "mloda.feature_groups"
_COMMUNITY = "mloda/community/"
_ENTERPRISE = "mloda/enterprise/"
_STRING_CONTEXT: dict[str, Any] = {**PERMISSIVE_OPTIONS.context, "pii_key_env": "SOME_ENV"}


@dataclass(frozen=True)
class Example:
    """A feature name an owning class accepts, with the options it is matched under."""

    package: str
    owner: Any
    name: str
    options: Options


def _string_options() -> Options:
    return Options(context=dict(_STRING_CONTEXT))


def _enterprise_examples() -> list[Example]:
    anonymizer = "mloda-enterprise-anonymizer"
    algorithms = AnonymizerFeatureGroup.PROPERTY_MAPPING[AnonymizerFeatureGroup.ALGORITHM].allowed_values or {}
    operations = BinaryExampleFeatureGroup.PROPERTY_MAPPING[BinaryExampleFeatureGroup.OPERATION].allowed_values or {}
    examples = [
        Example(anonymizer, AnonymizerFeatureGroup, f"email__{alg}_pseudonymized", _string_options())
        for alg in algorithms
    ]
    for alg in algorithms:
        config = {AnonymizerFeatureGroup.ALGORITHM: alg, "pii_key_env": "SOME_ENV", "in_features": "email"}
        examples.append(Example(anonymizer, AnonymizerFeatureGroup, "email_pseudonym", Options(context=config)))
    for op in operations:
        binary = {BinaryExampleFeatureGroup.OPERATION: op, "binary_input_columns": ["col_a"]}
        examples.append(
            Example("mloda-enterprise-binary-example", BinaryExampleFeatureGroup, "hashed", Options(context=binary))
        )
    return examples


def _community_explicit_examples() -> list[Example]:
    ledger = "mloda-community-german-ledger"
    fixtures = Path(german_ledger_reader.__file__).parent / "tests" / "fixtures"

    # A reader claims only a name with a dossier behind it, so each example points at a real fixture dossier.
    # Fresh per example: a reader's match writes a reserved key into the Options it is handed.
    def _gdpdu() -> Options:
        return Options({GdpduReader.__name__: str(fixtures / "dossier_a")})

    gdpdu = _gdpdu()
    datev = Options({DatevExtfReader.__name__: str(fixtures / "datev_ledermann")})
    return [
        Example(ledger, GdpduReader, JOURNAL, gdpdu),
        Example(ledger, DatevExtfReader, JOURNAL, datev),
        Example(ledger, SkrAccountFeatureGroup, "revenue", _gdpdu()),
        Example(ledger, SkrAccountFeatureGroup, "receivables", _gdpdu()),
        Example(ledger, SourcesFeatureGroup, "revenue__sources", _gdpdu()),
        Example("mloda-community-example", CommunityExampleFeatureGroup, "CommunityExampleFeatureGroup", Options()),
        Example("mloda-community-example-a", ExampleAFeatureGroup, "ExampleAFeatureGroup", Options()),
    ]


def _community_examples() -> list[Example]:
    packages = _feature_group_packages(_COMMUNITY)
    examples = _community_explicit_examples()
    for spec in FAMILIES:
        package = _package_of_module(spec.module, packages)
        owner = _load_class(spec)
        examples += [Example(package, owner, name, _string_options()) for name in generate_valid_names(spec)]
    return examples


def _feature_group_packages(prefix: str) -> dict[str, str]:
    return {
        name: cfg["path"].replace("/", ".")
        for name, cfg in _plugin_packages()
        if _FEATURE_GROUPS in cfg.get("entry_point_groups", []) and cfg["path"].startswith(prefix)
    }


def _package_of_module(module: str, packages: dict[str, str]) -> str:
    owners = [name for name, dotted in packages.items() if module.startswith(f"{dotted}.")]
    assert owners, f"{module} is not inside any mloda.feature_groups package; add its package to config/packages.toml."
    return max(owners, key=lambda name: len(packages[name]))


def _manifest_classes(prefix: str) -> list[Any]:
    """The manifests' FeatureGroup classes; Implementations never match names, their group does."""
    classes: list[Any] = []
    for dotted in _feature_group_packages(prefix).values():
        classes += [
            cls
            for cls in importlib.import_module(f"{dotted}.manifest").FEATURE_GROUPS
            if not issubclass(cls, Implementation)
        ]
    return classes


def find_cross_matches(candidates: Iterable[Any], examples: Iterable[Example]) -> list[str]:
    """Descriptions of every candidate class that accepts an example it does not own."""
    examples = list(examples)
    return [
        f"{cls.__module__}.{cls.__name__} accepts {example.name!r} owned by {example.package}"
        for cls in candidates
        for example in examples
        if cls.match_feature_group_criteria(example.name, example.options)
    ]


def _format(found: list[str], side: str) -> str:
    return (
        f"{side} FeatureGroups accept names owned by the other side. A license never influences matching, so "
        "core would raise 'Multiple feature groups found'. Make the name patterns disjoint:\n  " + "\n  ".join(found)
    )


def test_every_example_is_accepted_by_its_owner() -> None:
    rejected = [
        f"{e.owner.__name__} rejects {e.name!r}"
        for e in _enterprise_examples() + _community_explicit_examples()
        if not e.owner.match_feature_group_criteria(e.name, e.options)
    ]
    assert rejected == [], "Owners must accept their own examples (else the guard passes vacuously):\n  " + "\n  ".join(
        rejected
    )


def test_community_groups_accept_no_enterprise_name() -> None:
    found = find_cross_matches(_manifest_classes(_COMMUNITY), _enterprise_examples())
    assert found == [], _format(found, "Community")


def test_enterprise_groups_accept_no_community_name() -> None:
    found = find_cross_matches(_manifest_classes(_ENTERPRISE), _community_examples())
    assert found == [], _format(found, "Enterprise")


def test_every_feature_group_package_has_example_features() -> None:
    enterprise_rows = {e.package for e in _enterprise_examples()}
    community_rows = {e.package for e in _community_examples()}
    missing = [
        name
        for prefix, rows in ((_ENTERPRISE, enterprise_rows), (_COMMUNITY, community_rows))
        for name in _feature_group_packages(prefix)
        if name not in rows
    ]
    assert missing == [], "Add example features to this guard for package(s):\n  " + "\n  ".join(missing)


def test_every_manifest_class_is_covered_by_an_example_owner() -> None:
    for prefix, owners in (
        (_ENTERPRISE, {e.owner for e in _enterprise_examples()}),
        (_COMMUNITY, {e.owner for e in _community_examples()}),
    ):
        classes = _manifest_classes(prefix)
        uncovered = [c.__name__ for c in classes if not any(issubclass(c, o) for o in owners)]
        assert uncovered == [], f"Add example features for class(es) under {prefix}:\n  " + "\n  ".join(uncovered)
        idle = [o.__name__ for o in owners if not any(issubclass(c, o) for c in classes)]
        assert idle == [], f"Example owners with no manifest class under {prefix} (vacuous guard):\n  " + "\n  ".join(
            idle
        )


def test_detector_reports_planted_colliding_community_group() -> None:
    class _PrototypePseudonymizer(FeatureChainParserMixin, FeatureGroup):
        """Planted community group claiming the enterprise anonymizer's name."""

        PREFIX_PATTERN = r".*__([a-z0-9_]+)_pseudonymized$"
        MIN_IN_FEATURES = 1
        MAX_IN_FEATURES = 1
        PROPERTY_MAPPING: ClassVar = {
            "pseudonymization_algorithm": property_spec("Algorithm", allowed_values={"hmac_sha256": "HMAC"}),
            "pii_key_env": property_spec("Key environment variable"),
            DefaultOptionKeys.in_features: property_spec("Source column"),
        }

    enterprise = [e for e in _enterprise_examples() if e.name.endswith("_pseudonymized")][:1]
    found = find_cross_matches([_PrototypePseudonymizer], enterprise)
    assert len(found) == 1 and enterprise[0].name in found[0], found

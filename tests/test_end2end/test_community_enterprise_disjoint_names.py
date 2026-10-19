"""Community and enterprise FeatureGroups must never accept the same feature name (core has no precedence)."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar

from mloda.provider import DefaultOptionKeys, FeatureChainParserMixin, FeatureGroup, property_spec
from mloda.user import Options

from mloda.community.feature_groups.data_operations.tests.test_prefix_pattern_collisions import (
    FAMILIES,
    PERMISSIVE_OPTIONS,
    _load_class,
    generate_valid_names,
)
from mloda.community.feature_groups.example.community_example_feature_group import CommunityExampleFeatureGroup
from mloda.community.feature_groups.example.example_a.example_a_feature_group import ExampleAFeatureGroup
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
    config = {"pseudonymization_algorithm": "hmac_sha256", "pii_key_env": "SOME_ENV", "in_features": "email"}
    binary = {"binary_operation": "hash", "binary_input_columns": ["col_a"]}
    return [
        Example(anonymizer, AnonymizerFeatureGroup, "email__hmac_sha256_pseudonymized", _string_options()),
        Example(anonymizer, AnonymizerFeatureGroup, "email_pseudonym", Options(context=config)),
        Example("mloda-enterprise-binary-example", BinaryExampleFeatureGroup, "hashed", Options(context=binary)),
    ]


def _community_explicit_examples() -> list[Example]:
    return [
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
    classes: list[Any] = []
    for dotted in _feature_group_packages(prefix).values():
        classes += importlib.import_module(f"{dotted}.manifest").FEATURE_GROUPS
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


class _PrototypePseudonymizer(FeatureChainParserMixin, FeatureGroup):
    """Planted community group claiming the enterprise anonymizer's name."""

    PREFIX_PATTERN = r".*__(hmac_sha256|aes_siv|format_preserving)_pseudonymized$"
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1
    PROPERTY_MAPPING: ClassVar = {
        "pseudonymization_algorithm": property_spec("Algorithm", allowed_values={"hmac_sha256": "HMAC"}),
        "pii_key_env": property_spec("Key environment variable"),
        DefaultOptionKeys.in_features: property_spec("Source column"),
    }


def test_detector_reports_planted_colliding_community_group() -> None:
    enterprise = [e for e in _enterprise_examples() if e.name == "email__hmac_sha256_pseudonymized"]
    found = find_cross_matches([_PrototypePseudonymizer], enterprise)
    assert len(found) == 1 and "email__hmac_sha256_pseudonymized" in found[0], found

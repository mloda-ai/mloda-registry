"""Lint: data-operation families must not claim overlapping feature-name patterns.

Feature resolution is name-driven: the matcher routes a feature like
``value_int__sum_agg`` to a family by testing that family's declared
``matching_patterns()`` (one pattern for most families, several for
``frame_aggregate``). As the catalog grows, two families could claim overlapping
name patterns, producing order-dependent or ambiguous routing that nothing
currently catches. This module is the guardrail, in the same spirit as the
reflection invariants in ``test_framework_support_matrix.py``.

Both checks below run over each family's FULL vocabulary, read from its own
``example_feature_names()``, with the families themselves coming from
``installed_family_classes()``. A new family is therefore covered the moment it
joins the registry; that the registry matches the packages on disk is guarded by
``test_family_registry.py``.

- ``test_routing_is_exhaustive_over_family_vocabularies`` is the authoritative
  invariant: for every valid feature name of every family, exactly one family's
  ``match_feature_group_criteria`` must accept it. It uses the real router, so a
  raw-regex overlap that subtype validation disambiguates (e.g.
  ``window_aggregation`` rejecting ``sales__avg_7_day_window``) is not flagged.

- ``test_no_unexpected_pattern_overlaps`` is the blunter check: it collects every
  family's ``matching_patterns()`` and fails when a name of one family is also
  matched (raw regex) by another family's pattern, unless the ordered family PAIR
  is in :data:`KNOWN_PATTERN_OVERLAPS`. Allowlisting is per pair rather than per
  name because a single pair covers dozens of names; the pair maps to the expected
  overlap SIZE so that a new overlapping name shape inside an already allowlisted
  pair still gets noticed (``test_known_overlap_sizes_are_pinned``).

The vocabularies these checks read are kept non-vacuous per family by
``test_family_registry.test_example_feature_names_cover_every_catalog_subtype``; the
total below is only the global half of that guard.

The negative tests (``test_find_*_detects_*``) feed planted collisions to the
detector functions so the guardrail is proven to fire, not merely to pass.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mloda.core.abstract_plugins.components.options import Options
from mloda.community.feature_groups.data_operations.catalog import installed_family_classes

# The name-driven Options live in the family-registry lint, which owns the registry
# contract, so every module exercising name-driven matching uses the same instance.
from mloda.community.feature_groups.data_operations.tests.test_family_registry import PERMISSIVE_OPTIONS

FAMILY_CLASSES: tuple[type[Any], ...] = installed_family_classes()


#: ``(owner_family, other_family) -> number of overlapping owner names``: raw-regex
#: overlaps between two families' ``matching_patterns()`` that are known-safe because
#: the other family's ``match_feature_group_criteria`` rejects every overlapping name
#: via subtype validation. Each pair is asserted (below) to still be a real raw overlap,
#: to still be fully disambiguated at routing time, and to still overlap on exactly the
#: pinned number of names, so neither a stale entry nor a newly overlapping name shape
#: inside an allowlisted pair passes unnoticed.
KNOWN_PATTERN_OVERLAPS: Mapping[tuple[str, str], int] = {
    # frame_aggregate's time-window names end in ``_window``, which is all
    # window_aggregation's pattern requires; its agg-type validation rejects them.
    # 56 = 8 aggregation types x 7 time units.
    ("frame_aggregate", "window_aggregation"): 56,
}


@dataclass(frozen=True)
class Collision:
    """A feature name whose set of accepting families is not exactly its owner."""

    feature_name: str
    owner: str
    acceptors: tuple[str, ...]

    @property
    def kind(self) -> str:
        return "OWNER_REJECTS" if self.owner not in self.acceptors else "MULTI_MATCH"


def family_vocabulary(families: tuple[type[Any], ...]) -> dict[str, tuple[str, ...]]:
    """Each family's full valid-name vocabulary, keyed by ``FAMILY_NAME``."""
    return {str(cls.FAMILY_NAME): cls.example_feature_names() for cls in families}


def family_classes(families: tuple[type[Any], ...]) -> dict[str, Any]:
    """The family classes keyed by ``FAMILY_NAME``."""
    return {str(cls.FAMILY_NAME): cls for cls in families}


def collect_prefix_patterns(families: tuple[type[Any], ...]) -> dict[str, list[re.Pattern[str]]]:
    """Each family's declared ``matching_patterns()``, compiled, keyed by ``FAMILY_NAME``."""
    return {str(cls.FAMILY_NAME): [re.compile(pattern) for pattern in cls.matching_patterns()] for cls in families}


def find_pattern_overlaps(
    family_patterns: Mapping[str, list[re.Pattern[str]]],
    family_names: Mapping[str, tuple[str, ...]],
) -> dict[tuple[str, str], list[str]]:
    """Raw-regex overlaps, keyed by ``(owner_family, other_family)``.

    Maps each ordered family pair to the owner's feature names that the other
    family's patterns also match (``re.Pattern.search``). The allowlist is NOT
    applied here; callers filter against :data:`KNOWN_PATTERN_OVERLAPS`.
    """
    overlaps: dict[tuple[str, str], list[str]] = {}
    for owner, names in family_names.items():
        for name in names:
            for other, compiled in family_patterns.items():
                if other == owner:
                    continue
                if any(pattern.search(name) for pattern in compiled):
                    overlaps.setdefault((owner, other), []).append(name)
    return overlaps


def find_collisions(
    family_names: Mapping[str, tuple[str, ...]],
    options: Options,
    classes: Mapping[str, Any],
) -> list[Collision]:
    """Routing-level collisions via ``match_feature_group_criteria``.

    For each family's feature names, the set of families in ``classes`` that accept
    the name must be exactly ``{owner}``. Returns a :class:`Collision` for every
    name where that does not hold (owner rejects its own name, or more than one
    family accepts).
    """
    collisions: list[Collision] = []
    for owner, names in family_names.items():
        for name in names:
            acceptors = tuple(key for key, cls in classes.items() if cls.match_feature_group_criteria(name, options))
            if acceptors != (owner,):
                collisions.append(Collision(name, owner, acceptors))
    return collisions


def _format_collision(collision: Collision) -> str:
    if collision.kind == "OWNER_REJECTS":
        return (
            f"{collision.feature_name!r}: owner {collision.owner!r} does not accept its own "
            f"feature name (acceptors={list(collision.acceptors)}). "
            "Fix the family's vocabulary or its matcher."
        )
    return (
        f"{collision.feature_name!r}: routed to multiple families {list(collision.acceptors)} "
        f"(owner={collision.owner!r}). Two families claim overlapping feature-name patterns."
    )


# --- authoritative invariants -------------------------------------------------


def test_routing_is_exhaustive_over_family_vocabularies() -> None:
    """Every valid name of every family is accepted by exactly that family."""
    vocabulary = family_vocabulary(FAMILY_CLASSES)
    total = sum(len(names) for names in vocabulary.values())
    assert total > 5 * len(FAMILY_CLASSES), (
        f"Family vocabulary is near-vacuous ({total} names for {len(FAMILY_CLASSES)} families); "
        "example_feature_names() lost its live vocabulary, so exhaustive routing would pass trivially."
    )
    collisions = find_collisions(vocabulary, PERMISSIVE_OPTIONS, family_classes(FAMILY_CLASSES))
    assert collisions == [], "Routing collisions over the family vocabularies:\n" + "\n".join(
        _format_collision(c) for c in collisions
    )


def test_vocabulary_drift_is_caught_as_owner_rejects() -> None:
    """Self-proof: a name no family accepts surfaces as OWNER_REJECTS for its owner."""
    owner = str(FAMILY_CLASSES[0].FAMILY_NAME)
    planted = {owner: (*family_vocabulary(FAMILY_CLASSES)[owner], "col__NOTAREALOP_zzz")}
    collisions = find_collisions(planted, PERMISSIVE_OPTIONS, family_classes(FAMILY_CLASSES))
    assert any(c.kind == "OWNER_REJECTS" and c.feature_name == "col__NOTAREALOP_zzz" for c in collisions), (
        "Planted bogus name 'col__NOTAREALOP_zzz' was not flagged as OWNER_REJECTS for its owning "
        "family; the self-validation proof that vocabulary drift surfaces as a collision does not hold."
    )


def test_no_unexpected_pattern_overlaps() -> None:
    overlaps = find_pattern_overlaps(collect_prefix_patterns(FAMILY_CLASSES), family_vocabulary(FAMILY_CLASSES))
    unexpected = sorted(pair for pair in overlaps if pair not in KNOWN_PATTERN_OVERLAPS)
    assert unexpected == [], (
        "Unexpected matching-pattern overlaps (feature names of one family are also matched by "
        "another family's pattern):\n  "
        + "\n  ".join(
            f"{len(overlaps[(owner, other)])} {owner} name(s) also matched by {other}'s pattern, "
            f"e.g. {overlaps[(owner, other)][0]!r}"
            for owner, other in unexpected
        )
    )


def test_known_overlaps_are_present_and_still_disambiguated() -> None:
    """Each allowlisted pair must still overlap, and every overlapping name must still be rejected."""
    overlaps = find_pattern_overlaps(collect_prefix_patterns(FAMILY_CLASSES), family_vocabulary(FAMILY_CLASSES))
    classes = family_classes(FAMILY_CLASSES)
    for owner, other in sorted(KNOWN_PATTERN_OVERLAPS):
        names = overlaps.get((owner, other), [])
        assert names != [], (
            f"Allowlisted overlap {(owner, other)} no longer produces a raw-regex overlap; "
            "remove the stale KNOWN_PATTERN_OVERLAPS entry."
        )
        accepted = [name for name in names if classes[other].match_feature_group_criteria(name, PERMISSIVE_OPTIONS)]
        assert accepted == [], (
            f"Allowlisted overlap {(owner, other)} is no longer disambiguated: {other} now accepts "
            f"{accepted}. These are real collisions, not a safe overlap."
        )


def test_known_overlap_sizes_are_pinned() -> None:
    """Each allowlisted pair must still overlap on exactly the pinned number of names.

    A per-pair allowlist cannot notice a NEW overlapping name shape once the pair is listed, so
    the size is pinned separately from the safety property above; the two stay independent tests
    so a legitimate vocabulary growth reports as a size change while the safety check still runs.
    """
    overlaps = find_pattern_overlaps(collect_prefix_patterns(FAMILY_CLASSES), family_vocabulary(FAMILY_CLASSES))
    actual = {pair: len(overlaps.get(pair, [])) for pair in KNOWN_PATTERN_OVERLAPS}
    assert actual == dict(KNOWN_PATTERN_OVERLAPS), (
        f"Allowlisted overlap sizes changed: expected {dict(KNOWN_PATTERN_OVERLAPS)}, got {actual}. "
        "A family vocabulary grew (or shrank) a name shape the other family's pattern also raw-matches. "
        "Confirm test_known_overlaps_are_present_and_still_disambiguated passes (routing still rejects "
        "every overlapping name), then update the pinned count in KNOWN_PATTERN_OVERLAPS."
    )


# --- negative tests: prove the detectors actually fire ------------------------


class _AcceptAll:
    @classmethod
    def match_feature_group_criteria(cls, feature_name: object, options: object, _dac: object = None) -> bool:
        return True


class _AcceptNone:
    @classmethod
    def match_feature_group_criteria(cls, feature_name: object, options: object, _dac: object = None) -> bool:
        return False


def test_find_collisions_detects_multi_match() -> None:
    collisions = find_collisions(
        {"alpha": ("col__op_x",)}, PERMISSIVE_OPTIONS, {"alpha": _AcceptAll, "beta": _AcceptAll}
    )
    assert len(collisions) == 1
    assert collisions[0].kind == "MULTI_MATCH"
    assert set(collisions[0].acceptors) == {"alpha", "beta"}


def test_find_collisions_detects_owner_rejects() -> None:
    collisions = find_collisions({"alpha": ("col__op_x",)}, PERMISSIVE_OPTIONS, {"alpha": _AcceptNone})
    assert len(collisions) == 1
    assert collisions[0].kind == "OWNER_REJECTS"


def test_find_pattern_overlaps_detects_planted_overlap() -> None:
    patterns = {"alpha": [re.compile(r".*_zz$")], "beta": [re.compile(r".*_zz$")]}
    names = {"alpha": ("col__op_zz",), "beta": ("col__op_yy",)}
    overlaps = find_pattern_overlaps(patterns, names)
    assert overlaps == {("alpha", "beta"): ["col__op_zz"]}
    assert all(owner != other for owner, other in overlaps)

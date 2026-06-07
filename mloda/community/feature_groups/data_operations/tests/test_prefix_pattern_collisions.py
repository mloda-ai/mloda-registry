"""Lint: data-operation families must not claim overlapping feature-name patterns.

Feature resolution is name-driven: the matcher routes a feature like
``value_int__sum_agg`` to a family by testing that family's ``PREFIX_PATTERN``
(and, for ``frame_aggregate``, its four real module-level patterns). As the
catalog grows, two families could claim overlapping name patterns, producing
order-dependent or ambiguous routing that nothing currently catches. This module
is the guardrail (issue #248), in the same spirit as the reflection invariants
in ``test_framework_support_matrix.py``.

Two complementary checks run here:

- ``test_no_routing_collisions`` is the authoritative invariant. For every
  family's representative *valid* feature names, exactly one family's
  ``match_feature_group_criteria`` must accept the name. This uses the real
  router, so a raw-regex overlap that subtype validation disambiguates (e.g.
  ``window_aggregation`` rejecting ``sales__avg_7_day_window``) is not flagged.

- ``test_no_unexpected_pattern_overlaps`` is the blunter check the issue asks
  for: it collects every family's ``PREFIX_PATTERN`` and fails if a
  representative name of one family is also matched (raw regex) by another
  family's pattern, unless the pair is in :data:`KNOWN_PATTERN_OVERLAPS` (a
  documented allowlist whose entries double as regression guards that the
  overlap is still disambiguated at routing time). This check is sampled over
  the representative names (the issue's chosen primitive), not an exhaustive
  regex-intersection proof.

``test_every_family_is_covered`` guards against a new family ``base.py`` being
added without a registry entry, so nothing escapes the lint.

The negative tests (``test_find_*_detects_*``) feed planted collisions to the
detector functions so the guardrail is proven to fire, not merely to pass.
"""

from __future__ import annotations

import importlib
import re
from typing import Any
from dataclasses import dataclass
from pathlib import Path

from mloda.core.abstract_plugins.components.options import Options


REPO_ROOT = Path(__file__).resolve().parents[5]
DATA_OPERATIONS_ROOT = REPO_ROOT / "mloda" / "community" / "feature_groups" / "data_operations"

# Matches a class-level ``PREFIX_PATTERN = ...`` / ``PREFIX_PATTERN: ...`` definition,
# not a mere mention in a docstring or comment.
_PREFIX_DEF_RE = re.compile(r"^\s*PREFIX_PATTERN\s*[:=]", re.MULTILINE)

# Structural-only options: enough for families that require ``partition_by`` /
# ``order_by`` to accept their string-based names, but deliberately free of any
# operation-type key (``aggregation_type``, ``frame_type``, ...) so no family can
# match via the config-based path. That keeps matching purely name-driven, which
# is what this lint is about.
PERMISSIVE_OPTIONS = Options(context={"partition_by": ["g"], "order_by": "t"})


@dataclass(frozen=True)
class FamilySpec:
    """A data-operation family: where its base class lives, its representative
    valid feature names, and how to collect its matching patterns."""

    key: str
    module: str
    class_name: str
    representative_names: tuple[str, ...]
    # ``frame_aggregate``'s PREFIX_PATTERN is a rolling-only placeholder; its real
    # matching uses four module-level compiled patterns. When ``pattern_module_attrs``
    # is set these module attributes are collected, and ``ignore_prefix_pattern``
    # drops the misleading placeholder.
    pattern_module_attrs: tuple[str, ...] = ()
    ignore_prefix_pattern: bool = False


_DO = "mloda.community.feature_groups.data_operations"

FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec("aggregation", f"{_DO}.aggregation.base", "AggregationFeatureGroup", ("sales__sum_agg",)),
    FamilySpec(
        "binning", f"{_DO}.row_preserving.binning.base", "BinningFeatureGroup", ("value__bin_5", "value__qbin_10")
    ),
    FamilySpec(
        "datetime", f"{_DO}.row_preserving.datetime.base", "DateTimeFeatureGroup", ("ts__year", "ts__dayofweek")
    ),
    FamilySpec("ema", f"{_DO}.row_preserving.ema.base", "EmaFeatureGroup", ("price__ema_20",)),
    FamilySpec("ffill", f"{_DO}.row_preserving.ffill.base", "FfillFeatureGroup", ("sales__ffill",)),
    FamilySpec(
        "frame_aggregate",
        f"{_DO}.row_preserving.frame_aggregate.base",
        "FrameAggregateFeatureGroup",
        ("sales__sum_rolling_3", "sales__avg_7_day_window", "sales__cumsum", "sales__expanding_avg"),
        pattern_module_attrs=("_ROLLING_PATTERN", "_TIME_WINDOW_PATTERN", "_CUMULATIVE_PATTERN", "_EXPANDING_PATTERN"),
        ignore_prefix_pattern=True,
    ),
    FamilySpec("offset", f"{_DO}.row_preserving.offset.base", "OffsetFeatureGroup", ("sales__lag_1_offset",)),
    FamilySpec(
        "percentile", f"{_DO}.row_preserving.percentile.base", "PercentileFeatureGroup", ("sales__p50_percentile",)
    ),
    FamilySpec(
        "point_arithmetic",
        f"{_DO}.row_preserving.point_arithmetic.base",
        "PointArithmeticFeatureGroup",
        ("x__add_point",),
    ),
    FamilySpec("rank", f"{_DO}.row_preserving.rank.base", "RankFeatureGroup", ("sales__row_number_ranked",)),
    FamilySpec(
        "resample", f"{_DO}.row_changing.resample.base", "ResampleFeatureGroup", ("metric__resample_60_minute_mean",)
    ),
    FamilySpec(
        "scalar_aggregate",
        f"{_DO}.row_preserving.scalar_aggregate.base",
        "ScalarAggregateFeatureGroup",
        ("value__sum_scalar",),
    ),
    FamilySpec(
        "scalar_arithmetic",
        f"{_DO}.row_preserving.scalar_arithmetic.base",
        "ScalarArithmeticFeatureGroup",
        ("value__add_constant",),
    ),
    FamilySpec(
        "sessionization",
        f"{_DO}.row_preserving.sessionization.base",
        "SessionizationFeatureGroup",
        ("session__sessionize_30_minute",),
    ),
    FamilySpec("string", f"{_DO}.string.base", "StringFeatureGroup", ("name__upper",)),
    FamilySpec(
        "time_bucketization",
        f"{_DO}.row_preserving.time_bucketization.base",
        "TimeBucketizationFeatureGroup",
        ("ts__floor_30_minute",),
    ),
    FamilySpec(
        "window_aggregation",
        f"{_DO}.row_preserving.window_aggregation.base",
        "WindowAggregationFeatureGroup",
        ("sales__sum_window",),
    ),
)


# (owner_key, representative_name, other_key): raw-regex PREFIX_PATTERN overlaps that
# are known-safe because the other family's ``match_feature_group_criteria`` rejects
# the name via subtype validation. Each entry is asserted (below) to still be both a
# real raw overlap and disambiguated at routing time, so a stale entry fails loudly.
KNOWN_PATTERN_OVERLAPS: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("frame_aggregate", "sales__avg_7_day_window", "window_aggregation"),
    }
)


@dataclass(frozen=True)
class Collision:
    """A representative name whose set of accepting families is not exactly its owner."""

    representative_name: str
    owner: str
    acceptors: tuple[str, ...]

    @property
    def kind(self) -> str:
        return "OWNER_REJECTS" if self.owner not in self.acceptors else "MULTI_MATCH"


def _load_class(spec: FamilySpec) -> Any:
    return getattr(importlib.import_module(spec.module), spec.class_name)


def collect_prefix_patterns(families: tuple[FamilySpec, ...]) -> dict[str, list[re.Pattern[str]]]:
    """Collect each family's matching patterns as compiled regexes.

    Uses ``PREFIX_PATTERN`` by default. For families that declare
    ``pattern_module_attrs`` (``frame_aggregate``) the named module-level patterns
    are collected, and when ``ignore_prefix_pattern`` is set the placeholder
    ``PREFIX_PATTERN`` is skipped.
    """
    patterns: dict[str, list[re.Pattern[str]]] = {}
    for spec in families:
        module = importlib.import_module(spec.module)
        compiled: list[re.Pattern[str]] = []
        if not spec.ignore_prefix_pattern:
            compiled.append(re.compile(getattr(module, spec.class_name).PREFIX_PATTERN))
        for attr in spec.pattern_module_attrs:
            value = getattr(module, attr)
            compiled.append(value if isinstance(value, re.Pattern) else re.compile(value))
        patterns[spec.key] = compiled
    return patterns


def find_pattern_overlaps(
    family_patterns: dict[str, list[re.Pattern[str]]],
    families: tuple[FamilySpec, ...],
) -> list[tuple[str, str, str]]:
    """Raw-regex overlaps as ``(owner_key, representative_name, other_key)``.

    For each family's representative names, report every *other* family whose
    collected patterns also match the name (``re.Pattern.search``). The allowlist
    is NOT applied here; callers filter against :data:`KNOWN_PATTERN_OVERLAPS`.
    """
    overlaps: list[tuple[str, str, str]] = []
    for spec in families:
        for name in spec.representative_names:
            for other_key, compiled in family_patterns.items():
                if other_key == spec.key:
                    continue
                if any(pattern.search(name) for pattern in compiled):
                    overlaps.append((spec.key, name, other_key))
    return overlaps


def find_collisions(
    families: tuple[FamilySpec, ...],
    options: Options,
    classes: dict[str, Any],
) -> list[Collision]:
    """Routing-level collisions via ``match_feature_group_criteria``.

    For each family's representative names, the set of families in ``classes``
    that accept the name must be exactly ``{owner}``. Returns a :class:`Collision`
    for every name where that does not hold (owner rejects its own name, or more
    than one family accepts).
    """
    collisions: list[Collision] = []
    for spec in families:
        for name in spec.representative_names:
            acceptors = tuple(key for key, cls in classes.items() if cls.match_feature_group_criteria(name, options))
            if acceptors != (spec.key,):
                collisions.append(Collision(name, spec.key, acceptors))
    return collisions


def discover_uncovered_families() -> list[str]:
    """Dotted module paths of ``data_operations`` ``base.py`` files that define a
    ``PREFIX_PATTERN`` but are absent from :data:`FAMILIES`."""
    covered = {spec.module for spec in FAMILIES}
    missing: list[str] = []
    for base_py in DATA_OPERATIONS_ROOT.rglob("base.py"):
        if not _PREFIX_DEF_RE.search(base_py.read_text()):
            continue
        module = ".".join(base_py.relative_to(REPO_ROOT).with_suffix("").parts)
        if module not in covered:
            missing.append(module)
    return sorted(missing)


def _format_collision(collision: Collision) -> str:
    if collision.kind == "OWNER_REJECTS":
        return (
            f"{collision.representative_name!r}: owner {collision.owner!r} does not accept its own "
            f"representative name (acceptors={list(collision.acceptors)}). "
            "Fix the representative name or the family's matcher."
        )
    return (
        f"{collision.representative_name!r}: routed to multiple families {list(collision.acceptors)} "
        f"(owner={collision.owner!r}). Two families claim overlapping feature-name patterns."
    )


def _real_classes() -> dict[str, Any]:
    return {spec.key: _load_class(spec) for spec in FAMILIES}


# --- authoritative invariants -------------------------------------------------


def test_no_routing_collisions() -> None:
    collisions = find_collisions(FAMILIES, PERMISSIVE_OPTIONS, _real_classes())
    assert collisions == [], "Routing collisions across data-operation families:\n" + "\n".join(
        _format_collision(c) for c in collisions
    )


def test_no_unexpected_pattern_overlaps() -> None:
    overlaps = find_pattern_overlaps(collect_prefix_patterns(FAMILIES), FAMILIES)
    unexpected = [o for o in overlaps if o not in KNOWN_PATTERN_OVERLAPS]
    assert unexpected == [], (
        "Unexpected PREFIX_PATTERN overlaps (a representative name of one family is also "
        "matched by another family's pattern):\n  "
        + "\n  ".join(
            f"{name!r} (owner={owner}) also matched by {other}'s pattern" for owner, name, other in unexpected
        )
    )


def test_known_overlaps_are_present_and_still_disambiguated() -> None:
    actual = set(find_pattern_overlaps(collect_prefix_patterns(FAMILIES), FAMILIES))
    classes = _real_classes()
    for owner, name, other in KNOWN_PATTERN_OVERLAPS:
        assert (owner, name, other) in actual, (
            f"Allowlisted overlap {(owner, name, other)} no longer occurs as a raw-regex overlap; "
            "remove the stale KNOWN_PATTERN_OVERLAPS entry."
        )
        assert not classes[other].match_feature_group_criteria(name, PERMISSIVE_OPTIONS), (
            f"Allowlisted overlap {(owner, name, other)} is no longer disambiguated: {other} now "
            f"accepts {name!r}. This is a real collision, not a safe overlap."
        )


def test_every_family_is_covered() -> None:
    uncovered = discover_uncovered_families()
    assert uncovered == [], (
        "These data-operation base.py modules define a PREFIX_PATTERN but are missing from "
        "FAMILIES (extend it so the collision lint covers them):\n  " + "\n  ".join(uncovered)
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
    specs = (FamilySpec("alpha", "m", "C", ("col__op_x",)),)
    collisions = find_collisions(specs, PERMISSIVE_OPTIONS, {"alpha": _AcceptAll, "beta": _AcceptAll})
    assert len(collisions) == 1
    assert collisions[0].kind == "MULTI_MATCH"
    assert set(collisions[0].acceptors) == {"alpha", "beta"}


def test_find_collisions_detects_owner_rejects() -> None:
    specs = (FamilySpec("alpha", "m", "C", ("col__op_x",)),)
    collisions = find_collisions(specs, PERMISSIVE_OPTIONS, {"alpha": _AcceptNone})
    assert len(collisions) == 1
    assert collisions[0].kind == "OWNER_REJECTS"


def test_find_pattern_overlaps_detects_planted_overlap() -> None:
    specs = (
        FamilySpec("alpha", "m", "C", ("col__op_zz",)),
        FamilySpec("beta", "m", "C", ("col__op_yy",)),
    )
    patterns = {"alpha": [re.compile(r".*_zz$")], "beta": [re.compile(r".*_zz$")]}
    overlaps = find_pattern_overlaps(patterns, specs)
    assert ("alpha", "col__op_zz", "beta") in overlaps
    assert all(owner != other for owner, _name, other in overlaps)

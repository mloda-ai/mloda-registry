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
  ``test_routing_is_exhaustive_over_generated_names`` raises the bar from the
  hand-picked representatives to each family's full vocabulary, generated from
  the family's own live op-type table (see :data:`NAME_GENERATORS`), so a
  collision on an unsampled categorical variant cannot slip through.

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
from collections.abc import Callable
from dataclasses import dataclass, replace
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


# --- exhaustive valid-name generators -----------------------------------------


# One generator per ``FamilySpec.key``, each producing that family's full set of
# valid feature names from the family's OWN live vocabulary (imported, never
# re-listed here). This lets routing be verified over every categorical op-type
# variant instead of a hand-picked sample. The ``NAME_GENERATORS`` registry below
# maps each FAMILIES key to its generator, and ``generate_valid_names`` looks up
# and calls it.
def _gen_aggregation() -> tuple[str, ...]:
    # Vocabulary: AGGREGATION_TYPES (aggregation_base). Names: ``{col}__{type}_agg``.
    from mloda.community.feature_groups.data_operations.aggregation_base import AGGREGATION_TYPES

    return tuple(f"sales__{t}_agg" for t in AGGREGATION_TYPES)


def _gen_window_aggregation() -> tuple[str, ...]:
    # Vocabulary: window_aggregation.base.AGGREGATION_TYPES. Names: ``{col}__{type}_window``.
    from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import AGGREGATION_TYPES

    return tuple(f"sales__{t}_window" for t in AGGREGATION_TYPES)


def _gen_scalar_aggregate() -> tuple[str, ...]:
    # Vocabulary: scalar_aggregate.base.AGGREGATION_TYPES. Names: ``{col}__{type}_scalar``.
    from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.base import AGGREGATION_TYPES

    return tuple(f"value__{t}_scalar" for t in AGGREGATION_TYPES)


def _gen_scalar_arithmetic() -> tuple[str, ...]:
    # Vocabulary: scalar_arithmetic.base.ARITHMETIC_OPERATIONS. Names: ``{col}__{op}_constant``.
    from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
        ARITHMETIC_OPERATIONS,
    )

    return tuple(f"value__{op}_constant" for op in ARITHMETIC_OPERATIONS)


def _gen_point_arithmetic() -> tuple[str, ...]:
    # Vocabulary: point_arithmetic.base.ARITHMETIC_OPERATIONS. Names: ``{col}__{op}_point``.
    from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
        ARITHMETIC_OPERATIONS,
    )

    return tuple(f"x__{op}_point" for op in ARITHMETIC_OPERATIONS)


def _gen_rank() -> tuple[str, ...]:
    # Vocabulary: RankFeatureGroup.RANK_TYPES (rank/base.py). Names: ``{col}__{type}_ranked``.
    from mloda.community.feature_groups.data_operations.row_preserving.rank.base import RankFeatureGroup

    return tuple(f"sales__{t}_ranked" for t in RankFeatureGroup.RANK_TYPES)


def _gen_offset() -> tuple[str, ...]:
    # Vocabulary: OffsetFeatureGroup.OFFSET_TYPES (static) plus the inline dynamic prefixes
    # ``lag_/lead_/diff_/pct_change_`` + digit >= 1 accepted by ``_supports_offset_type``
    # (offset/base.py:123). Names: ``{col}__{type}_offset``.
    from mloda.community.feature_groups.data_operations.row_preserving.offset.base import OffsetFeatureGroup

    static = tuple(f"sales__{t}_offset" for t in OffsetFeatureGroup.OFFSET_TYPES)
    dynamic = tuple(f"sales__{p}1_offset" for p in ("lag_", "lead_", "diff_", "pct_change_"))
    return static + dynamic


def _gen_binning() -> tuple[str, ...]:
    # Vocabulary: BINNING_OPS (binning/base.py). Names: ``{col}__{op}_{N}``; both rep numerics
    # (bin_5, qbin_10) are emitted so reps stay a subset of the generated set.
    from mloda.community.feature_groups.data_operations.row_preserving.binning.base import BINNING_OPS

    return tuple(f"value__{op}_{n}" for op in BINNING_OPS for n in (5, 10))


def _gen_datetime() -> tuple[str, ...]:
    # Vocabulary: DATETIME_OPS (datetime/base.py). The op is the whole suffix: ``{col}__{op}``.
    from mloda.community.feature_groups.data_operations.row_preserving.datetime.base import DATETIME_OPS

    return tuple(f"ts__{c}" for c in DATETIME_OPS)


def _gen_string() -> tuple[str, ...]:
    # Vocabulary: STRING_OPS (string/base.py). The op is the whole suffix: ``{col}__{op}``.
    from mloda.community.feature_groups.data_operations.string.base import STRING_OPS

    return tuple(f"name__{op}" for op in STRING_OPS)


def _gen_percentile() -> tuple[str, ...]:
    # Pattern ``(p\d+)_percentile`` (percentile/base.py); numeric slot, one fixed value suffices.
    return ("sales__p50_percentile",)


def _gen_ema() -> tuple[str, ...]:
    # Pattern ``ema_\d+`` (ema/base.py); numeric slot, one fixed value suffices.
    return ("price__ema_20",)


def _gen_ffill() -> tuple[str, ...]:
    # Pattern ``__ffill`` (ffill/base.py); no vocabulary.
    return ("sales__ffill",)


def _gen_sessionization() -> tuple[str, ...]:
    # Vocabulary: SESSIONIZATION_UNITS (sessionization/base.py). Names: ``{col}__sessionize_{N}_{unit}``.
    from mloda.community.feature_groups.data_operations.row_preserving.sessionization.base import SESSIONIZATION_UNITS

    return tuple(f"session__sessionize_30_{unit}" for unit in SESSIONIZATION_UNITS)


def _gen_time_bucketization() -> tuple[str, ...]:
    # Vocabulary: TIME_BUCKETIZATION_OPS x TIME_BUCKETIZATION_UNITS (time_bucketization/base.py).
    # Names: ``{col}__{op}_{N}_{unit}``; _parse_bucket_op only accepts n=1 for the calendar units
    # in _CALENDAR_UNITS (base.py:103), so the numeric slot is 1 there and a fixed 30 otherwise.
    from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.base import (
        _CALENDAR_UNITS,
        TIME_BUCKETIZATION_OPS,
        TIME_BUCKETIZATION_UNITS,
    )

    return tuple(
        f"ts__{op}_{1 if unit in _CALENDAR_UNITS else 30}_{unit}"
        for op in TIME_BUCKETIZATION_OPS
        for unit in TIME_BUCKETIZATION_UNITS
    )


def _gen_resample() -> tuple[str, ...]:
    # Vocabulary: RESAMPLE_AGGS x RESAMPLE_UNITS (resample/base.py). Names: ``{col}__resample_{N}_{unit}_{agg}``.
    from mloda.community.feature_groups.data_operations.row_changing.resample.base import RESAMPLE_AGGS, RESAMPLE_UNITS

    return tuple(f"metric__resample_60_{unit}_{agg}" for unit in RESAMPLE_UNITS for agg in RESAMPLE_AGGS)


def _gen_frame_aggregate() -> tuple[str, ...]:
    # Vocabulary: _AGGREGATION_TYPES (and _TIME_UNITS) from frame_aggregate/base.py, expanded over
    # the four module-level patterns (_ROLLING/_TIME_WINDOW/_CUMULATIVE/_EXPANDING_PATTERN).
    from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
        _AGGREGATION_TYPES,
        _TIME_UNITS,
    )

    aggs = sorted(_AGGREGATION_TYPES)
    rolling = tuple(f"sales__{t}_rolling_3" for t in aggs)
    time_window = tuple(f"sales__{t}_7_{unit}_window" for t in aggs for unit in sorted(_TIME_UNITS))
    cumulative = tuple(f"sales__cum{t}" for t in aggs)
    expanding = tuple(f"sales__expanding_{t}" for t in aggs)
    return rolling + time_window + cumulative + expanding


NAME_GENERATORS: dict[str, Callable[[], tuple[str, ...]]] = {
    "aggregation": _gen_aggregation,
    "binning": _gen_binning,
    "datetime": _gen_datetime,
    "ema": _gen_ema,
    "ffill": _gen_ffill,
    "frame_aggregate": _gen_frame_aggregate,
    "offset": _gen_offset,
    "percentile": _gen_percentile,
    "point_arithmetic": _gen_point_arithmetic,
    "rank": _gen_rank,
    "resample": _gen_resample,
    "scalar_aggregate": _gen_scalar_aggregate,
    "scalar_arithmetic": _gen_scalar_arithmetic,
    "sessionization": _gen_sessionization,
    "string": _gen_string,
    "time_bucketization": _gen_time_bucketization,
    "window_aggregation": _gen_window_aggregation,
}


def generate_valid_names(spec: FamilySpec) -> tuple[str, ...]:
    """Exhaustive valid feature names for a family, from its live vocabulary.

    Returns the family's full set of valid feature names from its registered
    generator, or () if no generator is registered for the key."""
    gen = NAME_GENERATORS.get(spec.key)
    return gen() if gen is not None else ()


# --- authoritative invariants -------------------------------------------------


def test_no_routing_collisions() -> None:
    collisions = find_collisions(FAMILIES, PERMISSIVE_OPTIONS, _real_classes())
    assert collisions == [], "Routing collisions across data-operation families:\n" + "\n".join(
        _format_collision(c) for c in collisions
    )


def test_every_family_has_a_name_generator() -> None:
    missing = [spec.key for spec in FAMILIES if spec.key not in NAME_GENERATORS]
    assert missing == [], (
        "These families have no entry in NAME_GENERATORS (register a generator that builds each "
        "family's valid feature names from its live vocabulary):\n  " + "\n  ".join(missing)
    )


def test_representative_names_are_generated() -> None:
    for spec in FAMILIES:
        generated = set(generate_valid_names(spec))
        missing = sorted(set(spec.representative_names) - generated)
        assert missing == [], (
            f"{spec.key}: representative names {missing} are absent from the generated vocabulary. "
            "The sampled reps drifted from the exhaustive set; fix the generator or the reps."
        )


def test_routing_is_exhaustive_over_generated_names() -> None:
    exhaustive = tuple(replace(spec, representative_names=generate_valid_names(spec)) for spec in FAMILIES)
    total = sum(len(generate_valid_names(s)) for s in FAMILIES)
    assert total > len(FAMILIES), (
        "Generated vocabulary is vacuous (expected many names per family); NAME_GENERATORS is empty "
        "or under-populated, so exhaustive routing would pass trivially."
    )
    collisions = find_collisions(exhaustive, PERMISSIVE_OPTIONS, _real_classes())
    assert collisions == [], "Routing collisions over generated vocabulary:\n" + "\n".join(
        _format_collision(c) for c in collisions
    )


def test_generator_drift_is_caught_as_owner_rejects() -> None:
    spec = FAMILIES[0]
    bad = replace(spec, representative_names=(*generate_valid_names(spec), "col__NOTAREALOP_zzz"))
    collisions = find_collisions((bad,), PERMISSIVE_OPTIONS, _real_classes())
    assert any(c.kind == "OWNER_REJECTS" and c.representative_name == "col__NOTAREALOP_zzz" for c in collisions), (
        "Planted bogus name 'col__NOTAREALOP_zzz' was not flagged as OWNER_REJECTS for its owning "
        "family; the self-validation proof that generator drift surfaces as a collision does not hold."
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

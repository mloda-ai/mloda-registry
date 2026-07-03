"""Cross-cutting guards for ``return_data_type_rule`` across all ten data operations.

Decision (mloda core `PR #493 <https://github.com/mloda-ai/mloda/pull/493>`_):
``return_data_type_rule`` runs only AFTER a feature group has been SELECTED, and
mloda core calls it UNGUARDED (fail-fast). A raise from the rule is therefore a
bug in a committed component that SHOULD surface, not something to swallow. The
per-rule ``except Exception: return None`` catches in the data-operation bases
have been REMOVED. Core does NOT wrap the call.

These tests consolidate the per-operation ``TestReturnDataTypeRule`` spot checks
into two contracts that span every base class declaring a deterministic output
type, framed around the post-selection reality:

- ``test_completeness`` (the CI guard): for every supported *deterministic* op, a
  representative feature must make ``return_data_type_rule`` return its expected
  non-``None`` ``DataType``. With the catches removed, this is the spine that
  catches an extraction regression (a broken ``_extract_*`` helper) directly, as a
  CI failure, rather than the type quietly degrading to ``None``.

- ``test_matching_feature_never_raises`` (the post-selection invariant): because
  the rule is only ever called for a SELECTED (matching) feature, it must, for its
  representative MATCHING feature(s), return ``None`` or a ``DataType`` and never
  raise. Feeding unselected garbage directly is out of scope by construction: such
  features never reach the rule.

- ``test_matched_numeric_boundary_never_raises`` (the generalized numeric-boundary
  fuzzer): the same post-selection invariant, but proven for EVERY family instead of
  a hand-picked few. It reuses the ``FAMILIES`` registry from
  ``test_prefix_pattern_collisions.py``, takes each family's generated valid feature
  names, mutates every numeric slot to a non-positive/zero count (``0`` and ``00``),
  and asserts that any variant a family still SELECTS also types without raising. This
  replaces the old hand-picked ``BOUNDARY_CASES`` numeric rows (binning/resample), so
  a future op with a loose ``\\d+`` name pattern cannot silently reintroduce the gap.
  A family's numeric-axis coverage depends on its generator (``generate_valid_names``)
  emitting a digit-bearing exemplar; ``offset`` and ``rank`` do so explicitly for their
  dynamic ``lag_/lead_`` and ``ntile_/top_/bottom_`` numeric forms (whose ``>= 1`` guard
  lives in the extractor, not the pattern).
  The config-only axis that is not derivable from the name registry (a
  ``frame_aggregate`` config feature missing ``in_features``) stays explicit in
  ``test_matched_config_boundary_never_raises``.

The (base class, representative feature, expected ``DataType``) tuples mirror the
existing ``TestReturnDataTypeRule`` classes in each operation's ``tests/test_base.py``;
they are not invented here.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.user import DataType

# Reuse the sibling module's family registry (DRY): the same FAMILIES tuple, valid-name
# generators, class loader, and name-driven Options that drive the routing-collision lint.
# Importing the module-private ``_load_class`` from within the same test package is fine;
# the underscore only marks it module-private, not import-forbidden.
from mloda.community.feature_groups.data_operations.tests.test_prefix_pattern_collisions import (
    FAMILIES,
    PERMISSIVE_OPTIONS,
    _load_class,
    generate_valid_names,
)

from mloda.community.feature_groups.data_operations.aggregation.base import AggregationFeatureGroup
from mloda.community.feature_groups.data_operations.row_changing.resample.base import ResampleFeatureGroup
from mloda.community.feature_groups.data_operations.row_preserving.binning.base import BinningFeatureGroup
from mloda.community.feature_groups.data_operations.row_preserving.datetime.base import DateTimeFeatureGroup
from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
    FrameAggregateFeatureGroup,
)
from mloda.community.feature_groups.data_operations.row_preserving.offset.base import OffsetFeatureGroup
from mloda.community.feature_groups.data_operations.row_preserving.rank.base import RankFeatureGroup
from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.base import (
    ScalarAggregateFeatureGroup,
)
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)

# StringFeatureGroup lives under data_operations.string (not row_preserving).
from mloda.community.feature_groups.data_operations.string.base import StringFeatureGroup


# The ten base classes that declare a deterministic output type. Kept as a named
# tuple so both tests can assert they cover all ten.
ALL_BASE_CLASSES: tuple[type[FeatureGroup], ...] = (
    AggregationFeatureGroup,
    BinningFeatureGroup,
    DateTimeFeatureGroup,
    FrameAggregateFeatureGroup,
    OffsetFeatureGroup,
    RankFeatureGroup,
    ResampleFeatureGroup,
    ScalarAggregateFeatureGroup,
    StringFeatureGroup,
    WindowAggregationFeatureGroup,
)


def _opts(**context: object) -> Options:
    """Build an Options carrying only a planning context (matches the source tests)."""
    return Options(context=dict(context)) if context else Options()


# (base class, representative feature, expected non-None DataType) for every supported
# deterministic op, one row per op. Names and options copied verbatim from each
# operation's TestReturnDataTypeRule in tests/test_base.py; do not invent new ones.
COMPLETENESS_CASES: list[tuple[type[FeatureGroup], Feature, DataType]] = [
    # aggregation: count / nunique -> INT64 (sum/avg are input-dependent -> None, not listed)
    (AggregationFeatureGroup, Feature("value_int__count_agg", options=_opts(partition_by=["region"])), DataType.INT64),
    (
        AggregationFeatureGroup,
        Feature("value_int__nunique_agg", options=_opts(partition_by=["region"])),
        DataType.INT64,
    ),
    # binning: bin / qbin -> INT64
    (BinningFeatureGroup, Feature("value_int__bin_5", options=Options()), DataType.INT64),
    (BinningFeatureGroup, Feature("value_int__qbin_4", options=Options()), DataType.INT64),
    # datetime: all ops are deterministic integer extractions -> INT64
    (DateTimeFeatureGroup, Feature("timestamp__year", options=Options()), DataType.INT64),
    (DateTimeFeatureGroup, Feature("timestamp__month", options=Options()), DataType.INT64),
    (DateTimeFeatureGroup, Feature("timestamp__day", options=Options()), DataType.INT64),
    (DateTimeFeatureGroup, Feature("timestamp__hour", options=Options()), DataType.INT64),
    (DateTimeFeatureGroup, Feature("timestamp__minute", options=Options()), DataType.INT64),
    (DateTimeFeatureGroup, Feature("timestamp__second", options=Options()), DataType.INT64),
    (DateTimeFeatureGroup, Feature("timestamp__dayofweek", options=Options()), DataType.INT64),
    (DateTimeFeatureGroup, Feature("timestamp__is_weekend", options=Options()), DataType.INT64),
    (DateTimeFeatureGroup, Feature("timestamp__quarter", options=Options()), DataType.INT64),
    # frame_aggregate: count -> INT64 (sum etc. input-dependent -> None)
    (
        FrameAggregateFeatureGroup,
        Feature("sales__count_rolling_3", options=_opts(partition_by=["region"], order_by="timestamp")),
        DataType.INT64,
    ),
    # offset: pct_change -> DOUBLE (lag/lead/diff/first_value/last_value preserve input -> None)
    (
        OffsetFeatureGroup,
        Feature("value_int__pct_change_1_offset", options=_opts(partition_by=["region"], order_by="value_int")),
        DataType.DOUBLE,
    ),
    # rank: row_number / rank / dense_rank / ntile -> INT64, percent_rank -> DOUBLE
    (
        RankFeatureGroup,
        Feature("value_int__row_number_ranked", options=_opts(partition_by=["region"], order_by="value_int")),
        DataType.INT64,
    ),
    (
        RankFeatureGroup,
        Feature("value_int__rank_ranked", options=_opts(partition_by=["region"], order_by="value_int")),
        DataType.INT64,
    ),
    (
        RankFeatureGroup,
        Feature("value_int__dense_rank_ranked", options=_opts(partition_by=["region"], order_by="value_int")),
        DataType.INT64,
    ),
    (
        RankFeatureGroup,
        Feature("value_int__ntile_4_ranked", options=_opts(partition_by=["region"], order_by="value_int")),
        DataType.INT64,
    ),
    (
        RankFeatureGroup,
        Feature("value_int__percent_rank_ranked", options=_opts(partition_by=["region"], order_by="value_int")),
        DataType.DOUBLE,
    ),
    # resample: count -> INT64 (mean/sum input-dependent -> None)
    (ResampleFeatureGroup, Feature("value__resample_5_minute_count", options=Options()), DataType.INT64),
    # scalar_aggregate: count -> INT64 (sum input-dependent -> None)
    (ScalarAggregateFeatureGroup, Feature("value_int__count_scalar", options=Options()), DataType.INT64),
    # string: length -> INT64 (upper/lower/trim/reverse deferred -> None)
    (StringFeatureGroup, Feature("name__length", options=Options()), DataType.INT64),
    # window_aggregation: count / nunique -> INT64 (avg input-dependent -> None)
    (
        WindowAggregationFeatureGroup,
        Feature("value_int__count_window", options=_opts(partition_by=["region"])),
        DataType.INT64,
    ),
    (
        WindowAggregationFeatureGroup,
        Feature("value_int__nunique_window", options=_opts(partition_by=["region"])),
        DataType.INT64,
    ),
]


def test_completeness_covers_every_base_class() -> None:
    """Sanity: the completeness table must touch all ten base classes (catches a
    new deterministic operation added without a representative row here)."""
    covered = {cls for cls, _feature, _expected in COMPLETENESS_CASES}
    assert covered == set(ALL_BASE_CLASSES)


@pytest.mark.parametrize(
    "feature_group_class, feature, expected",
    COMPLETENESS_CASES,
    ids=[f"{cls.__name__}:{feature.name}" for cls, feature, _expected in COMPLETENESS_CASES],
)
def test_completeness(feature_group_class: type[FeatureGroup], feature: Feature, expected: DataType) -> None:
    """Every supported deterministic op returns its declared non-None DataType.

    With the per-rule catches removed, a broken ``_extract_*`` helper no longer
    degrades the output type to ``None`` silently; this assertion is the CI guard
    that turns such an extraction regression into a visible failure.
    """
    result = feature_group_class.return_data_type_rule(feature)
    assert result == expected


@pytest.mark.parametrize(
    "feature_group_class, feature",
    [(cls, feature) for cls, feature, _expected in COMPLETENESS_CASES],
    ids=[f"{cls.__name__}:{feature.name}" for cls, feature, _expected in COMPLETENESS_CASES],
)
def test_matching_feature_never_raises(feature_group_class: type[FeatureGroup], feature: Feature) -> None:
    """For a SELECTED (matching) feature, the rule returns None-or-DataType, never raises.

    The rule is only ever invoked post-selection (mloda core PR #493), so the
    relevant contract is over MATCHING features, not unselected garbage. Every base
    class in ``ALL_BASE_CLASSES`` is exercised here because ``COMPLETENESS_CASES``
    covers all ten (guarded by ``test_completeness_covers_every_base_class``).
    """
    result = feature_group_class.return_data_type_rule(feature)
    assert result is None or isinstance(result, DataType)


# --- generalized numeric-boundary fuzzer (registry-driven) --------------------
#
# The numeric axis of the pattern/extractor-alignment contract, generalized across
# EVERY family via the reused FAMILIES registry (was a hand-picked binning/resample
# list). For each family we take its generated valid feature names and mutate every
# maximal digit-run to a non-positive/zero count. A loose ``\d+`` pattern would still
# SELECT such a name, but its extractor (needing a positive count/size) may not be
# able to type it. The contract under test: whatever matching SELECTS, the rule must
# type without raising. Variants a family does not select are out of scope, because
# the framework never routes them to the rule.

_DIGIT_RUN = re.compile(r"\d+")


def numeric_boundary_variants(name: str) -> set[str]:
    """Boundary variants of a valid feature name: each maximal digit-run replaced
    by a non-positive/zero-count value ('0' and '00') that a loose \\d+ pattern
    would still select but whose extractor (needing a positive count/size) cannot
    handle. Names with no digit-run yield the empty set."""
    variants: set[str] = set()
    for match in _DIGIT_RUN.finditer(name):
        for boundary in ("0", "00"):
            variants.add(name[: match.start()] + boundary + name[match.end() :])
    return variants


def _build_numeric_boundary_cases() -> tuple[list[tuple[str, Any, str]], frozenset[str]]:
    """Enumerate ``(family_key, class, variant)`` boundary cases over ALL families.

    Iterates every family in ``FAMILIES`` (recording the iterated keys so the
    coverage guard can prove no family is skipped), generates each family's valid
    names, and dedups their numeric-boundary variants per family. Families whose
    generated names carry no digit contribute zero variants; that is expected and
    guarded against becoming universal by ``test_numeric_boundary_fuzz_is_not_vacuous``.
    """
    cases: list[tuple[str, Any, str]] = []
    iterated: set[str] = set()
    for spec in FAMILIES:
        iterated.add(spec.key)
        cls = _load_class(spec)
        variants: set[str] = set()
        for name in generate_valid_names(spec):
            variants |= numeric_boundary_variants(name)
        for variant in sorted(variants):
            cases.append((spec.key, cls, variant))
    return cases, frozenset(iterated)


_NUMERIC_BOUNDARY_CASES, _FUZZED_FAMILY_KEYS = _build_numeric_boundary_cases()

# Families whose live vocabulary is known to contain a numeric slot (verified against
# generate_valid_names before hard-coding). Each MUST contribute at least one boundary
# variant; a zero here signals the generator or the family vocabulary drifted, which the
# non-vacuity guard turns into a loud failure rather than silently shrinking coverage.
_KNOWN_NUMERIC_FAMILIES: frozenset[str] = frozenset(
    {
        "binning",
        "resample",
        "frame_aggregate",
        "offset",
        "rank",
        "ema",
        "percentile",
        "sessionization",
        "time_bucketization",
    }
)


def _boundary_case_raises(cls: Any, variant: str) -> bool:
    """Run the post-selection contract for one boundary variant; return True iff it is a gap.

    If the family does not SELECT the variant it is out of scope (the framework never
    routes it to the rule), so return ``False``. If it is selected, ``return_data_type_rule``
    must type it (``None`` or a ``DataType``) without raising: a raise returns ``True`` (the
    gap), and a non-``DataType``/non-``None`` return trips the inner assertion. Selection uses
    ``PERMISSIVE_OPTIONS`` so matching stays name-driven (the axis this fuzzer covers).

    Shared by the real parametrized guard and the self-proof negative test so the two
    cannot diverge.
    """
    if not cls.match_feature_group_criteria(variant, PERMISSIVE_OPTIONS):
        return False
    try:
        result = cls.return_data_type_rule(Feature(variant, options=PERMISSIVE_OPTIONS))
    except Exception:
        return True
    assert result is None or isinstance(result, DataType)
    return False


def test_numeric_boundary_fuzz_covers_every_family() -> None:
    """The fuzzer must iterate EVERY family in the registry (no family silently skipped).

    Piggybacks on ``test_prefix_pattern_collisions.test_every_family_is_covered``, which
    forces ``FAMILIES`` to stay complete as new op families are added.
    """
    assert _FUZZED_FAMILY_KEYS == {spec.key for spec in FAMILIES}, (
        "The numeric-boundary fuzzer did not iterate every family in FAMILIES; a family "
        "was skipped during case generation."
    )


def test_numeric_boundary_fuzz_is_not_vacuous() -> None:
    """The fuzzer must not silently collapse to (near) zero cases.

    Asserts a comfortably large total variant count and that each known-numeric family
    contributes at least one boundary variant. If a known-numeric key yields zero, that
    is a real drift signal in the generator or the family vocabulary; the assertion fails
    loudly rather than weakening coverage.
    """
    assert len(_NUMERIC_BOUNDARY_CASES) > 15, (
        f"Numeric-boundary fuzz is near-vacuous ({len(_NUMERIC_BOUNDARY_CASES)} cases); the "
        "generators or vocabularies likely lost their numeric slots."
    )
    per_family: dict[str, int] = {}
    for family_key, _cls, _variant in _NUMERIC_BOUNDARY_CASES:
        per_family[family_key] = per_family.get(family_key, 0) + 1
    missing = sorted(key for key in _KNOWN_NUMERIC_FAMILIES if per_family.get(key, 0) < 1)
    assert missing == [], (
        f"Known-numeric families produced no boundary variant: {missing}. Their generated valid "
        "names lost a digit slot (generator or vocabulary drift); fix the generator, do not drop the key."
    )


@pytest.mark.parametrize(
    "family_key, feature_group_class, variant",
    _NUMERIC_BOUNDARY_CASES,
    ids=[f"{family_key}:{variant}" for family_key, _cls, variant in _NUMERIC_BOUNDARY_CASES],
)
def test_matched_numeric_boundary_never_raises(family_key: str, feature_group_class: Any, variant: str) -> None:
    """Every family: a numeric-boundary variant it still SELECTS must type without raising.

    This is the generalized, registry-driven replacement for the old hand-picked numeric
    ``BOUNDARY_CASES``. A raise here is a REAL product gap (a loose numeric pattern selects a
    name whose extractor cannot handle a zero count/size); the fix belongs at the
    matching/validation layer (tighten the pattern or reject the input up front), not in a
    re-added catch. Variants a family does not select are out of scope and pass trivially.
    """
    assert not _boundary_case_raises(feature_group_class, variant), (
        f"{family_key}: matched numeric-boundary feature {variant!r} raised in return_data_type_rule. "
        "A loose numeric pattern selected a name whose extractor cannot handle a zero count/size. "
        "Fix at the matching/validation layer (tighten the pattern or reject the input), not with a catch."
    )


# --- self-proof: the boundary guard must actually fire ------------------------
#
# Mirrors test_prefix_pattern_collisions.test_find_collisions_detects_multi_match: feed a
# planted family that SELECTS a boundary variant and then raises, and assert the SHARED
# check flags it. This proves the guard is not trivially green.


class _AcceptsZeroAndRaises:
    @classmethod
    def match_feature_group_criteria(cls, feature_name: object, options: object, _dac: object = None) -> bool:
        return str(feature_name).endswith("_0")

    @classmethod
    def return_data_type_rule(cls, feature: object) -> DataType | None:
        raise ValueError("boom: extractor cannot handle n=0")


def test_boundary_guard_detects_a_selecting_raising_family() -> None:
    """A family that SELECTS a ``_0`` variant and raises must be flagged by the shared check."""
    assert _boundary_case_raises(_AcceptsZeroAndRaises, "value__bin_0"), (
        "The boundary guard did not flag a family that selects a numeric-boundary name and raises; "
        "the shared _boundary_case_raises check is not proving the guard fires."
    )


# --- config-only boundary axis (not derivable from the name registry) ---------
#
# The numeric axis above is now handled generically by the registry-driven fuzzer. This
# retained case covers the ONE boundary shape the name registry cannot express: a
# frame_aggregate CONFIG feature carrying a valid op/frame but NO in_features (source
# column). It stays explicit because it is options-driven, not name-driven.
BOUNDARY_CASES: list[tuple[type[FeatureGroup], str, Options]] = [
    (
        FrameAggregateFeatureGroup,
        "my_frame_agg",
        _opts(
            aggregation_type="count",
            frame_type="rolling",
            frame_size=3,
            partition_by=["region"],
            order_by="timestamp",
        ),
    ),
]


@pytest.mark.parametrize(
    "feature_group_class, feature_name, options",
    BOUNDARY_CASES,
    ids=[f"{cls.__name__}:{name}" for cls, name, _options in BOUNDARY_CASES],
)
def test_matched_config_boundary_never_raises(
    feature_group_class: type[FeatureGroup], feature_name: str, options: Options
) -> None:
    """Config-axis boundary: a frame_aggregate config feature missing ``in_features``.

    If ``match_feature_group_criteria`` SELECTS it, ``return_data_type_rule`` must type it
    without raising (``None`` or a ``DataType``). If it does not match, the framework never
    routes it to the rule, so it passes trivially. This axis is options-driven and cannot
    be generated from the name registry, so it stays explicit alongside the numeric fuzzer.
    """
    if not feature_group_class.match_feature_group_criteria(feature_name, options, None):
        return
    result = feature_group_class.return_data_type_rule(Feature(feature_name, options=options))
    assert result is None or isinstance(result, DataType)

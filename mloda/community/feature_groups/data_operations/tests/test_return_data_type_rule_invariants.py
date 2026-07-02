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

The (base class, representative feature, expected ``DataType``) tuples mirror the
existing ``TestReturnDataTypeRule`` classes in each operation's ``tests/test_base.py``;
they are not invented here.
"""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.user import DataType

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


# Pattern-BOUNDARY inputs: names/options that sit right at the edge of each pattern
# (zero bucket sizes, zero bins, a config feature missing in_features). These are the
# exact shapes where a loose pattern would SELECT a feature that the extractor then
# cannot handle. The contract under test: whatever matching SELECTS, the rule must be
# able to type without raising. If a case does not match, the framework never routes
# it to the rule, so it is out of scope (skipped implicitly by the guard).
BOUNDARY_CASES: list[tuple[type[FeatureGroup], str, Options]] = [
    (ResampleFeatureGroup, "value__resample_0_hour_mean", _opts(time_column="timestamp")),
    (ResampleFeatureGroup, "value__resample_00_hour_count", _opts(time_column="timestamp")),
    (BinningFeatureGroup, "value_int__bin_0", Options()),
    (BinningFeatureGroup, "value_int__qbin_0", Options()),
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
def test_matched_boundary_feature_never_raises(
    feature_group_class: type[FeatureGroup], feature_name: str, options: Options
) -> None:
    """Guards the pattern/extractor-alignment contract this change depends on.

    For each boundary input: if ``match_feature_group_criteria`` SELECTS it, then
    ``return_data_type_rule`` must type it without raising (``None`` or a ``DataType``).
    A pattern that selects a feature its extractor cannot handle would raise here.
    Inputs that do not match are never routed to the rule, so they pass trivially.
    """
    if not feature_group_class.match_feature_group_criteria(feature_name, options, None):
        return
    result = feature_group_class.return_data_type_rule(Feature(feature_name, options=options))
    assert result is None or isinstance(result, DataType)

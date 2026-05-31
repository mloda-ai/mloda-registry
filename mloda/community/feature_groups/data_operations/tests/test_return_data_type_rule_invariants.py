"""Cross-cutting guards for ``return_data_type_rule`` across all ten data operations.

These two tests codify the decision recorded in
``docs/guides/data-operation-patterns/16-return-data-type-rule.md`` for issue
`#244 <https://github.com/mloda-ai/mloda-registry/issues/244>`_. They consolidate
the per-operation ``TestReturnDataTypeRule`` spot checks into two contracts that
span every base class that declares a deterministic output type.

Two guarantees:

- ``test_completeness`` (doc option 5): for every supported *deterministic* op,
  a representative feature must make ``return_data_type_rule`` return its expected
  non-``None`` ``DataType``. This is the spine that catches the "hidden bug" path:
  a broken ``_extract_*`` helper that the broad ``except Exception`` would silently
  swallow into ``None``, disabling output-type validation with no signal. If an
  extraction regression ships, this test fails in CI rather than degrading quietly.

- ``test_invariant_never_raises`` (the explicit invariant): ``return_data_type_rule``
  is a planning-time hint that mloda core calls unguarded in ``engine.set_data_type``.
  A rule that raises crashes planning for the whole graph, so the rule must always
  RETURN (``None`` is an acceptable answer) and NEVER raise, even for malformed or
  garbage features. This pins the reason the broad catch exists so a future narrow
  cannot quietly regress it.

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

    A broken ``_extract_*`` helper would be swallowed by the rule's broad catch and
    silently return ``None`` (doc path 3). This assertion turns that into a CI failure.
    """
    result = feature_group_class.return_data_type_rule(feature)
    assert result == expected


# Malformed / garbage features: a name that matches no pattern, plus a config-style
# feature whose empty Options drives frame_aggregate's _extract_params through
# get_in_features() (which raises ValueError/TypeError) -- confirming the broad catch
# holds rather than letting the exception propagate.
MALFORMED_FEATURES: list[Feature] = [
    Feature("totally_unparseable_name"),
    # Empty in_features: frame_aggregate._extract_source_features -> get_in_features()
    # raises ValueError ("Input features not found in options"). Other rules just miss.
    Feature("frame_agg_no_in_features", options=Options()),
    # A garbage name with an explicit (empty-context) Options object.
    Feature("__bad__", options=Options()),
]


@pytest.mark.parametrize(
    "feature_group_class",
    ALL_BASE_CLASSES,
    ids=[cls.__name__ for cls in ALL_BASE_CLASSES],
)
def test_invariant_never_raises(feature_group_class: type[FeatureGroup]) -> None:
    """``return_data_type_rule`` must return (``None`` is acceptable) and never raise.

    Planning calls the rule unguarded; a raise crashes the whole graph. For every
    base class and every malformed feature, the rule must degrade to ``None`` or a
    ``DataType`` instead of propagating an exception.
    """
    for feature in MALFORMED_FEATURES:
        result = feature_group_class.return_data_type_rule(feature)
        assert result is None or isinstance(result, DataType)

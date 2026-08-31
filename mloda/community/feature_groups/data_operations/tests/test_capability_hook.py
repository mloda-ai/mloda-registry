"""Per-op, per-framework capability checks via ``supports_compute_framework`` (issue #247).

mloda core 0.9.0 evaluates ``FeatureGroup.supports_compute_framework(feature_name, options,
compute_framework)`` per feature at match time. The data_operations backends must override
it so that operations a backend cannot compute (e.g. ``median`` on SQLite) are rejected at
match time instead of failing later inside ``calculate_feature``. Backends stay conservative:
anything they cannot parse into an operation keeps the default ``True``.

The tests import only what the specific framework needs and skip when that framework's
optional dependency is missing.

The per-family capability matrices now live in each family's per-backend test module
(``tests/test_<backend>.py``) via the shared ``CapabilityHookTestMixin``
(``mloda/testing/feature_groups/data_operations/mixins/capability.py``). This module keeps only
the cross-family resolve_feature integration checks.

mloda core 0.11.0 removed the free function ``split_frameworks_by_capability``; the per-feature
capability check is unchanged and still public as the classmethod
``FeatureGroup.supports_compute_framework(feature_name, options, compute_framework_class)``. Also,
``resolve_feature(...).unsupported_compute_frameworks`` now reports only frameworks the *winning*
feature group itself rejects, not ones rejected by a sibling backend class, so a sibling's own
rejection (e.g. SQLite rejecting ``median``) is visible only in ``resolve_feature``'s elimination
text when the query is scoped to that class via ``feature_group=``, or by calling
``supports_compute_framework`` on it directly.
"""

from __future__ import annotations

import pytest
from mloda.user import Options

# ---------------------------------------------------------------------------
# Integration: resolve_feature surfaces the capability split
# ---------------------------------------------------------------------------


class TestResolveFeatureIntegration:
    def test_resolve_feature_splits_frameworks_for_median_scalar(self) -> None:
        """resolve_feature must surface SqliteFramework as rejected and PandasDataFrame as supported.

        resolve_feature evaluates matching under empty Options, and the group-by
        aggregation family requires partition_by to match, so the scalar aggregate
        family (matching string-based with empty Options) is the integration probe
        for the SQLite-rejects-median capability.

        Both queries scope via ``feature_group=`` to a concrete backend: without it, the shared,
        unrestricted base class is also a matching candidate and resolve_feature reports an
        ambiguous "Multiple feature groups found" instead of resolving to either backend. Both
        also pass a restricted ``enabled_feature_groups`` plugin_collector so an unrelated broken
        class from another test module can't fail the whole environment build.
        """
        pytest.importorskip("pandas")
        from mloda.steward import resolve_feature
        from mloda.user import PluginCollector

        from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.pandas_scalar_aggregate import (
            PandasScalarAggregate,
        )
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.sqlite_scalar_aggregate import (
            SqliteScalarAggregate,
        )

        plugin_collector = PluginCollector.enabled_feature_groups({PandasScalarAggregate, SqliteScalarAggregate})

        # SqliteScalarAggregate matches by name but supports_compute_framework rejects median for
        # its own (only) framework, SqliteFramework, so no candidate remains and the elimination
        # text names the rejected framework.
        sqlite_result = resolve_feature(
            "value__median_scalar", feature_group=SqliteScalarAggregate, plugin_collector=plugin_collector
        )
        assert sqlite_result.feature_group is None
        assert sqlite_result.error is not None
        assert "SqliteFramework" in sqlite_result.error

        # PandasScalarAggregate has no such restriction and resolves cleanly.
        pandas_result = resolve_feature(
            "value__median_scalar", feature_group=PandasScalarAggregate, plugin_collector=plugin_collector
        )
        assert pandas_result.feature_group is PandasScalarAggregate
        assert "PandasDataFrame" in pandas_result.supported_compute_frameworks
        assert pandas_result.unsupported_compute_frameworks == []

    def test_capability_split_rejects_sqlite_for_median_rolling_frame(self) -> None:
        """Each backend's own supports_compute_framework must reject/accept a median rolling frame.

        resolve_feature cannot be the probe here: frame aggregate's
        ``match_feature_group_criteria`` requires partition_by/order_by, which are absent
        under the empty Options resolve_feature evaluates matching with (the same reason the
        median-scalar test above uses the scalar family instead of the group-by family). The
        removed ``split_frameworks_by_capability`` used to batch this same per-class check for a
        caller-supplied list of candidates; this test calls each class's own
        ``supports_compute_framework`` directly against its own (single) compute framework
        instead, which is the exact per-feature hook that helper batched.
        """
        pytest.importorskip("pandas")
        pytest.importorskip("duckdb")

        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.duckdb_frame_aggregate import (
            DuckdbFrameAggregate,
        )
        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.pandas_frame_aggregate import (
            PandasFrameAggregate,
        )
        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.sqlite_frame_aggregate import (
            SqliteFrameAggregate,
        )

        feature_name = "value__median_rolling_3"

        assert SqliteFrameAggregate.supports_compute_framework(feature_name, Options(), SqliteFramework) is False
        assert PandasFrameAggregate.supports_compute_framework(feature_name, Options(), PandasDataFrame) is True
        assert DuckdbFrameAggregate.supports_compute_framework(feature_name, Options(), DuckDBFramework) is True

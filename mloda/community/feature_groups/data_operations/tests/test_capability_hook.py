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
"""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.options import Options


# ---------------------------------------------------------------------------
# Integration: resolve_feature surfaces the capability split
# ---------------------------------------------------------------------------


class TestResolveFeatureIntegration:
    def test_resolve_feature_splits_frameworks_for_median_scalar(self) -> None:
        """resolve_feature must list SqliteFramework as unsupported and PandasDataFrame as supported.

        resolve_feature evaluates matching under empty Options, and the group-by
        aggregation family requires partition_by to match, so the scalar aggregate
        family (matching string-based with empty Options) is the integration probe
        for the SQLite-rejects-median capability.
        """
        pytest.importorskip("pandas")
        from mloda.steward import resolve_feature

        # Importing the production classes registers them as FeatureGroup subclasses.
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.pandas_scalar_aggregate import (  # noqa: F401
            PandasScalarAggregate,
        )
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.sqlite_scalar_aggregate import (  # noqa: F401
            SqliteScalarAggregate,
        )

        result = resolve_feature("value__median_scalar")

        assert "SqliteFramework" in result.unsupported_compute_frameworks
        assert "PandasDataFrame" in result.supported_compute_frameworks

    def test_capability_split_rejects_sqlite_for_median_rolling_frame(self) -> None:
        """The capability split must reject SqliteFramework for a median rolling frame while keeping Pandas/DuckDB.

        resolve_feature cannot be the probe here: frame aggregate's
        ``match_feature_group_criteria`` requires partition_by/order_by, which are absent
        under the empty Options resolve_feature evaluates matching with (the same reason the
        median-scalar test above uses the scalar family instead of the group-by family). This
        test exercises ``split_frameworks_by_capability`` directly, which is exactly the
        mechanism resolve_feature uses internally to derive
        supported/unsupported_compute_frameworks.
        """
        pytest.importorskip("pandas")
        pytest.importorskip("duckdb")
        from mloda.core.prepare.identify_feature_group import split_frameworks_by_capability

        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.duckdb_frame_aggregate import (
            DuckdbFrameAggregate,
        )
        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.pandas_frame_aggregate import (
            PandasFrameAggregate,
        )
        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.sqlite_frame_aggregate import (
            SqliteFrameAggregate,
        )

        supported, rejected = split_frameworks_by_capability(
            [SqliteFrameAggregate, PandasFrameAggregate, DuckdbFrameAggregate],
            "value__median_rolling_3",
            Options(),
        )
        supported_names = {c.get_class_name() for c in supported}
        rejected_names = {c.get_class_name() for c in rejected}

        assert "SqliteFramework" in rejected_names
        assert "PandasDataFrame" in supported_names
        assert "DuckDBFramework" in supported_names

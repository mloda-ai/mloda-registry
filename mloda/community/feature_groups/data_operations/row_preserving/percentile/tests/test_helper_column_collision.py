"""Helper-column collision tests for percentile implementations.

These tests verify that user-provided columns whose names happen to match
internal hardcoded helper-column names survive the computation unmodified.

ReferencePercentile uses a ``__mloda_pctl_idx__`` helper column.
"""

from __future__ import annotations

import pytest


class TestReferencePercentileCollision:
    """User column named ``__mloda_pctl_idx__`` must survive ReferencePercentile."""

    def test_user_column_named_pctl_idx_survives(self) -> None:
        pa = pytest.importorskip("pyarrow")

        from mloda.testing.feature_groups.data_operations.row_preserving.percentile.reference import (
            ReferencePercentile,
        )

        table = pa.table(
            {
                "region": ["A", "A", "A", "B", "B"],
                "value": [10.0, 20.0, 30.0, 40.0, 50.0],
                "__mloda_pctl_idx__": ["u0", "u1", "u2", "u3", "u4"],
            }
        )

        result = ReferencePercentile._compute_percentile(
            table=table,
            feature_name="p50_value",
            source_col="value",
            partition_by=["region"],
            percentile=0.5,
        )

        assert "__mloda_pctl_idx__" in result.column_names
        assert result.column("__mloda_pctl_idx__").to_pylist() == ["u0", "u1", "u2", "u3", "u4"]
        assert "p50_value" in result.column_names
        assert result.column("p50_value").to_pylist() == [20.0, 20.0, 20.0, 45.0, 45.0]

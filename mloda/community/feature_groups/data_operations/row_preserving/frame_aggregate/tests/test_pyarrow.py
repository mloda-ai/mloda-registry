"""Tests for PyArrow frame aggregate implementation."""

from __future__ import annotations

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.pyarrow_frame_aggregate import (
    PyArrowFrameAggregate,
)


@pytest.fixture()
def sample_table() -> pa.Table:
    return pa.table(
        {
            "region": ["A", "A", "A", "A", "B", "B", "B"],
            "timestamp": [1, 2, 3, 4, 1, 2, 3],
            "value": [10, 20, 30, 40, 100, 200, 300],
        }
    )


class TestRollingWindow:
    def test_rolling_sum_window_2(self, sample_table: pa.Table) -> None:
        feature = Feature(
            "value__sum_rolling_2",
            options=Options(context={"partition_by": ["region"], "order_by": "timestamp"}),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PyArrowFrameAggregate.calculate_feature(sample_table, fs)
        col = result.column("value__sum_rolling_2").to_pylist()

        assert col[0] == 10
        assert col[1] == 30
        assert col[2] == 50
        assert col[3] == 70
        assert col[4] == 100
        assert col[5] == 300
        assert col[6] == 500

    def test_rolling_avg_window_3(self, sample_table: pa.Table) -> None:
        feature = Feature(
            "value__avg_rolling_3",
            options=Options(context={"partition_by": ["region"], "order_by": "timestamp"}),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PyArrowFrameAggregate.calculate_feature(sample_table, fs)
        col = result.column("value__avg_rolling_3").to_pylist()

        assert col[0] == 10.0
        assert col[1] == 15.0
        assert col[2] == 20.0
        assert col[3] == 30.0


class TestCumulative:
    def test_cumsum(self, sample_table: pa.Table) -> None:
        feature = Feature(
            "value__cumsum",
            options=Options(context={"partition_by": ["region"], "order_by": "timestamp"}),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PyArrowFrameAggregate.calculate_feature(sample_table, fs)
        col = result.column("value__cumsum").to_pylist()

        assert col[0] == 10
        assert col[1] == 30
        assert col[2] == 60
        assert col[3] == 100
        assert col[4] == 100
        assert col[5] == 300
        assert col[6] == 600

    def test_cummax(self, sample_table: pa.Table) -> None:
        feature = Feature(
            "value__cummax",
            options=Options(context={"partition_by": ["region"], "order_by": "timestamp"}),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PyArrowFrameAggregate.calculate_feature(sample_table, fs)
        col = result.column("value__cummax").to_pylist()

        assert col[0] == 10
        assert col[1] == 20
        assert col[2] == 30
        assert col[3] == 40
        assert col[4] == 100
        assert col[5] == 200
        assert col[6] == 300


class TestExpanding:
    def test_expanding_avg(self, sample_table: pa.Table) -> None:
        feature = Feature(
            "value__expanding_avg",
            options=Options(context={"partition_by": ["region"], "order_by": "timestamp"}),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PyArrowFrameAggregate.calculate_feature(sample_table, fs)
        col = result.column("value__expanding_avg").to_pylist()

        assert col[0] == 10.0
        assert col[1] == 15.0
        assert col[2] == 20.0
        assert col[3] == 25.0
        assert col[4] == 100.0
        assert col[5] == 150.0
        assert col[6] == 200.0


class TestConfigBased:
    def test_config_rolling(self, sample_table: pa.Table) -> None:
        feature = Feature(
            "my_rolling_sum",
            options=Options(
                context={
                    "aggregation_type": "sum",
                    "frame_type": "rolling",
                    "frame_size": 2,
                    "in_features": "value",
                    "partition_by": ["region"],
                    "order_by": "timestamp",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PyArrowFrameAggregate.calculate_feature(sample_table, fs)
        assert "my_rolling_sum" in result.column_names
        col = result.column("my_rolling_sum").to_pylist()
        assert col[0] == 10
        assert col[1] == 30


class TestNullHandling:
    def test_rolling_sum_skips_nulls(self) -> None:
        """Null values in the source column should be skipped by the aggregation."""
        table = pa.table(
            {
                "region": ["A", "A", "A", "A"],
                "ts": [1, 2, 3, 4],
                "value": [10, None, 30, 40],
            }
        )
        feature = Feature(
            "value__sum_rolling_3",
            options=Options(context={"partition_by": ["region"], "order_by": "ts"}),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PyArrowFrameAggregate.calculate_feature(table, fs)
        col = result.column("value__sum_rolling_3").to_pylist()

        # pos 0: window=[10] => 10
        # pos 1: window=[10, None] => 10
        # pos 2: window=[10, None, 30] => 40
        # pos 3: window=[None, 30, 40] => 70
        assert col == [10, 10, 40, 70]

    def test_cumsum_with_null_order_by(self) -> None:
        """Rows with null order_by values should be sorted last."""
        table = pa.table(
            {
                "region": ["A", "A", "A"],
                "ts": [None, 1, 2],
                "value": [100, 10, 20],
            }
        )
        feature = Feature(
            "value__cumsum",
            options=Options(context={"partition_by": ["region"], "order_by": "ts"}),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PyArrowFrameAggregate.calculate_feature(table, fs)
        col = result.column("value__cumsum").to_pylist()

        # Sorted by ts: (1, 10), (2, 20), (None, 100) -- None goes last
        # idx 1 (ts=1): cumsum=10
        # idx 2 (ts=2): cumsum=30
        # idx 0 (ts=None): cumsum=130
        assert col == [130, 10, 30]

    def test_all_null_values_returns_none(self) -> None:
        """When all values in the window are null, the result should be None."""
        table = pa.table(
            {
                "region": ["A", "A"],
                "ts": [1, 2],
                "value": [None, None],
            }
        )
        feature = Feature(
            "value__cumsum",
            options=Options(context={"partition_by": ["region"], "order_by": "ts"}),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PyArrowFrameAggregate.calculate_feature(table, fs)
        col = result.column("value__cumsum").to_pylist()

        assert col == [None, None]


class TestTimeWindow:
    def test_time_window_via_config(self) -> None:
        """Time-based window using config-based feature creation."""
        from datetime import datetime, timezone

        table = pa.table(
            {
                "region": ["A", "A", "A", "A"],
                "ts": [
                    datetime(2023, 1, 1, tzinfo=timezone.utc),
                    datetime(2023, 1, 2, tzinfo=timezone.utc),
                    datetime(2023, 1, 4, tzinfo=timezone.utc),
                    datetime(2023, 1, 5, tzinfo=timezone.utc),
                ],
                "value": [10, 20, 30, 40],
            }
        )
        feature = Feature(
            "my_time_sum",
            options=Options(
                context={
                    "aggregation_type": "sum",
                    "frame_type": "time",
                    "frame_size": 2,
                    "frame_unit": "day",
                    "in_features": "value",
                    "partition_by": ["region"],
                    "order_by": "ts",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PyArrowFrameAggregate.calculate_feature(table, fs)
        col = result.column("my_time_sum").to_pylist()

        # 2-day window:
        # Jan 1: window includes Jan 1 (within 2 days back) => [10] => 10
        # Jan 2: window includes Jan 1, Jan 2 => [10, 20] => 30
        # Jan 4: window includes Jan 2, Jan 4 (Jan 1 is >2 days before Jan 4) => [20, 30] => 50
        # Jan 5: window includes Jan 4, Jan 5 (Jan 2 is >2 days before Jan 5) => [30, 40] => 70
        assert col == [10, 30, 50, 70]


class TestRowPreserving:
    def test_output_row_count_equals_input(self, sample_table: pa.Table) -> None:
        feature = Feature(
            "value__sum_rolling_2",
            options=Options(context={"partition_by": ["region"], "order_by": "timestamp"}),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PyArrowFrameAggregate.calculate_feature(sample_table, fs)
        assert result.num_rows == sample_table.num_rows

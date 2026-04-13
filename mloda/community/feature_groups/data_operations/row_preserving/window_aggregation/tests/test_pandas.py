"""Tests for PandasWindowAggregation compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

import pandas as pd

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pandas_window_aggregation import (
    PandasWindowAggregation,
)
from mloda.testing.feature_groups.data_operations.mixins.pandas import PandasTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.window_aggregation import (
    WindowAggregationTestBase,
)


class TestPandasWindowAggregation(PandasTestMixin, WindowAggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PandasWindowAggregation


class TestPandasWindowModeVectorized:
    """Targeted tests for the vectorized window ``_compute_mode``.

    Verifies that the per-row broadcast produced by the vectorized
    implementation matches the insertion-order tie-breaking semantics of
    the old ``Counter``-based reducer and preserves the original row order.
    """

    def test_mode_broadcasts_to_every_row(self) -> None:
        data = pd.DataFrame({"region": ["A", "A", "A"], "value": [1, 2, 2]})
        result = PandasWindowAggregation._compute_mode(data, "mode_val", "value", ["region"])
        assert result["mode_val"].tolist() == [2, 2, 2]
        assert result["value"].tolist() == [1, 2, 2]

    def test_mode_tie_break_by_first_occurrence(self) -> None:
        data = pd.DataFrame({"region": ["A", "A", "A", "A"], "value": [3, 1, 1, 3]})
        result = PandasWindowAggregation._compute_mode(data, "mode_val", "value", ["region"])
        assert result["mode_val"].tolist() == [3, 3, 3, 3]

    def test_mode_preserves_original_row_order(self) -> None:
        data = pd.DataFrame(
            {
                "region": ["B", "A", "B", "A", "B"],
                "value": [9, 1, 9, 2, 7],
            }
        )
        result = PandasWindowAggregation._compute_mode(data, "mode_val", "value", ["region"])
        assert result["region"].tolist() == ["B", "A", "B", "A", "B"]
        assert result["value"].tolist() == [9, 1, 9, 2, 7]
        assert result["mode_val"].tolist() == [9, 1, 9, 1, 9]

    def test_mode_all_null_partition_broadcasts_none(self) -> None:
        data = pd.DataFrame(
            {
                "region": ["A", "A", "B", "B"],
                "value": [float("nan"), float("nan"), 4.0, 4.0],
            }
        )
        result = PandasWindowAggregation._compute_mode(data, "mode_val", "value", ["region"])
        assert pd.isna(result.loc[result["region"] == "A", "mode_val"]).all()
        assert (result.loc[result["region"] == "B", "mode_val"] == 4.0).all()

    def test_mode_null_in_partition_keys_groups_together(self) -> None:
        data = pd.DataFrame({"region": ["A", None, None, "A"], "value": [1, 9, 9, 1]})
        result = PandasWindowAggregation._compute_mode(data, "mode_val", "value", ["region"])
        assert result["mode_val"].tolist() == [1, 9, 9, 1]

    def test_mode_multi_key_partition_broadcasts(self) -> None:
        data = pd.DataFrame(
            {
                "region": ["A", "A", "A", "B", "B"],
                "category": ["x", "x", "y", "x", "x"],
                "value": [1, 1, 5, 7, 9],
            }
        )
        result = PandasWindowAggregation._compute_mode(data, "mode_val", "value", ["region", "category"])
        assert result["mode_val"].tolist() == [1, 1, 5, 7, 7]

    def test_mode_does_not_use_python_counter(self) -> None:
        """Guard against regressing to the Python ``collections.Counter`` path."""
        from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation import (
            pandas_window_aggregation,
        )

        assert not hasattr(pandas_window_aggregation, "Counter")

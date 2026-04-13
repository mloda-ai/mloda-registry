"""Tests for PandasAggregation compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

import pandas as pd

from mloda.community.feature_groups.data_operations.aggregation.pandas_aggregation import (
    PandasAggregation,
)
from mloda.testing.feature_groups.data_operations.aggregation.aggregation import (
    AggregationTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.pandas import PandasTestMixin


class TestPandasAggregation(PandasTestMixin, AggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PandasAggregation


class TestPandasModeVectorized:
    """Targeted tests for the vectorized ``_compute_mode`` implementation.

    These exercise the tie-breaking and null-handling semantics that were
    previously delivered by the per-group ``Counter``/Python-loop reducer.
    """

    def test_mode_single_partition_basic(self) -> None:
        data = pd.DataFrame({"region": ["A", "A", "A"], "value": [1, 2, 2]})
        result = PandasAggregation._compute_mode(data, "value__mode_agg", "value", ["region"])
        assert result["region"].tolist() == ["A"]
        assert result["value__mode_agg"].tolist() == [2]

    def test_mode_tie_break_by_first_occurrence(self) -> None:
        data = pd.DataFrame({"region": ["A", "A", "A", "A"], "value": [3, 1, 1, 3]})
        result = PandasAggregation._compute_mode(data, "value__mode_agg", "value", ["region"])
        assert result["value__mode_agg"].tolist() == [3]

    def test_mode_three_way_tie_picks_earliest(self) -> None:
        data = pd.DataFrame({"region": ["A"] * 6, "value": [5, 7, 9, 7, 5, 9]})
        result = PandasAggregation._compute_mode(data, "value__mode_agg", "value", ["region"])
        assert result["value__mode_agg"].tolist() == [5]

    def test_mode_all_null_partition_yields_none(self) -> None:
        data = pd.DataFrame(
            {
                "region": ["A", "A", "B", "B"],
                "value": [float("nan"), float("nan"), 4.0, 4.0],
            }
        )
        result = PandasAggregation._compute_mode(data, "value__mode_agg", "value", ["region"])
        result = result.sort_values("region").reset_index(drop=True)
        assert result["region"].tolist() == ["A", "B"]
        assert pd.isna(result.loc[0, "value__mode_agg"])
        assert result.loc[1, "value__mode_agg"] == 4.0

    def test_mode_nulls_in_source_are_ignored(self) -> None:
        data = pd.DataFrame({"region": ["A", "A", "A", "A"], "value": [float("nan"), 1.0, 2.0, 2.0]})
        result = PandasAggregation._compute_mode(data, "value__mode_agg", "value", ["region"])
        assert result["value__mode_agg"].tolist() == [2.0]

    def test_mode_null_in_partition_keys_groups_together(self) -> None:
        data = pd.DataFrame({"region": ["A", None, None, "A"], "value": [1, 9, 9, 1]})
        result = PandasAggregation._compute_mode(data, "value__mode_agg", "value", ["region"])
        result_map: dict[Any, Any] = {}
        for _, row in result.iterrows():
            key: Any = None if pd.isna(row["region"]) else row["region"]
            result_map[key] = row["value__mode_agg"]
        assert result_map == {"A": 1, None: 9}

    def test_mode_multi_key_partition(self) -> None:
        data = pd.DataFrame(
            {
                "region": ["A", "A", "A", "B", "B"],
                "category": ["x", "x", "y", "x", "x"],
                "value": [1, 1, 5, 7, 9],
            }
        )
        result = PandasAggregation._compute_mode(data, "value__mode_agg", "value", ["region", "category"])
        result = result.sort_values(["region", "category"]).reset_index(drop=True)
        assert result[["region", "category"]].values.tolist() == [
            ["A", "x"],
            ["A", "y"],
            ["B", "x"],
        ]
        assert result["value__mode_agg"].tolist() == [1, 5, 7]

    def test_mode_does_not_use_python_counter(self) -> None:
        """Guard against regressing to the Python ``collections.Counter`` path.

        The vectorized implementation must not import ``Counter`` in the
        aggregation module; this test pins that invariant.
        """
        from mloda.community.feature_groups.data_operations.aggregation import (
            pandas_aggregation,
        )

        assert not hasattr(pandas_aggregation, "Counter")

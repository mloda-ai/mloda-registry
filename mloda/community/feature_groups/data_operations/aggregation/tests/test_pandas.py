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

    def test_mode_source_col_equals_partition_by_single_key(self) -> None:
        """Bug 1: source_col appearing in partition_by must not raise.

        Mode of X grouped by X is X itself. The current implementation
        raises ``ValueError: cannot reindex on an axis with duplicate
        labels`` because ``data[partition_by + [source_col]]`` produces
        duplicate column labels.
        """
        data = pd.DataFrame({"value": [1, 2, 2, 3]})
        result = PandasAggregation._compute_mode(data, "value__mode_agg", "value", ["value"])
        result = result.sort_values("value").reset_index(drop=True)
        assert result["value"].tolist() == [1, 2, 3]
        assert result["value__mode_agg"].tolist() == [1, 2, 3]

    def test_mode_source_col_in_multi_key_partition(self) -> None:
        """Bug 1: source_col appearing among multi-key partition_by must not raise."""
        data = pd.DataFrame(
            {
                "region": ["A", "A", "B", "B"],
                "value": [1, 1, 2, 2],
            }
        )
        result = PandasAggregation._compute_mode(data, "value__mode_agg", "value", ["region", "value"])
        result = result.sort_values(["region", "value"]).reset_index(drop=True)
        assert result[["region", "value"]].values.tolist() == [["A", 1], ["B", 2]]
        assert result["value__mode_agg"].tolist() == [1, 2]

    def test_mode_source_col_equals_partition_by_with_null(self) -> None:
        """Bug 1: when source_col == partition_by and a value is null, that row's
        partition is still emitted but the feature value is NaN (matching the
        all-null-partition convention from ``test_mode_all_null_partition_yields_none``).
        """
        data = pd.DataFrame({"value": [1.0, 2.0, float("nan")]})
        result = PandasAggregation._compute_mode(data, "value__mode_agg", "value", ["value"])
        # three unique partition keys: 1.0, 2.0, NaN
        assert len(result) == 3
        non_null = result[result["value"].notna()].sort_values("value").reset_index(drop=True)
        assert non_null["value"].tolist() == [1.0, 2.0]
        assert non_null["value__mode_agg"].tolist() == [1.0, 2.0]
        null_row = result[result["value"].isna()]
        assert len(null_row) == 1
        assert pd.isna(null_row["value__mode_agg"]).all()

    def test_mode_accepts_tuple_partition_by_single_key(self) -> None:
        """Bug 3: partition_by may arrive as a tuple (mloda core hashability fix, #228).

        Other aggregations work because they pass partition_by straight to
        ``df.groupby``. The new mode path builds ``partition_by + [source_col]``
        which raises ``TypeError`` when partition_by is a tuple.
        """
        data = pd.DataFrame({"region": ["A", "A", "A"], "value": [1, 2, 2]})
        result = PandasAggregation._compute_mode(data, "value__mode_agg", "value", ("region",))
        assert result["region"].tolist() == ["A"]
        assert result["value__mode_agg"].tolist() == [2]

    def test_mode_accepts_tuple_partition_by_multi_key(self) -> None:
        """Bug 3: multi-key tuple partition_by must not crash."""
        data = pd.DataFrame(
            {
                "region": ["A", "A", "A", "B", "B"],
                "category": ["x", "x", "y", "x", "x"],
                "value": [1, 1, 5, 7, 9],
            }
        )
        result = PandasAggregation._compute_mode(data, "value__mode_agg", "value", ("region", "category"))
        result = result.sort_values(["region", "category"]).reset_index(drop=True)
        assert result[["region", "category"]].values.tolist() == [
            ["A", "x"],
            ["A", "y"],
            ["B", "x"],
        ]
        assert result["value__mode_agg"].tolist() == [1, 5, 7]


class TestComputeModeWinnersCollisions:
    """Bug 2: internal sentinel column names must not collide with user columns.

    ``compute_mode_winners`` uses the hard-coded temp names
    ``__mloda_mode_row_idx__``, ``__mloda_mode_count__`` and
    ``__mloda_mode_first_idx__``. If a user column shares any of these
    names the helper overwrites it silently and produces wrong results.
    """

    def test_compute_mode_winners_partition_collides_with_row_idx(self) -> None:
        from mloda.community.feature_groups.data_operations.pandas_helpers import (
            compute_mode_winners,
        )

        data = pd.DataFrame(
            {
                "__mloda_mode_row_idx__": ["A", "A", "B", "B", "B"],
                "v": [1, 1, 2, 3, 3],
            }
        )
        result = compute_mode_winners(data, "v", ["__mloda_mode_row_idx__"])
        result = result.sort_values("__mloda_mode_row_idx__").reset_index(drop=True)
        assert result["__mloda_mode_row_idx__"].tolist() == ["A", "B"]
        assert result["v"].tolist() == [1, 3]

    def test_compute_mode_winners_source_col_is_count_sentinel(self) -> None:
        from mloda.community.feature_groups.data_operations.pandas_helpers import (
            compute_mode_winners,
        )

        data = pd.DataFrame(
            {
                "region": ["A", "A", "A", "B", "B"],
                "__mloda_mode_count__": [10, 10, 20, 30, 30],
            }
        )
        result = compute_mode_winners(data, "__mloda_mode_count__", ["region"])
        result = result.sort_values("region").reset_index(drop=True)
        assert result["region"].tolist() == ["A", "B"]
        assert result["__mloda_mode_count__"].tolist() == [10, 30]

    def test_aggregation_mode_partition_collides_with_row_idx(self) -> None:
        data = pd.DataFrame(
            {
                "__mloda_mode_row_idx__": ["A", "A", "B", "B", "B"],
                "v": [1, 1, 2, 3, 3],
            }
        )
        result = PandasAggregation._compute_mode(data, "v__mode_agg", "v", ["__mloda_mode_row_idx__"])
        result = result.sort_values("__mloda_mode_row_idx__").reset_index(drop=True)
        assert result["__mloda_mode_row_idx__"].tolist() == ["A", "B"]
        assert result["v__mode_agg"].tolist() == [1, 3]

    def test_aggregation_mode_source_col_is_count_sentinel(self) -> None:
        data = pd.DataFrame(
            {
                "region": ["A", "A", "A", "B", "B"],
                "__mloda_mode_count__": [10, 10, 20, 30, 30],
            }
        )
        result = PandasAggregation._compute_mode(data, "out", "__mloda_mode_count__", ["region"])
        result = result.sort_values("region").reset_index(drop=True)
        assert result["region"].tolist() == ["A", "B"]
        assert result["out"].tolist() == [10, 30]

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
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.window_aggregation import (
    WindowAggregationTestBase,
)


class TestPandasWindowAggregation(ReservedColumnsTestMixin, PandasTestMixin, WindowAggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PandasWindowAggregation

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        return "value_int__sum_window"

    @classmethod
    def reserved_columns_partition_by(cls) -> list[str] | None:
        return ["region"]

    @classmethod
    def reserved_columns_order_by(cls) -> str | None:
        return None


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

    def test_mode_source_col_equals_partition_by_single_key(self) -> None:
        """Bug 1: source_col in partition_by must broadcast trivially.

        Within every partition defined by value, the value column is
        constant, so the mode equals the key.
        """
        data = pd.DataFrame({"value": [1, 2, 2, 3]})
        result = PandasWindowAggregation._compute_mode(data, "mode_val", "value", ["value"])
        assert result["value"].tolist() == [1, 2, 2, 3]
        assert result["mode_val"].tolist() == [1, 2, 2, 3]

    def test_mode_source_col_in_multi_key_partition(self) -> None:
        """Bug 1: source_col appearing among multi-key partition_by must broadcast correctly."""
        data = pd.DataFrame(
            {
                "region": ["A", "A", "B", "B"],
                "value": [1, 1, 2, 2],
            }
        )
        result = PandasWindowAggregation._compute_mode(data, "mode_val", "value", ["region", "value"])
        assert result["region"].tolist() == ["A", "A", "B", "B"]
        assert result["value"].tolist() == [1, 1, 2, 2]
        assert result["mode_val"].tolist() == [1, 1, 2, 2]

    def test_mode_source_col_equals_partition_by_with_null(self) -> None:
        """Bug 1: null source values broadcast as NaN when source_col == partition_by."""
        data = pd.DataFrame({"value": [1.0, 2.0, float("nan"), 2.0]})
        result = PandasWindowAggregation._compute_mode(data, "mode_val", "value", ["value"])
        assert result["value"].tolist()[:2] == [1.0, 2.0]
        assert result.loc[0, "mode_val"] == 1.0
        assert result.loc[1, "mode_val"] == 2.0
        assert pd.isna(result.loc[2, "mode_val"])
        assert result.loc[3, "mode_val"] == 2.0

    def test_mode_partition_collides_with_is_data_sentinel(self) -> None:
        """Bug 2: a user partition column named ``__mloda_mode_is_data__`` must
        still produce a correct mode broadcast. The current implementation uses
        that name as an internal sentinel and overwrites the column, yielding
        all-None output.
        """
        data = pd.DataFrame(
            {
                "__mloda_mode_is_data__": ["A", "A", "B", "B", "B"],
                "v": [1, 1, 2, 3, 3],
            }
        )
        result = PandasWindowAggregation._compute_mode(data, "mode_val", "v", ["__mloda_mode_is_data__"])
        assert result["__mloda_mode_is_data__"].tolist() == ["A", "A", "B", "B", "B"]
        assert result["v"].tolist() == [1, 1, 2, 3, 3]
        assert result["mode_val"].tolist() == [1, 1, 3, 3, 3]

    def test_mode_partition_collides_with_row_idx_sentinel(self) -> None:
        """Bug 2: a user partition column matching the helper's row-idx sentinel."""
        data = pd.DataFrame(
            {
                "__mloda_mode_row_idx__": ["A", "A", "B", "B", "B"],
                "v": [1, 1, 2, 3, 3],
            }
        )
        result = PandasWindowAggregation._compute_mode(data, "mode_val", "v", ["__mloda_mode_row_idx__"])
        assert result["mode_val"].tolist() == [1, 1, 3, 3, 3]

    def test_mode_source_col_is_count_sentinel(self) -> None:
        """Bug 2: a user source column matching the helper's count sentinel."""
        data = pd.DataFrame(
            {
                "region": ["A", "A", "A", "B", "B"],
                "__mloda_mode_count__": [10, 10, 20, 30, 30],
            }
        )
        result = PandasWindowAggregation._compute_mode(data, "mode_val", "__mloda_mode_count__", ["region"])
        assert result["mode_val"].tolist() == [10, 10, 10, 30, 30]

    def test_mode_accepts_tuple_partition_by_single_key(self) -> None:
        """Bug 3: tuple partition_by must not crash the window mode path.

        mloda core converts list options to tuples for hashability (#228).
        The shared helper builds ``partition_by + [source_col]`` which raises
        ``TypeError`` when partition_by is a tuple.
        """
        data = pd.DataFrame({"region": ["A", "A", "A"], "value": [1, 2, 2]})
        result = PandasWindowAggregation._compute_mode(data, "mode_val", "value", ("region",))
        assert result["mode_val"].tolist() == [2, 2, 2]
        assert result["value"].tolist() == [1, 2, 2]

    def test_mode_accepts_tuple_partition_by_multi_key(self) -> None:
        """Bug 3: multi-key tuple partition_by must broadcast correctly."""
        data = pd.DataFrame(
            {
                "region": ["A", "A", "A", "B", "B"],
                "category": ["x", "x", "y", "x", "x"],
                "value": [1, 1, 5, 7, 9],
            }
        )
        result = PandasWindowAggregation._compute_mode(data, "mode_val", "value", ("region", "category"))
        assert result["mode_val"].tolist() == [1, 1, 5, 7, 7]

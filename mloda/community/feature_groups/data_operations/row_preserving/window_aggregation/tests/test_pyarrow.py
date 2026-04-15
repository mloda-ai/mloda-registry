"""Tests for PyArrowWindowAggregation compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pyarrow_window_aggregation import (
    PyArrowWindowAggregation,
)
from mloda.testing.feature_groups.data_operations.mixins.pyarrow import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.window_aggregation import (
    WindowAggregationTestBase,
)


class TestPyArrowWindowAggregation(PyArrowTestMixin, WindowAggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowWindowAggregation

    @classmethod
    def supported_agg_types(cls) -> set[str]:
        return {
            "sum",
            "avg",
            "mean",
            "count",
            "min",
            "max",
            "std",
            "var",
            "std_pop",
            "std_samp",
            "var_pop",
            "var_samp",
            "nunique",
            "first",
            "last",
        }

    def test_collision_wa_idx(self) -> None:
        """User column named __mloda_wa_idx__ must survive PyArrowWindowAggregation."""
        self._run_collision_case("__mloda_wa_idx__")

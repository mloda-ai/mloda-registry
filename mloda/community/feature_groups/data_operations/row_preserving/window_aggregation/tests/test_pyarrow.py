"""Tests for PyArrowWindowAggregation compute implementation."""

from __future__ import annotations

import warnings
from typing import Any

from mloda.core.abstract_plugins.components.options import Options
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pyarrow_window_aggregation import (
    PyArrowWindowAggregation,
)
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.pyarrow import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.window_aggregation import (
    EXPECTED_FIRST_BY_REGION,
    EXPECTED_LAST_BY_REGION,
    WindowAggregationTestBase,
)


class TestPyArrowWindowAggregation(CapabilityHookTestMixin, PyArrowTestMixin, WindowAggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowWindowAggregation

    @classmethod
    def capability_unsupported(cls) -> tuple[tuple[str, Options], ...]:
        return (("value__mode_window", Options()),)

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

    def test_first_window_no_future_warning(self) -> None:
        """first with partition_by must not raise pyarrow's null_placement FutureWarning."""
        fs = make_feature_set("value_int__first_window", ["region"], order_by="value_int")
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__first_window")
        assert result_col == EXPECTED_FIRST_BY_REGION

    def test_last_window_no_future_warning(self) -> None:
        """last with partition_by must not raise pyarrow's null_placement FutureWarning."""
        fs = make_feature_set("value_int__last_window", ["region"], order_by="value_int")
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__last_window")
        assert result_col == EXPECTED_LAST_BY_REGION

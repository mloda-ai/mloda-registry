"""Tests for ResampleFeatureGroup base class."""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.options import Options

from mloda.community.feature_groups.data_operations.row_changing.resample.base import (
    ResampleFeatureGroup,
)


class TestReturnDataTypeRule:
    """return_data_type_rule should fix the output type only for deterministic ops.

    A bucket count always returns INT64. mean / sum depend on the input column
    type, so the rule must return None for them.
    """

    def test_count_returns_int64(self) -> None:
        from mloda.user import DataType, Feature

        feature = Feature("value__resample_5_minute_count", options=Options())
        assert ResampleFeatureGroup.return_data_type_rule(feature) == DataType.INT64

    @pytest.mark.parametrize("agg", ["mean", "sum"])
    def test_input_dependent_ops_return_none(self, agg: str) -> None:
        from mloda.user import Feature

        feature = Feature(f"value__resample_5_minute_{agg}", options=Options())
        assert ResampleFeatureGroup.return_data_type_rule(feature) is None

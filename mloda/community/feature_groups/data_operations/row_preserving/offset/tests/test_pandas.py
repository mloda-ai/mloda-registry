"""Tests for PandasOffset compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

from mloda.community.feature_groups.data_operations.row_preserving.offset.pandas_offset import PandasOffset
from mloda.testing.feature_groups.data_operations.mixins.pandas import PandasTestMixin
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.offset.offset import OffsetTestBase


class TestPandasOffset(ReservedColumnsTestMixin, PandasTestMixin, OffsetTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PandasOffset

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        return "value_int__lag_1_offset"

    @classmethod
    def reserved_columns_partition_by(cls) -> list[str] | None:
        return ["region"]

    @classmethod
    def reserved_columns_order_by(cls) -> str | None:
        return "value_int"

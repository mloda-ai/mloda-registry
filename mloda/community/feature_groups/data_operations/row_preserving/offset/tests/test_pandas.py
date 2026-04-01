"""Tests for PandasOffset compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

from mloda.community.feature_groups.data_operations.row_preserving.offset.pandas_offset import PandasOffset
from mloda.testing.feature_groups.data_operations.mixins.pandas import PandasTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.offset.offset import OffsetTestBase


class TestPandasOffset(PandasTestMixin, OffsetTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PandasOffset

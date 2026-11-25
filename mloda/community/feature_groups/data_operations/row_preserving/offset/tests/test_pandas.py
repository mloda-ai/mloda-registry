"""Tests for PandasOffset compute implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

pytest.importorskip("pandas")

from mloda.community.feature_groups.data_operations.row_preserving.offset.pandas_offset import PandasOffset
from mloda.testing.feature_groups.data_operations.mixins.frameworks import (
    PandasNullableIntTestMixin,
    PandasTestMixin,
)
from mloda.testing.feature_groups.data_operations.row_preserving.offset.offset import OffsetTestBase


class TestPandasOffset(PandasTestMixin, OffsetTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PandasOffset

    @classmethod
    def pinned_divergences(cls) -> Mapping[str, Mapping[str, list[Any]]]:
        return {
            **super().pinned_divergences(),
            # NaN and None partition keys merge, so lag crosses the reference partition boundary.
            "17-partition_keys-v__lag_1_offset": {"v__lag_1_offset": [None, 1, 2, 3]},
        }


class TestPandasOffsetNullableInt(PandasNullableIntTestMixin, TestPandasOffset):
    """Runs the offset suite on nullable pandas int columns."""

"""Tests for PolarsLazyOffset compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("polars")

from mloda.community.feature_groups.data_operations.row_preserving.offset.polars_lazy_offset import PolarsLazyOffset
from mloda.testing.feature_groups.data_operations.mixins.polars_lazy import PolarsLazyTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.offset.offset import OffsetTestBase


class TestPolarsLazyOffset(PolarsLazyTestMixin, OffsetTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyOffset

    def test_collision_orig_idx(self) -> None:
        """User column named __mloda_orig_idx must survive PolarsLazyOffset."""
        self._run_collision_case("__mloda_orig_idx")

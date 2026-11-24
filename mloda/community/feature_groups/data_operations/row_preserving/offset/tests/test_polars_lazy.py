"""Tests for PolarsLazyOffset compute implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

pytest.importorskip("polars")

from mloda.community.feature_groups.data_operations.row_preserving.offset.polars_lazy_offset import PolarsLazyOffset
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PolarsLazyTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.offset.offset import OffsetTestBase


class TestPolarsLazyOffset(PolarsLazyTestMixin, OffsetTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyOffset

    @classmethod
    def pinned_divergences(cls) -> Mapping[str, Mapping[str, list[Any]]]:
        return {
            **super().pinned_divergences(),
            # Narrow integer subtraction wraps before widening.
            "20-dtype-v__diff_1_offset": {"v__diff_1_offset": [None, 252]},
            "21-dtype-v__diff_1_offset": {"v__diff_1_offset": [None, -56]},
        }

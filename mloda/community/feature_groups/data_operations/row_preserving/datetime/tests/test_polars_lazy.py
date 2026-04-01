"""Tests for PolarsLazyDateTimeExtraction compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("polars")

from mloda.community.feature_groups.data_operations.row_preserving.datetime.polars_lazy_datetime import (
    PolarsLazyDateTimeExtraction,
)
from mloda.testing.feature_groups.data_operations.mixins.polars_lazy import PolarsLazyTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.datetime.datetime import (
    DateTimeTestBase,
)


class TestPolarsLazyDateTimeExtraction(PolarsLazyTestMixin, DateTimeTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyDateTimeExtraction

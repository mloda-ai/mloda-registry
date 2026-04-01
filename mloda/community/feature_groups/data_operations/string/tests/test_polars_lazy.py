"""Tests for PolarsLazyStringOps compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("polars")

from mloda.community.feature_groups.data_operations.string.polars_lazy_string import (
    PolarsLazyStringOps,
)
from mloda.testing.feature_groups.data_operations.mixins.polars_lazy import PolarsLazyTestMixin
from mloda.testing.feature_groups.data_operations.string import (
    StringTestBase,
)


class TestPolarsLazyStringOps(PolarsLazyTestMixin, StringTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyStringOps

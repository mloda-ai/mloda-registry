"""Tests for PandasStringOps compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

from mloda.community.feature_groups.data_operations.string.pandas_string import (
    PandasStringOps,
)
from mloda.testing.feature_groups.data_operations.mixins.pandas import (
    PandasNullableIntTestMixin,
    PandasTestMixin,
)
from mloda.testing.feature_groups.data_operations.string.string import (
    StringTestBase,
)


class TestPandasStringOps(PandasTestMixin, StringTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PandasStringOps


class TestPandasStringOpsNullableInt(PandasNullableIntTestMixin, TestPandasStringOps):
    """Runs the string suite on nullable pandas int columns."""

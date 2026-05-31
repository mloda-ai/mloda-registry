"""Tests for PandasFfill compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

from mloda.community.feature_groups.data_operations.row_preserving.ffill.pandas_ffill import (
    PandasFfill,
)
from mloda.testing.feature_groups.data_operations.mixins.pandas import PandasTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.ffill.ffill import (
    FfillTestBase,
)


class TestPandasFfill(PandasTestMixin, FfillTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PandasFfill

"""Tests for PandasResample compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

from mloda.community.feature_groups.data_operations.row_changing.resample.pandas_resample import (
    PandasResample,
)
from mloda.testing.feature_groups.data_operations.mixins.pandas import (
    PandasNullableIntTestMixin,
    PandasTestMixin,
)
from mloda.testing.feature_groups.data_operations.row_changing.resample.resample import (
    ResampleTestBase,
)


class TestPandasResample(PandasTestMixin, ResampleTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PandasResample


class TestPandasResampleNullableInt(PandasNullableIntTestMixin, TestPandasResample):
    """Runs the resample suite on nullable pandas int columns."""

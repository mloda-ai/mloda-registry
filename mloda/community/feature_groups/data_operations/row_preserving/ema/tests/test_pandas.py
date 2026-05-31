"""Tests for PandasEma compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

from mloda.community.feature_groups.data_operations.row_preserving.ema.pandas_ema import (
    PandasEma,
)
from mloda.testing.feature_groups.data_operations.mixins.pandas import PandasTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.ema.ema import (
    EmaTestBase,
)


class TestPandasEma(PandasTestMixin, EmaTestBase):
    """All value/semantics/error tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PandasEma

"""Tests for PandasSessionization compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

from mloda.community.feature_groups.data_operations.row_preserving.sessionization.pandas_sessionization import (
    PandasSessionization,
)
from mloda.testing.feature_groups.data_operations.mixins.pandas import PandasTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.sessionization.sessionization import (
    SessionizationTestBase,
)


class TestPandasSessionization(PandasTestMixin, SessionizationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PandasSessionization

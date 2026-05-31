"""Tests for PolarsLazySessionization compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("polars")

from mloda.community.feature_groups.data_operations.row_preserving.sessionization.polars_lazy_sessionization import (
    PolarsLazySessionization,
)
from mloda.testing.feature_groups.data_operations.mixins.polars_lazy import PolarsLazyTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.sessionization.sessionization import (
    SessionizationTestBase,
)


class TestPolarsLazySessionization(PolarsLazyTestMixin, SessionizationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazySessionization

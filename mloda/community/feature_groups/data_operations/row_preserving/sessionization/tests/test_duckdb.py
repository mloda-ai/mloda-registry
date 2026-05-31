"""Tests for DuckdbSessionization compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.row_preserving.sessionization.duckdb_sessionization import (
    DuckdbSessionization,
)
from mloda.testing.feature_groups.data_operations.mixins.duckdb import DuckdbTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.sessionization.sessionization import (
    SessionizationTestBase,
)


class TestDuckdbSessionization(DuckdbTestMixin, SessionizationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbSessionization

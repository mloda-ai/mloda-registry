"""Tests for DuckdbPercentile compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.row_preserving.percentile.duckdb_percentile import (
    DuckdbPercentile,
)
from mloda.testing.feature_groups.data_operations.row_preserving.percentile.percentile import (
    PercentileTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.duckdb import DuckdbTestMixin


class TestDuckdbPercentile(DuckdbTestMixin, PercentileTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbPercentile

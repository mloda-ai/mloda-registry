"""Tests for DuckdbEma: EMA is REJECTED, not computed.

DuckDB has no native exponentially weighted (EWM) aggregate, and a recursive /
Python emulation is forbidden by the CFW-backend rule. The DuckDB backend must
reject EMA up-front with a clear ValueError. This file inherits only the single
rejection assertion from ``EmaRejectionTestBase`` -- none of the value tests.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.row_preserving.ema.duckdb_ema import (
    DuckdbEma,
)
from mloda.testing.feature_groups.data_operations.mixins.duckdb import DuckdbTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.ema.ema import (
    EmaRejectionTestBase,
)


class TestDuckdbEmaRejected(DuckdbTestMixin, EmaRejectionTestBase):
    """EMA on DuckDB raises a clear ValueError."""

    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbEma

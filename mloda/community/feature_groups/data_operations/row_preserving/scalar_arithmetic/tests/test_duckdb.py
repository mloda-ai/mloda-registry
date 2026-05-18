"""Tests for DuckdbScalarArithmetic compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.duckdb_scalar_arithmetic import (
    DuckdbScalarArithmetic,
)
from mloda.testing.feature_groups.data_operations.row_preserving.scalar_arithmetic.scalar_arithmetic import (
    ScalarArithmeticTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.duckdb import DuckdbTestMixin


class TestDuckdbScalarArithmetic(DuckdbTestMixin, ScalarArithmeticTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbScalarArithmetic

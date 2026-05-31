"""Tests for DuckdbBinning compute implementation."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

duckdb = pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.row_preserving.binning.duckdb_binning import (
    DuckdbBinning,
)
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.duckdb import DuckdbTestMixin
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.binning.binning import (
    BinningTestBase,
)


class TestDuckdbBinning(ReservedColumnsTestMixin, DuckdbTestMixin, BinningTestBase):
    """Standard tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbBinning

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        return "value_int__bin_3"

    @classmethod
    def reserved_columns_partition_by(cls) -> list[str] | None:
        return None

    def test_reserved_prefixed_user_column_survives(self) -> None:
        """A user column matching the reserved ``__mloda_`` prefix is preserved intact.

        Because the DuckDB backend picks collision-free helper-column names via
        ``pick_helper_column_name``, a reserved-prefixed USER column is accepted
        (no error) and survives in the output unchanged, alongside the computed
        binning feature. This pins the behaviour-change beyond the shared mixin's
        weaker "result is not None" assertion.
        """
        colliding_name = "__MLODA_USER_COL__"
        base_table: pa.Table = self._arrow_table
        num_rows = base_table.num_rows
        colliding_table = base_table.append_column(
            colliding_name,
            pa.array(list(range(num_rows)), type=pa.int64()),
        )
        colliding_data = self.create_test_data(colliding_table)

        feature_name = self.reserved_columns_feature_name()
        fs = make_feature_set(
            feature_name,
            partition_by=self.reserved_columns_partition_by(),
            order_by=self.reserved_columns_order_by(),
        )
        result = self.implementation_class().calculate_feature(colliding_data, fs)

        assert isinstance(result, self.get_expected_type())
        column_names = result.to_arrow_table().column_names
        assert feature_name in column_names
        assert colliding_name in column_names

        result_user_col = self.extract_column(result, colliding_name)
        assert result_user_col == list(range(num_rows))

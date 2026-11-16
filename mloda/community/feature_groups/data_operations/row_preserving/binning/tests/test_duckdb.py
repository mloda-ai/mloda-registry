"""Tests for DuckdbBinning compute implementation."""

from __future__ import annotations

from typing import Any

import pytest
from mloda.core.abstract_plugins.components.data_types import DataType

duckdb = pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.row_preserving.binning.duckdb_binning import (
    DuckdbBinning,
)
from mloda.testing.feature_groups.data_operations.helpers import append_typed_columns, calculate_with, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.frameworks import DuckdbTestMixin
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

    @classmethod
    def reserved_columns_helper_names(cls) -> set[str]:
        return {"__mloda_rn0__", "__mloda_rn1__", "__mloda_rn2__"}

    @classmethod
    def reserved_columns_extra_feature_names(cls) -> tuple[str, ...]:
        return ("value_int__qbin_3",)

    def test_reserved_prefixed_user_column_survives(self) -> None:
        """A user column matching the reserved ``__mloda_`` prefix is preserved intact.

        Because the DuckDB backend picks collision-free helper-column names via
        ``pick_helper_column_name``, a reserved-prefixed USER column is accepted
        (no error) and survives in the output unchanged, alongside the computed
        binning feature. This pins the behaviour-change beyond the shared mixin's
        weaker "result is not None" assertion.
        """
        colliding_name = "__MLODA_USER_COL__"
        num_rows = self._source.row_count
        colliding_typed = append_typed_columns(self._source, {colliding_name: (DataType.INT64, list(range(num_rows)))})
        colliding_data = self.create_test_data(colliding_typed)

        feature_name = self.reserved_columns_feature_name()
        fs = make_feature_set(
            feature_name,
            partition_by=self.reserved_columns_partition_by(),
            order_by=self.reserved_columns_order_by(),
        )
        result = calculate_with(self.implementation_class(), colliding_data, fs)

        assert isinstance(result, self.get_expected_type())
        column_names = result.to_arrow_table().column_names
        assert feature_name in column_names
        assert colliding_name in column_names

        result_user_col = self.extract_column(result, colliding_name)
        assert result_user_col == list(range(num_rows))

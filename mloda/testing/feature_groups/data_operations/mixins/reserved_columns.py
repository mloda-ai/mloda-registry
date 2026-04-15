"""Reusable reserved-column guard test mixin for data-operations feature groups.

Provides a single standardized test that verifies input columns whose name
starts with the reserved ``__mloda_`` prefix are rejected, across every
framework implementation of a guarded data-operation. Each feature-group
test base mixes this in and overrides the abstract configuration methods
to adapt the generic test to its specific semantics (feature name,
partition keys, order key).

The test method uses a ``test_mixin_`` prefix to match the existing
convention for shared mixin tests (see ``MaskTestMixin``).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from mloda.testing.feature_groups.data_operations.helpers import make_feature_set


class ReservedColumnsTestMixin:
    """Mixin providing a standardized reserved-column collision test.

    Feature-group test bases mix this in and implement the configuration
    methods below to adapt the generic test to their semantics.

    Requires the host class to provide (from DataOpsTestBase):
    - ``implementation_class()``
    - ``_arrow_table`` attribute (set in setup_method)
    - ``create_test_data(arrow_table)``
    """

    # -- Configuration methods (override per feature group) --------------------

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        """Feature name for the reserved-columns test (e.g. 'value_int__sum_window')."""
        raise NotImplementedError

    @classmethod
    def reserved_columns_partition_by(cls) -> list[str] | None:
        """Partition key(s) for the reserved-columns test.

        Defaults to ``["region"]``, the canonical partition column in the shared
        data-operations test dataset. Override to return ``None`` for ops that
        do not partition (e.g. scalar aggregate, binning, datetime, string) or
        to a different column list when the op uses a non-standard partition.
        """
        return ["region"]

    @classmethod
    def reserved_columns_order_by(cls) -> str | None:
        """Order key for the reserved-columns test.

        Defaults to ``None``. Override for ops that require an order key
        (e.g. frame aggregate, offset).
        """
        return None

    # -- Concrete test method --------------------------------------------------

    def test_mixin_reserved_column_collision_rejected(self) -> None:
        """Mixin: input column matching the reserved ``__mloda_`` prefix is rejected.

        Uses an uppercase column name to verify the guard is case-insensitive
        (SQLite and DuckDB unquoted identifiers fold case).
        """
        colliding_name = "__MLODA_USER_COL__"
        base_table: pa.Table = self._arrow_table  # type: ignore[attr-defined]
        colliding_table = base_table.append_column(
            colliding_name,
            pa.array([0] * base_table.num_rows, type=pa.int64()),
        )
        colliding_data = self.create_test_data(colliding_table)  # type: ignore[attr-defined]
        fs = make_feature_set(
            self.reserved_columns_feature_name(),
            partition_by=self.reserved_columns_partition_by(),
            order_by=self.reserved_columns_order_by(),
        )
        with pytest.raises(ValueError, match=r"__MLODA_USER_COL__"):
            self.implementation_class().calculate_feature(colliding_data, fs)  # type: ignore[attr-defined]

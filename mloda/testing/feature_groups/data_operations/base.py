"""Shared abstract base class for all data-operations framework test suites.

``DataOpsTestBase`` eliminates the boilerplate that was duplicated across all
9 operation-specific test bases (scalar_aggregate, aggregation, window_aggregation,
offset, rank, binning, datetime, frame_aggregate, string). It provides:

- 5 abstract adapter methods (implementation_class, create_test_data,
  extract_column, get_row_count, get_expected_type) that framework mixins
  or concrete test classes implement once.
- Shared setup_method / teardown_method (connection lifecycle).
- A ``_skip_if_unsupported`` helper for operations not available on every framework.
- A ``_compare_with_pyarrow`` cross-framework comparison helper.

Operation-specific test bases (e.g. ``AggregationTestBase``) inherit from this
class and add their own expected values and concrete test methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pyarrow as pa
import pytest

from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.helpers import extract_column as _extract_column


class DataOpsTestBase(ABC):
    """Abstract base class shared by all data-operations test suites.

    Subclasses must implement the 5 adapter methods below (or inherit them
    from a framework mixin such as ``PandasTestMixin``).

    Operation-specific bases extend this with concrete test methods and
    expected-value constants.
    """

    # -- Abstract adapter methods (implement once per framework) ----------------

    @classmethod
    @abstractmethod
    def implementation_class(cls) -> Any:
        """Return the feature-group implementation class under test."""

    @abstractmethod
    def create_test_data(self, arrow_table: pa.Table) -> Any:
        """Convert the canonical PyArrow test table to the framework's native format."""

    @abstractmethod
    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        """Extract a column from the framework result as a Python list."""

    @abstractmethod
    def get_row_count(self, result: Any) -> int:
        """Return the number of rows in the framework result."""

    @abstractmethod
    def get_expected_type(self) -> Any:
        """Return the expected Python type of the framework result."""

    # -- PyArrow reference (for cross-framework comparison) --------------------

    @classmethod
    def pyarrow_implementation_class(cls) -> Any:
        """Return the PyArrow implementation for cross-framework comparison.

        Operation-specific bases override this to import the correct class.
        """
        raise NotImplementedError("Subclass must override pyarrow_implementation_class")

    # -- Setup / teardown ------------------------------------------------------

    def setup_method(self) -> None:
        """Create test data from the canonical 12-row dataset.

        Connection-based mixins (DuckdbTestMixin, SqliteTestMixin) create
        ``self.conn`` first, then call ``super().setup_method()``.
        """
        self._arrow_table = PyArrowDataOpsTestDataCreator.create()
        self.test_data = self.create_test_data(self._arrow_table)

    def teardown_method(self) -> None:
        """Close ``self.conn`` if set by a connection-based mixin."""
        conn = getattr(self, "conn", None)
        if conn is not None:
            conn.close()

    # -- Helpers ---------------------------------------------------------------

    def _skip_if_unsupported(self, op: str) -> None:
        """Skip a test if *op* is not in the framework's supported set.

        Works with any subclass that defines ``supported_agg_types``,
        ``supported_ops``, ``supported_offset_types``, or ``supported_rank_types``.
        """
        for attr in ("supported_agg_types", "supported_ops", "supported_offset_types", "supported_rank_types"):
            method = getattr(self, attr, None)
            if method is not None:
                if op not in method():
                    pytest.skip(f"{op} not supported by this framework")
                return
        pytest.skip(f"{op} not supported by this framework")

    # -- Cross-framework comparison --------------------------------------------

    def _compare_with_pyarrow(
        self,
        feature_name: str,
        *,
        partition_by: list[str] | None = None,
        order_by: str | None = None,
        use_approx: bool = False,
        rel: float = 1e-6,
    ) -> None:
        """Run a feature through this framework and PyArrow, assert results match."""
        try:
            pyarrow_cls = self.pyarrow_implementation_class()
        except NotImplementedError:
            pytest.skip("No PyArrow implementation for cross-framework comparison")

        from mloda.testing.feature_groups.data_operations.helpers import make_feature_set

        fs = make_feature_set(feature_name, partition_by=partition_by, order_by=order_by)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        ref = pyarrow_cls.calculate_feature(self._arrow_table, fs)

        result_col = self.extract_column(result, feature_name)
        ref_col = _extract_column(ref, feature_name)

        assert len(result_col) == len(ref_col), f"row count {len(result_col)} != reference {len(ref_col)}"
        if use_approx:
            for i, (ref_val, fw_val) in enumerate(zip(ref_col, result_col)):
                if ref_val is None:
                    assert fw_val is None, f"row {i}: expected None, got {fw_val}"
                else:
                    assert fw_val == pytest.approx(ref_val, rel=rel), f"row {i}: {fw_val} != reference {ref_val}"
        else:
            assert result_col == ref_col

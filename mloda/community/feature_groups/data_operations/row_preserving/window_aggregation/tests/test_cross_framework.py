"""Cross-framework comparison tests for window aggregation implementations.

Runs each aggregation through all four framework implementations and asserts
that every framework produces the same results as the PyArrow reference.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

pytest.importorskip("pandas")
pytest.importorskip("polars")

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator import PyArrowDataOpsTestDataCreator
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pandas_window_aggregation import (
    PandasWindowAggregation,
)
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.polars_lazy_window_aggregation import (
    PolarsLazyWindowAggregation,
)
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pyarrow_window_aggregation import (
    PyArrowWindowAggregation,
)
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.sql_window_aggregation import (
    SqlWindowAggregation,
)


@pytest.fixture
def sample_data() -> pa.Table:
    """Return the shared 12-row test dataset as a PyArrow Table."""
    return PyArrowDataOpsTestDataCreator.create()


def _make_feature_set(feature_name: str, partition_by: list[str]) -> FeatureSet:
    """Helper to build a FeatureSet with partition_by options."""
    feature = Feature(
        feature_name,
        options=Options(context={"partition_by": partition_by}),
    )
    fs = FeatureSet()
    fs.add(feature)
    return fs


def _extract_column(result: pa.Table, column_name: str) -> list[Any]:
    """Extract a result column as a Python list."""
    return list(result.column(column_name).to_pylist())


class TestCrossFrameworkComparison:
    """Compare all framework implementations against PyArrow reference."""

    FRAMEWORKS = {
        "pyarrow": PyArrowWindowAggregation,
        "sql": SqlWindowAggregation,
        "polars_lazy": PolarsLazyWindowAggregation,
        "pandas": PandasWindowAggregation,
    }

    def _run_all_frameworks(
        self,
        sample_data: pa.Table,
        feature_name: str,
        partition_by: list[str],
    ) -> dict[str, list[Any]]:
        """Run the given feature through all frameworks and return results keyed by name."""
        feature_set = _make_feature_set(feature_name, partition_by)
        results: dict[str, list[Any]] = {}
        for name, cls in self.FRAMEWORKS.items():
            result_table = cls.calculate_feature(sample_data, feature_set)
            results[name] = _extract_column(result_table, feature_name)
        return results

    def _assert_matches_reference(
        self,
        results: dict[str, list[Any]],
        use_approx: bool = False,
    ) -> None:
        """Assert every framework result matches the PyArrow reference."""
        reference = results["pyarrow"]
        for name, values in results.items():
            if name == "pyarrow":
                continue
            assert len(values) == len(reference), f"{name}: row count {len(values)} != reference {len(reference)}"
            if use_approx:
                for i, (ref_val, fw_val) in enumerate(zip(reference, values)):
                    if ref_val is None:
                        assert fw_val is None, f"{name} row {i}: expected None, got {fw_val}"
                    else:
                        assert fw_val == pytest.approx(ref_val, rel=1e-6), (
                            f"{name} row {i}: {fw_val} != reference {ref_val}"
                        )
            else:
                assert values == reference, f"{name} produced {values}, expected {reference}"

    def test_sum_cross_framework(self, sample_data: pa.Table) -> None:
        """Sum of value_int partitioned by region: all frameworks must match PyArrow."""
        results = self._run_all_frameworks(sample_data, "value_int__sum_groupby", ["region"])
        self._assert_matches_reference(results)

    def test_avg_cross_framework(self, sample_data: pa.Table) -> None:
        """Average of value_int partitioned by region: all frameworks must match PyArrow."""
        results = self._run_all_frameworks(sample_data, "value_int__avg_groupby", ["region"])
        self._assert_matches_reference(results, use_approx=True)

    def test_count_cross_framework(self, sample_data: pa.Table) -> None:
        """Count of value_int partitioned by region: all frameworks must match PyArrow."""
        results = self._run_all_frameworks(sample_data, "value_int__count_groupby", ["region"])
        self._assert_matches_reference(results)

    def test_min_cross_framework(self, sample_data: pa.Table) -> None:
        """Minimum of value_int partitioned by region: all frameworks must match PyArrow."""
        results = self._run_all_frameworks(sample_data, "value_int__min_groupby", ["region"])
        self._assert_matches_reference(results)

    def test_max_cross_framework(self, sample_data: pa.Table) -> None:
        """Maximum of value_int partitioned by region: all frameworks must match PyArrow."""
        results = self._run_all_frameworks(sample_data, "value_int__max_groupby", ["region"])
        self._assert_matches_reference(results)

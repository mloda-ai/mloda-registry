"""Security-focused tests for the column aggregation package.

Covers SQL injection attempts, invalid aggregation types, special characters
in column names, and type confusion via options.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.aggregation.base import (
    AGGREGATION_TYPES,
    ColumnAggregationFeatureGroup,
)
from mloda.community.feature_groups.data_operations.aggregation.pyarrow_aggregation import (
    PyArrowColumnAggregation,
)


def _make_feature_set(feature_name: str, options: Options | None = None) -> FeatureSet:
    feature = Feature(feature_name, options=options or Options())
    fs = FeatureSet()
    fs.add(feature)
    return fs


class TestSQLInjectionInFeatureNames:
    """Verify that SQL injection payloads in feature names are rejected."""

    SQL_INJECTION_PAYLOADS = [
        "value_int__sum_aggr; DROP TABLE users",
        "value_int__sum_aggr' OR '1'='1",
        "value_int__sum_aggr); DELETE FROM data--",
        "value_int__1 UNION SELECT * FROM secrets--_aggr",
    ]

    @pytest.mark.parametrize("malicious_name", SQL_INJECTION_PAYLOADS)
    def test_sql_injection_rejected_by_match(self, malicious_name: str) -> None:
        """SQL injection payloads must not match the feature group criteria."""
        options = Options()
        result = ColumnAggregationFeatureGroup.match_feature_group_criteria(malicious_name, options, None)
        assert result is False, f"Should reject: {malicious_name}"


class TestInvalidAggregationTypes:
    """Verify that invalid aggregation types are rejected at every level."""

    INVALID_TYPES = [
        "drop_table",
        "exec",
        "eval",
        "__import__",
        "SUM",
        "Sum",
        "",
    ]

    @pytest.mark.parametrize("bad_type", INVALID_TYPES)
    def test_invalid_type_rejected_by_pattern_match(self, bad_type: str) -> None:
        """Pattern-based feature names with invalid agg types must not match."""
        feature_name = f"value_int__{bad_type}_aggr"
        options = Options()
        result = ColumnAggregationFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is False, f"Should reject aggregation type: {bad_type}"

    @pytest.mark.parametrize("bad_type", INVALID_TYPES)
    def test_invalid_type_rejected_by_options_match(self, bad_type: str) -> None:
        """Options-based configuration with invalid agg types must not match."""
        if bad_type == "":
            pytest.skip("Empty string in options has different behavior path")
        options = Options(
            context={
                "aggregation_type": bad_type,
                "in_features": "value_int",
            }
        )
        result = ColumnAggregationFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False, f"Should reject aggregation type via options: {bad_type}"

    def test_pandas_rejects_unknown_type_at_compute(self) -> None:
        """Even if an invalid type somehow reaches compute, it raises ValueError."""
        pytest.importorskip("pandas")
        from mloda.community.feature_groups.data_operations.aggregation.pandas_aggregation import (
            PandasColumnAggregation,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        df = arrow_table.to_pandas()
        with pytest.raises(ValueError, match="[Uu]nsupported"):
            PandasColumnAggregation._compute_aggregation(df, "result_col", "value_int", "evil_type")

    def test_pyarrow_rejects_unknown_type_at_compute(self) -> None:
        """Even if an invalid type somehow reaches compute, it raises ValueError."""
        arrow_table = PyArrowDataOpsTestDataCreator.create()
        with pytest.raises(ValueError, match="[Uu]nsupported"):
            PyArrowColumnAggregation._compute_aggregation(arrow_table, "result_col", "value_int", "evil_type")


class TestSpecialCharactersInColumnNames:
    """Verify that special characters in source column names are handled safely."""

    def test_pattern_rejects_special_chars_in_agg_type(self) -> None:
        """Feature names with special chars in the agg type portion are rejected."""
        malicious_names = [
            "value_int__su'm_aggr",
            'value_int__su"m_aggr',
            "value_int__su;m_aggr",
            "value_int__su)m_aggr",
        ]
        options = Options()
        for name in malicious_names:
            result = ColumnAggregationFeatureGroup.match_feature_group_criteria(name, options, None)
            assert result is False, f"Should reject: {name}"

    def test_sqlite_quote_ident_handles_double_quotes(self) -> None:
        """Verify quote_ident properly escapes double quotes for SQL identifiers."""
        from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

        assert quote_ident('col"name') == '"col""name"'
        assert quote_ident("normal_col") == '"normal_col"'
        assert quote_ident("col'; DROP TABLE--") == '"col\'; DROP TABLE--"'


class TestTypeConfusionInOptions:
    """Verify that non-string types in options do not bypass validation."""

    def test_none_aggregation_type_rejected(self) -> None:
        """None as aggregation_type must not match."""
        options = Options(
            context={
                "aggregation_type": None,
                "in_features": "value_int",
            }
        )
        result = ColumnAggregationFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_integer_aggregation_type_rejected(self) -> None:
        """An integer as aggregation_type must not match."""
        options = Options(
            context={
                "aggregation_type": 42,
                "in_features": "value_int",
            }
        )
        result = ColumnAggregationFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_list_aggregation_type_rejected(self) -> None:
        """A list as aggregation_type must not match."""
        options = Options(
            context={
                "aggregation_type": ["sum", "max"],
                "in_features": "value_int",
            }
        )
        result = ColumnAggregationFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False


class TestAllowlistCompleteness:
    """Verify that every type in AGGREGATION_TYPES is covered by every backend."""

    def test_duckdb_covers_all_types(self) -> None:
        from mloda.community.feature_groups.data_operations.aggregation.duckdb_aggregation import (
            _DUCKDB_AGG_FUNCS,
        )

        for agg_type in AGGREGATION_TYPES:
            assert agg_type in _DUCKDB_AGG_FUNCS, f"DuckDB backend missing aggregation type: {agg_type}"

    def test_pandas_covers_all_types(self) -> None:
        """Pandas uses if/elif, so we verify by calling _compute_aggregation."""
        pytest.importorskip("pandas")
        from mloda.community.feature_groups.data_operations.aggregation.pandas_aggregation import (
            PandasColumnAggregation,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        df = arrow_table.to_pandas()
        for agg_type in AGGREGATION_TYPES:
            result = PandasColumnAggregation._compute_aggregation(df, f"test_{agg_type}", "value_int", agg_type)
            assert f"test_{agg_type}" in result.columns

    def test_pyarrow_covers_all_types(self) -> None:
        arrow_table = PyArrowDataOpsTestDataCreator.create()
        for agg_type in AGGREGATION_TYPES:
            result = PyArrowColumnAggregation._compute_aggregation(
                arrow_table, f"test_{agg_type}", "value_int", agg_type
            )
            assert f"test_{agg_type}" in result.column_names

    def test_polars_covers_all_types(self) -> None:
        polars = pytest.importorskip("polars")
        from mloda.community.feature_groups.data_operations.aggregation.polars_lazy_aggregation import (
            PolarsLazyColumnAggregation,
        )

        arrow_table = PyArrowDataOpsTestDataCreator.create()
        lf = polars.from_arrow(arrow_table).lazy()
        for agg_type in AGGREGATION_TYPES:
            result = PolarsLazyColumnAggregation._compute_aggregation(lf, f"test_{agg_type}", "value_int", agg_type)
            collected = result.collect()
            assert f"test_{agg_type}" in collected.columns

    def test_sqlite_covers_supported_types(self) -> None:
        """SQLite supports a subset of aggregation types."""
        from mloda.community.feature_groups.data_operations.aggregation.sqlite_aggregation import (
            _SQLITE_AGG_FUNCS,
        )

        # SQLite should cover at least the basic types
        basic_types = {"sum", "min", "max", "avg", "mean", "count"}
        for agg_type in basic_types:
            assert agg_type in _SQLITE_AGG_FUNCS, f"SQLite backend missing basic aggregation type: {agg_type}"


class TestCaseSensitivity:
    """Verify that aggregation types are case-sensitive (lowercase only)."""

    UPPERCASE_TYPES = ["SUM", "MIN", "MAX", "AVG", "MEAN", "COUNT", "STD", "VAR", "MEDIAN"]
    MIXED_CASE_TYPES = ["Sum", "Min", "Max", "Avg", "Mean", "Count", "Std", "Var", "Median"]

    @pytest.mark.parametrize("upper_type", UPPERCASE_TYPES)
    def test_uppercase_rejected(self, upper_type: str) -> None:
        feature_name = f"value_int__{upper_type}_aggr"
        options = Options()
        result = ColumnAggregationFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is False, f"Should reject uppercase: {upper_type}"

    @pytest.mark.parametrize("mixed_type", MIXED_CASE_TYPES)
    def test_mixed_case_rejected(self, mixed_type: str) -> None:
        feature_name = f"value_int__{mixed_type}_aggr"
        options = Options()
        result = ColumnAggregationFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is False, f"Should reject mixed case: {mixed_type}"

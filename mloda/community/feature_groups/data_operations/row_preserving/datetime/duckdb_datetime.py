"""DuckDB implementation for datetime extraction feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.row_preserving.datetime.base import (
    DateTimeFeatureGroup,
)

# DuckDB native function expressions for datetime extraction.
# dayofweek: DuckDB DAYOFWEEK returns 0=Sunday, 1=Monday, ..., 6=Saturday.
#   Convert to Python convention (0=Monday, 6=Sunday): (DAYOFWEEK(col) + 6) % 7
# is_weekend: DuckDB DAYOFWEEK 0=Sunday, 6=Saturday
_DUCKDB_DATETIME_EXPRS: dict[str, str] = {
    "year": "YEAR({col})",
    "month": "MONTH({col})",
    "day": "DAY({col})",
    "hour": "HOUR({col})",
    "minute": "MINUTE({col})",
    "second": "SECOND({col})",
    "dayofweek": "(DAYOFWEEK({col}) + 6) % 7",
    "is_weekend": "CASE WHEN {col} IS NULL THEN NULL WHEN DAYOFWEEK({col}) IN (0, 6) THEN 1 ELSE 0 END",
    "quarter": "QUARTER({col})",
}


class DuckdbDateTimeExtraction(DateTimeFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {DuckDBFramework}

    @classmethod
    def _compute_datetime(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        op: str,
    ) -> DuckdbRelation:
        expr_template = _DUCKDB_DATETIME_EXPRS.get(op)
        if expr_template is None:
            raise ValueError(f"Unsupported datetime operation for DuckDB: {op}")

        quoted_source = quote_ident(source_col)
        expr = expr_template.format(col=quoted_source)
        quoted_feature = quote_ident(feature_name)

        raw_sql = f"*, {expr} AS {quoted_feature}"
        result: DuckdbRelation = data.select(_raw_sql=raw_sql)
        return result

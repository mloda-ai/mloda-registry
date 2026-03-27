"""DuckDB implementation for string operation feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.string.base import (
    StringFeatureGroup,
)

# DuckDB native string function expressions.
_DUCKDB_STRING_EXPRS: dict[str, str] = {
    "upper": "UPPER({col})",
    "lower": "LOWER({col})",
    "trim": "TRIM({col})",
    "length": "LENGTH({col})",
    "reverse": "REVERSE({col})",
}


class DuckdbStringOps(StringFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {DuckDBFramework}

    @classmethod
    def _compute_string(
        cls,
        data: Any,
        feature_name: str,
        source_col: str,
        op: str,
    ) -> DuckdbRelation:
        expr_template = _DUCKDB_STRING_EXPRS.get(op)
        if expr_template is None:
            raise ValueError(f"Unsupported string operation for DuckDB: {op}")

        quoted_source = quote_ident(source_col)
        expr = expr_template.format(col=quoted_source)
        quoted_feature = quote_ident(feature_name)

        raw_sql = f"*, {expr} AS {quoted_feature}"
        result: DuckdbRelation = data.select(_raw_sql=raw_sql)
        return result

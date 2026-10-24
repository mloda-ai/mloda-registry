"""SQLite implementation for datetime extraction feature groups."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.datetime.base import (
    DateTimeFeatureGroup,
)

# SQLite strftime-based expressions for datetime extraction.
# dayofweek: SQLite %w returns 0=Sunday, 6=Saturday. Convert to 0=Monday:
#   (strftime('%w', col) + 6) % 7
_SQLITE_DATETIME_EXPRS: dict[str, str] = {
    "year": "CAST(strftime('%Y', {col}) AS INTEGER)",
    "month": "CAST(strftime('%m', {col}) AS INTEGER)",
    "day": "CAST(strftime('%d', {col}) AS INTEGER)",
    "hour": "CAST(strftime('%H', {col}) AS INTEGER)",
    "minute": "CAST(strftime('%M', {col}) AS INTEGER)",
    "second": "CAST(strftime('%S', {col}) AS INTEGER)",
    "dayofweek": "(CAST(strftime('%w', {col}) AS INTEGER) + 6) % 7",
    "is_weekend": "CASE WHEN {col} IS NULL THEN NULL WHEN strftime('%w', {col}) IN ('0', '6') THEN 1 ELSE 0 END",
    "quarter": "((CAST(strftime('%m', {col}) AS INTEGER) - 1) / 3 + 1)",
}


class SqliteDateTimeExtraction(Implementation[DateTimeFeatureGroup]):
    feature_group = DateTimeFeatureGroup
    compute_framework = SqliteFramework

    @classmethod
    def calculate(
        cls, group: type[DateTimeFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.extract_datetime(data, features, compute_datetime=cls._compute_datetime)

    @classmethod
    def _compute_datetime(
        cls,
        data: SqliteRelation,
        feature_name: str,
        source_col: str,
        op: str,
    ) -> SqliteRelation:
        expr_template = _SQLITE_DATETIME_EXPRS.get(op)
        if expr_template is None:
            raise ValueError(f"Unsupported datetime operation for SQLite: {op}")

        quoted_source = quote_ident(source_col)
        expr = expr_template.format(col=quoted_source)
        quoted_feature = quote_ident(feature_name)

        sql = " ".join(
            [
                "SELECT",
                f"{expr} AS {quoted_feature}",
                "FROM",
                f"{quote_ident(data.table_name)}",
                "ORDER BY rowid",
            ]
        )
        cursor = data.connection.execute(sql)
        rows = cursor.fetchall()

        result_values = [row[0] for row in rows]
        return data.append_column(feature_name, result_values)

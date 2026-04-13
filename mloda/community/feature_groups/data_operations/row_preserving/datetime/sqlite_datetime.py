"""SQLite implementation for datetime extraction feature groups."""

from __future__ import annotations

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.reserved_columns import assert_no_reserved_columns
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


class SqliteDateTimeExtraction(DateTimeFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SqliteFramework}

    @classmethod
    def _compute_datetime(
        cls,
        data: SqliteRelation,
        feature_name: str,
        source_col: str,
        op: str,
    ) -> SqliteRelation:
        assert_no_reserved_columns(data.columns, framework="SQLite", operation="datetime")

        expr_template = _SQLITE_DATETIME_EXPRS.get(op)
        if expr_template is None:
            raise ValueError(f"Unsupported datetime operation for SQLite: {op}")

        quoted_source = quote_ident(source_col)
        expr = expr_template.format(col=quoted_source)
        quoted_feature = quote_ident(feature_name)
        qrn = quote_ident("__mloda_rn__")

        sql = " ".join(
            [
                "SELECT",
                f"{expr} AS {quoted_feature},",
                f"ROW_NUMBER() OVER (ORDER BY rowid) AS {qrn}",
                "FROM",
                f"{quote_ident(data.table_name)}",
                "ORDER BY",
                qrn,
            ]
        )
        cursor = data.connection.execute(sql)
        rows = cursor.fetchall()

        result_values = [row[0] for row in rows]
        return data.append_column(feature_name, result_values)

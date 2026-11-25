"""SQLite implementation for datetime extraction feature groups."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.datetime.base import (
    DateTimeFeatureGroup,
)
from mloda.community.feature_groups.data_operations.sqlite_helpers import wall_clock_expr

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
    "dayofweek": "CAST((CAST(strftime('%w', {col}) AS INTEGER) + 6) % 7 AS INTEGER)",
    "is_weekend": "CAST(CASE WHEN {col} IS NULL THEN NULL WHEN strftime('%w', {col}) IN ('0', '6') THEN 1 ELSE 0 END AS INTEGER)",
    "quarter": "CAST(((CAST(strftime('%m', {col}) AS INTEGER) - 1) / 3 + 1) AS INTEGER)",
}


class SqliteDateTimeExtraction(Implementation[DateTimeFeatureGroup]):
    feature_group = DateTimeFeatureGroup
    compute_framework = SqliteFramework

    @classmethod
    def calculate(
        cls, group: type[DateTimeFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.extract_datetime(
            data, features, compute_datetime=cls._compute_datetime, is_date_source=cls._is_date_source
        )

    @staticmethod
    def _is_date_source(data: Any, source_col: str) -> bool:
        hints = data.type_hints
        hint = None if hints is None else dict(zip(data.columns, hints)).get(source_col)
        return hint is not None and pa.types.is_date(hint)

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
        # First 19 chars (second precision): older SQLite rounds a fractional pre-epoch value across midnight.
        expr = expr_template.format(col=f"substr({wall_clock_expr(quoted_source)}, 1, 19)")
        quoted_feature = quote_ident(feature_name)

        data._ensure_column_absent(feature_name)
        # Raw "*" projection keeps row order and the source type hints.
        return data.select(_raw_sql=f"*, {expr} AS {quoted_feature}")

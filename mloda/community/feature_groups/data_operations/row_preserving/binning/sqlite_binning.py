"""SQLite implementation for binning feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.binning.base import (
    BinningFeatureGroup,
)


class SqliteBinning(BinningFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {SqliteFramework}

    @classmethod
    def _compute_binning(
        cls,
        data: SqliteRelation,
        feature_name: str,
        source_col: str,
        op: str,
        n_bins: int,
    ) -> SqliteRelation:
        quoted_source = quote_ident(source_col)
        quoted_feature = quote_ident(feature_name)
        qrn = quote_ident("__mloda_rn__")
        table_name = quote_ident(data.table_name)

        if op == "bin":
            # Safety: all identifiers are quote_ident()-quoted, n_bins is int.
            expr = (
                f"CASE WHEN {quoted_source} IS NULL THEN NULL "  # nosec B608
                f"WHEN (SELECT MAX({quoted_source}) FROM {table_name}) = "
                f"(SELECT MIN({quoted_source}) FROM {table_name}) THEN 0 "
                f"ELSE MIN(CAST("
                f"({quoted_source} - (SELECT MIN({quoted_source}) FROM {table_name})) / "
                f"(((SELECT MAX({quoted_source}) FROM {table_name}) - "
                f"(SELECT MIN({quoted_source}) FROM {table_name})) / {n_bins}.0)"
                f" AS INTEGER), {n_bins - 1}) END"
            )
        elif op == "qbin":
            expr = (
                f"CASE WHEN {quoted_source} IS NULL THEN NULL "
                f"ELSE MIN(NTILE({n_bins}) OVER (ORDER BY {quoted_source}) - 1, {n_bins - 1}) END"
            )
        else:
            raise ValueError(f"Unsupported binning operation for SQLite: {op}")

        sql = " ".join(
            [
                "SELECT",
                f"{expr} AS {quoted_feature},",
                f"ROW_NUMBER() OVER (ORDER BY rowid) AS {qrn}",
                "FROM",
                table_name,
                "ORDER BY",
                qrn,
            ]
        )
        cursor = data.connection.execute(sql)
        rows = cursor.fetchall()

        result_values = [row[0] for row in rows]
        return data.append_column(feature_name, result_values)

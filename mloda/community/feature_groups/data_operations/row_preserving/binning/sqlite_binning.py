"""SQLite implementation for binning feature groups."""

from __future__ import annotations


from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import pick_helper_column_name, quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.binning.base import (
    BinningFeatureGroup,
)


class SqliteBinning(BinningFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
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
        table_name = quote_ident(data.table_name)

        if op == "bin":
            # Equal-width binning uses scalar subqueries (no window function), so this
            # path stays a fetch+append; ``ORDER BY rowid`` preserves input row order.
            # Safety: all identifiers are quote_ident()-quoted, n_bins is int.
            expr = (
                f"CASE WHEN {quoted_source} IS NULL THEN NULL "  # nosec
                f"WHEN (SELECT MAX({quoted_source}) FROM {table_name}) = "
                f"(SELECT MIN({quoted_source}) FROM {table_name}) THEN 0 "
                f"ELSE MIN(CAST("
                f"({quoted_source} - (SELECT MIN({quoted_source}) FROM {table_name})) / "
                f"(((SELECT MAX({quoted_source}) FROM {table_name}) - "
                f"(SELECT MIN({quoted_source}) FROM {table_name})) / {n_bins}.0)"
                f" AS INTEGER), {n_bins - 1}) END"
            )
            sql = " ".join(
                [
                    "SELECT",
                    f"{expr} AS {quoted_feature}",
                    "FROM",
                    table_name,
                    "ORDER BY rowid",
                ]
            )
            cursor = data.connection.execute(sql)
            rows = cursor.fetchall()
            result_values = [row[0] for row in rows]
            return data.append_column(feature_name, result_values)

        if op != "qbin":
            raise ValueError(f"Unsupported binning operation for SQLite: {op}")

        # Quantile binning is a real NTILE window with an expression partition key
        # (non-null rows only) wrapped in scalar MIN(..., n-1). Precompute the partition
        # key into a helper column, run NTILE, then apply the wrapper via a raw
        # projection (Pattern W). ``MIN(a, b)`` here is the 2-arg scalar min.
        original_cols = list(data.columns)
        rn = pick_helper_column_name(taken=set(data.columns) | {feature_name})
        part = pick_helper_column_name(taken=set(data.columns) | {feature_name, rn})
        ntile = pick_helper_column_name(taken=set(data.columns) | {feature_name, rn, part})
        rel = data.with_row_number(rn, order_by=["rowid"])
        rel = rel.select(_raw_sql=f"*, CASE WHEN {quoted_source} IS NOT NULL THEN 1 END AS {quote_ident(part)}")
        rel = rel.window(f"NTILE({n_bins})", ntile, partition_by=[part], order_by=[source_col])
        qntile = quote_ident(ntile)
        feat_expr = f"CASE WHEN {quoted_source} IS NULL THEN NULL ELSE MIN({qntile} - 1, {n_bins - 1}) END"
        proj = (
            ", ".join(quote_ident(c) for c in original_cols) + f", {quote_ident(rn)}, {feat_expr} AS {quoted_feature}"
        )
        rel = rel.select(_raw_sql=proj)
        rel = rel.order(rn)
        return rel.select(*original_cols, feature_name)
